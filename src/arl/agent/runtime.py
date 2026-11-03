"""Interfaz pública y confiable del agente: iniciar un turno, consultar su estado y reanudar.

Es la frontera del modelo de seguridad del MVP. Toda función toma el lock del thread (obligatorio:
sin ``lock_dir`` falla con ``LockingRequired``) y verifica, antes de leer o escribir, que el
thread pertenezca al principal del proceso. La identidad no es un parámetro: es la del
proceso, ligada al grafo en ``build_graph`` a partir del gateway MCP. Invocar el grafo
directamente queda fuera del modelo de seguridad (ver docs/agent.md).

Cada llamada a ``run_turn``, ``resume`` o ``continue_after_failure`` es una ejecución (``run_id``)
con eventos de traza best-effort: ``run_started`` ... ``run_finished`` (ver docs/observability.md).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from arl.agent.gateway import ToolCallFailed
from arl.agent.history import close_unanswered
from arl.agent.locks import LockingRequired, ThreadBusy, thread_lock
from arl.agent.redaction import LOGGER
from arl.agent.scope import BoundScope, RequestScope, ScopeMismatch, ScopeViolation
from arl.agent.state import ApprovalDecision, ApprovalMismatch, PendingWrite, PrincipalMismatch
from arl.tracing import recorder as tracing
from arl.tracing.recorder import JsonlTraceSink, RunContext

# Cada superpaso se persiste antes del siguiente: tras un crash se reanuda desde el último nodo
# completado y nunca se pierde una aprobación ya registrada.
DURABILITY = "sync"


class TurnStatus(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    TOOL_LIMIT = "tool_limit"
    TURN_LIMIT = "turn_limit"
    # Un fallo detuvo la ejecución antes de cualquier escritura: se puede continuar sin riesgo.
    INTERRUPTED = "interrupted"
    # Una escritura aprobada no pudo confirmarse: pudo haber ocurrido o no. Continuar la reintenta
    # con la misma clave de idempotencia, así que nunca se duplica.
    WRITE_UNCERTAIN = "write_uncertain"
    # Otro proceso está ejecutando el thread ahora: el estado se consultará cuando termine.
    BUSY = "busy"


class ThreadAwaitingApproval(RuntimeError):
    """No se acepta un mensaje nuevo mientras hay una escritura esperando decisión."""


class NoPendingApproval(RuntimeError):
    """Se intentó reanudar un thread que no está esperando aprobación."""


class ThreadNeedsRecovery(RuntimeError):
    """Hay una escritura aprobada sin confirmar: hay que continuarla, no iniciar otro turno."""


class NothingToContinue(RuntimeError):
    """``continue_after_failure`` sobre un thread que no quedó interrumpido."""


@dataclass(frozen=True)
class TurnResult:
    status: TurnStatus
    reply: str
    approval_request: dict[str, Any] | None = None
    approval_error: str | None = None  # por qué se rechazó la última decisión entregada
    error: str | None = None  # código y detalle seguro del fallo que detuvo la ejecución
    pending_action: dict[str, Any] | None = None  # la escritura aprobada sin confirmar
    run_id: str | None = None  # correlaciona el resultado con sus trazas y el log


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def bound_principal(graph: CompiledStateGraph) -> str:
    """Identidad del proceso con la que se construyó el grafo."""
    return graph.config["configurable"]["principal_id"]


def _check_owner(graph: CompiledStateGraph, values: dict[str, Any]) -> str:
    principal_id = bound_principal(graph)
    owner = values.get("principal_id")
    if owner is not None and owner != principal_id:
        raise PrincipalMismatch("el thread pertenece a otro principal")
    return principal_id


def _lock_dir(graph: CompiledStateGraph) -> Path:
    """Sin lock no se ejecuta: un grafo compilado aparte o con la config sobrescrita falla."""
    value = graph.config["configurable"].get("lock_dir")
    if not value:
        raise LockingRequired(
            "el grafo no tiene lock_dir: construilo con build_graph(lock_dir=...)"
        )
    return Path(value)


def _trace_sink(graph: CompiledStateGraph) -> JsonlTraceSink | None:
    value = graph.config["configurable"].get("trace_dir")
    return JsonlTraceSink(Path(value)) if value else None


# Rechazos esperables del runtime: la ejecución no llegó a correr (estado final ``refused``).
_REFUSALS = (
    LockingRequired,
    PrincipalMismatch,
    ApprovalMismatch,
    ThreadAwaitingApproval,
    NoPendingApproval,
    ThreadNeedsRecovery,
    NothingToContinue,
    ScopeMismatch,
    ScopeViolation,
)


async def _traced(
    graph: CompiledStateGraph,
    thread_id: str,
    operation: str,
    body: Callable[[RunContext], Awaitable[TurnResult]],
    scope_kind: str | None = None,
) -> TurnResult:
    """Una ejecución observable: ``run_started`` y siempre ``run_finished`` (salvo crash del
    proceso), con el estado final, la duración y los conteos. Las excepciones se re-lanzan."""
    sink = _trace_sink(graph)
    with tracing.run_scope(sink, thread_id=thread_id, principal_id=bound_principal(graph)) as run:
        run.emit("run_started", {"operation": operation, "scope": scope_kind})

        def finish(status: str, refusal: str | None = None) -> None:
            run.emit(
                "run_finished",
                {
                    "operation": operation,
                    "status": status,
                    "duration_ms": tracing.elapsed_ms(run.started),
                    "refusal": refusal,
                    "failure": run.failure,
                    "counts": run.snapshot_counts().model_dump(),
                },
            )

        try:
            result = await body(run)
        except ThreadBusy:
            finish(TurnStatus.BUSY, refusal="ThreadBusy")
            raise
        except _REFUSALS as exc:
            finish("refused", refusal=type(exc).__name__)
            raise
        except BaseException as exc:
            run.failure = run.failure or {
                "code": "unhandled",
                "operation": operation,
                "error_type": tracing.safe_id(type(exc).__name__),
            }
            finish("error")
            raise
        finish(result.status)
        return replace(result, run_id=run.run_id)


async def _invoke(
    graph: CompiledStateGraph, payload: Any, thread_id: str, run: RunContext
) -> TurnResult:
    """Ejecuta el grafo y convierte los fallos del MCP en un estado claro y reanudable.

    Los reintentos automáticos (acotados) ya ocurrieron dentro del nodo; si se agotaron o el
    fallo es definitivo, el checkpoint queda antes del nodo fallido. Se llama con el lock del
    thread tomado. El log lleva diagnóstico (operación, código, tipo de error, run_id), nunca
    el texto del error original.
    """
    config = {**thread_config(thread_id), "metadata": {"run_id": run.run_id}}
    try:
        await graph.ainvoke(payload, config, durability=DURABILITY)
    except ToolCallFailed as exc:
        result = await _result(graph, thread_id)
        run.failure = {"code": exc.code, "operation": exc.tool, "error_type": exc.error_type}
        LOGGER.warning(
            "ejecución detenida run_id=%s thread=%s operation=%s code=%s error_type=%s status=%s",
            run.run_id, thread_id, exc.tool, exc.code, exc.error_type, result.status,
        )  # fmt: skip
        if result.status is TurnStatus.WRITE_UNCERTAIN and result.pending_action:
            action = result.pending_action
            run.emit(
                "write_uncertain",
                {
                    "action_id": action["action_id"],
                    "idempotency_key": action["idempotency_key"],
                    "tool_call_id": tracing.safe_id(action["tool_call_id"]),
                    "task_id": action["task_id"],
                    "failure": run.failure,
                },
            )
        return replace(result, error=f"[{exc.code}] {exc.tool}")
    return await _result(graph, thread_id)


async def run_turn(
    graph: CompiledStateGraph, *, thread_id: str, text: str, scope: RequestScope
) -> TurnResult:
    """Inicia un turno con el **alcance** que la aplicación autorizó (``arl.agent.scope``).

    ``scope`` es obligatorio y lo declara la aplicación a partir de una entrada explícita del
    usuario; nunca del LLM ni del texto del pedido. Se liga a este thread y a la identidad del
    proceso, se sella y rige hasta que termine el turno, incluidas sus aprobaciones,
    reanudaciones y recuperaciones. ``resume`` y ``continue_after_failure`` no lo cambian.

    Política de historial (ver ``arl.agent.history``):

    - aprobación pendiente → ``ThreadAwaitingApproval`` (hay que decidir);
    - escritura aprobada sin confirmar → ``ThreadNeedsRecovery``: no se cierra ni se descarta,
      porque su resultado se desconoce;
    - lectura o preparación interrumpida → sus tool calls se cierran con ``[not_executed]``
      antes del mensaje nuevo (no tuvieron efectos).
    """

    if not isinstance(scope, RequestScope):
        raise TypeError("run_turn requiere un RequestScope declarado por la aplicación")

    async def body(run: RunContext) -> TurnResult:
        with thread_lock(_lock_dir(graph), thread_id):
            snapshot = await graph.aget_state(thread_config(thread_id))
            principal_id = _check_owner(graph, snapshot.values)  # antes de escribir en el thread
            bound = BoundScope.bind(scope, thread_id=thread_id, principal_id=principal_id)
            if snapshot.interrupts:
                raise ThreadAwaitingApproval(
                    f"el thread '{thread_id}' tiene una aprobación pendiente"
                )
            if snapshot.next == ("execute_write",):
                raise ThreadNeedsRecovery(
                    f"el thread '{thread_id}' requiere continue_after_failure"
                )
            closed = close_unanswered(snapshot.values.get("messages", []))
            if closed:
                run.emit(
                    "history_closed",
                    {
                        "tool_call_ids": [tracing.safe_id(m.tool_call_id) for m in closed],
                        "reason": "not_executed",
                    },
                )
            payload = {
                "messages": [*closed, HumanMessage(text)],
                "principal_id": principal_id,
                # Un alcance nuevo por turno lógico, con su registro de reservas vacío. No pisa
                # reservas abiertas: con una aprobación pendiente o una escritura incierta, el
                # turno ya se rechazó arriba.
                "request_scope": bound.model_dump(mode="json"),
                "scope_ledger": {},
            }
            return await _invoke(graph, payload, thread_id, run)

    kind = "read_only" if scope.kind == "explicit" and not scope.write_limit else scope.kind
    return await _traced(graph, thread_id, "run_turn", body, scope_kind=kind)


async def resume(
    graph: CompiledStateGraph, *, thread_id: str, decision: ApprovalDecision
) -> TurnResult:
    """Entrega la decisión humana. Se valida antes de tocar el grafo; el nodo vuelve a validar."""

    async def body(run: RunContext) -> TurnResult:
        with thread_lock(_lock_dir(graph), thread_id):
            snapshot = await graph.aget_state(thread_config(thread_id))
            if not snapshot.interrupts:
                raise NoPendingApproval(f"el thread '{thread_id}' no espera ninguna aprobación")
            _check_owner(graph, snapshot.values)
            request = snapshot.interrupts[0].value
            if (decision.action_id, decision.args_hash) != (
                request["action_id"],
                request["args_hash"],
            ):
                raise ApprovalMismatch("la decisión no corresponde a la acción pendiente")
            command = Command(resume=decision.model_dump())
            return await _invoke(graph, command, thread_id, run)

    return await _traced(graph, thread_id, "resume", body)


async def continue_after_failure(graph: CompiledStateGraph, *, thread_id: str) -> TurnResult:
    """Retoma una ejecución interrumpida desde su último checkpoint (mismo nodo, misma acción).

    Nunca se saltea la aprobación: si el fallo ocurrió antes de aprobar, se vuelve a pedir; si
    ocurrió en execute_write, se reintenta la acción ya aprobada con la misma clave.
    """

    async def body(run: RunContext) -> TurnResult:
        with thread_lock(_lock_dir(graph), thread_id):
            snapshot = await graph.aget_state(thread_config(thread_id))
            _check_owner(graph, snapshot.values)
            if snapshot.interrupts:
                raise ThreadAwaitingApproval(
                    f"el thread '{thread_id}' espera una decisión, no un reintento"
                )
            if not snapshot.next:
                raise NothingToContinue(
                    f"el thread '{thread_id}' no tiene una ejecución interrumpida"
                )
            uncertain = snapshot.next == ("execute_write",)
            pending = snapshot.values.get("pending_write") if uncertain else None
            run.emit(
                "recovery_started",
                {
                    "prior_status": "write_uncertain" if uncertain else "interrupted",
                    "resume_node": snapshot.next[0],
                    "action_id": pending["action_id"] if pending else None,
                },
            )
            return await _invoke(graph, None, thread_id, run)

    return await _traced(graph, thread_id, "continue_after_failure", body)


async def turn_result(graph: CompiledStateGraph, thread_id: str) -> TurnResult:
    """Estado visible de un thread; solo para su dueño (incluye la solicitud de aprobación).

    Si otro proceso lo está ejecutando, devuelve ``BUSY`` en vez de un estado intermedio.
    """
    lock_dir = _lock_dir(graph)
    _check_owner(graph, (await graph.aget_state(thread_config(thread_id))).values)
    try:
        with thread_lock(lock_dir, thread_id):
            return await _result(graph, thread_id)
    except ThreadBusy:
        return TurnResult(TurnStatus.BUSY, "")


async def _result(graph: CompiledStateGraph, thread_id: str) -> TurnResult:
    """Interna: todos sus llamadores ya verificaron el dueño y tienen el lock."""
    snapshot = await graph.aget_state(thread_config(thread_id))
    values = snapshot.values
    replies = [m for m in values.get("messages", []) if isinstance(m, AIMessage)]
    reply = replies[-1].text if replies else ""
    if snapshot.interrupts:
        return TurnResult(
            TurnStatus.AWAITING_APPROVAL,
            reply,
            snapshot.interrupts[0].value,
            values.get("approval_error"),
        )
    if snapshot.next == ("execute_write",):  # la escritura aprobada no llegó a confirmarse
        pending = PendingWrite.model_validate(values["pending_write"])
        # La clave permite reconciliar a mano (ledger y auditoría del servidor la registran).
        action = pending.approval_request() | {
            "idempotency_key": pending.idempotency_key,
            "tool_call_id": pending.tool_call_id,
        }
        return TurnResult(TurnStatus.WRITE_UNCERTAIN, reply, pending_action=action)
    if snapshot.next or values.get("status") == "running":  # ejecución que no terminó
        return TurnResult(TurnStatus.INTERRUPTED, reply)
    return TurnResult(TurnStatus(values["status"]), reply)


def approve(request: dict[str, Any]) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=request["action_id"], args_hash=request["args_hash"], decision="approve"
    )


def reject(request: dict[str, Any], reason: str | None = None) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=request["action_id"],
        args_hash=request["args_hash"],
        decision="reject",
        reason=reason,
    )
