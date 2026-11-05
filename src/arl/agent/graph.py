"""StateGraph del agente con aprobación humana de escrituras.

Flujo::

    START → begin_turn → call_model ─┬→ END (sin tool calls / límite de turnos)
                                     └→ dispatch
    dispatch ─┬→ execute_read ─→ dispatch
              ├→ prepare_write ─┬→ dispatch (tarea no visible)
              │                 └→ await_approval ─┬→ execute_write ────→ dispatch
              │                                    ├→ record_rejection ─→ dispatch
              │                                    └→ await_approval (decisión inválida)
              ├→ call_model (cola vacía)
              └→ END (límite de tool calls)

Efectos secundarios: ``execute_read`` (lectura, re-ejecutable), ``prepare_write`` (solo
lecturas) y ``execute_write`` (la única escritura, idempotente por clave). ``await_approval``
contiene únicamente ``interrupt()``: al reanudar se re-ejecuta desde el inicio sin efectos.
Ningún nodo decide permisos: la autorización es exclusiva del servidor MCP. El **alcance** del
turno (``arl.agent.scope``) es otra cosa: lo que la aplicación autorizó escribir. ``dispatch``
bloquea lo que queda fuera antes de pedir aprobación, ``prepare_write`` reserva el permiso y
``execute_write`` verifica la reserva antes de enviar la escritura.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import ValidationError

from arl.agent import outcomes
from arl.agent.gateway import McpGateway, ToolCallFailed, ToolResult, domain_code
from arl.agent.history import check_tool_protocol, normalize_response
from arl.agent.locks import LockingRequired
from arl.agent.scope import OutOfScope, ScopeViolation, load_bound, settle_state
from arl.agent.state import (
    AgentState,
    ApprovalDecision,
    ApprovalMismatch,
    Limits,
    PendingWrite,
    PrincipalMismatch,
    write_identity,
)
from arl.agent.tool_registry import (
    REGISTRY,
    GetProjectArgs,
    InvalidToolCall,
    ToolKind,
    UpdateTaskArgs,
    tool_definitions,
    validate_call,
)
from arl.tracing import recorder as tracing

SYSTEM_PROMPT = (
    "Sos un asistente de gestión de proyectos con datos sintéticos. La identidad del usuario la "
    "fija el sistema: nunca la pidas ni la incluyas en argumentos. Usá list_projects y "
    "get_project para consultar y update_task para proponer un cambio. La aprobación de cada "
    "cambio la pide el sistema al usuario fuera de este chat: nunca la pidas vos ni preguntes si "
    "confirma. Cuando update_task responde, la decisión ya se tomó: ERROR [rejected_by_user] es "
    "un rechazo del usuario a esa propuesta (no se aplicó ni queda pendiente: no vuelvas a pedir "
    "esa aprobación); [forbidden] es falta de permiso; [out_of_scope] es un cambio fuera de lo "
    "que pidió el usuario; un fallo técnico no confirma nada. Solo digas que un cambio se hizo "
    "si update_task lo confirmó. Si una tool devuelve ERROR, explicalo sin inventar datos."
)
TOOL_LIMIT_REPLY = "Detuve la ejecución: se alcanzó el límite de tool calls para este pedido."
TURN_LIMIT_REPLY = "Detuve la ejecución: se alcanzó el límite de pasos del modelo para este pedido."


class ToolCallingModel(Protocol):
    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> Any: ...


def is_retryable(exc: Exception) -> bool:
    """Solo fallos transitorios del MCP. Los resultados de dominio (forbidden, version_conflict,
    rechazo humano) no son excepciones y nunca se reintentan."""
    return isinstance(exc, ToolCallFailed) and exc.retryable


def _tool_message(tool_call_id: str, name: str, result: ToolResult) -> ToolMessage:
    if result.is_error:
        return ToolMessage(
            f"ERROR {result.text}", tool_call_id=tool_call_id, name=name, status="error"
        )
    return ToolMessage(result.content(), tool_call_id=tool_call_id, name=name)


def _error_message(call: dict[str, Any], code: str, detail: str) -> ToolMessage:
    return ToolMessage(
        f"ERROR [{code}] {detail}", tool_call_id=call["id"], name=call["name"], status="error"
    )


def build_graph(
    model: ToolCallingModel,
    gateway: McpGateway,
    checkpointer: BaseCheckpointSaver,
    limits: Limits | None = None,
    *,
    lock_dir: Path,
    trace_dir: Path | None = None,
) -> CompiledStateGraph:
    """``lock_dir``: directorio de locks por thread (ver ``arl.agent.locks``). Obligatorio: sin él,
    dos procesos podrían ejecutar el mismo thread a la vez sobre el mismo checkpointer.

    ``trace_dir``: trazas JSONL de cada ejecución (``arl.tracing``). Opcional y best-effort: sin
    él no se registran eventos, y un fallo al escribirlos nunca cambia el resultado."""
    if lock_dir is None:
        raise LockingRequired("build_graph requiere lock_dir (lock por thread entre procesos)")
    limits = limits or Limits()
    llm = model.bind_tools(tool_definitions())

    def check_principal(state: AgentState) -> None:
        if state.get("principal_id") != gateway.principal_id:
            raise PrincipalMismatch(
                "el thread no pertenece a la identidad con la que se ejecuta este proceso"
            )

    # --- Nodos puros ----------------------------------------------------------------------------

    def bound_scope(state: AgentState, config: RunnableConfig):
        bound = load_bound(state.get("request_scope"))
        bound.verify(thread_id=config["configurable"]["thread_id"],
                     principal_id=gateway.principal_id)  # fmt: skip
        return bound

    def begin_turn(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        check_principal(state)
        bound_scope(state, config)  # un alcance alterado o ajeno detiene el turno
        return {
            "queue": [],
            "pending_write": None,
            "approval": None,
            "approval_error": None,
            "tool_calls_used": 0,
            "model_turns": 0,
            "status": "running",
        }

    def dispatch(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Descarta tool calls inválidas o fuera del alcance y aplica el límite; no ejecuta nada.

        Una escritura fuera del alcance se bloquea acá: nunca llega a ``prepare_write`` ni al
        humano. El motivo es un código fijo, sin los valores pedidos."""
        bound = bound_scope(state, config)
        ledger = state.get("scope_ledger") or {}
        queue = list(state["queue"])
        used = state["tool_calls_used"]
        messages: list[BaseMessage] = []
        written = dict(state.get("write_outcomes") or {})
        while queue:
            if used >= limits.max_tool_calls:
                for call in queue:
                    _trace_route(call, "tool_limit")
                    _unprepared(written, call, "tool_limit", "not_executed")
                messages += [
                    _error_message(call, "tool_limit", "límite de tool calls alcanzado")
                    for call in queue
                ]
                messages.append(AIMessage(TOOL_LIMIT_REPLY))
                return {"queue": [], "messages": messages, "tool_calls_used": used,
                        "status": "tool_limit", "write_outcomes": written}  # fmt: skip
            try:
                spec, args = validate_call(queue[0]["name"], queue[0]["args"])
            except InvalidToolCall as exc:
                _trace_route(queue[0], "invalid_tool_call")
                _unprepared(written, queue[0], "invalid_tool_call", "not_executed")
                messages.append(_error_message(queue.pop(0), "invalid_tool_call", str(exc)))
                used += 1
                continue
            if spec.kind is ToolKind.WRITE:
                write = cast(UpdateTaskArgs, args)
                try:
                    bound.check(write.task_id, write.changes.changes(), ledger)
                except OutOfScope as exc:
                    _trace_route(queue[0], "out_of_scope", args, scope_reason=exc.reason)
                    _unprepared(written, queue[0], "out_of_scope", "out_of_scope",
                                write.task_id, write.changes.changes())  # fmt: skip
                    messages.append(_error_message(queue.pop(0), "out_of_scope",
                                                   f"({exc.reason}) {exc}"))  # fmt: skip
                    used += 1
                    continue
            decision = "execute" if spec.kind is ToolKind.READ else "requires_approval"
            _trace_route(queue[0], decision, args)
            break
        return {"queue": queue, "messages": messages, "tool_calls_used": used,
                "write_outcomes": written}  # fmt: skip

    def await_approval(state: AgentState) -> dict[str, Any]:
        """Solo interrupt(): se re-ejecuta al reanudar, por eso no tiene efectos previos.

        Una decisión inválida NO lanza excepción: LangGraph conservaría el valor de resume
        rechazado y lo volvería a entregar en la próxima reanudación. En cambio se registra
        ``approval_error`` y la ruta vuelve a este nodo, que pide la aprobación de nuevo.
        """
        pending = PendingWrite.model_validate(state["pending_write"])
        raw = interrupt(pending.approval_request())
        ids = {
            "action_id": pending.action_id,
            "tool_call_id": tracing.safe_id(pending.tool_call_id),
        }
        invalid = ids | {"decision": "invalid"}
        if state.get("principal_id") != gateway.principal_id:
            tracing.emit("approval_decided", **invalid, invalid_reason="principal")
            return {"approval": None, "approval_error": "principal distinto del dueño del thread"}
        try:
            decision = ApprovalDecision.model_validate(raw)
        except ValidationError as exc:
            tracing.emit("approval_decided", **invalid, invalid_reason="schema")
            fields = ", ".join(".".join(map(str, e["loc"])) or "(raíz)" for e in exc.errors())
            return {"approval": None, "approval_error": f"decisión inválida: {fields}"}
        if not decision.matches(pending):
            tracing.emit("approval_decided", **invalid, invalid_reason="other_action")
            return {"approval": None, "approval_error": "la decisión corresponde a otra acción"}
        tracing.emit(
            "approval_decided",
            **ids,
            decision=decision.decision,
            with_reason=decision.reason is not None,
        )
        return {"approval": decision.model_dump(), "approval_error": None}

    def record_rejection(state: AgentState) -> dict[str, Any]:
        pending = PendingWrite.model_validate(state["pending_write"])
        decision = ApprovalDecision.model_validate(state["approval"])
        reason = decision.reason or "sin motivo"
        ledger = _settle(state, pending, "released")  # nunca se envió: el permiso vuelve
        written = _settled(state, pending, "rejected_by_user", "rejected", "reject")
        result = json.dumps(outcomes.public(written[pending.action_id]), ensure_ascii=False)
        message = ToolMessage(
            f"ERROR [rejected_by_user] el usuario rechazó la modificación de "
            f"'{pending.task_id}' y la decisión ya está tomada: no se aplicó ni queda pendiente; "
            f"no vuelvas a pedir esta aprobación. Motivo: {reason}. Resultado: {result}",
            tool_call_id=pending.tool_call_id,
            name=pending.tool,
            status="error",
        )
        return {"messages": [message], "pending_write": None, "approval": None,
                "scope_ledger": ledger, "write_outcomes": written}  # fmt: skip

    # --- Nodos con efectos (lecturas MCP o la única escritura) -----------------------------------

    async def call_model(state: AgentState) -> dict[str, Any]:
        if state["model_turns"] >= limits.max_model_turns:
            return {"messages": [AIMessage(TURN_LIMIT_REPLY)], "queue": [], "status": "turn_limit"}
        # Última barrera: un proveedor real rechaza tool calls sin respuesta; nunca se le envían.
        check_tool_protocol(state["messages"])
        turn = state["model_turns"] + 1
        start = time.perf_counter()
        tracing.count("model_calls")
        try:
            response = await llm.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        except Exception as exc:
            tracing.emit(
                "model_call", model_turn=turn, duration_ms=tracing.elapsed_ms(start),
                outcome="error", error_type=type(exc).__name__,
            )  # fmt: skip
            raise
        # La identidad de las tool calls la fija el agente (riesgo R1): nunca el id del proveedor.
        response = normalize_response(response, state["messages"])
        tracing.emit(
            "model_call",
            model_turn=turn,
            duration_ms=tracing.elapsed_ms(start),
            outcome="ok",
            tool_calls=[
                {"tool_call_id": tracing.safe_id(c["id"]), "tool": tracing.safe_id(c["name"])}
                for c in response.tool_calls
            ],
        )
        queue = [
            {"id": call["id"], "name": call["name"], "args": call["args"]}
            for call in response.tool_calls
        ]
        return {
            "messages": [response],
            "queue": queue,
            "model_turns": turn,
            "status": "running" if queue else "completed",
        }

    async def execute_read(state: AgentState) -> dict[str, Any]:
        check_principal(state)
        call, *rest = state["queue"]
        _, args = validate_call(call["name"], call["args"])
        with tracing.step("execute_read", call["id"]):
            result = await gateway.call(call["name"], args.model_dump(mode="json"))
        return {
            "queue": rest,
            "messages": [_tool_message(call["id"], call["name"], result)],
            "tool_calls_used": state["tool_calls_used"] + 1,
        }

    async def prepare_write(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Lee el estado actual de la tarea y sella la acción. No escribe."""
        check_principal(state)
        call, *rest = state["queue"]
        args = cast(UpdateTaskArgs, validate_call(call["name"], call["args"])[1])
        update: dict[str, Any] = {"queue": rest, "tool_calls_used": state["tool_calls_used"] + 1}

        with tracing.step("prepare_write", call["id"]):
            located = await _locate_task(gateway, args.task_id)
        if located is None:
            detail = f"tarea '{args.task_id}' no encontrada"
            written = dict(state.get("write_outcomes") or {})
            _unprepared(written, call, "not_found", "not_found", args.task_id,
                        args.changes.changes())  # fmt: skip
            return update | {"messages": [_error_message(call, "not_found", detail)],
                             "write_outcomes": written}  # fmt: skip
        project_id, my_role, task = located

        changes = args.changes.changes()
        action_id, key = write_identity(config["configurable"]["thread_id"], call["id"])
        # Se vuelve a verificar al reservar (sin contar la propia reserva si el nodo se reintenta).
        ledger = {k: v for k, v in (state.get("scope_ledger") or {}).items() if k != action_id}
        try:
            grant_id = bound_scope(state, config).check(args.task_id, changes, ledger)
        except OutOfScope as exc:
            _trace_route(call, "out_of_scope", args, scope_reason=exc.reason)
            written = dict(state.get("write_outcomes") or {})
            _unprepared(written, call, "out_of_scope", "out_of_scope", args.task_id, changes)
            return update | {"messages": [_error_message(call, "out_of_scope",
                                                         f"({exc.reason}) {exc}")],
                             "write_outcomes": written}  # fmt: skip
        pending = PendingWrite.seal(
            action_id=action_id,
            tool_call_id=call["id"],
            tool="update_task",
            task_id=args.task_id,
            project_id=project_id,
            my_role=my_role,
            changes=changes,
            before={field: task[field] for field in changes},
            expected_version=task["version"],
            idempotency_key=key,
        )
        tracing.emit(
            "approval_requested",
            **pending.model_dump(include={"action_id", "args_hash", "task_id", "project_id",
                                          "my_role", "expected_version", "tool"}),
            tool_call_id=tracing.safe_id(call["id"]),
            changed_fields=sorted(changes),
        )  # fmt: skip
        entry = {"grant_id": grant_id, "state": "reserved", "task_id": args.task_id,
                 "args_hash": pending.args_hash}  # fmt: skip
        _trace_reservation(action_id, entry)
        return update | {
            "pending_write": pending.model_dump(),
            "approval": None,
            "approval_error": None,
            "scope_ledger": {**ledger, action_id: entry},
        }

    async def execute_write(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """La única escritura. Usa la acción aprobada y sellada, nunca argumentos del LLM.

        Un reintento (LangGraph o ``continue_after_failure``) es la misma acción con la misma
        reserva; si la escritura queda incierta, la reserva sigue tomada."""
        check_principal(state)
        pending = PendingWrite.model_validate(state["pending_write"])
        decision = ApprovalDecision.model_validate(state["approval"])
        if decision.decision != "approve" or not decision.matches(pending):
            raise ApprovalMismatch("execute_write sin una aprobación válida para esta acción")
        bound_scope(state, config)
        reservation = (state.get("scope_ledger") or {}).get(pending.action_id)
        if reservation is None or (reservation["state"], reservation["args_hash"]) != (
            "reserved",
            pending.args_hash,
        ):
            raise ScopeViolation("la acción aprobada no tiene una reserva del alcance del turno")
        with tracing.step("execute_write", pending.tool_call_id):
            result = await gateway.call(pending.tool, pending.server_arguments())
        _trace_write(pending, result)
        outcome = (
            domain_code(pending.tool, result.text)
            if result.is_error or result.data is None
            else result.data["outcome"]
        )
        replayed = bool(not result.is_error and result.data and result.data.get("replayed"))
        return {
            "messages": [_tool_message(pending.tool_call_id, pending.tool, result)],
            "scope_ledger": _settle(state, pending, settle_state(outcome)),
            "write_outcomes": _settled(
                state,
                pending,
                outcome or "unknown",
                outcomes.server_state(outcome),
                "approve",
                replayed=replayed,
            ),
            "pending_write": None,
            "approval": None,
        }

    # --- Rutas ------------------------------------------------------------------------------------

    def after_model(state: AgentState) -> str:
        return "dispatch" if state["queue"] else END

    def after_dispatch(state: AgentState) -> str:
        if state["status"] == "tool_limit":
            return END
        if not state["queue"]:
            return "call_model"
        kind = REGISTRY[state["queue"][0]["name"]].kind
        return "execute_read" if kind is ToolKind.READ else "prepare_write"

    def after_prepare(state: AgentState) -> str:
        return "await_approval" if state.get("pending_write") else "dispatch"

    def after_approval(state: AgentState) -> str:
        if state["approval"] is None:
            return "await_approval"  # decisión inválida: se vuelve a pedir
        return "execute_write" if state["approval"]["decision"] == "approve" else "record_rejection"

    # Solo los nodos que hablan con el MCP se reintentan, y solo ante fallos transitorios.
    # execute_write reintenta con la misma PendingWrite (misma clave): el servidor deduplica.
    mcp_retry = RetryPolicy(
        max_attempts=limits.max_attempts,
        initial_interval=limits.retry_initial_interval,
        backoff_factor=2.0,
        max_interval=limits.retry_max_interval,
        jitter=limits.retry_jitter,
        retry_on=is_retryable,
    )
    builder = StateGraph(AgentState)
    for node in (begin_turn, call_model, dispatch, await_approval, record_rejection):
        builder.add_node(node.__name__, node)
    for node in (execute_read, prepare_write, execute_write):
        builder.add_node(node.__name__, node, retry_policy=mcp_retry)
    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "call_model")
    builder.add_conditional_edges("call_model", after_model, ["dispatch", END])
    builder.add_conditional_edges(
        "dispatch", after_dispatch, ["execute_read", "prepare_write", "call_model", END]
    )
    builder.add_edge("execute_read", "dispatch")
    builder.add_conditional_edges("prepare_write", after_prepare, ["await_approval", "dispatch"])
    builder.add_conditional_edges(
        "await_approval", after_approval, ["execute_write", "record_rejection", "await_approval"]
    )
    builder.add_edge("execute_write", "dispatch")
    builder.add_edge("record_rejection", "dispatch")
    # El grafo queda ligado a la identidad del proceso: el runtime la toma de aquí, no del caller.
    return builder.compile(checkpointer=checkpointer).with_config(
        {
            "recursion_limit": limits.recursion_limit,
            "configurable": {
                "principal_id": gateway.principal_id,
                "lock_dir": str(lock_dir),
                "trace_dir": str(trace_dir) if trace_dir else None,
            },
        }
    )


def _settled(
    state: AgentState,
    pending: PendingWrite,
    code: str,
    outcome_state: str,
    approval: str,
    *,
    replayed: bool = False,
) -> dict[str, dict[str, Any]]:
    """El resultado de una acción preparada: decisión validada o respuesta del servidor."""
    written = dict(state.get("write_outcomes") or {})
    previous = written.get(pending.action_id)
    written[pending.action_id] = outcomes.entry(
        previous["seq"] if previous else len(written) + 1,
        action_id=pending.action_id,
        tool_call_id=pending.tool_call_id,
        task_id=pending.task_id,
        changes=pending.changes,
        code=code,
        state=outcome_state,
        approval=approval,
        replayed=replayed,
    )
    return written


def _unprepared(
    written: dict[str, dict[str, Any]],
    call: dict[str, Any],
    code: str,
    outcome_state: str,
    task_id: Any = None,
    changes: Any = None,
) -> None:
    """Una escritura que nunca llegó a prepararse: sin action_id ni aprobación."""
    if call.get("name") != "update_task":
        return
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    written[f"call:{call['id']}"] = outcomes.entry(
        len(written) + 1,
        action_id=None,
        tool_call_id=call["id"],
        task_id=task_id if task_id is not None else args.get("task_id"),
        changes=changes if changes is not None else args.get("changes"),
        code=code,
        state=outcome_state,
    )


def _settle(state: AgentState, pending: PendingWrite, new_state: str) -> dict[str, Any]:
    """Cierra la reserva de la acción (consumida, liberada, rechazada o invalidada)."""
    ledger = dict(state.get("scope_ledger") or {})
    entry = ledger.get(pending.action_id)
    if entry is None:
        return ledger
    ledger[pending.action_id] = {**entry, "state": new_state}
    _trace_reservation(pending.action_id, ledger[pending.action_id])
    return ledger


def _trace_reservation(action_id: str, entry: dict[str, Any]) -> None:
    tracing.emit(
        "scope_reservation",
        action_id=action_id,
        grant_id=entry["grant_id"],
        task_id=tracing.safe_id(entry["task_id"]),
        state=entry["state"],
    )


def _trace_route(
    call: dict[str, Any], decision: str, args: Any = None, *, scope_reason: str | None = None
) -> None:
    """Evento ``tool_call_routed``: lo que pidió el LLM, sin valores (solo ids y nombres)."""
    data: dict[str, Any] = {
        "tool_call_id": tracing.safe_id(call["id"]),
        "tool": tracing.safe_id(call["name"]),
        "decision": decision,
        "scope_reason": scope_reason,
    }
    if isinstance(args, GetProjectArgs):
        data["project_id"] = args.project_id
    elif isinstance(args, UpdateTaskArgs):
        data |= {"task_id": args.task_id, "changed_fields": sorted(args.changes.changes())}
    tracing.emit("tool_call_routed", **data)


def _trace_write(pending: PendingWrite, result: ToolResult) -> None:
    """Evento ``write_result``: lo que el servidor confirma que ejecutó (o rechazó)."""
    data: dict[str, Any] = {
        "action_id": pending.action_id,
        "idempotency_key": pending.idempotency_key,
        "tool_call_id": tracing.safe_id(pending.tool_call_id),
        "task_id": pending.task_id,
    }
    if result.is_error or result.data is None:
        data["outcome"] = domain_code(pending.tool, result.text) or "error"
    else:
        data |= {
            "outcome": result.data["outcome"],
            "replayed": result.data["replayed"],
            "version": result.data["task"]["version"],
            "audit_id": result.data["audit_id"],
        }
    tracing.emit("write_result", **data)


async def _locate_task(gateway: McpGateway, task_id: str) -> tuple[str, str, dict[str, Any]] | None:
    """Busca la tarea entre los proyectos visibles (lecturas MCP; sin permisos propios)."""
    listing = await gateway.call("list_projects", {})
    if listing.is_error or listing.data is None:
        return None
    for project in listing.data["result"]:
        detail = await gateway.call("get_project", {"project_id": project["id"]})
        if detail.is_error or detail.data is None:  # p. ej. acceso revocado entre lecturas
            continue
        for task in detail.data["tasks"]:
            if task["id"] == task_id:
                return project["id"], detail.data["my_role"], task
    return None
