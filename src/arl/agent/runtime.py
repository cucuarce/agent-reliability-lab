"""Interfaz pública y confiable del agente: iniciar un turno, consultar su estado y reanudar.

Es la frontera del modelo de seguridad del MVP. Toda función verifica, antes de leer o escribir,
que el thread pertenezca al principal del proceso. La identidad no es un parámetro: es la del
proceso, ligada al grafo en ``build_graph`` a partir del gateway MCP. Invocar el grafo
directamente queda fuera del modelo de seguridad (ver docs/agent.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from arl.agent.state import ApprovalDecision, ApprovalMismatch, PrincipalMismatch

# Cada superpaso se persiste antes del siguiente: tras un crash se reanuda desde el último nodo
# completado y nunca se pierde una aprobación ya registrada.
DURABILITY = "sync"


class TurnStatus(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    TOOL_LIMIT = "tool_limit"
    TURN_LIMIT = "turn_limit"


class ThreadAwaitingApproval(RuntimeError):
    """No se acepta un mensaje nuevo mientras hay una escritura esperando decisión."""


class NoPendingApproval(RuntimeError):
    """Se intentó reanudar un thread que no está esperando aprobación."""


@dataclass(frozen=True)
class TurnResult:
    status: TurnStatus
    reply: str
    approval_request: dict[str, Any] | None = None
    approval_error: str | None = None  # por qué se rechazó la última decisión entregada


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


async def run_turn(graph: CompiledStateGraph, *, thread_id: str, text: str) -> TurnResult:
    config = thread_config(thread_id)
    snapshot = await graph.aget_state(config)
    principal_id = _check_owner(graph, snapshot.values)  # antes de escribir nada en el thread
    if snapshot.interrupts:
        raise ThreadAwaitingApproval(f"el thread '{thread_id}' tiene una aprobación pendiente")
    payload = {"messages": [HumanMessage(text)], "principal_id": principal_id}
    await graph.ainvoke(payload, config, durability=DURABILITY)
    return await turn_result(graph, thread_id)


async def resume(
    graph: CompiledStateGraph, *, thread_id: str, decision: ApprovalDecision
) -> TurnResult:
    """Entrega la decisión humana. Se valida antes de tocar el grafo; el nodo vuelve a validar."""
    config = thread_config(thread_id)
    snapshot = await graph.aget_state(config)
    if not snapshot.interrupts:
        raise NoPendingApproval(f"el thread '{thread_id}' no espera ninguna aprobación")
    _check_owner(graph, snapshot.values)
    request = snapshot.interrupts[0].value
    if (decision.action_id, decision.args_hash) != (request["action_id"], request["args_hash"]):
        raise ApprovalMismatch("la decisión no corresponde a la acción pendiente")
    await graph.ainvoke(Command(resume=decision.model_dump()), config, durability=DURABILITY)
    return await turn_result(graph, thread_id)


async def continue_after_failure(graph: CompiledStateGraph, *, thread_id: str) -> TurnResult:
    """Reintenta desde el último checkpoint (p. ej. tras un crash dentro de execute_write)."""
    config = thread_config(thread_id)
    _check_owner(graph, (await graph.aget_state(config)).values)
    await graph.ainvoke(None, config, durability=DURABILITY)
    return await turn_result(graph, thread_id)


async def turn_result(graph: CompiledStateGraph, thread_id: str) -> TurnResult:
    """Estado visible de un thread; solo para su dueño (incluye la solicitud de aprobación)."""
    snapshot = await graph.aget_state(thread_config(thread_id))
    values = snapshot.values
    _check_owner(graph, values)
    replies = [m for m in values.get("messages", []) if isinstance(m, AIMessage)]
    reply = replies[-1].text if replies else ""
    if snapshot.interrupts:
        return TurnResult(
            TurnStatus.AWAITING_APPROVAL,
            reply,
            snapshot.interrupts[0].value,
            values.get("approval_error"),
        )
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
