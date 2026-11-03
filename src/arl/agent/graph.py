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
Ningún nodo decide permisos: la autorización es exclusiva del servidor MCP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import ValidationError

from arl.agent.gateway import McpGateway, ToolCallFailed, ToolResult
from arl.agent.history import check_tool_protocol
from arl.agent.locks import LockingRequired
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
    InvalidToolCall,
    ToolKind,
    UpdateTaskArgs,
    tool_definitions,
    validate_call,
)

SYSTEM_PROMPT = (
    "Sos un asistente de gestión de proyectos con datos sintéticos. La identidad del usuario la "
    "fija el sistema: nunca la pidas ni la incluyas en argumentos. Usá list_projects y "
    "get_project para consultar. Para modificar una tarea usá update_task: un humano debe "
    "aprobar cada cambio antes de que se ejecute. Si una tool devuelve ERROR, explicalo sin "
    "inventar datos."
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
) -> CompiledStateGraph:
    """``lock_dir``: directorio de locks por thread (ver ``arl.agent.locks``). Obligatorio: sin él,
    dos procesos podrían ejecutar el mismo thread a la vez sobre el mismo checkpointer."""
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

    def begin_turn(state: AgentState) -> dict[str, Any]:
        check_principal(state)
        return {
            "queue": [],
            "pending_write": None,
            "approval": None,
            "approval_error": None,
            "tool_calls_used": 0,
            "model_turns": 0,
            "status": "running",
        }

    def dispatch(state: AgentState) -> dict[str, Any]:
        """Descarta tool calls inválidas y aplica el límite; no ejecuta nada."""
        queue = list(state["queue"])
        used = state["tool_calls_used"]
        messages: list[BaseMessage] = []
        while queue:
            if used >= limits.max_tool_calls:
                messages += [
                    _error_message(call, "tool_limit", "límite de tool calls alcanzado")
                    for call in queue
                ]
                messages.append(AIMessage(TOOL_LIMIT_REPLY))
                return {"queue": [], "messages": messages, "tool_calls_used": used,
                        "status": "tool_limit"}  # fmt: skip
            try:
                validate_call(queue[0]["name"], queue[0]["args"])
                break
            except InvalidToolCall as exc:
                messages.append(_error_message(queue.pop(0), "invalid_tool_call", str(exc)))
                used += 1
        return {"queue": queue, "messages": messages, "tool_calls_used": used}

    def await_approval(state: AgentState) -> dict[str, Any]:
        """Solo interrupt(): se re-ejecuta al reanudar, por eso no tiene efectos previos.

        Una decisión inválida NO lanza excepción: LangGraph conservaría el valor de resume
        rechazado y lo volvería a entregar en la próxima reanudación. En cambio se registra
        ``approval_error`` y la ruta vuelve a este nodo, que pide la aprobación de nuevo.
        """
        pending = PendingWrite.model_validate(state["pending_write"])
        raw = interrupt(pending.approval_request())
        if state.get("principal_id") != gateway.principal_id:
            return {"approval": None, "approval_error": "principal distinto del dueño del thread"}
        try:
            decision = ApprovalDecision.model_validate(raw)
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) or "(raíz)" for e in exc.errors())
            return {"approval": None, "approval_error": f"decisión inválida: {fields}"}
        if not decision.matches(pending):
            return {"approval": None, "approval_error": "la decisión corresponde a otra acción"}
        return {"approval": decision.model_dump(), "approval_error": None}

    def record_rejection(state: AgentState) -> dict[str, Any]:
        pending = PendingWrite.model_validate(state["pending_write"])
        decision = ApprovalDecision.model_validate(state["approval"])
        reason = decision.reason or "sin motivo"
        message = ToolMessage(
            f"ERROR [rejected_by_user] el usuario rechazó la modificación de "
            f"'{pending.task_id}'. Motivo: {reason}",
            tool_call_id=pending.tool_call_id,
            name=pending.tool,
            status="error",
        )
        return {"messages": [message], "pending_write": None, "approval": None}

    # --- Nodos con efectos (lecturas MCP o la única escritura) -----------------------------------

    async def call_model(state: AgentState) -> dict[str, Any]:
        if state["model_turns"] >= limits.max_model_turns:
            return {"messages": [AIMessage(TURN_LIMIT_REPLY)], "queue": [], "status": "turn_limit"}
        # Última barrera: un proveedor real rechaza tool calls sin respuesta; nunca se le envían.
        check_tool_protocol(state["messages"])
        response = await llm.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        queue = [
            {"id": call["id"], "name": call["name"], "args": call["args"]}
            for call in response.tool_calls
        ]
        return {
            "messages": [response],
            "queue": queue,
            "model_turns": state["model_turns"] + 1,
            "status": "running" if queue else "completed",
        }

    async def execute_read(state: AgentState) -> dict[str, Any]:
        check_principal(state)
        call, *rest = state["queue"]
        _, args = validate_call(call["name"], call["args"])
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

        located = await _locate_task(gateway, args.task_id)
        if located is None:
            detail = f"tarea '{args.task_id}' no encontrada"
            return update | {"messages": [_error_message(call, "not_found", detail)]}
        project_id, my_role, task = located

        changes = args.changes.changes()
        action_id, key = write_identity(config["configurable"]["thread_id"], call["id"])
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
        return update | {
            "pending_write": pending.model_dump(),
            "approval": None,
            "approval_error": None,
        }

    async def execute_write(state: AgentState) -> dict[str, Any]:
        """La única escritura. Usa la acción aprobada y sellada, nunca argumentos del LLM."""
        check_principal(state)
        pending = PendingWrite.model_validate(state["pending_write"])
        decision = ApprovalDecision.model_validate(state["approval"])
        if decision.decision != "approve" or not decision.matches(pending):
            raise ApprovalMismatch("execute_write sin una aprobación válida para esta acción")
        result = await gateway.call(pending.tool, pending.server_arguments())
        return {
            "messages": [_tool_message(pending.tool_call_id, pending.tool, result)],
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
            },
        }
    )


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
