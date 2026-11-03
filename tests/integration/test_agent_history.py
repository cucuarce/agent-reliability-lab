"""Historial consistente: lo que recibe el modelo respeta el contrato de tool calling.

Contrato de los proveedores (Anthropic y OpenAI): cada tool call de un mensaje del asistente
recibe exactamente una respuesta, inmediatamente después de ese mensaje y antes de cualquier
otro mensaje de usuario o asistente. ``assert_provider_contract`` lo verifica de forma
independiente del código del agente.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.history import HistoryInconsistent
from arl.agent.runtime import (
    ThreadNeedsRecovery,
    TurnStatus,
    approve,
    continue_after_failure,
    resume,
    run_turn,
    thread_config,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, Step, call, say
from arl.agent.state import write_identity

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

THREAD = "t-hist"
KEY = write_identity(THREAD, "call_0_0")[1]


def assert_provider_contract(messages: list[BaseMessage]) -> None:
    pending: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, f"respuesta sin tool call: {message!r}"
            pending.remove(message.tool_call_id)
            continue
        assert not pending, f"tool calls sin respuesta antes de {type(message).__name__}: {pending}"
        if isinstance(message, AIMessage):
            pending = [tc["id"] for tc in message.tool_calls]
    assert not pending, f"tool calls sin respuesta al final: {pending}"


def _tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


async def _messages(graph) -> list[BaseMessage]:
    return (await graph.aget_state(thread_config(THREAD))).values["messages"]


async def test_new_turn_after_an_interrupted_read_closes_every_pending_call(
    open_faulty_agent, conn, invariants
) -> None:
    """Dos lecturas en un mensaje; la primera cae de forma persistente y la segunda nunca se
    ejecutó. Un turno nuevo cierra ambas como no ejecutadas, antes del mensaje nuevo."""
    model = ScriptedChatModel(
        [
            Step(calls=(("list_projects", {}), ("get_project", {"project_id": "apollo"}))),
            say("Nuevo turno."),
        ]
    )
    plan = {("list_projects", EVERY_CALL): Fault.DISCONNECT_BEFORE}
    async with open_faulty_agent("alice", model, plan) as (graph, faulty):
        first = await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
        assert first.status is TurnStatus.INTERRUPTED
        done = await run_turn(graph, thread_id=THREAD, text="dejalo, otra cosa", scope=READ_ONLY)
        messages = await _messages(graph)

    assert done.status is TurnStatus.COMPLETED
    for prompt in model.prompts:
        assert_provider_contract([m for m in prompt if m.type != "system"])
    assert_provider_contract(messages)
    closed = _tool_messages(messages)
    assert [m.tool_call_id for m in closed] == ["call_0_0", "call_0_1"]
    assert all(m.status == "error" and "[not_executed]" in m.text for m in closed)
    assert isinstance(messages[messages.index(closed[-1]) + 1], HumanMessage)
    assert [name for name, _ in faulty.attempts if name == "get_project"] == []
    invariants(conn, set())


async def test_new_turn_after_an_interrupted_write_preparation_closes_it_without_writing(
    open_faulty_agent, conn, invariants
) -> None:
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Otro tema.")]
    )
    plan = {("get_project", EVERY_CALL): Fault.TIMEOUT_BEFORE}
    async with open_faulty_agent("alice", model, plan) as (graph, faulty):
        assert (
            await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        ).status is (TurnStatus.INTERRUPTED)
        done = await run_turn(graph, thread_id=THREAD, text="mejor no", scope=READ_ONLY)
        state = await graph.aget_state(thread_config(THREAD))

    assert done.status is TurnStatus.COMPLETED
    assert_provider_contract(state.values["messages"])
    assert_provider_contract([m for m in model.prompts[-1] if m.type != "system"])
    (closed,) = _tool_messages(state.values["messages"])
    assert "[not_executed]" in closed.text
    assert state.values["pending_write"] is None
    assert [name for name, _ in faulty.attempts if name == "update_task"] == []
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == ("todo", 1)
    invariants(conn, set())


async def test_uncertain_write_is_never_closed_and_keeps_action_and_key(
    open_faulty_agent, conn, invariants
) -> None:
    """La escritura pudo haberse confirmado: no se inventa éxito ni fracaso. Un turno nuevo se
    rechaza sin tocar el historial; la acción y su clave quedan para reconciliar."""
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )
    plan = {("update_task", EVERY_CALL): Fault.DISCONNECT_AFTER}
    async with open_faulty_agent("alice", model, plan) as (graph, faulty):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        uncertain = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        before = await _messages(graph)
        with pytest.raises(ThreadNeedsRecovery):
            await run_turn(graph, thread_id=THREAD, text="olvidalo", scope=READ_ONLY)
        after = await _messages(graph)
        faulty.clear()
        done = await continue_after_failure(graph, thread_id=THREAD)
        final = await _messages(graph)

    assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
    assert uncertain.pending_action["idempotency_key"] == KEY
    assert after == before  # ni cierre inventado ni mensaje nuevo
    assert _tool_messages(before) == []
    assert done.status is TurnStatus.COMPLETED
    assert_provider_contract(final)
    keys = {args["idempotency_key"] for name, args in faulty.attempts if name == "update_task"}
    assert keys == {KEY}
    invariants(conn, {KEY})


async def test_the_model_is_never_called_with_an_unanswered_tool_call(open_faulty_agent) -> None:
    """Fuera del runtime (grafo invocado directo) el mensaje nuevo queda después de la tool call
    sin respuesta: ``call_model`` se niega antes de invocar al proveedor."""
    model = ScriptedChatModel([call("list_projects"), say("no debería llegar")])
    plan = {("list_projects", EVERY_CALL): Fault.DISCONNECT_BEFORE}
    async with open_faulty_agent("alice", model, plan) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
        prompts = len(model.prompts)
        with pytest.raises(HistoryInconsistent):
            await graph.ainvoke(
                {"messages": [HumanMessage("hola")], "principal_id": "alice"},
                thread_config(THREAD),
            )
    assert len(model.prompts) == prompts
