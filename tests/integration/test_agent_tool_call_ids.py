"""Identidad de las tool calls con proveedores reales (riesgo R1 de la etapa 6B).

``action_id`` y la clave de idempotencia se derivan del thread y del ``tool_call_id``. Un
proveedor real puede repetir ids (el mismo ``call_0`` en cada respuesta, o dos veces en la misma)
y el mismo id de mensaje. Si el agente los usara tal cual, dos operaciones lógicas distintas
compartirían acción y clave, y un mensaje nuevo reemplazaría a uno anterior en el historial.
El agente fija su propia identidad: ``call_<n.º de mensajes del asistente previos>_<índice>``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.history import check_tool_protocol
from arl.agent.runtime import (
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

pytestmark = pytest.mark.anyio

THREAD = "t-ids"
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))
REOPEN_APO3 = RequestScope.writes(allow_update("apo-3", status=["in_progress"]))


@dataclass
class ProviderLikeModel(ScriptedChatModel):
    """Guion como ``ScriptedChatModel``, pero con los ids que elegiría un proveedor descuidado."""

    call_id: str | None = "call_0"  # None: conserva los ids del guion
    message_id: str | None = "resp-1"  # None: sin id de mensaje

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        scripted = await super().ainvoke(messages, config)
        calls = [{**c, "id": self.call_id or c["id"]} for c in scripted.tool_calls]
        raw = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
            for c in calls
        ]  # fmt: skip
        return AIMessage(
            content=scripted.content,
            tool_calls=calls,
            id=self.message_id,
            additional_kwargs={"tool_calls": raw} if raw else {},
            response_metadata={"model_name": "fake-provider", "finish_reason": "tool_calls"},
        )


def _audit(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [tuple(r) for r in conn.execute("SELECT task_id, idempotency_key FROM audit_log")]


async def _messages(graph) -> list[BaseMessage]:
    return (await graph.aget_state(thread_config(THREAD))).values["messages"]


def _pairs(messages: list[BaseMessage]) -> tuple[list[str], list[str]]:
    calls = [c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    answers = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    return calls, answers


async def test_repeated_provider_ids_across_turns_are_distinct_operations(open_agent, conn) -> None:
    """Dos turnos, dos escrituras distintas, el mismo id del proveedor: ambas se aplican."""
    model = ProviderLikeModel([
        call("update_task", task_id="apo-3", changes={"status": "done"}), say("cerrada"),
        call("update_task", task_id="apo-3", changes={"status": "in_progress"}), say("reabierta"),
    ])  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        first = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id=THREAD, decision=approve(first.approval_request))
        assert done.status is TurnStatus.COMPLETED
        second = await run_turn(graph, thread_id=THREAD, text="reabrila", scope=REOPEN_APO3)
        assert second.approval_request["action_id"] != first.approval_request["action_id"]
        again = await resume(graph, thread_id=THREAD, decision=approve(second.approval_request))
        assert again.status is TurnStatus.COMPLETED
        messages = await _messages(graph)
    rows = _audit(conn)
    assert [task for task, _ in rows] == ["apo-3", "apo-3"]
    assert len({key for _, key in rows}) == 2  # claves de idempotencia distintas
    status = conn.execute("SELECT status FROM tasks WHERE id = 'apo-3'").fetchone()[0]
    assert status == "in_progress"
    calls, answers = _pairs(messages)
    assert calls == answers == ["call_0_0", "call_2_0"]
    check_tool_protocol(messages)


async def test_repeated_message_ids_never_replace_earlier_history(open_agent) -> None:
    model = ProviderLikeModel(
        [call("get_project", project_id="apollo"), say("Apollo tiene 4 tareas.")], call_id=None
    )
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id=THREAD, text="resumime apollo",
                                scope=RequestScope.read_only())  # fmt: skip
        messages = await _messages(graph)
    assert result.status is TurnStatus.COMPLETED and result.reply == "Apollo tiene 4 tareas."
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]  # fmt: skip
    check_tool_protocol(messages)


async def test_duplicate_ids_in_one_response_get_one_answer_each(open_agent) -> None:
    model = ProviderLikeModel(
        [Step(calls=(("list_projects", {}), ("get_project", {"project_id": "apollo"}))),
         say("listo")]
    )  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="leé", scope=RequestScope.read_only())
        messages = await _messages(graph)
    calls, answers = _pairs(messages)
    assert calls == answers == ["call_0_0", "call_0_1"]
    tools = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.name for m in tools] == ["list_projects", "get_project"]
    assert not any(str(m.content).startswith("ERROR") for m in tools)


async def test_provider_copies_of_tool_calls_are_not_kept_in_history(open_agent) -> None:
    """``additional_kwargs['tool_calls']`` conserva los ids del proveedor: si quedara en el
    historial, el próximo pedido podría enviarlos en lugar de los normalizados."""
    model = ProviderLikeModel([call("list_projects"), say("listo")])
    async with open_agent("alice", model) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="leé", scope=RequestScope.read_only())
        messages = await _messages(graph)
    for message in messages:
        if isinstance(message, AIMessage):
            assert "tool_calls" not in message.additional_kwargs
            assert message.id != "resp-1"
            assert message.invalid_tool_calls == []
    assert model.prompts[-1][-2].tool_calls[0]["id"] == "call_0_0"  # lo que ve el modelo


async def test_identity_is_stable_across_a_restart_while_awaiting_approval(
    open_agent, conn
) -> None:
    script = [call("update_task", task_id="apo-3", changes={"status": "done"}), say("listo")]
    async with open_agent("alice", ProviderLikeModel(script)) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
    action_id, key = write_identity(THREAD, "call_0_0")
    assert paused.approval_request["action_id"] == action_id
    async with open_agent("alice", ProviderLikeModel(script)) as (graph, _):  # proceso nuevo
        state = await graph.aget_state(thread_config(THREAD))
        assert state.interrupts[0].value["action_id"] == action_id
        assert list(state.values["scope_ledger"]) == [action_id]
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    assert done.status is TurnStatus.COMPLETED
    assert _audit(conn) == [("apo-3", key)]


async def test_recovery_reuses_the_normalized_identity(open_faulty_agent, conn) -> None:
    script = [call("update_task", task_id="apo-3", changes={"status": "done"}), say("listo")]
    plan = {("update_task", EVERY_CALL): Fault.DISCONNECT_AFTER}  # confirma y pierde la respuesta
    async with open_faulty_agent("alice", ProviderLikeModel(script), plan) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        uncertain = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
    action_id, key = write_identity(THREAD, "call_0_0")
    assert uncertain.pending_action["idempotency_key"] == key
    async with open_faulty_agent("alice", ProviderLikeModel(script)) as (graph, faulty):
        recovered = await continue_after_failure(graph, thread_id=THREAD)
        ledger = (await graph.aget_state(thread_config(THREAD))).values["scope_ledger"]
    assert recovered.status is TurnStatus.COMPLETED
    (sent,) = [args for name, args in faulty.delivered if name == "update_task"]
    assert sent["idempotency_key"] == key
    assert {k: v["state"] for k, v in ledger.items()} == {action_id: "consumed"}
    assert _audit(conn) == [("apo-3", key)]  # una sola escritura
