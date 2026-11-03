"""Estado de un thread interrumpido por un fallo: debe ser legible y no debe poder descartarse."""

from __future__ import annotations

import pytest

from arl.agent.runtime import TurnStatus, approve, resume, run_turn, turn_result
from arl.agent.scripted_model import ScriptedChatModel, call, say

pytestmark = pytest.mark.anyio

THREAD = "t-rec"


def _write_script() -> ScriptedChatModel:
    return ScriptedChatModel(
        [
            call("update_task", task_id="apo-3", changes={"status": "done"}),
            say("Listo."),
            say("Otro turno."),
        ]
    )


async def _crash_inside_execute_write(open_agent, monkeypatch) -> None:
    async with open_agent("alice", _write_script()) as (graph, gateway):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
        real_call = gateway.call

        async def crash(name, arguments):
            if name == "update_task":
                raise RuntimeError("caída simulada")
            return await real_call(name, arguments)

        monkeypatch.setattr(gateway, "call", crash)
        with pytest.raises(RuntimeError):
            await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))


async def test_state_of_a_thread_stopped_inside_execute_write_is_readable(
    open_agent, monkeypatch
) -> None:
    await _crash_inside_execute_write(open_agent, monkeypatch)
    async with open_agent("alice", _write_script()) as (graph, _):
        result = await turn_result(graph, THREAD)
    assert result.status is TurnStatus.WRITE_UNCERTAIN
    assert result.pending_action["task_id"] == "apo-3"


async def test_new_message_cannot_discard_an_unconfirmed_write(open_agent, monkeypatch, conn):
    await _crash_inside_execute_write(open_agent, monkeypatch)
    async with open_agent("alice", _write_script()) as (graph, _):
        from arl.agent.runtime import ThreadNeedsRecovery

        with pytest.raises(ThreadNeedsRecovery):
            await run_turn(graph, thread_id=THREAD, text="olvidate y hacé otra cosa")
        state = await graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert state.values["pending_write"] is not None  # la acción aprobada sigue registrada
