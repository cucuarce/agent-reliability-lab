"""Resultado incierto: la escritura se confirmó en el servidor pero la respuesta se perdió o llegó
mal formada. El agente no declara éxito sin evidencia ni abre otra operación con otra clave."""

from __future__ import annotations

import json

import pytest

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.runtime import (
    ThreadNeedsRecovery,
    TurnStatus,
    approve,
    continue_after_failure,
    resume,
    run_turn,
    thread_config,
    turn_result,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import write_identity

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

THREAD = "t-unc"
KEY = write_identity(THREAD, "call_0_0")[1]


def _model() -> ScriptedChatModel:
    return ScriptedChatModel(
        [
            call("update_task", task_id="apo-3", changes={"status": "done"}),
            say("Listo."),
            # Si el LLM pudiera volver a intentarlo, propondría la misma acción con otra tool call.
            call("update_task", task_id="apo-3", changes={"status": "done"}),
        ]
    )


def _keys(faulty) -> list[str]:
    return [args["idempotency_key"] for name, args in faulty.attempts if name == "update_task"]


def _audits(conn) -> list[tuple]:
    rows = conn.execute("SELECT idempotency_key, task_id FROM audit_log").fetchall()
    return [tuple(row) for row in rows]


async def _write_messages(graph) -> list:
    messages = (await graph.aget_state(thread_config(THREAD))).values["messages"]
    return [m for m in messages if m.type == "tool" and m.name == "update_task"]


@pytest.mark.parametrize(
    "fault", [Fault.MALFORMED, Fault.EMPTY, Fault.WRONG_RESOURCE, Fault.DISCONNECT_AFTER]
)
async def test_committed_write_with_bad_or_lost_response_is_reconciled_by_replay(
    open_faulty_agent, conn, invariants, fault
) -> None:
    async with open_faulty_agent("alice", _model(), {("update_task", EVERY_CALL): fault}) as (
        graph,
        faulty,
    ):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        uncertain = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        # El servidor confirmó, pero el agente no tiene evidencia: no declara éxito.
        assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
        assert await _write_messages(graph) == []
        assert _audits(conn) == [(KEY, "apo-3")]
        with pytest.raises(ThreadNeedsRecovery):
            await run_turn(graph, thread_id=THREAD, text="intentalo de nuevo", scope=READ_ONLY)
        faulty.clear()  # el servicio vuelve a responder bien
        done = await continue_after_failure(graph, thread_id=THREAD)
        (evidence,) = await _write_messages(graph)

    assert done.status is TurnStatus.COMPLETED
    outcome = json.loads(evidence.content)
    assert (outcome["replayed"], outcome["outcome"]) == (True, "updated")  # evidencia del servidor
    assert set(_keys(faulty)) == {KEY}  # nunca una clave nueva para el mismo intento
    assert _audits(conn) == [(KEY, "apo-3")]
    invariants(conn, {KEY})


async def test_persistently_bad_response_stays_uncertain_without_new_operations(
    open_faulty_agent, conn, invariants
) -> None:
    plan = {("update_task", EVERY_CALL): Fault.MALFORMED}
    async with open_faulty_agent("alice", _model(), plan) as (graph, faulty):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        results = [await continue_after_failure(graph, thread_id=THREAD) for _ in range(3)]
        state = await turn_result(graph, THREAD)

    assert {r.status for r in results} == {TurnStatus.WRITE_UNCERTAIN}
    assert state.status is TurnStatus.WRITE_UNCERTAIN
    assert state.error is None or state.error.startswith("[invalid_response]")
    # La acción aprobada y su clave quedan visibles para reconciliar a mano.
    assert state.pending_action["action_id"] == paused.approval_request["action_id"]
    assert state.pending_action["idempotency_key"] == KEY
    assert _keys(faulty) == [KEY] * 4  # 1 + 3 continuaciones, sin reintentos automáticos
    assert _audits(conn) == [(KEY, "apo-3")]  # la primera confirmó; las demás fueron replay
    invariants(conn, {KEY})
