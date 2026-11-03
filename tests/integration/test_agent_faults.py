"""Fault injection reproducible entre el agente y el servidor MCP (en memoria).

Cada test inyecta fallos con un plan determinístico (``FaultyClient``), verifica el estado
resultante y, al final, las invariantes de seguridad del dominio (``invariants``).
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from mcp.shared.exceptions import MCPError

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.gateway import ToolResponseInvalid, ToolUnavailable
from arl.agent.graph import is_retryable
from arl.agent.runtime import (
    NothingToContinue,
    ThreadAwaitingApproval,
    ThreadNeedsRecovery,
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    thread_config,
    turn_result,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import Limits, write_identity
from arl.domain.models import Principal, TaskPatch
from arl.services.project_service import ProjectService

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

THREAD = "t-fault"
KEY = write_identity(THREAD, "call_0_0")[1]  # la clave de la primera escritura del guion
NO_KEYS: set[str] = set()


def _write_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )


def _read_model() -> ScriptedChatModel:
    return ScriptedChatModel([call("list_projects"), say("Hay proyectos.")])


def _calls(faulty, tool: str) -> list[dict]:
    return [args for name, args in faulty.attempts if name == tool]


def _task(conn, task_id: str = "apo-3") -> tuple:
    return tuple(
        conn.execute("SELECT status, version FROM tasks WHERE id = ?", (task_id,)).fetchone()
    )


async def _paused(open_faulty_agent, user: str = "alice") -> dict:
    async with open_faulty_agent(user, _write_model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
    assert paused.status is TurnStatus.AWAITING_APPROVAL
    return paused.approval_request


async def _last_tool_content(graph) -> dict:
    messages = (await graph.aget_state(thread_config(THREAD))).values["messages"]
    return json.loads([m for m in messages if m.type == "tool"][-1].content)


# --- Lecturas: transitorios y permanentes ---------------------------------------------------------


@pytest.mark.parametrize(
    "fault", [Fault.DISCONNECT_BEFORE, Fault.TIMEOUT_AFTER, Fault.SERVER_CRASH]
)
async def test_transient_read_failure_is_retried(
    open_faulty_agent, conn, invariants, fault
) -> None:
    async with open_faulty_agent("alice", _read_model(), {("list_projects", 0): fault}) as (
        graph,
        faulty,
    ):
        result = await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
    assert (result.status, result.reply) == (TurnStatus.COMPLETED, "Hay proyectos.")
    assert len(_calls(faulty, "list_projects")) == 2
    invariants(conn, NO_KEYS)


async def test_persistent_read_failure_stops_cleanly_and_can_continue(
    open_faulty_agent, conn, invariants
) -> None:
    plan = {("list_projects", EVERY_CALL): Fault.TIMEOUT_BEFORE}
    async with open_faulty_agent("alice", _read_model(), plan) as (graph, faulty):
        result = await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
        attempts = len(_calls(faulty, "list_projects"))
    assert result.status is TurnStatus.INTERRUPTED
    assert result.error == "[unavailable] list_projects"
    assert attempts == 3  # Limits.max_attempts: acotado, sin bucle

    async with open_faulty_agent("alice", _read_model()) as (graph, _):  # el servicio volvió
        assert (await turn_result(graph, THREAD)).status is TurnStatus.INTERRUPTED
        done = await continue_after_failure(graph, thread_id=THREAD)
    assert done.status is TurnStatus.COMPLETED
    invariants(conn, NO_KEYS)


async def test_failure_while_preparing_a_write_never_skips_approval(
    open_faulty_agent, conn, invariants
) -> None:
    plan = {("get_project", EVERY_CALL): Fault.DISCONNECT_BEFORE}
    async with open_faulty_agent("alice", _write_model(), plan) as (graph, faulty):
        result = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
    assert result.status is TurnStatus.INTERRUPTED
    assert _calls(faulty, "update_task") == []

    async with open_faulty_agent("alice", _write_model()) as (graph, faulty):
        resumed = await continue_after_failure(graph, thread_id=THREAD)
    assert resumed.status is TurnStatus.AWAITING_APPROVAL  # vuelve a pedir aprobación
    assert _calls(faulty, "update_task") == []
    assert _task(conn) == ("todo", 1)
    invariants(conn, NO_KEYS)


# --- Escrituras: reintentos seguros con la misma clave -------------------------------------------


@pytest.mark.parametrize(
    ("fault", "delivered_first"),
    [
        (Fault.TIMEOUT_BEFORE, False),
        (Fault.DISCONNECT_BEFORE, False),
        (Fault.SERVER_CRASH, False),
        (Fault.TIMEOUT_AFTER, True),
        (Fault.DISCONNECT_AFTER, True),
        (Fault.SERVER_CRASH_AFTER, True),
    ],
)
async def test_transient_write_failure_retries_with_the_same_key(
    open_faulty_agent, conn, invariants, fault, delivered_first
) -> None:
    request = await _paused(open_faulty_agent)
    async with open_faulty_agent("alice", _write_model(), {("update_task", 0): fault}) as (
        graph,
        faulty,
    ):
        done = await resume(graph, thread_id=THREAD, decision=approve(request))
        outcome = await _last_tool_content(graph)

    attempts = _calls(faulty, "update_task")
    assert len(attempts) == 2
    assert {a["idempotency_key"] for a in attempts} == {KEY}
    assert attempts[0] == attempts[1]  # mismos argumentos exactos en el reintento
    assert done.status is TurnStatus.COMPLETED
    assert outcome["replayed"] is delivered_first  # si llegó, el reintento es un replay
    assert _task(conn) == ("done", 2)
    invariants(conn, {KEY})


@pytest.mark.parametrize(
    ("fault", "written"), [(Fault.DISCONNECT_AFTER, True), (Fault.TIMEOUT_BEFORE, False)]
)
async def test_unconfirmed_write_is_reported_as_uncertain_and_recovered_once(
    open_faulty_agent, conn, invariants, fault, written
) -> None:
    request = await _paused(open_faulty_agent)
    plan = {("update_task", EVERY_CALL): fault}
    async with open_faulty_agent("alice", _write_model(), plan) as (graph, faulty):
        result = await resume(graph, thread_id=THREAD, decision=approve(request))
        with pytest.raises(ThreadNeedsRecovery):  # no se puede descartar con un turno nuevo
            await run_turn(graph, thread_id=THREAD, text="olvidalo", scope=READ_ONLY)
    assert result.status is TurnStatus.WRITE_UNCERTAIN
    assert result.error == "[unavailable] update_task"
    assert result.pending_action["action_id"] == request["action_id"]
    assert len(_calls(faulty, "update_task")) == 3
    assert _task(conn) == (("done", 2) if written else ("todo", 1))  # el agente no lo sabe
    invariants(conn, {KEY})

    async with open_faulty_agent("alice", _write_model()) as (graph, faulty):  # otro proceso
        assert (await turn_result(graph, THREAD)).status is TurnStatus.WRITE_UNCERTAIN
        done = await continue_after_failure(graph, thread_id=THREAD)
        outcome = await _last_tool_content(graph)
    assert done.status is TurnStatus.COMPLETED
    assert outcome["replayed"] is written
    assert {a["idempotency_key"] for a in _calls(faulty, "update_task")} == {KEY}
    assert _task(conn) == ("done", 2)
    invariants(conn, {KEY})


# --- Fallos definitivos: sin reintento automático -------------------------------------------------


async def test_forbidden_is_not_retried(open_faulty_agent, conn, invariants) -> None:
    request = await _paused(open_faulty_agent, user="carol")  # viewer
    async with open_faulty_agent("carol", _write_model()) as (graph, faulty):
        await resume(graph, thread_id=THREAD, decision=approve(request))
    assert len(_calls(faulty, "update_task")) == 1
    invariants(conn, NO_KEYS)


async def test_version_conflict_is_not_retried(open_faulty_agent, conn, invariants) -> None:
    request = await _paused(open_faulty_agent)
    ProjectService(conn).update_task(
        Principal(user_id="bob"), "apo-3", TaskPatch(title="Editada"), expected_version=1,
        idempotency_key="human-edit-0001",
    )  # fmt: skip
    async with open_faulty_agent("alice", _write_model()) as (graph, faulty):
        done = await resume(graph, thread_id=THREAD, decision=approve(request))
    assert done.status is TurnStatus.COMPLETED
    assert len(_calls(faulty, "update_task")) == 1
    invariants(conn, {"human-edit-0001"})


async def test_rejected_action_is_never_sent(open_faulty_agent, conn, invariants) -> None:
    request = await _paused(open_faulty_agent)
    async with open_faulty_agent("alice", _write_model()) as (graph, faulty):
        await resume(graph, thread_id=THREAD, decision=reject(request))
    assert _calls(faulty, "update_task") == []
    invariants(conn, NO_KEYS)


# --- Respuestas MCP incorrectas o inesperadas ----------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "fault"),
    [
        ("list_projects", Fault.MALFORMED),
        ("list_projects", Fault.EMPTY),
        ("list_projects", Fault.PROTOCOL_ERROR),
        ("get_project", Fault.WRONG_RESOURCE),
    ],
)
async def test_invalid_read_responses_stop_without_retry(
    open_faulty_agent, conn, invariants, tool, fault
) -> None:
    model = ScriptedChatModel([call("get_project", project_id="apollo"), say("ok")])
    if tool == "list_projects":
        model = _read_model()
    async with open_faulty_agent("alice", model, {(tool, EVERY_CALL): fault}) as (graph, faulty):
        result = await run_turn(graph, thread_id=THREAD, text="leé", scope=READ_ONLY)
    assert result.status is TurnStatus.INTERRUPTED
    assert result.error == f"[invalid_response] {tool}"
    assert len(_calls(faulty, tool)) == 1
    invariants(conn, NO_KEYS)


@pytest.mark.parametrize(
    "fault", [Fault.MALFORMED, Fault.WRONG_RESOURCE, Fault.EMPTY, Fault.PROTOCOL_ERROR]
)
async def test_invalid_write_response_is_uncertain_and_not_retried(
    open_faulty_agent, conn, invariants, fault
) -> None:
    request = await _paused(open_faulty_agent)
    async with open_faulty_agent("alice", _write_model(), {("update_task", 0): fault}) as (
        graph,
        faulty,
    ):
        result = await resume(graph, thread_id=THREAD, decision=approve(request))
        assert len(_calls(faulty, "update_task")) == 1
        assert result.status is TurnStatus.WRITE_UNCERTAIN
        assert result.error == "[invalid_response] update_task"
        done = await continue_after_failure(graph, thread_id=THREAD)  # reconciliación manual
    assert done.status is TurnStatus.COMPLETED
    assert _task(conn) == ("done", 2)
    invariants(conn, {KEY})


# --- Secretos, aislamiento, límites y estados -----------------------------------------------------


async def test_transport_errors_do_not_leak_sensitive_text(
    open_faulty_agent, conn, caplog, monkeypatch
) -> None:
    secret = "password=hunter2 en C:/secretos/app.db"

    async def leaky_transport(name, arguments, **kwargs):
        raise OSError(secret)

    caplog.set_level(logging.DEBUG)
    async with open_faulty_agent("alice", _read_model()) as (graph, faulty):
        monkeypatch.setattr(faulty, "call_tool", leaky_transport)
        result = await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
        state = await graph.aget_state(thread_config(THREAD))
    assert result.status is TurnStatus.INTERRUPTED
    for text in (caplog.text, str(result), str(state.values)):
        assert "hunter2" not in text and "secretos" not in text


async def test_failure_in_one_thread_does_not_affect_another(open_faulty_agent, conn) -> None:
    """El guion del modelo falso depende del historial de cada thread: el thread B usa un modelo
    de lectura y el mismo plan de fallos (que solo afecta escrituras)."""
    request = await _paused(open_faulty_agent)
    plan = {("update_task", EVERY_CALL): Fault.DISCONNECT_BEFORE}
    async with open_faulty_agent("alice", _write_model(), plan) as (graph, _):
        uncertain = await resume(graph, thread_id=THREAD, decision=approve(request))
    async with open_faulty_agent("alice", _read_model(), plan) as (graph, _):
        other = await run_turn(graph, thread_id="t-otro", text="¿proyectos?", scope=READ_ONLY)
        still = await turn_result(graph, THREAD)
    assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
    assert other.status is TurnStatus.COMPLETED
    assert still.status is TurnStatus.WRITE_UNCERTAIN
    assert still.pending_action == uncertain.pending_action


async def test_retries_are_bounded_with_capped_backoff(open_faulty_agent, conn) -> None:
    limits = Limits(
        max_attempts=4, retry_initial_interval=0.01, retry_max_interval=0.02, retry_jitter=False
    )
    plan = {("list_projects", EVERY_CALL): Fault.DISCONNECT_BEFORE}
    async with open_faulty_agent("alice", _read_model(), plan, limits) as (graph, faulty):
        start = time.perf_counter()
        result = await run_turn(graph, thread_id=THREAD, text="¿proyectos?", scope=READ_ONLY)
        elapsed = time.perf_counter() - start
    assert result.status is TurnStatus.INTERRUPTED
    assert len(_calls(faulty, "list_projects")) == 4
    assert elapsed < 1.0  # esperas: 0.01 + 0.02 + 0.02 (tope)


async def test_continue_requires_an_interrupted_run(open_faulty_agent) -> None:
    async with open_faulty_agent("alice", _write_model()) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        with pytest.raises(ThreadAwaitingApproval):
            await continue_after_failure(graph, thread_id=THREAD)
    async with open_faulty_agent("alice", _read_model()) as (graph, _):
        await run_turn(graph, thread_id="t-ok", text="hola", scope=READ_ONLY)
        with pytest.raises(NothingToContinue):
            await continue_after_failure(graph, thread_id="t-ok")


def test_only_transient_failures_are_retryable() -> None:
    assert is_retryable(ToolUnavailable("update_task", "timeout"))
    assert not is_retryable(ToolResponseInvalid("update_task", "contrato"))
    assert not is_retryable(MCPError(-32000, "Connection closed"))  # sin clasificar: no
    assert not is_retryable(RuntimeError("bug"))
