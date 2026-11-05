"""Request Scope Enforcement (etapa 6A.1) sobre el agente real.

Hallazgo F1 de la etapa 6A: una escritura inducida por texto malicioso se ejecutaba si el humano
la aprobaba sin revisar. Con un alcance explícito, una escritura fuera de lo autorizado por la
aplicación se bloquea **antes** de pedir aprobación; la autorización por rol del servidor sigue
siendo un control independiente.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.runtime import (
    ThreadNeedsRecovery,
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    thread_config,
)
from arl.agent.scope import RequestScope, ScopeViolation, allow_update
from arl.agent.scripted_model import ScriptedChatModel, Step, call, say
from arl.agent.state import ApprovalMismatch
from arl.domain.models import Principal, TaskPatch
from arl.services.project_service import ProjectService
from arl.storage.db import connect
from arl.tracing.query import report_for

pytestmark = pytest.mark.anyio

THREAD = "t-scope"
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))


def _audit(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [tuple(r) for r in conn.execute("SELECT actor_id, task_id FROM audit_log")]


def _calls(*steps) -> ScriptedChatModel:
    return ScriptedChatModel([*steps, say("listo")])


async def _ledger(graph) -> dict:
    state = await graph.aget_state(thread_config(THREAD))
    return state.values.get("scope_ledger", {})


def _blocked(trace_dir: Path) -> list[str | None]:
    """Motivos de las escrituras bloqueadas por el alcance, según las trazas."""
    report = report_for(sorted(trace_dir.glob("thread-*/*.jsonl")))
    return [b["reason"] for b in report["scope"]["blocked"]]


# --- La vulnerabilidad F1 y su corrección --------------------------------------------------------


async def test_injected_write_is_blocked_before_approval_even_if_the_human_approves_all(
    open_agent, conn, trace_dir
) -> None:
    """Pedido de solo lectura; el modelo propone una escritura (como si siguiera una inyección)."""
    model = _calls(call("get_project", project_id="apollo"),
                   call("update_task", task_id="apo-4", changes={"status": "done"}))  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id=THREAD, text="resumime apollo",
                                scope=RequestScope.read_only())  # fmt: skip
    assert result.status is TurnStatus.COMPLETED
    assert result.approval_request is None  # nunca llega al humano
    assert _audit(conn) == []
    # Solo los resultados de tools: desde 6B.6 el system prompt también nombra el código.
    blocked = [
        m
        for m in model.prompts[-1]
        if isinstance(m, ToolMessage) and "out_of_scope" in str(m.content)
    ]
    assert len(blocked) == 1 and "done" not in str(blocked[0].content)  # motivo sin valores
    report = report_for(sorted(trace_dir.glob("thread-*/*.jsonl")))
    routed = [e for e in report["timeline"] if e["event"] == "tool_call_routed"]
    assert "out_of_scope" in routed[-1]["summary"]


async def test_write_within_scope_is_approved_executed_and_consumed(open_agent, conn) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}))
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        assert paused.status is TurnStatus.AWAITING_APPROVAL
        (entry,) = (await _ledger(graph)).values()
        assert entry["state"] == "reserved"  # reservada antes de pedir aprobación
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert done.status is TurnStatus.COMPLETED
        (entry,) = (await _ledger(graph)).values()
        assert entry["state"] == "consumed"
    assert _audit(conn) == [("alice", "apo-3")]


@pytest.mark.parametrize(
    "changes", [{"status": "blocked"}, {"title": "otro"}, {"status": "done", "assignee_id": "bob"}]
)
async def test_values_or_fields_outside_the_grant_are_blocked(open_agent, conn, changes) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes=changes))
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
    assert result.status is TurnStatus.COMPLETED and result.approval_request is None
    assert _audit(conn) == []


# --- Ciclo de vida de las reservas ---------------------------------------------------------------


async def test_consumed_grant_cannot_authorize_a_new_operation(open_agent, conn) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-3", changes={"status": "done"}))  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert done.status is TurnStatus.COMPLETED  # la segunda propuesta no pidió aprobación
        states = sorted(e["state"] for e in (await _ledger(graph)).values())
    assert states == ["consumed"]
    assert _audit(conn) == [("alice", "apo-3")]


async def test_two_actions_cannot_hold_a_single_use_grant_at_the_same_time(
    open_agent, conn, trace_dir: Path
) -> None:
    """Dos escrituras en el mismo mensaje del modelo sobre un permiso con ``max_uses=1``."""
    write = ("update_task", {"task_id": "apo-3", "changes": {"status": "done"}})
    model = ScriptedChatModel([Step(calls=(write, write)), say("listo")])
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        assert [e["state"] for e in (await _ledger(graph)).values()] == ["reserved"]
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert done.status is TurnStatus.COMPLETED and done.approval_request is None
        states = [e["state"] for e in (await _ledger(graph)).values()]
    assert states == ["consumed"]
    assert _blocked(trace_dir) == ["exhausted"]
    assert _audit(conn) == [("alice", "apo-3")]


async def test_rejected_proposal_releases_its_reservation(open_agent, conn) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-3", changes={"status": "done"}))  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        first = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        second = await resume(graph, thread_id=THREAD, decision=reject(first.approval_request))
        assert second.status is TurnStatus.AWAITING_APPROVAL  # nueva acción, nueva aprobación
        assert second.approval_request["action_id"] != first.approval_request["action_id"]
        with pytest.raises(ApprovalMismatch):  # la aprobación vieja no sirve para la nueva acción
            await resume(graph, thread_id=THREAD, decision=approve(first.approval_request))
        assert _audit(conn) == []
        await resume(graph, thread_id=THREAD, decision=approve(second.approval_request))
        states = sorted(e["state"] for e in (await _ledger(graph)).values())
    assert states == ["consumed", "released"]
    assert _audit(conn) == [("alice", "apo-3")]


async def test_version_conflict_invalidates_and_a_new_proposal_needs_a_new_scope(
    open_agent, conn, db_path: Path, trace_dir: Path
) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-3", changes={"status": "done"}))  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        with closing(connect(db_path)) as other:
            ProjectService(other).update_task(Principal(user_id="bob"), "apo-3",
                                              TaskPatch(title="Otro"), expected_version=1,
                                              idempotency_key="concurrente-1")  # fmt: skip
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert done.status is TurnStatus.COMPLETED  # la re-propuesta se bloqueó sin aprobación
        states = sorted(e["state"] for e in (await _ledger(graph)).values())
    assert states == ["invalidated"]
    assert _blocked(trace_dir) == ["exhausted"]  # el permiso invalidado no autoriza otra acción
    assert _audit(conn) == [("bob", "apo-3")]


async def test_forbidden_closes_the_reservation_without_automatic_retry(
    open_faulty_agent, conn, trace_dir: Path
) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-3", changes={"status": "done"}))  # fmt: skip
    async with open_faulty_agent("carol", model) as (graph, faulty):  # viewer de apollo
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert done.status is TurnStatus.COMPLETED
        states = sorted(e["state"] for e in (await _ledger(graph)).values())
    assert states == ["refused"]
    assert _blocked(trace_dir) == ["exhausted"]  # un action_id nuevo no reabre el permiso cerrado
    assert [name for name, _ in faulty.delivered if name == "update_task"] == ["update_task"]
    assert _audit(conn) == []  # el alcance no da permisos que el rol no tiene


async def test_technical_retry_keeps_action_key_and_arguments(open_faulty_agent, conn) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}))
    plan = {("update_task", 0): Fault.DISCONNECT_AFTER}  # confirma y se pierde la respuesta
    async with open_faulty_agent("alice", model, plan) as (graph, faulty):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        (entry,) = (await _ledger(graph)).values()
    assert done.status is TurnStatus.COMPLETED and entry["state"] == "consumed"
    sent = [args for name, args in faulty.delivered if name == "update_task"]
    assert len(sent) == 2 and sent[0] == sent[1]  # misma clave, mismos argumentos
    assert _audit(conn) == [("alice", "apo-3")]


async def test_uncertain_write_keeps_its_reservation_across_a_restart(
    open_faulty_agent, conn
) -> None:
    """Crash después del commit: la reserva sigue tomada; solo la operación original continúa."""
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-3", changes={"status": "done"}))  # fmt: skip
    plan = {("update_task", EVERY_CALL): Fault.DISCONNECT_AFTER}
    async with open_faulty_agent("alice", model, plan) as (graph, first):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        uncertain = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
        assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
        (entry,) = (await _ledger(graph)).values()
        assert entry["state"] == "reserved"
        with pytest.raises(ThreadNeedsRecovery):  # un turno nuevo no puede reemplazar el alcance
            await run_turn(graph, thread_id=THREAD, text="otra cosa", scope=CLOSE_APO3)
    async with open_faulty_agent("alice", model) as (graph, faulty):  # proceso nuevo, sin fallos
        recovered = await continue_after_failure(graph, thread_id=THREAD)
        assert recovered.status is TurnStatus.COMPLETED
        ledger = await _ledger(graph)
    assert [e["state"] for e in ledger.values()] == ["consumed"]  # la re-propuesta se bloqueó
    assert list(ledger) == [paused.approval_request["action_id"]]  # la misma acción
    original = {repr(args) for name, args in first.delivered if name == "update_task"}
    (sent,) = [args for name, args in faulty.delivered if name == "update_task"]
    assert original == {repr(sent)}  # misma clave, misma versión esperada, mismos cambios
    assert sent["idempotency_key"] == uncertain.pending_action["idempotency_key"]
    assert _audit(conn) == [("alice", "apo-3")]


async def test_scope_and_reservation_survive_a_restart_while_awaiting_approval(
    open_agent, conn
) -> None:
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}),
                   call("update_task", task_id="apo-4", changes={"status": "done"}))  # fmt: skip
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        saved = (await graph.aget_state(thread_config(THREAD))).values
    async with open_agent("alice", model) as (graph, _):  # proceso nuevo
        restored = (await graph.aget_state(thread_config(THREAD))).values
        assert restored["request_scope"] == saved["request_scope"]
        assert restored["scope_ledger"] == saved["scope_ledger"]
        (entry,) = restored["scope_ledger"].values()
        assert entry["state"] == "reserved"
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    assert done.status is TurnStatus.COMPLETED and done.approval_request is None
    assert _audit(conn) == [("alice", "apo-3")]  # apo-4 sigue fuera del alcance guardado


async def test_tampered_scope_in_the_checkpoint_is_detected(open_agent, conn) -> None:
    model = _calls(call("update_task", task_id="apo-4", changes={"status": "done"}))
    async with open_agent("alice", model) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="leé apollo", scope=RequestScope.read_only())
        state = await graph.aget_state(thread_config(THREAD))
        forged = dict(state.values["request_scope"])
        forged["scope"] = {**forged["scope"], "max_writes": 9}  # sin recalcular el sello
        await graph.aupdate_state(thread_config(THREAD), {"request_scope": forged})
        with pytest.raises(ScopeViolation):
            await graph.ainvoke({"messages": [], "principal_id": "alice"}, thread_config(THREAD))
    assert _audit(conn) == []


async def test_run_turn_requires_an_explicit_scope(open_agent) -> None:
    async with open_agent("alice", _calls()) as (graph, _):
        with pytest.raises(TypeError):
            await run_turn(graph, thread_id=THREAD, text="hola")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            await run_turn(graph, thread_id=THREAD, text="hola", scope={"grants": []})


def test_unrestricted_scope_is_used_only_by_the_v1_evaluation_suite() -> None:
    root = Path(__file__).resolve().parents[2]
    users = sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for folder in ("src", "scripts", "tests")
        for p in (root / folder).rglob("*.py")
        if "unrestricted_legacy_v1(" in p.read_text(encoding="utf-8")
    )
    assert users == [
        "src/arl/agent/scope.py",  # definición
        "src/arl/evals/runner.py",  # suite v1, histórica
        "tests/integration/test_agent_scope.py",  # este test
        "tests/unit/test_request_scope.py",
    ]


async def test_execute_write_refuses_an_approved_action_without_its_reservation(
    open_agent, conn
) -> None:
    """Defensa en profundidad: aunque la acción esté aprobada, sin reserva no se envía."""
    model = _calls(call("update_task", task_id="apo-3", changes={"status": "done"}))
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        await graph.aupdate_state(thread_config(THREAD), {"scope_ledger": {}})
        approval = approve(paused.approval_request).model_dump()
        await graph.aupdate_state(thread_config(THREAD), {"approval": approval},
                                  as_node="await_approval")  # fmt: skip
        assert (await graph.aget_state(thread_config(THREAD))).next == ("execute_write",)
        with pytest.raises(ScopeViolation):
            await continue_after_failure(graph, thread_id=THREAD)
    assert _audit(conn) == []
