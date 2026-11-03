"""Trazas JSONL: correlación, reconstrucción, fallos, datos sensibles y almacenamiento roto."""

from __future__ import annotations

import inspect
import io
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.locks import ThreadBusy, thread_lock
from arl.agent.redaction import LOGGER
from arl.agent.runtime import (
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import write_identity
from arl.tracing import recorder
from arl.tracing.events import EVENT_SCHEMAS, validate_event
from arl.tracing.logging_setup import UNSAFE_DEBUG_LOGGERS, configure_logging
from arl.tracing.query import load_events, load_paths, reconstruct, render, thread_paths

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

THREAD = "t-trace"
KEY = write_identity(THREAD, "call_1_0")[1]  # update_task en el segundo mensaje del modelo
CANARY = "CANARIO-confidencial-7731"
HELPER = Path(__file__).with_name("crash_during_write.py")


def _model(changes: dict | None = None) -> ScriptedChatModel:
    """Lectura, propuesta de escritura (requiere aprobación) y respuesta final."""
    return ScriptedChatModel(
        [
            call("get_project", project_id="apollo"),
            call("update_task", task_id="apo-3", changes=changes or {"status": "done"}),
            say("Listo."),
        ]
    )


def _all_events(trace_dir: Path) -> list[dict]:
    events, invalid = load_paths(sorted(trace_dir.glob("thread-*/*.jsonl")))
    assert invalid == 0
    return events


def _of(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e["event"] == kind]


def _raw(trace_dir: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in trace_dir.rglob("*.jsonl"))


async def _approved_flow(open_faulty_agent, plan=None):
    async with open_faulty_agent("alice", _model(), plan) as (graph, faulty):
        paused = await run_turn(
            graph, thread_id=THREAD, text="mirá apollo y cerrá apo-3", scope=CLOSE_APO3
        )
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    return paused, done, faulty


# --- Correlación y reconstrucción ----------------------------------------------------------------


async def test_events_correlate_thread_run_tool_call_and_audit(
    open_faulty_agent, trace_dir: Path, conn, invariants
) -> None:
    paused, done, _ = await _approved_flow(open_faulty_agent)
    assert (paused.status, done.status) == (TurnStatus.AWAITING_APPROVAL, TurnStatus.COMPLETED)
    events = _all_events(trace_dir)

    # Cada línea cumple el esquema versionado; seq contiguo por ejecución.
    for line in _raw(trace_dir).splitlines():
        validate_event(json.loads(line))
    runs = {paused.run_id: [], done.run_id: []}
    for e in events:
        assert e["thread_id"] == THREAD and e["principal_id"] == "alice"
        runs[e["run_id"]].append(e["seq"])
    assert all(sorted(seqs) == list(range(1, len(seqs) + 1)) for seqs in runs.values())

    # Solicitado (LLM) → aprobado (humano) → ejecutado (servidor), ligados por ids.
    (routed,) = [e for e in _of(events, "tool_call_routed") if e["data"]["tool"] == "update_task"]
    (requested,) = _of(events, "approval_requested")
    (decided,) = _of(events, "approval_decided")
    (written,) = _of(events, "write_result")
    assert routed["run_id"] == requested["run_id"] == paused.run_id
    assert decided["run_id"] == written["run_id"] == done.run_id
    assert routed["data"]["tool_call_id"] == requested["data"]["tool_call_id"] == "call_1_0"
    assert requested["data"]["action_id"] == decided["data"]["action_id"]
    assert written["data"]["action_id"] == paused.approval_request["action_id"]
    assert routed["data"]["changed_fields"] == ["status"]

    # La traza apunta a la fuente de verdad: misma clave y mismo id que el audit_log.
    audit = conn.execute("SELECT id, idempotency_key FROM audit_log").fetchall()
    assert [tuple(r) for r in audit] == [(written["data"]["audit_id"], KEY)]
    assert written["data"]["idempotency_key"] == KEY

    report = reconstruct(events)
    assert report["final_status"] == "completed" and report["complete"]
    assert [r["operation"] for r in report["runs"]] == ["run_turn", "resume"]
    (write,) = report["writes"]
    assert write["state"] == "executed" and write["decisions"] == ["approve"]
    assert write["requested"] == {"task_id": "apo-3", "changed_fields": ["status"]}
    assert report["counts"]["model_calls"] == 3 and report["counts"]["retries"] == 0
    invariants(conn, {KEY})


async def test_rejected_write_is_reconstructed_as_never_sent(open_faulty_agent, trace_dir) -> None:
    async with open_faulty_agent("alice", _model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        await resume(graph, thread_id=THREAD, decision=reject(paused.approval_request, CANARY))
    events = _all_events(trace_dir)
    (write,) = reconstruct(events)["writes"]
    assert write["state"] == "rejected" and write["attempts"] == []
    assert _of(events, "approval_decided")[0]["data"]["with_reason"] is True
    assert CANARY not in _raw(trace_dir)  # el motivo es texto libre: no se registra


async def test_retries_are_traced_with_attempts_and_the_same_key(
    open_faulty_agent, trace_dir: Path, conn, invariants
) -> None:
    _, done, _ = await _approved_flow(
        open_faulty_agent, {("update_task", 0): Fault.DISCONNECT_AFTER}
    )
    assert done.status is TurnStatus.COMPLETED
    events = _all_events(trace_dir)
    writes = [e["data"] for e in _of(events, "mcp_call") if e["data"]["tool"] == "update_task"]
    assert [(w["attempt"], w["outcome"], w.get("replayed")) for w in writes] == [
        (1, "unavailable", None),
        (2, "ok", True),  # el servidor había confirmado: el reintento es un replay
    ]
    assert {w["idempotency_key"] for w in writes} == {KEY}
    report = reconstruct(events)
    (retry,) = report["retries"]
    assert (retry["node"], retry["attempts"]) == ("execute_write", 2)
    assert report["writes"][0]["state"] == "executed"
    assert "replay" in report["writes"][0]["note"]
    invariants(conn, {KEY})


async def test_persistent_uncertain_write_is_never_reported_as_failed(
    open_faulty_agent, trace_dir: Path
) -> None:
    plan = {("update_task", EVERY_CALL): Fault.TIMEOUT_AFTER}
    _, uncertain, _ = await _approved_flow(open_faulty_agent, plan)
    async with open_faulty_agent("alice", _model(), plan) as (graph, _):
        again = await continue_after_failure(graph, thread_id=THREAD)
    assert uncertain.status is again.status is TurnStatus.WRITE_UNCERTAIN

    events = _all_events(trace_dir)
    assert len(_of(events, "write_uncertain")) == 2
    assert {e["data"]["prior_status"] for e in _of(events, "recovery_started")} == {
        "write_uncertain"
    }
    report = reconstruct(events)
    assert report["final_status"] == "write_uncertain"
    (write,) = report["writes"]
    assert write["state"] == "uncertain"
    assert write["idempotency_keys"] == [KEY]
    assert len(write["attempts"]) == 6  # 3 intentos acotados por ejecución, misma clave
    text = render(report)
    assert "UNCERTAIN" in text and "No es un fallo" in text
    assert "fallida" not in text.lower() and "failed" not in text.lower()


async def test_crash_between_processes_is_reconstructed_and_recovered(
    open_agent, db_path: Path, checkpoints_path: Path, trace_dir: Path, tmp_path: Path, conn
) -> None:
    async with open_agent("alice", _model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(paused.approval_request), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(HELPER), str(db_path), str(checkpoints_path), THREAD, str(request)],
        capture_output=True, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 75, proc.stderr.decode("utf-8", "replace")

    crashed = [r for r in reconstruct(_all_events(trace_dir))["runs"] if not r["complete"]]
    assert [r["operation"] for r in crashed] == ["resume"]  # sin run_finished: murió a mitad
    partial = reconstruct(_all_events(trace_dir))
    assert partial["final_status"] is None and not partial["complete"]
    assert partial["writes"][0]["state"] == "uncertain"  # aprobada, sin evidencia de resultado

    async with open_agent("alice", _model()) as (graph, _):
        done = await continue_after_failure(graph, thread_id=THREAD)
    report = reconstruct(*load_paths(thread_paths(trace_dir, THREAD)))
    assert done.status is TurnStatus.COMPLETED
    assert [r["operation"] for r in report["runs"]] == [
        "run_turn",
        "resume",
        "continue_after_failure",
    ]
    assert report["final_status"] == "completed" and report["complete"]
    (write,) = report["writes"]
    assert write["state"] == "executed" and write["result"]["replayed"] is True
    assert conn.execute("SELECT count(*) FROM audit_log").fetchone()[0] == 1


async def test_busy_run_is_traced_with_its_own_final_status(
    open_agent, lock_dir: Path, trace_dir: Path
) -> None:
    async with open_agent("alice", _model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3", scope=CLOSE_APO3)
        with thread_lock(lock_dir, THREAD), pytest.raises(ThreadBusy):
            await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    report = reconstruct(_all_events(trace_dir))
    assert [r["status"] for r in report["runs"]] == ["awaiting_approval", "busy"]
    assert report["final_status"] == "awaiting_approval"  # el rechazo no cambia el estado


# --- Datos sensibles -------------------------------------------------------------------------


async def test_sensitive_values_never_reach_the_traces(
    open_faulty_agent, trace_dir: Path, monkeypatch
) -> None:
    model = _model({"title": f"password={CANARY}", "status": "blocked"})
    async with open_faulty_agent("alice", model) as (graph, faulty):
        paused = await run_turn(
            graph,
            thread_id=THREAD,
            text=f"usá la clave {CANARY}",
            scope=RequestScope.writes(allow_update("apo-3", title=None, status=["blocked"])),
        )
        real = faulty.call_tool

        async def leaky(name, arguments, **kwargs):
            if name == "update_task":
                raise OSError(f"token={CANARY} en C:/secretos/{CANARY}")
            return await real(name, arguments, **kwargs)

        monkeypatch.setattr(faulty, "call_tool", leaky)
        await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    raw = _raw(trace_dir)
    assert CANARY not in raw and "secretos" not in raw
    fields = [e["data"].get("changed_fields") for e in _all_events(trace_dir)]
    assert ["status", "title"] in fields  # se registran los nombres, no los valores


def test_free_text_identifiers_are_hashed_not_recorded() -> None:
    assert recorder.safe_id("call_0_1") == "call_0_1"
    hashed = recorder.safe_id(f"id con {CANARY} y espacios")
    assert hashed.startswith("h-") and CANARY not in hashed


# --- Fallos del almacenamiento de trazas ---------------------------------------------------------


async def test_unwritable_trace_dir_never_changes_the_business_result(
    open_faulty_agent, trace_dir: Path, conn, invariants, caplog
) -> None:
    trace_dir.write_text("no es un directorio", encoding="utf-8")  # mkdir falla en cada evento
    caplog.set_level(logging.WARNING, logger=LOGGER.name)
    paused, done, faulty = await _approved_flow(open_faulty_agent)
    assert (paused.status, done.status) == (TurnStatus.AWAITING_APPROVAL, TurnStatus.COMPLETED)
    assert len([1 for name, _ in faulty.attempts if name == "update_task"]) == 1
    warnings = [r.getMessage() for r in caplog.records if "trazas" in r.getMessage()]
    assert len(warnings) == 2  # uno por ejecución, no uno por evento
    assert all("error_type=" in w for w in warnings)
    invariants(conn, {KEY})


async def test_trace_failure_right_after_the_commit_does_not_retry_the_write(
    open_faulty_agent, conn, invariants, monkeypatch
) -> None:
    """Si emitir ``write_result`` fallara con una excepción, LangGraph reintentaría
    ``execute_write``: una escritura nueva. El registrador absorbe el fallo."""
    real_write = recorder.JsonlTraceSink.write

    def broken_after_commit(self, run, line):
        if '"event":"write_result"' in line:
            raise OSError("disco lleno")
        real_write(self, run, line)

    monkeypatch.setattr(recorder.JsonlTraceSink, "write", broken_after_commit)
    _, done, faulty = await _approved_flow(open_faulty_agent)
    assert done.status is TurnStatus.COMPLETED
    assert len([1 for name, _ in faulty.attempts if name == "update_task"]) == 1
    invariants(conn, {KEY})


def test_events_outside_the_schema_are_dropped_not_written(tmp_path: Path) -> None:
    sink = recorder.JsonlTraceSink(tmp_path)
    with recorder.run_scope(sink, thread_id=THREAD, principal_id="alice") as run:
        recorder.emit("model_call", model_turn=1, duration_ms=1.0, outcome="ok", prompt=CANARY)
        recorder.emit("evento_inventado")
        recorder.emit("run_started", operation="run_turn")
    assert run.dropped == 2
    (path,) = tmp_path.rglob("*.jsonl")
    events, invalid = load_events(path)
    assert [e["event"] for e in events] == ["run_started"] and invalid == 0
    assert CANARY not in path.read_text(encoding="utf-8")


def test_truncated_lines_are_counted_not_fatal(tmp_path: Path) -> None:
    sink = recorder.JsonlTraceSink(tmp_path)
    with recorder.run_scope(sink, thread_id=THREAD, principal_id="alice"):
        recorder.emit("run_started", operation="run_turn")
    (path,) = tmp_path.rglob("*.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"v":1,"ts":"2026-01-01T00:00')  # línea cortada por un crash
    events, invalid = load_events(path)
    assert (len(events), invalid) == (1, 1)
    report = reconstruct(events, invalid)
    assert report["invalid_lines"] == 1 and report["complete"] is False


def test_schema_covers_every_event_type() -> None:
    from arl.tracing.events import json_schema

    schema = json_schema()
    assert schema["schema"] == "arl.trace/2"  # 6A.1: alcance del turno
    assert set(schema["events"]) == set(EVENT_SCHEMAS)


# --- Logging de dependencias ---------------------------------------------------------------------


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    saved = (
        root.level,
        list(root.handlers),
        {n: logging.getLogger(n).level for n in UNSAFE_DEBUG_LOGGERS},
    )
    yield
    root.setLevel(saved[0])
    root.handlers[:] = saved[1]
    for name, level in saved[2].items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize("allow_unsafe_debug", [False, True])
async def test_debug_logging_does_not_dump_checkpoints_unless_explicitly_allowed(
    open_faulty_agent, restore_logging, allow_unsafe_debug
) -> None:
    """Con DEBUG global, ``aiosqlite`` volcaría los checkpoints (mensajes completos). Queda en
    INFO salvo opt-in explícito; el caso ``True`` demuestra que el riesgo es real."""
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream, allow_unsafe_debug=allow_unsafe_debug)
    async with open_faulty_agent("alice", _model()) as (graph, _):
        await run_turn(
            graph, thread_id=THREAD, text=f"texto del usuario {CANARY}", scope=CLOSE_APO3
        )
    assert (CANARY in stream.getvalue()) is allow_unsafe_debug
    assert "DEBUG" in stream.getvalue()  # el resto del DEBUG sí se emite


def test_unsafe_debug_is_off_by_default_everywhere(restore_logging) -> None:
    """Sin opt-in, ``aiosqlite`` nunca baja de INFO, y ningún otro módulo del producto toca
    niveles o handlers globales (``basicConfig`` o ``setLevel`` saltearían esta protección)."""
    assert inspect.signature(configure_logging).parameters["allow_unsafe_debug"].default is False
    configure_logging("DEBUG")
    for name in UNSAFE_DEBUG_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() >= logging.INFO
    src = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        path.name
        for path in src.rglob("*.py")
        if path.name != "logging_setup.py"
        and re.search(r"basicConfig|\.setLevel\(|allow_unsafe_debug", path.read_text("utf-8"))
    ]
    assert offenders == []


def test_logging_setup_rejects_invalid_levels(restore_logging) -> None:
    with pytest.raises(ValueError):
        configure_logging("VERBOSO")
