"""Crash real entre el commit del servidor y el checkpoint, y costo de prepare_write."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arl.agent.runtime import TurnStatus, continue_after_failure, run_turn, thread_config
from arl.agent.scripted_model import ScriptedChatModel, call, say

pytestmark = pytest.mark.anyio

HELPER = Path(__file__).with_name("crash_during_write.py")
CRASH_EXIT_CODE = 75  # crash_during_write.CRASH_EXIT_CODE
THREAD = "t-crash"


def _write_script() -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )


def _domain(conn) -> tuple[tuple, int, int]:
    task = tuple(conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone())
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    keys = conn.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
    return task, audits, keys


async def test_process_killed_after_commit_before_checkpoint_resumes_once(
    open_agent, db_path: Path, checkpoints_path: Path, tmp_path: Path, conn
) -> None:
    # Proceso 1: deja el thread pausado esperando aprobación.
    async with open_agent("alice", _write_script()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(paused.approval_request), encoding="utf-8")

    # Proceso 2 (subproceso real): aprueba y muere con os._exit tras el commit del servidor.
    proc = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            str(db_path),
            str(checkpoints_path),
            THREAD,
            str(request_file),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == CRASH_EXIT_CODE, proc.stderr.decode("utf-8", "replace")
    assert _domain(conn) == (("done", 2), 1, 1)  # el servidor sí confirmó la escritura

    # Proceso 3: el checkpoint quedó antes de execute_write; al continuar, el servidor hace replay.
    async with open_agent("alice", _write_script()) as (graph, _):
        snapshot = await graph.aget_state(thread_config(THREAD))
        assert snapshot.next == ("execute_write",)
        assert snapshot.values["pending_write"]["action_id"] == paused.approval_request["action_id"]
        done = await continue_after_failure(graph, thread_id=THREAD)
        messages = (await graph.aget_state(thread_config(THREAD))).values["messages"]

    assert (done.status, done.reply) == (TurnStatus.COMPLETED, "Listo.")
    assert json.loads(messages[-2].content)["replayed"] is True
    assert _domain(conn) == (("done", 2), 1, 1)  # una sola modificación y una sola auditoría


# --- Costo de la búsqueda N+1 en prepare_write ---------------------------------------------------


@pytest.mark.parametrize(
    ("task_id", "expected_reads"),
    [
        ("apo-3", ["list_projects", "get_project:apollo"]),  # primer proyecto visible
        ("cas-2", ["list_projects", "get_project:apollo", "get_project:cassini"]),
        ("bor-1", ["list_projects", "get_project:apollo", "get_project:cassini"]),  # no visible
    ],
)
async def test_prepare_write_lookup_cost(
    open_agent, monkeypatch, task_id: str, expected_reads: list[str]
) -> None:
    """Medición del N+1: 1 list_projects + hasta P get_project (P = proyectos visibles)."""
    model = ScriptedChatModel([call("update_task", task_id=task_id, changes={"status": "done"})])
    async with open_agent("alice", model) as (graph, gateway):
        reads: list[str] = []
        real_call = gateway.call

        async def counting_call(name, arguments):
            reads.append(f"{name}:{arguments['project_id']}" if "project_id" in arguments else name)
            return await real_call(name, arguments)

        monkeypatch.setattr(gateway, "call", counting_call)
        await run_turn(graph, thread_id="t-cost", text="cambiá la tarea")

    assert reads == expected_reads
