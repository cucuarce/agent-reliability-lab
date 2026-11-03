"""Fallos reales entre procesos: timeouts, caída del servidor stdio y concurrencia."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client, StdioServerParameters

from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.runtime import (
    TurnStatus,
    approve,
    continue_after_failure,
    resume,
    run_turn,
    thread_config,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import Limits, write_identity

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

HERE = Path(__file__).parent
THREAD = "t-proc"
KEY = write_identity(THREAD, "call_0_0")[1]
FAST = Limits(retry_initial_interval=0.05, retry_max_interval=0.1, retry_jitter=False)


def _model(status: str = "done") -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": status}), say("Listo.")]
    )


def _faulty_server(db_path: Path, fault: str, delay_s: float = 2.0) -> StdioServerParameters:
    env = {
        **os.environ,
        "ARL_PRINCIPAL": "alice",
        "ARL_DATA_DIR": str(db_path.parent),
        "ARL_DB_NAME": db_path.name,
        "ARL_TEST_FAULT": fault,
        "ARL_TEST_DELAY_S": str(delay_s),
    }
    return StdioServerParameters(
        command=sys.executable, args=[str(HERE / "faulty_mcp_server.py")], env=env
    )


async def _pause(open_agent, user: str = "alice", status: str = "done", thread: str = THREAD):
    async with open_agent(user, _model(status)) as (graph, _):
        return (
            await run_turn(
                graph,
                thread_id=thread,
                text="cambiá apo-3",
                scope=RequestScope.writes(allow_update("apo-3", status=[status])),
            )
        ).approval_request


async def _resume_against(params, checkpoints: Path, request: dict, timeout_s: float):
    async with (
        AsyncSqliteSaver.from_conn_string(str(checkpoints)) as saver,
        Client(params) as client,
    ):
        gateway = McpGateway(client, "alice", timeout_s=timeout_s)
        graph = build_graph(_model(), gateway, saver, FAST, lock_dir=checkpoints.parent / "locks")
        return await resume(graph, thread_id=THREAD, decision=approve(request))


def _apo3(conn) -> tuple:
    return tuple(conn.execute("SELECT status, version FROM tasks WHERE id='apo-3'").fetchone())


async def _last_tool(open_agent, user: str = "alice", thread: str = THREAD) -> dict:
    async with open_agent(user, _model()) as (graph, _):
        messages = (await graph.aget_state(thread_config(thread))).values["messages"]
    return json.loads([m for m in messages if m.type == "tool"][-1].content)


# --- Timeouts y caídas reales del servidor ---------------------------------------------------


@pytest.mark.parametrize(
    ("fault", "replayed"),
    [
        # El cliente expira antes de que exista la escritura: el reintento (misma clave) la hace;
        # cuando la primera petición despierta, el servidor la resuelve como replay.
        ("delay_before_write", False),
        # La escritura ya ocurrió cuando el cliente expira: el reintento recibe el replay.
        ("delay_after_write", True),
    ],
)
async def test_real_timeout_around_the_write_is_resolved_by_retry(
    open_agent, db_path, checkpoints_path, conn, invariants, fault, replayed
) -> None:
    """Timeout real (servidor en otro proceso, read_timeout del SDK). El servidor MCP atiende el
    reintento en paralelo con la petición lenta, así que el reintento automático resuelve."""
    request = await _pause(open_agent)
    result = await _resume_against(
        _faulty_server(db_path, fault), checkpoints_path, request, timeout_s=0.5
    )
    assert result.status is TurnStatus.COMPLETED
    assert (await _last_tool(open_agent))["replayed"] is replayed
    assert _apo3(conn) == ("done", 2)
    invariants(conn, {KEY})


async def test_real_server_crash_after_commit_is_uncertain_then_recovered(
    open_agent, db_path, checkpoints_path, conn, invariants
) -> None:
    """El proceso servidor muere justo después de confirmar: la conexión queda muerta, los
    reintentos se agotan y el estado es WRITE_UNCERTAIN; otro proceso recupera con replay."""
    request = await _pause(open_agent)
    result = await _resume_against(
        _faulty_server(db_path, "crash_after_write"), checkpoints_path, request, timeout_s=5
    )
    assert result.status is TurnStatus.WRITE_UNCERTAIN
    assert result.error == "[unavailable] update_task"
    assert _apo3(conn) == ("done", 2)  # ocurrió, aunque el agente no pudo confirmarlo

    async with open_agent("alice", _model()) as (graph, _):  # proceso nuevo, servidor sano
        done = await continue_after_failure(graph, thread_id=THREAD)
    assert done.status is TurnStatus.COMPLETED
    assert (await _last_tool(open_agent))["replayed"] is True
    invariants(conn, {KEY})


# --- Concurrencia entre procesos -------------------------------------------------------------


def _spawn_workers(jobs: list[list[str]], barrier: Path) -> list[dict]:
    """Lanza los procesos, espera a que todos estén listos y los suelta a la vez."""
    barrier.mkdir()
    env = {**os.environ, "ARL_TEST_BARRIER_DIR": str(barrier)}
    procs = [
        subprocess.Popen(
            [sys.executable, str(HERE / "agent_worker.py"), *job],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        for job in jobs
    ]
    deadline = time.monotonic() + 60
    while len(list(barrier.glob("ready-*"))) < len(jobs):
        assert time.monotonic() < deadline, "los workers no quedaron listos"
        time.sleep(0.01)
    (barrier / "go").touch()
    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, stderr.decode("utf-8", "replace")
        outputs.append(json.loads(stdout.decode("utf-8").strip().splitlines()[-1]))
    return outputs


async def test_same_approval_from_concurrent_processes_runs_exactly_once(
    open_agent, db_path, checkpoints_path, conn, invariants, tmp_path
) -> None:
    """Cuatro procesos entregan la misma aprobación a la vez (p. ej. aprobar en dos terminales).

    El lock por thread serializa: exactamente uno ejecuta; el resto recibe ThreadBusy (mientras
    corre) o NoPendingApproval (después). Sin lock, varios ejecutaban y leían estados ajenos.
    """
    request = await _pause(open_agent)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    job = ["alice", "done", str(db_path), str(checkpoints_path), THREAD, str(request_file)]

    outputs = _spawn_workers([job] * 4, tmp_path / "barrier")

    runs = [o for o in outputs if "status" in o]
    refusals = [o["exception"] for o in outputs if "exception" in o]
    assert [run["status"] for run in runs] == ["completed"], outputs
    assert set(refusals) <= {"ThreadBusy", "NoPendingApproval"}, outputs
    assert json.loads(runs[0]["tool"])["replayed"] is False  # el único que ejecutó, escribió
    assert _apo3(conn) == ("done", 2)
    invariants(conn, {KEY})
    async with open_agent("alice", _model()) as (graph, _):
        final = await graph.aget_state(thread_config(THREAD))
    answers = [m for m in final.values["messages"] if getattr(m, "tool_call_id", "") == "call_0_0"]
    assert len(answers) == 1
    assert final.values["status"] == "completed"


async def test_competing_principals_in_separate_processes_have_one_winner(
    open_agent, db_path, checkpoints_path, conn, invariants, tmp_path
) -> None:
    """alice (owner) y bob (editor) aprueban cambios distintos sobre la misma versión de apo-3."""
    jobs = []
    for user, status in (("alice", "done"), ("bob", "blocked")):
        request = await _pause(open_agent, user=user, status=status, thread=f"t-{user}")
        path = tmp_path / f"{user}.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        jobs.append([user, status, str(db_path), str(checkpoints_path), f"t-{user}", str(path)])

    outputs = _spawn_workers(jobs, tmp_path / "barrier")

    tools = [o["tool"] for o in outputs]
    winners = [t for t in tools if not t.startswith("ERROR")]
    assert len(winners) == 1, tools
    assert sum("[version_conflict]" in t for t in tools) == 1, tools
    assert _apo3(conn)[1] == 2
    keys = {write_identity(f"t-{u}", "call_0_0")[1] for u in ("alice", "bob")}
    invariants(conn, keys)
