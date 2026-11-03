"""Lock por thread: serializa ejecuciones del mismo thread y se libera ante errores."""

from __future__ import annotations

from pathlib import Path

import pytest

from arl.agent.locks import ThreadBusy, lock_path, thread_lock
from arl.agent.runtime import TurnStatus, approve, resume, run_turn, turn_result
from arl.agent.scripted_model import ScriptedChatModel, call, say

pytestmark = pytest.mark.anyio

THREAD = "t-lock"


def _write_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )


def test_second_holder_is_refused_and_lock_is_released_after_errors(tmp_path: Path) -> None:
    with thread_lock(tmp_path, THREAD), pytest.raises(ThreadBusy), thread_lock(tmp_path, THREAD):
        pass
    with pytest.raises(RuntimeError), thread_lock(tmp_path, THREAD):
        raise RuntimeError("falla dentro de la sección crítica")
    with thread_lock(tmp_path, THREAD):  # liberado: se puede volver a tomar
        pass
    with thread_lock(tmp_path, THREAD), thread_lock(tmp_path, "otro-thread"):
        pass  # threads distintos no se bloquean entre sí


def test_lock_file_name_never_uses_the_thread_id_as_a_path(tmp_path: Path) -> None:
    path = lock_path(tmp_path, "../../etc/passwd")
    assert path.parent == tmp_path
    assert ".." not in path.name and "passwd" not in path.name


async def test_runtime_refuses_to_run_a_thread_that_is_already_running(
    open_agent, lock_dir: Path, conn
) -> None:
    async with open_agent("alice", _write_model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
        with thread_lock(lock_dir, THREAD):  # otro proceso lo está ejecutando
            with pytest.raises(ThreadBusy):
                await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
            with pytest.raises(ThreadBusy):
                await run_turn(graph, thread_id=THREAD, text="otra cosa")
            assert (await turn_result(graph, THREAD)).status is TurnStatus.BUSY
        done = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    assert done.status is TurnStatus.COMPLETED
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == ("done", 2)


async def test_busy_status_does_not_reveal_threads_of_other_principals(
    open_agent, lock_dir: Path
) -> None:
    from arl.agent.state import PrincipalMismatch

    async with open_agent("alice", _write_model()) as (graph, _):
        await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
    async with open_agent("bob", _write_model()) as (graph, _):
        with thread_lock(lock_dir, THREAD), pytest.raises(PrincipalMismatch):
            await turn_result(graph, THREAD)  # primero el dueño, después el lock


# --- Locking obligatorio ------------------------------------------------------------------------


async def test_building_a_persistent_graph_without_lock_dir_fails(checkpoints_path: Path) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from arl.agent.graph import build_graph
    from arl.agent.locks import LockingRequired

    async with AsyncSqliteSaver.from_conn_string(str(checkpoints_path)) as saver:
        with pytest.raises(LockingRequired):
            build_graph(_write_model(), _FakeGateway(), saver, lock_dir=None)  # type: ignore[arg-type]


class _FakeGateway:
    principal_id = "alice"


def _files(db: Path) -> dict[str, bytes]:
    """Base de checkpoints con su WAL: cualquier escritura cambia alguno de los dos."""
    return {p.name: p.read_bytes() for p in db.parent.glob(db.name + "*")}


@pytest.mark.parametrize("operation", ["run_turn", "resume", "continue", "turn_result"])
async def test_every_runtime_entry_point_refuses_to_run_without_a_lock(
    open_agent, checkpoints_path: Path, conn, operation
) -> None:
    """Un grafo cuya configuración perdió ``lock_dir`` (compilado aparte o con la config
    sobrescrita) no se ejecuta sin exclusión mutua: falla antes de leer o escribir el thread."""
    from arl.agent.locks import LockingRequired
    from arl.agent.runtime import continue_after_failure

    async with open_agent("alice", _write_model()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
        unlocked = graph.with_config({"configurable": {"lock_dir": None}})
        before = _files(checkpoints_path)
        calls = {
            "run_turn": lambda: run_turn(unlocked, thread_id=THREAD, text="otra"),
            "resume": lambda: resume(
                unlocked, thread_id=THREAD, decision=approve(paused.approval_request)
            ),
            "continue": lambda: continue_after_failure(unlocked, thread_id=THREAD),
            "turn_result": lambda: turn_result(unlocked, THREAD),
        }
        with pytest.raises(LockingRequired):
            await calls[operation]()
        assert _files(checkpoints_path) == before
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == ("todo", 1)


def test_lock_held_by_another_process_is_released_when_it_crashes(tmp_path: Path) -> None:
    """Dos procesos reales: mientras el hijo tiene el lock, el padre recibe ThreadBusy; al matar
    al hijo (sin liberar nada) el sistema operativo libera el lock."""
    import subprocess
    import sys
    import time

    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time, pathlib\n"
            "from arl.agent.locks import thread_lock\n"
            "with thread_lock(pathlib.Path(sys.argv[1]), sys.argv[2]):\n"
            "    pathlib.Path(sys.argv[3]).write_text('ok')\n"
            "    time.sleep(60)\n",
            str(tmp_path / "locks"),
            THREAD,
            str(ready),
        ]
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert child.poll() is None and time.monotonic() < deadline, "el hijo no tomó el lock"
            time.sleep(0.05)
        with pytest.raises(ThreadBusy), thread_lock(tmp_path / "locks", THREAD):
            pass
    finally:
        child.kill()  # crash: no ejecuta el finally ni libera el lock
        child.wait(timeout=30)
    # Windows libera los locks de un proceso terminado de forma asíncrona (documentado en
    # LockFileEx): durante unos milisegundos se sigue recibiendo ThreadBusy, nunca una ejecución
    # doble. Se espera la liberación con un tope.
    deadline = time.monotonic() + 10
    while True:
        try:
            with thread_lock(tmp_path / "locks", THREAD):
                break
        except ThreadBusy:
            assert time.monotonic() < deadline, "el lock no se liberó tras el crash"
            time.sleep(0.02)
