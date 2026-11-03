"""``python -m arl.tracing``: listar, reconstruir y exportar ejecuciones (solo lectura)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.runtime import approve, resume, run_turn
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.tracing.__main__ import main
from arl.tracing.query import REPORT_SCHEMA

pytestmark = pytest.mark.anyio

THREAD = "t-cli"


async def _uncertain_run(open_faulty_agent) -> str:
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("ok")]
    )
    plan = {("update_task", EVERY_CALL): Fault.DISCONNECT_AFTER}
    async with open_faulty_agent("alice", model, plan) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
        result = await resume(graph, thread_id=THREAD, decision=approve(paused.approval_request))
    return result.run_id


def _tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


async def test_show_reconstructs_a_run_and_exports_a_reusable_report(
    open_faulty_agent, trace_dir: Path, tmp_path: Path, capsys
) -> None:
    run_id = await _uncertain_run(open_faulty_agent)
    before = _tree(trace_dir)

    assert main(["show", run_id, "--dir", str(trace_dir)]) == 0
    text = capsys.readouterr().out
    assert "WRITE_UNCERTAIN" in text and "reintentos:" in text and "No es un fallo" in text

    assert main(["show", run_id, "--dir", str(trace_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == REPORT_SCHEMA and report["final_status"] == "write_uncertain"
    assert report["counts"]["retries"] == 2

    exported = tmp_path / "report.json"
    exported.write_text(json.dumps(report), encoding="utf-8")
    assert main(["show", str(exported)]) == 0  # un reporte exportado se vuelve a mostrar igual
    assert capsys.readouterr().out.strip() == text.strip()

    (jsonl,) = trace_dir.rglob(f"{run_id}.jsonl")
    assert main(["show", str(jsonl)]) == 0
    assert capsys.readouterr().out.strip() == text.strip()
    assert _tree(trace_dir) == before  # la consulta nunca modifica las trazas


async def test_runs_and_thread_list_every_execution(open_faulty_agent, trace_dir: Path, capsys):
    run_id = await _uncertain_run(open_faulty_agent)
    assert main(["runs", "--dir", str(trace_dir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and lines[0].startswith(run_id)  # la más reciente primero
    assert main(["thread", THREAD, "--dir", str(trace_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [r["status"] for r in report["runs"]] == ["awaiting_approval", "write_uncertain"]


def test_missing_traces_and_foreign_files_exit_with_2(tmp_path: Path, capsys) -> None:
    assert main(["show", "abc123", "--dir", str(tmp_path)]) == 2
    assert main(["thread", "nadie", "--dir", str(tmp_path)]) == 2
    other = tmp_path / "otro.json"
    other.write_text('{"schema": "otra-cosa"}', encoding="utf-8")
    assert main(["show", str(other)]) == 2
    assert main(["show", "../../etc/passwd", "--dir", str(tmp_path)]) == 2  # no es un run_id
    assert "arl.tracing:" in capsys.readouterr().err


def test_schema_command_prints_the_versioned_json_schema(capsys) -> None:
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "arl.trace/1"
