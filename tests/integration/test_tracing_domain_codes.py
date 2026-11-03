"""Regresión del defecto D1 (encontrado por la evaluación 6A): códigos de dominio en las trazas.

El SDK MCP v2 entrega un ``ToolError`` como ``CallToolResult(is_error=True)`` con el texto
``"Error executing tool <tool>: <mensaje>"`` (``mcp/server/mcpserver/tools/base.py``). El
gateway buscaba ``[codigo]`` al inicio del texto y registraba ``code="unknown"`` para todo
error de dominio real. Este archivo prueba el parser con los prefijos reales y con entradas
inesperadas, y el recorrido completo: servidor real -> trazas -> CLI que usa ``/analyze-run``.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pytest

from arl.agent.gateway import domain_code
from arl.agent.runtime import approve, resume, run_turn
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.domain.models import Principal, TaskPatch
from arl.services.project_service import ProjectService
from arl.storage.db import connect
from arl.tracing.__main__ import main as tracing_cli
from arl.tracing.query import report_for

pytestmark = pytest.mark.anyio


# --- Parser --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "text", "code"),
    [
        # Formato real del SDK para un ToolError (verificado contra el servidor en memoria abajo).
        (
            "update_task",
            "Error executing tool update_task: [forbidden] el rol 'viewer'…",
            "forbidden",
        ),
        (
            "get_project",
            "Error executing tool get_project: [not_found] proyecto 'zeta'…",
            "not_found",
        ),
        (
            "update_task",
            "Error executing tool update_task: [invalid_change] 'dave'…",
            "invalid_change",
        ),
        (
            "update_task",
            "Error executing tool update_task: [version_conflict] tarea…",
            "version_conflict",
        ),
        (
            "update_task",
            "Error executing tool update_task: [idempotency_conflict] la clave…",
            "idempotency_conflict",
        ),
        ("update_task", "[forbidden] sin prefijo (formato del dominio)", "forbidden"),
    ],
)
def test_domain_codes_are_read_from_the_real_sdk_format(tool, text, code) -> None:
    assert domain_code(tool, text) == code


@pytest.mark.parametrize(
    ("tool", "text"),
    [
        ("update_task", "Error executing tool update_task"),  # crash opaco: sin código
        ("update_task", "Error executing tool update_task: 1 validation error for …"),
        ("update_task", "Error executing tool update_task: [desconocido] algo"),  # no es de dominio
        ("update_task", "Error executing tool update_task: [Forbidden] mayúsculas"),
        ("update_task", "Error executing tool update_task: [forbidden"),  # malformado
        ("update_task", "Error executing tool update_task: texto [forbidden] en el medio"),
        ("update_task", "Error executing tool get_project: [forbidden] prefijo de otra tool"),
        ("update_task", "Error executing tool update_task:[forbidden] sin espacio"),
        ("update_task", " [forbidden] con espacio inicial"),
        ("update_task", ""),
    ],
)
def test_unexpected_formats_are_never_mapped_to_a_known_code(tool, text) -> None:
    assert domain_code(tool, text) is None


# --- Recorrido completo: servidor real -> trazas -> /analyze-run ---------------------------------


def _model(task_id: str, changes: dict) -> ScriptedChatModel:
    return ScriptedChatModel([call("update_task", task_id=task_id, changes=changes), say("ok")])


async def _approved(open_agent, user: str, model: ScriptedChatModel, before=None) -> None:
    async with open_agent(user, model) as (graph, _):
        paused = await run_turn(graph, thread_id=f"t-{user}", text="cambiá la tarea")
        if before:
            before()
        await resume(graph, thread_id=f"t-{user}", decision=approve(paused.approval_request))


def _codes(trace_dir: Path) -> tuple[list[str], list[str]]:
    report = report_for(sorted(trace_dir.glob("thread-*/*.jsonl")))
    errors = [e["code"] for e in report["errors"] if e["event"] == "mcp_call"]
    outcomes = [w["result"]["outcome"] for w in report["writes"] if w["result"]]
    return errors, outcomes


async def test_forbidden_reaches_the_traces(open_agent, trace_dir) -> None:
    await _approved(open_agent, "carol", _model("apo-3", {"status": "done"}))  # viewer
    assert _codes(trace_dir) == (["forbidden"], ["forbidden"])


async def test_invalid_change_reaches_the_traces(open_agent, trace_dir) -> None:
    await _approved(open_agent, "alice", _model("apo-3", {"assignee_id": "dave"}))
    assert _codes(trace_dir) == (["invalid_change"], ["invalid_change"])


async def test_version_conflict_reaches_the_traces(open_agent, trace_dir, db_path) -> None:
    def concurrent_edit() -> None:
        with closing(connect(db_path)) as conn:
            ProjectService(conn).update_task(
                Principal(user_id="bob"), "apo-3", TaskPatch(title="Otro"),
                expected_version=1, idempotency_key="concurrente-1",
            )  # fmt: skip

    await _approved(open_agent, "alice", _model("apo-3", {"status": "done"}), concurrent_edit)
    assert _codes(trace_dir) == (["version_conflict"], ["version_conflict"])


async def test_not_found_on_read_reaches_the_traces_and_the_cli(
    open_agent, trace_dir, capsys
) -> None:
    model = ScriptedChatModel([call("get_project", project_id="zeta"), say("no está")])
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id="t-read", text="mostrame zeta")
    assert _codes(trace_dir) == (["not_found"], [])
    # Lo que ve /analyze-run: la CLI de consulta, en texto y en JSON.
    assert tracing_cli(["show", result.run_id, "--dir", str(trace_dir)]) == 0
    assert "code=not_found" in capsys.readouterr().out
    assert tracing_cli(["show", result.run_id, "--dir", str(trace_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [e["code"] for e in report["errors"]] == ["not_found"]


async def test_cli_shows_the_reason_of_a_refused_write(open_agent, trace_dir, capsys) -> None:
    await _approved(open_agent, "carol", _model("apo-3", {"status": "done"}))
    assert tracing_cli(["thread", "t-carol", "--dir", str(trace_dir)]) == 0
    text = capsys.readouterr().out
    assert "REFUSED_BY_SERVER" in text and "[forbidden]" in text and "code=forbidden" in text
    assert "unknown" not in text
