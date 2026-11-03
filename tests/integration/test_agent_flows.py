"""Flujos del agente: lectura (1), escritura aprobada (2), rechazada (3) y sin permisos (4)."""

from __future__ import annotations

import json
import sqlite3

import pytest
from langchain_core.messages import ToolMessage

from arl.agent.runtime import TurnStatus, approve, reject, resume, run_turn
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.tool_registry import INJECTED_WRITE_ARGS

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

IDENTITY_FIELDS = {"user_id", "principal", "principal_id", "actor_id", "role"}


def _db(conn: sqlite3.Connection, task_id: str = "apo-3") -> tuple[tuple, int, int]:
    task = tuple(
        conn.execute("SELECT status, version FROM tasks WHERE id = ?", (task_id,)).fetchone()
    )
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    keys = conn.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
    return task, audits, keys


async def _tool_messages(graph, thread_id: str) -> list[ToolMessage]:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [m for m in state.values["messages"] if isinstance(m, ToolMessage)]


def _write_script(task_id: str = "apo-3", **changes) -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id=task_id, changes=changes or {"status": "done"}), say("Listo.")]
    )


# --- 1. Consulta de proyectos y tareas mediante MCP ----------------------------------------------


async def test_read_flow_queries_projects_and_tasks_without_approval(open_agent, conn) -> None:
    model = ScriptedChatModel(
        [call("list_projects"), call("get_project", project_id="apollo"), say("Apollo: 4 tareas.")]
    )
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id="t-read", text="¿Apollo?", scope=READ_ONLY)
        tools = await _tool_messages(graph, "t-read")

    assert (result.status, result.reply, result.approval_request) == (
        TurnStatus.COMPLETED,
        "Apollo: 4 tareas.",
        None,
    )
    listing, detail = (json.loads(m.content) for m in tools)
    assert [p["id"] for p in listing["result"]] == ["apollo", "cassini"]
    assert [t["id"] for t in detail["tasks"]] == ["apo-1", "apo-2", "apo-3", "apo-4"]
    assert model.prompts[-1][-1].content == tools[-1].content  # el modelo vio el resultado real
    assert _db(conn) == (("todo", 1), 0, 0)


async def test_llm_facing_tools_hide_identity_and_injected_arguments(open_agent) -> None:
    model = ScriptedChatModel([say("hola")])
    async with open_agent("alice", model):
        pass
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in model.bound_tools}
    assert set(tools) == {"list_projects", "get_project", "update_task"}
    for schema in tools.values():
        assert not set(schema.get("properties", {})) & (IDENTITY_FIELDS | INJECTED_WRITE_ARGS)
    assert set(tools["update_task"]["properties"]) == {"task_id", "changes"}


# --- 2. Escritura aprobada -----------------------------------------------------------------------


async def test_approved_write_runs_only_after_approval(open_agent, conn) -> None:
    async with open_agent("alice", _write_script()) as (graph, _):
        paused = await run_turn(graph, thread_id="t-w", text="cerrá apo-3", scope=CLOSE_APO3)
        assert paused.status is TurnStatus.AWAITING_APPROVAL
        assert _db(conn) == (("todo", 1), 0, 0)  # nada escrito antes de aprobar

        request = paused.approval_request
        assert {
            k: request[k] for k in ("task_id", "project_id", "my_role", "expected_version")
        } == {
            "task_id": "apo-3",
            "project_id": "apollo",
            "my_role": "owner",
            "expected_version": 1,
        }
        assert (request["before"], request["changes"]) == ({"status": "todo"}, {"status": "done"})

        done = await resume(graph, thread_id="t-w", decision=approve(request))
        tools = await _tool_messages(graph, "t-w")

    assert (done.status, done.reply) == (TurnStatus.COMPLETED, "Listo.")
    assert json.loads(tools[-1].content)["outcome"] == "updated"
    assert _db(conn) == (("done", 2), 1, 1)
    actor, key = conn.execute("SELECT actor_id, idempotency_key FROM audit_log").fetchone()
    assert actor == "alice" and key.startswith("arl-")  # clave generada fuera del LLM


# --- 3. Escritura rechazada ----------------------------------------------------------------------


async def test_rejected_write_never_reaches_the_server(open_agent, conn) -> None:
    async with open_agent("alice", _write_script()) as (graph, _):
        paused = await run_turn(graph, thread_id="t-r", text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(
            graph,
            thread_id="t-r",
            decision=reject(paused.approval_request, "todavía no"),
        )
        tools = await _tool_messages(graph, "t-r")

    assert done.status is TurnStatus.COMPLETED
    assert tools[-1].status == "error"
    assert "[rejected_by_user]" in tools[-1].content and "todavía no" in tools[-1].content
    assert _db(conn) == (("todo", 1), 0, 0)


# --- 4. Usuario sin permisos ---------------------------------------------------------------------


async def test_viewer_write_is_rejected_by_the_server_after_approval(open_agent, conn) -> None:
    """El agente no decide permisos: la aprobación muestra el rol y el servidor rechaza."""
    async with open_agent("carol", _write_script()) as (graph, _):
        paused = await run_turn(graph, thread_id="t-v", text="cerrá apo-3", scope=CLOSE_APO3)
        assert paused.approval_request["my_role"] == "viewer"
        await resume(graph, thread_id="t-v", decision=approve(paused.approval_request))
        tools = await _tool_messages(graph, "t-v")

    assert "[forbidden]" in tools[-1].content
    assert _db(conn) == (("todo", 1), 0, 0)


async def test_non_member_cannot_even_reach_approval(open_agent, conn) -> None:
    async with open_agent("dave", _write_script()) as (graph, _):
        result = await run_turn(graph, thread_id="t-d", text="cerrá apo-3", scope=CLOSE_APO3)
        tools = await _tool_messages(graph, "t-d")

    assert result.status is TurnStatus.COMPLETED and result.approval_request is None
    assert "[not_found]" in tools[-1].content
    assert _db(conn) == (("todo", 1), 0, 0)
