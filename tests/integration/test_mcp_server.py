"""Servidor MCP en memoria: Client(server) -> MCPServer -> ProjectService -> SQLite."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from arl.domain.models import Principal
from arl.mcp_server.server import create_server
from arl.services.project_service import ProjectService

pytestmark = pytest.mark.anyio

TOOLS = {"list_projects", "get_project", "update_task"}
IDENTITY_FIELDS = {
    "user_id", "user", "username", "principal", "principal_id", "actor", "actor_id",
    "identity", "as_user", "on_behalf_of", "impersonate", "role",
}  # fmt: skip

Connect = Callable[[str], AbstractAsyncContextManager[Client]]


@pytest.fixture
def connect_as(db_path: Path) -> Connect:
    @asynccontextmanager
    async def factory(user_id: str) -> AsyncIterator[Client]:
        async with Client(create_server(db_path, Principal(user_id=user_id))) as client:
            yield client

    return factory


def _update_args(task_id: str = "apo-3", key: str = "mcp-key-0001", **overrides: Any) -> dict:
    return {
        "task_id": task_id,
        "changes": {"status": "done"},
        "expected_version": 1,
        "idempotency_key": key,
    } | overrides


def _error_text(result) -> str:
    assert result.is_error is True
    assert result.structured_content is None
    return result.content[0].text


def _property_names(schema: Any) -> set[str]:
    """Todos los nombres de propiedad del schema, incluidos $defs anidados."""
    names: set[str] = set()
    if isinstance(schema, dict):
        names |= set(schema.get("properties", {}))
        for value in schema.values():
            names |= _property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            names |= _property_names(item)
    return names


def _db_state(conn: sqlite3.Connection) -> tuple[list[tuple], int, int]:
    versions = conn.execute("SELECT id, status, version FROM tasks ORDER BY id").fetchall()
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    keys = conn.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
    return [tuple(row) for row in versions], audits, keys


# --- Descubrimiento y schemas ---------------------------------------------------------------------


async def test_discovery_lists_exactly_three_tools_with_hints(connect_as: Connect) -> None:
    async with connect_as("alice") as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    assert tools["list_projects"].annotations.read_only_hint is True
    assert tools["get_project"].annotations.read_only_hint is True
    write = tools["update_task"].annotations
    assert (write.read_only_hint, write.destructive_hint, write.idempotent_hint) == (
        False,
        False,
        True,
    )
    assert all(tool.description for tool in tools.values())


async def test_input_schemas_expose_no_identity_fields(connect_as: Connect) -> None:
    async with connect_as("alice") as client:
        tools = (await client.list_tools()).tools
    for tool in tools:
        assert not _property_names(tool.input_schema) & IDENTITY_FIELDS, tool.name


async def test_update_task_schema_is_strict(connect_as: Connect) -> None:
    async with connect_as("alice") as client:
        (tool,) = [t for t in (await client.list_tools()).tools if t.name == "update_task"]
    schema = tool.input_schema
    assert set(schema["required"]) == {"task_id", "changes", "expected_version", "idempotency_key"}
    patch = schema["$defs"]["TaskPatch"]
    assert patch["additionalProperties"] is False
    assert set(patch["properties"]) == {"title", "status", "assignee_id"}
    assert schema["properties"]["expected_version"]["minimum"] == 1


# --- Ejecuciones exitosas -----------------------------------------------------------------------


async def test_list_projects_returns_only_visible_projects(connect_as: Connect) -> None:
    async with connect_as("alice") as client:
        result = await client.call_tool("list_projects", {})
    assert result.is_error is False
    assert result.structured_content == {
        "result": [
            {"id": "apollo", "name": "Apollo", "my_role": "owner"},
            {"id": "cassini", "name": "Cassini", "my_role": "viewer"},
        ]
    }


async def test_get_project_returns_tasks_with_versions(connect_as: Connect) -> None:
    async with connect_as("carol") as client:
        result = await client.call_tool("get_project", {"project_id": "apollo"})
    detail = result.structured_content
    assert detail["my_role"] == "viewer"
    assert [(t["id"], t["version"]) for t in detail["tasks"]] == [
        ("apo-1", 1),
        ("apo-2", 1),
        ("apo-3", 1),
        ("apo-4", 1),
    ]


async def test_update_task_writes_through_service_to_sqlite(connect_as: Connect, conn) -> None:
    async with connect_as("bob") as client:
        result = await client.call_tool("update_task", _update_args())
    assert result.is_error is False
    assert result.structured_content["outcome"] == "updated"
    assert result.structured_content["task"]["version"] == 2
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == ("done", 2)
    audit = conn.execute("SELECT actor_id, idempotency_key FROM audit_log").fetchall()
    assert [tuple(r) for r in audit] == [("bob", "mcp-key-0001")]  # actor = principal del servidor


async def test_retry_through_mcp_is_idempotent(connect_as: Connect, conn) -> None:
    async with connect_as("alice") as client:
        first = await client.call_tool("update_task", _update_args())
        retry = await client.call_tool("update_task", _update_args())
    assert retry.structured_content == first.structured_content | {"replayed": True}
    assert _db_state(conn)[1:] == (1, 1)


# --- Rechazos por permisos y no-filtración -------------------------------------------------------


async def test_viewer_write_is_forbidden_and_db_unchanged(connect_as: Connect, conn) -> None:
    before = _db_state(conn)
    async with connect_as("carol") as client:
        result = await client.call_tool("update_task", _update_args())
    assert "[forbidden]" in _error_text(result)
    assert _db_state(conn) == before


async def test_foreign_and_missing_resources_get_identical_errors(connect_as: Connect) -> None:
    async with connect_as("dave") as client:
        foreign = await client.call_tool("get_project", {"project_id": "apollo"})
        missing = await client.call_tool("get_project", {"project_id": "zeus"})
        foreign_task = await client.call_tool("update_task", _update_args("apo-3"))
        missing_task = await client.call_tool("update_task", _update_args("nope-1"))
    assert _error_text(foreign).replace("apollo", "X") == _error_text(missing).replace("zeus", "X")
    assert _error_text(foreign_task).replace("apo-3", "X") == _error_text(missing_task).replace(
        "nope-1", "X"
    )
    assert "[not_found]" in _error_text(foreign_task)
    assert "Apollo" not in _error_text(foreign)  # ni el nombre del proyecto ajeno


async def test_identity_arguments_cannot_change_the_principal(connect_as: Connect, conn) -> None:
    """Los argumentos de nivel superior no declarados los ignora el SDK; dentro de ``changes``
    se rechazan. En ningún caso modifican la identidad, que fija el servidor."""
    before = _db_state(conn)
    async with connect_as("dave") as client:
        listed = await client.call_tool("list_projects", {"user_id": "alice"})
        read = await client.call_tool("get_project", {"project_id": "apollo", "user_id": "alice"})
        write = await client.call_tool(
            "update_task", _update_args(principal="alice", actor_id="alice")
        )
        nested = await client.call_tool(
            "update_task", _update_args(changes={"status": "done", "user_id": "alice"})
        )
    assert listed.structured_content == {
        "result": [{"id": "borealis", "name": "Borealis", "my_role": "editor"}]
    }
    assert "[not_found]" in _error_text(read)
    assert "[not_found]" in _error_text(write)
    assert "extra_forbidden" in _error_text(nested)
    assert _db_state(conn) == before


# --- Errores de validación y de dominio ----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"task_id": "APO 3"},
        {"changes": {}},
        {"changes": {"status": "archived"}},
        {"changes": {"title": None}},
        {"expected_version": 0},
        {"idempotency_key": "short"},
    ],
)
async def test_invalid_inputs_are_rejected_before_the_service(
    connect_as: Connect, conn, overrides: dict
) -> None:
    before = _db_state(conn)
    async with connect_as("alice") as client:
        result = await client.call_tool("update_task", _update_args(**overrides))
    assert "Error executing tool update_task" in _error_text(result)
    assert _db_state(conn) == before


async def test_domain_conflicts_map_to_stable_codes(connect_as: Connect) -> None:
    async with connect_as("alice") as client:
        await client.call_tool("update_task", _update_args())
        stale = await client.call_tool(
            "update_task", _update_args(key="mcp-key-0002", changes={"status": "blocked"})
        )
        reused = await client.call_tool("update_task", _update_args(changes={"status": "todo"}))
        bad_assignee = await client.call_tool(
            "update_task",
            _update_args(key="mcp-key-0003", changes={"assignee_id": "dave"}, expected_version=2),
        )
    assert "[version_conflict]" in _error_text(stale)
    assert "[idempotency_conflict]" in _error_text(reused)
    assert "[invalid_change]" in _error_text(bad_assignee)


async def test_unexpected_errors_are_hidden_from_the_client(
    connect_as: Connect, monkeypatch
) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("secreto: C:/datos/app.db, SELECT * FROM users")

    monkeypatch.setattr(ProjectService, "list_projects", crash)
    async with connect_as("alice") as client:
        result = await client.call_tool("list_projects", {})
    assert _error_text(result) == "Error executing tool list_projects"


# --- Aislamiento entre usuarios ------------------------------------------------------------------


async def test_concurrent_clients_stay_isolated(connect_as: Connect, conn) -> None:
    results: dict[str, Any] = {}

    async def session(user_id: str, key: str) -> None:
        async with connect_as(user_id) as client:
            results[user_id] = (
                await client.call_tool("list_projects", {}),
                await client.call_tool("update_task", _update_args(key=key)),
            )

    async with anyio.create_task_group() as group:
        group.start_soon(session, "alice", "shared-key-1")
        group.start_soon(session, "dave", "shared-key-1")  # misma clave, otro principal

    assert [p["id"] for p in results["dave"][0].structured_content["result"]] == ["borealis"]
    assert results["alice"][1].structured_content["outcome"] == "updated"
    assert "[not_found]" in _error_text(results["dave"][1])
    assert _db_state(conn)[1:] == (1, 1)
