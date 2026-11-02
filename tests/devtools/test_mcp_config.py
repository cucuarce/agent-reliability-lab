""".mcp.json: el servidor que usa Claude Code es de solo lectura por autorización del servidor.

El test de punta a punta lanza el comando exacto de .mcp.json por stdio (como Claude Code) contra
una base de desarrollo temporal e intenta escribir en todos los proyectos.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from arl.storage import seed

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def server_config() -> dict:
    config = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    assert set(config["mcpServers"]) == {"arl"}
    return config["mcpServers"]["arl"]


def test_registered_server_uses_read_only_principal_and_separate_db(server_config: dict) -> None:
    env = server_config["env"]
    assert env["ARL_PRINCIPAL"] == "dev-viewer"
    assert env["ARL_DB_NAME"] not in ("", "app.db")
    assert "ARL_DATA_DIR" not in env  # usa el default local, fuera de OneDrive
    assert server_config["args"][-2:] == ["-m", "arl.mcp_server"]


def test_claude_code_asks_before_update_task(repo_root: Path) -> None:
    settings = json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert "mcp__arl__update_task" in settings["permissions"]["ask"]


@pytest.mark.anyio
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv no está en PATH")
async def test_registered_server_rejects_writes_end_to_end(
    server_config: dict, tmp_path: Path
) -> None:
    db_name = server_config["env"]["ARL_DB_NAME"]
    principal = server_config["env"]["ARL_PRINCIPAL"]
    seed.main(
        ["--db-name", db_name, "--read-only-user", principal], {"ARL_DATA_DIR": str(tmp_path)}
    )
    params = StdioServerParameters(
        command=server_config["command"],
        args=[a.replace("${CLAUDE_PROJECT_DIR:-.}", str(REPO_ROOT)) for a in server_config["args"]],
        env={**os.environ, **server_config["env"], "ARL_DATA_DIR": str(tmp_path)},
    )

    async with Client(params) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        projects = await client.call_tool("list_projects", {})
        writes = [
            await client.call_tool(
                "update_task",
                {
                    "task_id": task_id,
                    "changes": {"status": "done"},
                    "expected_version": 1,
                    "idempotency_key": f"claude-code-{task_id}",
                },
            )
            for task_id in ("apo-3", "bor-2", "cas-2")
        ]

    assert tools == {"list_projects", "get_project", "update_task"}
    assert {p["my_role"] for p in projects.structured_content["result"]} == {"viewer"}
    for result in writes:
        assert result.is_error is True
        assert "[forbidden]" in result.content[0].text

    with closing(sqlite3.connect(tmp_path / db_name)) as conn:
        assert conn.execute("SELECT count(*) FROM audit_log").fetchone()[0] == 0
        assert conn.execute("SELECT max(version) FROM tasks").fetchone()[0] == 1
