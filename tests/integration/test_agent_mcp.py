"""Integración del agente con el servidor MCP: contrato de tools y servidor real por stdio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client
from mcp.server import MCPServer

from arl.agent.gateway import ContractMismatch, McpGateway, stdio_server_parameters
from arl.agent.graph import build_graph
from arl.agent.runtime import TurnStatus, approve, resume, run_turn, thread_config
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio


async def test_gateway_rejects_a_server_with_a_different_contract() -> None:
    impostor = MCPServer("impostor")

    @impostor.tool()
    def list_projects() -> list[str]:
        return []

    async with Client(impostor) as client:
        with pytest.raises(ContractMismatch):
            await McpGateway(client, "alice").verify_contract()


async def test_agent_against_real_stdio_server_process(db_path: Path, tmp_path: Path, conn) -> None:
    """Agente → stdio → ``python -m arl.mcp_server`` (otro proceso) → ProjectService → SQLite."""
    model = ScriptedChatModel(
        [
            call("get_project", project_id="apollo"),
            call("update_task", task_id="apo-3", changes={"status": "done"}),
            say("Hecho."),
        ]
    )
    params = stdio_server_parameters("alice", data_dir=db_path.parent, db_name=db_path.name)
    async with (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver,
        Client(params) as client,
    ):
        gateway = McpGateway(client, "alice")
        await gateway.verify_contract()
        graph = build_graph(model, gateway, saver, lock_dir=tmp_path / "locks")
        paused = await run_turn(graph, thread_id="t-io", text="cerrá apo-3", scope=CLOSE_APO3)
        assert paused.status is TurnStatus.AWAITING_APPROVAL
        done = await resume(graph, thread_id="t-io", decision=approve(paused.approval_request))
        state = await graph.aget_state(thread_config("t-io"))

    assert done.status is TurnStatus.COMPLETED
    tools = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert json.loads(tools[-1].content)["outcome"] == "updated"
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == ("done", 2)
