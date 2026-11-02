"""Contratos de las librerías fijadas en uv.lock.

La arquitectura depende de APIs concretas del SDK MCP v2 y de LangGraph 1.x. Si una actualización
las cambia, estos tests fallan antes que el código de producto.
"""

from __future__ import annotations

import sqlite3
from importlib.metadata import version
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import Client
from mcp.server import MCPServer


def _major(dist: str) -> int:
    return int(version(dist).split(".")[0])


def test_pinned_major_versions() -> None:
    assert _major("mcp") == 2
    assert _major("langgraph") == 1
    assert _major("langgraph-checkpoint-sqlite") == 3


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_v2_in_memory_roundtrip() -> None:
    """MCPServer + Client(server) en memoria: la base de los tests de integración de la etapa 2."""
    server = MCPServer("contract")

    @server.tool()
    def echo(text: str) -> str:
        """Devuelve el texto recibido."""
        return text

    async with Client(server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == ["echo"]
        result = await client.call_tool("echo", {"text": "hola"})
        assert result.is_error is False
        assert result.structured_content == {"result": "hola"}


class _State(TypedDict):
    calls: list[str]
    decision: str


def test_langgraph_interrupt_reexecutes_node_and_resumes_from_sqlite(tmp_path) -> None:
    """Supuesto clave del diseño: al reanudar, el nodo con interrupt() se re-ejecuta desde el
    inicio, y el checkpoint en SQLite permite reanudar con otra instancia del grafo."""
    executions: list[str] = []

    def approval(state: _State) -> dict:
        executions.append("approval")
        decision = interrupt({"question": "¿aprobar?"})
        return {"decision": decision}

    def build(conn: sqlite3.Connection):
        builder = StateGraph(_State)
        builder.add_node("approval", approval)
        builder.add_edge(START, "approval")
        builder.add_edge("approval", END)
        return builder.compile(checkpointer=SqliteSaver(conn))

    db = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "t-1"}}

    with sqlite3.connect(db, check_same_thread=False) as conn:
        first = build(conn).invoke({"calls": [], "decision": ""}, config)
        assert first["__interrupt__"][0].value == {"question": "¿aprobar?"}

    with sqlite3.connect(db, check_same_thread=False) as conn:
        resumed = build(conn).invoke(Command(resume="approve"), config)

    assert resumed["decision"] == "approve"
    assert executions == ["approval", "approval"]
