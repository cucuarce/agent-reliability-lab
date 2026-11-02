from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from mcp import Client

from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.scripted_model import ScriptedChatModel
from arl.agent.state import Limits
from arl.domain.models import Principal
from arl.mcp_server.server import create_server
from arl.services.project_service import ProjectService
from arl.storage.db import connect
from arl.storage.seed import create_database

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Base sembrada en un archivo real (WAL), fuera de OneDrive."""
    path = tmp_path / "app.db"
    create_database(path).close()
    return path


@pytest.fixture
def open_conn(db_path: Path) -> Iterator[Callable[[], sqlite3.Connection]]:
    """Fábrica de conexiones independientes, como procesos distintos sobre la misma base."""
    opened: list[sqlite3.Connection] = []

    def factory() -> sqlite3.Connection:
        conn = connect(db_path)
        opened.append(conn)
        return conn

    yield factory
    for conn in opened:
        conn.close()


@pytest.fixture
def conn(open_conn: Callable[[], sqlite3.Connection]) -> sqlite3.Connection:
    return open_conn()


@pytest.fixture
def service(conn: sqlite3.Connection) -> ProjectService:
    return ProjectService(conn, clock=lambda: FIXED_NOW)


@pytest.fixture
def as_user() -> Callable[[str], Principal]:
    return lambda user_id: Principal(user_id=user_id)


AgentProcess = tuple[CompiledStateGraph, McpGateway]


@pytest.fixture
def checkpoints_path(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints.db"


@pytest.fixture
def open_agent(
    db_path: Path, checkpoints_path: Path
) -> Callable[..., AbstractAsyncContextManager[AgentProcess]]:
    """Un "proceso" del agente: checkpointer SQLite + cliente MCP (en memoria) + grafo.

    Abrir y cerrar dos veces el contexto equivale a reiniciar el proceso: solo persisten los
    archivos ``app.db`` y ``checkpoints.db``.
    """

    @asynccontextmanager
    async def factory(
        user_id: str, model: ScriptedChatModel, limits: Limits | None = None
    ) -> AsyncIterator[AgentProcess]:
        server = create_server(db_path, Principal(user_id=user_id))
        async with (
            AsyncSqliteSaver.from_conn_string(str(checkpoints_path)) as saver,
            Client(server) as client,
        ):
            gateway = McpGateway(client, user_id)
            await gateway.verify_contract()
            yield build_graph(model, gateway, saver, limits), gateway

    return factory
