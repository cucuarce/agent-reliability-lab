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

from arl.agent.faults import FaultPlan, FaultyClient
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
def lock_dir(checkpoints_path: Path) -> Path:
    """Locks por thread compartidos por todos los procesos del test (y los subprocesos)."""
    return checkpoints_path.parent / "locks"


@pytest.fixture
def trace_dir(checkpoints_path: Path) -> Path:
    """Trazas JSONL de todas las ejecuciones del test (también las de subprocesos)."""
    return checkpoints_path.parent / "traces"


@pytest.fixture
def open_agent(
    db_path: Path, checkpoints_path: Path, lock_dir: Path, trace_dir: Path
) -> Callable[..., AbstractAsyncContextManager[AgentProcess]]:
    """Un "proceso" del agente: checkpointer SQLite + cliente MCP (en memoria) + grafo.

    Abrir y cerrar dos veces el contexto equivale a reiniciar el proceso: solo persisten los
    archivos ``app.db``, ``checkpoints.db`` y los locks.
    """
    locks = lock_dir

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
            yield (
                build_graph(model, gateway, saver, limits, lock_dir=locks, trace_dir=trace_dir),
                gateway,
            )

    return factory


# Reintentos sin espera para que los tests de fallos sean rápidos y deterministas.
FAST_RETRY = Limits(retry_initial_interval=0.0, retry_max_interval=0.0, retry_jitter=False)


@pytest.fixture
def open_faulty_agent(
    db_path: Path, checkpoints_path: Path, lock_dir: Path, trace_dir: Path
) -> Callable[..., AbstractAsyncContextManager[tuple[CompiledStateGraph, FaultyClient]]]:
    """Como ``open_agent``, pero con un ``FaultyClient`` entre el gateway y el servidor."""
    locks = lock_dir

    @asynccontextmanager
    async def factory(
        user_id: str,
        model: ScriptedChatModel,
        plan: FaultPlan | None = None,
        limits: Limits = FAST_RETRY,
    ) -> AsyncIterator[tuple[CompiledStateGraph, FaultyClient]]:
        server = create_server(db_path, Principal(user_id=user_id))
        async with (
            AsyncSqliteSaver.from_conn_string(str(checkpoints_path)) as saver,
            Client(server) as client,
        ):
            faulty = FaultyClient(client, plan or {})
            gateway = McpGateway(faulty, user_id)  # type: ignore[arg-type]
            yield (
                build_graph(model, gateway, saver, limits, lock_dir=locks, trace_dir=trace_dir),
                faulty,
            )

    return factory


@pytest.fixture
def invariants() -> Callable[[sqlite3.Connection, set[str]], None]:
    return check_invariants


def check_invariants(conn: sqlite3.Connection, approved_keys: set[str]) -> None:
    """Invariantes de seguridad del dominio, verificables después de cualquier fallo.

    - solo se escribieron acciones aprobadas, sin duplicados;
    - cada auditoría tiene su entrada de ledger (escritura atómica);
    - la versión de cada tarea es 1 + número de modificaciones auditadas.
    """
    rows = conn.execute("SELECT actor_id, idempotency_key, task_id FROM audit_log").fetchall()
    audits = [tuple(row) for row in rows]
    keys = [key for _, key, _ in audits]
    assert len(keys) == len(set(keys)), "escritura duplicada"
    assert set(keys) <= approved_keys, "escritura sin aprobación"
    ledger = {tuple(r) for r in conn.execute("SELECT principal_id, key FROM idempotency_keys")}
    assert {(actor, key) for actor, key, _ in audits} <= ledger, "auditoría sin ledger"
    for task_id, version in conn.execute("SELECT id, version FROM tasks").fetchall():
        modifications = sum(1 for _, _, audited in audits if audited == task_id)
        assert version == 1 + modifications, f"versión inconsistente en {task_id}"
