from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arl.domain.models import Principal
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
