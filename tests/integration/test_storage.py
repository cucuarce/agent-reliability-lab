from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from arl.storage.db import SCHEMA_VERSION, connect, init_schema
from arl.storage.seed import create_database, load_seed


def _dump_hash(conn: sqlite3.Connection) -> str:
    return hashlib.sha256("\n".join(conn.iterdump()).encode("utf-8")).hexdigest()


def test_seed_is_deterministic(tmp_path: Path) -> None:
    with (
        closing(create_database(tmp_path / "a.db")) as first,
        closing(create_database(tmp_path / "b.db")) as second,
    ):
        assert _dump_hash(first) == _dump_hash(second)


def test_reset_recreates_database(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    conn = create_database(path)
    conn.execute("UPDATE tasks SET status = 'done', version = 2 WHERE id = 'apo-3'")
    conn.close()
    with pytest.raises(RuntimeError, match="ya tiene datos"):
        create_database(path)
    conn = create_database(path, reset=True)
    assert conn.execute("SELECT version FROM tasks WHERE id = 'apo-3'").fetchone()[0] == 1
    conn.close()


def test_seed_fixtures_are_consistent() -> None:
    data = load_seed()
    assert {u.id for u in data.users} == {"alice", "bob", "carol", "dave", "erin"}
    assert len(data.tasks) == 10


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    conn = connect(tmp_path / "other.db")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="no soportado"):
        init_schema(conn)
    conn.close()


def test_database_constraints_defend_themselves(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = 'apo-1'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE tasks SET assignee_id = 'ghost' WHERE id = 'apo-1'")  # FK
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO memberships VALUES ('erin', 'apollo', 'admin')")


def test_audit_log_is_append_only(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO audit_log (occurred_at, actor_id, action, project_id, task_id, "
        "idempotency_key, before_json, after_json) "
        "VALUES ('t', 'a', 'x', 'p', 't1', 'k', '{}', '{}')"
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE audit_log SET actor_id = 'mallory'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM audit_log")


def test_wal_mode_enabled(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
