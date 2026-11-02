"""Conexión SQLite, schema y límites transaccionales explícitos."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5_000


def connect(path: Path | str) -> sqlite3.Connection:
    """Abre una conexión configurada.

    ``isolation_level=None`` desactiva las transacciones implícitas del módulo ``sqlite3``: cada
    transacción se abre y se cierra explícitamente con :func:`transaction`.
    """
    conn = sqlite3.connect(path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection, *, write: bool) -> Iterator[sqlite3.Connection]:
    """Una transacción completa: COMMIT al salir, ROLLBACK ante cualquier excepción.

    Las escrituras usan ``BEGIN IMMEDIATE``: toman el lock de escritura antes de la primera
    lectura, así la verificación (authz, versión, ledger) y la escritura ven el mismo estado y
    dos escritores concurrentes se serializan en lugar de fallar a mitad de camino.
    """
    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_schema(conn: sqlite3.Connection) -> None:
    """Crea el schema si la base está vacía; falla si la base tiene otra versión."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == SCHEMA_VERSION:
        return
    if current != 0:
        raise RuntimeError(f"schema v{current} no soportado (se esperaba v{SCHEMA_VERSION})")
    script = files("arl.storage").joinpath("schema.sql").read_text(encoding="utf-8")
    try:
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")
    except sqlite3.Error:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
