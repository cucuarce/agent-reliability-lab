"""Ledger de idempotencia y auditoría append-only.

Se escriben dentro de la misma transacción que la modificación de la tarea: o quedan las tres
cosas (tarea, ledger, auditoría) o no queda ninguna.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: str
    request_hash: str
    response_json: str


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    occurred_at: str
    actor_id: str
    action: str
    project_id: str
    task_id: str
    idempotency_key: str
    before: dict[str, Any]
    after: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_hash(operation: str, params: dict[str, Any]) -> str:
    """Huella de los parámetros de una operación: misma clave + distinta huella = conflicto."""
    payload = canonical_json({"operation": operation, "params": params})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find(conn: sqlite3.Connection, principal_id: str, key: str) -> LedgerEntry | None:
    row = conn.execute(
        """
        SELECT operation, request_hash, response_json FROM idempotency_keys
        WHERE principal_id = ? AND key = ?
        """,
        (principal_id, key),
    ).fetchone()
    return LedgerEntry.model_validate(dict(row)) if row else None


def record(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    key: str,
    operation: str,
    request_hash: str,
    response_json: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO idempotency_keys
            (principal_id, key, operation, request_hash, response_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (principal_id, key, operation, request_hash, response_json, created_at),
    )


def append_audit(
    conn: sqlite3.Connection,
    *,
    occurred_at: str,
    actor_id: str,
    action: str,
    project_id: str,
    task_id: str,
    idempotency_key: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO audit_log (occurred_at, actor_id, action, project_id, task_id,
                               idempotency_key, before_json, after_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            occurred_at,
            actor_id,
            action,
            project_id,
            task_id,
            idempotency_key,
            canonical_json(before),
            canonical_json(after),
        ),
    )
    return int(cursor.lastrowid)  # type: ignore[arg-type]  # siempre definido tras un INSERT


def list_audit(conn: sqlite3.Connection, task_id: str | None = None) -> list[AuditEntry]:
    query = "SELECT * FROM audit_log"
    params: tuple[str, ...] = ()
    if task_id is not None:
        query += " WHERE task_id = ?"
        params = (task_id,)
    entries = []
    for row in conn.execute(query + " ORDER BY id", params).fetchall():
        data = dict(row)
        data["before"] = json.loads(data.pop("before_json"))
        data["after"] = json.loads(data.pop("after_json"))
        entries.append(AuditEntry.model_validate(data))
    return entries
