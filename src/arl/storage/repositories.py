"""Consultas SQL del dominio. Sin lógica de autorización: eso es responsabilidad de ProjectService.

Todas las funciones reciben la conexión de la transacción en curso.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from arl.domain.models import Project, ProjectSummary, Role, Task

_TASK_COLUMNS = "id, project_id, title, description, status, assignee_id, version"
_UPDATABLE_COLUMNS = frozenset({"title", "status", "assignee_id"})


def _task(row: sqlite3.Row) -> Task:
    return Task.model_validate(dict(row))


def get_role(conn: sqlite3.Connection, user_id: str, project_id: str) -> Role | None:
    row = conn.execute(
        "SELECT role FROM memberships WHERE user_id = ? AND project_id = ?",
        (user_id, project_id),
    ).fetchone()
    return Role(row["role"]) if row else None


def list_projects_for_user(conn: sqlite3.Connection, user_id: str) -> list[ProjectSummary]:
    rows = conn.execute(
        """
        SELECT p.id, p.name, m.role AS my_role
        FROM projects AS p JOIN memberships AS m ON m.project_id = p.id
        WHERE m.user_id = ?
        ORDER BY p.id
        """,
        (user_id,),
    ).fetchall()
    return [ProjectSummary.model_validate(dict(row)) for row in rows]


def get_project(conn: sqlite3.Connection, project_id: str) -> Project | None:
    row = conn.execute(
        "SELECT id, name, description FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    return Project.model_validate(dict(row)) if row else None


def list_tasks(conn: sqlite3.Connection, project_id: str) -> list[Task]:
    rows = conn.execute(
        f"SELECT {_TASK_COLUMNS} FROM tasks WHERE project_id = ? ORDER BY id", (project_id,)
    ).fetchall()
    return [_task(row) for row in rows]


def get_task(conn: sqlite3.Connection, task_id: str) -> Task | None:
    row = conn.execute(f"SELECT {_TASK_COLUMNS} FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _task(row) if row else None


def update_task(
    conn: sqlite3.Connection, task_id: str, changes: dict[str, Any], expected_version: int
) -> bool:
    """Aplica ``changes`` solo si la versión sigue siendo ``expected_version`` (compare-and-set).

    Devuelve False si otra escritura ganó la carrera. Los nombres de columna salen de una lista
    blanca fija; los valores van siempre como parámetros.
    """
    unknown = changes.keys() - _UPDATABLE_COLUMNS
    if unknown or not changes:
        raise ValueError(f"columnas no actualizables: {sorted(unknown)}")
    assignments = ", ".join(f"{column} = ?" for column in sorted(changes))
    values = [changes[column] for column in sorted(changes)]
    cursor = conn.execute(
        f"UPDATE tasks SET {assignments}, version = version + 1 WHERE id = ? AND version = ?",
        (*values, task_id, expected_version),
    )
    return cursor.rowcount == 1
