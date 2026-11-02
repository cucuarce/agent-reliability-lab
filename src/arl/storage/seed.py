"""Datos sintéticos: validación de fixtures y creación determinística de app.db."""

from __future__ import annotations

import json
import sqlite3
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from arl.domain.models import Membership, Project, Task, User
from arl.storage.db import connect, init_schema, transaction


class SeedData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    users: tuple[User, ...]
    projects: tuple[Project, ...]
    memberships: tuple[Membership, ...]
    tasks: tuple[Task, ...]

    @model_validator(mode="after")
    def _assignees_are_members(self) -> SeedData:
        members = {(m.user_id, m.project_id) for m in self.memberships}
        for task in self.tasks:
            if task.assignee_id and (task.assignee_id, task.project_id) not in members:
                raise ValueError(f"{task.id}: '{task.assignee_id}' no es miembro del proyecto")
        return self


def load_seed(path: Path | None = None) -> SeedData:
    """Carga y valida los fixtures; por defecto, los empaquetados con ``arl``."""
    source = path or files("arl.storage").joinpath("seed_data.json")
    return SeedData.model_validate(json.loads(source.read_text(encoding="utf-8")))


def seed(conn: sqlite3.Connection, data: SeedData) -> None:
    """Inserta los fixtures en una sola transacción. Requiere una base sin usuarios."""
    with transaction(conn, write=True):
        if conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            raise RuntimeError("la base ya tiene datos; usá reset=True para recrearla")
        conn.executemany("INSERT INTO users (id, name) VALUES (:id, :name)", _rows(data.users))
        conn.executemany(
            "INSERT INTO projects (id, name, description) VALUES (:id, :name, :description)",
            _rows(data.projects),
        )
        conn.executemany(
            "INSERT INTO memberships (user_id, project_id, role) "
            "VALUES (:user_id, :project_id, :role)",
            _rows(data.memberships),
        )
        conn.executemany(
            "INSERT INTO tasks (id, project_id, title, description, status, assignee_id, version) "
            "VALUES (:id, :project_id, :title, :description, :status, :assignee_id, :version)",
            _rows(data.tasks),
        )


def create_database(
    path: Path, data: SeedData | None = None, *, reset: bool = False
) -> sqlite3.Connection:
    """Crea (o recrea con ``reset``) una base con schema y datos sintéticos."""
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        init_schema(conn)
        seed(conn, data or load_seed())
    except BaseException:
        conn.close()  # en Windows una conexión abierta bloquea el archivo
        raise
    return conn


def _rows(models: tuple[BaseModel, ...]) -> list[dict]:
    return [model.model_dump(mode="json") for model in models]
