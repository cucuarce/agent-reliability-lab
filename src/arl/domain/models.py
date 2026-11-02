"""Modelos del dominio. Inmutables: los cambios pasan siempre por ProjectService."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
IdempotencyKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{8,128}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    OWNER = "owner"


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


class Principal(_Frozen):
    """Identidad **simulada** del actor de una operación. No autentica a nadie.

    Solo valida la *estructura* del identificador (formato canónico); no verifica credenciales ni
    que la persona sea quien dice ser. Su garantía es de procedencia: lo construye código de
    confianza (la CLI, o el servidor MCP a partir de su entorno de ejecución), nunca argumentos
    generados por el LLM. ProjectService no acepta un ``user_id`` suelto: exige un Principal.
    """

    user_id: Identifier


class User(_Frozen):
    id: Identifier
    name: str


class Project(_Frozen):
    id: Identifier
    name: str
    description: str = ""


class Membership(_Frozen):
    user_id: Identifier
    project_id: Identifier
    role: Role


class Task(_Frozen):
    id: Identifier
    project_id: Identifier
    title: Title
    description: str = ""
    status: TaskStatus
    assignee_id: Identifier | None = None
    version: int = Field(ge=1)


class TaskPatch(_Frozen):
    """Cambios solicitados sobre una tarea. Solo cuentan los campos enviados explícitamente:
    ``assignee_id=None`` significa "desasignar", omitirlo significa "no tocar"."""

    title: Title | None = None
    status: TaskStatus | None = None
    assignee_id: Identifier | None = None

    @model_validator(mode="after")
    def _validate(self) -> TaskPatch:
        if not self.model_fields_set:
            raise ValueError("el patch no contiene cambios")
        for name in ("title", "status"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"'{name}' no puede ser null")
        return self

    def changes(self) -> dict[str, Any]:
        """Campos enviados, serializados para SQL/JSON (orden estable)."""
        return self.model_dump(mode="json", include=self.model_fields_set)


class ProjectSummary(_Frozen):
    id: Identifier
    name: str
    my_role: Role


class ProjectDetail(_Frozen):
    project: Project
    my_role: Role
    tasks: tuple[Task, ...]


class UpdateOutcome(StrEnum):
    """``UNCHANGED``: el patch coincidía con el estado actual; sin versión ni auditoría nuevas."""

    UPDATED = "updated"
    UNCHANGED = "unchanged"


class UpdateResult(_Frozen):
    task: Task
    outcome: UpdateOutcome
    audit_id: int | None
    replayed: bool = False

    @model_validator(mode="after")
    def _audit_matches_outcome(self) -> UpdateResult:
        if (self.audit_id is None) != (self.outcome is UpdateOutcome.UNCHANGED):
            raise ValueError("audit_id debe existir si y solo si la tarea se modificó")
        return self
