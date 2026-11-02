"""Registro determinístico de tools del agente: qué ve el LLM y qué es lectura o escritura.

La clasificación READ/WRITE está en código, no en los hints MCP (que el spec define como no
confiables). Los argumentos que ve el LLM son un subconjunto estricto de los del servidor: la
versión esperada y la clave de idempotencia las inyecta el agente, nunca el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from arl.domain.models import Identifier, TaskPatch


class ToolKind(StrEnum):
    READ = "read"
    WRITE = "write"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListProjectsArgs(_Args):
    pass


class GetProjectArgs(_Args):
    project_id: Identifier


class UpdateTaskArgs(_Args):
    task_id: Identifier
    changes: TaskPatch


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    args_model: type[_Args]
    description: str


REGISTRY: MappingProxyType[str, ToolSpec] = MappingProxyType(
    {
        "list_projects": ToolSpec(
            "list_projects",
            ToolKind.READ,
            ListProjectsArgs,
            "Lista los proyectos visibles para el usuario actual y su rol en cada uno.",
        ),
        "get_project": ToolSpec(
            "get_project",
            ToolKind.READ,
            GetProjectArgs,
            "Devuelve un proyecto visible con sus tareas (id, título, estado, responsable).",
        ),
        "update_task": ToolSpec(
            "update_task",
            ToolKind.WRITE,
            UpdateTaskArgs,
            "Propone modificar title, status y/o assignee_id de una tarea. La modificación "
            "requiere aprobación humana antes de ejecutarse.",
        ),
    }
)

# Argumentos del servidor que el agente inyecta y el LLM nunca ve.
INJECTED_WRITE_ARGS = frozenset({"expected_version", "idempotency_key"})


class InvalidToolCall(ValueError):
    """Tool desconocida o argumentos inválidos. El mensaje no repite valores de entrada."""


def tool_definitions() -> list[dict[str, Any]]:
    """Definiciones para ``bind_tools`` (formato de función estándar)."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_model.model_json_schema(),
            },
        }
        for spec in REGISTRY.values()
    ]


def validate_call(name: str, args: Any) -> tuple[ToolSpec, _Args]:
    spec = REGISTRY.get(name)
    if spec is None:
        raise InvalidToolCall(f"tool desconocida: '{name}'")
    try:
        return spec, spec.args_model.model_validate(args)
    except ValidationError as exc:
        problems = ", ".join(
            f"{'.'.join(str(p) for p in error['loc']) or '(raíz)'} ({error['type']})"
            for error in exc.errors()
        )
        raise InvalidToolCall(f"argumentos inválidos para {name}: {problems}") from None
