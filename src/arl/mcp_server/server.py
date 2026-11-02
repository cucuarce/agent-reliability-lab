"""Servidor MCP: adaptador fino sobre ProjectService (SDK MCP Python v2, ``MCPServer``).

- Identidad: el ``Principal`` se fija al construir el servidor a partir del entorno del proceso
  (ver ``__main__``). Ninguna tool recibe identidad como argumento.
- Autorización: íntegramente en ProjectService; este módulo no contiene reglas.
- Errores: ``DomainError`` -> ``ToolError`` con un código estable (``[not_found] ...``). Cualquier
  otra excepción la oculta el SDK: el cliente solo ve ``Error executing tool <name>``.
- Una conexión SQLite por llamada: el SDK ejecuta las tools síncronas en hilos de trabajo y una
  conexión ``sqlite3`` no se comparte entre hilos.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from arl import __version__
from arl.domain.errors import DomainError
from arl.domain.models import (
    IdempotencyKey,
    Identifier,
    Principal,
    ProjectDetail,
    ProjectSummary,
    TaskPatch,
    UpdateResult,
)
from arl.services.project_service import ProjectService
from arl.storage.db import connect

SERVER_NAME = "arl"
INSTRUCTIONS = (
    "Gestión de proyectos con datos sintéticos de Agent Reliability Lab. La identidad del usuario "
    "la fija el servidor; las tools no aceptan identidad como argumento. Para modificar una tarea, "
    "leé primero el proyecto con get_project y usá la 'version' actual como expected_version."
)

ProjectId = Annotated[Identifier, Field(description="ID del proyecto, p. ej. 'apollo'.")]
TaskId = Annotated[Identifier, Field(description="ID de la tarea, p. ej. 'apo-3'.")]
Changes = Annotated[
    TaskPatch,
    Field(description="Campos a modificar: title, status y/o assignee_id (null = desasignar)."),
]
ExpectedVersion = Annotated[
    int,
    Field(ge=1, description="Versión de la tarea leída con get_project (control de concurrencia)."),
]
Key = Annotated[
    IdempotencyKey,
    Field(description="Clave única por operación; reintentar con la misma clave no duplica."),
]

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


def create_server(db_path: Path, principal: Principal) -> MCPServer:
    """Construye el servidor ligado a una base y a un principal de confianza."""
    if not isinstance(principal, Principal):
        raise TypeError("se requiere un Principal construido por código de confianza")
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version=__version__)

    @contextmanager
    def project_service() -> Iterator[ProjectService]:
        with closing(connect(db_path)) as conn:
            try:
                yield ProjectService(conn)
            except DomainError as exc:
                raise ToolError(f"[{exc.code}] {exc}") from exc

    @server.tool(annotations=_READ_ONLY)
    def list_projects() -> list[ProjectSummary]:
        """Lista los proyectos visibles para el usuario actual y su rol en cada uno."""
        with project_service() as service:
            return service.list_projects(principal)

    @server.tool(annotations=_READ_ONLY)
    def get_project(project_id: ProjectId) -> ProjectDetail:
        """Devuelve un proyecto visible con todas sus tareas (incluida la versión de cada una)."""
        with project_service() as service:
            return service.get_project(principal, project_id)

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    def update_task(
        task_id: TaskId, changes: Changes, expected_version: ExpectedVersion, idempotency_key: Key
    ) -> UpdateResult:
        """Modifica una tarea si el usuario actual tiene permiso de escritura en su proyecto.

        Falla con version_conflict si la tarea cambió desde expected_version. Si los cambios ya
        coinciden con el estado actual devuelve outcome='unchanged' sin modificar nada.
        """
        with project_service() as service:
            return service.update_task(
                principal,
                task_id,
                changes,
                expected_version=expected_version,
                idempotency_key=idempotency_key,
            )

    return server
