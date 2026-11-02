"""Casos de uso sobre proyectos y tareas: único punto de acceso al dominio.

Cada operación recibe un :class:`Principal` construido por código de confianza; ningún método
acepta un ``user_id`` como parámetro de negocio. La autorización, el control de versión, el ledger
de idempotencia y la auditoría se resuelven dentro de una única transacción SQLite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import TypeAdapter

from arl.domain.errors import (
    Forbidden,
    IdempotencyConflict,
    InvalidChange,
    NotFound,
    VersionConflict,
)
from arl.domain.models import (
    IdempotencyKey,
    Principal,
    ProjectDetail,
    ProjectSummary,
    TaskPatch,
    UpdateOutcome,
    UpdateResult,
)
from arl.domain.policy import Action, is_allowed
from arl.storage import ledger, repositories
from arl.storage.db import transaction

UPDATE_TASK_OPERATION = "update_task"
_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(IdempotencyKey)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_principal(principal: object) -> Principal:
    if not isinstance(principal, Principal):
        raise TypeError("se requiere un Principal construido por código de confianza")
    return principal


class ProjectService:
    def __init__(
        self, conn: sqlite3.Connection, *, clock: Callable[[], datetime] = _utc_now
    ) -> None:
        self._conn = conn
        self._clock = clock

    def list_projects(self, principal: Principal) -> list[ProjectSummary]:
        """Solo los proyectos donde el principal es miembro."""
        user_id = _require_principal(principal).user_id
        with transaction(self._conn, write=False) as conn:
            return repositories.list_projects_for_user(conn, user_id)

    def get_project(self, principal: Principal, project_id: str) -> ProjectDetail:
        """Proyecto con sus tareas. Inexistente y ajeno producen el mismo ``NotFound``."""
        user_id = _require_principal(principal).user_id
        with transaction(self._conn, write=False) as conn:
            role = repositories.get_role(conn, user_id, project_id)
            if role is None or not is_allowed(role, Action.READ_PROJECT):
                raise NotFound("proyecto", project_id)
            project = repositories.get_project(conn, project_id)
            if project is None:  # imposible con foreign keys; no se confía en eso
                raise NotFound("proyecto", project_id)
            tasks = repositories.list_tasks(conn, project_id)
            return ProjectDetail(project=project, my_role=role, tasks=tuple(tasks))

    def update_task(
        self,
        principal: Principal,
        task_id: str,
        patch: TaskPatch,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> UpdateResult:
        """Modifica una tarea de forma autorizada, versionada, idempotente y auditada.

        Orden de verificación (todo dentro de ``BEGIN IMMEDIATE``):

        1. visibilidad: tarea inexistente o de un proyecto ajeno -> ``NotFound``;
        2. ledger: misma clave y mismos parámetros -> se devuelve la respuesta original
           (``replayed=True``) sin escribir; misma clave con otros parámetros ->
           ``IdempotencyConflict``;
        3. rol: sin permiso de escritura -> ``Forbidden``;
        4. reglas del cambio (el assignee debe ser miembro del proyecto) -> ``InvalidChange``;
        5. versión: ``expected_version`` desactualizada -> ``VersionConflict``;
        6. cambios efectivos: si el patch coincide con el estado actual, ``UNCHANGED`` (sin
           versión nueva ni auditoría); si no, UPDATE de los campos que cambian + auditoría;
        7. ledger con la respuesta, confirmado en la misma transacción.

        Política de revocación: el replay exige visibilidad **actual** del proyecto (paso 1 antes
        del 2). Quien perdió la membresía recibe ``NotFound`` y nunca la respuesta guardada. Quien
        solo perdió el permiso de escritura (p. ej. editor -> viewer) sí recibe el replay: la
        escritura ya ocurrió con permiso y la respuesta solo contiene datos que todavía puede leer.

        Los errores no se registran en el ledger: reintentar una operación fallida la vuelve a
        evaluar contra el estado actual. ``UNCHANGED`` sí se registra, para que un reintento
        devuelva la misma respuesta aunque la tarea haya cambiado después.
        """
        user_id = _require_principal(principal).user_id
        key = _KEY_ADAPTER.validate_python(idempotency_key)
        changes = patch.changes()
        fingerprint = ledger.request_hash(
            UPDATE_TASK_OPERATION,
            {"task_id": task_id, "changes": changes, "expected_version": expected_version},
        )

        with transaction(self._conn, write=True) as conn:
            task = repositories.get_task(conn, task_id)
            role = repositories.get_role(conn, user_id, task.project_id) if task else None
            if task is None or not is_allowed(role, Action.READ_PROJECT):
                raise NotFound("tarea", task_id)

            previous = ledger.find(conn, user_id, key)
            if previous is not None:
                if (previous.operation, previous.request_hash) != (
                    UPDATE_TASK_OPERATION,
                    fingerprint,
                ):
                    raise IdempotencyConflict(key)
                original = UpdateResult.model_validate_json(previous.response_json)
                return original.model_copy(update={"replayed": True})

            if not is_allowed(role, Action.UPDATE_TASK):
                raise Forbidden(f"el rol '{role}' no permite modificar tareas de este proyecto")

            assignee = changes.get("assignee_id")
            if (
                assignee is not None
                and repositories.get_role(conn, assignee, task.project_id) is None
            ):
                raise InvalidChange(f"'{assignee}' no es miembro del proyecto")

            if task.version != expected_version:
                raise VersionConflict(task_id, expected_version, task.version)

            now = self._clock().isoformat()
            current = task.model_dump(mode="json")
            effective = {
                field: value for field, value in changes.items() if current[field] != value
            }
            if not effective:
                result = UpdateResult(task=task, outcome=UpdateOutcome.UNCHANGED, audit_id=None)
            else:
                if not repositories.update_task(conn, task_id, effective, expected_version):
                    raise VersionConflict(task_id, expected_version, task.version)
                updated = repositories.get_task(conn, task_id)
                if updated is None:
                    raise RuntimeError(f"la tarea '{task_id}' desapareció dentro de la transacción")
                audit_id = ledger.append_audit(
                    conn,
                    occurred_at=now,
                    actor_id=user_id,
                    action=UPDATE_TASK_OPERATION,
                    project_id=task.project_id,
                    task_id=task_id,
                    idempotency_key=key,
                    before=current,
                    after=updated.model_dump(mode="json"),
                )
                result = UpdateResult(
                    task=updated, outcome=UpdateOutcome.UPDATED, audit_id=audit_id
                )

            ledger.record(
                conn,
                principal_id=user_id,
                key=key,
                operation=UPDATE_TASK_OPERATION,
                request_hash=fingerprint,
                response_json=result.model_dump_json(),
                created_at=now,
            )
            return result
