"""Políticas de update_task: revocación entre escritura y replay, y updates sin cambios."""

from __future__ import annotations

import pytest

from arl.domain.errors import Forbidden, IdempotencyConflict, NotFound, VersionConflict
from arl.domain.models import TaskPatch, TaskStatus, UpdateOutcome
from arl.storage import ledger

KEY = "thread-9:call-1"
DONE = TaskPatch(status=TaskStatus.DONE)


def _version(conn, task_id: str = "apo-3") -> int:
    return conn.execute("SELECT version FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


# --- Revocación de permisos entre la escritura y el replay ------------------------------------


def test_revoked_membership_gets_not_found_instead_of_stored_response(service, as_user, conn):
    """Política: el replay exige visibilidad actual; no se devuelve la respuesta guardada."""
    service.update_task(as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    conn.execute("DELETE FROM memberships WHERE user_id = 'bob' AND project_id = 'apollo'")

    with pytest.raises(NotFound) as exc:
        service.update_task(as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    assert "done" not in str(exc.value)  # nada del estado guardado se filtra en el error
    assert str(exc.value) == str(NotFound("tarea", "apo-3"))  # igual que una tarea inexistente


def test_downgraded_role_still_gets_replay_of_completed_write(service, as_user, conn):
    """Política: perder solo el permiso de escritura no oculta el resultado de una escritura
    ya realizada; la respuesta contiene datos que el principal todavía puede leer."""
    first = service.update_task(
        as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
    )
    conn.execute(
        "UPDATE memberships SET role = 'viewer' WHERE user_id = 'bob' AND project_id = 'apollo'"
    )

    replay = service.update_task(
        as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
    )
    assert replay.replayed is True and replay.task == first.task
    with pytest.raises(Forbidden):  # una escritura nueva sí exige el permiso actual
        service.update_task(
            as_user("bob"),
            "apo-3",
            TaskPatch(status=TaskStatus.TODO),
            expected_version=2,
            idempotency_key="thread-9:call-2",
        )
    assert len(ledger.list_audit(conn)) == 1


# --- Updates sin cambios efectivos ---------------------------------------------------------------


def test_noop_returns_unchanged_without_new_version_or_audit(service, as_user, conn):
    result = service.update_task(
        as_user("alice"),
        "apo-1",
        DONE,
        expected_version=1,
        idempotency_key=KEY,  # ya está done
    )
    assert result.outcome is UpdateOutcome.UNCHANGED
    assert result.audit_id is None
    assert result.task.version == 1 and _version(conn, "apo-1") == 1
    assert ledger.list_audit(conn) == []


def test_noop_is_idempotent_even_after_later_writes(service, as_user, conn):
    alice = as_user("alice")
    first = service.update_task(alice, "apo-1", DONE, expected_version=1, idempotency_key=KEY)
    service.update_task(
        alice,
        "apo-1",
        TaskPatch(status=TaskStatus.TODO),
        expected_version=1,
        idempotency_key="thread-9:call-2",
    )
    retry = service.update_task(alice, "apo-1", DONE, expected_version=1, idempotency_key=KEY)
    assert retry.replayed is True
    assert retry.model_copy(update={"replayed": False}) == first
    with pytest.raises(IdempotencyConflict):
        service.update_task(
            alice,
            "apo-1",
            TaskPatch(status=TaskStatus.BLOCKED),
            expected_version=1,
            idempotency_key=KEY,
        )


def test_noop_still_requires_write_permission_and_current_version(service, as_user, conn):
    with pytest.raises(Forbidden):  # carol es viewer de apollo
        service.update_task(
            as_user("carol"), "apo-1", DONE, expected_version=1, idempotency_key=KEY
        )
    service.update_task(
        as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key="thread-9:call-3"
    )
    with pytest.raises(VersionConflict):  # mismo valor, pero versión desactualizada
        service.update_task(
            as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key="thread-9:call-4"
        )


def test_partial_noop_applies_and_audits_only_effective_changes(service, as_user, conn):
    result = service.update_task(
        as_user("alice"),
        "apo-2",  # in_progress, asignada a bob
        TaskPatch(status=TaskStatus.IN_PROGRESS, assignee_id="carol"),
        expected_version=1,
        idempotency_key=KEY,
    )
    assert result.outcome is UpdateOutcome.UPDATED
    assert (result.task.version, result.task.assignee_id) == (2, "carol")
    (audit,) = ledger.list_audit(conn, "apo-2")
    changed = {k for k in audit.after if audit.after[k] != audit.before[k]}
    assert changed == {"assignee_id", "version"}
