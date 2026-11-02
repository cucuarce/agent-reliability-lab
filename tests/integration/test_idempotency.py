"""Idempotencia y límites transaccionales de update_task.

Cubre los criterios de la etapa: reintento sin duplicar, misma clave con otros parámetros ->
conflicto, reinicio sin segundo efecto, atomicidad ante fallas y concurrencia real.
"""

from __future__ import annotations

import threading
from contextlib import closing

import pytest
from pydantic import ValidationError

from arl.domain.errors import IdempotencyConflict, NotFound, VersionConflict
from arl.domain.models import TaskPatch, TaskStatus
from arl.services.project_service import ProjectService
from arl.storage import ledger
from arl.storage.db import connect

DONE = TaskPatch(status=TaskStatus.DONE)
KEY = "thread-1:call-1"


def _counts(conn) -> tuple[int, int, int]:
    """(versión de apo-3, filas de auditoría, filas del ledger)."""
    version = conn.execute("SELECT version FROM tasks WHERE id = 'apo-3'").fetchone()[0]
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    keys = conn.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
    return version, audits, keys


def test_retry_with_same_key_and_params_does_not_duplicate(service, as_user, conn) -> None:
    alice = as_user("alice")
    first = service.update_task(alice, "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    retry = service.update_task(alice, "apo-3", DONE, expected_version=1, idempotency_key=KEY)

    assert first.replayed is False and retry.replayed is True
    assert retry.model_copy(update={"replayed": False}) == first  # misma respuesta
    assert _counts(conn) == (2, 1, 1)


@pytest.mark.parametrize(
    ("patch", "expected_version", "task_id"),
    [
        (TaskPatch(status=TaskStatus.BLOCKED), 1, "apo-3"),  # otros cambios
        (DONE, 2, "apo-3"),  # otra versión esperada
        (DONE, 1, "apo-2"),  # otra tarea
    ],
)
def test_same_key_with_different_params_conflicts(
    service, as_user, conn, patch, expected_version, task_id
) -> None:
    alice = as_user("alice")
    service.update_task(alice, "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    with pytest.raises(IdempotencyConflict):
        service.update_task(
            alice, task_id, patch, expected_version=expected_version, idempotency_key=KEY
        )
    assert _counts(conn) == (2, 1, 1)


def test_restart_does_not_cause_a_second_effect(open_conn, as_user) -> None:
    """Simula un reinicio: el reintento llega por otra conexión y otra instancia del servicio."""
    before_restart = open_conn()
    ProjectService(before_restart).update_task(
        as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
    )
    before_restart.close()

    after_restart = open_conn()
    retry = ProjectService(after_restart).update_task(
        as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
    )
    assert retry.replayed is True
    assert _counts(after_restart) == (2, 1, 1)


@pytest.mark.parametrize("failing", ["append_audit", "record"])
def test_failure_mid_transaction_rolls_back_everything(
    service, as_user, conn, monkeypatch, failing: str
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("falla simulada")

    monkeypatch.setattr(ledger, failing, boom)
    with pytest.raises(RuntimeError, match="falla simulada"):
        service.update_task(
            as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
        )
    assert _counts(conn) == (1, 0, 0)  # ni la tarea, ni auditoría, ni ledger
    assert not conn.in_transaction

    monkeypatch.undo()
    result = service.update_task(
        as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=KEY
    )
    assert result.replayed is False
    assert _counts(conn) == (2, 1, 1)


def test_keys_are_scoped_per_principal(service, as_user, conn) -> None:
    """Otro principal con la misma clave no recibe la respuesta ajena: se evalúa desde cero."""
    service.update_task(as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    with pytest.raises(NotFound):
        service.update_task(as_user("dave"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    with pytest.raises(VersionConflict):  # bob es editor, pero la versión 1 ya no existe
        service.update_task(as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    assert _counts(conn) == (2, 1, 1)


def test_replay_requires_current_visibility(service, as_user, conn) -> None:
    """Si el principal perdió acceso, el reintento no devuelve la respuesta guardada."""
    service.update_task(as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)
    conn.execute("DELETE FROM memberships WHERE user_id = 'bob' AND project_id = 'apollo'")
    with pytest.raises(NotFound):
        service.update_task(as_user("bob"), "apo-3", DONE, expected_version=1, idempotency_key=KEY)


@pytest.mark.parametrize("key", ["", "short", "x" * 129, "con espacios 123", "ñandú-123"])
def test_invalid_keys_are_rejected_before_touching_the_db(service, as_user, conn, key) -> None:
    with pytest.raises(ValidationError):
        service.update_task(
            as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=key
        )
    assert _counts(conn) == (1, 0, 0)


def _run_concurrently(db_path, as_user, keys: list[str]) -> list[object]:
    barrier = threading.Barrier(len(keys))
    outcomes: list[object] = [None] * len(keys)

    def worker(index: int) -> None:
        # Una conexión propia por hilo, abierta y cerrada en el hilo: como procesos separados.
        with closing(connect(db_path)) as conn:
            service = ProjectService(conn)
            barrier.wait()
            try:
                outcomes[index] = service.update_task(
                    as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key=keys[index]
                )
            except Exception as exc:
                outcomes[index] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(keys))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return outcomes


def test_concurrent_retries_with_same_key_write_once(db_path, open_conn, as_user) -> None:
    outcomes = _run_concurrently(db_path, as_user, [KEY] * 8)
    assert not [o for o in outcomes if isinstance(o, Exception)]
    assert sum(not o.replayed for o in outcomes) == 1
    assert _counts(open_conn()) == (2, 1, 1)


def test_concurrent_writers_with_same_version_have_one_winner(db_path, open_conn, as_user) -> None:
    outcomes = _run_concurrently(db_path, as_user, [f"writer-key-{i}" for i in range(8)])
    winners = [o for o in outcomes if not isinstance(o, Exception)]
    losers = [o for o in outcomes if isinstance(o, Exception)]
    assert len(winners) == 1
    assert all(isinstance(o, VersionConflict) for o in losers)
    assert _counts(open_conn()) == (2, 1, 1)
