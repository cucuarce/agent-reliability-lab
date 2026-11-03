"""``RequestScope``: lo que la aplicación autoriza escribir en un turno (etapa 6A.1).

El alcance lo declara la aplicación (nunca el LLM ni una tool), el runtime lo liga al thread y al
principal y lo sella. Cada propuesta de escritura tiene un estado en el registro del turno; los
permisos reservados, consumidos, rechazados por el servidor o invalidados cuentan para los
límites, así que nunca se sobreasignan.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arl.agent.scope import (
    COUNTED_STATES,
    BoundScope,
    OutOfScope,
    RequestScope,
    ScopeMismatch,
    allow_update,
    settle_state,
)

THREAD, ALICE = "t-1", "alice"


def _bound(scope: RequestScope) -> BoundScope:
    return BoundScope.bind(scope, thread_id=THREAD, principal_id=ALICE)


def _check(scope: RequestScope, task_id: str, changes: dict, ledger: dict | None = None) -> str:
    return _bound(scope).check(task_id, changes, ledger or {})


# --- Coincidencia de recursos, campos y valores -------------------------------------------------


def test_read_only_scope_blocks_every_write() -> None:
    with pytest.raises(OutOfScope) as exc:
        _check(RequestScope.read_only(), "apo-3", {"status": "done"})
    assert exc.value.reason == "read_only"


def test_grant_allows_only_its_tasks_fields_and_values() -> None:
    scope = RequestScope.writes(allow_update("apo-3", status=["done", "blocked"]))
    assert _check(scope, "apo-3", {"status": "done"}).startswith("g-")
    for task, changes, reason in [
        ("apo-4", {"status": "done"}, "resource"),
        ("apo-3", {"title": "otro"}, "field"),
        ("apo-3", {"status": "done", "title": "otro"}, "field"),
        ("apo-3", {"status": "todo"}, "value"),
    ]:
        with pytest.raises(OutOfScope) as exc:
            _check(scope, task, changes)
        assert exc.value.reason == reason, (task, changes)


def test_any_value_of_a_field_can_be_allowed_explicitly() -> None:
    scope = RequestScope.writes(allow_update("apo-3", title=None))
    assert _check(scope, "apo-3", {"title": "cualquier título"})


def test_unassigning_is_a_value_that_must_be_allowed() -> None:
    scope = RequestScope.writes(allow_update("apo-3", assignee_id=["bob"]))
    with pytest.raises(OutOfScope):
        _check(scope, "apo-3", {"assignee_id": None})
    assert _check(RequestScope.writes(allow_update("apo-3", assignee_id=[None])), "apo-3",
                  {"assignee_id": None})  # fmt: skip


def test_out_of_scope_reasons_never_echo_values() -> None:
    scope = RequestScope.writes(allow_update("apo-3", status=["done"]))
    with pytest.raises(OutOfScope) as exc:
        _check(scope, "apo-3", {"status": "blocked"})
    assert "blocked" not in str(exc.value)


# --- Límites: reservas + consumos nunca superan max_uses ni max_writes ---------------------------


def test_reserved_consumed_refused_and_invalidated_count_released_does_not() -> None:
    scope = RequestScope.writes(allow_update("apo-3", status=["done"]))
    grant = _check(scope, "apo-3", {"status": "done"})
    for state in ("reserved", "consumed", "refused", "invalidated"):
        assert state in COUNTED_STATES
        with pytest.raises(OutOfScope) as exc:
            _check(scope, "apo-3", {"status": "done"}, {"a1": {"grant_id": grant, "state": state}})
        assert exc.value.reason == "exhausted"
    released = {"a1": {"grant_id": grant, "state": "released"}}
    assert _check(scope, "apo-3", {"status": "done"}, released) == grant


def test_max_writes_caps_the_turn_across_grants() -> None:
    scope = RequestScope.writes(
        allow_update("apo-3", status=["done"]), allow_update("apo-2", status=["done"]),
        max_writes=1,
    )  # fmt: skip
    grant = _check(scope, "apo-3", {"status": "done"})
    with pytest.raises(OutOfScope) as exc:
        _check(scope, "apo-2", {"status": "done"}, {"a1": {"grant_id": grant, "state": "consumed"}})
    assert exc.value.reason == "max_writes"


@pytest.mark.parametrize("state", ["reserved", "consumed", "refused", "invalidated"])
def test_max_writes_counts_open_reservations_and_closed_operations(state) -> None:
    scope = RequestScope.writes(
        allow_update("apo-3", status=["done"]), allow_update("apo-2", status=["done"]),
        max_writes=1,
    )  # fmt: skip
    grant = _check(scope, "apo-3", {"status": "done"})
    with pytest.raises(OutOfScope) as exc:
        _check(scope, "apo-2", {"status": "done"}, {"a1": {"grant_id": grant, "state": state}})
    assert exc.value.reason == "max_writes"
    released = {"a1": {"grant_id": grant, "state": "released"}}
    assert _check(scope, "apo-2", {"status": "done"}, released) != grant  # liberada: no cuenta


def test_max_uses_allows_several_writes_with_one_grant() -> None:
    scope = RequestScope.writes(allow_update("apo-3", "apo-2", status=["done"], max_uses=2))
    grant = _check(scope, "apo-3", {"status": "done"})
    ledger = {"a1": {"grant_id": grant, "state": "consumed"}}
    assert _check(scope, "apo-2", {"status": "done"}, ledger) == grant
    ledger["a2"] = {"grant_id": grant, "state": "reserved"}
    with pytest.raises(OutOfScope):
        _check(scope, "apo-3", {"status": "done"}, ledger)


@pytest.mark.parametrize(
    ("outcome", "state"),
    [
        ("updated", "consumed"), ("unchanged", "consumed"),
        ("forbidden", "refused"), ("not_found", "refused"), ("invalid_change", "refused"),
        ("idempotency_conflict", "refused"), (None, "refused"),  # código desconocido: cerrar
        ("version_conflict", "invalidated"),
    ],
)  # fmt: skip
def test_server_outcomes_settle_the_reservation(outcome, state) -> None:
    assert settle_state(outcome) == state


# --- Integridad y vínculo con el thread ----------------------------------------------------------


def test_bound_scope_is_sealed_and_detects_tampering() -> None:
    bound = _bound(RequestScope.writes(allow_update("apo-3", status=["done"])))
    data = bound.model_dump()
    assert BoundScope.model_validate(data) == bound
    data["scope"]["grants"][0]["task_ids"] = ["apo-4"]
    with pytest.raises(ValidationError, match="alterado"):
        BoundScope.model_validate(data)


def test_seal_is_not_authentication_a_checkpoint_writer_can_recompute_it() -> None:
    """Límite documentado: sin secreto, quien pueda escribir el checkpoint recalcula el sha256 y
    el alcance falsificado pasa la verificación. El sello solo detecta cambios sin recalcular."""
    narrow = _bound(RequestScope.writes(allow_update("apo-3", status=["done"])))
    wide = _bound(RequestScope.writes(allow_update("apo-3", "apo-4", status=None, max_uses=9)))
    forged = narrow.model_dump() | {"scope": wide.scope.model_dump(), "scope_hash": wide.scope_hash}
    assert BoundScope.model_validate(forged).scope == wide.scope


def test_bound_scope_belongs_to_one_thread_and_principal() -> None:
    bound = _bound(RequestScope.read_only())
    bound.verify(thread_id=THREAD, principal_id=ALICE)
    with pytest.raises(ScopeMismatch):
        bound.verify(thread_id="otro", principal_id=ALICE)
    with pytest.raises(ScopeMismatch):
        bound.verify(thread_id=THREAD, principal_id="bob")


def test_declarations_are_strict_and_immutable() -> None:
    with pytest.raises(ValidationError):
        allow_update("apo-3", prioridad=["alta"])  # campo que update_task no tiene
    with pytest.raises(ValidationError):
        allow_update()  # sin recursos
    with pytest.raises(ValidationError):
        allow_update("apo-3")  # sin campos
    with pytest.raises(ValidationError):
        RequestScope.model_validate({"grants": [], "extra": 1})
    scope = RequestScope.read_only()
    with pytest.raises(ValidationError):
        scope.max_writes = 5  # type: ignore[misc]


def test_unrestricted_scope_is_explicitly_legacy() -> None:
    bound = _bound(RequestScope.unrestricted_legacy_v1())
    assert bound.scope.kind == "unrestricted_legacy_v1"
    assert bound.check("cas-2", {"title": "x"}, {}) == "unrestricted"
