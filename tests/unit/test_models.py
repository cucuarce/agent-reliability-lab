from __future__ import annotations

import pytest
from pydantic import ValidationError

from arl.domain.models import Principal, Task, TaskPatch, TaskStatus, UpdateOutcome, UpdateResult


def test_patch_only_reports_explicitly_sent_fields() -> None:
    assert TaskPatch(status=TaskStatus.DONE).changes() == {"status": "done"}
    assert TaskPatch(assignee_id=None).changes() == {"assignee_id": None}  # desasignar
    assert TaskPatch(title="  Nuevo título  ").changes() == {"title": "Nuevo título"}


@pytest.mark.parametrize(
    "payload",
    [
        {},  # sin cambios
        {"status": None},
        {"title": None},
        {"title": "   "},
        {"title": "x" * 201},
        {"status": "archived"},
        {"assignee_id": "Alice; DROP TABLE tasks"},
        # Identidad o campos de control enviados por el LLM: se rechazan, no se ignoran.
        {"status": "done", "user_id": "alice"},
        {"status": "done", "version": 99},
        {"status": "done", "project_id": "cassini"},
    ],
)
def test_patch_rejects_invalid_payloads(payload: dict) -> None:
    with pytest.raises(ValidationError):
        TaskPatch.model_validate(payload)


@pytest.mark.parametrize("user_id", ["", "Alice", "alice bob", "../alice", "a" * 65])
def test_principal_requires_canonical_identifier(user_id: str) -> None:
    with pytest.raises(ValidationError):
        Principal(user_id=user_id)


@pytest.mark.parametrize(
    ("outcome", "audit_id"), [(UpdateOutcome.UNCHANGED, 7), (UpdateOutcome.UPDATED, None)]
)
def test_update_result_requires_audit_only_when_updated(outcome, audit_id) -> None:
    task = Task(id="t-1", project_id="p", title="T", status=TaskStatus.TODO, version=1)
    with pytest.raises(ValidationError, match="audit_id"):
        UpdateResult(task=task, outcome=outcome, audit_id=audit_id)


def test_principal_is_immutable() -> None:
    principal = Principal(user_id="alice")
    with pytest.raises(ValidationError):
        principal.user_id = "bob"  # type: ignore[misc]
