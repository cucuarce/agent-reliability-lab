"""Autorización y no-filtración de información en ProjectService."""

from __future__ import annotations

import pytest

from arl.domain.errors import Forbidden, InvalidChange, NotFound, VersionConflict
from arl.domain.models import Role, TaskPatch, TaskStatus
from arl.services.project_service import ProjectService
from arl.storage import ledger

DONE = TaskPatch(status=TaskStatus.DONE)


def test_list_projects_only_returns_memberships(service: ProjectService, as_user) -> None:
    summaries = service.list_projects(as_user("alice"))
    assert [(p.id, p.my_role) for p in summaries] == [
        ("apollo", Role.OWNER),
        ("cassini", Role.VIEWER),
    ]
    assert service.list_projects(as_user("erin")) == []
    assert service.list_projects(as_user("ghost")) == []  # principal inexistente: sin acceso


def test_get_project_returns_tasks_and_role(service: ProjectService, as_user) -> None:
    detail = service.get_project(as_user("carol"), "apollo")
    assert detail.my_role is Role.VIEWER
    assert [t.id for t in detail.tasks] == ["apo-1", "apo-2", "apo-3", "apo-4"]


def test_foreign_and_missing_projects_are_indistinguishable(
    service: ProjectService, as_user
) -> None:
    with pytest.raises(NotFound) as foreign:
        service.get_project(as_user("dave"), "apollo")
    with pytest.raises(NotFound) as missing:
        service.get_project(as_user("dave"), "zeus")
    assert type(foreign.value) is type(missing.value)
    assert str(foreign.value).replace("apollo", "X") == str(missing.value).replace("zeus", "X")


def test_update_on_foreign_task_is_not_found_not_forbidden(
    service: ProjectService, as_user, conn
) -> None:
    for task_id in ("apo-3", "nope-1"):
        with pytest.raises(NotFound):
            service.update_task(
                as_user("dave"), task_id, DONE, expected_version=1, idempotency_key="key-dave-1"
            )
    assert ledger.list_audit(conn) == []


def test_viewer_cannot_update(service: ProjectService, as_user, conn) -> None:
    with pytest.raises(Forbidden):
        service.update_task(
            as_user("carol"), "apo-3", DONE, expected_version=1, idempotency_key="key-carol-1"
        )
    assert service.get_project(as_user("carol"), "apollo").tasks[2].version == 1
    assert ledger.list_audit(conn) == []


@pytest.mark.parametrize("user_id", ["alice", "bob"])  # owner y editor
def test_owner_and_editor_can_update(service: ProjectService, as_user, user_id: str) -> None:
    result = service.update_task(
        as_user(user_id), "apo-3", DONE, expected_version=1, idempotency_key=f"key-{user_id}-1"
    )
    assert (result.task.status, result.task.version, result.replayed) == ("done", 2, False)


def test_assignee_must_be_project_member(service: ProjectService, as_user) -> None:
    with pytest.raises(InvalidChange):
        service.update_task(
            as_user("alice"),
            "apo-3",
            TaskPatch(assignee_id="dave"),
            expected_version=1,
            idempotency_key="key-assign-1",
        )
    result = service.update_task(
        as_user("alice"),
        "apo-3",
        TaskPatch(assignee_id="carol"),
        expected_version=1,
        idempotency_key="key-assign-2",
    )
    assert result.task.assignee_id == "carol"


def test_stale_version_is_rejected(service: ProjectService, as_user, conn) -> None:
    service.update_task(
        as_user("alice"), "apo-3", DONE, expected_version=1, idempotency_key="key-first-1"
    )
    with pytest.raises(VersionConflict) as exc:
        service.update_task(
            as_user("bob"),
            "apo-3",
            TaskPatch(status=TaskStatus.BLOCKED),
            expected_version=1,
            idempotency_key="key-stale-1",
        )
    assert (exc.value.expected, exc.value.current) == (1, 2)
    assert len(ledger.list_audit(conn, "apo-3")) == 1


def test_service_rejects_raw_user_ids(service: ProjectService) -> None:
    with pytest.raises(TypeError, match="Principal"):
        service.list_projects("alice")  # type: ignore[arg-type]
