"""La política de autorización, verificada combinación por combinación."""

from __future__ import annotations

import itertools

import pytest

from arl.domain.models import Role
from arl.domain.policy import Action, is_allowed

EXPECTED = {
    (None, Action.READ_PROJECT): False,
    (None, Action.UPDATE_TASK): False,
    (Role.VIEWER, Action.READ_PROJECT): True,
    (Role.VIEWER, Action.UPDATE_TASK): False,
    (Role.EDITOR, Action.READ_PROJECT): True,
    (Role.EDITOR, Action.UPDATE_TASK): True,
    (Role.OWNER, Action.READ_PROJECT): True,
    (Role.OWNER, Action.UPDATE_TASK): True,
}


def test_matrix_covers_every_role_and_action() -> None:
    assert set(EXPECTED) == set(itertools.product([None, *Role], Action))


@pytest.mark.parametrize(("role", "action"), list(EXPECTED))
def test_is_allowed(role: Role | None, action: Action) -> None:
    assert is_allowed(role, action) is EXPECTED[(role, action)]
