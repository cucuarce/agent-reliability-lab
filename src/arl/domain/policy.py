"""Política de autorización: función pura de (rol, acción), sin I/O ni intervención del LLM."""

from __future__ import annotations

from enum import StrEnum

from arl.domain.models import Role


class Action(StrEnum):
    READ_PROJECT = "read_project"
    UPDATE_TASK = "update_task"


_ALLOWED_ROLES: dict[Action, frozenset[Role]] = {
    Action.READ_PROJECT: frozenset({Role.VIEWER, Role.EDITOR, Role.OWNER}),
    Action.UPDATE_TASK: frozenset({Role.EDITOR, Role.OWNER}),
}


def is_allowed(role: Role | None, action: Action) -> bool:
    """``role=None`` significa "no es miembro del proyecto": nunca está permitido."""
    return role is not None and role in _ALLOWED_ROLES[action]
