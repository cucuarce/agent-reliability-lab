from __future__ import annotations

import pytest
from pydantic import ValidationError

from arl.agent.state import ApprovalDecision, PendingWrite, _keep_principal, write_identity
from arl.agent.tool_registry import REGISTRY, InvalidToolCall, ToolKind, validate_call


def _pending(**overrides) -> PendingWrite:
    fields = {
        "action_id": "a1",
        "tool_call_id": "call_0_0",
        "tool": "update_task",
        "task_id": "apo-3",
        "project_id": "apollo",
        "my_role": "owner",
        "changes": {"status": "done"},
        "before": {"status": "todo"},
        "expected_version": 1,
        "idempotency_key": "arl-key-0001",
    } | overrides
    return PendingWrite.seal(**fields)


def test_registry_classifies_reads_and_writes_deterministically() -> None:
    kinds = {name: spec.kind for name, spec in REGISTRY.items()}
    assert kinds == {
        "list_projects": ToolKind.READ,
        "get_project": ToolKind.READ,
        "update_task": ToolKind.WRITE,
    }


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("delete_project", {}, "tool desconocida"),
        ("get_project", {}, "project_id (missing)"),
        ("get_project", {"project_id": "apollo", "user_id": "alice"}, "user_id (extra_forbidden)"),
        ("update_task", {"task_id": "apo-3", "changes": {}}, "changes (value_error)"),
    ],
)
def test_invalid_calls_are_described_without_echoing_values(name, args, fragment) -> None:
    with pytest.raises(InvalidToolCall) as exc:
        validate_call(name, args)
    assert fragment in str(exc.value)
    assert "alice" not in str(exc.value)


def test_write_identity_is_deterministic_and_valid_as_idempotency_key() -> None:
    first = write_identity("thread-1", "call_0_0")
    assert first == write_identity("thread-1", "call_0_0")
    assert first != write_identity("thread-2", "call_0_0")
    action_id, key = first
    assert len(action_id) == 16 and key.startswith("arl-") and len(key) == 44


def test_pending_write_seal_detects_any_tampering() -> None:
    pending = _pending()
    PendingWrite.model_validate(pending.model_dump())  # íntegra
    for field, value in [
        ("changes", {"status": "blocked"}),
        ("expected_version", 2),
        ("task_id", "apo-4"),
        ("before", {"status": "done"}),
        ("idempotency_key", "arl-otra-clave"),
    ]:
        with pytest.raises(ValidationError, match="alterada"):
            PendingWrite.model_validate(pending.model_dump() | {field: value})


def test_approval_decision_only_decides() -> None:
    pending = _pending()
    decision = ApprovalDecision(
        action_id=pending.action_id, args_hash=pending.args_hash, decision="approve"
    )
    assert decision.matches(pending)
    assert not decision.matches(_pending(changes={"status": "blocked"}))
    with pytest.raises(ValidationError):
        ApprovalDecision.model_validate(decision.model_dump() | {"expected_version": 5})


def test_principal_of_a_thread_cannot_change() -> None:
    assert _keep_principal(None, "alice") == "alice"
    assert _keep_principal("alice", "alice") == "alice"
    assert _keep_principal("alice", None) == "alice"
    assert _keep_principal("alice", "bob") == "alice"  # se ignora, sin excepción (ver docstring)
