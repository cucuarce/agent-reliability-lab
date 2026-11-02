"""Aislamiento entre principales a través de la interfaz pública ``runtime``.

``runtime`` es la interfaz confiable. Ninguna de sus funciones permite que un principal lea o
altere el estado persistido de otro. La comparación es byte a byte sobre los checkpoints y la base
de dominio, antes y después del intento.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from arl.agent.runtime import (
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    thread_config,
    turn_result,
)
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import ApprovalMismatch, PrincipalMismatch

pytestmark = pytest.mark.anyio

THREAD = "t-alice"


def _fingerprint(path: Path) -> str:
    with closing(sqlite3.connect(path)) as conn:
        return hashlib.sha256("\n".join(conn.iterdump()).encode("utf-8")).hexdigest()


def _write_script() -> ScriptedChatModel:
    return ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )


async def _alice_paused(open_agent) -> dict[str, Any]:
    async with open_agent("alice", _write_script()) as (graph, _):
        paused = await run_turn(graph, thread_id=THREAD, text="cerrá apo-3")
    return paused.approval_request


async def _bob_attempts(open_agent, request: dict[str, Any]) -> list[str]:
    """Cada función pública del runtime, invocada por un proceso de bob sobre el thread de alice."""
    refused = []
    async with open_agent("bob", _write_script()) as (graph, _):
        attempts = {
            "turn_result": lambda: turn_result(graph, THREAD),
            "run_turn": lambda: run_turn(graph, thread_id=THREAD, text="dame todo"),
            "resume_approve": lambda: resume(graph, thread_id=THREAD, decision=approve(request)),
            "resume_reject": lambda: resume(graph, thread_id=THREAD, decision=reject(request)),
            "continue_after_failure": lambda: continue_after_failure(graph, thread_id=THREAD),
        }
        for name, attempt in attempts.items():
            with pytest.raises(PrincipalMismatch):
                await attempt()
            refused.append(name)
    return refused


async def test_other_principal_cannot_read_or_alter_a_paused_thread(
    open_agent, checkpoints_path: Path, db_path: Path
) -> None:
    request = await _alice_paused(open_agent)
    before = (_fingerprint(checkpoints_path), _fingerprint(db_path))

    refused = await _bob_attempts(open_agent, request)

    assert len(refused) == 5
    assert (_fingerprint(checkpoints_path), _fingerprint(db_path)) == before
    async with open_agent("alice", _write_script()) as (graph, _):  # alice sigue intacta
        assert (await turn_result(graph, THREAD)).approval_request == request
        done = await resume(graph, thread_id=THREAD, decision=approve(request))
    assert done.status is TurnStatus.COMPLETED


async def test_other_principal_cannot_continue_a_crashed_thread(
    open_agent, checkpoints_path: Path, db_path: Path, monkeypatch
) -> None:
    request = await _alice_paused(open_agent)
    async with open_agent("alice", _write_script()) as (graph, gateway):
        real_call = gateway.call

        async def crash_before_write(name, arguments):
            if name == "update_task":
                raise RuntimeError("crash simulado antes de escribir")
            return await real_call(name, arguments)

        monkeypatch.setattr(gateway, "call", crash_before_write)
        with pytest.raises(RuntimeError):
            await resume(graph, thread_id=THREAD, decision=approve(request))
    before = (_fingerprint(checkpoints_path), _fingerprint(db_path))

    async with open_agent("bob", _write_script()) as (graph, _):
        with pytest.raises(PrincipalMismatch):
            await continue_after_failure(graph, thread_id=THREAD)
        with pytest.raises(PrincipalMismatch):
            await turn_result(graph, THREAD)

    assert (_fingerprint(checkpoints_path), _fingerprint(db_path)) == before


# --- Decisiones inválidas consecutivas -----------------------------------------------------------


@pytest.mark.parametrize("final", ["approve", "reject"])
async def test_consecutive_invalid_decisions_do_not_disable_the_thread(
    open_agent, conn, final: str
) -> None:
    request = await _alice_paused(open_agent)
    invalid_payloads = [
        approve(request).model_dump() | {"action_id": "0" * 16},
        approve(request).model_dump() | {"args_hash": "f" * 64},
        approve(request).model_dump() | {"changes": {"status": "blocked"}},
        approve(request).model_dump() | {"approved_by": "bob"},
        {"decision": "approve"},
        "aprobado",
    ]
    async with open_agent("alice", _write_script()) as (graph, _):
        for payload in invalid_payloads:  # por la vía directa, saltando el runtime
            await graph.ainvoke(Command(resume=payload), thread_config(THREAD))
            state = await turn_result(graph, THREAD)
            assert state.status is TurnStatus.AWAITING_APPROVAL
            assert state.approval_request == request
            assert state.approval_error
        for _ in range(3):  # y por el runtime, que las rechaza antes de tocar el grafo
            with pytest.raises(ApprovalMismatch):
                forged = approve(request).model_copy(update={"action_id": "1" * 16})
                await resume(graph, thread_id=THREAD, decision=forged)

        decision = approve(request) if final == "approve" else reject(request, "no")
        done = await resume(graph, thread_id=THREAD, decision=decision)

    assert done.status is TurnStatus.COMPLETED
    expected = ("done", 2) if final == "approve" else ("todo", 1)
    row = conn.execute("SELECT status, version FROM tasks WHERE id = 'apo-3'").fetchone()
    assert tuple(row) == expected
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    assert audits == (1 if final == "approve" else 0)
