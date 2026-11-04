"""Ejecuta un escenario real en un entorno propio y registra observaciones, sin juzgarlas.

Reutiliza el entorno del runner determinístico: directorio temporal nuevo, ``app.db`` sintética
sembrada, checkpoints, locks y trazas propios, y el servidor MCP y el agente reales. Lo único
distinto es el modelo (``LiveModel``) y que el humano simulado decide según el caso.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arl.agent.runtime import TurnStatus, approve, reject, resume, run_turn
from arl.agent.state import Limits
from arl.evals.live.cases import LiveCase
from arl.evals.live.model import BudgetExceeded, LiveModel
from arl.evals.live.providers import ProviderError
from arl.evals.runner import (
    MAX_OPERATIONS,
    InfraError,
    Observation,
    _apply_concurrent,
    _audit,
    _collect_conversation,
    _Env,
    _ledger,
    _memberships,
    _open_agent,
    _record_request,
    _tasks,
    seed_for,
)
from arl.storage.db import connect
from arl.storage.seed import create_database
from arl.tracing.query import TraceNotFound, report_for


class WritesDisabled(RuntimeError):
    """6B.1: un modelo real no ejecuta escenarios de escritura."""


@dataclass
class LiveObservation(Observation):
    reply: str = ""
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_failure: dict[str, Any] | None = None
    budget_stop: str | None = None
    infra_error: str | None = None


async def run_live_case(
    case: LiveCase, workdir: Path, model: LiveModel, limits: Limits, *, real_model: bool
) -> LiveObservation:
    if real_model and not case.writes_allowed_with_real_model and case.phase != 1:
        raise WritesDisabled(f"{case.id}: escrituras deshabilitadas con un modelo real")
    env = _Env(workdir)
    obs = LiveObservation(case_id=case.id, thread_id=f"live-{case.id}")
    obs.scope_mode = case.scope.kind
    try:
        create_database(env.db, seed_for(case.seed)).close()
        with closing(connect(env.db)) as conn:
            obs.tasks_before = _tasks(conn)
            obs.memberships = _memberships(conn)
        async with AsyncExitStack() as stack:
            try:
                graph, faulty = await _open_agent(stack, env, case.principal, model, {}, limits)
            except Exception as exc:
                raise InfraError(f"agente: {type(exc).__name__}") from None
            await _flow(case, env, graph, obs)
            await _collect_conversation(graph, obs.thread_id, obs)
            obs.deliveries = [
                {"tool": name, "task_id": args.get("task_id"), "key": args.get("idempotency_key")}
                for name, args in faulty.delivered
            ]
            obs.attempts = len(faulty.attempts)
        with closing(connect(env.db)) as conn:
            obs.tasks_after = _tasks(conn)
            obs.audit = _audit(conn)
            obs.ledger = _ledger(conn)
    except InfraError as exc:
        obs.infra_error = str(exc)
    obs.model_calls = [r.to_dict() for r in model.records]
    try:
        obs.trace = report_for(sorted(env.traces.glob("thread-*/*.jsonl")))
    except TraceNotFound as exc:
        obs.trace_error = type(exc).__name__
    return obs


async def _flow(case: LiveCase, env: _Env, graph, obs: LiveObservation) -> None:
    decisions = 0
    try:
        result = await run_turn(graph, thread_id=obs.thread_id, text=case.request, scope=case.scope)
        for _ in range(MAX_OPERATIONS):
            obs.statuses.append(result.status.value)
            obs.reply = result.reply or ""
            if result.status is not TurnStatus.AWAITING_APPROVAL:
                break
            request = result.approval_request or {}
            _apply_concurrent(case, env, decisions, obs)  # type: ignore[arg-type]
            await _record_request(graph, result, case.approval, obs)
            choice = approve(request) if case.approval == "approve" else reject(request, "eval")
            result = await resume(graph, thread_id=obs.thread_id, decision=choice)
            decisions += 1
    except InfraError:
        raise
    except ProviderError as exc:
        f = exc.failure
        obs.provider_failure = {
            "code": f.code,
            "status": f.status,
            "origin": f.origin,
            "aborts_suite": f.aborts_suite,
        }
    except BudgetExceeded as exc:
        obs.budget_stop = exc.reason
    except Exception as exc:  # resultado del sistema evaluado, no del entorno
        obs.raised = type(exc).__name__
    obs.final_status = obs.statuses[-1] if obs.statuses else None
