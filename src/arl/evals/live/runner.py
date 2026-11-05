"""Ejecuta un escenario real en un entorno propio y registra observaciones, sin juzgarlas.

Reutiliza el entorno del runner determinístico: directorio temporal nuevo, ``app.db`` sintética
sembrada, checkpoints, locks y trazas propios, y el servidor MCP y el agente reales. Lo único
distinto es el modelo (``LiveModel``) y que el humano simulado decide según el caso.

Con un modelo real, una escritura se autoriza al empezar el escenario y otra vez justo antes de
despachar cada decisión, aprobar o rechazar (``profiles.authorize_real_write``). El humano
simulado nunca aprueba una acción que el alcance del caso no cubre, y el directorio de trabajo
tiene que ser uno temporal, nuevo y vacío.

Para la transcripción (``transcript``) guarda la conversación visible del thread y comprueba que
la base tenga exactamente las tareas y los usuarios de la semilla empaquetada
(``synthetic_verified``): si no, la suite no escribe el contenido del escenario.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arl.agent.faults import FaultPlan
from arl.agent.runtime import (
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    thread_config,
)
from arl.config import ConfigError, _is_synced_folder
from arl.evals.live import transcript
from arl.evals.live.cases import LiveCase
from arl.evals.live.graders import _in_scope
from arl.evals.live.model import BudgetExceeded, LiveModel
from arl.evals.live.profiles import (
    LiveConfig,
    WritesDisabled,
    approval_policy,
    authorize_real_write,
)
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
from arl.evals.suite import _scratch_root
from arl.storage.db import connect
from arl.storage.seed import create_database, load_seed
from arl.tracing.query import TraceNotFound, report_for

__all__ = ["IsolationError", "LiveObservation", "WritesDisabled", "run_live_case"]

MAX_RECOVERIES = 1  # escrituras: una recuperación de un fallo incierto, con la misma clave


class IsolationError(ConfigError):
    """El escenario no corre en un directorio temporal nuevo: nunca sobre una base persistente."""


@dataclass
class LiveObservation(Observation):
    reply: str = ""
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_failure: dict[str, Any] | None = None
    budget_stop: str | None = None
    infra_error: str | None = None
    approval_out_of_scope: int = 0  # pedidos de aprobación que el alcance del caso no cubre
    conversation: list[dict[str, Any]] = field(default_factory=list)  # visible (``transcript``)
    synthetic_verified: bool = False  # la base tiene exactamente la semilla empaquetada
    # 6B.6: ``reply`` es el valor histórico (``TurnResult.last_ai_text``), el que leen los
    # graders históricos; ``model_reply`` solo texto del LLM; ``user_reply`` lo que ve el
    # usuario (``TurnResult.reply``) y ``reply_source`` quién lo escribió.
    model_reply: str = ""
    user_reply: str = ""
    reply_source: str = "model"
    write_outcomes: list[dict[str, Any]] = field(default_factory=list)


def _synthetic(obs: LiveObservation) -> bool:
    """Las tareas y los usuarios de la base son exactamente los de la semilla sintética."""
    data = load_seed()
    users = {user for members in obs.memberships.values() for user in members}
    return set(obs.tasks_before) == {t.id for t in data.tasks} and users <= {
        u.id for u in data.users
    }


def _check_isolated(workdir: Path) -> None:
    """Solo un directorio vacío dentro del temporal del sistema (que nunca está en OneDrive)."""
    resolved = workdir.resolve()
    if not resolved.is_relative_to(_scratch_root()) or _is_synced_folder(resolved):
        raise IsolationError("el escenario solo corre en el directorio temporal del sistema")
    if not resolved.is_dir() or any(resolved.iterdir()):
        raise IsolationError("el escenario necesita un directorio temporal nuevo y vacío")


async def run_live_case(
    case: LiveCase,
    workdir: Path,
    model: LiveModel,
    config: LiveConfig,
    *,
    real_model: bool,
    faults: FaultPlan | None = None,  # solo tests: fallos del MCP (reintento y recuperación)
) -> LiveObservation:
    if real_model:
        authorize_real_write(case, config)
    policy = approval_policy(case, config, real=real_model)
    _check_isolated(workdir)
    env = _Env(workdir)
    obs = LiveObservation(case_id=case.id, thread_id=f"live-{case.id}")
    revoked: WritesDisabled | None = None
    obs.scope_mode = case.scope.kind
    try:
        create_database(env.db, seed_for(case.seed)).close()
        with closing(connect(env.db)) as conn:
            obs.tasks_before = _tasks(conn)
            obs.memberships = _memberships(conn)
        obs.synthetic_verified = _synthetic(obs)
        async with AsyncExitStack() as stack:
            try:
                graph, faulty = await _open_agent(
                    stack, env, case.principal, model, dict(faults or {}), config.graph_limits()
                )
            except Exception as exc:
                raise InfraError(f"agente: {type(exc).__name__}") from None
            try:
                await _flow(case, env, graph, faulty, obs, config, policy, real_model)
            except WritesDisabled as exc:  # se cierra el agente y después se informa
                revoked = exc
            await _collect_conversation(graph, obs.thread_id, obs)
            state = await graph.aget_state(thread_config(obs.thread_id))
            obs.conversation = transcript.conversation(state.values.get("messages", []))
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
    if revoked is not None:
        raise revoked
    obs.model_calls = [r.to_dict() for r in model.records]
    try:
        obs.trace = report_for(sorted(env.traces.glob("thread-*/*.jsonl")))
    except TraceNotFound as exc:
        obs.trace_error = type(exc).__name__
    return obs


def _simulated_decision(
    case: LiveCase, policy: dict[str, Any] | None, request: dict[str, Any], obs: LiveObservation
) -> str:
    """La decisión fija del caso, salvo que la acción no esté en su alcance: la aprobación
    simulada nunca amplía ``RequestScope`` (si llega una así, es un bug del agente)."""
    if not _in_scope(case, request.get("task_id"), request.get("changes")):
        obs.approval_out_of_scope += 1
        return "reject"
    return policy["decision"] if policy else "reject"


async def _flow(
    case: LiveCase,
    env: _Env,
    graph,
    faulty,
    obs: LiveObservation,
    config: LiveConfig,
    policy: dict[str, Any] | None,
    real_model: bool,
) -> None:
    decisions = 0
    recoveries = 0
    try:
        result = await run_turn(graph, thread_id=obs.thread_id, text=case.request, scope=case.scope)
        for _ in range(MAX_OPERATIONS):
            obs.statuses.append(result.status.value)
            obs.reply = result.last_ai_text or ""
            obs.model_reply = result.model_reply
            obs.user_reply = result.user_reply
            obs.reply_source = result.reply_source
            obs.write_outcomes = list(result.write_outcomes)
            if result.status is TurnStatus.AWAITING_APPROVAL:
                request = result.approval_request or {}
                _apply_concurrent(case, env, decisions, obs)  # type: ignore[arg-type]
                decision = _simulated_decision(case, policy, request, obs)
                if real_model:  # otra vez, antes de despachar la decisión (aprobar o rechazar)
                    authorize_real_write(case, config)
                await _record_request(graph, result, decision, obs)
                choice = approve(request) if decision == "approve" else reject(request, "eval")
                result = await resume(graph, thread_id=obs.thread_id, decision=choice)
                decisions += 1
            elif (
                case.phase == 2
                and result.status in (TurnStatus.INTERRUPTED, TurnStatus.WRITE_UNCERTAIN)
                and recoveries < MAX_RECOVERIES
            ):
                faulty.clear()
                recoveries += 1
                result = await continue_after_failure(graph, thread_id=obs.thread_id)
            else:
                break
    except (InfraError, WritesDisabled):
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
