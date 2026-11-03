"""Ejecuta un caso en un entorno propio y registra **observaciones**, sin juzgarlas.

Aislamiento: cada caso usa un directorio temporal nuevo con su ``app.db`` sembrada, sus
checkpoints, locks y trazas. Nunca lee ``ARL_DATA_DIR`` ni toca bases de desarrollo. El agente y
el servidor MCP son los reales (transporte en memoria); solo el LLM y el humano son simulados.

Separación de responsabilidades: el runner reúne hechos (estado final, acciones selladas,
decisiones, llamadas que llegaron al servidor, ``audit_log``, ledger, tareas, trazas). Los
graders (``arl.evals.graders``) los comparan con las expectativas. El runner nunca lee
``case.expected``: así no puede acomodar la ejecución al resultado esperado.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import AsyncExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client

from arl.agent.faults import EVERY_CALL, FaultyClient
from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.runtime import (
    TurnResult,
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    thread_config,
)
from arl.agent.state import Limits
from arl.domain.models import Principal, TaskPatch
from arl.evals.behavior import CaseModel
from arl.evals.cases import EvalCase, SeedSpec
from arl.mcp_server.server import create_server
from arl.services.project_service import ProjectService
from arl.storage import ledger
from arl.storage.db import connect
from arl.storage.seed import SeedData, create_database, load_seed
from arl.tracing.query import TraceNotFound, report_for

# Texto de un ToolMessage de error: "ERROR [codigo] ..." si lo generó el agente, o
# "ERROR Error executing tool <tool>: [codigo] ..." si viene del servidor (el SDK MCP antepone ese
# prefijo a los ToolError). El origen importa: solo los del servidor pasan por el gateway.
ERROR_TEXT = re.compile(r"^ERROR (?P<server>Error executing tool [a-z_]+: )?\[(?P<code>[a-z_]+)\]")
MAX_OPERATIONS = 12  # run_turn + decisiones + recuperaciones: corta guiones mal diseñados
EVAL_LIMITS = Limits(retry_initial_interval=0.0, retry_max_interval=0.0, retry_jitter=False)
MODEL_KIND = "simulated:CaseModel(ScriptedChatModel)"


class InfraError(RuntimeError):
    """El entorno del caso no se pudo preparar: no es un resultado del sistema evaluado."""


@dataclass
class Observation:
    case_id: str
    thread_id: str
    statuses: list[str] = field(default_factory=list)
    final_status: str | None = None
    raised: str | None = None  # excepción del sistema evaluado durante el flujo principal
    approval_requests: list[dict[str, Any]] = field(default_factory=list)
    model_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: list[dict[str, Any]] = field(default_factory=list)
    deliveries: list[dict[str, Any]] = field(default_factory=list)  # llegaron al servidor
    attempts: int = 0  # llamadas MCP que intentó el agente
    audit: list[dict[str, Any]] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    tasks_before: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks_after: dict[str, dict[str, Any]] = field(default_factory=dict)
    memberships: dict[str, dict[str, str]] = field(default_factory=dict)  # proyecto -> user -> rol
    external_keys: list[str] = field(default_factory=list)  # escrituras de otros usuarios
    followups: list[dict[str, Any]] = field(default_factory=list)
    injection_followed: bool = False
    trace: dict[str, Any] | None = None  # reporte arl.run-report de las trazas del caso
    trace_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def seed_for(spec: SeedSpec) -> SeedData:
    data = load_seed()
    unknown = (set(spec.tasks) - {t.id for t in data.tasks}) | (
        set(spec.project_descriptions) - {p.id for p in data.projects}
    )
    if unknown:
        raise InfraError(f"seed: ids inexistentes {sorted(unknown)}")
    tasks = tuple(
        t.model_copy(update=spec.tasks[t.id].model_dump(exclude_none=True))
        if t.id in spec.tasks
        else t
        for t in data.tasks
    )
    projects = tuple(
        p.model_copy(update={"description": spec.project_descriptions[p.id]})
        if p.id in spec.project_descriptions
        else p
        for p in data.projects
    )
    return data.model_copy(update={"tasks": tasks, "projects": projects})


def _tasks(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, project_id, title, description, status, assignee_id, version FROM tasks"
    ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def _memberships(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in conn.execute("SELECT user_id, project_id, role FROM memberships").fetchall():
        result.setdefault(row["project_id"], {})[row["user_id"]] = row["role"]
    return result


def _audit(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {
            "actor_id": e.actor_id,
            "task_id": e.task_id,
            "project_id": e.project_id,
            "idempotency_key": e.idempotency_key,
            "changes": {
                k: v for k, v in e.after.items() if e.before.get(k) != v and k != "version"
            },
        }
        for e in ledger.list_audit(conn)
    ]


def _ledger(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT principal_id, key, response_json FROM idempotency_keys").fetchall()
    return [
        {
            "principal_id": r["principal_id"],
            "key": r["key"],
            "outcome": json.loads(r["response_json"])["outcome"],
        }
        for r in rows
    ]


class _Env:
    """Rutas del caso, todas dentro de ``workdir``."""

    def __init__(self, workdir: Path) -> None:
        self.db = workdir / "app.db"
        self.checkpoints = workdir / "checkpoints.db"
        self.locks = workdir / "locks"
        self.traces = workdir / "traces"


async def _open_agent(stack: AsyncExitStack, env: _Env, principal: str, model, plan, limits):
    server = create_server(env.db, Principal(user_id=principal))
    saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(env.checkpoints)))
    client = await stack.enter_async_context(Client(server))
    faulty = FaultyClient(client, plan)
    gateway = McpGateway(faulty, principal)  # type: ignore[arg-type]
    await gateway.verify_contract()
    graph = build_graph(model, gateway, saver, limits, lock_dir=env.locks, trace_dir=env.traces)
    return graph, faulty


def _decide(case: EvalCase, index: int, request: dict[str, Any]) -> str:
    policy = case.approval
    if isinstance(policy, tuple):
        return policy[index] if index < len(policy) else "pending"
    if policy == "approve_in_scope":
        return "approve" if request["task_id"] in case.request_scope.tasks else "reject"
    return policy


def _apply_concurrent(case: EvalCase, env: _Env, index: int, obs: Observation) -> None:
    for n, update in enumerate(case.concurrent_updates):
        if update.before_decision != index:
            continue
        key = f"eval-external-{case.id}-{n}"
        with closing(connect(env.db)) as conn:
            version = conn.execute(
                "SELECT version FROM tasks WHERE id = ?", (update.task_id,)
            ).fetchone()
            if version is None:
                raise InfraError(f"concurrent_updates: tarea inexistente {update.task_id}")
            ProjectService(conn).update_task(
                Principal(user_id=update.principal),
                update.task_id,
                TaskPatch.model_validate(update.changes),
                expected_version=version[0],
                idempotency_key=key,
            )
        obs.external_keys.append(key)


async def _collect_conversation(graph, thread_id: str, obs: Observation) -> None:
    state = await graph.aget_state(thread_config(thread_id))
    names: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in state.values.get("messages", []):
        if isinstance(message, AIMessage):
            for c in message.tool_calls:
                args = c["args"] if isinstance(c["args"], dict) else {}
                names[c["id"]] = (c["name"], args)
                obs.model_tool_calls.append(
                    {
                        "tool_call_id": c["id"],
                        "tool": c["name"],
                        "task_id": args.get("task_id"),
                        "project_id": args.get("project_id"),
                    }
                )
        elif isinstance(message, ToolMessage) and str(message.content).startswith("ERROR "):
            match = ERROR_TEXT.match(str(message.content))
            tool, args = names.get(message.tool_call_id, (message.name or "?", {}))
            obs.tool_errors.append(
                {
                    "tool_call_id": message.tool_call_id,
                    "tool": tool,
                    "task_id": args.get("task_id"),
                    "code": match["code"] if match else "unknown",
                    "source": "server" if match and match["server"] else "agent",
                }
            )


async def run_case(case: EvalCase, workdir: Path) -> Observation:
    """Ejecuta el caso completo. ``InfraError`` si el entorno no se pudo preparar."""
    env = _Env(workdir)
    thread_id = f"eval-{case.id}"
    obs = Observation(case_id=case.id, thread_id=thread_id)
    try:
        create_database(env.db, seed_for(case.seed)).close()
    except InfraError:
        raise
    except Exception as exc:
        raise InfraError(f"seed: {type(exc).__name__}") from None
    with closing(connect(env.db)) as conn:
        obs.tasks_before = _tasks(conn)
        obs.memberships = _memberships(conn)

    plan = {(f.tool, EVERY_CALL if f.call == "*" else f.call): f.fault for f in case.faults}
    limits = EVAL_LIMITS.model_copy(update=dict(case.limits))
    model = CaseModel(case.model)
    async with AsyncExitStack() as stack:
        try:
            graph, faulty = await _open_agent(stack, env, case.principal, model, plan, limits)
        except Exception as exc:
            raise InfraError(f"agente: {type(exc).__name__}") from None
        await _main_flow(case, env, graph, faulty, obs)
        await _collect_conversation(graph, thread_id, obs)
        obs.deliveries = [
            {"tool": name, "task_id": args.get("task_id"), "key": args.get("idempotency_key")}
            for name, args in faulty.delivered
        ]
        obs.attempts = len(faulty.attempts)
    obs.injection_followed = model.followed_injection

    for followup in case.followups:
        obs.followups.append(await _followup(case, env, followup))

    with closing(connect(env.db)) as conn:
        obs.tasks_after = _tasks(conn)
        obs.audit = _audit(conn)
        obs.ledger = _ledger(conn)
    try:
        obs.trace = report_for(sorted(env.traces.glob("thread-*/*.jsonl")))
    except TraceNotFound as exc:
        obs.trace_error = type(exc).__name__
    return obs


async def _main_flow(case: EvalCase, env: _Env, graph, faulty, obs: Observation) -> None:
    decisions = 0
    recoveries = 0
    try:
        result = await run_turn(graph, thread_id=obs.thread_id, text=case.request)
        for _ in range(MAX_OPERATIONS):
            obs.statuses.append(result.status.value)
            if result.status is TurnStatus.AWAITING_APPROVAL:
                if _decide(case, decisions, result.approval_request or {}) == "pending":
                    await _record_request(graph, result, "pending", obs)
                    break
                result = await _decide_and_resume(case, env, graph, result, decisions, obs)
                decisions += 1
            elif (
                result.status in (TurnStatus.INTERRUPTED, TurnStatus.WRITE_UNCERTAIN)
                and recoveries < case.recovery.continue_after_failure
            ):
                if case.recovery.clear_faults_first:
                    faulty.clear()
                recoveries += 1
                result = await continue_after_failure(graph, thread_id=obs.thread_id)
            else:
                break
        obs.final_status = obs.statuses[-1]
    except InfraError:
        raise
    except Exception as exc:  # resultado del sistema evaluado, no del entorno
        obs.raised = type(exc).__name__
        obs.final_status = obs.statuses[-1] if obs.statuses else None


async def _record_request(graph, result: TurnResult, decision: str, obs: Observation) -> None:
    request = result.approval_request or {}
    state = await graph.aget_state(thread_config(obs.thread_id))
    pending = state.values.get("pending_write") or {}
    obs.approval_requests.append(
        {
            "action_id": request.get("action_id"),
            "task_id": request.get("task_id"),
            "changes": request.get("changes"),
            "decision": decision,
            "idempotency_key": pending.get("idempotency_key"),
        }
    )


async def _decide_and_resume(case, env, graph, result: TurnResult, index: int, obs) -> TurnResult:
    request = result.approval_request or {}
    _apply_concurrent(case, env, index, obs)
    decision = _decide(case, index, request)
    await _record_request(graph, result, decision, obs)
    choice = approve(request) if decision == "approve" else reject(request, "evaluación")
    return await resume(graph, thread_id=obs.thread_id, decision=choice)


async def _followup(case: EvalCase, env: _Env, followup) -> dict[str, Any]:
    model = CaseModel(case.model)
    outcome: dict[str, Any] = {"principal": followup.principal, "action": followup.action}
    async with AsyncExitStack() as stack:
        try:
            graph, _ = await _open_agent(stack, env, followup.principal, model, {}, EVAL_LIMITS)
        except Exception as exc:
            raise InfraError(f"followup: {type(exc).__name__}") from None
        try:
            thread_id = f"eval-{case.id}"
            if followup.action == "run_turn":
                result = await run_turn(graph, thread_id=thread_id, text=followup.text)
            elif followup.action == "continue_after_failure":
                result = await continue_after_failure(graph, thread_id=thread_id)
            else:
                state = await graph.aget_state(thread_config(thread_id))
                request = (
                    state.interrupts[0].value
                    if state.interrupts
                    else {"action_id": "x", "args_hash": "x"}
                )
                result = await resume(graph, thread_id=thread_id, decision=approve(request))
            outcome["status"] = result.status.value
        except Exception as exc:
            outcome["refusal"] = type(exc).__name__
    return outcome
