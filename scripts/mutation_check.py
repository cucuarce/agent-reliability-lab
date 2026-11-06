"""Chequeo de mutaciones: ¿los tests fallan ante las regresiones que dicen prevenir?

Cada mutante reintroduce un defecto concreto de identidad, aprobación o tolerancia a fallos. La
suite del agente debe fallar con cada uno ("KILLED"); un mutante "SURVIVED" es un hueco de tests.

Seguridad: nunca toca el árbol de trabajo. Copia el repositorio a un directorio temporal, muta
solo la copia y ejecuta pytest allí con ``PYTHONPATH`` apuntando a la copia (verificado antes de
empezar). Un Ctrl+C o un kill no pueden dejar código mutado en el repo.

Línea base obligatoria: si la suite sin mutar no pasa en la copia, aborta (exit 2), porque un
test roto "mataría" mutantes que no detecta.

Estados de cada mutante (6B.11): ``KILLED`` (pytest falla con tests identificados), ``SURVIVED``
(pytest pasa), ``TIMEOUT`` (superó ``--timeout``; se mata el árbol de procesos) y ``ERROR``
(pytest no terminó por un test: error interno, de uso, sin tests). Solo ``KILLED`` cuenta como
detectado. Un mutante con ``tests`` propios (focalizados) corre solo esos; los demás, ``TESTS``
completa. Con ``--results`` cada mutante deja una línea JSONL al terminar (commit, configuración,
estado, duración, código de salida y tests), así una corrida cortada conserva lo hecho.

Uso:
    uv run python scripts/mutation_check.py            # todos los mutantes (-x: rápido)
    uv run python scripts/mutation_check.py --list
    uv run python scripts/mutation_check.py --check-fragments   # sin pytest ni mutaciones
    uv run python scripts/mutation_check.py --group P1 --timeout 300 --results <afuera>.jsonl
    uv run python scripts/mutation_check.py --names-file <nombres>.txt --list   # selección exacta
    uv run python scripts/mutation_check.py -k runtime --all-failures

Exit: 0 todos detectados, 1 hay sobrevivientes, timeouts o errores, 2 abortado (línea base,
fragmento inválido, mutante que no compila, tests focalizados fuera de ``TESTS`` o restauración
fallida).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/unit/test_agent_components.py",
    "tests/integration/test_agent_flows.py",
    "tests/integration/test_agent_safety.py",
    "tests/integration/test_agent_isolation.py",
    "tests/integration/test_agent_crash.py",
    "tests/integration/test_agent_mcp.py",
    "tests/integration/test_agent_recovery_state.py",
    "tests/integration/test_agent_faults.py",
    "tests/integration/test_agent_faults_process.py",
    "tests/integration/test_agent_locks.py",
    "tests/integration/test_agent_history.py",
    "tests/integration/test_agent_log_safety.py",
    "tests/integration/test_agent_uncertain.py",
    "tests/integration/test_tracing.py",
    "tests/integration/test_tracing_cli.py",
    "tests/integration/test_tracing_integrity.py",
    "tests/integration/test_tracing_domain_codes.py",
    "tests/unit/test_evals_graders.py",
    "tests/integration/test_evals_suite.py",
    "tests/unit/test_request_scope.py",
    "tests/integration/test_agent_scope.py",
    "tests/integration/test_evals_scope.py",
    "tests/unit/test_tool_call_normalization.py",
    "tests/integration/test_agent_tool_call_ids.py",
    "tests/unit/test_live_provider.py",
    "tests/unit/test_live_model.py",
    "tests/unit/test_live_env_file.py",
    "tests/unit/test_live_truthfulness.py",
    "tests/integration/test_live_suite.py",
    "tests/integration/test_live_first_write.py",
    "tests/integration/test_live_reject_write.py",
    # 6B.11: dominio (autorización, versión, idempotencia), respuesta visible, transcripción y
    # escrituras reales de L09 a L14. Los mutantes P1 corren solo sus tests focalizados.
    "tests/unit/test_policy.py",
    "tests/integration/test_service_authorization.py",
    "tests/integration/test_idempotency.py",
    "tests/integration/test_update_policies.py",
    "tests/integration/test_mcp_server.py",
    "tests/integration/test_agent_outcomes.py",
    "tests/unit/test_outcomes_blocked_wording.py",
    "tests/unit/test_outcomes_limit_notice.py",
    "tests/integration/test_live_replies.py",
    "tests/unit/test_live_communication.py",
    "tests/integration/test_live_transcript.py",
    "tests/integration/test_live_external_writes.py",
    "tests/integration/test_live_close_write.py",
    "tests/integration/test_live_forbidden_write.py",
    "tests/integration/test_live_read_then_write.py",
    "tests/integration/test_live_conflict_write.py",
    "tests/integration/test_live_injected_write.py",
]
_COPY_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "var", "htmlcov",
    # bases SQLite, reportes y transcripciones: nunca entran a la copia (6B.11)
    "*.db", "*.db-*", "*.sqlite", "*.sqlite3", "*.jsonl", "live-reports", "live-transcripts",
)  # fmt: skip
# Variables que el proceso de pytest nunca hereda: los tests que necesitan una credencial la
# fijan ellos mismos (canarios).
_SECRET_ENV_MARKERS = ("API_KEY", "SECRET", "TOKEN", "PASSWORD")


def COPY_IGNORE(directory: str, names: list[str]) -> set[str]:
    """Además, los secretos nunca se copian al directorio temporal (sí ``.env.example``)."""
    secrets = {
        n
        for n in names
        if n == ".env"
        or (n.startswith(".env.") and n != ".env.example")
        or n.endswith((".pem", ".key"))
    }
    return set(_COPY_IGNORE_PATTERNS(directory, names)) | secrets


RUNTIME = "src/arl/agent/runtime.py"
GRAPH = "src/arl/agent/graph.py"
STATE = "src/arl/agent/state.py"
REGISTRY = "src/arl/agent/tool_registry.py"
GATEWAY = "src/arl/agent/gateway.py"
LOCKS = "src/arl/agent/locks.py"
REDACTION = "src/arl/agent/redaction.py"
EVAL_GRADERS = "src/arl/evals/graders.py"
EVAL_METRICS = "src/arl/evals/metrics.py"
EVAL_RUNNER = "src/arl/evals/runner.py"
SCOPE = "src/arl/agent/scope.py"
HISTORY = "src/arl/agent/history.py"
LIVE_MODEL = "src/arl/evals/live/model.py"
LIVE_PROVIDERS = "src/arl/evals/live/providers.py"
LIVE_RUNNER = "src/arl/evals/live/runner.py"
LIVE_GRADERS = "src/arl/evals/live/graders.py"
LIVE_TRUTH = "src/arl/evals/live/truthfulness.py"
LIVE_SUITE = "src/arl/evals/live/suite.py"
LIVE_PROFILES = "src/arl/evals/live/profiles.py"
LIVE_CASES = "src/arl/evals/live/cases.py"
LIVE_CLI = "src/arl/evals/live/__main__.py"
RECORDER = "src/arl/tracing/recorder.py"
QUERY = "src/arl/tracing/query.py"
LOGGING = "src/arl/tracing/logging_setup.py"


SERVICE = "src/arl/services/project_service.py"
POLICY = "src/arl/domain/policy.py"
OUTCOMES = "src/arl/agent/outcomes.py"
TRANSCRIPT = "src/arl/evals/live/transcript.py"


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    # Tests focalizados (archivo o ``archivo::test``, siempre de un archivo de ``TESTS``). Vacío:
    # la lista ``TESTS`` completa, como los mutantes históricos.
    tests: tuple[str, ...] = ()
    group: str = "historical"


HISTORICAL = [
    # --- Identidad y aprobación (etapa 3) ---
    Mutant("runtime: _check_owner no verifica", RUNTIME,
           "if owner is not None and owner != principal_id:", "if False:"),
    Mutant("runtime.resume: no compara action_id/args_hash", RUNTIME,
           "            if (decision.action_id, decision.args_hash) != (\n"
           '                request["action_id"],\n                request["args_hash"],\n'
           "            ):",
           "            if False:"),
    Mutant("runtime.turn_result: sin chequeo de dueño", RUNTIME,
           "    _check_owner(graph, (await graph.aget_state(thread_config(thread_id))).values)\n",
           ""),
    Mutant("runtime.continue_after_failure: sin chequeo de dueño", RUNTIME,
           "            _check_owner(graph, snapshot.values)\n            if snapshot.interrupts:",
           "            if snapshot.interrupts:"),
    Mutant("graph.check_principal desactivado", GRAPH,
           '        if state.get("principal_id") != gateway.principal_id:\n            raise',
           "        if False:\n            raise"),
    Mutant("await_approval: no verifica el principal del proceso", GRAPH,
           '        if state.get("principal_id") != gateway.principal_id:\n'
           "            tracing.emit(",
           "        if False:\n            tracing.emit("),
    Mutant("await_approval: no verifica que la decisión sea de la acción", GRAPH,
           "        if not decision.matches(pending):\n            tracing.emit(",
           "        if False:\n            tracing.emit("),
    Mutant("await_approval: lanza ante decisión inválida (envenena el thread)", GRAPH,
           'return {"approval": None, "approval_error": "la decisión corresponde a otra acción"}',
           'raise ApprovalMismatch("mutante")'),
    Mutant("execute_write: no exige aprobación válida", GRAPH,
           'if decision.decision != "approve" or not decision.matches(pending):', "if False:"),
    Mutant("dispatch: sin límite de tool calls", GRAPH,
           "            if used >= limits.max_tool_calls:", "            if False:"),
    Mutant("reducer de principal lanza (envenena el thread)", STATE,
           "    return current or new",
           '    if current and new and current != new:\n        raise PrincipalMismatch("m")\n'
           "    return current or new"),
    Mutant("reducer de principal acepta el nuevo principal", STATE,
           "    return current or new", "    return new or current"),
    Mutant("PendingWrite: sello sin verificar", STATE,
           'if _sha256(self.model_dump(exclude={"args_hash"})) != self.args_hash:', "if False:"),
    Mutant("ApprovalDecision acepta campos extra", STATE,
           'argumentos nuevos (``extra=forbid``)."""\n\n'
           '    model_config = ConfigDict(frozen=True, extra="forbid")',
           'argumentos nuevos."""\n\n    model_config = ConfigDict(frozen=True, extra="ignore")'),
    Mutant("args del LLM aceptan campos extra", REGISTRY,
           'model_config = ConfigDict(extra="forbid", frozen=True)',
           'model_config = ConfigDict(extra="ignore", frozen=True)'),
    Mutant("idempotency_key ignora la tool call", STATE,
           'digest = _sha256({"thread_id": thread_id, "tool_call_id": tool_call_id})',
           'digest = _sha256({"thread_id": thread_id})'),
    # --- Tolerancia a fallos (etapa 4) ---
    Mutant("is_retryable: reintenta también fallos definitivos", GRAPH,
           "return isinstance(exc, ToolCallFailed) and exc.retryable",
           "return isinstance(exc, ToolCallFailed)"),
    Mutant("is_retryable: nunca reintenta", GRAPH,
           "return isinstance(exc, ToolCallFailed) and exc.retryable", "return False"),
    Mutant("nodos MCP sin RetryPolicy", GRAPH,
           "builder.add_node(node.__name__, node, retry_policy=mcp_retry)",
           "builder.add_node(node.__name__, node)"),
    Mutant("gateway: error opaco del servidor tratado como dominio", GATEWAY,
           'if text.strip() == f"Error executing tool {name}":', "if False:"),
    Mutant("gateway: no valida la respuesta contra el contrato", GATEWAY,
           "return ToolResult(False, text, _validate_output(name, arguments, "
           "result.structured_content))",
           "return ToolResult(False, text, result.structured_content)"),
    Mutant("gateway: lanza dentro del except (el error original queda en __context__)", GATEWAY,
           "            failure = ToolUnavailable(\n"
           '                name, "transporte no disponible", error_type=type(exc).__name__\n'
           "            )",
           '            raise ToolUnavailable(\n'
           '                name, "transporte no disponible", error_type=type(exc).__name__\n'
           "            ) from None"),
    Mutant("runtime: run_turn descarta una escritura incierta", RUNTIME,
           '            if snapshot.next == ("execute_write",):\n'
           "                raise ThreadNeedsRecovery(",
           "            if False:\n                raise ThreadNeedsRecovery("),
    Mutant("locks: thread_lock no excluye", LOCKS,
           "        try:\n            _try_lock(handle)", "        try:\n            pass"),
    Mutant("runtime: sin lock_dir sigue sin exclusión (un directorio por llamada)", RUNTIME,
           "    if not value:\n        raise LockingRequired(",
           '    if not value:\n        return Path(__file__).parent / f".nolock-{uuid.uuid4()}"\n'
           "        raise LockingRequired("),
    Mutant("build_graph: acepta construir sin lock_dir", GRAPH,
           "    if lock_dir is None:\n        raise LockingRequired(",
           "    if False:\n        raise LockingRequired("),
    Mutant("runtime: turn_result no informa WRITE_UNCERTAIN", RUNTIME,
           'if snapshot.next == ("execute_write",):  # la escritura aprobada',
           "if False:  # la escritura aprobada"),
    Mutant("runtime: los fallos MCP no se convierten en estado", RUNTIME,
           "    except ToolCallFailed as exc:", "    except ZeroDivisionError as exc:"),
    Mutant("continue_after_failure: acepta threads esperando aprobación", RUNTIME,
           "            if snapshot.interrupts:\n                raise ThreadAwaitingApproval(\n"
           "                    f\"el thread '{thread_id}' espera una decisión, no un reintento\"",
           "            if False:\n                raise ThreadAwaitingApproval(\n"
           "                    f\"el thread '{thread_id}' espera una decisión, no un reintento\""),
    # --- Historial, resultado incierto y logs (revisión de etapa 4) ---
    Mutant("run_turn: no cierra las tool calls interrumpidas", RUNTIME,
           'closed = close_unanswered(snapshot.values.get("messages", []))', "closed = []"),
    Mutant("call_model: envía el historial sin verificar el contrato", GRAPH,
           '        check_tool_protocol(state["messages"])\n', ""),
    Mutant("WRITE_UNCERTAIN no expone la clave para reconciliar", RUNTIME,
           '            "idempotency_key": pending.idempotency_key,\n'
           '            "tool_call_id": pending.tool_call_id,',
           '            "tool_call_id": pending.tool_call_id,'),
    Mutant("logger del agente sin filtro de secretos", REDACTION,
           "LOGGER.addFilter(RedactingFilter())", "pass"),
    Mutant("runtime: el log de fallos no incluye el tipo de error", RUNTIME,
           "exc.tool, exc.code, exc.error_type, result.status",
           'exc.tool, exc.code, "?", result.status'),
    # --- Observabilidad (etapa 5) ---
    Mutant("trazas: emit propaga excepciones (LangGraph reintentaría la escritura)", RECORDER,
           "        except Exception as exc:  # best-effort: nunca afecta la ejecución",
           "        except ZeroDivisionError as exc:"),
    Mutant("trazas: no valida el evento contra el esquema", RECORDER,
           "model = EVENT_SCHEMAS[event].model_validate(data)",
           "model = EVENT_SCHEMAS[event].model_construct(**data)"),
    Mutant("trazas: tool_call_routed registra valores en vez de nombres de campos", GRAPH,
           '"changed_fields": sorted(args.changes.changes())}',
           '"changed_fields": sorted(map(str, args.changes.changes().values()))}'),
    Mutant("trazas: BUSY no deja run_finished", RUNTIME,
           "        except ThreadBusy:\n"
           '            finish(TurnStatus.BUSY, refusal="ThreadBusy")\n',
           "        except ThreadBusy:\n"),
    Mutant("consulta: escritura incierta informada como fallida", QUERY,
           'return "uncertain", note, "evidence"', 'return "failed", note, "evidence"'),
    Mutant("consulta: un reporte importado se cree tal cual", QUERY,
           "    rebuilt = reconstruct(events, counts[0], unsupported_lines=counts[1])",
           "    rebuilt = report"),
    Mutant("consulta: los duplicados cuentan dos veces", QUERY,
           "        elif unique[key] == e:", "        elif False:"),
    Mutant("consulta: sin decisión en las trazas, un envío se informa como no preparado", QUERY,
           'if "approve" in w["decisions"] or w["attempts"] or w["uncertain_runs"]:',
           'if "approve" in w["decisions"]:'),
    Mutant("consulta: run_id validado con isalnum()", QUERY,
           "if RUN_ID.fullmatch(run_id) else []", "if run_id.isalnum() else []"),
    Mutant("trazas: un evento descartado no deja hueco en seq", RECORDER,
           "        self.seq += 1\n        try:\n"
           "            model = EVENT_SCHEMAS[event].model_validate(data)\n",
           "        try:\n            model = EVENT_SCHEMAS[event].model_validate(data)\n"
           "            self.seq += 1\n"),
    Mutant("logging: aiosqlite queda en DEBUG", LOGGING,
           "logging.NOTSET if allow_unsafe_debug else max(numeric, logging.INFO)",
           "logging.NOTSET"),
    Mutant("trazas: el código de dominio ignora el prefijo del SDK (defecto D1)", GATEWAY,
           '    body = text.removeprefix(f"Error executing tool {tool}: ")', "    body = text"),
    Mutant("trazas: un código desconocido se registra como de dominio", GATEWAY,
           "match.group(1) if match and match.group(1) in DOMAIN_CODES else None",
           "match.group(1) if match else None"),
    # --- Alcance del turno (etapa 6A.1) ---
    Mutant("alcance: dispatch no verifica el alcance (F1 vuelve)", GRAPH,
           "                    bound.check(write.task_id, write.changes.changes(), ledger)",
           "                    pass"),
    Mutant("alcance: prepare_write reserva sin volver a verificar", GRAPH,
           "            grant_id = bound_scope(state, config).check(args.task_id, changes, ledger)",
           '            grant_id = "sin-verificar"'),
    Mutant("alcance: execute_write no exige la reserva", GRAPH,
           '            raise ScopeViolation("la acción aprobada no tiene una reserva del alcance '
           'del turno")',
           "            pass"),
    Mutant("alcance: el rechazo humano consume el permiso", GRAPH,
           '        ledger = _settle(state, pending, "released")',
           '        ledger = _settle(state, pending, "consumed")'),
    Mutant("alcance: las reservas abiertas no cuentan (sobreasignación)", SCOPE,
           'COUNTED_STATES = frozenset({"reserved", "consumed", "refused", "invalidated"})',
           'COUNTED_STATES = frozenset({"consumed", "refused", "invalidated"})'),
    Mutant("alcance: un rechazo definitivo del servidor devuelve el permiso", SCOPE,
           'COUNTED_STATES = frozenset({"reserved", "consumed", "refused", "invalidated"})',
           'COUNTED_STATES = frozenset({"reserved", "consumed", "invalidated"})'),
    Mutant("alcance: version_conflict libera en vez de invalidar", SCOPE,
           '        return "invalidated"', '        return "released"'),
    Mutant("alcance: el sello no se verifica", SCOPE,
           "        if _sha256(body) != self.scope_hash:", "        if False:"),
    Mutant("alcance: no se verifica el thread ni el principal", SCOPE,
           "        if (self.thread_id, self.principal_id) != (thread_id, principal_id):",
           "        if False:"),
    Mutant("evals: la medición 'antes' corre con la defensa", EVAL_RUNNER,
           "    if case.scope is not None and enforce:", "    if case.scope is not None:"),
    # --- Identidad de las tool calls con proveedores reales (etapa 6B.1, R1) ---
    Mutant("R1: call_model usa los ids del proveedor", GRAPH,
           '        response = normalize_response(response, state["messages"])', "        pass"),
    Mutant("R1: el id depende solo del índice (se repite entre turnos)", HISTORY,
           '{"id": f"call_{turn}_{i}", "name"', '{"id": f"call_0_{i}", "name"'),
    # --- Evaluación con LLM real (etapa 6B.1) ---
    Mutant("live: no se verifica el presupuesto antes de llamar", LIVE_MODEL,
           "            if reason := meter.refusal(worst, self.limits.call_timeout_s):",
           "            if reason := None:"),
    Mutant("live: los intentos no cuentan como llamadas", LIVE_MODEL,
           "            meter.calls += 1", "            pass"),
    Mutant("live: se reintenta antes de lo que pide Retry-After", LIVE_MODEL,
           "        if wait > self.limits.max_retry_wait_s:", "        if False:"),
    Mutant("live: se reintentan errores no transitorios", LIVE_MODEL,
           "        if not failure.retryable or attempt >= self.limits.max_retries:",
           "        if attempt >= self.limits.max_retries:"),
    Mutant("live: una cota de tokens incorrecta no detiene la suite", LIVE_MODEL,
           '                self._break("prompt_estimate_exceeded")', "                pass"),
    Mutant("live: tool_use_failed se cuenta como error del proveedor", LIVE_PROVIDERS,
           '        return "model" if self.code == "tool_use_failed" else "provider"',
           '        return "provider"'),
    Mutant("live: un 401 no detiene la suite", LIVE_PROVIDERS,
           '_ABORTS_SUITE = frozenset({"auth", "forbidden", "not_found"})',
           '_ABORTS_SUITE = frozenset({"forbidden", "not_found"})'),
    Mutant("live: el .env sobrescribe la credencial del entorno", LIVE_PROVIDERS,
           "    if API_KEY_ENV in environ:", "    if False:"),
    Mutant("live: del .env se toman todas las variables", LIVE_PROVIDERS,
           "    value = dotenv_values(path, interpolate=False).get(API_KEY_ENV)",
           "    values = dotenv_values(path, interpolate=False)\n"
           "    environ.update({k: v for k, v in values.items() if v is not None})\n"
           "    value = values.get(API_KEY_ENV)"),
    Mutant("live: una cuota agotada no detiene la suite", LIVE_PROVIDERS,
           "        return self.code in _ABORTS_SUITE or self.code in _QUOTA_EXHAUSTED",
           "        return self.code in _ABORTS_SUITE"),
    Mutant("live: run acepta escenarios que el perfil no autoriza", LIVE_CLI,
           "    unauthorized = sorted(set(args.only) - set(config.allowed_cases))",
           "    unauthorized = []"),
    Mutant("live: el servidor MCP recibe la clave del proveedor", GATEWAY,
           "        **_without_credentials(os.environ),", "        **os.environ,"),
    Mutant("live: un modelo real puede correr escenarios de escritura", LIVE_RUNNER,
           "    if real_model:\n        authorize_real_write(case, config)\n    policy",
           "    policy"),
    Mutant("live: la CLI acepta escenarios de escritura", LIVE_CLI,
           "    for case in writes:  # lista cerrada, perfil que la habilita y configuración"
           " registrada\n        authorize_real_write(case, config)\n", ""),
    Mutant("live: la CLI no exige --confirm-real-writes", LIVE_CLI,
           "    if writes and not args.confirm_real_writes:", "    if False:"),
    Mutant("live: la lista cerrada no limita las escrituras reales", LIVE_PROFILES,
           "    if case.id not in live_cases.REAL_WRITE_CASES:", "    if False:"),
    Mutant("live: el perfil no necesita habilitar la escritura", LIVE_PROFILES,
           "    if case.id not in config.real_writes:", "    if False:"),
    Mutant("live: una configuración no registrada habilita escrituras", LIVE_PROFILES,
           "    if config not in PROFILES.values():", "    if False:"),
    Mutant("live: un perfil amplía la lista cerrada", LIVE_PROFILES,
           "        if outside := sorted(set(self.real_writes)"
           " - set(live_cases.REAL_WRITE_CASES)):",
           "        if outside := []:"),
    Mutant("live: la escritura no se vuelve a autorizar antes de despacharse", LIVE_RUNNER,
           "                if real_model:  # otra vez, antes de despachar la decisión",
           "                if False:  # otra vez, antes de despachar la decisión"),
    Mutant("live: el rechazo se despacha sin volver a autorizar", LIVE_RUNNER,
           "                if real_model:  # otra vez, antes de despachar la decisión",
           '                if decision == "approve" and real_model:  # otra vez'),
    Mutant("live: la autorización no verifica la decisión permitida", LIVE_PROFILES,
           '    if policy is None or policy["decision"] not in'
           " live_cases.REAL_WRITE_CASES[case.id]:",
           "    if policy is None:"),
    Mutant("live: un perfil cambia la decisión permitida", LIVE_PROFILES,
           "            if decision is not None and decision not in"
           " live_cases.REAL_WRITE_CASES[case_id]:",
           "            if False:"),
    Mutant("live: L11 también puede aprobar", LIVE_CASES,
           '"L11": frozenset({"reject"})', '"L11": frozenset({"approve", "reject"})'),
    Mutant("live: la lista cerrada se puede modificar en ejecución", LIVE_CASES,
           "REAL_WRITE_CASES: Mapping[str, frozenset[str]] = MappingProxyType(",
           "REAL_WRITE_CASES: Mapping[str, frozenset[str]] = dict("),
    Mutant("live: reject-write no admite 6 pedidos con su peor caso", LIVE_PROFILES,
           '    "reject-write": LiveConfig(\n'
           "        scenario_budget=Budget(max_model_calls=6, max_total_tokens=40_000,",
           '    "reject-write": LiveConfig(\n'
           "        scenario_budget=Budget(max_model_calls=6, max_total_tokens=24_000,"),
    Mutant("live: la aprobación simulada amplía el alcance", LIVE_RUNNER,
           '    if not _in_scope(case, request.get("task_id"), request.get("changes")):',
           "    if False:"),
    Mutant("live: approval_within_scope nunca falla", LIVE_GRADERS,
           '            not getattr(obs, "approval_out_of_scope", 0),', "            True,"),
    Mutant("live: el runner acepta un directorio fuera del temporal", LIVE_RUNNER,
           "    if not resolved.is_relative_to(_scratch_root()) or _is_synced_folder(resolved):",
           "    if False:"),
    Mutant("live: el runner acepta un directorio con datos previos", LIVE_RUNNER,
           "    if not resolved.is_dir() or any(resolved.iterdir()):",
           "    if not resolved.is_dir():"),
    Mutant("live: un escenario empieza sin presupuesto completo", LIVE_SUITE,
           "if remaining is None or not remaining.covers(config.scenario_budget):",
           "if remaining is None:"),
    Mutant("veracidad: afirmar una escritura no ejecutada pasa como consistente", LIVE_TRUTH,
           '                finding = "consistent" if done else "false_affirmation"',
           '                finding = "consistent"'),
    Mutant("veracidad: se ignora la negación cercana", LIVE_TRUTH,
           "            if _NEGATION.search(window):", "            if False:"),
    Mutant("veracidad: una escritura ajena cuenta como del agente", LIVE_TRUTH,
           "    executed = executed_tasks(own_audit)",
           "    executed = executed_tasks(own_audit) | {t for t, a in obs.tasks_after.items()"
           " if a != obs.tasks_before.get(t)}"),
    Mutant("veracidad: un false_claim no falla la corrida", LIVE_GRADERS,
           '    status = {"false_claim": "fail",', '    status = {"false_claim": "pass",'),
    Mutant("live: los bloqueos por alcance no se cuentan", LIVE_GRADERS,
           '    blocked = [e for e in obs.tool_errors if e["code"] == "out_of_scope"]',
           "    blocked = []"),
    Mutant("live: los 429 que agotan el presupuesto no se informan", LIVE_GRADERS,
           '        if c["outcome"] == "error" and c["origin"] == "provider"',
           "        if False"),
    Mutant("live: un 429 reintentado se informa como causa contribuyente", LIVE_GRADERS,
           "    if not obs.budget_stop:\n        return []\n", ""),
    Mutant("live: una escritura real corre sin política de aprobación explícita", LIVE_PROFILES,
           "    elif real:\n        raise ApprovalPolicyMissing",
           "    elif False:\n        raise ApprovalPolicyMissing"),
    Mutant("live: se aceptan aprobaciones distintas del caso y del perfil", LIVE_PROFILES,
           "    if len(set(declared.values())) > 1:", "    if False:"),
    Mutant("live: un perfil declara aprobaciones de escenarios que no autoriza", LIVE_PROFILES,
           "        extra = sorted(set(self.approvals) - set(self.allowed_cases or ()))",
           "        extra = []"),
    Mutant("live: el runner corre una escritura sin decisión de aprobación", LIVE_RUNNER,
           "    policy = approval_policy(case, config, real=real_model)",
           '    policy = {"decision": "approve", "source": "mutant", "actor": "simulated"}'),
    Mutant("live: el intento de ataque se cuenta como ejecución", LIVE_GRADERS,
           '        "executed": any(matches(a["task_id"], a["changes"]) for a in _own_audit(obs)),',
           '        "executed": bool(attempts),'),
    # --- Evaluador (etapa 6A): un grader roto no puede pasar desapercibido ---
    Mutant("evals: approved_only nunca falla", EVAL_GRADERS,
           '            all(a["idempotency_key"] in approved for a in own),', "            True,"),
    Mutant("evals: una escritura incierta reportada como confirmada pasa", EVAL_GRADERS,
           '                    "uncertain" in states.values(),', "                    True,"),
    Mutant("evals: falta de trazas cuenta como pass", EVAL_GRADERS,
           'GraderResult("trace_error_codes", "functional", "not_evaluable", "sin trazas")',
           'GraderResult("trace_error_codes", "functional", "pass", "sin trazas")'),
    Mutant("evals: las garantías propuestas cambian el veredicto", EVAL_GRADERS,
           '    counted = [g for g in graders if g.kind != "proposed"]', "    counted = graders"),
    Mutant("evals: los casos no evaluables entran al denominador funcional", EVAL_METRICS,
           '        and all(g["status"] != "not_evaluable" for g in _graders(r, "functional"))\n',
           ""),
    Mutant("evals: un ataque bloqueado se clasifica como ejecutado", EVAL_GRADERS,
           '    return {"outcome": "blocked", "by": reasons or ["not_executed"]}',
           '    return {"outcome": "executed", "with_human_approval": False}'),
    Mutant("evals: toda ejecución de un ataque cuenta como insegura", EVAL_METRICS,
           '        and any(g["status"] == "fail" for g in _graders(r, "safety"))\n',
           ""),
    Mutant("evals: el humano atento aprueba todo", EVAL_RUNNER,
           'return "approve" if request["task_id"] in case.request_scope.tasks else "reject"',
           'return "approve"'),
]  # fmt: skip


# --- P1: regresiones críticas de seguridad (etapa 6B.11) -----------------------------------------
# Cada uno reintroduce un defecto real (no equivalente) y corre solo sus tests focalizados. Cuando
# hay dos, uno prueba la regla en su capa y el otro el efecto de punta a punta (ver docs).
_SVC = "tests/integration/test_service_authorization.py"
_IDEM = "tests/integration/test_idempotency.py"
_SCOPE_UNIT = "tests/unit/test_request_scope.py::test_grant_allows_only_its_tasks_fields_and_values"
_INJECTED = "tests/integration/test_live_injected_write.py"
_CONFLICT = "tests/integration/test_live_conflict_write.py"
_EXTERNAL = "tests/integration/test_live_external_writes.py"
_CAS_BLOCK_OLD = (
    "            if task.version != expected_version:\n"
    "                raise VersionConflict(task_id, expected_version, task.version)\n"
    "\n"
    "            now = self._clock().isoformat()\n"
    '            current = task.model_dump(mode="json")\n'
    "            effective = {\n"
    "                field: value for field, value in changes.items() if current[field] != value\n"
    "            }\n"
    "            if not effective:\n"
    "                result = UpdateResult(task=task, outcome=UpdateOutcome.UNCHANGED, "
    "audit_id=None)\n"
    "            else:\n"
    "                if not repositories.update_task(conn, task_id, effective, expected_version):\n"
)
# Sin el chequeo de versión **y** con el compare-and-set sobre la versión actual: cada capa sola
# es equivalente (la otra frena la escritura vieja), así que se mutan las dos.
_CAS_BLOCK_NEW = _CAS_BLOCK_OLD.replace(
    "            if task.version != expected_version:\n"
    "                raise VersionConflict(task_id, expected_version, task.version)\n"
    "\n",
    "",
).replace(
    "(conn, task_id, effective, expected_version)", "(conn, task_id, effective, task.version)"
)

P1 = [
    Mutant("P1 dominio: un viewer puede modificar tareas (política)", POLICY,
           "    Action.UPDATE_TASK: frozenset({Role.EDITOR, Role.OWNER}),",
           "    Action.UPDATE_TASK: frozenset({Role.VIEWER, Role.EDITOR, Role.OWNER}),",
           ("tests/unit/test_policy.py", f"{_SVC}::test_viewer_cannot_update"), "P1"),
    Mutant("P1 dominio: un no miembro lee el proyecto como viewer", SERVICE,
           "            if role is None or not is_allowed(role, Action.READ_PROJECT):\n"
           '                raise NotFound("proyecto", project_id)',
           '            role = role or "viewer"',
           (f"{_SVC}::test_foreign_and_missing_projects_are_indistinguishable",), "P1"),
    Mutant("P1 dominio: update_task no verifica el rol de escritura", SERVICE,
           "            if not is_allowed(role, Action.UPDATE_TASK):", "            if False:",
           (f"{_SVC}::test_viewer_cannot_update",), "P1"),
    Mutant("P1 dominio: sin control de versión ni compare-and-set", SERVICE,
           _CAS_BLOCK_OLD, _CAS_BLOCK_NEW,
           (f"{_SVC}::test_stale_version_is_rejected",
            f"{_IDEM}::test_concurrent_writers_with_same_version_have_one_winner"), "P1"),
    Mutant("P1 idempotencia: un reintento no devuelve la respuesta guardada", SERVICE,
           "            if previous is not None:", "            if False:",
           (f"{_IDEM}::test_retry_with_same_key_and_params_does_not_duplicate",), "P1"),
    Mutant("P1 idempotencia: misma clave con otros parámetros hace replay", SERVICE,
           "                    raise IdempotencyConflict(key)", "                    pass",
           (f"{_IDEM}::test_same_key_with_different_params_conflicts",), "P1"),
    Mutant("P1 alcance: covers no verifica la tarea (resource)", SCOPE,
           '        if task_id not in self.task_ids:\n            return "resource"',
           '        if False:\n            return "resource"',
           (_SCOPE_UNIT,
            f"{_INJECTED}::test_a_only_apo_4_is_blocked_before_approval_and_apo_3_stays_open"),
           "P1"),
    Mutant("P1 alcance: covers no verifica los campos (field)", SCOPE,
           '        if set(changes) - set(self.fields):\n            return "field"\n'
           "        for name, value in changes.items():\n"
           "            allowed = self.fields[name]",
           "        for name, value in changes.items():\n"
           "            allowed = self.fields.get(name)",
           (_SCOPE_UNIT,
            f"{_INJECTED}::test_e_null_padding_on_apo_3_is_blocked_and_the_correction_writes_once"),
           "P1"),
    Mutant("P1 alcance: covers no verifica el valor (value)", SCOPE,
           "            if allowed is not None and value not in allowed:",
           "            if False:",
           (_SCOPE_UNIT,
            "tests/unit/test_request_scope.py::test_unassigning_is_a_value_that_must_be_allowed"),
           "P1"),
    Mutant("P1 respuesta: con escrituras se muestra el texto del modelo", OUTCOMES,
           "    if not lines:\n        return model_reply, \"model\"",
           "    if True:\n        return model_reply, \"model\"",
           ("tests/integration/test_agent_outcomes.py"
            "::test_an_executed_write_is_reported_from_the_server_confirmation",
            f"{_CONFLICT}::test_user_reply_never_claims_the_write_the_server_refused"), "P1"),
    Mutant("P1 respuesta: version_conflict se informa como ejecutada", OUTCOMES,
           '    "version_conflict": "conflict",', '    "version_conflict": "executed",',
           (f"{_CONFLICT}::test_l13_keeps_bobs_change_and_never_applies_alices",
            f"{_EXTERNAL}::test_bobs_change_is_never_attributed_to_the_agent"), "P1"),
    Mutant("P1 transcripción: no se redactan secretos", TRANSCRIPT,
           "        return _SECRETS.subn(REDACTED, value)", "        return value, 0",
           ("tests/integration/test_live_transcript.py"
            "::test_transcript_never_copies_reasoning_headers_or_credentials",), "P1"),
    Mutant("P1 atribución: _own_audit cuenta escrituras de otros actores", EVAL_GRADERS,
           '    return [a for a in obs.audit if a["idempotency_key"] not in external]',
           "    return list(obs.audit)",
           (f"{_EXTERNAL}::test_bobs_change_is_never_attributed_to_the_agent",
            f"{_CONFLICT}::test_l13_keeps_bobs_change_and_never_applies_alices"), "P1"),
    Mutant("P1 atribución: writes_executed suma filas de otros actores", LIVE_SUITE,
           '                sum(r["observation"]["own_audit_rows"] for r in ran),',
           '                sum(r["observation"]["audit_rows"] for r in ran),',
           (f"{_EXTERNAL}::test_bobs_concurrent_change_is_recorded_as_an_external_write",), "P1"),
    Mutant("P1 atribución: writes.effective cuenta el ledger de otros actores", LIVE_SUITE,
           '    own = Counter(row["outcome"] for row in obs.ledger if row["key"] not in external)',
           '    own = Counter(row["outcome"] for row in obs.ledger)',
           (f"{_EXTERNAL}::test_bobs_ledger_entry_is_not_an_effective_write_of_the_agent",), "P1"),
    Mutant("P1 atribución: toda fila de audit_log se marca como propia", LIVE_SUITE,
           '                "own": a["idempotency_key"] not in external,',
           '                "own": True,',
           (f"{_EXTERNAL}::test_bobs_concurrent_change_is_recorded_as_an_external_write",), "P1"),
]  # fmt: skip

MUTANTS = HISTORICAL + P1

KILLED, SURVIVED, TIMEOUT, ERROR = "KILLED", "SURVIVED", "TIMEOUT", "ERROR"
DEFAULT_TIMEOUT_S = 900.0  # por mutante
DEFAULT_BASELINE_TIMEOUT_S = 1800.0


def child_env(copy: Path) -> dict[str, str]:
    """El entorno de pytest: importa la copia y nunca hereda credenciales."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_ENV_MARKERS)
    }
    env["PYTHONPATH"] = str(copy / "src")
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Termina pytest y todos sus hijos (servidores MCP, subprocesos de los tests)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, check=False,
        )  # fmt: skip
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


@dataclass(frozen=True)
class PytestRun:
    returncode: int | None  # None: timeout
    stdout: str
    duration_s: float

    @property
    def timed_out(self) -> bool:
        return self.returncode is None


def run_pytest(
    copy: Path, tests: list[str], stop_first: bool, timeout: float | None = None
) -> PytestRun:
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    if stop_first:
        args.insert(4, "-x")
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
        if os.name == "nt"
        else {"start_new_session": True}
    )
    started = time.monotonic()
    proc = subprocess.Popen(
        args, cwd=copy, env=child_env(copy), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", **group,
    )  # fmt: skip
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, _ = proc.communicate()
        code = None
    except BaseException:  # Ctrl+C: el árbol de pytest nunca queda vivo
        _kill_tree(proc)
        proc.wait()
        raise
    return PytestRun(code, stdout or "", round(time.monotonic() - started, 2))


def failed_tests(output: str) -> list[str]:
    names = {
        line[7:].split(" - ")[0].split("::")[-1].split("[")[0]
        for line in output.splitlines()
        if line.startswith(("FAILED ", "ERROR ")) and "::" in line  # no logs capturados
    }
    return sorted(names)


def classify(run: PytestRun, killers: list[str]) -> str:
    """Solo un fallo de tests identificado detecta el mutante. pytest: 0 pasa, 1 tests
    fallidos, 2 interrumpido o error de colección, 3 error interno, 4 uso, 5 sin tests."""
    if run.timed_out:
        return TIMEOUT
    if run.returncode == 0:
        return SURVIVED
    if run.returncode in (1, 2) and killers:
        return KILLED
    return ERROR


def check_isolation(copy: Path) -> None:
    """Aborta si el paquete importado no es el de la copia (mutar no tendría efecto)."""
    imported = subprocess.run(
        [sys.executable, "-c", "import arl; print(arl.__file__)"],
        cwd=copy, env=child_env(copy), capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    if not Path(imported).resolve().is_relative_to(copy.resolve()):
        sys.exit(f"ABORT: se importa {imported}, no la copia en {copy}")


def tests_for(mutant: Mutant) -> list[str]:
    return list(mutant.tests) or list(TESTS)


def catalog_problems(root: Path, mutants: list[Mutant]) -> list[str]:
    """Problemas estáticos del catálogo, sin pytest ni mutaciones en disco: fragmento único,
    mutante que compila, nombres únicos y tests focalizados dentro de ``TESTS``."""
    problems = []
    names = [m.name for m in mutants]
    problems += [f"nombre repetido: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    for mutant in mutants:
        target = root / mutant.path
        if not target.is_file():
            problems.append(f"'{mutant.name}': no existe {mutant.path}")
            continue
        original = target.read_text(encoding="utf-8")
        count = original.count(mutant.old)
        if count != 1:
            problems.append(f"'{mutant.name}': el fragmento aparece {count} veces")
            continue
        if mutant.old == mutant.new:
            problems.append(f"'{mutant.name}': el mutante no cambia nada")
        try:
            compile(original.replace(mutant.old, mutant.new), str(target), "exec")
        except SyntaxError as exc:
            problems.append(f"'{mutant.name}': no es Python válido (línea {exc.lineno})")
        for test in mutant.tests:
            file = test.split("::")[0]
            if file not in TESTS:
                problems.append(f"'{mutant.name}': {file} no está en TESTS")
            elif not (root / file).is_file():
                problems.append(f"'{mutant.name}': no existe {file}")
    return problems


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RestoreError(RuntimeError):
    pass


@contextlib.contextmanager
def mutated(target: Path, mutant: Mutant) -> Iterator[None]:
    """Aplica el mutante a la copia y la restaura siempre (también ante excepciones o Ctrl+C);
    verifica la restauración por hash."""
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace(mutant.old, mutant.new), encoding="utf-8")
    try:
        yield
    finally:
        target.write_text(original, encoding="utf-8")
        if _sha(target.read_text(encoding="utf-8")) != _sha(original):
            raise RestoreError(f"no se pudo restaurar {mutant.path}")


def git_state(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


class Results:
    """JSONL incremental: una línea por evento, escrita y vaciada al terminar cada mutante."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _results_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).resolve()
    if path.is_relative_to(ROOT.resolve()):
        sys.exit("ABORT: --results va fuera del repositorio (no se versionan resultados)")
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


NAMES_FILE_GROUP = "historical"


def select_by_names(text: str, mutants: list[Mutant]) -> tuple[list[Mutant], list[str]]:
    """``(seleccionados, problemas)``. Un nombre exacto del catálogo por línea (las vacías y las
    que empiezan con ``#`` se ignoran); nunca posiciones ni subcadenas. Solo mutantes del grupo
    ``historical``. Se devuelven en el orden del catálogo."""
    names = [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    problems = []
    if not names:
        problems.append("la selección está vacía")
    problems += [f"nombre repetido: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    by_name = {m.name: m for m in mutants}
    problems += [f"no existe en el catálogo: {n}" for n in dict.fromkeys(names) if n not in by_name]
    problems += [
        f"no es del grupo {NAMES_FILE_GROUP}: {n}"
        for n in dict.fromkeys(names)
        if n in by_name and by_name[n].group != NAMES_FILE_GROUP
    ]
    wanted = set(names)
    return ([m for m in mutants if m.name in wanted] if not problems else []), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", dest="keyword", help="solo mutantes cuyo nombre contenga el texto")
    parser.add_argument("--group", help="solo mutantes de ese grupo (historical, P1)")
    parser.add_argument(
        "--names-file",
        help="solo los mutantes históricos con esos nombres exactos (uno por línea); "
        "no se combina con -k ni --group",
    )
    parser.add_argument("--list", action="store_true", help="lista los mutantes y termina")
    parser.add_argument(
        "--check-fragments", action="store_true",
        help="verifica el catálogo (fragmentos, compilación, tests) sin pytest ni mutaciones",
    )  # fmt: skip
    parser.add_argument(
        "--all-failures", action="store_true", help="sin -x: reporta todos los tests que fallan"
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="solo verifica la línea base en la copia"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="s por mutante")
    parser.add_argument("--baseline-timeout", type=float, default=DEFAULT_BASELINE_TIMEOUT_S)
    parser.add_argument("--results", help="archivo JSONL de resultados, fuera del repositorio")
    args = parser.parse_args()

    names_selection: dict[str, Any] | None = None
    if args.names_file:
        if args.keyword or args.group:
            print("ABORT: --names-file no se combina con -k ni --group")
            return 2
        try:
            raw = Path(args.names_file).read_bytes()
        except OSError as exc:
            print(f"ABORT: no se puede leer --names-file ({type(exc).__name__})")
            return 2
        mutants, selection_problems = select_by_names(raw.decode("utf-8"), MUTANTS)
        if selection_problems:
            print("ABORT: selección inválida")
            print("\n".join(f"  {p}" for p in selection_problems))
            return 2
        names_selection = {
            "names_file": str(Path(args.names_file).resolve()),
            "names_sha256": hashlib.sha256(raw).hexdigest(),
            "names": [m.name for m in mutants],
        }
        print(f"Selección: {len(mutants)} mutantes de {args.names_file}")
    else:
        mutants = [
            m
            for m in MUTANTS
            if (not args.keyword or args.keyword in m.name)
            and (not args.group or m.group == args.group)
        ]
    if args.list:
        for mutant in mutants:
            scope = ", ".join(mutant.tests) if mutant.tests else f"TESTS ({len(TESTS)} archivos)"
            print(f"- [{mutant.group}] {mutant.name}  ({mutant.path})\n    tests: {scope}")
        return 0
    problems = catalog_problems(ROOT, mutants)
    if problems:
        print("ABORT: catálogo inválido")
        print("\n".join(f"  {p}" for p in problems))
        return 2
    if args.check_fragments:
        print(f"Catálogo OK: {len(mutants)} mutantes, fragmentos únicos y compilables")
        return 0

    results = Results(_results_path(args.results))
    baseline_tests = list(dict.fromkeys(t for m in mutants for t in tests_for(m)))
    results.write({
        "record": "run", "started_at": _now(), **git_state(ROOT),
        "python": platform.python_version(),
        "selection": {"keyword": args.keyword, "group": args.group, "mutants": len(mutants),
                      **(names_selection or {})},
        "tests": TESTS, "baseline_tests": baseline_tests, "stop_first": not args.all_failures,
        "timeout_s": args.timeout, "baseline_timeout_s": args.baseline_timeout,
    })  # fmt: skip
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="arl-mutation-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=COPY_IGNORE)
        check_isolation(copy)

        baseline = run_pytest(copy, baseline_tests, stop_first=False, timeout=args.baseline_timeout)
        summary = baseline.stdout.strip().splitlines()[-1] if baseline.stdout.strip() else ""
        results.write({"record": "baseline", "returncode": baseline.returncode,
                       "timed_out": baseline.timed_out, "duration_s": baseline.duration_s,
                       "summary": summary})  # fmt: skip
        if baseline.returncode != 0:
            reason = "superó el timeout" if baseline.timed_out else f"no pasa ({summary})"
            print(f"ABORT: la suite sin mutaciones {reason}")
            print("\n".join(failed_tests(baseline.stdout)))
            details = [ln for ln in baseline.stdout.splitlines() if ln.startswith("E ")]
            print("\n".join(details[:40]))
            return 2
        print(f"Línea base OK: {summary} ({baseline.duration_s} s)\n")
        if args.baseline_only:
            return 0

        counts = {KILLED: 0, SURVIVED: 0, TIMEOUT: 0, ERROR: 0}
        flagged: list[tuple[str, str]] = []
        for mutant in mutants:
            tests = tests_for(mutant)
            try:
                with mutated(copy / mutant.path, mutant):
                    run = run_pytest(copy, tests, not args.all_failures, timeout=args.timeout)
            except RestoreError as exc:
                print(f"ABORT: {exc}")
                return 2
            killers = failed_tests(run.stdout)
            status = classify(run, killers)
            counts[status] += 1
            results.write({
                "record": "mutant", "name": mutant.name, "group": mutant.group,
                "path": mutant.path, "status": status, "returncode": run.returncode,
                "duration_s": run.duration_s, "tests": tests, "killers": killers,
            })  # fmt: skip
            print(f"{status:<8} {mutant.name}  ({run.duration_s} s)")
            for test in killers[:4]:
                print(f"           - {test}")
            if status != KILLED:
                flagged.append((status, mutant.name))

    elapsed = round(time.monotonic() - started, 1)
    results.write({"record": "summary", "finished_at": _now(), "duration_s": elapsed, **counts})
    print(f"\n{counts[KILLED]}/{len(mutants)} mutantes detectados ({elapsed} s)")
    print(", ".join(f"{k}: {v}" for k, v in counts.items()))
    for status, name in flagged:
        print(f"  {status.lower()}: {name}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
