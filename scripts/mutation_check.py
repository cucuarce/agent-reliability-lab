"""Chequeo de mutaciones: ¿los tests fallan ante las regresiones que dicen prevenir?

Cada mutante reintroduce un defecto concreto de identidad, aprobación o tolerancia a fallos. La
suite del agente debe fallar con cada uno ("KILLED"); un mutante "SURVIVED" es un hueco de tests.

Seguridad: nunca toca el árbol de trabajo. Copia el repositorio a un directorio temporal, muta
solo la copia y ejecuta pytest allí con ``PYTHONPATH`` apuntando a la copia (verificado antes de
empezar). Un Ctrl+C o un kill no pueden dejar código mutado en el repo.

Línea base obligatoria: si la suite sin mutar no pasa en la copia, aborta (exit 2), porque un
test roto "mataría" mutantes que no detecta.

Uso:
    uv run python scripts/mutation_check.py            # todos los mutantes (-x: rápido)
    uv run python scripts/mutation_check.py --list
    uv run python scripts/mutation_check.py -k runtime --all-failures

Exit: 0 todos detectados, 1 hay sobrevivientes, 2 abortado (línea base, fragmento inválido o
mutante que no compila).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
]
COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "var", "*.db", "htmlcov"
)

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
RECORDER = "src/arl/tracing/recorder.py"
QUERY = "src/arl/tracing/query.py"
LOGGING = "src/arl/tracing/logging_setup.py"


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str


MUTANTS = [
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


def run_pytest(copy: Path, tests: list[str], stop_first: bool) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(copy / "src")}
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    if stop_first:
        args.insert(4, "-x")
    return subprocess.run(
        args, cwd=copy, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def failed_tests(output: str) -> list[str]:
    names = {
        line[7:].split(" - ")[0].split("::")[-1].split("[")[0]
        for line in output.splitlines()
        if line.startswith(("FAILED ", "ERROR ")) and "::" in line  # no logs capturados
    }
    return sorted(names)


def check_isolation(copy: Path) -> None:
    """Aborta si el paquete importado no es el de la copia (mutar no tendría efecto)."""
    env = {**os.environ, "PYTHONPATH": str(copy / "src")}
    imported = subprocess.run(
        [sys.executable, "-c", "import arl; print(arl.__file__)"],
        cwd=copy, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    if not Path(imported).resolve().is_relative_to(copy.resolve()):
        sys.exit(f"ABORT: se importa {imported}, no la copia en {copy}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", dest="keyword", help="solo mutantes cuyo nombre contenga el texto")
    parser.add_argument("--list", action="store_true", help="lista los mutantes y termina")
    parser.add_argument(
        "--all-failures", action="store_true", help="sin -x: reporta todos los tests que fallan"
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="solo verifica la línea base en la copia"
    )
    args = parser.parse_args()

    mutants = [m for m in MUTANTS if not args.keyword or args.keyword in m.name]
    if args.list:
        for mutant in mutants:
            print(f"- {mutant.name}  ({mutant.path})")
        return 0

    with tempfile.TemporaryDirectory(prefix="arl-mutation-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=COPY_IGNORE)
        check_isolation(copy)

        baseline = run_pytest(copy, TESTS, stop_first=False)
        summary = baseline.stdout.strip().splitlines()[-1] if baseline.stdout.strip() else ""
        if baseline.returncode != 0:
            print(f"ABORT: la suite sin mutaciones no pasa ({summary})")
            print("\n".join(failed_tests(baseline.stdout)))
            details = [ln for ln in baseline.stdout.splitlines() if ln.startswith("E ")]
            print("\n".join(details[:40]))
            return 2
        print(f"Línea base OK: {summary}\n")
        if args.baseline_only:
            return 0

        survivors = []
        for mutant in mutants:
            target = copy / mutant.path
            original = target.read_text(encoding="utf-8")
            if original.count(mutant.old) != 1:
                print(f"ABORT: '{mutant.name}': el fragmento aparece {original.count(mutant.old)} "
                      "veces (¿cambió el código?)")  # fmt: skip
                return 2
            mutated = original.replace(mutant.old, mutant.new)
            try:  # un mutante que no compila "mataría" la suite por otro motivo
                compile(mutated, str(target), "exec")
            except SyntaxError as exc:
                print(f"ABORT: '{mutant.name}' no es Python válido (línea {exc.lineno})")
                return 2
            target.write_text(mutated, encoding="utf-8")
            try:
                run = run_pytest(copy, TESTS, stop_first=not args.all_failures)
            finally:
                target.write_text(original, encoding="utf-8")
            killers = failed_tests(run.stdout)
            status = "KILLED  " if run.returncode != 0 else "SURVIVED"
            print(f"{status} {mutant.name}")
            for test in killers[:4]:
                print(f"           - {test}")
            if run.returncode == 0:
                survivors.append(mutant.name)

    print(f"\n{len(mutants) - len(survivors)}/{len(mutants)} mutantes detectados")
    for name in survivors:
        print(f"  sobrevivió: {name}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
