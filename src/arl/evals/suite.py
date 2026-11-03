"""Ejecución de la suite y reportes (JSON + resumen legible).

Reglas de los reportes:

- Cada ejecución escribe en un directorio **nuevo** (``<UTC>-<commit>``); si existe, falla. Nunca
  sobrescribe resultados históricos ni escribe en ``evals/cases``.
- Los entornos de los casos viven en un directorio temporal fuera de OneDrive y se borran al
  terminar; las bases de desarrollo (``ARL_DATA_DIR``) nunca se abren.
- Un error de infraestructura en un caso no detiene la suite: el caso queda como ``error``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anyio

from arl import __version__
from arl.config import ConfigError, _is_synced_folder
from arl.evals import graders, metrics
from arl.evals.cases import CaseFile, suite_fingerprint
from arl.evals.runner import EVAL_LIMITS, MODEL_KIND, InfraError, run_case, scope_for

# Versión de la suite según el esquema de sus casos: v1 (6A, sin alcance) y v2 (6A.1).
SUITE_VERSIONS = {"arl.eval-case/1": "6a.1", "arl.eval-case/2": "6a1.1"}
REPORT_SCHEMA = "arl.eval-report/1"
SAFETY_CAVEAT = (
    "> 0 violaciones de garantías implementadas **no** significa ausencia de riesgo: solo cubre "
    "estos escenarios simulados y las garantías que existen. Un ataque ejecutado con aprobación "
    "humana no viola ninguna (ver `request_scope`)."
)


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or "unknown", "dirty": None if status is None else bool(status)}


def _scratch_root() -> Path:
    root = Path(tempfile.gettempdir()).resolve()
    if _is_synced_folder(root):
        raise ConfigError(f"el directorio temporal {root} está en OneDrive")
    return root


async def _evaluate(case_file: CaseFile, workdir: Path, enforce_scope: bool) -> dict[str, Any]:
    case = case_file.case
    base = {
        "case_id": case.id,
        "category": case.category.value,
        "path": case_file.path,
        "case_sha256": case_file.sha256,
        "covers_tests": list(case.covers_tests),
        "scope_mode": scope_for(case, enforce=enforce_scope).kind,  # el mismo que usa run_case
    }
    start = time.perf_counter()
    try:
        obs = await run_case(case, workdir, enforce_scope=enforce_scope)
    except InfraError as exc:
        return base | {
            "verdict": "error",
            "error": str(exc),
            "graders": [],
            "attack": None,
            "findings": [],
            "duration_ms": _ms(start),
        }
    results, attack = graders.grade(case, obs)
    findings = [g.to_dict() for g in results if g.kind == "proposed" and g.status == "fail"]
    return base | {
        "verdict": graders.verdict(results),
        "graders": [g.to_dict() for g in results],
        "attack": attack,
        "findings": findings,
        "scope_stats": graders.scope_stats(case, obs),
        "observation": _summary(obs),
        "duration_ms": _ms(start),
    }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _summary(obs) -> dict[str, Any]:
    """Lo observado, sin textos libres del modelo ni de las tareas."""
    return {
        "statuses": obs.statuses,
        "final_status": obs.final_status,
        "raised": obs.raised,
        "approval_requests": [
            {k: r[k] for k in ("task_id", "changes", "decision")} for r in obs.approval_requests
        ],
        "model_tool_calls": [
            {k: c[k] for k in ("tool", "task_id", "project_id")} for c in obs.model_tool_calls
        ],
        "blocked": graders.observed_blocks(obs),
        "audit": [{k: a[k] for k in ("actor_id", "task_id", "changes")} for a in obs.audit],
        "ledger_entries": len(obs.ledger),
        "server_deliveries": [d["tool"] for d in obs.deliveries],
        "agent_attempts": obs.attempts,
        "injection_followed": obs.injection_followed,
        "followups": obs.followups,
        "trace": None
        if obs.trace is None
        else {
            "final_status": obs.trace["final_status"],
            "integrity_ok": obs.trace["integrity"]["ok"],
            "runs": len(obs.trace["runs"]),
            "retries": obs.trace["counts"].get("retries", 0),
            "write_states": [w["state"] for w in obs.trace["writes"]],
        },
        "trace_error": obs.trace_error,
    }


async def run_suite(
    cases: list[CaseFile], root: Path, *, enforce_scope: bool = True
) -> dict[str, Any]:
    started = datetime.now(UTC)
    results = []
    with tempfile.TemporaryDirectory(prefix="arl-evals-", dir=_scratch_root()) as scratch:
        for case_file in cases:
            workdir = Path(scratch) / case_file.case.id
            workdir.mkdir()
            results.append(await _evaluate(case_file, workdir, enforce_scope))
    modes = sorted({r["scope_mode"] for r in results})
    return {
        "schema": REPORT_SCHEMA,
        "metadata": {
            "suite_version": ", ".join(sorted({SUITE_VERSIONS[c.case.schema_] for c in cases})),
            # True solo si todos los casos corrieron con un alcance explícito. Los casos v1 corren
            # siempre con el histórico, y --without-scope-enforcement aplica el histórico a los v2.
            "scope_enforcement": modes == ["explicit"],
            "scope_modes": modes,
            "case_schema": ", ".join(sorted({c.case.schema_ for c in cases})),
            "suite_fingerprint": suite_fingerprint(cases),
            **git_state(root),
            "arl_version": __version__,
            "model": MODEL_KIND,
            "human": "simulado, política por caso: approve/reject/approve_in_scope/pending",
            "judge": "graders determinísticos (sin LLM-as-judge)",
            "config": {
                "limits": EVAL_LIMITS.model_dump(),
                "transport": "MCP en memoria (servidor y agente reales)",
                "python": platform.python_version(),
                "langgraph": version("langgraph"),
                "mcp": version("mcp"),
            },
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "data": "sintéticos (arl/storage/seed_data.json + seed por caso)",
        },
        "metrics": metrics.compute(results),
        "results": results,
        "findings": [{"case_id": r["case_id"], **f} for r in results for f in r["findings"]],
    }


def run(cases: list[CaseFile], root: Path, *, enforce_scope: bool = True) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        return await run_suite(cases, root, enforce_scope=enforce_scope)

    return anyio.run(main)


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    """Directorio nuevo por ejecución; nunca sobrescribe."""
    meta = report["metadata"]
    stamp = meta["started_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    target = out_dir / f"{stamp}-{meta['commit'][:8]}"
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "summary.md").write_text(render(report), encoding="utf-8")
    return target


def _rate(m: dict[str, Any]) -> str:
    rate = "n/a" if m["rate"] is None else f"{m['rate'] * 100:.1f} %"
    return f"{m['numerator']}/{m['denominator']} ({rate})"


def render(report: dict[str, Any]) -> str:
    """Resumen legible. Tolera reportes incompletos (claves faltantes se muestran como '?')."""
    meta = report.get("metadata", {})
    m = report.get("metrics", {})
    cases = m.get("cases", {})
    dirty = {True: " (con cambios sin commit)", False: "", None: " (estado git desconocido)"}
    fingerprint = str(meta.get("suite_fingerprint", "?"))[:12]
    lines = [
        "# Evaluación determinística (etapa 6A)",
        "",
        f"- Suite: {meta.get('suite_version', '?')} · huella `{fingerprint}`",
        f"- Commit: `{str(meta.get('commit', '?'))[:12]}`{dirty.get(meta.get('dirty'), '')}",
        f"- Modelo: {meta.get('model', '?')} · Humano: {meta.get('human', '?')}",
        f"- Juez: {meta.get('judge', '?')} · Datos: {meta.get('data', '?')}",
        "",
        "## Casos por categoría",
        "",
        "| Categoría | Aprobados | Fallidos | No evaluables | Error infra |",
        "|---|---|---|---|---|",
    ]
    for category, n in cases.get("by_category", {}).items():
        lines.append(
            f"| {category} | {n.get('passed', 0)} | {n.get('failed', 0)} | "
            f"{n.get('not_evaluable', 0)} | {n.get('error', 0)} |"
        )
    lines.append(
        f"| **total** | {cases.get('passed', 0)} | {cases.get('failed', 0)} | "
        f"{cases.get('not_evaluable', 0)} | {cases.get('error', 0)} |"
    )
    for title, group in (
        ("Métricas funcionales", "functional"),
        ("Métricas de seguridad", "safety"),
        ("Alcance del turno (casos v2)", "scope"),
        ("Garantías propuestas (no implementadas)", "proposed"),
    ):
        lines += ["", f"## {title}", "", "| Métrica | Valor | Definición |", "|---|---|---|"]
        for name, metric in m.get(group, {}).items():
            lines.append(f"| `{name}` | {_rate(metric)} | {metric['definition']} |")
        if group == "safety":
            lines += ["", SAFETY_CAVEAT]
    lines += ["", "## Casos que no pasaron", ""]
    problems = [r for r in report.get("results", []) if r.get("verdict") != "passed"]
    for r in problems:
        why = r.get("error") or "; ".join(
            f"{g['name']}: {g['detail'] or g['status']}"
            for g in r.get("graders", [])
            if g["status"] != "pass" and g["kind"] != "proposed"
        )
        lines.append(f"- `{r.get('case_id', '?')}` ({r.get('verdict', '?')}): {why}")
    if not problems:
        lines.append("- ninguno")
    lines += ["", "## Hallazgos (garantías no implementadas)", ""]
    for f in report.get("findings", []):
        lines.append(f"- `{f['case_id']}` · {f['name']}: {f['detail']}")
    if not report.get("findings"):
        lines.append("- ninguno")
    return "\n".join(lines) + "\n"
