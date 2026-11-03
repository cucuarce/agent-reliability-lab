"""Consulta de trazas: reconstruye una ejecución (o un thread completo) desde sus eventos.

El reporte (``arl.run-report/1``) es JSON estable para evaluaciones futuras. Reglas que nunca se
rompen al interpretar:

- Una escritura aprobada sin ``write_result`` es **incierta**, nunca "fallida": pudo haberse
  confirmado en el servidor. La evidencia definitiva está en el ``audit_log`` (por
  ``idempotency_key``), no en las trazas.
- Una ejecución sin ``run_finished`` quedó **incompleta** (el proceso terminó a mitad de camino):
  su estado final se desconoce.
- La ausencia de un evento nunca es evidencia: los estados que solo se deducen de lo que falta
  llevan ``basis: "absence"``. Duplicados, conflictos, huecos de ``seq``, versiones desconocidas y
  mezclas de threads se informan en ``integrity``; los eventos en conflicto se descartan.
- Los archivos son datos no confiables: solo se usa lo que pasa el esquema ``arl.trace/1``. Un
  reporte exportado se reconstruye desde sus eventos validados; el resto del archivo se ignora.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arl.tracing.events import SUPPORTED_VERSIONS, validate_event
from arl.tracing.recorder import run_file

REPORT_SCHEMA = "arl.run-report/3"
RUN_ID = re.compile(r"^[0-9a-f]{16}$")
MAX_FILE_BYTES = 64 * 1024 * 1024  # una ejecución ocupa KB: más es un archivo ajeno


class TraceNotFound(LookupError):
    """No hay trazas para el identificador o la ruta indicada, o el archivo no es legible."""


@dataclass
class Scan:
    """Lo leído de uno o más archivos: eventos válidos y líneas descartadas, por motivo."""

    events: list[dict[str, Any]] = field(default_factory=list)
    invalid: int = 0  # no es UTF-8, no es JSON o no cumple el esquema (p. ej. cortada)
    unsupported: int = 0  # sobre de otra versión de arl.trace


def _read_bytes(path: Path) -> bytes:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise TraceNotFound(f"{path.name}: supera {MAX_FILE_BYTES} bytes")
        return path.read_bytes()
    except OSError as exc:
        failure = TraceNotFound(f"no se pudo leer {path.name} ({type(exc).__name__})")
    raise failure


def scan_paths(paths: list[Path]) -> Scan:
    scan = Scan()
    for path in paths:
        for line in _read_bytes(path).splitlines():
            if line.strip():
                _scan_line(line, scan)
    return scan


def _scan_line(line: bytes, scan: Scan) -> None:
    try:
        raw = json.loads(line.decode("utf-8"))
    except ValueError:  # también UnicodeDecodeError
        scan.invalid += 1
        return
    version = raw.get("v") if isinstance(raw, dict) else None
    if isinstance(version, int) and version not in SUPPORTED_VERSIONS:
        scan.unsupported += 1
        return
    try:
        scan.events.append(validate_event(raw).model_dump())
    except (ValueError, ValidationError):
        scan.invalid += 1


def load_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Eventos válidos del archivo y cantidad de líneas descartadas (inválidas o de otra
    versión; ``scan_paths`` las distingue)."""
    scan = scan_paths([path])
    return scan.events, scan.invalid + scan.unsupported


def run_path(trace_dir: Path, run_id: str) -> Path:
    matches = sorted(trace_dir.glob(f"thread-*/{run_id}.jsonl")) if RUN_ID.fullmatch(run_id) else []
    if not matches:
        raise TraceNotFound(f"no hay trazas de la ejecución '{run_id}' en {trace_dir}")
    return matches[0]


def thread_paths(trace_dir: Path, thread_id: str) -> list[Path]:
    folder = run_file(trace_dir, thread_id, "x").parent
    paths = sorted(folder.glob("*.jsonl"))
    if not paths:
        raise TraceNotFound(f"no hay trazas del thread '{thread_id}' en {trace_dir}")
    return paths


def load_paths(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    scan = scan_paths(paths)
    return scan.events, scan.invalid + scan.unsupported


def report_for(paths: list[Path]) -> dict[str, Any]:
    scan = scan_paths(paths)
    return reconstruct(scan.events, scan.invalid, unsupported_lines=scan.unsupported)


def load_report(path: Path) -> dict[str, Any]:
    """Reporte exportado con ``--json``. Es un dato no confiable: se reconstruye desde sus
    eventos (validados de nuevo) y nunca se muestra lo que trae calculado. Si lo calculado no
    coincide con la reconstrucción, ``integrity.report_mismatch`` lo indica."""
    invalid = TraceNotFound(f"{path.name} no es un reporte {REPORT_SCHEMA} válido")
    try:
        report = json.loads(_read_bytes(path).decode("utf-8"))
    except ValueError:  # también UnicodeDecodeError
        report = None
    source = report.get("source") if isinstance(report, dict) else None
    if not isinstance(source, dict) or report.get("schema") != REPORT_SCHEMA:  # type: ignore[union-attr]
        raise invalid
    counts = [source.get("invalid_lines"), source.get("unsupported_version_lines")]
    if not isinstance(source.get("events"), list) or not all(
        type(n) is int and n >= 0 for n in counts
    ):
        raise invalid
    events = []
    for raw in source["events"]:
        try:
            events.append(validate_event(raw).model_dump())
        except (ValueError, ValidationError):
            raise invalid from None
    rebuilt = reconstruct(events, counts[0], unsupported_lines=counts[1])
    if rebuilt != report:
        rebuilt["integrity"].update(report_mismatch=True, ok=False)
    return rebuilt


def list_runs(trace_dir: Path) -> list[dict[str, Any]]:
    """Una fila por ejecución, de la más reciente a la más antigua."""
    rows = []
    for path in trace_dir.glob("thread-*/*.jsonl"):
        events, _ = load_events(path)
        if events:
            rows.append(_run_summary(events))
    return sorted(rows, key=lambda r: r["started_at"], reverse=True)


# --- Reconstrucción --------------------------------------------------------------------------


def _ts(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(event["ts"])


def _run_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    events = sorted(events, key=lambda e: e["seq"])
    first = events[0]
    started = next((e for e in events if e["event"] == "run_started"), first)
    finished = next((e for e in events if e["event"] == "run_finished"), None)
    data = finished["data"] if finished else {}
    return {
        "run_id": first["run_id"],
        "thread_id": first["thread_id"],
        "principal_id": first["principal_id"],
        "operation": started["data"].get("operation"),
        "started_at": first["ts"],
        "complete": finished is not None,
        "status": data.get("status"),  # None: incompleta, estado desconocido
        "duration_ms": data.get("duration_ms"),
        "refusal": data.get("refusal"),
        "failure": data.get("failure"),
        "counts": data.get("counts"),
        "events": len(events),
    }


def _summary_line(event: dict[str, Any]) -> str:
    d = event["data"]
    kind = event["event"]
    if kind == "run_started":
        return d["operation"]
    if kind == "recovery_started":
        action = f" action={d['action_id']}" if d.get("action_id") else ""
        return f"prior={d['prior_status']} node={d['resume_node']}{action}"
    if kind == "history_closed":
        return f"{d['reason']}: {', '.join(d['tool_call_ids'])}"
    if kind == "model_call":
        calls = ", ".join(f"{c['tool']}({c['tool_call_id']})" for c in d.get("tool_calls", []))
        return f"turn={d['model_turn']} {d['outcome']} {d['duration_ms']}ms {calls}".rstrip()
    if kind == "tool_call_routed":
        target = d.get("task_id") or d.get("project_id") or ""
        why = f" ({d['scope_reason']})" if d.get("scope_reason") else ""
        return f"{d['tool']}({d['tool_call_id']}) -> {d['decision']}{why} {target}".rstrip()
    if kind == "scope_reservation":
        return f"action={d['action_id']} {d['task_id']} grant={d['grant_id']} {d['state']}"
    if kind == "approval_requested":
        return (
            f"action={d['action_id']} {d['task_id']} fields={','.join(d['changed_fields'])} "
            f"v{d['expected_version']} role={d['my_role']}"
        )
    if kind == "approval_decided":
        reason = f" ({d['invalid_reason']})" if d.get("invalid_reason") else ""
        return f"action={d['action_id']} {d['decision']}{reason}"
    if kind == "mcp_call":
        detail = d["outcome"]
        if d.get("error_type"):
            detail += f" ({d['error_type']})"
        elif d.get("code"):
            detail += f" [{d['code']}]"
        if d.get("replayed"):
            detail += " replayed"
        where = f" {d['node']}#{d['attempt']}" if d.get("node") else ""
        return f"{d['tool']}{where} {detail} {d['duration_ms']}ms"
    if kind == "write_result":
        replay = " replayed" if d.get("replayed") else ""
        return f"action={d['action_id']} {d['task_id']} {d['outcome']}{replay} v{d.get('version')}"
    if kind == "write_uncertain":
        return f"action={d['action_id']} key={d['idempotency_key']} (resultado desconocido)"
    if kind == "run_finished":
        extra = d.get("refusal") or (d.get("failure") or {}).get("code") or ""
        return f"{d['status']} {d['duration_ms']}ms {extra}".rstrip()
    return ""


def _retries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        d = e["data"]
        if e["event"] == "mcp_call" and d.get("node"):
            groups[(e["run_id"], d["node"], d["tool_call_id"])].append(d)
    rows = []
    for (run_id, node, tool_call_id), calls in groups.items():
        attempts = max(c["attempt"] for c in calls)
        if attempts > 1:
            rows.append(
                {
                    "run_id": run_id,
                    "node": node,
                    "tool_call_id": tool_call_id,
                    "attempts": attempts,
                    "outcomes": [f"#{c['attempt']} {c['tool']}:{c['outcome']}" for c in calls],
                }
            )
    return rows


def _errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        d = e["data"]
        problem = None
        if e["event"] == "mcp_call" and d["outcome"] != "ok":
            problem = {
                "operation": d["tool"],
                "code": d.get("code"),
                "error_type": d.get("error_type"),
            }
        elif e["event"] == "model_call" and d["outcome"] == "error":
            problem = {
                "operation": "model",
                "code": "model_error",
                "error_type": d.get("error_type"),
            }
        elif e["event"] == "approval_decided" and d["decision"] == "invalid":
            problem = {
                "operation": "approval",
                "code": "invalid_decision",
                "error_type": None,
                "detail": d.get("invalid_reason"),
            }
        elif e["event"] == "run_finished" and (d.get("failure") or d.get("refusal")):
            failure = d.get("failure") or {}
            problem = {
                "operation": failure.get("operation", d["operation"]),
                "code": failure.get("code", d["status"]),
                "error_type": failure.get("error_type", d.get("refusal")),
            }
        if problem:
            rows.append({"run_id": e["run_id"], "seq": e["seq"], "event": e["event"], **problem})
    return rows


def _writes(events: list[dict[str, Any]], last_run_complete: bool) -> list[dict[str, Any]]:
    """Por tool call de escritura: solicitada -> aprobada -> ejecutada, con su evidencia."""
    writes: dict[str, dict[str, Any]] = {}

    def entry(tool_call_id: str) -> dict[str, Any]:
        return writes.setdefault(
            tool_call_id,
            {
                "tool_call_id": tool_call_id,
                "requested": None,
                "action": None,
                "decisions": [],
                "attempts": [],
                "result": None,
                "uncertain_runs": [],
            },
        )

    write_events = {"approval_requested", "approval_decided", "write_result", "write_uncertain"}
    for e in events:
        d = e["data"]
        kind = e["event"]
        if kind == "tool_call_routed" and d["decision"] == "requires_approval":
            entry(d["tool_call_id"])["requested"] = {
                "task_id": d.get("task_id"),
                "changed_fields": d.get("changed_fields"),
            }
        elif kind == "mcp_call" and d["tool"] == "update_task" and d.get("tool_call_id"):
            entry(d["tool_call_id"])["attempts"].append(
                {
                    "run_id": e["run_id"],
                    "attempt": d.get("attempt"),
                    "outcome": d["outcome"],
                    "idempotency_key": d.get("idempotency_key"),
                }
            )
        elif kind in write_events:
            w = entry(d["tool_call_id"])
            if kind == "approval_requested":
                keys = ("action_id", "task_id", "project_id", "my_role", "changed_fields",
                        "expected_version")  # fmt: skip
                w["action"] = {k: d[k] for k in keys}
            elif kind == "approval_decided":
                w["decisions"].append(d["decision"])
            elif kind == "write_result":
                w["result"] = {
                    k: d.get(k) for k in ("outcome", "replayed", "version", "audit_id")
                } | {"idempotency_key": d["idempotency_key"], "run_id": e["run_id"]}
            else:
                w["uncertain_runs"].append(e["run_id"])

    for w in writes.values():
        w["idempotency_keys"] = sorted({a["idempotency_key"] for a in w["attempts"]} - {None})
        w["state"], w["note"], w["basis"] = _write_state(w, last_run_complete)
    return list(writes.values())


ABSENCE = (
    " (deducido por ausencia de eventos: pueden estar en otra ejecución o haberse perdido; "
    "verificar audit_log)"
)


def _scope(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Alcance del turno: escrituras bloqueadas antes de pedir aprobación y estado final de cada
    reserva (la evidencia de lo escrito sigue siendo el audit_log)."""
    blocked = [
        {"run_id": e["run_id"], "seq": e["seq"], "tool_call_id": e["data"]["tool_call_id"],
         "task_id": e["data"].get("task_id"), "reason": e["data"].get("scope_reason")}
        for e in events
        if e["event"] == "tool_call_routed" and e["data"]["decision"] == "out_of_scope"
    ]  # fmt: skip
    reservations: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "scope_reservation":
            d = e["data"]
            reservations[d["action_id"]] = {k: d[k] for k in ("grant_id", "task_id", "state")}
    kinds = sorted({e["data"]["scope"] for e in events
                    if e["event"] == "run_started" and e["data"].get("scope")})  # fmt: skip
    return {"kinds": kinds, "blocked": blocked, "reservations": reservations}


def _write_state(w: dict[str, Any], last_run_complete: bool) -> tuple[str, str, str]:
    """(estado, nota, basis): ``evidence`` si un evento lo prueba, ``absence`` si solo se deduce
    de eventos que faltan."""
    if w["result"] is not None:
        outcome = w["result"]["outcome"]
        if outcome in ("updated", "unchanged"):
            how = " (replay: ya estaba confirmada)" if w["result"]["replayed"] else ""
            return "executed", f"el servidor confirmó '{outcome}'{how}", "evidence"
        return "refused_by_server", f"el servidor respondió [{outcome}]: no se aplicó", "evidence"
    # Se envió al servidor (intentos o write_uncertain) aunque falte la decisión en las trazas.
    if "approve" in w["decisions"] or w["attempts"] or w["uncertain_runs"]:
        why = "sin write_result" + ("" if last_run_complete else "; la última ejecución no terminó")
        if "approve" not in w["decisions"]:
            why += "; la aprobación no figura en estas trazas"
        note = (
            f"aprobada, resultado desconocido ({why}). No es un fallo: verificar audit_log por "
            "idempotency_key o continuar con la misma clave"
        )
        return "uncertain", note, "evidence"
    if "reject" in w["decisions"]:
        return "rejected", "rechazada por el humano: nunca se envió al servidor", "evidence"
    if w["action"] is not None:
        return "awaiting_approval", "preparada y sellada, sin decisión" + ABSENCE, "absence"
    return "not_prepared", "solicitada por el modelo, sin preparación" + ABSENCE, "absence"


def _deduplicate(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Un evento por (run_id, seq). Las copias idénticas se cuentan; si dos versiones difieren,
    ninguna es evidencia y se descartan las dos."""
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    conflicts: set[tuple[str, int]] = set()
    duplicates = 0
    for e in events:
        key = (e["run_id"], e["seq"])
        if key not in unique:
            unique[key] = e
        elif unique[key] == e:
            duplicates += 1
        else:
            conflicts.add(key)
    kept = [e for key, e in unique.items() if key not in conflicts]
    return kept, duplicates, [{"run_id": r, "seq": n} for r, n in sorted(conflicts)]


def _seq_gaps(by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """``seq`` que faltan en cada ejecución: eventos descartados al emitir, perdidos o editados."""
    gaps = []
    for run_id, evs in sorted(by_run.items()):
        seqs = {e["seq"] for e in evs}
        missing = sorted(set(range(1, max(seqs) + 1)) - seqs)
        if missing:
            gaps.append({"run_id": run_id, "missing": missing[:50]})
    return gaps


def reconstruct(
    events: list[dict[str, Any]], invalid_lines: int = 0, *, unsupported_lines: int = 0
) -> dict[str, Any]:
    """Reporte de una o más ejecuciones (un run o un thread completo)."""
    source = {  # orden canónico: el reporte no depende del orden de las líneas
        "events": sorted(events, key=lambda e: (e["run_id"], e["seq"])),
        "invalid_lines": invalid_lines,
        "unsupported_version_lines": unsupported_lines,
    }
    events, duplicates, conflicts = _deduplicate(events)
    if not events:
        raise TraceNotFound(
            f"no hay eventos válidos para reconstruir (líneas inválidas: {invalid_lines}; de "
            f"una versión no soportada: {unsupported_lines}; en conflicto: {len(conflicts)})"
        )
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_run[e["run_id"]].append(e)
    runs = sorted((_run_summary(evs) for evs in by_run.values()), key=lambda r: r["started_at"])
    order = {r["run_id"]: i for i, r in enumerate(runs)}
    ordered = sorted(events, key=lambda e: (order[e["run_id"]], e["seq"]))
    origin = _ts(ordered[0])
    # El estado del thread lo da la última ejecución que corrió: un rechazo (busy/refused) no
    # cambia el estado (p. ej. un run_turn rechazado sobre un thread WRITE_UNCERTAIN).
    executed = [r for r in runs if r["status"] not in ("busy", "refused")]
    last = executed[-1] if executed else runs[-1]
    counts: dict[str, int] = defaultdict(int)
    for r in runs:
        for key, value in (r["counts"] or {}).items():
            counts[key] += value
    retries = _retries(ordered)
    counts["retries"] = sum(r["attempts"] - 1 for r in retries)
    thread_ids = sorted({r["thread_id"] for r in runs})
    integrity: dict[str, Any] = {
        "invalid_lines": invalid_lines,
        "unsupported_version_lines": unsupported_lines,
        "duplicate_events": duplicates,
        "conflicting_events": conflicts,
        "seq_gaps": _seq_gaps(by_run),
        "thread_ids": thread_ids,
        "incomplete_runs": [r["run_id"] for r in runs if not r["complete"]],
        "report_mismatch": False,
    }
    integrity["ok"] = not _integrity_problems(integrity)
    return {
        "schema": REPORT_SCHEMA,
        "thread_id": last["thread_id"],
        "principal_id": last["principal_id"],
        "final_status": last["status"],  # None si la última ejecución quedó incompleta
        "complete": last["complete"],
        "duration_ms": round(sum(r["duration_ms"] or 0 for r in runs), 3),
        "runs": runs,
        "counts": dict(counts),
        "retries": retries,
        "errors": _errors(ordered),
        "writes": _writes(ordered, last["complete"]),
        "invalid_lines": invalid_lines,
        "integrity": integrity,
        "scope": _scope(ordered),
        "timeline": [
            {
                "t_ms": round((_ts(e) - origin).total_seconds() * 1000, 3),
                "run_id": e["run_id"],
                "seq": e["seq"],
                "event": e["event"],
                "summary": _summary_line(e),
            }
            for e in ordered
        ],
        "source": source,  # eventos validados: un reporte importado se reconstruye desde acá
    }


def render(report: dict[str, Any]) -> str:
    """Texto legible del reporte (la CLI lo imprime; ``--json`` exporta el reporte entero)."""
    lines = [
        f"thread {report['thread_id']}  principal {report['principal_id']}",
        f"estado final: {_status_text(report)}  duración total: {report['duration_ms']} ms",
        "",
        "ejecuciones:",
    ]
    for r in report["runs"]:
        status = r["status"] or "INCOMPLETA (sin run_finished: el proceso terminó a mitad)"
        lines.append(f"  {r['run_id']}  {r['operation']:<22} {status}  {r['duration_ms']} ms")
    lines += ["", "cronología:"]
    for t in report["timeline"]:
        lines.append(
            f"  +{t['t_ms']:>10.3f} ms  {t['run_id'][:8]} #{t['seq']:<3} "
            f"{t['event']:<18} {t['summary']}"
        )
    if report["retries"]:
        lines += ["", "reintentos:"]
        for r in report["retries"]:
            lines.append(
                f"  {r['node']} {r['tool_call_id']}: {r['attempts']} intentos "
                f"[{', '.join(r['outcomes'])}]"
            )
    if report["errors"]:
        lines += ["", "errores:"]
        for e in report["errors"]:
            lines.append(
                f"  {e['run_id'][:8]} #{e['seq']} {e['event']}: {e['operation']} "
                f"code={e['code']} error_type={e['error_type']}"
            )
    if report["writes"]:
        lines += ["", "escrituras (solicitada -> aprobada -> ejecutada):"]
        for w in report["writes"]:
            keys = ", ".join(w["idempotency_keys"]) or "-"
            lines.append(
                f"  {w['tool_call_id']}: {w['state'].upper()}  decisiones={w['decisions']}"
                f"  intentos={len(w['attempts'])}  claves={keys}"
            )
            lines.append(f"      {w['note']}")
    scope = report.get("scope") or {}
    if scope.get("blocked") or scope.get("reservations"):
        lines += ["", f"alcance del turno ({', '.join(scope['kinds']) or 'sin declarar'}):"]
        for b in scope["blocked"]:
            lines.append(f"  bloqueada antes de aprobar: {b['tool_call_id']} {b['task_id']} "
                         f"({b['reason']})")  # fmt: skip
        for action_id, r in scope["reservations"].items():
            lines.append(f"  reserva action={action_id} {r['task_id']} {r['state']}")
    problems = _integrity_problems(report["integrity"])
    if problems:
        lines += ["", "integridad de las trazas:"] + [f"  {text}" for _, text in problems]
    return "\n".join(lines)


def _integrity_problems(integrity: dict[str, Any]) -> list[tuple[str, str]]:
    """(tipo, detalle) de cada problema de integridad; vacío si las trazas son consistentes."""
    problems = []
    if n := integrity["invalid_lines"]:
        problems.append(("líneas inválidas", f"líneas inválidas ignoradas: {n}"))
    if n := integrity["unsupported_version_lines"]:
        problems.append(("versión no soportada", f"líneas de otra versión ignoradas: {n}"))
    if n := integrity["duplicate_events"]:
        problems.append(("duplicados", f"eventos duplicados ignorados: {n}"))
    for c in integrity["conflicting_events"]:
        detail = f"{c['run_id'][:8]} #{c['seq']} tiene dos versiones distintas: se descartan"
        problems.append(("conflictos", detail))
    for g in integrity["seq_gaps"]:
        problems.append(("huecos de seq", f"{g['run_id'][:8]}: faltan los seq {g['missing']}"))
    if len(integrity["thread_ids"]) > 1:
        detail = f"eventos de varios threads: {', '.join(integrity['thread_ids'])}"
        problems.append(("varios threads", detail))
    if integrity["report_mismatch"]:
        detail = "el reporte importado no coincidía con sus eventos: se muestra la reconstrucción"
        problems.append(("reporte alterado", detail))
    return problems


def _status_text(report: dict[str, Any]) -> str:
    if not report["complete"]:
        text = "DESCONOCIDO (la última ejecución quedó incompleta)"
    else:
        text = str(report["final_status"]).upper()
    kinds = dict.fromkeys(kind for kind, _ in _integrity_problems(report["integrity"]))
    if kinds:
        text += f"  [integridad dudosa: {', '.join(kinds)}]"
    return text
