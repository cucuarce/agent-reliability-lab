"""``python -m arl.evals.live``: evaluación conductual con un LLM real (etapa 6B.1, solo Groq).

    python -m arl.evals.live config [--profile P]        # configuración y credencial (sí/no)
    python -m arl.evals.live dry-run [--only ID ...]     # sin inferencia y sin red
    python -m arl.evals.live models                      # GET /models (no hace inferencia)
    python -m arl.evals.live run --profile first-batch --only L01 L02 --confirm-real-calls

``run`` hace inferencia real. Exige ``--confirm-real-calls``, ``--only`` y un perfil que autorice
esos escenarios (``first-batch``: L01 y L02), no admite cambiar las repeticiones del perfil y
solo acepta escenarios de fase 1 (solo lectura).

La credencial sale de ``GROQ_API_KEY``: del entorno o, si no está definida ahí, del ``.env`` del
directorio actual. Nunca se imprime ni se guarda.

Exit: 0 si todas las corridas pasan; 1 si alguna falla o no es evaluable; 2 ante un error de
uso, configuración o credencial.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from arl.config import ConfigError
from arl.evals.live.cases import ApprovalPolicyMissing, InvalidLiveCase, load_live_cases
from arl.evals.live.providers import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    CredentialSource,
    MissingCredential,
    ProviderError,
    credential_present,
    list_models,
    load_env_file,
)
from arl.evals.live.suite import (
    PROFILES,
    LiveConfig,
    WritesDisabled,
    authorize_real_write,
    run_live_suite,
)
from arl.evals.live.transcript import TranscriptUnavailable, TranscriptWriter
from arl.tracing.logging_setup import LOG_LEVEL_ENV, configure_logging

ALTERNATIVE_MODEL = "openai/gpt-oss-120b"
ENV_FILE = Path(".env")  # relativo al directorio actual: la raíz del repo
_SOURCES = {"environment": "entorno", "env_file": ".env", None: None}


TOKENS_COUNTED_NOTE = (
    "\\* Tokens contados contra el presupuesto: los informados por el proveedor en cada llamada "
    "con uso informado; la cota del peor caso (prompt estimado + salida máxima) en las demás "
    "(dry-run, fallos o uso no informado)."
)
# Solo cambia cómo se muestra: la clave del reporte JSON se conserva (reportes históricos).
_DISPLAY_NAMES = {"tokens_charged_worst_case": "tokens_counted*"}


def _rate(m: dict[str, Any]) -> str:
    rate = "n/a" if m["rate"] is None else f"{m['rate'] * 100:.1f} %"
    return f"{m['numerator']}/{m['denominator']} ({rate})"


def render(report: dict[str, Any]) -> str:
    meta, m = report["metadata"], report["metrics"]
    lines = [
        f"# Evaluación con LLM real — {meta['mode']} ({meta['provider']}: {meta['model']})",
        f"suite {meta['suite_version']} · casos {meta['case_schema']} · commit "
        f"{meta['commit'][:8]}{' (con cambios)' if meta['dirty'] else ''}"
        + (f" · detenida: {meta['stopped']}" if meta["stopped"] else ""),
        "",
        "| escenario | rep | fase | veredicto | clase | causa | llamadas | uso informado | "
        "tokens contados* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        usage = r.get("scenario_usage") or {}
        calls = (r.get("observation") or {}).get("model_calls") or []
        informed = f"{sum(c['usage_reported'] for c in calls)}/{len(calls)}" if usage else ""
        cause = " ".join(
            [r.get("cause") or ""]
            + [
                f"(+{c['origin']}:{c['code']}, {c['attempts']} intentos)"
                for c in r.get("contributing_causes", [])
            ]
        ).strip()
        lines.append(
            f"| {r['case_id']} | {r['repetition']} | {r['phase']} | {r['verdict']} | "
            f"{r['error_class'] or ''} | {cause} | {usage.get('calls', '')} | "
            f"{informed} | {usage.get('tokens_charged', '')} |"
        )
    lines += ["", TOKENS_COUNTED_NOTE, ""]
    if contributing := m.get("runs", {}).get("contributing_causes"):  # desde 6b1.3
        lines.append(f"- runs.contributing_causes: {contributing}")
    groups = ("functional", "safety", "attacks", "scope", "truthfulness", "communication",
              "format", "provider")  # fmt: skip
    for group in (*groups, "usage"):
        if group not in m:  # reportes 6b1.1, anteriores a scope y truthfulness
            continue
        for name, value in m[group].items():
            shown = _rate(value) if isinstance(value, dict) and "rate" in value else value
            lines.append(f"- {group}.{_DISPLAY_NAMES.get(name, name)}: {shown}")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    meta = report["metadata"]
    stamp = meta["started_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    target = out_dir / f"{stamp}-{meta['mode']}-{meta['commit'][:8]}"
    n = 1
    while target.exists():
        n += 1
        target = target.with_name(f"{target.name.split('~')[0]}~{n}")
    target.mkdir(parents=True)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "summary.md").write_text(render(report), encoding="utf-8")
    return target


def _config(args: argparse.Namespace) -> LiveConfig:
    base = PROFILES[args.profile]
    update: dict[str, Any] = {"spec": base.spec.model_copy(update={"model": args.model})}
    if args.repeat is not None:
        update["repeat"] = args.repeat
    return LiveConfig.model_validate(base.model_dump() | update)


def _show_config(config: LiveConfig, source: CredentialSource | None) -> None:
    print(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "credential": {
                    API_KEY_ENV: "definida" if credential_present() else "no definida",
                    "origen": _SOURCES[source],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arl.evals.live",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("config", "dry-run", "run"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, default=Path("evals/live/cases"))
        p.add_argument("--out", type=Path, default=Path("evals/live-reports"))
        p.add_argument("--only", nargs="+")
        p.add_argument("--profile", choices=sorted(PROFILES), default="default")
        p.add_argument("--repeat", type=int)  # por defecto, el del perfil
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--no-write", action="store_true")
        p.add_argument("--no-transcript", action="store_true")  # por defecto, sí (sintéticos)
        if name == "run":
            p.add_argument("--confirm-real-calls", action="store_true")
            p.add_argument("--confirm-real-writes", action="store_true")
    sub.add_parser("models")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    configure_logging(os.environ.get(LOG_LEVEL_ENV, "ERROR"))
    try:
        source = load_env_file(ENV_FILE)  # antes de cualquier ChatGroq o GET /models
    except MissingCredential as exc:
        print(f"arl.evals.live: {exc}", file=sys.stderr)
        return 2

    if args.command == "models":
        return _models()
    config = _config(args)
    if args.command == "config":
        _show_config(config, source)
        return 0
    try:
        cases = load_live_cases(args.cases)
        if args.only:
            unknown = set(args.only) - {c.case.id for c in cases}
            if unknown:
                raise InvalidLiveCase(f"escenarios inexistentes: {sorted(unknown)}")
            cases = [c for c in cases if c.case.id in args.only]
        if args.command == "run":
            _check_real_run(args, cases, config)
        mode = "live" if args.command == "run" else "dry-run"
        writer, unavailable = _transcript_writer(args, mode)
        extra = {"transcript": writer} if writer is not None else {}
        report = anyio.run(lambda: run_live_suite(cases, Path.cwd(), config, mode=mode, **extra))
    except (
        InvalidLiveCase,
        ApprovalPolicyMissing,
        WritesDisabled,
        ConfigError,
        MissingCredential,
        UsageError,
    ) as exc:
        print(f"arl.evals.live: {exc}", file=sys.stderr)
        return 2
    if unavailable is not None:
        report["metadata"]["transcript"] = {"status": "disabled", "reason": unavailable}
    print(render(report))
    if not args.no_write:
        print(f"reporte: {write_report(report, args.out)}")
    if writer is not None:
        print(f"transcript: {writer.path}")
    runs = report["metrics"]["runs"]
    return 0 if runs["passed"] == runs["total"] else 1


class UsageError(ValueError):
    pass


def _transcript_writer(
    args: argparse.Namespace, mode: str
) -> tuple[TranscriptWriter | None, str | None]:
    """La transcripción local, salvo ``--no-write``/``--no-transcript``. Si no hay un lugar
    seguro (fuera de OneDrive y del repo), la corrida sigue sin ella y el reporte lo dice."""
    if args.no_write or args.no_transcript:
        return None, None
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{mode}-{uuid.uuid4().hex[:8]}"
    try:
        return TranscriptWriter.create(run_id, root=Path.cwd()), None
    except TranscriptUnavailable as exc:
        print(f"arl.evals.live: transcripción desactivada: {exc}", file=sys.stderr)
        return None, exc.code


def _check_real_run(args: argparse.Namespace, cases, config: LiveConfig) -> None:
    if not args.only:
        raise UsageError("run exige --only con los escenarios autorizados")
    if not args.confirm_real_calls:
        raise UsageError("run hace inferencia real: agregá --confirm-real-calls")
    writes = [c.case for c in cases if c.case.phase != 1]
    if writes and not args.confirm_real_writes:
        raise UsageError("escrituras con un modelo real: agregá --confirm-real-writes")
    if args.confirm_real_writes and not writes:
        raise UsageError("--confirm-real-writes solo se usa con escenarios de escritura")
    if config.allowed_cases is None:
        raise UsageError(f"el perfil {args.profile!r} no autoriza inferencia real")
    unauthorized = sorted(set(args.only) - set(config.allowed_cases))
    if unauthorized:
        raise UsageError(f"el perfil {args.profile!r} no autoriza {unauthorized}")
    if config.repeat != PROFILES[args.profile].repeat:
        raise UsageError("run no admite cambiar las repeticiones del perfil")
    for case in writes:  # lista cerrada, perfil que la habilita y configuración registrada
        authorize_real_write(case, config)
    if not credential_present():
        raise MissingCredential(f"{API_KEY_ENV} no está definida en el entorno")


def _models() -> int:
    try:
        models = list_models()
    except MissingCredential as exc:
        print(f"arl.evals.live: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"arl.evals.live: {exc} (HTTP {exc.failure.status})", file=sys.stderr)
        return 2
    ids = {m["id"]: m for m in models}
    for wanted in (DEFAULT_MODEL, ALTERNATIVE_MODEL):
        info = ids.get(wanted)
        state = (
            "no disponible"
            if info is None
            else (f"disponible (active={info['active']}, context_window={info['context_window']})")
        )
        print(f"{wanted}: {state}")
    print(f"{len(models)} modelos en la cuenta: " + ", ".join(sorted(ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
