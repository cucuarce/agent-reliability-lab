"""``python -m arl.tracing``: consulta de trazas por línea de comandos (solo lectura).

    python -m arl.tracing runs [--dir D] [--limit N]
    python -m arl.tracing show <run_id | archivo.jsonl | reporte.json> [--dir D] [--json]
    python -m arl.tracing thread <thread_id> [--dir D] [--json]
    python -m arl.tracing schema

``--dir`` por defecto: ``<ARL_DATA_DIR>/traces``. Exit: 0 ok, 2 sin trazas o entrada inválida.
Nunca modifica trazas, checkpoints ni la base: solo lee, y no importa nada que pueda escribir.

Las entradas son datos no confiables: un ``run_id`` debe tener la forma exacta de un id
(16 hexadecimales) y de cualquier archivo solo se muestra lo que pasa el esquema ``arl.trace/1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from arl.config import ConfigError, resolve_data_dir
from arl.tracing.events import json_schema
from arl.tracing.query import (
    TraceNotFound,
    list_runs,
    load_report,
    render,
    report_for,
    run_path,
    thread_paths,
)


def default_trace_dir() -> Path:
    return resolve_data_dir() / "traces"


def _report_for(target: str, trace_dir: Path | None) -> dict[str, Any]:
    path = Path(target)
    if path.suffix == ".json" and path.is_file():  # reporte exportado antes con --json
        report = load_report(path)
        if report["integrity"]["report_mismatch"]:
            print("arl.tracing: aviso: el reporte no coincide con sus eventos; se muestra la "
                  "reconstrucción", file=sys.stderr)  # fmt: skip
        return report
    if path.suffix == ".jsonl" and path.is_file():
        return report_for([path])
    return report_for([run_path(trace_dir or default_trace_dir(), target)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arl.tracing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runs = sub.add_parser("runs", help="lista las ejecuciones registradas")
    runs.add_argument("--dir", type=Path)
    runs.add_argument("--limit", type=int, default=20)
    show = sub.add_parser("show", help="reconstruye una ejecución")
    show.add_argument("target", help="run_id, archivo .jsonl de una ejecución o reporte .json")
    show.add_argument("--dir", type=Path)
    show.add_argument("--json", action="store_true", help="exporta el reporte completo en JSON")
    thread = sub.add_parser("thread", help="reconstruye todas las ejecuciones de un thread")
    thread.add_argument("thread_id")
    thread.add_argument("--dir", type=Path)
    thread.add_argument("--json", action="store_true")
    sub.add_parser("schema", help="JSON Schema de los eventos (arl.trace/1)")
    args = parser.parse_args(argv)
    # Salida UTF-8 siempre: la consola de Windows (cp1252) no puede codificar todo el texto y la
    # salida suele leerse por un pipe (p. ej. la Skill analyze-run).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        if args.command == "schema":
            print(json.dumps(json_schema(), ensure_ascii=False, indent=2))
            return 0
        trace_dir = args.dir or default_trace_dir()
        if args.command == "runs":
            for row in list_runs(trace_dir)[: args.limit]:
                status = row["status"] or "INCOMPLETA"
                print(f"{row['run_id']}  {row['started_at']}  {row['thread_id']:<20} "
                      f"{row['operation']:<22} {status}")  # fmt: skip
            return 0
        if args.command == "show":
            report = _report_for(args.target, args.dir)
        else:
            report = report_for(thread_paths(trace_dir, args.thread_id))
    except (TraceNotFound, ConfigError) as exc:
        print(f"arl.tracing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
