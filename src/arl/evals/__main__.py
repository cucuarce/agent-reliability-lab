"""``python -m arl.evals``: ejecuta la suite de evaluación determinística.

    python -m arl.evals [--cases evals/cases] [--out evals/reports] [--only ID ...] [--no-write]

Exit: 0 si todos los casos pasan; 1 si alguno falla, es no evaluable o tuvo un error de
infraestructura; 2 si los casos son inválidos. El resumen se imprime siempre.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from arl.config import ConfigError
from arl.evals.cases import InvalidCase, load_cases
from arl.evals.suite import render, run, write_report
from arl.tracing.logging_setup import LOG_LEVEL_ENV, configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arl.evals", description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evals/cases"))
    parser.add_argument("--out", type=Path, default=Path("evals/reports"))
    parser.add_argument("--only", nargs="+", help="ids de casos a ejecutar")
    parser.add_argument("--no-write", action="store_true", help="no escribe el reporte")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    # Los fallos inyectados producen reintentos esperados: solo errores a stderr, con el mismo
    # logging seguro del servidor (redacción y sin el DEBUG de aiosqlite).
    configure_logging(os.environ.get(LOG_LEVEL_ENV, "ERROR"))
    try:
        cases = load_cases(args.cases)
        if args.only:
            unknown = set(args.only) - {c.case.id for c in cases}
            if unknown:
                raise InvalidCase(f"casos inexistentes: {sorted(unknown)}")
            cases = [c for c in cases if c.case.id in args.only]
        report = run(cases, Path.cwd())
    except (InvalidCase, ConfigError) as exc:
        print(f"arl.evals: {exc}", file=sys.stderr)
        return 2
    print(render(report))
    if not args.no_write:
        print(f"reporte: {write_report(report, args.out)}")
    counts = report["metrics"]["cases"]
    return 0 if counts["passed"] == counts["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
