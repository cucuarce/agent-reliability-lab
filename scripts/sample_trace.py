"""Genera trazas de ejemplo con datos sintéticos (agente real, LLM simulado, fallos inyectados).

Escenarios (cada uno en su propio thread):

- ``demo-retry``: lectura, escritura aprobada; el primer intento pierde la respuesta después del
  commit y el reintento automático con la misma clave termina en replay → ``completed``.
- ``demo-uncertain``: escritura aprobada cuya respuesta se pierde siempre → ``write_uncertain``
  (se deja así, para analizarla).

Uso:
    uv run python scripts/sample_trace.py --out <directorio fuera de OneDrive>

Imprime un JSON con los run_id y las rutas. Escribe solo dentro de ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anyio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client

from arl.agent.faults import EVERY_CALL, Fault, FaultPlan, FaultyClient
from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.runtime import approve, resume, run_turn
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import Limits
from arl.config import resolve_data_dir
from arl.domain.models import Principal
from arl.mcp_server.server import create_server
from arl.storage.seed import create_database
from arl.tracing.logging_setup import configure_logging
from arl.tracing.recorder import run_file

FAST = Limits(retry_initial_interval=0.0, retry_max_interval=0.0, retry_jitter=False)


async def _scenario(data: Path, db: Path, thread: str, plan: FaultPlan) -> dict[str, Any]:
    model = ScriptedChatModel(
        [
            call("get_project", project_id="apollo"),
            call("update_task", task_id="apo-3", changes={"status": "done"}),
            say("Listo: apo-3 quedó en done."),
        ]
    )
    async with (
        AsyncSqliteSaver.from_conn_string(str(data / "checkpoints.db")) as saver,
        Client(create_server(db, Principal(user_id="alice"))) as client,
    ):
        gateway = McpGateway(FaultyClient(client, plan), "alice")  # type: ignore[arg-type]
        graph = build_graph(
            model, gateway, saver, FAST, lock_dir=data / "locks", trace_dir=data / "traces"
        )
        paused = await run_turn(graph, thread_id=thread, text="Revisá apollo y cerrá apo-3")
        done = await resume(graph, thread_id=thread, decision=approve(paused.approval_request))
    return {
        "thread_id": thread,
        "runs": [paused.run_id, done.run_id],
        "final_status": done.status.value,
        "files": [str(run_file(data / "traces", thread, r)) for r in (paused.run_id, done.run_id)],
    }


async def generate(out: Path) -> dict[str, Any]:
    data = resolve_data_dir({"ARL_DATA_DIR": str(out)})  # rechaza OneDrive (SQLite)
    data.mkdir(parents=True, exist_ok=True)
    # Una base por escenario: ambos parten del mismo estado sembrado (apo-3 en todo).
    dbs = {name: data / f"{name}.db" for name in ("retry", "uncertain")}
    for db in dbs.values():
        if not db.exists():
            create_database(db).close()
    plan_retry: FaultPlan = {("update_task", 0): Fault.DISCONNECT_AFTER}
    plan_uncertain: FaultPlan = {("update_task", EVERY_CALL): Fault.TIMEOUT_AFTER}
    return {
        "trace_dir": str(data / "traces"),
        "scenarios": [
            await _scenario(data, dbs["retry"], "demo-retry", plan_retry),
            await _scenario(data, dbs["uncertain"], "demo-uncertain", plan_uncertain),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging("ERROR")  # los reintentos inyectados son esperables
    print(json.dumps(anyio.run(generate, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
