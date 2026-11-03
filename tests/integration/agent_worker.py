"""Proceso de agente independiente que aprueba una escritura pendiente. Auxiliar de tests.

Uso: python agent_worker.py <principal> <status> <app.db> <checkpoints.db> <thread> <request.json>

Imprime en stdout un JSON con el resultado (o el tipo de excepción) para que el test lo agregue.
Usa el mismo directorio de locks que el resto de los procesos del test.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import anyio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client

from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.runtime import approve, resume, thread_config
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.domain.models import Principal
from arl.mcp_server.server import create_server


async def _barrier() -> None:
    """Con ``ARL_TEST_BARRIER_DIR``: avisa que está listo y espera la señal ``go`` común, para
    que todos los procesos aprueben a la vez (arranque ya pagado)."""
    barrier = os.environ.get("ARL_TEST_BARRIER_DIR")
    if not barrier:
        return
    (Path(barrier) / f"ready-{os.getpid()}").touch()
    with anyio.fail_after(60):
        while not (Path(barrier) / "go").exists():
            await anyio.sleep(0.005)


async def main(principal: str, status: str, db: str, checkpoints: str, thread: str, req: str):
    request = json.loads(Path(req).read_text(encoding="utf-8"))
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": status}), say("Listo.")]
    )
    async with (
        AsyncSqliteSaver.from_conn_string(checkpoints) as saver,
        Client(create_server(Path(db), Principal(user_id=principal))) as client,
    ):
        gateway = McpGateway(client, principal)
        graph = build_graph(model, gateway, saver, lock_dir=Path(checkpoints).parent / "locks")
        await _barrier()
        try:
            result = await resume(graph, thread_id=thread, decision=approve(request))
        except Exception as exc:  # el test agrega el tipo; no hay datos sensibles en juego
            traceback.print_exc()  # a stderr, para diagnosticar fallos inesperados
            print(json.dumps({"exception": type(exc).__name__}))
            return
        messages = (await graph.aget_state(thread_config(thread))).values["messages"]
        tool = [m for m in messages if m.type == "tool"][-1].content
        print(json.dumps({"status": result.status.value, "tool": tool}))


if __name__ == "__main__":
    anyio.run(main, *sys.argv[1:7])
