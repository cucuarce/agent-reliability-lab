"""Proceso auxiliar de ``test_agent_crash.py`` (no es un test: pytest no lo recolecta).

Reanuda la aprobación de un thread y termina el proceso con ``os._exit`` inmediatamente después de
que el servidor confirma la escritura: no corre ningún ``finally`` y LangGraph no llega a guardar el
checkpoint de ``execute_write``. Es el equivalente a matar el proceso en el peor momento.

Uso: python crash_during_write.py <app.db> <checkpoints.db> <thread_id> <approval_request.json>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client

from arl.agent.gateway import McpGateway
from arl.agent.graph import build_graph
from arl.agent.runtime import approve, resume
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.domain.models import Principal
from arl.mcp_server.server import create_server

CRASH_EXIT_CODE = 75


async def main(db_path: str, checkpoints: str, thread_id: str, request_file: str) -> None:
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "done"}), say("Listo.")]
    )
    async with (
        AsyncSqliteSaver.from_conn_string(checkpoints) as saver,
        Client(create_server(Path(db_path), Principal(user_id="alice"))) as client,
    ):
        gateway = McpGateway(client, "alice")
        real_call = gateway.call

        async def die_after_commit(name, arguments):
            result = await real_call(name, arguments)
            if name == "update_task" and not result.is_error:
                os._exit(CRASH_EXIT_CODE)
            return result

        gateway.call = die_after_commit  # type: ignore[method-assign]
        graph = build_graph(model, gateway, saver)
        await resume(graph, thread_id=thread_id, decision=approve(request))
    sys.exit(1)  # no debería llegar acá


if __name__ == "__main__":
    anyio.run(main, *sys.argv[1:5])
