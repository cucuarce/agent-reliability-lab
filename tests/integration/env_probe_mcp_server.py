"""Servidor MCP mínimo (stdio) que informa si una variable existe en su entorno. Auxiliar de tests.

Se lanza con el mismo entorno que ``stdio_server_parameters`` prepara para ``arl.mcp_server``.
Devuelve solo un booleano, nunca el valor.
"""

from __future__ import annotations

import os

from mcp.server import MCPServer

server = MCPServer("env-probe")


@server.tool()
def env_has(name: str) -> bool:
    """True si la variable está definida en el entorno del proceso."""
    return name in os.environ


if __name__ == "__main__":
    server.run()
