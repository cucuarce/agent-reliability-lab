"""``python -m arl.mcp_server``: servidor MCP por stdio.

Contexto de ejecución confiable (lo fija quien lanza el proceso, no el LLM):

- ``ARL_PRINCIPAL``: identidad **simulada** del usuario. No hay autenticación real.
- ``ARL_DATA_DIR`` / ``ARL_DB_NAME``: ubicación de la base (ver ``arl.config``).

El proceso valida la configuración al arrancar y falla rápido (exit 2) si algo no cuadra.
stdout queda reservado al protocolo MCP: los diagnósticos van a stderr.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from contextlib import closing

from mcp.server import MCPServer
from pydantic import ValidationError

from arl.config import ConfigError, app_db_path
from arl.domain.models import Principal
from arl.mcp_server.server import create_server
from arl.storage import repositories
from arl.storage.db import SCHEMA_VERSION, connect

PRINCIPAL_ENV = "ARL_PRINCIPAL"


def build_from_env(env: Mapping[str, str]) -> MCPServer:
    raw_principal = env.get(PRINCIPAL_ENV)
    if not raw_principal:
        raise ConfigError(f"falta {PRINCIPAL_ENV} (identidad simulada del usuario)")
    try:
        principal = Principal(user_id=raw_principal)
    except ValidationError:
        raise ConfigError(f"{PRINCIPAL_ENV} tiene un formato inválido") from None

    db_path = app_db_path(env)
    if not db_path.is_file():
        raise ConfigError(
            f"no existe la base {db_path}; creala con `python -m arl.storage.seed --db-name "
            f"{db_path.name}`"
        )
    with closing(connect(db_path)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ConfigError(f"{db_path} tiene schema v{version}; se esperaba v{SCHEMA_VERSION}")
        if not repositories.user_exists(conn, principal.user_id):
            raise ConfigError(f"el principal '{principal.user_id}' no existe en {db_path.name}")
    return create_server(db_path, principal)


def main() -> int:
    try:
        server = build_from_env(os.environ)
    except ConfigError as exc:
        print(f"arl.mcp_server: {exc}", file=sys.stderr)
        return 2
    server.run()  # stdio
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
