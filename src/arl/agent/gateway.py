"""Cliente MCP del agente: llamadas a las tools del registro y verificación de contrato.

La identidad no viaja en ninguna llamada: la fija el proceso del servidor MCP (``ARL_PRINCIPAL``).
``principal_id`` se guarda solo para que el grafo verifique que un thread se reanuda con la misma
identidad con la que se creó.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from arl.agent.tool_registry import INJECTED_WRITE_ARGS, REGISTRY, ToolKind


class ContractMismatch(RuntimeError):
    """Las tools del servidor no coinciden con el registro del agente."""


@dataclass(frozen=True)
class ToolResult:
    is_error: bool
    text: str
    data: dict[str, Any] | None

    def content(self) -> str:
        """Contenido para el ToolMessage: JSON estructurado si existe, si no el texto."""
        if self.is_error or self.data is None:
            return self.text
        return json.dumps(self.data, ensure_ascii=False, sort_keys=True)


class McpGateway:
    def __init__(self, client: Client, principal_id: str) -> None:
        self._client = client
        self.principal_id = principal_id

    async def verify_contract(self) -> None:
        """Falla si el servidor no expone exactamente las tools y argumentos esperados."""
        tools = {tool.name: tool for tool in (await self._client.list_tools()).tools}
        if set(tools) != set(REGISTRY):
            raise ContractMismatch(f"tools del servidor: {sorted(tools)}")
        for name, spec in REGISTRY.items():
            expected = set(spec.args_model.model_fields)
            if spec.kind is ToolKind.WRITE:
                expected |= INJECTED_WRITE_ARGS
            actual = set(tools[name].input_schema.get("properties", {}))
            if actual != expected:
                raise ContractMismatch(f"{name}: argumentos {sorted(actual)} != {sorted(expected)}")

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in REGISTRY:
            raise ValueError(f"tool fuera del registro: {name}")
        result = await self._client.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return ToolResult(result.is_error, text, result.structured_content)


def stdio_server_parameters(
    principal_id: str, *, data_dir: Path, db_name: str = "app.db"
) -> StdioServerParameters:
    """Lanza ``python -m arl.mcp_server`` con la identidad simulada en su entorno."""
    env = {
        **os.environ,
        "ARL_PRINCIPAL": principal_id,
        "ARL_DATA_DIR": str(data_dir),
        "ARL_DB_NAME": db_name,
    }
    return StdioServerParameters(command=sys.executable, args=["-m", "arl.mcp_server"], env=env)
