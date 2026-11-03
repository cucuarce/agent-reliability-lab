"""Inyección de fallos determinística entre el agente y el servidor MCP (tests y evaluación).

``FaultyClient`` envuelve un ``mcp.Client`` y, según un plan ``(tool, n-ésima llamada) -> Fault``,
lanza las mismas excepciones que el SDK produce en fallos reales (verificadas contra mcp 2.3:
``MCPError(REQUEST_TIMEOUT)`` y ``MCPError(CONNECTION_CLOSED)``) o altera la respuesta. Así se
ejercita la clasificación real del ``McpGateway``.

Los fallos ``*_BEFORE`` no llegan al servidor; los ``*_AFTER`` llegan, el servidor procesa (y en
una escritura confirma) y se pierde o corrompe la respuesta: el caso de resultado incierto.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
from mcp_types import CONNECTION_CLOSED, METHOD_NOT_FOUND, REQUEST_TIMEOUT

EVERY_CALL: Literal["*"] = "*"


class Fault(StrEnum):
    TIMEOUT_BEFORE = "timeout_before"
    DISCONNECT_BEFORE = "disconnect_before"
    PROTOCOL_ERROR = "protocol_error"
    SERVER_CRASH = "server_crash"  # error opaco del servidor sin ejecutar la tool
    TIMEOUT_AFTER = "timeout_after"
    DISCONNECT_AFTER = "disconnect_after"
    SERVER_CRASH_AFTER = "server_crash_after"  # la tool se ejecutó y luego falló la respuesta
    MALFORMED = "malformed"
    WRONG_RESOURCE = "wrong_resource"
    EMPTY = "empty"


_NOT_DELIVERED = {
    Fault.TIMEOUT_BEFORE,
    Fault.DISCONNECT_BEFORE,
    Fault.PROTOCOL_ERROR,
    Fault.SERVER_CRASH,
}

FaultPlan = Mapping[tuple[str, int | Literal["*"]], Fault]


def _opaque_error(name: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"Error executing tool {name}")], is_error=True
    )


def _corrupt(fault: Fault, result: CallToolResult) -> CallToolResult:
    if fault is Fault.MALFORMED:
        data: dict[str, Any] | None = {"unexpected": True}
    elif fault is Fault.EMPTY:
        data = None
    else:  # WRONG_RESOURCE: mismo formato, otro recurso
        data = copy.deepcopy(result.structured_content) or {}
        target = data.get("task") or data.get("project") or {}
        target["id"] = "otro-recurso"
    return CallToolResult(content=result.content, structured_content=data, is_error=False)


class FaultyClient:
    def __init__(self, client: Client, plan: FaultPlan) -> None:
        self._client = client
        self._plan = dict(plan)
        self.attempts: list[tuple[str, dict[str, Any]]] = []  # lo que intentó el agente
        self.delivered: list[tuple[str, dict[str, Any]]] = []  # lo que llegó al servidor

    def clear(self) -> None:
        """Quita los fallos pendientes (el "servicio se recuperó")."""
        self._plan.clear()

    async def list_tools(self, **kwargs: Any) -> Any:
        return await self._client.list_tools(**kwargs)

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        index = sum(1 for tool, _ in self.attempts if tool == name)
        self.attempts.append((name, copy.deepcopy(arguments)))
        fault = self._plan.get((name, index)) or self._plan.get((name, EVERY_CALL))

        if fault is Fault.TIMEOUT_BEFORE:
            raise MCPError(REQUEST_TIMEOUT, f"Timed out waiting for '{name}'")
        if fault is Fault.DISCONNECT_BEFORE:
            raise MCPError(CONNECTION_CLOSED, "Connection closed")
        if fault is Fault.PROTOCOL_ERROR:
            raise MCPError(METHOD_NOT_FOUND, "Method not found")
        if fault is Fault.SERVER_CRASH:
            return _opaque_error(name)

        result = await self._client.call_tool(name, arguments, **kwargs)
        self.delivered.append((name, copy.deepcopy(arguments)))
        if fault is None or fault in _NOT_DELIVERED:
            return result
        if fault is Fault.TIMEOUT_AFTER:
            raise MCPError(REQUEST_TIMEOUT, f"Timed out waiting for '{name}'")
        if fault is Fault.DISCONNECT_AFTER:
            raise MCPError(CONNECTION_CLOSED, "Connection closed")
        if fault is Fault.SERVER_CRASH_AFTER:
            return _opaque_error(name)
        return _corrupt(fault, result)
