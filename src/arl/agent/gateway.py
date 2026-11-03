"""Cliente MCP del agente: llamadas a las tools del registro, clasificación de fallos y contrato.

La identidad no viaja en ninguna llamada: la fija el proceso del servidor MCP (``ARL_PRINCIPAL``).
``principal_id`` se guarda solo para que el grafo verifique que un thread se reanuda con la misma
identidad con la que se creó.

Clasificación de cada llamada:

- **Resultado de dominio** (``[forbidden]``, ``[version_conflict]``, ...): ``ToolResult`` con
  ``is_error``. Definitivo: nunca se reintenta.
- **``ToolUnavailable``** (timeout, conexión cerrada, error opaco del servidor): transitorio. En
  una escritura, el resultado es **incierto** (el servidor pudo haber confirmado).
- **``ToolResponseInvalid``** (respuesta fuera de contrato, error de protocolo): definitivo.

Los mensajes de las excepciones son fijos y no conservan el error original: su texto podría
contener rutas o datos sensibles y LangGraph registra cada reintento con su traceback. Por eso se
lanzan fuera del bloque ``except``: sin ``__cause__`` ni ``__context__`` (``from None`` solo oculta
el contexto al imprimir; el objeto lo seguiría guardando). Para diagnosticar quedan ``code``,
``tool`` y ``error_type`` (el nombre del tipo original).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT
from pydantic import BaseModel, ConfigDict, ValidationError

from arl.agent.tool_registry import INJECTED_WRITE_ARGS, REGISTRY, ToolKind
from arl.domain.errors import DomainError
from arl.domain.models import ProjectDetail, ProjectSummary, UpdateResult
from arl.tracing import recorder as tracing

DEFAULT_CALL_TIMEOUT_S = 10.0
_TRANSPORT_ERRORS = (
    TimeoutError,
    OSError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
)


class ContractMismatch(RuntimeError):
    """Las tools del servidor no coinciden con el registro del agente."""


class ToolCallFailed(RuntimeError):
    """La llamada no produjo un resultado utilizable. ``code`` es estable y seguro de mostrar."""

    code = "tool_call_failed"
    retryable = False

    def __init__(self, tool: str, detail: str, *, error_type: str | None = None) -> None:
        super().__init__(f"[{self.code}] {tool}: {detail}")
        self.tool = tool
        self.error_type = error_type or type(self).__name__


class ToolUnavailable(ToolCallFailed):
    code = "unavailable"
    retryable = True


class ToolResponseInvalid(ToolCallFailed):
    code = "invalid_response"


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


class _ProjectList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: list[ProjectSummary]


def _validate_output(name: str, arguments: dict[str, Any], data: Any) -> dict[str, Any]:
    """La respuesta exitosa debe cumplir el contrato y referirse al recurso pedido."""
    if not isinstance(data, dict):
        raise ToolResponseInvalid(name, "respuesta sin contenido estructurado")
    model = _parse_output(name, data)
    if model is None:  # el ValidationError incluye los valores recibidos: no se propaga
        raise ToolResponseInvalid(
            name, "la respuesta no cumple el contrato", error_type="ValidationError"
        )
    if isinstance(model, ProjectDetail) and model.project.id != arguments["project_id"]:
        raise ToolResponseInvalid(name, "la respuesta corresponde a otro proyecto")
    if isinstance(model, UpdateResult) and model.task.id != arguments["task_id"]:
        raise ToolResponseInvalid(name, "la respuesta corresponde a otra tarea")
    return model.model_dump(mode="json")


def _parse_output(name: str, data: dict[str, Any]) -> BaseModel | None:
    schema: type[BaseModel] = {"list_projects": _ProjectList, "get_project": ProjectDetail}.get(
        name, UpdateResult
    )
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


class McpGateway:
    def __init__(
        self, client: Client, principal_id: str, *, timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    ) -> None:
        self._client = client
        self.principal_id = principal_id
        self.timeout_s = timeout_s

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
        start = time.perf_counter()
        try:
            result = await self._call(name, arguments)
        except ToolCallFailed as exc:
            _trace_call(name, arguments, start, failure=exc)
            raise
        _trace_call(name, arguments, start, result=result)
        return result

    async def _call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        failure: ToolCallFailed | None = None
        try:
            result = await self._client.call_tool(
                name, arguments, read_timeout_seconds=self.timeout_s
            )
        except MCPError as exc:
            failure = _classify_mcp_error(name, exc.error.code)
        except _TRANSPORT_ERRORS as exc:
            failure = ToolUnavailable(
                name, "transporte no disponible", error_type=type(exc).__name__
            )
        # Se lanza fuera del ``except``: así no queda ``__context__`` con el texto original, que
        # LangGraph volcaría en el log de cada reintento.
        if failure is not None:
            raise failure

        text = "\n".join(block.text for block in result.content if block.type == "text")
        if result.is_error:
            if text.strip() == f"Error executing tool {name}":  # falla opaca del servidor
                raise ToolUnavailable(name, "error interno del servidor")
            return ToolResult(True, text, None)
        return ToolResult(False, text, _validate_output(name, arguments, result.structured_content))


DOMAIN_CODE = re.compile(r"^\[([a-z_]+)\]")
# Códigos que produce el servidor (``DomainError.code`` de cada subclase de ``arl.domain.errors``).
DOMAIN_CODES = frozenset(cls.code for cls in DomainError.__subclasses__())


def domain_code(tool: str, text: str) -> str | None:
    """Código de un error de dominio devuelto por ``tool``, o ``None`` si no se reconoce.

    El SDK MCP v2 entrega un ``ToolError`` solo como texto: ``"Error executing tool <tool>: "``
    seguido del mensaje, que el servidor arma como ``"[codigo] detalle"``. Se acepta únicamente
    ese prefijo exacto (o el mensaje sin prefijo) y un código conocido. Nada se infiere: otro
    prefijo, un código desconocido o un texto malformado devuelven ``None``.
    """
    body = text.removeprefix(f"Error executing tool {tool}: ")
    match = DOMAIN_CODE.match(body)
    return match.group(1) if match and match.group(1) in DOMAIN_CODES else None


def _trace_call(
    name: str,
    arguments: dict[str, Any],
    start: float,
    *,
    failure: ToolCallFailed | None = None,
    result: ToolResult | None = None,
) -> None:
    """Evento ``mcp_call``: identificadores, resultado y duración; nunca argumentos ni textos."""
    tracing.count("mcp_calls")
    data: dict[str, Any] = {"tool": name, "duration_ms": tracing.elapsed_ms(start)}
    if failure is not None:
        tracing.count("mcp_failures")
        data |= {"outcome": failure.code, "code": failure.code, "error_type": failure.error_type}
    elif result is not None and result.is_error:
        data |= {"outcome": "domain_error", "code": domain_code(name, result.text) or "unknown"}
    else:
        data["outcome"] = "ok"
        if result is not None and result.data is not None and "replayed" in result.data:
            data["replayed"] = result.data["replayed"]
    for key in ("project_id", "task_id", "idempotency_key"):
        if key in arguments:
            data[key] = tracing.safe_id(arguments[key])
    tracing.emit("mcp_call", **data, **tracing.current_step())


def _classify_mcp_error(name: str, code: int) -> ToolCallFailed:
    if code == REQUEST_TIMEOUT:
        return ToolUnavailable(name, "timeout esperando la respuesta", error_type="MCPError")
    if code == CONNECTION_CLOSED:
        return ToolUnavailable(name, "conexión con el servidor cerrada", error_type="MCPError")
    return ToolResponseInvalid(name, f"error de protocolo MCP {code}", error_type="MCPError")


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
