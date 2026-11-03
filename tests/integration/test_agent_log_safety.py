"""Seguridad de logs: ningún secreto simulado llega a excepciones, tracebacks, estado ni logs.

No alcanza con ``from None``: la excepción original seguiría en ``__context__`` (que leen
formateadores y reporteros de errores) aunque el traceback impreso la oculte. Cada fallo se
inspecciona por todas las vías por las que su texto podría salir.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import anyio
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
from mcp_types import CONNECTION_CLOSED, INTERNAL_ERROR, REQUEST_TIMEOUT

from arl.agent.gateway import McpGateway, ToolCallFailed
from arl.agent.redaction import LOGGER, redact
from arl.agent.runtime import TurnStatus, run_turn, thread_config
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say

# Alcance explícito de cada turno (etapa 6A.1): exactamente lo que el test escribe.
READ_ONLY = RequestScope.read_only()
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))

pytestmark = pytest.mark.anyio

SECRET = "hunter2-s3cr3t"
LEAKY = f"password={SECRET} token=Bearer {SECRET} en C:/secretos/{SECRET}.db"


def _raising(exc: BaseException):
    async def call_tool(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        raise exc

    return call_tool


def _returning(result: CallToolResult):
    async def call_tool(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        return result

    return call_tool


class _Client:
    def __init__(self, call_tool) -> None:
        self.call_tool = call_tool


def _every_surface(exc: BaseException) -> list[str]:
    """Todo lo que un logger, un formateador o un reportero de errores podría volcar."""
    surfaces = [str(exc), repr(exc), repr(exc.args), repr(vars(exc))]
    surfaces += ["".join(traceback.format_exception(exc))]
    surfaces += [repr(getattr(exc, "__notes__", []))]
    for linked in (exc.__cause__, exc.__context__):
        if linked is not None:
            surfaces.append(repr(linked))
    return surfaces


MALFORMED_WITH_SECRET = CallToolResult(
    content=[TextContent(type="text", text=LEAKY)],
    structured_content={"result": [{"id": LEAKY}]},
    is_error=False,
)


@pytest.mark.parametrize(
    ("client", "code", "error_type"),
    [
        (_raising(OSError(LEAKY)), "unavailable", "OSError"),
        (_raising(TimeoutError(LEAKY)), "unavailable", "TimeoutError"),
        (_raising(anyio.BrokenResourceError(LEAKY)), "unavailable", "BrokenResourceError"),
        (_raising(MCPError(REQUEST_TIMEOUT, LEAKY)), "unavailable", "MCPError"),
        (_raising(MCPError(CONNECTION_CLOSED, LEAKY)), "unavailable", "MCPError"),
        (_raising(MCPError(INTERNAL_ERROR, LEAKY)), "invalid_response", "MCPError"),
        (_returning(MALFORMED_WITH_SECRET), "invalid_response", "ValidationError"),
    ],
)
async def test_gateway_failures_carry_diagnostics_but_no_secret(client, code, error_type) -> None:
    gateway = McpGateway(_Client(client), "alice")  # type: ignore[arg-type]
    with pytest.raises(ToolCallFailed) as raised:
        await gateway.call("list_projects", {})
    exc = raised.value
    assert (exc.code, exc.tool, exc.error_type) == (code, "list_projects", error_type)
    assert exc.__cause__ is None and exc.__context__ is None
    for surface in _every_surface(exc):
        assert SECRET not in surface and "secretos" not in surface


async def test_failed_turn_logs_trace_operation_and_type_without_secrets(
    open_faulty_agent, caplog, monkeypatch
) -> None:
    model = ScriptedChatModel([call("list_projects"), say("ok")])
    caplog.set_level(logging.DEBUG)  # todo: LangGraph (reintentos con traceback), SDK, arl
    async with open_faulty_agent("alice", model) as (graph, faulty):
        monkeypatch.setattr(faulty, "call_tool", _raising(OSError(LEAKY)))
        result = await run_turn(graph, thread_id="t-log", text="¿proyectos?", scope=READ_ONLY)
        state = await graph.aget_state(thread_config("t-log"))

    assert result.status is TurnStatus.INTERRUPTED
    assert result.run_id
    for text in (caplog.text, str(result), str(state.values)):
        assert SECRET not in text and "secretos" not in text
    retries = [r for r in caplog.records if r.exc_info and r.name.startswith("langgraph")]
    assert retries, "LangGraph debería registrar los reintentos con su traceback"
    (stopped,) = [r for r in caplog.records if r.name == LOGGER.name]
    message = stopped.getMessage()
    for expected in (result.run_id, "list_projects", "unavailable", "OSError", "t-log"):
        assert expected in message


def test_redaction_masks_secrets_in_messages_arguments_and_tracebacks(caplog) -> None:
    """Segunda barrera: aunque un mensaje del agente incluyera un secreto, el filtro del logger
    lo enmascara (mensaje, argumentos y traceback)."""
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    shapes = f"password={SECRET} Authorization: Bearer {SECRET}"
    try:
        raise RuntimeError(shapes)
    except RuntimeError:
        LOGGER.exception("fallo con %s y api_key=%s", shapes, SECRET)
    assert caplog.records
    assert SECRET not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "RuntimeError" in caplog.text  # el diagnóstico (tipo) se conserva


@pytest.mark.parametrize(
    "text",
    [
        f"password={SECRET}",
        f"PASSWORD: {SECRET}",
        f"api_key={SECRET}",
        f"api-key: {SECRET}",
        f"secret={SECRET}",
        f"Authorization: Bearer {SECRET}",
        f"token={SECRET}",
        f"key sk-{SECRET}abcdef",
    ],
)
def test_redact_masks_common_secret_shapes(text: str) -> None:
    assert SECRET not in redact(text)


def test_redact_keeps_ordinary_diagnostics() -> None:
    text = "[unavailable] update_task: transporte no disponible (OSError) run_id=abc123"
    assert redact(text) == text
