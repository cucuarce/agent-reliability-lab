"""Contrato de tool calling del historial: cada tool call recibe exactamente una respuesta.

Los proveedores reales (Anthropic, OpenAI) rechazan un historial en el que un mensaje del
asistente con tool calls no va seguido inmediatamente de una respuesta por cada una. Un fallo
que detiene la ejecución deja tool calls sin respuesta; esta política decide qué hacer con ellas:

- **Lectura o preparación de escritura interrumpida:** no hubo efectos, así que un turno nuevo
  las cierra con ``[not_executed]`` antes del mensaje nuevo (``close_unanswered``).
- **Escritura aprobada sin confirmar:** nunca se cierra. Su resultado se desconoce y cerrarla
  inventaría éxito o fracaso. El runtime exige ``continue_after_failure``, que la reintenta con la
  misma clave de idempotencia.

``check_tool_protocol`` es la última barrera: ``call_model`` la ejecuta antes de invocar al LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

NOT_EXECUTED = (
    "ERROR [not_executed] la llamada no se completó por un fallo y se cerró al iniciar un turno "
    "nuevo. No tuvo efectos: no se envió ninguna escritura."
)


class HistoryInconsistent(RuntimeError):
    """El historial no respeta el contrato de tool calling: no se envía al proveedor."""


def unanswered_tool_calls(messages: Sequence[BaseMessage]) -> list[dict]:
    """Tool calls del último mensaje del asistente que todavía no tienen respuesta."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            answered = {m.tool_call_id for m in messages[index + 1 :] if isinstance(m, ToolMessage)}
            return [call for call in message.tool_calls if call["id"] not in answered]
        if not isinstance(message, ToolMessage):
            return []  # después del último asistente hay un mensaje de usuario
    return []


def close_unanswered(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """Respuestas ``[not_executed]`` para las tool calls pendientes, en su orden original.

    Solo es correcto si ninguna tuvo efectos: el llamador debe haber descartado antes el caso de
    una escritura aprobada sin confirmar.
    """
    return [
        ToolMessage(NOT_EXECUTED, tool_call_id=call["id"], name=call["name"], status="error")
        for call in unanswered_tool_calls(messages)
    ]


def check_tool_protocol(messages: Sequence[BaseMessage]) -> None:
    """Falla si alguna tool call queda sin respuesta, se responde dos veces o fuera de lugar."""
    pending: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise HistoryInconsistent("respuesta de tool sin una tool call pendiente")
            pending.remove(message.tool_call_id)
            continue
        if pending:
            raise HistoryInconsistent(f"{len(pending)} tool call(s) sin respuesta")
        if isinstance(message, AIMessage):
            pending = [call["id"] for call in message.tool_calls]
    if pending:
        raise HistoryInconsistent(f"{len(pending)} tool call(s) sin respuesta")
