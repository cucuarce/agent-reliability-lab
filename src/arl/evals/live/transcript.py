"""Transcripción local de cada corrida (``transcript.jsonl``, ``arl.live-transcript/1``).

El reporte guarda métricas sin texto; la transcripción guarda lo necesario para reconstruir una
corrida sintética: el pedido, las respuestas visibles del modelo (incluida la final completa, tal
como la escribió el modelo: ``model_reply``; aparte, ``user_reply``, la que vio el usuario),
las tool calls con sus argumentos y resultados, las decisiones del humano simulado, los bloqueos
del alcance, los errores y las métricas por llamada.

- **Solo lo visible.** El texto de ``AIMessage.content`` (los bloques de texto), nunca
  ``additional_kwargs`` (donde el proveedor pone el razonamiento), headers ni objetos del
  proveedor. Cada evento tiene una lista explícita de campos (``EVENT_FIELDS``): un campo de más
  es un error, no se serializa nada por reflexión.
- **Solo datos sintéticos verificados.** El runner comprueba que la base temporal tiene
  exactamente las tareas y los usuarios de la semilla empaquetada. Si no, el escenario se
  retiene (``withheld``): no se escribe ningún contenido.
- **Sin secretos.** Antes de escribir, todo texto pasa por ``redact`` (claves de Groq, tokens
  ``sk-``, ``Bearer``, ``api_key=``); el reporte cuenta los reemplazos.
- **Fuera del repo y de OneDrive.** ``<ARL_DATA_DIR>/live-transcripts/<run_id>/``; si el
  directorio de datos está sincronizado o dentro del repositorio, no hay transcripción
  (``TranscriptUnavailable``) y la corrida sigue sin ella.
- **Se escribe a medida que avanza** (una línea por evento, en modo append): lo ya escrito queda
  aunque un escenario falle, se bloquee una herramienta, se agote el presupuesto o se aborte.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from arl.agent.graph import TOOL_LIMIT_REPLY, TURN_LIMIT_REPLY
from arl.config import ConfigError, resolve_data_dir

SCHEMA = "arl.live-transcript/2"  # /2 (6B.6): final con model_reply y user_reply
DIR_NAME = "live-transcripts"
FILE_NAME = "transcript.jsonl"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_AGENT_REPLIES = frozenset({TOOL_LIMIT_REPLY, TURN_LIMIT_REPLY})  # las escribe el grafo, no el LLM
_SCENARIO = ("case_id", "repetition")
EVENT_FIELDS: dict[str, frozenset[str]] = {
    name: frozenset({"event", *fields})
    for name, fields in {
        "header": ("schema", "run_id", "created_at"),
        "suite_start": ("mode", "inference", "provider", "model", "suite_version", "config"),
        "scenario_start": (*_SCENARIO, "phase", "thread_id", "principal", "scope",
                           "approval_policy", "synthetic"),
        "user_request": (*_SCENARIO, "text"),
        "agent_note": (*_SCENARIO, "text"),  # SystemMessage del agente en el historial
        "model_call": (*_SCENARIO, "call_index", "turn", "attempt", "outcome", "latency_ms",
                       "estimated_prompt_tokens", "input_tokens", "output_tokens",
                       "reasoning_tokens", "usage_reported", "code", "origin", "status",
                       "finish_reason", "tool_calls", "invalid_tool_calls", "text_tool_call",
                       "model_name"),
        "model_response": (*_SCENARIO, "turn", "origin", "content", "tool_calls"),
        "tool_result": (*_SCENARIO, "turn", "tool_call_id", "tool", "is_error", "content"),
        "approval": (*_SCENARIO, "index", "task_id", "tool_call_id", "changes", "decision",
                     "source", "actor"),
        "scope_block": (*_SCENARIO, "tool_call_id", "tool", "task_id", "reason"),
        "error": (*_SCENARIO, "kind", "detail"),
        "final": (*_SCENARIO, "final_status", "statuses", "model_reply", "user_reply",
                  "reply_source", "write_outcomes"),
        "persistence": (*_SCENARIO, "tasks", "audit", "ledger", "writes"),
        "evaluation": (*_SCENARIO, "verdict", "error_class", "cause", "graders", "truthfulness",
                       "communication"),
        "withheld": (*_SCENARIO, "reason"),
        "scenario_aborted": (*_SCENARIO, "error_type"),
    }.items()
}  # fmt: skip
_CALL_FIELDS = EVENT_FIELDS["model_call"] - {"event", *_SCENARIO, "call_index", "turn"}

_SECRETS = re.compile(
    r"gsk_[A-Za-z0-9_]{10,}|\bsk-[A-Za-z0-9_-]{16,}|\bbearer\s+[A-Za-z0-9._~+/=-]{10,}"
    r"|\b(?:api[_-]?key|authorization|x-api-key)\b\s*[=:]\s*\S+",
    re.IGNORECASE,
)
REDACTED = "[redactado]"


class TranscriptUnavailable(RuntimeError):
    """No se puede garantizar un lugar local, no sincronizado y fuera del repo."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def redact(value: Any) -> tuple[Any, int]:
    """Reemplaza lo que parece una credencial en todo texto, también dentro de listas y dicts."""
    if isinstance(value, str):
        return _SECRETS.subn(REDACTED, value)
    if isinstance(value, Mapping):
        total, out = 0, {}
        for k, v in value.items():
            out[str(k)], n = redact(v)
            total += n
        return out, total
    if isinstance(value, list | tuple):
        total, items = 0, []
        for v in value:
            item, n = redact(v)
            items.append(item)
            total += n
        return items, total
    return value, 0


def visible_text(content: Any) -> str:
    """El texto visible de un mensaje: los bloques de texto, nunca razonamiento ni otros."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else ():
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def conversation(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Los mensajes del thread como eventos, con el turno del modelo de cada uno."""
    events: list[dict[str, Any]] = []
    turn = 0
    turn_of: dict[str, int] = {}
    for message in messages:
        if isinstance(message, HumanMessage):
            events.append({"event": "user_request", "text": visible_text(message.content)})
        elif isinstance(message, SystemMessage):  # nota del agente: nunca se atribuye al LLM
            events.append({"event": "agent_note", "text": visible_text(message.content)})
        elif isinstance(message, AIMessage):
            text = visible_text(message.content)
            origin = "agent" if text in _AGENT_REPLIES and not message.tool_calls else "model"
            if origin == "model":
                turn += 1
            calls = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "args": c["args"] if isinstance(c["args"], dict) else {},
                }
                for c in message.tool_calls
            ]
            turn_of.update({c["id"]: turn for c in calls})
            events.append(
                {
                    "event": "model_response",
                    "turn": turn,
                    "origin": origin,
                    "content": text,
                    "tool_calls": calls,
                }
            )
        elif isinstance(message, ToolMessage):
            text = visible_text(message.content)
            events.append(
                {
                    "event": "tool_result",
                    "turn": turn_of.get(message.tool_call_id),
                    "tool_call_id": message.tool_call_id,
                    "tool": message.name,
                    "is_error": message.status == "error" or text.startswith("ERROR "),
                    "content": text,
                }
            )
    return events


def model_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Métricas por llamada al proveedor; los reintentos comparten el turno (``attempt`` > 0)."""
    turn = 0
    events = []
    for index, record in enumerate(records, start=1):
        if record.get("attempt") == 0:
            turn += 1
        events.append(
            {"event": "model_call", "call_index": index, "turn": turn}
            | {k: record.get(k) for k in sorted(_CALL_FIELDS)}
        )
    return events


@dataclass
class TranscriptWriter:
    run_id: str
    path: Path
    events: int = 0
    redactions: int = 0
    withheld: int = 0

    @classmethod
    def create(
        cls, run_id: str, *, root: Path, env: Mapping[str, str] | None = None
    ) -> TranscriptWriter:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"run_id inválido: {run_id!r}")
        try:
            base = resolve_data_dir(env)
        except ConfigError:
            raise TranscriptUnavailable(
                "data_dir_synced", "el directorio de datos está dentro de OneDrive"
            ) from None
        if base.is_relative_to(root.resolve()):
            raise TranscriptUnavailable(
                "inside_repository", "el directorio de datos está dentro del repositorio"
            )
        target = base / DIR_NAME / run_id
        target.mkdir(parents=True, exist_ok=False)
        writer = cls(run_id, target / FILE_NAME)
        writer.write(
            {
                "event": "header",
                "schema": SCHEMA,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        return writer

    def write(self, event: Mapping[str, Any]) -> None:
        kind = event.get("event")
        allowed = EVENT_FIELDS.get(kind) if isinstance(kind, str) else None
        if allowed is None:
            raise ValueError(f"evento desconocido: {kind!r}")
        if extra := sorted(set(event) - allowed):
            raise ValueError(f"campos no permitidos en {kind}: {extra}")
        clean, n = redact(dict(event))
        self.redactions += n
        self.withheld += kind == "withheld"
        line = json.dumps(clean, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        self.events += 1

    def describe(self) -> dict[str, Any]:
        return {
            "status": "written",
            "schema": SCHEMA,
            "run_id": self.run_id,
            "path": f"<ARL_DATA_DIR>/{DIR_NAME}/{self.run_id}/{FILE_NAME}",
            "events": self.events,
            "redactions": self.redactions,
            "withheld": self.withheld,
        }
