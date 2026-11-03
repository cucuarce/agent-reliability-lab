"""Configuración de logging de los procesos del producto.

- Un único handler a stderr (stdout del servidor MCP es del protocolo) con el filtro de
  redacción aplicado a **todos** los registros, también los de dependencias (LangGraph, SDK).
- Loggers de dependencias que vuelcan contenido a nivel DEBUG quedan como mínimo en INFO, aunque
  se pida DEBUG. Verificado con un canario en un turno real (memoria y stdio): solo ``aiosqlite``
  lo filtró, porque registra los parámetros SQL, es decir, checkpoints completos con mensajes y
  resultados de tools. ``allow_unsafe_debug=True`` lo habilita a conciencia (solo desarrollo).

La redacción es heurística: no garantiza confidencialidad. La protección real es no producir el
texto (mensajes fijos en excepciones, trazas con lista blanca, dependencias peligrosas en INFO).
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from arl.agent.redaction import RedactingFilter

LOG_LEVEL_ENV = "ARL_LOG_LEVEL"
# logger → por qué no puede quedar en DEBUG
UNSAFE_DEBUG_LOGGERS = {
    "aiosqlite": "registra los parámetros SQL: checkpoints completos (mensajes y resultados)",
}
_HANDLER_NAME = "arl-stderr"


def configure_logging(
    level: str | int = "WARNING", *, stream: TextIO | None = None, allow_unsafe_debug: bool = False
) -> logging.Handler:
    """Idempotente: reemplaza el handler propio si ya estaba instalado."""
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        raise ValueError(f"nivel de log inválido: {level!r}")
    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)
    root.setLevel(numeric)
    for name in UNSAFE_DEBUG_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.NOTSET if allow_unsafe_debug else max(numeric, logging.INFO)
        )
    return handler
