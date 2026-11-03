"""Logs del agente sin secretos: segunda barrera, no la única.

La primera barrera es no producir texto sensible: el gateway lanza excepciones con mensajes fijos
y sin ``__cause__`` ni ``__context__`` (ver ``McpGateway.call``). Este filtro cubre lo que se
escape igual: enmascara formas habituales de secretos en el mensaje, los argumentos y el
traceback de cada registro del logger ``arl.agent``. Conserva tipo de error, operación y trace_id.

Es heurístico: un secreto sin forma reconocible (sin ``password=``, ``Bearer``, ``sk-``...) no
se detecta. Por eso no reemplaza a la primera barrera.
"""

from __future__ import annotations

import logging
import re

REDACTED = "[REDACTED]"
_PATTERNS = (
    # clave=valor o clave: valor (password, token, secret, api_key, authorization, ...)
    re.compile(
        r"(?i)\b(pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|authorization|"
        r"credentials?)(\s*[=:]\s*)(?:bearer\s+)?[^\s,;'\"]+"
    ),
    re.compile(r"(?i)\b(bearer)(\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(sk)(-)[A-Za-z0-9_-]{8,}"),
)


def redact(text: str) -> str:
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return text


class RedactingFilter(logging.Filter):
    """Reescribe el registro ya formateado: mensaje con argumentos y traceback."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:
            formatted = logging.Formatter().formatException(record.exc_info)
            record.exc_text = redact(formatted)
            record.exc_info = None  # el formateador usa exc_text ya enmascarado
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


LOGGER = logging.getLogger("arl.agent")
LOGGER.addFilter(RedactingFilter())
