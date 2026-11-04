"""Proveedor real de la evaluación conductual (etapa 6B.1): solo Groq.

- La credencial sale **solo** de ``GROQ_API_KEY``: la lee el SDK. Este módulo nunca la recibe
  como argumento, nunca la guarda y nunca la registra (solo comprueba si está definida). Si el
  entorno no la define, ``load_env_file`` la toma de un ``.env`` local (y nada más de ese archivo).
- ``classify`` convierte cada excepción del proveedor en un código fijo, sin conservar su
  mensaje: el texto de un error de Groq puede incluir la salida del modelo.
- ``tool_use_failed`` (Groq no pudo interpretar la tool call que generó el modelo) es un error
  de **formato del modelo**, no del proveedor.
- Los reintentos no los hace el SDK (``max_retries=0``): los hace ``LiveModel``, para que cada
  intento cuente en el presupuesto.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
DEFAULT_MODEL = "openai/gpt-oss-20b"

FailureCode = Literal[
    "rate_limited",
    "timeout",
    "server_error",
    "connection",
    "auth",
    "forbidden",
    "not_found",
    "request_too_large",
    "context_length",
    "tool_use_failed",
    "bad_request",
    "unknown",
]
Origin = Literal["provider", "model"]

_RETRYABLE = frozenset({"rate_limited", "timeout", "server_error", "connection"})
# Se consulta sobre el fallo final, después de los reintentos. La cuenta o el modelo no sirven
# (401, 403, 404), o la cuota se agotó (un 429 que persiste o pide esperar más de lo permitido;
# un 413 por el límite de tokens del plan): seguir solo gastaría el presupuesto.
_ABORTS_SUITE = frozenset({"auth", "forbidden", "not_found"})
_QUOTA_EXHAUSTED = frozenset({"rate_limited", "request_too_large"})


@dataclass(frozen=True)
class ProviderFailure:
    code: FailureCode
    status: int | None = None
    retry_after_s: float | None = None

    @property
    def origin(self) -> Origin:
        return "model" if self.code == "tool_use_failed" else "provider"

    @property
    def retryable(self) -> bool:
        return self.code in _RETRYABLE

    @property
    def aborts_suite(self) -> bool:
        return self.code in _ABORTS_SUITE or self.code in _QUOTA_EXHAUSTED


class ProviderError(RuntimeError):
    """Fallo del proveedor (o de formato del modelo, si ``failure.origin == "model"``), con un
    mensaje fijo. Se lanza fuera del ``except`` original: no conserva ``__context__``."""

    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(f"proveedor: {failure.code}")
        self.failure = failure


class MissingCredential(RuntimeError):
    pass


def _retry_after(headers: httpx.Headers | None) -> float | None:
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            value = float(headers.get(name, ""))
        except ValueError:
            continue  # p. ej. una fecha HTTP: no se interpreta, se usa el backoff propio
        if value >= 0:
            return value / scale
    return None


def _body_code(body: object) -> str | None:
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("code"), str):
            return inner["code"]
    return None


def status_code(status: int, body: object = None) -> FailureCode:
    if status == 401:
        return "auth"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "not_found"
    if status == 413:
        return "request_too_large"  # Groq: el pedido supera los tokens por minuto del plan
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status == 400:
        code = _body_code(body)
        if code == "tool_use_failed":
            return "tool_use_failed"
        if code == "context_length_exceeded":
            return "context_length"
        return "bad_request"
    return "unknown"


def classify(exc: BaseException) -> ProviderFailure:
    """Código fijo para una excepción del proveedor. Nunca conserva el mensaje original."""
    try:
        import groq
    except ImportError:  # pragma: no cover - el extra ``live`` no está instalado
        groq = None  # type: ignore[assignment]
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        return ProviderFailure("timeout")
    if groq is not None:
        if isinstance(exc, groq.APITimeoutError):
            return ProviderFailure("timeout")
        if isinstance(exc, groq.APIConnectionError):
            return ProviderFailure("connection")
        if isinstance(exc, groq.APIStatusError):
            return ProviderFailure(
                status_code(exc.status_code, exc.body),
                exc.status_code,
                _retry_after(exc.response.headers),
            )
    if isinstance(exc, httpx.HTTPStatusError):
        return ProviderFailure(
            status_code(exc.response.status_code),
            exc.response.status_code,
            _retry_after(exc.response.headers),
        )
    if isinstance(exc, httpx.TransportError):
        return ProviderFailure("connection")
    return ProviderFailure("unknown")


def credential_present() -> bool:
    """Solo si la variable está definida y no vacía; nunca su valor."""
    return bool(os.environ.get(API_KEY_ENV))


CredentialSource = Literal["environment", "env_file"]


def load_env_file(
    path: Path, environ: MutableMapping[str, str] = os.environ
) -> CredentialSource | None:
    """Toma ``GROQ_API_KEY`` de un ``.env`` local si el entorno no la define. Devuelve de dónde
    sale la credencial (o None si no hay), nunca su valor.

    - **No sobrescribe:** si la variable ya existe en el entorno, aunque esté vacía, el archivo
      no se lee.
    - **Solo esa clave:** el resto del archivo se ignora. Un ``.env`` no puede cambiar
      ``ARL_PRINCIPAL``, ``ARL_DATA_DIR`` ni nada más del proceso, cuyo entorno hereda el
      servidor MCP.
    - Sin interpolación: el valor se toma literal, sin expandir ``${...}`` del entorno.
    """
    if API_KEY_ENV in environ:
        return "environment" if environ[API_KEY_ENV] else None
    if not path.is_file():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - el extra ``live`` no está instalado
        raise MissingCredential("para leer .env hace falta el extra live (python-dotenv)") from exc
    value = dotenv_values(path, interpolate=False).get(API_KEY_ENV)
    if not value:
        return None
    environ[API_KEY_ENV] = value
    return "env_file"


class GroqSpec(BaseModel):
    """Modelo y parámetros de inferencia. No contiene credenciales."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["groq"] = "groq"
    model: str = Field(default=DEFAULT_MODEL, pattern=r"^[A-Za-z0-9._/:-]{1,128}$")
    temperature: float = Field(default=0.0, ge=0, le=2)
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


def make_groq_model(spec: GroqSpec, *, max_output_tokens: int, timeout_s: float) -> Any:
    """``ChatGroq`` sin reintentos propios. No hace ninguna llamada de red al construirse."""
    if not credential_present():
        raise MissingCredential(f"{API_KEY_ENV} no está definida en el entorno")
    from langchain_groq import ChatGroq

    kwargs: dict[str, Any] = {}
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort
    return ChatGroq(  # la clave la lee el SDK de GROQ_API_KEY
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=max_output_tokens,
        request_timeout=timeout_s,
        max_retries=0,
        **kwargs,
    )


def list_models(*, transport: httpx.BaseTransport | None = None, timeout_s: float = 10.0):
    """``GET /models``: modelos disponibles para la cuenta. No hace inferencia.

    Devuelve solo id, estado, ventana de contexto y dueño de cada modelo."""
    if not credential_present():
        raise MissingCredential(f"{API_KEY_ENV} no está definida en el entorno")
    failure: ProviderFailure | None = None
    with httpx.Client(transport=transport, timeout=timeout_s) as client:
        try:
            response = client.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {os.environ[API_KEY_ENV]}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
        except Exception as exc:
            failure = classify(exc)
    if failure is not None:
        raise ProviderError(failure)
    return [
        {k: m.get(k) for k in ("id", "active", "context_window", "owned_by")}
        for m in data
        if isinstance(m, dict)
    ]
