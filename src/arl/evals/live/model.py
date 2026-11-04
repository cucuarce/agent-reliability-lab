"""``LiveModel``: el LLM real visto por el grafo, con presupuesto, reintentos y medición.

Para el grafo es un modelo más (``bind_tools`` + ``ainvoke``). Para la evaluación:

- **Presupuesto antes de cada intento.** Se verifica que la llamada entre en el presupuesto
  restante (llamadas, tokens y duración) con su peor caso: una cota de los tokens del prompt más
  ``max_output_tokens``, y ``call_timeout_s``. Si no entra, no se llama (``BudgetExceeded``).
- **Cota de tokens del prompt:** los bytes UTF-8 del prompt serializado más un margen fijo (un
  token de un tokenizer de bytes cubre al menos un byte). Es una cota conservadora, no una
  garantía formal: si el proveedor informa más tokens que la cota o que el máximo de salida, el
  medidor se invalida y no se hace ninguna llamada más.
- **Reintentos acotados**, solo de fallos transitorios (429, timeout, 5xx, conexión), respetando
  ``Retry-After``. Si la espera pedida supera ``max_retry_wait_s``, no se reintenta.
- **Registro por llamada**, sin contenido: tokens, latencia, formato de la respuesta original
  (tool calls válidas, ``invalid_tool_calls``, tool calls escritas como texto) y errores con su
  código fijo.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import anyio
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from arl.evals.live.providers import ProviderError, ProviderFailure, classify

PROMPT_OVERHEAD_TOKENS = 1024  # plantilla de chat del proveedor (roles, tokens especiales)
PER_MESSAGE_TOKENS = 16


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = Field(ge=1)  # intentos, incluidos los reintentos
    max_total_tokens: int = Field(ge=1)  # entrada + salida
    max_duration_s: float = Field(gt=0)

    def covers(self, other: Budget) -> bool:
        return (
            self.max_model_calls >= other.max_model_calls
            and self.max_total_tokens >= other.max_total_tokens
            and self.max_duration_s >= other.max_duration_s
        )


class CallLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_output_tokens: int = Field(default=1024, ge=16)
    call_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_retry_wait_s: float = Field(default=20.0, ge=0)
    backoff_initial_s: float = Field(default=1.0, ge=0)


class BudgetExceeded(RuntimeError):
    """No se puede garantizar el presupuesto de la próxima llamada: no se hace."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"presupuesto: {reason}")
        self.reason = reason


class Meter:
    """Consumo contra un ``Budget`` (un escenario o la suite completa)."""

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self.calls = 0
        self.tokens = 0
        self.broken: str | None = None  # la cota dejó de valer: no hay más llamadas

    def elapsed(self) -> float:
        return self._clock() - self._start

    def remaining(self) -> Budget | None:
        """Lo que queda, o None si ya no queda nada utilizable."""
        calls = self.budget.max_model_calls - self.calls
        tokens = self.budget.max_total_tokens - self.tokens
        seconds = self.budget.max_duration_s - self.elapsed()
        if self.broken or calls < 1 or tokens < 1 or seconds <= 0:
            return None
        return Budget(max_model_calls=calls, max_total_tokens=tokens, max_duration_s=seconds)

    def refusal(self, tokens: int, seconds: float) -> str | None:
        if self.broken:
            return self.broken
        if self.calls + 1 > self.budget.max_model_calls:
            return "max_model_calls"
        if self.tokens + tokens > self.budget.max_total_tokens:
            return "max_total_tokens"
        if self.elapsed() + seconds > self.budget.max_duration_s:
            return "max_duration_s"
        return None


@dataclass
class CallRecord:
    attempt: int
    outcome: str  # ok | error
    latency_ms: float
    estimated_prompt_tokens: int
    code: str | None = None  # error: código fijo (ver providers.classify)
    origin: str | None = None  # error: provider | model
    status: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_reported: bool = False
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    text_tool_call: bool = False
    finish_reason: str | None = None
    model_name: str | None = None
    system_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prompt_upper_bound(messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]) -> int:
    """Cota superior conservadora de los tokens del prompt (ver el docstring del módulo)."""
    size = len(json.dumps(list(tools), ensure_ascii=False).encode("utf-8"))
    for m in messages:
        size += len(str(m.content).encode("utf-8")) + PER_MESSAGE_TOKENS
        for call in getattr(m, "tool_calls", None) or []:
            size += len(json.dumps(call, ensure_ascii=False, default=str).encode("utf-8"))
    return size + PROMPT_OVERHEAD_TOKENS


# --- Formato de la respuesta original ------------------------------------------------------------

_TAG = re.compile(r"<\s*/?\s*(tool_call|function_call|tool_use)\b", re.IGNORECASE)


def detect_text_tool_call(text: str, tool_names: Iterable[str]) -> bool:
    """¿El modelo escribió una tool call como texto en lugar de usar tool calling? Nunca se
    ejecuta: el grafo solo ejecuta ``AIMessage.tool_calls``."""
    names = set(tool_names)
    if not text or _TAG.search(text):
        return bool(text)
    if any(re.search(rf"\bfunctions\.{re.escape(n)}\b", text) for n in names):
        return True
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, dict):
            function = value.get("function")
            name = function.get("name") if isinstance(function, dict) else value.get("name")
            has_args = any(k in value for k in ("arguments", "parameters", "args"))
            if name in names and (has_args or isinstance(function, dict)):
                return True
    return False


def response_format(response: AIMessage, tool_names: Iterable[str]) -> dict[str, Any]:
    meta = response.response_metadata or {}
    content = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "tool_calls": len(response.tool_calls),
        "invalid_tool_calls": len(response.invalid_tool_calls),
        "text_tool_call": detect_text_tool_call(content, tool_names),
        "finish_reason": meta.get("finish_reason"),
        "model_name": meta.get("model_name"),
        "system_fingerprint": meta.get("system_fingerprint"),
    }


def count_null_padding(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Propuestas de ``update_task`` con campos en ``null`` explícito. La semántica no cambia
    (``assignee_id: null`` desasigna): se mide aparte de los intentos de ataque."""
    proposals = [c for c in calls if c.get("tool") == "update_task"]
    fields: Counter[str] = Counter()
    with_nulls = 0
    for c in proposals:
        nulls = [k for k, v in (c.get("changes") or {}).items() if v is None]
        with_nulls += bool(nulls)
        fields.update(nulls)
    return {"proposals": len(proposals), "with_nulls": with_nulls, "null_fields": dict(fields)}


# --- El modelo -----------------------------------------------------------------------------------


def _usage(response: AIMessage) -> tuple[int | None, int | None, int | None]:
    usage = response.usage_metadata
    if not usage:
        return None, None, None
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    return usage.get("input_tokens"), usage.get("output_tokens"), reasoning


@dataclass
class LiveModel:
    """Envuelve un chat model de LangChain (``ChatGroq`` o, en dry-run, uno simulado)."""

    inner: Any
    limits: CallLimits
    meters: tuple[Meter, ...]
    sleep: Callable[[float], Any] = anyio.sleep
    records: list[CallRecord] = field(default_factory=list)
    _bound: Any = None
    _tools: list[dict[str, Any]] = field(default_factory=list)

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> LiveModel:
        self._tools = list(tools)
        self._bound = self.inner.bind_tools(tools, **kwargs)
        return self

    def _tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in self._tools}

    def _reserve(self, estimate: int) -> None:
        worst = estimate + self.limits.max_output_tokens
        for meter in self.meters:
            if reason := meter.refusal(worst, self.limits.call_timeout_s):
                raise BudgetExceeded(reason)
        for meter in self.meters:
            meter.calls += 1

    def _charge(self, tokens: int) -> None:
        for meter in self.meters:
            meter.tokens += tokens

    def _break(self, reason: str) -> None:
        for meter in self.meters:
            meter.broken = reason

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        if self._bound is None:
            raise RuntimeError("LiveModel sin bind_tools")
        attempt = 0
        while True:
            estimate = prompt_upper_bound(messages, self._tools)
            self._reserve(estimate)
            start = time.perf_counter()
            failure: ProviderFailure | None = None
            try:
                with anyio.fail_after(self.limits.call_timeout_s):
                    response = await self._bound.ainvoke(messages)
            except Exception as exc:
                failure = classify(exc)
            latency = round((time.perf_counter() - start) * 1000, 1)
            if failure is None:
                return self._accept(response, attempt, latency, estimate)
            self._charge(estimate + self.limits.max_output_tokens)  # uso desconocido: peor caso
            self.records.append(
                CallRecord(
                    attempt=attempt,
                    outcome="error",
                    latency_ms=latency,
                    estimated_prompt_tokens=estimate,
                    code=failure.code,
                    origin=failure.origin,
                    status=failure.status,
                )
            )
            wait = self._retry_wait(failure, attempt)
            if wait is None:
                raise ProviderError(failure)
            await self.sleep(wait)
            attempt += 1

    def _retry_wait(self, failure: ProviderFailure, attempt: int) -> float | None:
        if not failure.retryable or attempt >= self.limits.max_retries:
            return None
        wait = failure.retry_after_s
        if wait is None:
            wait = self.limits.backoff_initial_s * 2**attempt
        if wait > self.limits.max_retry_wait_s:
            return None  # respetar Retry-After significa no llamar antes
        needed = wait + self.limits.call_timeout_s
        if any(m.elapsed() + needed > m.budget.max_duration_s for m in self.meters):
            return None
        return wait

    def _accept(self, response: AIMessage, attempt: int, latency: float, estimate: int):
        input_tokens, output_tokens, reasoning = _usage(response)
        reported = input_tokens is not None and output_tokens is not None
        if reported:
            self._charge(input_tokens + output_tokens)  # type: ignore[operator]
            if input_tokens > estimate:  # type: ignore[operator]
                self._break("prompt_estimate_exceeded")
            if output_tokens > self.limits.max_output_tokens:  # type: ignore[operator]
                self._break("output_cap_exceeded")
        else:
            self._charge(estimate + self.limits.max_output_tokens)
        self.records.append(
            CallRecord(
                attempt=attempt,
                outcome="ok",
                latency_ms=latency,
                estimated_prompt_tokens=estimate,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning,
                usage_reported=reported,
                **response_format(response, self._tool_names()),
            )
        )
        return response
