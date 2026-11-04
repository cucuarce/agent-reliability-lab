"""``LiveModel``: presupuesto verificado antes de cada intento, reintentos acotados y medición."""

from __future__ import annotations

from typing import Any

import anyio
import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from arl.agent.tool_registry import tool_definitions
from arl.evals.live.model import (
    Budget,
    BudgetExceeded,
    CallLimits,
    LiveModel,
    Meter,
    prompt_upper_bound,
)
from arl.evals.live.providers import ProviderError

pytestmark = pytest.mark.anyio
MESSAGES = [HumanMessage("¿Qué proyectos veo?")]
URL = "https://api.groq.com/openai/v1/chat/completions"
USAGE = {"input_tokens": 900, "output_tokens": 40, "total_tokens": 940}


def _rate_limited(retry_after: str | None = None) -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", URL))
    return groq.RateLimitError("limit", response=response, body=None)


def _auth() -> groq.AuthenticationError:
    response = httpx.Response(401, request=httpx.Request("POST", URL))
    return groq.AuthenticationError("bad key", response=response, body=None)


class FakeChat:
    """Chat model falso: cada llamada consume el siguiente elemento (respuesta o excepción)."""

    def __init__(self, outcomes: list[Any], delay: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.delay = delay

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> FakeChat:
        return self

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        self.calls += 1
        if self.delay:
            await anyio.sleep(self.delay)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _ok(**kwargs: Any) -> AIMessage:
    return AIMessage(
        "Apollo y Cassini.",
        usage_metadata=USAGE,
        response_metadata={"finish_reason": "stop"},
        **kwargs,
    )


def _model(
    inner: FakeChat, *, budget: Budget | None = None, sleeps: list[float] | None = None, **limits
) -> LiveModel:
    budget = budget or Budget(max_model_calls=10, max_total_tokens=100_000, max_duration_s=60)
    waits = sleeps if sleeps is not None else []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    model = LiveModel(inner, CallLimits(call_timeout_s=5, **limits), (Meter(budget),), sleep=sleep)
    return model.bind_tools(tool_definitions())


async def test_successful_call_records_usage_latency_and_format_without_content() -> None:
    model = _model(FakeChat([_ok()]))
    response = await model.ainvoke(MESSAGES)
    assert response.content == "Apollo y Cassini."
    (record,) = model.records
    assert (record.outcome, record.input_tokens, record.output_tokens) == ("ok", 900, 40)
    assert record.usage_reported and record.finish_reason == "stop"
    assert "Apollo" not in str(record.to_dict())  # nunca el contenido
    assert model.meters[0].tokens == 940 and model.meters[0].calls == 1


async def test_a_call_that_may_not_fit_the_budget_is_never_made() -> None:
    estimate = prompt_upper_bound(MESSAGES, tool_definitions())
    tight = Budget(max_model_calls=10, max_total_tokens=estimate + 100, max_duration_s=60)
    inner = FakeChat([_ok()])
    model = _model(inner, budget=tight, max_output_tokens=1024)  # peor caso: estimate + 1024
    with pytest.raises(BudgetExceeded, match="max_total_tokens"):
        await model.ainvoke(MESSAGES)
    assert inner.calls == 0


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (Budget(max_model_calls=1, max_total_tokens=100_000, max_duration_s=60), "max_model_calls"),
        (Budget(max_model_calls=9, max_total_tokens=100_000, max_duration_s=4), "max_duration_s"),
    ],
)
async def test_calls_and_duration_are_checked_before_each_attempt(budget, reason) -> None:
    inner = FakeChat([_ok(), _ok()])
    model = _model(inner, budget=budget)  # call_timeout_s=5
    if reason == "max_model_calls":
        await model.ainvoke(MESSAGES)
    with pytest.raises(BudgetExceeded, match=reason):
        await model.ainvoke(MESSAGES)
    assert inner.calls == (1 if reason == "max_model_calls" else 0)


async def test_rate_limit_is_retried_honoring_retry_after_and_counted() -> None:
    sleeps: list[float] = []
    inner = FakeChat([_rate_limited("3"), _ok()])
    model = _model(inner, sleeps=sleeps)
    await model.ainvoke(MESSAGES)
    assert sleeps == [3.0] and inner.calls == 2
    assert [r.code for r in model.records] == ["rate_limited", None]
    assert model.meters[0].calls == 2  # cada intento cuenta


async def test_retries_are_bounded_and_then_raise_a_fixed_provider_error() -> None:
    inner = FakeChat([_rate_limited(), _rate_limited(), _rate_limited(), _ok()])
    sleeps: list[float] = []
    model = _model(inner, sleeps=sleeps, max_retries=2, backoff_initial_s=1)
    with pytest.raises(ProviderError) as exc:
        await model.ainvoke(MESSAGES)
    assert exc.value.failure.code == "rate_limited" and inner.calls == 3
    assert sleeps == [1, 2]  # backoff exponencial sin Retry-After
    assert exc.value.__context__ is None and exc.value.__cause__ is None


async def test_a_retry_after_longer_than_allowed_is_not_shortened() -> None:
    """Con duración de sobra, solo ``max_retry_wait_s`` decide: no se acorta la espera pedida."""
    inner = FakeChat([_rate_limited("120"), _ok()])
    roomy = Budget(max_model_calls=10, max_total_tokens=100_000, max_duration_s=3600)
    sleeps: list[float] = []
    model = _model(inner, budget=roomy, sleeps=sleeps, max_retry_wait_s=20)
    with pytest.raises(ProviderError):
        await model.ainvoke(MESSAGES)
    assert inner.calls == 1 and sleeps == []  # ni se espera ni se reintenta
    inner = FakeChat([_rate_limited("15"), _ok()])
    model = _model(inner, budget=roomy, sleeps=sleeps, max_retry_wait_s=20)
    await model.ainvoke(MESSAGES)
    assert sleeps == [15.0] and inner.calls == 2  # dentro del máximo: se respeta tal cual


async def test_auth_errors_are_never_retried() -> None:
    inner = FakeChat([_auth(), _ok()])
    model = _model(inner)
    with pytest.raises(ProviderError) as exc:
        await model.ainvoke(MESSAGES)
    assert exc.value.failure.aborts_suite and inner.calls == 1


async def test_each_call_has_a_hard_timeout() -> None:
    inner = FakeChat([_ok(), _ok()], delay=10)
    model = _model(inner, max_retries=0)
    model.limits = model.limits.model_copy(update={"call_timeout_s": 0.05})
    with pytest.raises(ProviderError) as exc:
        await model.ainvoke(MESSAGES)
    assert exc.value.failure.code == "timeout"


async def test_unreported_usage_and_failures_charge_the_worst_case() -> None:
    estimate = prompt_upper_bound(MESSAGES, tool_definitions())
    model = _model(FakeChat([AIMessage("sin uso informado")]), max_output_tokens=100)
    await model.ainvoke(MESSAGES)
    assert model.meters[0].tokens == estimate + 100 and not model.records[0].usage_reported


@pytest.mark.parametrize(
    ("usage", "reason"),
    [
        (
            {"input_tokens": 10**6, "output_tokens": 1, "total_tokens": 10**6 + 1},
            "prompt_estimate_exceeded",
        ),
        ({"input_tokens": 10, "output_tokens": 5000, "total_tokens": 5010}, "output_cap_exceeded"),
    ],
)
async def test_if_the_bound_proves_wrong_no_further_call_is_made(usage, reason) -> None:
    inner = FakeChat([AIMessage("x", usage_metadata=usage), _ok()])
    model = _model(
        inner, budget=Budget(max_model_calls=10, max_total_tokens=10**7, max_duration_s=60)
    )
    await model.ainvoke(MESSAGES)
    with pytest.raises(BudgetExceeded, match=reason):
        await model.ainvoke(MESSAGES)
    assert inner.calls == 1


def test_scenario_admission_requires_the_whole_scenario_budget() -> None:
    clock: list[float] = [0.0]
    suite = Meter(
        Budget(max_model_calls=12, max_total_tokens=60_000, max_duration_s=600),
        clock=lambda: clock[0],
    )
    scenario = Budget(max_model_calls=6, max_total_tokens=30_000, max_duration_s=180)
    assert suite.remaining().covers(scenario)
    suite.calls, suite.tokens = 7, 0
    assert not suite.remaining().covers(scenario)  # quedan 5 llamadas: no se empieza
    suite.calls, clock[0] = 0, 500.0
    assert not suite.remaining().covers(scenario)  # quedan 100 s
