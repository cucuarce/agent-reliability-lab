"""Evaluación con LLM real (etapa 6B.1): clasificación de errores y formato de las respuestas.

Sin red: las excepciones son las reales del SDK ``groq``, construidas con respuestas ``httpx`` en
memoria. Un error del proveedor nunca se confunde con un error de formato del modelo, ni con un
fallo del agente, y su mensaje (que puede incluir la salida del modelo) nunca se conserva.
"""

from __future__ import annotations

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from arl.evals.live.model import count_null_padding, detect_text_tool_call
from arl.evals.live.providers import ProviderError, classify

URL = "https://api.groq.com/openai/v1/chat/completions"
SECRET = "texto-sensible-del-modelo"


def _status(cls, status: int, *, body=None, headers=None):
    request = httpx.Request("POST", URL)
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls(f"Error code: {status} - {SECRET}", response=response, body=body)


@pytest.mark.parametrize(
    ("exc", "code", "origin", "retryable", "aborts"),
    [
        # aborts: sobre el fallo final. Un 429 que persiste es una cuota agotada.
        (_status(groq.RateLimitError, 429), "rate_limited", "provider", True, True),
        (_status(groq.InternalServerError, 503), "server_error", "provider", True, False),
        (groq.APITimeoutError(httpx.Request("POST", URL)), "timeout", "provider", True, False),
        (TimeoutError(), "timeout", "provider", True, False),  # límite propio por llamada
        (
            groq.APIConnectionError(request=httpx.Request("POST", URL)),
            "connection",
            "provider",
            True,
            False,
        ),
        (_status(groq.AuthenticationError, 401), "auth", "provider", False, True),
        (_status(groq.PermissionDeniedError, 403), "forbidden", "provider", False, True),
        (_status(groq.NotFoundError, 404), "not_found", "provider", False, True),
        (_status(groq.APIStatusError, 413), "request_too_large", "provider", False, True),
        (
            _status(
                groq.BadRequestError,
                400,
                body={"error": {"code": "tool_use_failed", "failed_generation": SECRET}},
            ),
            "tool_use_failed",
            "model",
            False,
            False,
        ),
        (
            _status(groq.BadRequestError, 400, body={"code": "context_length_exceeded"}),
            "context_length",
            "provider",
            False,
            False,
        ),
        (
            _status(groq.BadRequestError, 400, body={"error": {"code": "otro"}}),
            "bad_request",
            "provider",
            False,
            False,
        ),
        (ValueError(SECRET), "unknown", "provider", False, False),
    ],
)
def test_provider_errors_are_classified_by_type_and_code(exc, code, origin, retryable, aborts):
    failure = classify(exc)
    assert (failure.code, failure.origin, failure.retryable, failure.aborts_suite) == (
        code,
        origin,
        retryable,
        aborts,
    )
    assert SECRET not in repr(failure)


def test_tool_use_failed_is_a_model_format_error_not_a_provider_failure() -> None:
    body = {"error": {"code": "tool_use_failed", "failed_generation": SECRET}}
    failure = classify(_status(groq.BadRequestError, 400, body=body))
    assert failure.origin == "model" and failure.status == 400


@pytest.mark.parametrize(
    ("headers", "seconds"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, None),
        ({}, None),
    ],
)
def test_retry_after_is_read_from_the_headers(headers, seconds) -> None:
    assert classify(_status(groq.RateLimitError, 429, headers=headers)).retry_after_s == seconds


def test_provider_error_keeps_a_fixed_message_and_no_context() -> None:
    failure = classify(_status(groq.RateLimitError, 429))
    error = ProviderError(failure)
    assert str(error) == "proveedor: rate_limited" and SECRET not in str(error)
    assert error.__cause__ is None and error.__context__ is None


# --- Formato de las respuestas del modelo --------------------------------------------------------

TOOLS = {"list_projects", "get_project", "update_task"}


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "update_task", "arguments": {"task_id": "apo-4"}}',
        'Llamo: ```json\n{"name": "get_project", "parameters": {"project_id": "apollo"}}\n```',
        '<tool_call>{"name": "list_projects", "arguments": {}}</tool_call>',
        '{"type": "function", "function": {"name": "update_task", "arguments": "{}"}}',
        'functions.update_task({"task_id": "apo-3"})',
    ],
)
def test_tool_calls_written_as_text_are_detected(text) -> None:
    assert detect_text_tool_call(text, TOOLS)


@pytest.mark.parametrize(
    "text",
    [
        "Apollo tiene 4 tareas; apo-4 está bloqueada.",
        '{"name": "Apollo", "tareas": 4}',
        "Podés usar update_task para cambiar una tarea.",
        "",
    ],
)
def test_ordinary_answers_are_not_tool_calls(text) -> None:
    assert not detect_text_tool_call(text, TOOLS)


def test_null_padding_counts_explicit_nulls_but_keeps_their_meaning() -> None:
    """``assignee_id: null`` explícito significa "desasignar" (semántica sin cambios): se mide
    aparte para no confundirlo con un ataque."""
    calls = [
        {"tool": "update_task", "changes": {"status": "done", "assignee_id": None, "title": None}},
        {"tool": "update_task", "changes": {"status": "done"}},
        {"tool": "get_project", "changes": None},
    ]
    assert count_null_padding(calls) == {
        "proposals": 2,
        "with_nulls": 1,
        "null_fields": {"assignee_id": 1, "title": 1},
    }


def test_invalid_tool_calls_are_counted_from_the_original_response() -> None:
    from arl.evals.live.model import response_format

    response = AIMessage(
        '{"name": "get_project", "arguments": {}}',
        invalid_tool_calls=[
            {
                "id": "x",
                "name": "update_task",
                "args": "{",
                "error": SECRET,
                "type": "invalid_tool_call",
            }
        ],
        response_metadata={"finish_reason": "stop", "model_name": "openai/gpt-oss-20b"},
    )
    fmt = response_format(response, TOOLS)
    assert fmt == {
        "tool_calls": 0,
        "invalid_tool_calls": 1,
        "text_tool_call": True,
        "finish_reason": "stop",
        "model_name": "openai/gpt-oss-20b",
        "system_fingerprint": None,
    }
