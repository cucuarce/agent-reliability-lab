"""``normalize_response``: la identidad de las tool calls la fija el agente (riesgo R1)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from arl.agent.history import normalize_response

HISTORY = [
    HumanMessage("hola"),
    AIMessage("", tool_calls=[{"id": "call_0_0", "name": "list_projects", "args": {}}]),
    ToolMessage("[]", tool_call_id="call_0_0"),
]


def _call(call_id: str | None, name: str = "get_project") -> dict:
    return {"id": call_id, "name": name, "args": {"project_id": "apollo"}, "type": "tool_call"}


def test_ids_depend_only_on_the_position_in_the_thread() -> None:
    response = AIMessage("", tool_calls=[_call("call_0"), _call("call_0"), _call(None)], id="x")
    normalized = normalize_response(response, HISTORY)
    assert [c["id"] for c in normalized.tool_calls] == ["call_1_0", "call_1_1", "call_1_2"]
    assert normalized.id is None  # add_messages le asigna uno nuevo
    assert normalize_response(response, HISTORY).tool_calls == normalized.tool_calls  # estable


def test_scripted_model_ids_are_unchanged() -> None:
    response = AIMessage("", tool_calls=[_call("call_1_0")])
    assert normalize_response(response, HISTORY).tool_calls[0]["id"] == "call_1_0"


def test_provider_copies_and_invalid_calls_are_dropped_but_usage_is_kept() -> None:
    usage = {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
    response = AIMessage(
        "",
        tool_calls=[_call("provider-id")],
        invalid_tool_calls=[{"id": "bad", "name": "update_task", "args": "{", "error": "json",
                             "type": "invalid_tool_call"}],
        additional_kwargs={"tool_calls": [{"id": "provider-id"}], "reasoning_content": "..."},
        response_metadata={"model_name": "m", "finish_reason": "tool_calls"},
        usage_metadata=usage,
    )  # fmt: skip
    normalized = normalize_response(response, HISTORY)
    assert normalized.additional_kwargs == {} and normalized.invalid_tool_calls == []
    assert normalized.usage_metadata == usage
    assert normalized.response_metadata["finish_reason"] == "tool_calls"
    assert normalized.tool_calls[0]["args"] == {"project_id": "apollo"}


def test_content_blocks_keep_only_text() -> None:
    response = AIMessage(
        [{"type": "text", "text": "Apollo "}, {"type": "tool_use", "id": "t", "name": "x"},
         {"type": "reasoning", "text": "secreto"}, {"type": "text", "text": "tiene 4 tareas."}]
    )  # fmt: skip
    assert normalize_response(response, HISTORY).content == "Apollo tiene 4 tareas."
