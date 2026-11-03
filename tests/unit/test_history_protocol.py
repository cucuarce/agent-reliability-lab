"""``check_tool_protocol`` y ``unanswered_tool_calls``: el contrato de tool calling."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from arl.agent.history import HistoryInconsistent, check_tool_protocol, unanswered_tool_calls


def _ai(*ids: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"id": i, "name": "list_projects", "args": {}} for i in ids])


def _tool(call_id: str) -> ToolMessage:
    return ToolMessage("ok", tool_call_id=call_id, name="list_projects")


def test_complete_history_is_accepted() -> None:
    check_tool_protocol([HumanMessage("hola"), _ai("a", "b"), _tool("a"), _tool("b"), _ai()])


@pytest.mark.parametrize(
    "messages",
    [
        [HumanMessage("hola"), _tool("x")],  # respuesta sin tool call
        [_ai("a"), _tool("a"), _tool("a")],  # respuesta duplicada
        [_ai("a", "b"), _tool("a"), HumanMessage("otra")],  # mensaje en medio
        [_ai("a")],  # pendiente al final
    ],
)
def test_contract_violations_are_refused(messages) -> None:
    with pytest.raises(HistoryInconsistent):
        check_tool_protocol(messages)


def test_unanswered_calls_only_look_at_the_last_assistant_message() -> None:
    assert [c["id"] for c in unanswered_tool_calls([_ai("a", "b"), _tool("a")])] == ["b"]
    assert unanswered_tool_calls([_ai("a"), HumanMessage("x")]) == []
    assert unanswered_tool_calls([HumanMessage("x")]) == []
