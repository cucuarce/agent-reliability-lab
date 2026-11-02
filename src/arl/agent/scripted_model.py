"""LLM falso y determinístico para tests y para la suite de evaluación sin costo.

Elige la respuesta según cuántas respuestas del asistente ya hay en el historial, no con un
contador interno: así una instancia nueva (p. ej. tras reiniciar el proceso) continúa en el mismo
punto del guion. Los IDs de tool call también son determinísticos.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


@dataclass(frozen=True)
class Step:
    text: str = ""
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()


def say(text: str) -> Step:
    return Step(text=text)


def call(name: str, **args: Any) -> Step:
    return Step(calls=((name, args),))


@dataclass
class ScriptedChatModel:
    steps: Sequence[Step]
    repeat_last: bool = False  # para simular un modelo que entra en bucle
    bound_tools: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[list[BaseMessage]] = field(default_factory=list)

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        self.prompts.append(list(messages))
        turn = sum(isinstance(m, AIMessage) for m in messages)
        if turn < len(self.steps):
            step = self.steps[turn]
        elif self.repeat_last and self.steps:
            step = self.steps[-1]
        else:
            step = say("(fin del guion)")
        return AIMessage(
            content=step.text,
            tool_calls=[
                {"id": f"call_{turn}_{i}", "name": name, "args": args, "type": "tool_call"}
                for i, (name, args) in enumerate(step.calls)
            ],
        )
