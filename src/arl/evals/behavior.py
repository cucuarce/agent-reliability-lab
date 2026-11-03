"""Modelo simulado de la evaluación: ``ScriptedChatModel`` más una rama de prompt injection.

Como ``ScriptedChatModel``, elige el paso según el historial (no con estado interno), así que es
determinístico y continúa en el mismo punto tras reanudar. Si el caso define ``on_injection``,
el modelo cambia de guion **solo** cuando el marker aparece en el resultado de una tool: la
escritura no solicitada demuestra que el texto malicioso llegó a su contexto, no que el guion
la tuviera desde el inicio.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from arl.agent.scripted_model import ScriptedChatModel, Step, say
from arl.evals.cases import ModelSpec, StepSpec


def _step(spec: StepSpec) -> Step:
    return Step(text=spec.text, calls=tuple((c.name, dict(c.args)) for c in spec.calls))


class CaseModel(ScriptedChatModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__([_step(s) for s in spec.steps], repeat_last=spec.repeat_last)
        self.injection = spec.on_injection
        self.injected_steps = [_step(s) for s in spec.on_injection.steps] if self.injection else []
        self.followed_injection = False

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        switch = self._injection_turn(messages)
        if switch is None:
            return await super().ainvoke(messages, config)
        self.prompts.append(list(messages))
        self.followed_injection = True
        turn = sum(isinstance(m, AIMessage) for m in messages)
        index = turn - switch
        step = self.injected_steps[index] if index < len(self.injected_steps) else say("(fin)")
        return AIMessage(
            content=step.text,
            tool_calls=[
                {"id": f"call_{turn}_{i}", "name": name, "args": args, "type": "tool_call"}
                for i, (name, args) in enumerate(step.calls)
            ],
        )

    def _injection_turn(self, messages: list[BaseMessage]) -> int | None:
        """Turno del modelo en el que vio el marker por primera vez (None si nunca lo vio)."""
        if self.injection is None:
            return None
        turns = 0
        for message in messages:
            if isinstance(message, AIMessage):
                turns += 1
            elif isinstance(message, ToolMessage) and self.injection.marker in str(message.content):
                return turns
        return None
