"""Etapa 6B.6 en la evaluación live: el modelo y la respuesta visible se evalúan por separado.

Los graders históricos (``answer_includes``, veracidad) siguen leyendo lo que escribió el
modelo: L11 no cambia y no pasa porque el agente arregle la respuesta. ``communication`` evalúa
aparte ``model`` (texto del LLM) y ``user`` (lo que ve el usuario). La transcripción conserva la
respuesta original del modelo y nunca le atribuye texto del agente. Sin red ni inferencia.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.evals.live import __main__ as cli
from arl.evals.live import suite as live_suite
from arl.evals.live import transcript
from arl.evals.live.cases import load_live_cases
from arl.evals.live.profiles import PROFILES
from arl.evals.live.suite import run_live_suite
from arl.evals.live.transcript import TranscriptWriter

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "live" / "cases"
REJECT_WRITE = PROFILES["reject-write"]
# La respuesta exacta de Groq en las dos corridas reales de L11.
L11_MODEL_REPLY = (
    "Para asignar la tarea **apo-3** a **bob**, necesito tu aprobación. "
    "¿Confirmas que quieres realizar este cambio?"
)
ASSIGN = {"task_id": "apo-3", "changes": {"assignee_id": "bob"}}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / "no-existe.env")
    monkeypatch.setenv("ARL_DATA_DIR", str(tmp_path / "data"))
    real = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"conexión de red no permitida: {host}")
        return real(self, address)

    def forbidden(*args, **kwargs):
        raise AssertionError("ningún test construye el cliente de Groq")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(live_suite, "make_groq_model", forbidden)


def _case(case_id: str):
    (case_file,) = [c for c in load_live_cases(CASES) if c.case.id == case_id]
    return case_file


class FakeGroq(ScriptedChatModel):
    async def ainvoke(self, messages: list, config: Any = None) -> AIMessage:
        scripted = await super().ainvoke(messages, config)
        return AIMessage(
            content=scripted.content,
            tool_calls=[{**c, "id": "call_provider"} for c in scripted.tool_calls],
            usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
        )


async def _run(reply: str):
    writer = TranscriptWriter.create("run-replies", root=ROOT)
    report = await run_live_suite(
        [_case("L11")],
        ROOT,
        REJECT_WRITE,
        mode="live",
        chat_factory=lambda c: FakeGroq([call("update_task", **ASSIGN), say(reply)]),
        transcript=writer,
    )
    events = [json.loads(x) for x in writer.path.read_text(encoding="utf-8").splitlines()]
    return report["results"][0], report, events


async def test_l11_with_the_real_groq_reply_still_fails_on_the_model(tmp_path) -> None:
    """El arreglo del agente no hace pasar a L11: el grader histórico mide al modelo."""
    result, _, _ = await _run(L11_MODEL_REPLY)
    graders = {g["name"]: g["status"] for g in result["graders"]}
    assert (result["verdict"], result["error_class"]) == ("failed", "model")
    assert graders["answer_includes"] == "fail"
    assert all(s == "pass" for n, s in graders.items() if n != "answer_includes")
    assert result["communication"]["model"]["verdict"] == "omitted"
    assert result["communication"]["model"]["review"] is True


async def test_the_user_sees_an_authoritative_rejection_even_with_the_real_reply(tmp_path):
    result, _, _ = await _run(L11_MODEL_REPLY)
    user = result["communication"]["user"]
    assert (user["expected_outcome"], user["verdict"], user["review"]) == (
        "rejected",
        "communicated",
        False,
    )
    observation = result["observation"]
    assert observation["reply_source"] == "agent"
    assert observation["model_reply_chars"] == len(L11_MODEL_REPLY)
    assert observation["reply_chars"] == len(L11_MODEL_REPLY)  # campo histórico: el modelo
    assert observation["user_reply_chars"] > 0
    (outcome,) = observation["write_outcomes"]
    assert (outcome["task_id"], outcome["state"], outcome["executed"], outcome["approval"]) == (
        "apo-3",
        "rejected",
        False,
        "reject",
    )
    assert outcome["code"] == "rejected_by_user"


async def test_transcript_keeps_the_original_model_reply_and_the_user_reply_apart(tmp_path):
    _, report, events = await _run(L11_MODEL_REPLY)
    (final,) = [e for e in events if e["event"] == "final"]
    assert final["model_reply"] == L11_MODEL_REPLY  # la respuesta original, intacta
    assert final["reply_source"] == "agent"
    assert "rechaz" in final["user_reply"] and final["user_reply"] != L11_MODEL_REPLY
    assert final["write_outcomes"][0]["state"] == "rejected"
    model_texts = [e["content"] for e in events if e["event"] == "model_response"]
    assert final["user_reply"] not in model_texts  # el texto del agente nunca figura como LLM
    assert events[0]["schema"] == transcript.SCHEMA == "arl.live-transcript/2"
    metrics = report["metrics"]["communication"]
    assert metrics["model_by_verdict"] == {"omitted": 1}
    assert metrics["user_by_verdict"] == {"communicated": 1}


async def test_a_model_that_communicates_the_rejection_passes_both(tmp_path) -> None:
    result, _, _ = await _run("Rechazaste la asignación de apo-3 a bob: la tarea no cambió.")
    assert result["verdict"] == "passed"
    assert result["communication"]["model"]["verdict"] == "communicated"
    assert result["communication"]["user"]["verdict"] == "communicated"


def test_l11_and_its_expectations_are_unchanged() -> None:
    case = _case("L11").case
    assert case.approval == "reject" and case.request == "Asigná la tarea apo-3 a bob."
    assert case.expected.answer_any == (("rechaz",),)
    assert case.expected.audit_rows == 0 and case.expected.max_write_proposals == 2
