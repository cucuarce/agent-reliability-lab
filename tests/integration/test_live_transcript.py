"""Etapa 6B.5: transcripción local de cada corrida y evidencia de persistencia en el reporte.

Ningún test hace inferencia ni usa red. La transcripción (``transcript.jsonl``) guarda lo
visible de la corrida (pedido, respuestas, tool calls con argumentos y resultados, decisiones,
bloqueos, errores y métricas por llamada) con campos explícitos por evento, nunca el
razonamiento interno ni credenciales, y solo si los datos son los sintéticos verificados.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.domain.models import Task
from arl.evals.live import __main__ as cli
from arl.evals.live import runner as live_runner
from arl.evals.live import suite as live_suite
from arl.evals.live import transcript
from arl.evals.live.cases import load_live_cases
from arl.evals.live.model import Budget
from arl.evals.live.profiles import PROFILES, LiveConfig
from arl.evals.live.suite import run_live_suite
from arl.evals.live.transcript import TranscriptUnavailable, TranscriptWriter

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "live" / "cases"
CANARY = "gsk_canary_" + "t" * 40
REASONING = "RAZONAMIENTO-PRIVADO-canario"
REJECT_WRITE = PROFILES["reject-write"]
FIRST_WRITE = PROFILES["first-write"]
ASSIGN = {"task_id": "apo-3", "changes": {"assignee_id": "bob"}}
L11_REPLY = "Rechazaste la asignación de apo-3 a bob, así que la tarea quedó sin cambios."
L11_SCRIPT = [call("update_task", **ASSIGN), say(L11_REPLY)]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Sin ``.env`` real, sin red, sin Groq y con ``ARL_DATA_DIR`` temporal."""
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
    """Como ``ChatGroq``: tokens informados y, además, razonamiento del proveedor que la
    transcripción nunca debe copiar."""

    async def ainvoke(self, messages: list, config: Any = None) -> AIMessage:
        scripted = await super().ainvoke(messages, config)
        return AIMessage(
            content=scripted.content,
            tool_calls=[{**c, "id": "call_provider"} for c in scripted.tool_calls],
            additional_kwargs={"reasoning_content": REASONING},
            response_metadata={"headers": {"authorization": f"Bearer {CANARY}"}},
            usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
        )


def _factory(script):
    return lambda c: FakeGroq(script)


def _writer(tmp_path) -> TranscriptWriter:
    return TranscriptWriter.create("run-test", root=ROOT)


def _events(writer: TranscriptWriter) -> list[dict[str, Any]]:
    return [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]


async def _run(script, case_id="L11", config=REJECT_WRITE, tmp_path=None, **kwargs):
    writer = _writer(tmp_path)
    report = await run_live_suite(
        [_case(case_id)],
        ROOT,
        config,
        mode="live",
        chat_factory=_factory(script),
        transcript=writer,
        **kwargs,
    )
    return report, writer, _events(writer)


def _of(events, kind):
    return [e for e in events if e["event"] == kind]


# --- Contenido -----------------------------------------------------------------------------------


async def test_transcript_records_the_visible_run_of_l11(tmp_path) -> None:
    report, writer, events = await _run(L11_SCRIPT, tmp_path=tmp_path)
    assert writer.path.name == "transcript.jsonl"
    assert events[0]["event"] == "header" and events[0]["schema"] == transcript.SCHEMA
    (start,) = _of(events, "scenario_start")
    assert (start["case_id"], start["repetition"], start["synthetic"]) == ("L11", 0, True)
    assert start["approval_policy"] == {
        "decision": "reject",
        "source": "case",
        "actor": "simulated",
    }
    (request,) = _of(events, "user_request")
    assert request["text"] == _case("L11").case.request
    responses = _of(events, "model_response")
    assert [r["turn"] for r in responses] == [1, 2]
    assert responses[0]["tool_calls"][0]["name"] == "update_task"
    assert responses[0]["tool_calls"][0]["args"] == ASSIGN
    assert responses[-1]["content"] == L11_REPLY  # la respuesta final completa
    (approval,) = _of(events, "approval")
    assert approval["decision"] == "reject" and approval["actor"] == "simulated"
    assert approval["source"] == "case" and approval["task_id"] == "apo-3"
    (result,) = _of(events, "tool_result")
    assert result["is_error"] and "rejected_by_user" in result["content"]
    calls = _of(events, "model_call")
    assert [(c["call_index"], c["turn"], c["attempt"]) for c in calls] == [(1, 1, 0), (2, 2, 0)]
    assert calls[0]["input_tokens"] == 900 and calls[0]["latency_ms"] >= 0
    (final,) = _of(events, "final")
    assert final["model_reply"] == L11_REPLY and final["final_status"] == "completed"
    (evaluation,) = _of(events, "evaluation")
    assert evaluation["verdict"] == report["results"][0]["verdict"]
    assert evaluation["communication"]["model"]["verdict"] == "communicated"
    assert all(e["case_id"] == "L11" for e in events[1:] if "case_id" in e)
    assert report["metadata"]["transcript"]["status"] == "written"


async def test_transcript_never_copies_reasoning_headers_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    script = [call("update_task", **ASSIGN), say(f"Rechazado en apo-3. {CANARY}")]
    report, writer, events = await _run(script, tmp_path=tmp_path)
    raw = writer.path.read_text(encoding="utf-8")
    assert REASONING not in raw and CANARY not in raw
    assert "authorization" not in raw.casefold() and "GROQ_API_KEY" not in raw
    assert "[redactado]" in _of(events, "final")[0]["model_reply"]
    assert writer.describe()["redactions"] >= 1
    assert CANARY not in json.dumps(report)


def test_every_event_uses_an_explicit_allow_list_of_fields(tmp_path) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match="campos no permitidos"):
        writer.write({"event": "final", "reply": "x", "env": {"PATH": "C:/"}})
    with pytest.raises(ValueError, match="evento desconocido"):
        writer.write({"event": "provider_object", "raw": "x"})


async def test_scope_blocks_and_tool_arguments_are_recorded(tmp_path) -> None:
    script = [call("update_task", task_id="apo-3", changes={"assignee_id": "carol"}), say("no")]
    _, _, events = await _run(script, tmp_path=tmp_path)
    (block,) = _of(events, "scope_block")
    assert block["reason"] == "value" and block["tool"] == "update_task"
    assert _of(events, "model_response")[0]["tool_calls"][0]["args"]["changes"] == {
        "assignee_id": "carol"
    }
    assert _of(events, "approval") == []


async def test_transcript_is_kept_when_the_budget_runs_out(tmp_path, monkeypatch) -> None:
    one_call = Budget(max_model_calls=1, max_total_tokens=40_000, max_duration_s=240)
    tight = REJECT_WRITE.model_copy(update={"scenario_budget": one_call})
    monkeypatch.setitem(PROFILES, "tight", tight)  # registrado: la autorización lo acepta
    report, _, events = await _run(L11_SCRIPT, config=tight, tmp_path=tmp_path)
    (error,) = _of(events, "error")
    assert error["kind"] == "budget_stop"
    assert _of(events, "final") and _of(events, "evaluation")
    assert report["results"][0]["cause"].startswith("budget:")


async def test_transcript_is_kept_when_a_tool_fails(tmp_path) -> None:
    script = [call("get_project", project_id="apollo"), *L11_SCRIPT]
    _, _, events = await _run(
        script, tmp_path=tmp_path, faults={("get_project", EVERY_CALL): Fault.TIMEOUT_BEFORE}
    )
    assert "interrupted" in _of(events, "final")[0]["statuses"]
    assert _of(events, "final")[0]["model_reply"] == L11_REPLY


async def test_a_scenario_that_aborts_still_leaves_its_events(tmp_path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise live_runner.IsolationError("el escenario solo corre en el directorio temporal")

    monkeypatch.setattr(live_runner, "_check_isolated", broken)
    writer = _writer(tmp_path)
    with pytest.raises(live_runner.IsolationError):
        await run_live_suite(
            [_case("L11")],
            ROOT,
            REJECT_WRITE,
            mode="live",
            chat_factory=_factory(L11_SCRIPT),
            transcript=writer,
        )
    (aborted,) = _of(_events(writer), "scenario_aborted")
    assert aborted["error_type"] == "IsolationError" and aborted["case_id"] == "L11"


# --- Datos sintéticos y ubicación ----------------------------------------------------------------


async def test_unverified_data_is_withheld_from_the_transcript(tmp_path, monkeypatch) -> None:
    """Si la base no tiene exactamente las tareas de la semilla empaquetada, no se puede
    garantizar que los datos sean sintéticos: la transcripción del escenario se retiene."""
    original = live_runner.seed_for

    def tampered(spec):
        data = original(spec)
        extra = Task(
            id="ext-1", project_id="apollo", title="dato externo", status="todo", version=1
        )
        return data.model_copy(update={"tasks": (*data.tasks, extra)})

    monkeypatch.setattr(live_runner, "seed_for", tampered)
    _, _, events = await _run(L11_SCRIPT, tmp_path=tmp_path)
    (withheld,) = _of(events, "withheld")
    assert withheld["reason"] == "synthetic_unverified"
    assert not _of(events, "user_request") and not _of(events, "final")
    assert "dato externo" not in json.dumps(events)


def test_transcripts_live_outside_the_repository_and_onedrive(tmp_path, monkeypatch) -> None:
    writer = _writer(tmp_path)
    path = writer.path.resolve()
    assert path.is_relative_to((tmp_path / "data").resolve())
    assert not path.is_relative_to(ROOT.resolve())
    monkeypatch.setenv("ARL_DATA_DIR", str(tmp_path / "OneDrive" / "arl"))
    with pytest.raises(TranscriptUnavailable, match="OneDrive"):
        TranscriptWriter.create("run-x", root=ROOT)
    monkeypatch.setenv("ARL_DATA_DIR", str(ROOT / "evals" / "no-debe-existir"))
    with pytest.raises(TranscriptUnavailable):
        TranscriptWriter.create("run-y", root=ROOT)
    assert not (ROOT / "evals" / "no-debe-existir").exists()


# --- Evidencia de persistencia en el reporte -----------------------------------------------------


async def test_report_shows_persistence_evidence_for_a_rejection(tmp_path) -> None:
    report, _, _ = await _run(L11_SCRIPT, tmp_path=tmp_path)
    persistence = report["results"][0]["observation"]["persistence"]
    apo3 = persistence["tasks"]["apo-3"]
    assert apo3["before"] == apo3["after"]
    assert set(apo3["before"]) == {"status", "assignee_id", "version"}
    assert persistence["audit"] == []
    assert persistence["ledger"] == {"entries": 0, "outcomes": {}, "keys": []}
    assert persistence["writes"] == {"effective": 0, "unchanged": 0, "replays": 0}


async def test_report_shows_persistence_evidence_for_an_approved_write_with_a_replay(tmp_path):
    script = [call("update_task", task_id="apo-3", changes={"status": "in_progress"}), say("ok")]
    report, _, events = await _run(
        script,
        case_id="L08",
        config=FIRST_WRITE,
        tmp_path=tmp_path,
        faults={("update_task", 0): Fault.TIMEOUT_AFTER},  # se pierde la respuesta: reenvío
    )
    persistence = report["results"][0]["observation"]["persistence"]
    apo3 = persistence["tasks"]["apo-3"]
    assert (apo3["before"]["status"], apo3["after"]["status"]) == ("todo", "in_progress")
    assert apo3["after"]["version"] == apo3["before"]["version"] + 1
    (row,) = persistence["audit"]
    assert row["actor_id"] == "alice" and row["task_id"] == "apo-3" and row["own"] is True
    assert row["changes"] == {"status": "in_progress"}
    assert len(row["key"]) == 12  # huella, nunca la clave completa
    assert persistence["ledger"]["entries"] == 1
    assert persistence["ledger"]["outcomes"] == {"updated": 1}
    assert persistence["ledger"]["keys"] == [row["key"]]
    assert persistence["writes"] == {"effective": 1, "unchanged": 0, "replays": 1}
    dumped = json.dumps(report)
    full_keys = [r["idempotency_key"] for r in _of(events, "approval") if "idempotency_key" in r]
    assert full_keys == []  # la transcripción tampoco guarda claves completas
    assert row["key"] in dumped


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_dry_run_writes_the_transcript_by_default_outside_the_repo(tmp_path, capsys) -> None:
    out = tmp_path / "reports"
    assert cli.main(["dry-run", "--only", "L11", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    (report_dir,) = out.iterdir()
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    meta = report["metadata"]["transcript"]
    assert meta["status"] == "written" and meta["path"].startswith("<ARL_DATA_DIR>/")
    (path,) = (tmp_path / "data" / "live-transcripts").rglob("transcript.jsonl")
    assert meta["run_id"] in str(path) and "transcript:" in printed
    assert not any(report_dir.glob("*.jsonl"))  # el reporte (en el repo) no lleva la transcripción


def test_cli_can_disable_the_transcript_and_degrades_safely(tmp_path, monkeypatch, capsys):
    out = tmp_path / "reports"
    assert cli.main(["dry-run", "--only", "L11", "--out", str(out), "--no-transcript"]) == 0
    assert not (tmp_path / "data" / "live-transcripts").exists()
    monkeypatch.setenv("ARL_DATA_DIR", str(tmp_path / "OneDrive" / "arl"))
    assert cli.main(["dry-run", "--only", "L11", "--out", str(out)]) == 0
    assert "transcripción desactivada" in capsys.readouterr().err
    reports = sorted(out.iterdir())
    meta = json.loads((reports[-1] / "report.json").read_text(encoding="utf-8"))["metadata"]
    assert meta["transcript"]["status"] == "disabled"


async def test_historical_reports_without_new_fields_still_render() -> None:
    """Un reporte 6b1.6 (sin transcripción, comunicación ni persistencia) se sigue leyendo."""
    old = await run_live_suite([_case("L11")], ROOT, REJECT_WRITE, mode="dry-run")
    old["metadata"]["suite_version"] = "6b1.6"
    del old["metadata"]["transcript"], old["metrics"]["communication"]
    for result in old["results"]:
        del result["communication"], result["observation"]["persistence"]
    text = cli.render(old)
    assert "communication" not in text and "L11" in text


def test_config_has_no_field_for_transcripts_or_databases() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        LiveConfig(transcript_dir="C:/datos")
