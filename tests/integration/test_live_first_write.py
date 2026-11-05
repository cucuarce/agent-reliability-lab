"""Etapa 6B.3: habilitación controlada de la primera escritura con un modelo real (L08).

Ningún test hace inferencia ni usa red: el "modelo real" es un chat model falso que reemplaza a
``ChatGroq``. Se verifica que solo L08, solo con el perfil ``first-write`` y solo con los dos
flags de la CLI, pueda escribir; y que el alcance, la aprobación simulada, la idempotencia y el
aislamiento no se debiliten.
"""

from __future__ import annotations

import json
import socket
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from arl.agent import scope as agent_scope
from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.config import ConfigError
from arl.evals.live import __main__ as cli
from arl.evals.live import cases as live_cases
from arl.evals.live import profiles as live_profiles
from arl.evals.live import runner as live_runner
from arl.evals.live import suite as live_suite
from arl.evals.live.cases import ApprovalPolicyMissing, load_live_cases
from arl.evals.live.model import CallLimits, LiveModel, Meter
from arl.evals.live.profiles import PROFILES, LiveConfig, WritesDisabled
from arl.evals.live.runner import IsolationError, run_live_case
from arl.evals.live.suite import run_live_suite
from arl.storage import ledger
from arl.storage.db import connect

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "live" / "cases"
CANARY = "gsk_canary_" + "w" * 40
FIRST_WRITE = PROFILES["first-write"]
BOTH = ("--confirm-real-calls", "--confirm-real-writes")
UPDATE = {"task_id": "apo-3", "changes": {"status": "in_progress"}}
L08_SCRIPT = [call("update_task", **UPDATE), say("Listo: apo-3 quedó en in_progress.")]
OTHER_WRITES = ("L09", "L10", "L11", "L12", "L13", "L14")
OUTSIDE_THE_LIST = ("L12", "L13")  # L11 (6B.4), L10 (6B.7), L09 y L14 (6B.8) están en la lista


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Sin ``.env`` real, sin red y sin cliente de Groq: 0 llamadas al proveedor."""
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / "no-existe.env")
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
    """Reemplaza a ``ChatGroq``: guion fijo, uso de tokens informado, sin red."""

    async def ainvoke(self, messages: list, config: Any = None) -> AIMessage:
        scripted = await super().ainvoke(messages, config)
        return AIMessage(
            content=scripted.content,
            tool_calls=[{**c, "id": "call_provider"} for c in scripted.tool_calls],
            usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
        )


def _never(c):
    raise AssertionError("no debía crear el cliente del modelo")


def _model(script, config: LiveConfig = FIRST_WRITE) -> LiveModel:
    return LiveModel(FakeGroq(script), CallLimits(), (Meter(config.scenario_budget),))


def _audit_rows(workdir: Path) -> list:
    with closing(connect(workdir / "app.db")) as conn:
        return ledger.list_audit(conn)


# --- Lista cerrada, perfil y configuración -----------------------------------------------------


def test_only_first_write_enables_l08_and_only_to_approve() -> None:
    # 6B.4 agregó L11 (solo reject, perfil reject-write): test_live_reject_write.py
    assert live_cases.REAL_WRITE_CASES["L08"] == frozenset({"approve"})
    assert FIRST_WRITE.real_writes == ("L08",)
    assert [n for n, p in PROFILES.items() if "L08" in p.real_writes] == ["first-write"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allowed_cases": ("L12",), "real_writes": ("L12",)}, "lista cerrada"),
        ({"allowed_cases": ("L01",), "real_writes": ("L08",)}, "no autoriza"),
        ({"real_writes": ("L08",)}, "no autoriza"),
        ({"db_path": "C:/datos/app.db"}, "Extra inputs"),  # ninguna base persistente o externa
        ({"data_dir": "C:/datos"}, "Extra inputs"),
    ],
    ids=["amplia-la-lista", "no-autorizado", "sin-allowed", "db_path", "data_dir"],
)
def test_a_profile_can_never_widen_the_closed_list_or_point_to_a_database(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LiveConfig(**kwargs)


async def test_l08_writes_under_first_write_with_a_fake_provider() -> None:
    """La única combinación habilitada, invocada desde Python: escribe una vez, en la SQLite
    temporal, con la aprobación simulada declarada por el perfil."""
    fakes: list[FakeGroq] = []

    def factory(c):
        fakes.append(FakeGroq(L08_SCRIPT))
        return fakes[-1]

    report = await run_live_suite(
        [_case("L08")], ROOT, FIRST_WRITE, mode="live", chat_factory=factory
    )
    (result,) = report["results"]
    assert (result["verdict"], result["error_class"]) == ("passed", None)
    assert result["approval_policy"] == {
        "decision": "approve",
        "source": "profile",
        "actor": "simulated",
    }
    assert result["observation"]["audit_rows"] == 1
    assert all(g["status"] == "pass" for g in result["graders"] if g["kind"] == "safety")
    assert len(fakes) == 1 and len(fakes[0].prompts) == 2  # el falso, nunca Groq
    assert CANARY not in json.dumps(report)


def _unregistered(**update) -> LiveConfig:
    return FIRST_WRITE.model_copy(update=update)


@pytest.mark.parametrize(
    ("case_id", "config", "message"),
    [
        ("L08", PROFILES["third-batch"], "no habilita"),  # autorizado pero sin real_writes
        ("L08", LiveConfig(allowed_cases=("L08",), approvals={"L08": "approve"}), "no habilita"),
        ("L08", _unregistered(repeat=2), "perfil registrado"),
        ("L08", _unregistered(max_tool_calls=8), "perfil registrado"),
        *[(w, FIRST_WRITE, "lista cerrada") for w in OUTSIDE_THE_LIST],
        ("L11", FIRST_WRITE, "no habilita"),  # en la lista, pero first-write no lo habilita
        ("L10", FIRST_WRITE, "no habilita"),  # ídem (6B.7)
        ("L09", FIRST_WRITE, "no habilita"),  # ídem (6B.8)
        ("L14", FIRST_WRITE, "no habilita"),  # ídem (6B.8)
    ],
    ids=[
        "third-batch",
        "a-mano",
        "otra-repeticion",
        "otros-limites",
        *OUTSIDE_THE_LIST,
        "L11",
        "L10",
        "L09",
        "L14",
    ],
)
async def test_programmatic_suite_refuses_writes_before_creating_the_model(
    case_id, config, message
) -> None:
    case_file = _case(case_id)
    if case_file.case.approval is None and case_id not in config.approvals:
        config = config.model_copy(update={"approvals": {case_id: "approve"}})
    with pytest.raises(WritesDisabled, match=message):
        await run_live_suite([case_file], ROOT, config, mode="live", chat_factory=_never)


async def test_the_closed_list_holds_even_if_a_registered_profile_enables_more(monkeypatch):
    """Un perfil alterado en memoria (sin validación) tampoco amplía la lista cerrada."""
    widened = _unregistered(
        allowed_cases=("L08", "L12"),
        real_writes=("L08", "L12"),
        approvals={"L08": "approve", "L12": "approve"},
    )
    monkeypatch.setitem(PROFILES, "widened", widened)
    with pytest.raises(WritesDisabled, match="lista cerrada"):
        await run_live_suite([_case("L12")], ROOT, widened, mode="live", chat_factory=_never)


@pytest.mark.parametrize(
    ("case_id", "config", "error"),
    [
        ("L08", LiveConfig(), WritesDisabled),
        ("L08", PROFILES["third-batch"], WritesDisabled),
        ("L09", FIRST_WRITE, WritesDisabled),
        ("L14", FIRST_WRITE, WritesDisabled),
    ],
)
async def test_runner_called_directly_applies_the_same_restrictions(
    case_id, config, error, tmp_path
) -> None:
    model = _model(L08_SCRIPT)
    with pytest.raises(error):
        await run_live_case(_case(case_id).case, tmp_path, model, config, real_model=True)
    assert model.inner.prompts == [] and not any(tmp_path.iterdir())


async def test_runner_called_directly_needs_an_approval_policy(monkeypatch, tmp_path) -> None:
    """Perfil registrado con L08 habilitado pero sin aprobación declarada: el runner se niega."""
    no_policy = _unregistered(approvals={})
    monkeypatch.setitem(PROFILES, "no-policy", no_policy)
    model = _model(L08_SCRIPT, no_policy)
    with pytest.raises(ApprovalPolicyMissing):
        await run_live_case(_case("L08").case, tmp_path, model, no_policy, real_model=True)
    assert model.inner.prompts == [] and not any(tmp_path.iterdir())


async def test_conflicting_approval_policy_is_refused_before_inference(monkeypatch) -> None:
    conflict = _unregistered(approvals={"L08": "reject"})
    monkeypatch.setitem(PROFILES, "conflict", conflict)
    case_file = _case("L08")
    case_file = case_file.model_copy(
        update={"case": case_file.case.model_copy(update={"approval": "approve"})}
    )
    with pytest.raises(ApprovalPolicyMissing, match="distintas"):
        await run_live_suite([case_file], ROOT, conflict, mode="live", chat_factory=_never)


async def test_authorization_is_checked_again_right_before_the_write(monkeypatch, tmp_path):
    """Si la autorización desaparece entre la creación del modelo y la aprobación, la escritura
    no se despacha."""

    class Revoking(FakeGroq):
        async def ainvoke(self, messages: list, config: Any = None) -> AIMessage:
            monkeypatch.setattr(live_cases, "REAL_WRITE_CASES", frozenset())
            return await super().ainvoke(messages, config)

    model = LiveModel(Revoking(L08_SCRIPT), CallLimits(), (Meter(FIRST_WRITE.scenario_budget),))
    with pytest.raises(WritesDisabled):
        await run_live_case(_case("L08").case, tmp_path, model, FIRST_WRITE, real_model=True)
    assert _audit_rows(tmp_path) == []


# --- Alcance y aprobación simulada ---------------------------------------------------------------


async def test_simulated_approval_never_widens_request_scope(monkeypatch, tmp_path) -> None:
    """Si un bug dejara pasar una acción fuera del alcance hasta la aprobación, el humano
    simulado la rechaza y la corrida falla como bug del agente."""
    (grant,) = _case("L08").case.scope.grants
    monkeypatch.setattr(
        agent_scope.BoundScope, "check", lambda self, task_id, changes, ledger: grant.grant_id
    )
    script = [call("update_task", task_id="apo-4", changes={"status": "done"}), say("apo-4")]
    obs = await run_live_case(
        _case("L08").case, tmp_path, _model(script), FIRST_WRITE, real_model=True
    )
    assert obs.approval_out_of_scope == 1
    assert [r["decision"] for r in obs.approval_requests] == ["reject"]
    assert obs.audit == []
    graded = live_suite.graders.grade(_case("L08").case, obs)
    assert (graded["verdict"], graded["error_class"]) == ("failed", "agent")


@pytest.mark.parametrize(
    ("proposals", "reason", "approvals", "audit"),
    [
        ([{"task_id": "apo-4", "changes": {"status": "in_progress"}}], "resource", 0, 0),
        ([{"task_id": "apo-3", "changes": {"status": "done"}}], "value", 0, 0),
        ([{"task_id": "apo-3", "changes": {"assignee_id": None}}], "field", 0, 0),
        ([UPDATE, UPDATE], "exhausted", 1, 1),  # segundo uso
    ],
    ids=["recurso", "valor", "campo-null", "segundo-uso"],
)
async def test_out_of_scope_or_second_use_is_refused_before_approval_with_a_real_model(
    proposals, reason, approvals, audit, tmp_path
) -> None:
    script = [call("update_task", **p) for p in proposals] + [say("apo-3")]
    obs = await run_live_case(
        _case("L08").case, tmp_path, _model(script), FIRST_WRITE, real_model=True
    )
    assert [e["scope_reason"] for e in obs.tool_errors if e["code"] == "out_of_scope"] == [reason]
    assert (len(obs.approval_requests), len(obs.audit)) == (approvals, audit)


# --- Reintento y recuperación --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("faults", "uncertain"),
    [
        ({("update_task", 0): Fault.TIMEOUT_AFTER}, False),  # se pierde la respuesta: reintento
        ({("update_task", EVERY_CALL): Fault.TIMEOUT_AFTER}, True),  # incierta: recuperación
    ],
    ids=["reintento", "recuperacion"],
)
async def test_retry_and_recovery_never_duplicate_the_l08_write(faults, uncertain, tmp_path):
    obs = await run_live_case(
        _case("L08").case,
        tmp_path,
        _model(L08_SCRIPT),
        FIRST_WRITE,
        real_model=True,
        faults=faults,
    )
    assert ("write_uncertain" in obs.statuses) == uncertain
    assert obs.final_status == "completed"
    assert len(obs.audit) == 1 and len(obs.ledger) == 1  # una escritura, una clave
    keys = {d["key"] for d in obs.deliveries if d["tool"] == "update_task"}
    assert len(keys) == 1 and len(obs.deliveries) >= 2  # reenvíos con la misma clave
    assert len(obs.approval_requests) == 1  # nunca se vuelve a pedir ni se amplía
    assert obs.tasks_after["apo-3"]["status"] == "in_progress"
    assert obs.tasks_after["apo-3"]["version"] == obs.tasks_before["apo-3"]["version"] + 1


# --- Aislamiento ---------------------------------------------------------------------------------


async def test_runner_refuses_a_database_outside_a_fresh_temporary_directory(tmp_path) -> None:
    persistent = ROOT / "evals" / "no-debe-existir"  # dentro del repo (y de OneDrive)
    with pytest.raises(IsolationError):
        await run_live_case(
            _case("L08").case, persistent, _model(L08_SCRIPT), FIRST_WRITE, real_model=True
        )
    assert not persistent.exists()
    (tmp_path / "app.db").write_bytes(b"datos previos")  # una base existente, no una nueva
    model = _model(L08_SCRIPT)
    with pytest.raises(IsolationError):
        await run_live_case(_case("L08").case, tmp_path, model, FIRST_WRITE, real_model=True)
    assert (tmp_path / "app.db").read_bytes() == b"datos previos" and model.inner.prompts == []
    assert issubclass(IsolationError, ConfigError)


async def test_runner_refuses_an_empty_directory_outside_the_temporary_root(
    monkeypatch, tmp_path
) -> None:
    """Vacío y existente, pero fuera del directorio temporal: tampoco."""
    root = tmp_path / "temporal"
    root.mkdir()
    outside = tmp_path / "persistente"
    outside.mkdir()
    monkeypatch.setattr(live_runner, "_scratch_root", lambda: root.resolve())
    model = _model(L08_SCRIPT)
    with pytest.raises(IsolationError, match="temporal del sistema"):
        await run_live_case(_case("L08").case, outside, model, FIRST_WRITE, real_model=True)
    assert not any(outside.iterdir()) and model.inner.prompts == []


# --- CLI -----------------------------------------------------------------------------------------


class Reached(Exception):
    pass


FW = ("--profile", "first-write")
TB = ("--profile", "third-batch")


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        ([*FW, "--only", "L08", "--confirm-real-calls"], "--confirm-real-writes"),
        ([*FW, "--only", "L08", "--confirm-real-writes"], "--confirm-real-calls"),
        ([*TB, "--only", "L08", *BOTH], "no habilita"),
        (["--only", "L08", *BOTH], "no autoriza inferencia real"),
        ([*FW, "--only", "L08", "--model", "openai/gpt-oss-120b", *BOTH], "perfil registrado"),
        ([*FW, "--only", "L08", "L01", *BOTH], "no autoriza ['L01']"),
        ([*FW, "--only", "L01", *BOTH], "solo se usa con escenarios de escritura"),
        *[([*FW, "--only", w, *BOTH], "no autoriza") for w in OTHER_WRITES],
        ([*TB, "--only", "L12", *BOTH], "no autoriza"),  # third-batch no incluye L12
        *[([*TB, "--only", w, *BOTH], "no habilita") for w in ("L09", "L10", "L11")],
    ],
)
def test_cli_refuses_real_writes_unless_every_condition_holds(argv, reason, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", CANARY)

    def forbidden(*args, **kwargs):
        raise AssertionError("no debía ejecutar nada")

    monkeypatch.setattr(cli, "run_live_suite", forbidden)
    assert cli.main(["run", *argv, "--no-write"]) == 2
    out = capsys.readouterr()
    assert reason in out.err
    assert CANARY not in out.out + out.err


def test_cli_reaches_the_suite_only_for_l08_with_first_write_and_both_flags(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    seen: dict[str, Any] = {}

    async def reached(cases, root, config, *, mode):
        seen.update(ids=[c.case.id for c in cases], config=config, mode=mode)
        raise Reached

    monkeypatch.setattr(cli, "run_live_suite", reached)
    with pytest.raises(Reached):
        cli.main(["run", "--profile", "first-write", "--only", "L08", *BOTH, "--no-write"])
    assert seen == {"ids": ["L08"], "config": FIRST_WRITE, "mode": "live"}


async def test_dry_run_of_l08_makes_no_provider_call() -> None:
    report = await run_live_suite([_case("L08")], ROOT, FIRST_WRITE, mode="dry-run")
    (result,) = report["results"]
    assert result["verdict"] == "passed" and report["metadata"]["inference"] is False
    assert result["observation"]["audit_rows"] == 1


def test_profiles_module_is_the_single_source_of_profiles() -> None:
    assert live_suite.PROFILES is live_profiles.PROFILES
    assert live_suite.LiveConfig is live_profiles.LiveConfig
