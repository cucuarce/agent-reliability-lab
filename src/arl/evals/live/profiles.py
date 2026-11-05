"""Perfiles de la evaluación con LLM real y la autorización de escrituras con un modelo real.

Una escritura con un modelo real necesita, a la vez:

1. que el escenario esté en la lista cerrada del código (``cases.REAL_WRITE_CASES``), con la
   única decisión del humano simulado que esa lista le permite (L08, L09 y L10 ``approve``,
   L11 ``reject``): ni el caso ni el perfil la cambian;
2. que el perfil la habilite (``LiveConfig.real_writes``), y que la configuración sea
   exactamente la de un perfil registrado (``PROFILES``): un ``LiveConfig`` armado a mano o
   modificado no habilita nada;
3. una política de aprobación explícita (``approval_policy``);
4. en la CLI, ``--confirm-real-calls`` y ``--confirm-real-writes``.

``authorize_real_write`` lo verifica la CLI, la suite antes de crear el cliente del modelo, el
runner al empezar el escenario y otra vez justo antes de despachar cada decisión (aprobar o
rechazar).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arl.agent.state import Limits
from arl.evals.live import cases as live_cases
from arl.evals.live.cases import ApprovalPolicyMissing, LiveCase
from arl.evals.live.model import Budget, CallLimits
from arl.evals.live.providers import GroqSpec


class WritesDisabled(RuntimeError):
    """Escritura con un modelo real sin habilitación: se detiene antes de cualquier inferencia."""


class LiveConfig(BaseModel):
    """Todo lo que define una ejecución. No contiene credenciales."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: GroqSpec = GroqSpec()
    call: CallLimits = CallLimits()
    scenario_budget: Budget = Budget(max_model_calls=6, max_total_tokens=30_000, max_duration_s=180)
    suite_budget: Budget = Budget(max_model_calls=14, max_total_tokens=70_000, max_duration_s=480)
    max_model_turns: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=6, ge=1)
    repeat: int = Field(default=1, ge=1, le=10)
    # Escenarios que este perfil autoriza con inferencia real; None: ninguno (``run`` se niega).
    allowed_cases: tuple[str, ...] | None = None
    # Decisión del humano simulado por escenario de escritura, declarada por el perfil para los
    # casos que no la declaran. No habilita escrituras: eso lo deciden los bloqueos de fase 2.
    approvals: dict[str, Literal["approve", "reject"]] = {}
    # Escenarios de escritura que este perfil habilita con un modelo real. Nunca más que los de
    # la lista cerrada (``cases.REAL_WRITE_CASES``) ni que los que el perfil autoriza.
    real_writes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _approvals_authorized(self) -> LiveConfig:
        extra = sorted(set(self.approvals) - set(self.allowed_cases or ()))
        if extra:
            raise ValueError(f"el perfil no autoriza los escenarios con aprobación {extra}")
        return self

    @model_validator(mode="after")
    def _real_writes_closed(self) -> LiveConfig:
        if unauthorized := sorted(set(self.real_writes) - set(self.allowed_cases or ())):
            raise ValueError(f"el perfil no autoriza los escenarios con escritura {unauthorized}")
        if outside := sorted(set(self.real_writes) - set(live_cases.REAL_WRITE_CASES)):
            raise ValueError(f"escrituras reales fuera de la lista cerrada: {outside}")
        for case_id in self.real_writes:
            decision = self.approvals.get(case_id)
            if decision is not None and decision not in live_cases.REAL_WRITE_CASES[case_id]:
                raise ValueError(f"{case_id}: decisión no permitida con un modelo real: {decision}")
        return self

    def graph_limits(self) -> Limits:
        return Limits(
            max_model_turns=self.max_model_turns,
            max_tool_calls=self.max_tool_calls,
            retry_initial_interval=0.0,
            retry_max_interval=0.0,
            retry_jitter=False,
        )


# Primera tanda con inferencia real: L01 y L02, solo lectura, una corrida cada uno, en serie y
# con a lo sumo 8 pedidos a Groq (incluidos los reintentos). Cada escenario tiene la mitad, así
# la admisión deja empezar el segundo aunque el primero gaste todo lo suyo.
PROFILES: dict[str, LiveConfig] = {
    "default": LiveConfig(),
    "first-batch": LiveConfig(
        scenario_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=150),
        suite_budget=Budget(max_model_calls=8, max_total_tokens=48_000, max_duration_s=300),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L01", "L02"),
    ),
    # Segunda tanda: L03-L07 (solo lectura, con dos inyecciones), una corrida cada uno, en serie,
    # a lo sumo 20 pedidos: 4 por escenario, así la admisión deja empezar el quinto aunque los
    # anteriores gasten todo lo suyo. La duración de la tanda deja margen para el armado de cada
    # escenario (servidor MCP y SQLite temporal), que corre fuera de las llamadas al modelo.
    "second-batch": LiveConfig(
        scenario_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=150),
        suite_budget=Budget(max_model_calls=20, max_total_tokens=120_000, max_duration_s=900),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L03", "L04", "L05", "L06", "L07"),
    ),
    # Repetibilidad de las inyecciones: L05 y L06, tres corridas cada uno (L05 x3 y después
    # L06 x3), en serie, a lo sumo 24 pedidos: 4 por corrida.
    "injection-repeat": LiveConfig(
        scenario_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=150),
        suite_budget=Budget(max_model_calls=24, max_total_tokens=144_000, max_duration_s=1080),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=3,
        allowed_cases=("L05", "L06"),
    ),
    # Tercera tanda: L08-L11 (escritura), una corrida cada uno, en serie, a lo sumo 16 pedidos.
    # Solo para el dry-run: no habilita escrituras reales (``real_writes`` vacío), así que ``run``
    # se niega aunque el perfil las liste.
    "third-batch": LiveConfig(
        scenario_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=150),
        suite_budget=Budget(max_model_calls=16, max_total_tokens=96_000, max_duration_s=720),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L08", "L09", "L10", "L11"),
    ),
    # Primera escritura con un modelo real: solo L08, una corrida, a lo sumo 4 pedidos a Groq.
    # Declara la aprobación simulada de L08 (el caso no la declara) y es el único perfil que
    # habilita una escritura real. La CLI exige además --confirm-real-writes.
    "first-write": LiveConfig(
        scenario_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=150),
        suite_budget=Budget(max_model_calls=4, max_total_tokens=24_000, max_duration_s=180),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L08",),
        approvals={"L08": "approve"},
        real_writes=("L08",),
    ),
    # Rechazo explícito con un modelo real: solo L11, una corrida. El caso declara ``reject`` y
    # la lista cerrada no le permite otra decisión. A lo sumo 6 pedidos a Groq (incluidos los
    # reintentos), con los mismos 4 turnos y 4 tool calls; los tokens alcanzan para 6 pedidos con
    # su peor caso (prompt estimado + 1024 de salida, unos 6000 cada uno).
    "reject-write": LiveConfig(
        scenario_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=240),
        suite_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=300),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L11",),
        real_writes=("L11",),
    ),
    # Denegación de permisos del servidor con un modelo real: solo L10, una corrida. carol es
    # viewer de apollo: el alcance cubre la operación y el humano simulado la aprueba (el caso no
    # declara la decisión: la declara el perfil, y la lista cerrada solo permite ``approve``),
    # pero ``ProjectService`` la rechaza con ``Forbidden``. Mismos límites que reject-write: a lo
    # sumo 6 pedidos (incluidos los reintentos), 4 turnos y 4 tool calls, y tokens para 6
    # pedidos con su peor caso.
    "forbidden-write": LiveConfig(
        scenario_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=240),
        suite_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=300),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L10",),
        approvals={"L10": "approve"},
        real_writes=("L10",),
    ),
    # Cerrar una tarea con un modelo real: solo L09, una corrida (alice, owner de apollo,
    # ``apo-3.status=done``). El caso no declara la decisión: la declara el perfil, y la lista
    # cerrada solo permite ``approve``. Mide ``null_padding``: un campo no pedido (p. ej.
    # ``assignee_id: null``, que desasigna) queda fuera del alcance y se bloquea antes de la
    # aprobación. Mismos límites que reject-write y forbidden-write.
    "close-write": LiveConfig(
        scenario_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=240),
        suite_budget=Budget(max_model_calls=6, max_total_tokens=40_000, max_duration_s=300),
        max_model_turns=4,
        max_tool_calls=4,
        repeat=1,
        allowed_cases=("L09",),
        approvals={"L09": "approve"},
        real_writes=("L09",),
    ),
}


def approval_policy(case: LiveCase, config: LiveConfig, *, real: bool) -> dict[str, Any] | None:
    """La decisión del humano simulado de un escenario de escritura y de dónde sale.

    Con un modelo real tiene que ser explícita: del caso o del perfil, y si ambos la declaran,
    la misma. Si no, ``ApprovalPolicyMissing``. En dry-run, un caso histórico sin política usa
    ``approve`` y el reporte lo marca (``dry_run_default``)."""
    if case.phase != 2:
        return None
    declared = {
        source: decision
        for source, decision in (
            ("case", case.approval),
            ("profile", config.approvals.get(case.id)),
        )
        if decision is not None
    }
    if len(set(declared.values())) > 1:
        raise ApprovalPolicyMissing(
            f"{case.id}: el caso y el perfil declaran aprobaciones distintas"
        )
    if declared:
        source, decision = next(iter(declared.items()))
    elif real:
        raise ApprovalPolicyMissing(f"{case.id}: escritura sin política de aprobación explícita")
    else:
        source, decision = "dry_run_default", "approve"
    return {"decision": decision, "source": source, "actor": "simulated"}


def authorize_real_write(case: LiveCase, config: LiveConfig) -> None:
    """``WritesDisabled`` salvo que esta escritura con un modelo real esté habilitada."""
    if case.phase != 2:
        return
    if case.id not in live_cases.REAL_WRITE_CASES:
        raise WritesDisabled(f"{case.id}: escrituras deshabilitadas (fuera de la lista cerrada)")
    if case.id not in config.real_writes:
        raise WritesDisabled(f"{case.id}: el perfil no habilita escrituras con un modelo real")
    if config not in PROFILES.values():
        raise WritesDisabled(f"{case.id}: la configuración no es la de un perfil registrado")
    policy = approval_policy(case, config, real=True)
    if policy is None or policy["decision"] not in live_cases.REAL_WRITE_CASES[case.id]:
        decision = policy and policy["decision"]
        raise WritesDisabled(f"{case.id}: decisión no permitida con un modelo real: {decision}")
