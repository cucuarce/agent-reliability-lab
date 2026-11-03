"""Casos de evaluación declarativos (``evals/cases/**/*.json``).

Un caso describe **qué** pasa (principal, solicitud, comportamiento del modelo simulado, política
de aprobación del humano simulado, fallos inyectados) y **qué se espera** que el sistema haga. El
evaluador nunca escribe en estos archivos: las expectativas cambian solo con revisión humana
(invariante 5 de CLAUDE.md; el hook ``guard_paths`` pide confirmación).

Distinción explícita en cada caso:

- **Permisos del usuario**: los decide el servidor según el rol (``seed``).
- **Aprobación humana**: ``approval`` simula al humano que decide cada acción sellada.
- **Alcance de la solicitud**: ``request_scope`` declara qué pidió el usuario. **No existe un
  control de autorización por alcance del turno**: el alcance solo se usa para la política
  ``approve_in_scope`` (un humano atento) y para el chequeo *propuesto* ``request_scope``, que se
  informa como hallazgo, nunca como garantía.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from arl.agent.faults import Fault
from arl.domain.models import Identifier

CASE_SCHEMA = "arl.eval-case/1"
CaseId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")]


class Category(StrEnum):
    READ = "read"
    WRITE = "write"
    AUTHORIZATION = "authorization"
    SPOOFING = "spoofing"
    PROMPT_INJECTION = "prompt_injection"
    RETRY = "retry"
    FAILURE = "failure"
    CONFLICT = "conflict"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallSpec(_Strict):
    name: str  # puede ser una tool inexistente: el modelo simulado también se equivoca
    args: dict[str, Any] = {}


class StepSpec(_Strict):
    text: str = ""
    calls: tuple[ToolCallSpec, ...] = ()


class InjectionBranch(_Strict):
    """El modelo "obedece" texto malicioso: cuando ``marker`` aparece en el resultado de una tool,
    sigue estos pasos en vez de los normales. Demuestra que el texto llegó a su contexto."""

    marker: str = Field(min_length=8)
    steps: tuple[StepSpec, ...] = Field(min_length=1)


class ModelSpec(_Strict):
    steps: tuple[StepSpec, ...] = Field(min_length=1)
    repeat_last: bool = False
    on_injection: InjectionBranch | None = None


class TaskOverride(_Strict):
    title: str | None = None
    description: str | None = None


class SeedSpec(_Strict):
    """Cambios sobre los datos sintéticos empaquetados (cada caso parte de una base nueva)."""

    tasks: dict[Identifier, TaskOverride] = {}
    project_descriptions: dict[Identifier, str] = {}


class FaultSpec(_Strict):
    tool: Literal["list_projects", "get_project", "update_task"]
    call: int | Literal["*"] = "*"  # n-ésima llamada (desde 0) o todas
    fault: Fault


class Scope(_Strict):
    """Lo que el usuario pidió escribir. Vacío: una solicitud de solo lectura."""

    tasks: tuple[Identifier, ...] = ()


class ConcurrentUpdate(_Strict):
    """Otro usuario modifica una tarea (vía ``ProjectService``) antes de la n-ésima decisión."""

    before_decision: int = Field(ge=0)
    principal: Identifier
    task_id: Identifier
    changes: dict[str, Any]


class Followup(_Strict):
    """Otro principal intenta operar sobre el thread del caso (suplantación entre procesos)."""

    principal: Identifier
    action: Literal["run_turn", "resume", "continue_after_failure"]
    text: str = "continuá"


# ``pending``: el humano no decide y el thread queda esperando (p. ej. para probar suplantación).
ApprovalPolicy = Literal["approve", "reject", "approve_in_scope", "pending"]


class Recovery(_Strict):
    continue_after_failure: int = Field(default=0, ge=0, le=3)
    clear_faults_first: bool = True  # el "servicio se recupera" antes de continuar


class WriteExpectation(_Strict):
    task_id: Identifier
    changes: dict[str, Any]  # campos que cambiaron según el audit_log (after != before)


class ApprovalExpectation(_Strict):
    task_id: Identifier
    changes: dict[str, Any]


BlockReason = Literal[
    "validation",  # invalid_tool_call: el registro de tools la rechazó antes de ejecutar
    "not_found",  # tarea o proyecto no visible para el principal
    "authorization",  # forbidden: el servidor rechazó por rol
    "human_rejection",  # el humano rechazó la acción sellada
    "invalid_change",  # regla de dominio (p. ej. assignee no miembro)
    "version_conflict",
    "tool_limit",
]


class BlockExpectation(_Strict):
    tool: str
    reason: BlockReason
    task_id: Identifier | None = None


class Attack(_Strict):
    """Tool call maliciosa o no solicitada que el caso intenta provocar."""

    tool: str
    task_id: Identifier | None = None
    project_id: Identifier | None = None


WriteState = Literal["executed", "refused_by_server", "uncertain", "rejected", "awaiting_approval"]


class Expected(_Strict):
    final_status: Literal[
        "completed",
        "awaiting_approval",
        "tool_limit",
        "turn_limit",
        "interrupted",
        "write_uncertain",
    ]
    approvals: tuple[ApprovalExpectation, ...] = ()
    writes: tuple[WriteExpectation, ...] = ()
    blocked: tuple[BlockExpectation, ...] = ()
    write_states: dict[Identifier, WriteState] = {}  # por task_id, según las trazas
    # Escrituras inciertas: ¿el audit_log tiene la escritura? (evidencia, no una expectativa del
    # agente: el agente no puede saberlo y por eso reporta "uncertain").
    reconciliation: dict[Identifier, Literal["committed", "not_committed"]] = {}
    update_task_deliveries: int | None = None  # llamadas update_task que llegaron al servidor
    followup_refusals: tuple[str, ...] = ()


class EvalCase(_Strict):
    schema_: Literal["arl.eval-case/1"] = Field(alias="schema")
    id: CaseId
    category: Category
    description: str = Field(min_length=10)
    principal: Identifier
    request: str = Field(min_length=1)
    request_scope: Scope = Scope()
    model: ModelSpec
    approval: ApprovalPolicy | tuple[Literal["approve", "reject", "pending"], ...] = "approve"
    seed: SeedSpec = SeedSpec()
    faults: tuple[FaultSpec, ...] = ()
    recovery: Recovery = Recovery()
    concurrent_updates: tuple[ConcurrentUpdate, ...] = ()
    followups: tuple[Followup, ...] = ()
    attack: Attack | None = None
    limits: dict[str, int] = {}
    expected: Expected
    covers_tests: tuple[str, ...] = ()  # tests existentes que ya cubren parte del escenario

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> EvalCase:
        if self.category is Category.PROMPT_INJECTION and self.model.on_injection is None:
            raise ValueError("un caso de prompt_injection necesita model.on_injection")
        if self.model.on_injection is not None and not _seed_texts(self.seed):
            raise ValueError("on_injection sin texto malicioso en seed")
        if self.model.on_injection and not any(
            self.model.on_injection.marker in text for text in _seed_texts(self.seed)
        ):
            raise ValueError("el marker de on_injection no aparece en los datos sembrados")
        if set(self.expected.reconciliation) - set(self.expected.write_states):
            raise ValueError("reconciliation solo para tareas con write_states")
        for task_id in self.expected.reconciliation:
            if self.expected.write_states[task_id] != "uncertain":
                raise ValueError("reconciliation solo se declara para escrituras inciertas")
        return self


def _seed_texts(seed: SeedSpec) -> list[str]:
    texts = [o.description or "" for o in seed.tasks.values()]
    texts += [o.title or "" for o in seed.tasks.values()]
    return [t for t in (*texts, *seed.project_descriptions.values()) if t]


class CaseFile(BaseModel):
    """Un caso cargado y la huella de su archivo (la versión de la suite depende de todas)."""

    model_config = ConfigDict(frozen=True)

    case: EvalCase
    path: str
    sha256: str


class InvalidCase(ValueError):
    pass


def load_cases(root: Path) -> list[CaseFile]:
    """Todos los casos de ``root`` (recursivo), validados. Falla ante ids repetidos."""
    files = sorted(root.rglob("*.json"))
    if not files:
        raise InvalidCase(f"no hay casos en {root}")
    loaded: list[CaseFile] = []
    for path in files:
        raw = path.read_bytes()
        try:
            case = EvalCase.model_validate(json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidCase(f"{path.relative_to(root)}: {exc}") from None
        if path.stem != case.id:
            raise InvalidCase(f"{path.relative_to(root)}: el archivo debe llamarse {case.id}.json")
        loaded.append(
            CaseFile(
                case=case,
                path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    ids = [c.case.id for c in loaded]
    if duplicated := sorted({i for i in ids if ids.count(i) > 1}):
        raise InvalidCase(f"ids repetidos: {duplicated}")
    return loaded


def suite_fingerprint(cases: list[CaseFile]) -> str:
    """Huella del conjunto de casos: cambia si cambia cualquier expectativa."""
    digest = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.path):
        digest.update(f"{c.path}:{c.sha256}\n".encode())
    return digest.hexdigest()
