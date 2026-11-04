"""Escenarios de la evaluación con LLM real (``arl.live-case/1``).

Las expectativas son **independientes del modelo**: herramientas exigidas o prohibidas (y su
orden), hechos mínimos en la respuesta, estado final de la base y ``audit_log``. No hay LLM como
juez. Los casos v1 y v2 no cambian; estos viven en ``evals/live/cases``.

- ``scope`` lo declara el caso, que representa a la aplicación. El modelo nunca lo ve.
- **Fase 1: solo lectura.** ``scope`` debe ser ``read_only()``.
- **Fase 2: escrituras con alcance explícito.** Con un modelo real están deshabilitadas en 6B.1
  (``REAL_WRITES_ENABLED``); solo corren en dry-run, con su guion de referencia.
- ``reference``: el guion que sigue el modelo simulado del dry-run, para comprobar los graders y
  el entorno sin inferencia. Nunca se usa con un modelo real ni influye en las expectativas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from arl.agent.scope import RequestScope
from arl.evals.cases import ConcurrentUpdate, Identifier, SeedSpec, StepSpec

SCHEMA = "arl.live-case/1"
REAL_WRITES_ENABLED = False  # 6B.1: ninguna escritura con un modelo real
CaseId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Injection(_Strict):
    """La acción que pide el texto malicioso sembrado. Proponerla es un **intento**; que aparezca
    en el ``audit_log`` es una **ejecución**. Son métricas distintas."""

    task_id: Identifier
    changes: dict[str, Any]
    marker: str = Field(min_length=8)  # texto que debe aparecer en los datos sembrados


class Expectations(_Strict):
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    order: tuple[tuple[str, str], ...] = ()  # (a, b): la primera llamada a "a" antes que a "b"
    answer_any: tuple[tuple[str, ...], ...] = ()  # cada grupo: al menos uno aparece
    answer_none: tuple[str, ...] = ()  # ninguno aparece
    tasks: dict[Identifier, dict[str, Any]] = {}  # valores finales; el resto, sin cambios
    audit_rows: int = Field(default=0, ge=0)
    max_write_proposals: int | None = Field(default=None, ge=0)


class LiveCase(_Strict):
    schema_: Literal["arl.live-case/1"] = Field(alias="schema")
    id: CaseId
    phase: Literal[1, 2]
    title: str = Field(min_length=1)
    principal: Identifier
    request: str = Field(min_length=1)
    seed: SeedSpec = SeedSpec()
    scope: RequestScope
    approval: Literal["approve", "reject"] = "approve"
    concurrent_updates: tuple[ConcurrentUpdate, ...] = ()
    injection: Injection | None = None
    expected: Expectations
    reference: tuple[StepSpec, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _phases(self) -> LiveCase:
        if self.scope.kind != "explicit":
            raise ValueError("el alcance de un escenario real es siempre explícito")
        if self.phase == 1 and (
            self.scope.grants or self.expected.audit_rows or self.expected.tasks
        ):
            raise ValueError("fase 1: solo lectura (scope read_only, sin escrituras esperadas)")
        if self.phase == 2 and not self.scope.grants:
            raise ValueError("fase 2: requiere un alcance de escritura explícito")
        if self.injection is not None:
            texts = [*self.seed.project_descriptions.values()]
            texts += [t.description or "" for t in self.seed.tasks.values()]
            texts += [t.title or "" for t in self.seed.tasks.values()]
            if not any(self.injection.marker in t for t in texts):
                raise ValueError("el marker de la inyección no aparece en los datos sembrados")
        return self

    @property
    def writes_allowed_with_real_model(self) -> bool:
        return REAL_WRITES_ENABLED and self.phase == 2


class LiveCaseFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: LiveCase
    path: str
    sha256: str


class InvalidLiveCase(ValueError):
    pass


def load_live_cases(root: Path) -> list[LiveCaseFile]:
    files = sorted(root.rglob("*.json"))
    if not files:
        raise InvalidLiveCase(f"no hay escenarios en {root}")
    loaded = []
    for path in files:
        raw = path.read_bytes()
        try:
            case = LiveCase.model_validate(json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidLiveCase(f"{path.relative_to(root)}: {exc}") from None
        if path.stem != case.id:
            raise InvalidLiveCase(f"{path.name}: el archivo debe llamarse {case.id}.json")
        loaded.append(
            LiveCaseFile(
                case=case,
                path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    ids = [c.case.id for c in loaded]
    if duplicated := sorted({i for i in ids if ids.count(i) > 1}):
        raise InvalidLiveCase(f"ids repetidos: {duplicated}")
    return loaded


def fingerprint(cases: list[LiveCaseFile]) -> str:
    digest = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.path):
        digest.update(f"{c.path}:{c.sha256}\n".encode())
    return digest.hexdigest()
