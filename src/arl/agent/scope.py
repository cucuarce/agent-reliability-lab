"""Alcance de un turno (``RequestScope``): qué escrituras autorizó la aplicación (etapa 6A.1).

Corrige el hallazgo F1 de la etapa 6A: una escritura que el usuario no pidió (p. ej. inducida por
texto malicioso en una tarea) se ejecutaba si el humano la aprobaba sin revisar.

Quién decide qué:

- **La aplicación** declara el alcance de cada turno a partir de una entrada explícita del usuario
  (``run_turn(..., scope=...)``). Nunca sale del LLM, de una tool ni del texto del pedido. No es
  autenticación: la identidad sigue siendo simulada.
- **El runtime** lo liga al thread y al principal del proceso y lo sella (``BoundScope``).
- **El grafo** (código determinístico) bloquea toda escritura fuera del alcance **antes** de pedir
  aprobación, reserva el permiso al preparar la acción y lo verifica otra vez al ejecutarla.
- **El servidor** sigue decidiendo los permisos por rol; el alcance nunca los amplía.

Registro del turno (``scope_ledger``, por ``action_id``)::

    reserved ─ aprobada y confirmada (updated / unchanged / replay) ─► consumed
    reserved ─ rechazo humano ───────────────────────────────────────► released
    reserved ─ WRITE_UNCERTAIN ──────────────────────────────────────► reserved (retenida)
    reserved ─ rechazo definitivo del servidor ──────────────────────► refused
    reserved ─ version_conflict ─────────────────────────────────────► invalidated

``reserved``, ``consumed``, ``refused`` e ``invalidated`` cuentan para ``max_uses`` y
``max_writes``: un permiso consumido, cerrado o invalidado nunca habilita una operación nueva en el
mismo turno. Un reintento técnico es la **misma** acción (mismo ``action_id``, clave y
``args_hash``) y usa su propia reserva.

El sello (``scope_hash``, sha256) detecta cambios accidentales o del LLM. **No es autenticación
criptográfica**: quien pueda escribir los checkpoints puede recalcularlo.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arl.domain.models import Identifier, TaskPatch

# Estados que ocupan un permiso. ``released`` (rechazo humano) lo devuelve.
COUNTED_STATES = frozenset({"reserved", "consumed", "refused", "invalidated"})
UPDATE_FIELDS = frozenset(TaskPatch.model_fields)
Reason = Literal["read_only", "resource", "field", "value", "exhausted", "max_writes", "no_scope"]


class OutOfScope(Exception):
    """La escritura no está autorizada por el alcance del turno. ``reason`` es un código fijo:
    nunca incluye los valores pedidos."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(f"la acción no está autorizada por el alcance del turno ({reason})")
        self.reason = reason


class ScopeMismatch(RuntimeError):
    """El alcance no corresponde a este thread o a esta identidad."""


class ScopeViolation(RuntimeError):
    """El alcance guardado fue alterado, falta, o una ejecución no tiene su reserva."""


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WriteGrant(_Frozen):
    """Permiso de escritura: tareas, campos y valores autorizados. ``None`` = cualquier valor
    válido de ese campo (debe declararse explícitamente)."""

    tool: Literal["update_task"] = "update_task"
    task_ids: tuple[Identifier, ...] = Field(min_length=1)
    fields: dict[str, tuple[Any, ...] | None] = Field(min_length=1)
    max_uses: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _known_fields(self) -> WriteGrant:
        if unknown := set(self.fields) - UPDATE_FIELDS:
            raise ValueError(f"campos que update_task no tiene: {sorted(unknown)}")
        return self

    @property
    def grant_id(self) -> str:
        return "g-" + _sha256(self.model_dump(mode="json"))[:12]

    def covers(self, task_id: str, changes: dict[str, Any]) -> Reason | None:
        """``None`` si cubre el cambio; si no, el primer motivo (sin contar el cupo)."""
        if task_id not in self.task_ids:
            return "resource"
        if set(changes) - set(self.fields):
            return "field"
        for name, value in changes.items():
            allowed = self.fields[name]
            if allowed is not None and value not in allowed:
                return "value"
        return None


def allow_update(*task_ids: str, max_uses: int = 1, **fields: Any) -> WriteGrant:
    """``allow_update("apo-3", status=["done"])``. Un campo con ``None`` admite cualquier valor."""
    return WriteGrant(
        task_ids=task_ids,
        fields={k: None if v is None else tuple(v) for k, v in fields.items()},
        max_uses=max_uses,
    )


class RequestScope(_Frozen):
    """Declaración de la aplicación para un turno. Inmutable."""

    kind: Literal["explicit", "unrestricted_legacy_v1"] = "explicit"
    grants: tuple[WriteGrant, ...] = ()
    max_writes: int | None = Field(default=None, ge=0)  # None: la suma de max_uses

    @classmethod
    def read_only(cls) -> RequestScope:
        return cls()

    @classmethod
    def writes(cls, *grants: WriteGrant, max_writes: int | None = None) -> RequestScope:
        return cls(grants=grants, max_writes=max_writes)

    @classmethod
    def unrestricted_legacy_v1(cls) -> RequestScope:
        """**Solo** para reproducir la suite de evaluación v1 (anterior a 6A.1), que no declara
        alcance. Cualquier otro uso lo detecta ``test_unrestricted_scope_is_used_only_by_…``."""
        return cls(kind="unrestricted_legacy_v1")

    @property
    def write_limit(self) -> int:
        if self.max_writes is not None:
            return self.max_writes
        return sum(g.max_uses for g in self.grants)


class BoundScope(_Frozen):
    """El alcance ligado a un thread y a un principal, sellado. Es lo que guarda el checkpoint."""

    thread_id: str
    principal_id: str
    scope: RequestScope
    scope_hash: str

    @classmethod
    def bind(cls, scope: RequestScope, *, thread_id: str, principal_id: str) -> BoundScope:
        if not isinstance(scope, RequestScope):
            raise TypeError("scope debe ser un RequestScope construido por la aplicación")
        body = {"thread_id": thread_id, "principal_id": principal_id,
                "scope": scope.model_dump(mode="json")}  # fmt: skip
        return cls(thread_id=thread_id, principal_id=principal_id, scope=scope,
                   scope_hash=_sha256(body))  # fmt: skip

    @model_validator(mode="after")
    def _sealed(self) -> BoundScope:
        body = {"thread_id": self.thread_id, "principal_id": self.principal_id,
                "scope": self.scope.model_dump(mode="json")}  # fmt: skip
        if _sha256(body) != self.scope_hash:
            raise ValueError("el alcance del turno fue alterado después de fijarse")
        return self

    def verify(self, *, thread_id: str, principal_id: str) -> None:
        if (self.thread_id, self.principal_id) != (thread_id, principal_id):
            raise ScopeMismatch("el alcance pertenece a otro thread o a otra identidad")

    def check(self, task_id: str, changes: dict[str, Any], ledger: dict[str, Any]) -> str:
        """``grant_id`` que autoriza el cambio con cupo libre, u ``OutOfScope``."""
        scope = self.scope
        if scope.kind == "unrestricted_legacy_v1":
            return "unrestricted"
        if not scope.grants or scope.write_limit == 0:
            raise OutOfScope("read_only")
        used = [entry for entry in ledger.values() if entry["state"] in COUNTED_STATES]
        reasons = []
        for grant in scope.grants:
            reason = grant.covers(task_id, changes)
            if reason is None:
                if sum(1 for e in used if e["grant_id"] == grant.grant_id) >= grant.max_uses:
                    reason = "exhausted"
                elif len(used) >= scope.write_limit:
                    reason = "max_writes"
                else:
                    return grant.grant_id
            reasons.append(reason)
        # El motivo más específico entre los permisos evaluados (el que más se acercó).
        order: list[Reason] = ["max_writes", "exhausted", "value", "field", "resource"]
        raise OutOfScope(next(r for r in order if r in reasons))


def load_bound(raw: Any) -> BoundScope:
    """El alcance guardado en el estado; ``ScopeViolation`` si falta o fue alterado."""
    if raw is None:
        raise ScopeViolation("el thread no tiene alcance: iniciá el turno con run_turn(scope=...)")
    try:
        return BoundScope.model_validate(raw)
    except ValueError:
        raise ScopeViolation("el alcance del turno fue alterado o es inválido") from None


def settle_state(outcome: str | None) -> str:
    """Estado final de una reserva según la respuesta válida del servidor."""
    if outcome in ("updated", "unchanged"):
        return "consumed"
    if outcome == "version_conflict":
        return "invalidated"
    return "refused"  # forbidden, not_found, invalid_change, idempotency_conflict o desconocido
