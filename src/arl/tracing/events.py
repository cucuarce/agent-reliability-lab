"""Esquema versionado de los eventos de traza (``arl.trace`` v1).

Cada línea JSONL es un ``Envelope`` con un ``data`` validado por el modelo de su tipo de evento.
Todos los modelos usan ``extra="forbid"``: un campo nuevo exige cambiar el esquema a propósito, y
el registrador descarta (sin lanzar) cualquier evento que no lo cumpla. Así el esquema funciona
también como lista blanca de lo que se puede registrar.

Qué **nunca** se registra: texto de mensajes (usuario, modelo o tools), prompts, valores de los
cambios propuestos (títulos, descripciones), textos de error, rutas ni checkpoints. Solo
identificadores, nombres de campos, códigos, tipos de error, conteos y duraciones.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SCHEMA_VERSION = 2
# v2 (etapa 6A.1) es un superconjunto de v1: agrega el alcance del turno (``run_started.scope``,
# la decisión ``out_of_scope`` y el evento ``scope_reservation``). Los lectores aceptan ambas.
SUPPORTED_VERSIONS = (1, 2)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SafeId = Annotated[str, StringConstraints(pattern=_SAFE_ID.pattern)]


def safe_id(value: Any) -> str:
    """Identificadores que vienen del LLM o del caller: se conservan si tienen forma de id; si
    no, se reemplazan por un hash (nunca se registra texto libre)."""
    text = str(value)
    if _SAFE_ID.fullmatch(text):
        return text
    return "h-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Operation = Literal["run_turn", "resume", "continue_after_failure"]
ScopeReason = Literal[
    "read_only", "resource", "field", "value", "exhausted", "max_writes", "no_scope"
]
RunStatus = Literal[
    "completed",
    "awaiting_approval",
    "tool_limit",
    "turn_limit",
    "interrupted",
    "write_uncertain",
    "busy",
    "refused",
    "error",
]


class ToolCallRef(_Data):
    tool_call_id: SafeId
    tool: SafeId


class Failure(_Data):
    code: SafeId  # p. ej. unavailable, invalid_response
    operation: SafeId  # tool MCP o nodo
    error_type: SafeId  # nombre del tipo de excepción original


class RunStarted(_Data):
    operation: Operation
    scope: Literal["explicit", "read_only", "unrestricted_legacy_v1"] | None = None


class RecoveryStarted(_Data):
    """``continue_after_failure`` retoma una ejecución detenida."""

    prior_status: Literal["interrupted", "write_uncertain"]
    resume_node: SafeId
    action_id: SafeId | None = None  # la escritura aprobada que se reintenta, si la hay


class HistoryClosed(_Data):
    tool_call_ids: list[SafeId]
    reason: Literal["not_executed"]


class ModelCall(_Data):
    model_turn: int
    duration_ms: float
    outcome: Literal["ok", "error"]
    tool_calls: list[ToolCallRef] = []
    error_type: SafeId | None = None


class ToolCallRouted(_Data):
    """Decisión del gate: qué se hace con cada tool call que propuso el LLM (lo *solicitado*)."""

    tool_call_id: SafeId
    tool: SafeId
    decision: Literal[
        "execute", "requires_approval", "invalid_tool_call", "tool_limit", "out_of_scope"
    ]
    scope_reason: ScopeReason | None = None  # solo para out_of_scope: código fijo, sin valores
    project_id: SafeId | None = None
    task_id: SafeId | None = None
    changed_fields: list[SafeId] | None = None  # solo nombres, nunca valores


class ApprovalRequested(_Data):
    """La acción sellada que se le muestra al humano (lo que se *aprobaría*)."""

    action_id: SafeId
    args_hash: SafeId
    tool_call_id: SafeId
    tool: SafeId
    task_id: SafeId
    project_id: SafeId
    my_role: SafeId
    changed_fields: list[SafeId]
    expected_version: int


class ApprovalDecided(_Data):
    action_id: SafeId
    tool_call_id: SafeId
    decision: Literal["approve", "reject", "invalid"]
    invalid_reason: Literal["principal", "schema", "other_action"] | None = None
    with_reason: bool = False  # el motivo de un rechazo es texto libre: no se registra


class McpCall(_Data):
    tool: SafeId
    outcome: Literal["ok", "domain_error", "unavailable", "invalid_response"]
    duration_ms: float
    node: SafeId | None = None
    tool_call_id: SafeId | None = None
    attempt: int | None = None  # intento del nodo dentro de la ejecución (reintentos)
    code: SafeId | None = None  # código de dominio o de fallo
    error_type: SafeId | None = None
    project_id: SafeId | None = None
    task_id: SafeId | None = None
    idempotency_key: SafeId | None = None  # correlaciona con audit_log / idempotency_keys
    replayed: bool | None = None


class WriteResult(_Data):
    """Lo *ejecutado*: respuesta válida del servidor para la acción aprobada."""

    action_id: SafeId
    idempotency_key: SafeId
    tool_call_id: SafeId
    task_id: SafeId
    outcome: SafeId  # updated | unchanged | código de dominio (forbidden, version_conflict)
    replayed: bool | None = None
    version: int | None = None
    audit_id: int | None = None


class ScopeReservation(_Data):
    """Estado de la reserva de un permiso del alcance para una acción (ver ``arl.agent.scope``)."""

    action_id: SafeId
    grant_id: SafeId
    task_id: SafeId
    state: Literal["reserved", "consumed", "released", "refused", "invalidated"]


class WriteUncertain(_Data):
    """La escritura aprobada no pudo confirmarse: pudo ocurrir o no. No es un fallo."""

    action_id: SafeId
    idempotency_key: SafeId
    tool_call_id: SafeId
    task_id: SafeId
    failure: Failure | None = None


class Counts(_Data):
    model_calls: int = 0
    mcp_calls: int = 0
    mcp_failures: int = 0


class RunFinished(_Data):
    operation: Operation
    status: RunStatus
    duration_ms: float
    refusal: SafeId | None = None  # para busy/refused: el motivo (tipo de excepción)
    failure: Failure | None = None
    counts: Counts


EVENT_SCHEMAS: dict[str, type[_Data]] = {
    "run_started": RunStarted,
    "recovery_started": RecoveryStarted,
    "history_closed": HistoryClosed,
    "model_call": ModelCall,
    "tool_call_routed": ToolCallRouted,
    "approval_requested": ApprovalRequested,
    "approval_decided": ApprovalDecided,
    "mcp_call": McpCall,
    "write_result": WriteResult,
    "write_uncertain": WriteUncertain,
    "scope_reservation": ScopeReservation,
    "run_finished": RunFinished,
}


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    v: Literal[1, 2] = SCHEMA_VERSION
    ts: str  # ISO 8601 con zona horaria
    seq: int = Field(ge=1)  # orden dentro de la ejecución: 1, 2, 3...
    event: Literal[tuple(EVENT_SCHEMAS)]  # type: ignore[valid-type]
    run_id: SafeId
    thread_id: SafeId
    principal_id: SafeId
    data: dict[str, Any]

    @field_validator("ts")
    @classmethod
    def _aware_timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value).tzinfo is None:  # ValueError si no es ISO 8601
            raise ValueError("ts sin zona horaria")
        return value


def validate_event(raw: dict[str, Any]) -> Envelope:
    """Valida el sobre y el ``data`` de su tipo. Lanza ``ValidationError`` si no cumple."""
    envelope = Envelope.model_validate(raw)
    EVENT_SCHEMAS[envelope.event].model_validate(envelope.data)
    return envelope


def json_schema() -> dict[str, Any]:
    """JSON Schema del sobre y de cada tipo de ``data`` (documentación generada)."""
    return {
        "schema": f"arl.trace/{SCHEMA_VERSION}",
        "envelope": Envelope.model_json_schema(),
        "events": {name: model.model_json_schema() for name, model in EVENT_SCHEMAS.items()},
    }
