"""Estado del grafo y modelos de la aprobación humana.

En el estado (que persiste el checkpointer) solo se guardan mensajes y tipos JSON simples; los
modelos Pydantic validan esos dicts al leerlos.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrincipalMismatch(RuntimeError):
    """Un thread se intentó continuar con una identidad distinta de la que lo creó."""


class ApprovalMismatch(RuntimeError):
    """La decisión humana no corresponde a la acción pendiente (antigua, ajena o alterada)."""


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tool_calls: int = Field(default=8, ge=1)  # tool calls del LLM por turno de usuario
    max_model_turns: int = Field(default=6, ge=1)  # invocaciones del LLM por turno de usuario
    # Reintentos automáticos de fallos transitorios del MCP (por nodo, backoff exponencial acotado).
    # Espera por reintento: min(max_interval, initial * 2^(n-1)) + jitter de LangGraph (0 a 1 s).
    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_interval: float = Field(default=0.5, ge=0)
    retry_max_interval: float = Field(default=4.0, ge=0)
    retry_jitter: bool = True

    @property
    def recursion_limit(self) -> int:
        """Red de seguridad de LangGraph, holgada respecto de los límites propios."""
        return 4 * self.max_tool_calls + 3 * self.max_model_turns + 10


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_identity(thread_id: str, tool_call_id: str) -> tuple[str, str]:
    """``(action_id, idempotency_key)`` derivados del thread y de la tool call, fuera del LLM.

    Determinísticos: si ``prepare_write`` se re-ejecuta, la acción y la clave son las mismas.
    """
    digest = _sha256({"thread_id": thread_id, "tool_call_id": tool_call_id})
    return digest[:16], f"arl-{digest[:40]}"


class PendingWrite(BaseModel):
    """Acción de escritura preparada, pendiente de aprobación. ``args_hash`` la sella."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str
    tool_call_id: str
    tool: Literal["update_task"]
    task_id: str
    project_id: str
    my_role: str
    changes: dict[str, Any]
    before: dict[str, Any]
    expected_version: int
    idempotency_key: str
    args_hash: str

    @classmethod
    def seal(cls, **fields: Any) -> PendingWrite:
        """Construye la acción con ``args_hash`` = hash de todos los demás campos."""
        return cls(**fields, args_hash=_sha256(fields))

    @model_validator(mode="after")
    def _check_seal(self) -> PendingWrite:
        if _sha256(self.model_dump(exclude={"args_hash"})) != self.args_hash:
            raise ValueError("la acción pendiente fue alterada después de prepararse")
        return self

    def server_arguments(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "changes": self.changes,
            "expected_version": self.expected_version,
            "idempotency_key": self.idempotency_key,
        }

    def approval_request(self) -> dict[str, Any]:
        """Lo que ve el humano en el interrupt: la acción exacta y su sello."""
        return {
            "type": "write_approval",
            "action_id": self.action_id,
            "args_hash": self.args_hash,
            "tool": self.tool,
            "task_id": self.task_id,
            "project_id": self.project_id,
            "my_role": self.my_role,
            "before": self.before,
            "changes": self.changes,
            "expected_version": self.expected_version,
        }


class ApprovalDecision(BaseModel):
    """Respuesta humana. Solo decide: no puede aportar argumentos nuevos (``extra=forbid``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str
    args_hash: str
    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=500)

    def matches(self, pending: PendingWrite) -> bool:
        return (self.action_id, self.args_hash) == (pending.action_id, pending.args_hash)


def _keep_principal(current: str | None, new: str | None) -> str | None:
    """Reducer: el primer principal de un thread queda fijo; los intentos de cambiarlo se ignoran.

    No lanza excepción a propósito: LangGraph persiste las escrituras de entrada antes de aplicar
    reducers, y un reducer que falla dejaría el thread ilegible para su dueño. El rechazo lo hacen
    ``runtime.run_turn`` (antes de invocar) y el nodo ``begin_turn`` (contra el proceso).
    """
    return current or new


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    principal_id: Annotated[str, _keep_principal]
    queue: list[dict[str, Any]]  # tool calls del último mensaje del LLM, aún sin procesar
    pending_write: dict[str, Any] | None  # PendingWrite serializado
    approval: dict[str, Any] | None  # ApprovalDecision serializada
    approval_error: str | None  # última decisión rechazada por inválida (se vuelve a pedir)
    tool_calls_used: int
    model_turns: int
    status: Literal["running", "completed", "tool_limit", "turn_limit"]
    # Alcance del turno (``arl.agent.scope``): lo fija ``run_turn`` desde la aplicación; ningún
    # nodo lo modifica. ``scope_ledger`` lo escriben solo nodos determinísticos (por action_id).
    request_scope: dict[str, Any] | None
    scope_ledger: dict[str, dict[str, Any]]
    # Resultado de cada escritura propuesta en el turno (``arl.agent.outcomes``), por action_id
    # (o ``call:<tool_call_id>`` si no llegó a prepararse). Solo lo escriben nodos deterministas.
    write_outcomes: dict[str, dict[str, Any]]
