"""Registro de eventos de traza: best-effort, nunca interfiere con la ejecución.

El runtime abre una ejecución (``run_scope``) y la publica en una ``ContextVar``. Los nodos del
grafo y el gateway llaman a ``emit``, que no hace nada fuera de una ejecución. Garantías:

- **``emit`` nunca lanza.** Un fallo al validar, serializar o escribir descarta el evento, suma
  ``dropped`` y deja un aviso en el log (una vez por ejecución). Como ``emit`` se llama desde los
  nodos, una excepción ahí haría que LangGraph reintentara el nodo: con ``execute_write``, una
  nueva llamada de escritura. Por eso no puede lanzar.
- **Un archivo por ejecución** (``<trace_dir>/thread-<hash>/<run_id>.jsonl``): lo escribe un solo
  proceso, así que no hay escrituras concurrentes sobre el mismo archivo. Cada evento se agrega y
  se cierra el archivo de inmediato: un crash pierde a lo sumo el evento en curso.

Las trazas son operativas. La fuente de verdad de lo que se escribió es el ``audit_log`` del
servidor (ver docs/observability.md).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from arl.agent.redaction import LOGGER
from arl.tracing.events import EVENT_SCHEMAS, Counts, Envelope, safe_id


class TraceSink(Protocol):
    def write(self, run: RunContext, line: str) -> None: ...


def run_file(trace_dir: Path, thread_id: str, run_id: str) -> Path:
    """Ruta del archivo de una ejecución. El thread_id se usa solo como hash, nunca como ruta."""
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:32]
    return trace_dir / f"thread-{digest}" / f"{run_id}.jsonl"


class JsonlTraceSink:
    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir

    def write(self, run: RunContext, line: str) -> None:
        path = run_file(self.trace_dir, run.thread_id, run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


@dataclass
class RunContext:
    sink: TraceSink | None
    thread_id: str
    principal_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started: float = field(default_factory=time.perf_counter)
    seq: int = 0
    dropped: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {"model_calls": 0, "mcp_calls": 0, "mcp_failures": 0}
    )
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)
    failure: dict[str, str] | None = None  # code/operation/error_type del fallo final

    def emit(self, event: str, data: dict[str, Any]) -> None:
        # El seq se asigna antes de validar: un evento descartado deja un hueco que el lector
        # informa (la ausencia queda a la vista, no se disimula).
        self.seq += 1
        try:
            model = EVENT_SCHEMAS[event].model_validate(data)
            payload = model.model_dump(mode="json", exclude_none=True)
            envelope = Envelope(
                ts=datetime.now(UTC).isoformat(timespec="microseconds"),
                seq=self.seq,
                event=event,  # type: ignore[arg-type]
                run_id=self.run_id,
                thread_id=safe_id(self.thread_id),
                principal_id=safe_id(self.principal_id),
                data=payload,
            )
            if self.sink is not None:
                self.sink.write(self, envelope.model_dump_json())
        except Exception as exc:  # best-effort: nunca afecta la ejecución
            self.dropped += 1
            if self.dropped == 1:
                LOGGER.warning(
                    "trazas: evento descartado run_id=%s event=%s error_type=%s",
                    self.run_id, safe_id(event), type(exc).__name__,
                )  # fmt: skip

    def snapshot_counts(self) -> Counts:
        return Counts(**self.counts)


_RUN: ContextVar[RunContext | None] = ContextVar("arl_trace_run", default=None)
# (nodo, tool_call_id, intento) del paso en curso: el gateway lo adjunta a cada llamada MCP.
_STEP: ContextVar[tuple[str, str, int] | None] = ContextVar("arl_trace_step", default=None)


def current_run() -> RunContext | None:
    return _RUN.get()


def emit(event: str, **data: Any) -> None:
    """Registra un evento en la ejecución en curso; sin ejecución, no hace nada."""
    run = _RUN.get()
    if run is not None:
        run.emit(event, data)


def count(name: str) -> None:
    run = _RUN.get()
    if run is not None:
        run.counts[name] += 1


@contextmanager
def run_scope(sink: TraceSink | None, *, thread_id: str, principal_id: str) -> Iterator[RunContext]:
    run = RunContext(sink=sink, thread_id=thread_id, principal_id=principal_id)
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


@contextmanager
def step(node: str, tool_call_id: str) -> Iterator[None]:
    """Un intento de un nodo con efectos. LangGraph reintenta el nodo completo: cada entrada
    cuenta como un intento nuevo de esa tool call dentro de la ejecución."""
    run = _RUN.get()
    attempt = 1
    if run is not None:
        key = (node, tool_call_id)
        attempt = run.attempts.get(key, 0) + 1
        run.attempts[key] = attempt
    token = _STEP.set((node, safe_id(tool_call_id), attempt))
    try:
        yield
    finally:
        _STEP.reset(token)


def current_step() -> dict[str, Any]:
    current = _STEP.get()
    if current is None:
        return {}
    node, tool_call_id, attempt = current
    return {"node": node, "tool_call_id": tool_call_id, "attempt": attempt}


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
