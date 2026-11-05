"""El aviso de límite en ``user_reply`` cuando el turno tuvo escrituras (6B.8).

Con resultados de escrituras, el aviso del grafo ("Detuve la ejecución…") se leía como si la
escritura se hubiera detenido. En ``user_reply`` se reemplaza por una línea fija que separa lo
confirmado por el servidor, lo que no se aplicó y lo que quedó sin confirmar, y dice que el
modelo no dio una respuesta final. ``TURN_LIMIT_REPLY``, el historial y ``last_ai_text`` no
cambian; sin escrituras, el aviso se muestra igual que antes.
"""

from __future__ import annotations

import pytest

from arl.agent.graph import TOOL_LIMIT_REPLY, TURN_LIMIT_REPLY
from arl.agent.outcomes import compose, entry


def _outcome(seq: int, state: str, task: str = "apo-3", code: str = "x") -> dict:
    return entry(
        seq,
        action_id=f"a{seq}",
        tool_call_id=f"call_{seq}_0",
        task_id=task,
        changes={"status": "done"},
        code=code,
        state=state,
        approval="approve" if state in ("executed", "uncertain", "denied") else None,
    )


def _notice_line(reply: str) -> str:
    return reply.splitlines()[-1]


def test_a_confirmed_write_is_separated_from_the_missing_final_reply() -> None:
    reply, source = compose("", [_outcome(1, "executed")], TURN_LIMIT_REPLY)
    assert source == "agent"
    assert reply.splitlines()[0] == (
        "apo-3: se aplicó el cambio (status → done); lo confirmó el servidor."
    )
    assert _notice_line(reply) == (
        "El modelo no llegó a dar una respuesta final: el sistema detuvo el pedido al alcanzar "
        "el límite de pasos del modelo. Según el sistema: 1 cambio confirmado por el servidor."
    )
    assert "Detuve la ejecución" not in reply


def test_blocked_and_rejected_proposals_are_counted_as_not_applied() -> None:
    outcomes = [
        _outcome(1, "out_of_scope"),
        _outcome(2, "executed"),
        _outcome(3, "rejected", task="apo-2"),
    ]
    reply, _ = compose("", outcomes, TURN_LIMIT_REPLY)
    assert _notice_line(reply) == (
        "El modelo no llegó a dar una respuesta final: el sistema detuvo el pedido al alcanzar "
        "el límite de pasos del modelo. Según el sistema: 1 cambio confirmado por el servidor; "
        "2 propuestas no aplicadas (rechazadas o bloqueadas)."
    )


def test_only_blocked_proposals_say_that_nothing_was_confirmed() -> None:
    reply, _ = compose("", [_outcome(1, "out_of_scope"), _outcome(2, "denied")], TOOL_LIMIT_REPLY)
    assert _notice_line(reply) == (
        "El modelo no llegó a dar una respuesta final: el sistema detuvo el pedido al alcanzar "
        "el límite de tool calls. Según el sistema: ningún cambio confirmado por el servidor; "
        "2 propuestas no aplicadas (rechazadas o bloqueadas)."
    )


def test_an_uncertain_write_is_never_counted_as_confirmed_or_not_applied() -> None:
    reply, _ = compose("", [_outcome(1, "uncertain", code="write_uncertain")], TURN_LIMIT_REPLY)
    assert "no se pudo confirmar si el cambio" in reply.splitlines()[0]
    assert _notice_line(reply) == (
        "El modelo no llegó a dar una respuesta final: el sistema detuvo el pedido al alcanzar "
        "el límite de pasos del modelo. Según el sistema: ningún cambio confirmado por el "
        "servidor; 1 cambio sin confirmar (puede haberse aplicado)."
    )


def test_a_pending_approval_is_never_counted_as_applied_or_uncertain() -> None:
    reply, _ = compose("", [_outcome(1, "pending", code="pending_approval")], TOOL_LIMIT_REPLY)
    assert _notice_line(reply).endswith(
        "ningún cambio confirmado por el servidor; 1 cambio pendiente de aprobación."
    )


def test_unchanged_counts_as_confirmed_by_the_server() -> None:
    reply, _ = compose("", [_outcome(1, "unchanged")], TURN_LIMIT_REPLY)
    assert _notice_line(reply).endswith("Según el sistema: 1 cambio confirmado por el servidor.")


@pytest.mark.parametrize("notice", [TOOL_LIMIT_REPLY, TURN_LIMIT_REPLY])
def test_without_writes_the_notice_is_shown_as_before(notice) -> None:
    assert compose("", [], notice) == (notice, "agent")


def test_without_a_limit_there_is_no_notice_line() -> None:
    reply, _ = compose("Listo.", [_outcome(1, "executed")], None)
    assert reply == "apo-3: se aplicó el cambio (status → done); lo confirmó el servidor."


def test_the_graph_notices_are_unchanged() -> None:
    assert TURN_LIMIT_REPLY == (
        "Detuve la ejecución: se alcanzó el límite de pasos del modelo para este pedido."
    )
    assert TOOL_LIMIT_REPLY.startswith("Detuve la ejecución")
