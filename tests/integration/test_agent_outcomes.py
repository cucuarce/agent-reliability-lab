"""Etapa 6B.6: resultado de cada escritura por acción y respuesta visible autoritativa.

``TurnResult`` separa lo que escribió el modelo (``model_reply``) de lo que ve el usuario
(``user_reply``). Cuando el turno tuvo escrituras propuestas, ``user_reply`` lo arma el agente
con el estado autoritativo de cada acción (``write_outcomes``): la decisión validada del
mecanismo de aprobación para un rechazo; la respuesta del servidor para una escritura. El turno
no se corta: después de un rechazo el modelo puede volver a proponer, con su alcance y su
aprobación. ``reply`` es lo que se muestra (igual a ``user_reply``).
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from arl.agent.faults import EVERY_CALL, Fault
from arl.agent.graph import SYSTEM_PROMPT, TOOL_LIMIT_REPLY
from arl.agent.runtime import (
    TurnStatus,
    approve,
    continue_after_failure,
    reject,
    resume,
    run_turn,
    turn_result,
)
from arl.agent.scope import RequestScope, allow_update
from arl.agent.scripted_model import ScriptedChatModel, call, say
from arl.agent.state import Limits

pytestmark = pytest.mark.anyio

# La respuesta exacta de Groq en las dos corridas reales de L11 (6B.4 y 6B.5).
L11_MODEL_REPLY = (
    "Para asignar la tarea **apo-3** a **bob**, necesito tu aprobación. "
    "¿Confirmas que quieres realizar este cambio?"
)
ASSIGN = {"task_id": "apo-3", "changes": {"assignee_id": "bob"}}
ASSIGN_BOB = RequestScope.writes(allow_update("apo-3", assignee_id=["bob"]))
CLOSE_APO3 = RequestScope.writes(allow_update("apo-3", status=["done"]))
CLOSE_BOTH = RequestScope.writes(
    allow_update("apo-3", status=["done"]), allow_update("apo-2", status=["done"])
)
CLOSE = {"task_id": "apo-3", "changes": {"status": "done"}}


def _db(conn: sqlite3.Connection, task_id: str = "apo-3") -> tuple:
    row = conn.execute(
        "SELECT status, assignee_id, version FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    audits = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
    keys = conn.execute("SELECT count(*) FROM idempotency_keys").fetchone()[0]
    return tuple(row), audits, keys


async def _messages(graph, thread_id: str) -> list:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return state.values["messages"]


def _states(result) -> list[tuple]:
    return [(o["task_id"], o["state"], o["executed"], o["approval"]) for o in result.write_outcomes]


# --- El caso real de L11 -------------------------------------------------------------------------


async def test_l11_reply_asking_for_approval_again_is_not_what_the_user_sees(open_agent, conn):
    model = ScriptedChatModel([call("update_task", **ASSIGN), say(L11_MODEL_REPLY)])
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="Asigná apo-3 a bob.", scope=ASSIGN_BOB)
        done = await resume(graph, thread_id="t", decision=reject(paused.approval_request, "eval"))

    assert done.status is TurnStatus.COMPLETED
    assert done.model_reply == L11_MODEL_REPLY  # lo que dijo el modelo, intacto
    assert done.reply == done.user_reply  # ``reply`` es lo que se muestra (6B.6)
    assert done.last_ai_text == L11_MODEL_REPLY  # el valor histórico, para evaluaciones
    assert done.reply_source == "agent"
    assert "apo-3" in done.user_reply and "rechaz" in done.user_reply.casefold()
    assert "no se aplicó" in done.user_reply
    for misleading in ("necesito tu aprobación", "¿confirm", "se aplicó el cambio", "pendiente"):
        assert misleading not in done.user_reply.casefold()
    assert _states(done) == [("apo-3", "rejected", False, "reject")]
    (outcome,) = done.write_outcomes
    assert outcome["code"] == "rejected_by_user"
    assert outcome["action_id"] == paused.approval_request["action_id"]
    assert outcome["changes"] == {"assignee_id": "bob"}
    assert _db(conn) == (("todo", None, 1), 0, 0)


async def test_the_rejection_tool_message_is_structured_and_keeps_its_prefix(open_agent) -> None:
    model = ScriptedChatModel([call("update_task", **ASSIGN), say(L11_MODEL_REPLY)])
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="Asigná apo-3 a bob.", scope=ASSIGN_BOB)
        await resume(graph, thread_id="t", decision=reject(paused.approval_request, "todavía no"))
        messages = await _messages(graph, "t")

    (tool,) = [m for m in messages if isinstance(m, ToolMessage)]
    (proposal,) = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
    assert tool.tool_call_id == proposal.tool_calls[0]["id"]  # correlación intacta
    assert tool.status == "error"
    assert tool.content.startswith("ERROR [rejected_by_user] ")  # lo que leen los parsers
    assert "todavía no" in tool.content and "no vuelvas a pedir" in tool.content
    structured = json.loads(tool.content.split("Resultado: ", 1)[1])
    assert structured == {
        "action_id": paused.approval_request["action_id"],
        "task_id": "apo-3",
        "code": "rejected_by_user",
        "state": "rejected",
        "executed": False,
        "approval": "reject",
    }


def test_the_model_instructions_explain_out_of_band_approval_and_final_rejections() -> None:
    prompt = SYSTEM_PROMPT.casefold()
    assert "fuera de este chat" in prompt
    assert "rejected_by_user" in prompt and "ya se tomó" in prompt
    assert "no vuelvas a pedir" in prompt
    for code in ("forbidden", "out_of_scope"):
        assert code in prompt
    assert "solo digas que un cambio se hizo si update_task lo confirmó" in prompt


# --- Un resultado por acción, de la fuente autoritativa -----------------------------------------


async def test_an_executed_write_is_reported_from_the_server_confirmation(open_agent, conn):
    model = ScriptedChatModel([call("update_task", **CLOSE), say("Listo.")])
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)
        assert _states(paused) == [("apo-3", "pending", False, None)]
        assert "espera tu aprobación" in paused.user_reply
        done = await resume(graph, thread_id="t", decision=approve(paused.approval_request))

    assert (done.model_reply, done.reply_source) == ("Listo.", "agent")
    assert done.reply == done.user_reply != "Listo."
    assert _states(done) == [("apo-3", "executed", True, "approve")]
    assert done.write_outcomes[0]["code"] == "updated"
    assert done.write_outcomes[0]["replayed"] is False
    assert "apo-3" in done.user_reply and "se aplicó" in done.user_reply
    assert _db(conn) == (("done", None, 2), 1, 1)


async def test_a_turn_without_writes_shows_the_model_reply(open_agent) -> None:
    model = ScriptedChatModel([call("list_projects"), say("Hay 2 proyectos.")])
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(
            graph, thread_id="t", text="¿proyectos?", scope=RequestScope.read_only()
        )
    assert (result.user_reply, result.model_reply, result.reply_source) == (
        "Hay 2 proyectos.",
        "Hay 2 proyectos.",
        "model",
    )
    assert result.write_outcomes == []


async def test_a_permission_denied_by_the_server_is_reported_as_denied(open_agent, conn) -> None:
    model = ScriptedChatModel([call("update_task", **CLOSE), say("Listo, la cerré.")])
    async with open_agent("carol", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id="t", decision=approve(paused.approval_request))

    assert _states(done) == [("apo-3", "denied", False, "approve")]
    assert done.write_outcomes[0]["code"] == "forbidden"
    assert "permiso" in done.user_reply and "no se aplicó" in done.user_reply
    assert "la cerré" not in done.user_reply  # la afirmación falsa del modelo no se muestra
    assert done.model_reply == "Listo, la cerré."
    assert _db(conn) == (("todo", None, 1), 0, 0)


async def test_an_out_of_scope_proposal_is_reported_without_an_action(open_agent, conn) -> None:
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-4", changes={"status": "done"}), say("Hecho.")]
    )
    async with open_agent("alice", model) as (graph, _):
        result = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)

    assert _states(result) == [("apo-4", "out_of_scope", False, None)]
    assert result.write_outcomes[0]["action_id"] is None  # nunca se preparó ni se aprobó
    assert result.write_outcomes[0]["code"] == "out_of_scope"
    assert "fuera" in result.user_reply and "apo-4" in result.user_reply
    assert _db(conn, "apo-4")[1:] == (0, 0)


async def test_an_unchanged_write_is_not_reported_as_a_change(open_agent, conn) -> None:
    keep = RequestScope.writes(allow_update("apo-3", status=["todo"]))
    model = ScriptedChatModel(
        [call("update_task", task_id="apo-3", changes={"status": "todo"}), say("Listo.")]
    )
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="dejá apo-3 en todo", scope=keep)
        done = await resume(graph, thread_id="t", decision=approve(paused.approval_request))

    assert _states(done) == [("apo-3", "unchanged", False, "approve")]
    assert done.write_outcomes[0]["code"] == "unchanged"
    assert "ya tenía" in done.user_reply and "se aplicó" not in done.user_reply
    assert _db(conn) == (("todo", None, 1), 0, 1)


async def test_a_replayed_write_is_reported_once_without_duplicating(open_faulty_agent, conn):
    model = ScriptedChatModel([call("update_task", **CLOSE), say("Listo.")])
    plan = {("update_task", 0): Fault.TIMEOUT_AFTER}  # se pierde la respuesta: reenvío
    async with open_faulty_agent("alice", model, plan) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)
        done = await resume(graph, thread_id="t", decision=approve(paused.approval_request))

    assert _states(done) == [("apo-3", "executed", True, "approve")]
    assert done.write_outcomes[0]["replayed"] is True
    assert "sin duplicar" in done.user_reply
    assert _db(conn) == (("done", None, 2), 1, 1)


async def test_an_uncertain_write_is_never_reported_as_executed(open_faulty_agent, conn) -> None:
    model = ScriptedChatModel([call("update_task", **CLOSE), say("Listo.")])
    plan = {("update_task", EVERY_CALL): Fault.TIMEOUT_AFTER}
    async with open_faulty_agent("alice", model, plan) as (graph, faulty):
        paused = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)
        uncertain = await resume(graph, thread_id="t", decision=approve(paused.approval_request))
        assert uncertain.status is TurnStatus.WRITE_UNCERTAIN
        assert _states(uncertain) == [("apo-3", "uncertain", None, "approve")]
        assert "no se pudo confirmar" in uncertain.user_reply
        assert "se aplicó el cambio" not in uncertain.user_reply
        faulty.clear()
        done = await continue_after_failure(graph, thread_id="t")

    assert _states(done) == [("apo-3", "executed", True, "approve")]
    assert _db(conn) == (("done", None, 2), 1, 1)


# --- Resultados mixtos, varias propuestas y el turno que sigue -----------------------------------


async def test_a_rejected_action_proposed_again_needs_its_own_approval(open_agent, conn) -> None:
    """El rechazo no corta el turno ni veta la acción: se vuelve a proponer, pasa otra vez por
    el alcance y la aprobación, y cada acción se informa por separado (como scope-rejected-
    releases en la suite v2)."""
    model = ScriptedChatModel(
        [call("update_task", **CLOSE), call("update_task", **CLOSE), say("Listo.")]
    )
    async with open_agent("alice", model) as (graph, _):
        first = await run_turn(graph, thread_id="t", text="cerrá apo-3", scope=CLOSE_APO3)
        second = await resume(graph, thread_id="t", decision=reject(first.approval_request))
        assert second.status is TurnStatus.AWAITING_APPROVAL  # nueva aprobación, no un veto
        assert second.approval_request["action_id"] != first.approval_request["action_id"]
        assert _states(second) == [
            ("apo-3", "rejected", False, "reject"),
            ("apo-3", "pending", False, None),
        ]
        done = await resume(graph, thread_id="t", decision=approve(second.approval_request))

    assert _states(done) == [
        ("apo-3", "rejected", False, "reject"),
        ("apo-3", "executed", True, "approve"),
    ]
    lines = done.user_reply.splitlines()
    assert len(lines) == 2 and "rechaz" in lines[0] and "se aplicó" in lines[1]
    assert _db(conn) == (("done", None, 2), 1, 1)


async def test_mixed_outcomes_on_two_tasks_are_reported_separately(open_agent, conn) -> None:
    model = ScriptedChatModel(
        [
            call("update_task", **CLOSE),
            call("update_task", task_id="apo-2", changes={"status": "done"}),
            say("Cerré las dos."),
        ]
    )
    async with open_agent("alice", model) as (graph, _):
        first = await run_turn(graph, thread_id="t", text="cerrá apo-3 y apo-2", scope=CLOSE_BOTH)
        second = await resume(graph, thread_id="t", decision=approve(first.approval_request))
        done = await resume(graph, thread_id="t", decision=reject(second.approval_request))

    assert _states(done) == [
        ("apo-3", "executed", True, "approve"),
        ("apo-2", "rejected", False, "reject"),
    ]
    executed, rejected = done.user_reply.splitlines()
    assert "apo-3" in executed and "se aplicó" in executed
    assert "apo-2" in rejected and "rechaz" in rejected
    assert "las dos" not in done.user_reply and done.model_reply == "Cerré las dos."


async def test_history_checkpoint_and_tool_call_ids_stay_consistent(open_agent) -> None:
    """No se agrega ningún mensaje al historial: los ids siguen la numeración de siempre, el
    resultado se reconstruye desde el checkpoint y el turno siguiente empieza limpio."""
    model = ScriptedChatModel(
        [call("update_task", **ASSIGN), say(L11_MODEL_REPLY), say("apo-3 está sin asignar.")]
    )
    async with open_agent("alice", model) as (graph, _):
        paused = await run_turn(graph, thread_id="t", text="Asigná apo-3 a bob.", scope=ASSIGN_BOB)
        done = await resume(graph, thread_id="t", decision=reject(paused.approval_request))
    async with open_agent("alice", model) as (graph, _):  # otro "proceso": solo el checkpoint
        again = await turn_result(graph, "t")
        assert (again.user_reply, again.write_outcomes) == (done.user_reply, done.write_outcomes)
        nxt = await run_turn(graph, thread_id="t", text="¿apo-3?", scope=RequestScope.read_only())
        messages = await _messages(graph, "t")

    ai = [m for m in messages if isinstance(m, AIMessage)]
    assert [m.text for m in ai] == ["", L11_MODEL_REPLY, "apo-3 está sin asignar."]
    assert [c["id"] for m in ai for c in m.tool_calls] == ["call_0_0"]
    assert (nxt.write_outcomes, nxt.reply_source) == ([], "model")


async def test_agent_written_text_is_never_attributed_to_the_model(open_agent) -> None:
    model = ScriptedChatModel(
        [call("list_projects"), call("list_projects"), call("list_projects"), say("x")]
    )
    async with open_agent("alice", model, Limits(max_tool_calls=2)) as (graph, _):
        result = await run_turn(graph, thread_id="t", text="listá", scope=RequestScope.read_only())
    assert result.status is TurnStatus.TOOL_LIMIT
    assert result.reply == TOOL_LIMIT_REPLY  # aviso del agente, sin escrituras
    assert result.model_reply == ""  # el texto del límite lo escribió el agente, no el LLM
    assert (result.user_reply, result.reply_source) == (TOOL_LIMIT_REPLY, "agent")
