# Agente LangGraph con aprobación humana

`StateGraph` propio que consulta proyectos por MCP y propone modificaciones que solo se ejecutan
después de una aprobación humana ligada a la acción exacta.

> **No hay autenticación real:** la identidad es **simulada** (`ARL_PRINCIPAL` del proceso
> servidor). **El LLM también es simulado:** los tests usan un modelo falso determinístico
> (`ScriptedChatModel`) y todavía no hay un proveedor real integrado.

## Modelo de seguridad del MVP

- **Interfaz confiable: `arl.agent.runtime`.** `run_turn`, `resume`, `continue_after_failure` y
  `turn_result` verifican, antes de leer o escribir, que el thread pertenezca al principal del
  proceso. Un principal ajeno recibe `PrincipalMismatch`, y los checkpoints y la base quedan
  idénticos byte a byte (`test_agent_isolation.py`).
- **Fuera del modelo:** invocar el grafo directamente (`graph.ainvoke`, `aupdate_state`) o leer
  el archivo de checkpoints. Ahí los nodos siguen impidiendo ejecuciones indebidas (defensa en
  profundidad), pero pueden quedar entradas en el historial de un thread ajeno, y el archivo es
  legible por cualquiera con acceso local.
- **Autoridad sobre los datos:** el servidor MCP (`ProjectService`), con la identidad de su propio
  proceso.

### Quién puede aprobar

La aprobación no la decide `action_id`/`args_hash` solo. Esos dos campos ligan la decisión a la
acción; **quién** puede decidir lo determina la identidad del proceso:

1. Solo el **dueño del thread** puede aprobar o rechazar. El dueño es el principal que lo creó,
   fijado en el estado y sin posibilidad de cambiarlo.
2. `resume` toma la identidad del proceso (la del gateway, ligada al grafo) y la compara con el
   dueño **antes** de entregar la decisión. `await_approval` vuelve a compararla al reanudar.
3. La decisión no puede declarar un aprobador: `ApprovalDecision` rechaza campos extra como
   `approved_by`.
4. La escritura aprobada se ejecuta con esa misma identidad. El servidor la autoriza contra su
   política (un viewer que aprueba recibe `[forbidden]`) y la audita con `actor_id` = aprobador =
   dueño del thread.

Tests: `test_thread_cannot_be_resumed_by_another_principal`,
`test_other_principal_cannot_read_or_alter_a_paused_thread`,
`test_consecutive_invalid_decisions_do_not_disable_the_thread` (incluye `approved_by`) y
`test_viewer_write_is_rejected_by_the_server_after_approval`.

No existe la aprobación delegada (que una persona apruebe acciones de otra). Sería una extensión
posterior y necesitaría una política de aprobadores en el servidor.

## Componentes

| Módulo | Responsabilidad |
|---|---|
| `agent/tool_registry.py` | Registro READ/WRITE en código y args que ve el LLM (sin identidad, versión ni clave) |
| `agent/gateway.py` | Cliente MCP: llamadas y verificación de contrato con el servidor |
| `agent/state.py` | `AgentState`, `PendingWrite` sellada (hash), `ApprovalDecision`, `Limits` |
| `agent/graph.py` | Nodos y rutas del grafo |
| `agent/runtime.py` | `run_turn`, `resume`, `continue_after_failure`, `turn_result` |
| `agent/scripted_model.py` | LLM falso determinístico (reanudable: decide según el historial) |
| `agent/locks.py` | Lock por thread entre procesos, obligatorio (ver [reliability.md](reliability.md)) |
| `agent/history.py` | Contrato de tool calling: cierre de llamadas interrumpidas y verificación antes del LLM |
| `agent/scope.py` | Alcance del turno (`RequestScope`): escrituras autorizadas por la aplicación, reservas y sello (ver [scope.md](scope.md)) |
| `agent/redaction.py` | Logger `arl.agent` con filtro de secretos (segunda barrera) |
| `tracing/` | Eventos de traza por ejecución, consulta y logging (ver [observability.md](observability.md)) |
| `agent/faults.py` | Inyección de fallos determinística entre el agente y el MCP (tests y evaluación) |

Clasificación de fallos, reintentos y estados `INTERRUPTED` / `WRITE_UNCERTAIN` / `BUSY`: ver
[reliability.md](reliability.md).

## Grafo

```
START → begin_turn → call_model ─┬→ END (sin tool calls / límite de turnos)
                                 └→ dispatch
dispatch ─┬→ execute_read ─→ dispatch
          ├→ prepare_write ─┬→ dispatch (tarea no visible)
          │                 └→ await_approval ─┬→ execute_write ────→ dispatch
          │                                    ├→ record_rejection ─→ dispatch
          │                                    └→ await_approval (decisión inválida)
          ├→ call_model (cola vacía)
          └→ END (límite de tool calls)
```

| Nodo | Efectos | Qué hace |
|---|---|---|
| `begin_turn` | — | Verifica el principal del thread contra el del proceso y resetea los contadores del turno |
| `call_model` | LLM | Verifica el contrato de tool calling del historial, invoca el modelo y encola sus tool calls; aplica `max_model_turns` |
| `dispatch` | — | Descarta tool calls desconocidas o inválidas (`[invalid_tool_call]`), aplica `max_tool_calls` y elige la ruta |
| `execute_read` | lectura MCP | Ejecuta `list_projects` / `get_project` sin aprobación |
| `prepare_write` | lecturas MCP | Ubica la tarea, toma `version`, genera `action_id` + `idempotency_key` y sella la `PendingWrite` |
| `await_approval` | — | Solo `interrupt()`; valida la decisión y, si es inválida, vuelve a pedirla |
| `execute_write` | **escritura MCP** | Ejecuta exactamente la acción sellada y aprobada |
| `record_rejection` | — | Responde la tool call con `[rejected_by_user]` |

## Interrupt / resume

1. `prepare_write` guarda en el estado una `PendingWrite` sellada: `args_hash` es el hash de todos
   sus campos. El checkpointer persiste el estado en SQLite (`AsyncSqliteSaver`, `durability="sync"`).
2. `await_approval` llama a `interrupt(pending.approval_request())`. El grafo se detiene y la
   solicitud queda en el checkpoint: sobrevive a un reinicio del proceso.
3. `runtime.resume(graph, thread_id, decision)` valida **antes** de tocar el grafo:
   - que haya una aprobación pendiente;
   - que el thread pertenezca al principal del proceso;
   - que `action_id` y `args_hash` coincidan.

   Después envía `Command(resume=decision)`.
4. LangGraph re-ejecuta `await_approval` desde el inicio. `interrupt()` devuelve la decisión, y el
   nodo vuelve a validarla como defensa en profundidad.
5. `approve` lleva a `execute_write`, que llama a `update_task` con `expected_version` e
   `idempotency_key` de la acción sellada. `reject` lleva a `record_rejection`.

### Comportamientos de LangGraph verificados en esta etapa

- **Valor de resume rechazado.** Si un nodo lanza una excepción después de `interrupt()`, el valor
  rechazado queda registrado y se vuelve a entregar en la reanudación siguiente, aunque esta sea
  correcta. Por eso `await_approval` nunca lanza ante una decisión inválida.
- **Reducers.** Las escrituras de entrada se persisten antes de aplicar los reducers. Un reducer
  que falla deja el thread ilegible (`aget_state` falla). Por eso el reducer del principal ignora
  los cambios en lugar de lanzar.
- **`aupdate_state` sobre un thread pausado** consume el interrupt.

## Resultado de cada escritura y respuesta visible (6B.6)

Motivo: en las dos corridas reales de L11 el modelo recibió `ERROR [rejected_by_user]` y
contestó "necesito tu aprobación. ¿Confirmas…?", como si la decisión siguiera pendiente. El
agente sabía con certeza que se había rechazado, pero solo lo comunicaba el texto del LLM.

- **`write_outcomes`** (estado del agente, en el checkpoint; `src/arl/agent/outcomes.py`): un
  resultado por `action_id` (o `call:<tool_call_id>` si la propuesta no llegó a prepararse), con
  `code`, `state`, `executed`, `approval` y `replayed`. Lo escriben solo nodos deterministas, desde
  la fuente autoritativa: `record_rejection` (la decisión validada contra `action_id` y
  `args_hash`), `execute_write` (la respuesta del servidor: `updated`, `unchanged`, replay,
  `forbidden`, `not_found`, `invalid_change`, `version_conflict`) y `dispatch`/`prepare_write`
  (fuera del alcance, tarea no visible, tool call inválida, límite). Proponer `update_task` no es
  un resultado. `pending` y `uncertain` (`WRITE_UNCERTAIN`, `executed: null`) los deriva el
  runtime del checkpoint. Se reinicia en cada turno, como `scope_ledger`.
- **`TurnResult`** suma `model_reply` (solo texto del LLM: nunca `TOOL_LIMIT_REPLY` ni
  `TURN_LIMIT_REPLY`), `user_reply` (lo que se muestra), `reply_source` (`model` | `agent`) y
  `write_outcomes`. Con escrituras en el turno, `user_reply` lo arma el agente con una línea por
  acción (resultados mixtos por separado); sin escrituras es `model_reply`. **`reply` es lo que
  se muestra** (igual a `user_reply`): el campo obvio es el seguro. `model_reply` queda como
  evidencia de lo que generó el LLM y `last_ai_text` conserva el valor que `reply` tenía antes
  (último texto del asistente), solo para evaluaciones comparables con las históricas.
- **Ningún mensaje del modelo se modifica ni se inventa**: los ids de tool calls y el checkpoint
  no cambian fuera de `write_outcomes`. El turno no se corta al rechazar: el permiso se libera y
  el modelo puede volver a proponer, con su alcance y su aprobación (no hay veto persistente).
- **Aviso de límite con escrituras (6B.8).** Si el turno termina por `turn_limit` o `tool_limit`
  y hubo resultados, `user_reply` no repite "Detuve la ejecución…" (se leía como si la escritura
  se hubiera detenido): la última línea dice que el modelo no llegó a dar una respuesta final y
  cuenta, según el sistema, los cambios confirmados por el servidor, las propuestas no aplicadas
  (rechazadas o bloqueadas) y los cambios sin confirmar o pendientes. `TURN_LIMIT_REPLY`, el
  historial y `last_ai_text` no cambian; sin escrituras, el aviso es el de siempre. Tests:
  `tests/unit/test_outcomes_limit_notice.py`.
- **Turno siguiente.** El historial guarda el texto original del modelo, que el usuario quizá no
  vio. Si el turno anterior terminó con `reply_source = "agent"`, `run_turn` agrega antes del
  mensaje nuevo una **nota del sistema** (`SystemMessage`, no del modelo ni del usuario) que
  resume lo que el usuario vio y aclara que esas decisiones ya están tomadas. Queda en el
  checkpoint como parte del historial. Test de dos turnos: `tests/integration/test_agent_turn_context.py`.
- **La nota nunca interpola `user_reply`**: este puede llevar valores que propuso el LLM (un
  `title` libre, un `task_id` o un valor fuera del alcance, quizá copiados de contenido de una
  tool), y un `SystemMessage` les daría prioridad alta. `outcomes.system_note` la arma con
  plantillas fijas por estado desde `write_outcomes`: el `tool_call_id` solo si tiene el formato
  del agente (`call_<n>_<i>`), los campos solo si son de `update_task`, y una tarea o un valor
  solo si la aplicación los declaró en el alcance del turno (un campo con `None` no declara
  valores). Lo demás se omite. Solo afirma que no queda nada pendiente si ningún resultado es
  `pending` o `uncertain`.
- **Instrucciones** (`SYSTEM_PROMPT`): la aprobación la pide el sistema fuera del chat;
  `rejected_by_user` es una decisión ya tomada; no volver a pedirla; solo afirmar cambios
  confirmados; distinguir rechazo, `forbidden`, `out_of_scope` y fallo técnico. El `ToolMessage`
  del rechazo conserva `ERROR [rejected_by_user]` y agrega `Resultado: {…}` con `action_id`,
  `code`, `state`, `executed` y `approval`.

Tests: `tests/integration/test_agent_outcomes.py` (la respuesta real de L11, ejecutada,
rechazada, denegada, fuera del alcance, `unchanged`, replay, `write_uncertain`, pendiente,
resultados mixtos, la misma tarea propuesta dos veces, checkpoint e ids, y texto del agente que
nunca se atribuye al modelo).

## Garantías y dónde se prueban

| # | Garantía | Tests (`tests/integration/`) |
|---|---|---|
| 1 | Lecturas por MCP sin aprobación | `test_read_flow_queries_projects_and_tasks_without_approval` |
| 2 | Nada se escribe antes de aprobar | `test_approved_write_runs_only_after_approval` |
| 3 | El rechazo no llega al servidor | `test_rejected_write_never_reaches_the_server` |
| 4 | Sin permisos: rechaza el servidor | `test_viewer_write_is_rejected_by_the_server_after_approval`, `test_non_member_cannot_even_reach_approval` |
| 5 | Reanudación tras reiniciar el proceso | `test_resume_after_process_restart` |
| 6 | Aprobación antigua o ajena | `test_consecutive_invalid_decisions_do_not_disable_the_thread`, `test_execute_write_refuses_an_approval_injected_into_the_state`, `test_approval_for_another_action_is_refused`, `test_invalid_decision_sent_directly_to_graph_does_not_poison_the_thread`, `test_old_approval_cannot_approve_a_later_action`, `test_tampered_pending_action_is_detected`, `test_execution_uses_exactly_the_approved_arguments` |
| 7 | Versión cambiada en pausa | `test_task_changed_while_paused_yields_version_conflict` |
| 8 | Reintento sin duplicar | `test_crash_after_server_commit_retries_without_duplicate`, `test_process_killed_after_commit_before_checkpoint_resumes_once` (subproceso real terminado con `os._exit`), `test_two_approved_writes_in_one_thread_get_distinct_keys` |
| 9 | Suplantación | `test_other_principal_cannot_read_or_alter_a_paused_thread`, `test_other_principal_cannot_continue_a_crashed_thread`, `test_identity_or_control_arguments_from_llm_are_rejected`, `test_claiming_another_identity_in_the_prompt_changes_nothing`, `test_thread_cannot_be_resumed_by_another_principal`, `test_thread_cannot_be_continued_by_another_principal` |
| 10 | Bucles controlados | `test_tool_call_loop_stops_at_the_tool_limit`, `test_many_parallel_calls_in_one_message_are_capped`, `test_model_turn_loop_stops_at_the_turn_limit` |

## Limitaciones

- **Sin LLM real.** Las pruebas usan un guion determinístico; el comportamiento con un modelo real
  se mide en la suite conductual (etapa 6). El proveedor sigue sin decidirse.
- **Acceso directo al grafo:** queda fuera del modelo de seguridad (ver arriba).
- **Búsqueda N+1 en `prepare_write`.** Para ubicar la tarea hace 1 `list_projects` + hasta P
  `get_project` (P = proyectos visibles) y corta al encontrarla. Cada llamada es un round-trip MCP
  y abre una conexión SQLite en el servidor. Medido en `test_prepare_write_lookup_cost`: 2 lecturas
  si la tarea está en el primer proyecto, P+1 en el peor caso (incluso cuando la tarea no es
  visible). Estas lecturas no cuentan en `max_tool_calls`. Con los datos sintéticos (P ≤ 3) el
  costo es despreciable. Si P crece, conviene una tool `get_task` en el servidor (decisión
  pendiente).
- **Un thread no puede reanudarse después de alterar `pending_write`.** El sello lo detecta, pero no
  hay herramienta de recuperación: hay que abrir un thread nuevo.
- **Los permisos no se pre-verifican en el agente.** Un viewer recibe la solicitud de aprobación
  (con `my_role: viewer` visible) y el servidor rechaza al ejecutar. Es una decisión deliberada para
  no duplicar reglas.
