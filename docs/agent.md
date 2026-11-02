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
| `call_model` | LLM | Invoca el modelo y encola sus tool calls; aplica `max_model_turns` |
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
