# Tolerancia a fallos y confiabilidad (etapa 4)

> Todo lo de esta página se verificó con datos sintéticos, un LLM simulado y procesos locales.
> No es una garantía de comportamiento en producción.

## Clasificación de fallos

El `McpGateway` clasifica cada llamada MCP de una de estas tres maneras:

| Clase | Ejemplos | Reintento automático | Resultado si persiste |
|---|---|---|---|
| **Dominio** (definitivo) | `[forbidden]`, `[not_found]`, `[version_conflict]`, `[idempotency_conflict]`, `[invalid_change]`, rechazo humano | **Nunca** | Se le informa al LLM como `ToolMessage` de error |
| **`ToolUnavailable`** (transitorio) | `MCPError(REQUEST_TIMEOUT)`, `MCPError(CONNECTION_CLOSED)`, `OSError`/`anyio` de transporte, error opaco `Error executing tool X` | Sí, acotado | Lectura → `INTERRUPTED`; escritura → **`WRITE_UNCERTAIN`** |
| **`ToolResponseInvalid`** (definitivo) | Respuesta fuera de contrato, de otro recurso, vacía; otro `MCPError` de protocolo | No | Lectura → `INTERRUPTED`; escritura → **`WRITE_UNCERTAIN`** |

Los códigos de error del SDK se verificaron contra mcp 2.3 con un servidor real: un timeout da
`-32001` y una conexión cerrada da `-32000`.

**Reintentos.** Se usa la `RetryPolicy` de LangGraph en los nodos `execute_read`,
`prepare_write` y `execute_write`, con `retry_on=is_retryable`.
- Valores por defecto: 3 intentos y backoff `min(4 s, 0.5 s × 2^(n-1))` más un jitter de 0 a 1 s
  (`Limits`).
- `execute_write` reintenta con la misma `PendingWrite` sellada, es decir, con la **misma clave de
  idempotencia**. El servidor deduplica.

## Estados del runtime

| `TurnStatus` | Significado | Qué hacer |
|---|---|---|
| `INTERRUPTED` | Un fallo detuvo la ejecución **antes de cualquier escritura** | `continue_after_failure`. Si faltaba aprobar, se vuelve a pedir la aprobación. |
| `WRITE_UNCERTAIN` | Una escritura **aprobada** no pudo confirmarse: puede haber ocurrido o no | `continue_after_failure` la reintenta con la misma clave: replay si ocurrió, ejecución si no. `run_turn` se rechaza (`ThreadNeedsRecovery`) para no descartarla. Ver [reconciliación](#resultado-incierto-y-reconciliación). |
| `BUSY` | Otro proceso está ejecutando el thread | Reintentar más tarde |

`TurnResult.error` lleva un código estable (`[unavailable] update_task`), `trace_id` correlaciona
el resultado con el log de la ejecución y `pending_action` muestra la acción aprobada que quedó sin
confirmar, con su `idempotency_key`. Ningún mensaje incluye el texto del error original (ver
[logs](#logs-sin-secretos)).

## Política de historial

Contrato de los proveedores reales (Anthropic, OpenAI): cada tool call de un mensaje del asistente
recibe exactamente una respuesta, inmediatamente después de ese mensaje. Un fallo que detiene la
ejecución deja tool calls sin respuesta. Qué pasa con ellas depende de si pudieron tener efectos:

| Estado del thread | `continue_after_failure` | `run_turn` (mensaje nuevo) |
|---|---|---|
| `INTERRUPTED` en una lectura | Reintenta la lectura | Cierra las tool calls pendientes con `[not_executed]` y después agrega el mensaje |
| `INTERRUPTED` al preparar una escritura | Vuelve a preparar y **vuelve a pedir aprobación** | Igual: `[not_executed]`; nunca se envió una escritura |
| `AWAITING_APPROVAL` | `ThreadAwaitingApproval` | `ThreadAwaitingApproval`: hay que aprobar o rechazar (`[rejected_by_user]`) |
| `WRITE_UNCERTAIN` | Reintenta con la **misma clave** | `ThreadNeedsRecovery`: **nunca se cierra**, porque cerrarla inventaría éxito o fracaso |

Ejemplo: el modelo pidió dos lecturas en un mensaje y la primera cayó de forma persistente. El
usuario escribe "dejalo, otra cosa". El historial que recibe el modelo queda así:

```
AI    tool_calls=[call_0_0 list_projects, call_0_1 get_project]
Tool  call_0_0  ERROR [not_executed] la llamada no se completó por un fallo ... No tuvo efectos
Tool  call_0_1  ERROR [not_executed] ...
Human "dejalo, otra cosa"
```

Los cierres van **antes** del mensaje nuevo (`run_turn` los agrega en la misma entrada), así que se
cumple el contrato. `[not_executed]` es un hecho verificable: las lecturas y la preparación no
tienen efectos, y el runtime descarta antes el único caso que sí puede tenerlos.

**Última barrera:** `call_model` ejecuta `check_tool_protocol` antes de invocar al LLM y falla con
`HistoryInconsistent` si el historial viola el contrato. Solo es alcanzable invocando el grafo
directamente (fuera del modelo de seguridad): el proveedor nunca recibe ese historial.

## Locking obligatorio

LangGraph no serializa las ejecuciones de un mismo thread entre procesos. Lo comprobamos con
procesos reales: varios procesos reanudaban a la vez, bifurcaban los checkpoints y leían estados
ajenos. El `runtime` toma un lock de archivo del sistema operativo, no bloqueante, en cada
operación (`arl.agent.locks`).

El lock no es opcional:

- `lock_dir` es un argumento obligatorio de `build_graph`. Con `None` falla con `LockingRequired`.
- `run_turn`, `resume`, `continue_after_failure` y `turn_result` leen `lock_dir` de la
  configuración del grafo. Si falta (un grafo compilado aparte, o con la config sobrescrita),
  fallan con `LockingRequired` **antes** de leer o escribir el thread. No hay modo "sin lock".

```python
build_graph(model, gateway, saver)  # TypeError: falta lock_dir
build_graph(model, gateway, saver, lock_dir=None)  # LockingRequired
graph = build_graph(model, gateway, saver, lock_dir=data_dir / "locks")
await run_turn(graph.with_config({"configurable": {"lock_dir": None}}), ...)  # LockingRequired
```

Mientras otro proceso ejecuta el thread, `run_turn`, `resume` y `continue_after_failure` lanzan
`ThreadBusy` y `turn_result` devuelve `BUSY`. Si el proceso que tiene el lock muere, el sistema
operativo lo libera. **En Windows esa liberación es asíncrona:** durante unos milisegundos después
del kill se sigue recibiendo `ThreadBusy`. Es seguro (nunca hay dos ejecuciones) y el llamador
reintenta. Apareció de forma intermitente en
`test_lock_held_by_another_process_is_released_when_it_crashes`, que ahora espera la liberación con
un tope de 10 s.

**Limitación:** el lock es local a una máquina. Procesos en máquinas distintas, o sobre un sistema
de archivos de red, no se excluyen.

## Resultado incierto y reconciliación

Caso: el servidor confirmó la escritura, pero la respuesta se perdió (timeout o desconexión) o
llegó mal formada (fuera de contrato, vacía o de otra tarea).

1. El agente no tiene evidencia, así que no declara éxito. El thread queda en `WRITE_UNCERTAIN`
   y el historial no recibe ninguna respuesta para esa tool call.
2. La acción sellada sigue en el estado. `pending_action` expone `action_id`, `task_id`,
   `changes`, `expected_version` e `idempotency_key`.
3. **No se puede generar otra operación para el mismo intento.** `run_turn` está bloqueado, así
   que el LLM no puede volver a proponer la acción con otra tool call (que tendría otra clave).
   `continue_after_failure` reintenta solo la acción aprobada, con su clave.
4. **Reconciliación automática:** al continuar, el servidor encuentra la clave en su ledger y
   devuelve la respuesta original con `replayed: true`. Esa es la evidencia: recién entonces el
   resultado entra al historial y el turno se completa.
5. **Si el servidor sigue respondiendo mal**, el estado sigue en `WRITE_UNCERTAIN`. Se probó con 3
   continuaciones: la misma clave las 4 veces y una sola auditoría. La reconciliación manual
   consiste en consultar `audit_log` e `idempotency_keys` del servidor por esa clave. Si la clave
   está, la escritura ocurrió. Si no está, no ocurrió, y continuar la ejecutará una sola vez. El
   agente no tiene una herramienta para cerrar el estado a mano: es un riesgo pendiente.

## Logs sin secretos

Tres barreras. Ninguna depende de `from None`:

1. **No producir el texto.** El gateway clasifica el fallo dentro del `except` y lanza **fuera**
   de él, con un mensaje fijo. La excepción no tiene `__cause__` ni `__context__`: `from None`
   solo oculta el contexto al imprimir, pero el objeto lo seguiría guardando para cualquier
   formateador o reportero de errores. Tampoco se propaga el `ValidationError` de una respuesta
   mal formada, que incluye los valores recibidos.
2. **Diagnóstico sin el texto original.** Cada `ToolCallFailed` lleva `code`, `tool` y
   `error_type` (el nombre del tipo original, p. ej. `OSError`). El runtime registra en
   `arl.agent`: `ejecución detenida trace_id=… thread=… operation=update_task code=unavailable
   error_type=OSError status=write_uncertain`.
3. **Filtro de redacción** en el logger `arl.agent`: enmascara `password=`, `token=`, `api_key=`,
   `Authorization: Bearer …` y `sk-…` en el mensaje, los argumentos y el traceback. Es heurístico:
   un secreto sin forma reconocible (p. ej. una ruta) no se detecta. Por eso es la segunda
   barrera y no la primera.

`test_agent_log_safety.py` inyecta secretos simulados en 7 tipos de fallo y verifica `str`,
`repr`, `args`, atributos, traceback formateado, notas, `__cause__` y `__context__`. También
revisa los logs a nivel DEBUG (incluidos los reintentos de LangGraph), el `TurnResult` y el estado
persistido.

## Matriz de fallos y resultados

T = transitorio · D = definitivo · I = resultado incierto. Todos los escenarios terminan
verificando las invariantes del dominio: solo escrituras aprobadas, sin duplicados, cada auditoría
con su entrada de ledger y versión = 1 + modificaciones.

| # | Fallo | Clase | Antes de la etapa 4 | Ahora | Test |
|---|---|---|---|---|---|
| 1 | Desconexión, timeout o error opaco en una lectura (una vez) | T | Excepción cruda | Reintento con éxito | `test_transient_read_failure_is_retried` (×3) |
| 2 | Lectura caída de forma persistente | T | Excepción cruda | 3 intentos, `INTERRUPTED`, se puede continuar después | `test_persistent_read_failure_stops_cleanly_and_can_continue` |
| 3 | Fallo al preparar una escritura | T | Excepción cruda | `INTERRUPTED`; al continuar **vuelve a pedir aprobación** | `test_failure_while_preparing_a_write_never_skips_approval` |
| 4 | Timeout, desconexión o crash **antes** de que llegue la escritura | I | Excepción; reintento manual | Reintento con la misma clave: 1 escritura | `test_transient_write_failure_retries_with_the_same_key` (×3) |
| 5 | Timeout, desconexión o crash **después** del commit | I | Excepción; reintento manual | Reintento con la misma clave: replay, 1 escritura | ídem (×3) |
| 6 | Escritura sin confirmar de forma persistente | I | Excepción; **un turno nuevo descartaba la acción** | `WRITE_UNCERTAIN`; `run_turn` bloqueado; otro proceso recupera una vez | `test_unconfirmed_write_is_reported_as_uncertain_and_recovered_once` (×2) |
| 7 | `[forbidden]`, `[version_conflict]`, rechazo humano | D | Sin reintento | Sin reintento (1 llamada, o 0 si se rechazó) | `test_forbidden_is_not_retried`, `test_version_conflict_is_not_retried`, `test_rejected_action_is_never_sent` |
| 8 | Respuesta de lectura mal formada, vacía, de otro recurso o con error de protocolo | D | `KeyError` o se aceptaba | `INTERRUPTED` `[invalid_response]`, sin reintento | `test_invalid_read_responses_stop_without_retry` (×4) |
| 9 | Respuesta de escritura mal formada, de otra tarea, vacía o con error de protocolo | I | Se aceptaba cualquier cosa | `WRITE_UNCERTAIN`, sin reintento automático; reconciliación manual | `test_invalid_write_response_is_uncertain_and_not_retried` (×4) |
| 10 | Error de transporte con texto sensible | — | Llegaba a los logs vía `__cause__` | No aparece en logs, resultado ni estado | `test_transport_errors_do_not_leak_sensitive_text` |
| 11 | `turn_result` sobre un thread caído | — | `ValueError: 'running'` | `WRITE_UNCERTAIN` / `INTERRUPTED` | `test_state_of_a_thread_stopped_inside_execute_write_is_readable` |
| 12 | Fallo en un thread mientras otro sigue | — | — | El otro thread no se ve afectado | `test_failure_in_one_thread_does_not_affect_another` |
| 13 | **Timeout real** antes y después del commit (servidor stdio en otro proceso) | I | — | El reintento automático resuelve: 1 escritura | `test_real_timeout_around_the_write_is_resolved_by_retry` (×2) |
| 14 | **El proceso servidor muere** tras el commit | I | — | `WRITE_UNCERTAIN`; otro proceso recupera con replay | `test_real_server_crash_after_commit_is_uncertain_then_recovered` |
| 15 | **El proceso agente muere** (`os._exit`) entre el commit y el checkpoint | I | Probado en la etapa 3 | Igual; además el lock se libera | `test_process_killed_after_commit_before_checkpoint_resumes_once` |
| 16 | **4 procesos aprueban la misma acción a la vez** | — | Varios ejecutaban, estados `interrupted` falsos, `ValueError` intermitente | Exactamente 1 ejecuta; el resto recibe `ThreadBusy` o `NoPendingApproval` | `test_same_approval_from_concurrent_processes_runs_exactly_once` |
| 17 | 2 principales en procesos distintos sobre la misma versión | D | Probado solo en el servicio | 1 gana, 1 `[version_conflict]` | `test_competing_principals_in_separate_processes_have_one_winner` |
| 18 | Thread ejecutándose en otro proceso | — | Sin protección | `ThreadBusy` / `BUSY`; no revela threads ajenos | `test_agent_locks.py` |
| 19 | Reintentos acotados con backoff limitado | — | — | N intentos exactos; espera con tope | `test_retries_are_bounded_with_capped_backoff` |
| 20 | Turno nuevo tras una lectura interrumpida (2 tool calls; una nunca se ejecutó) | T | Tool calls sin respuesta antes del mensaje nuevo: un proveedor real lo rechazaría | Ambas cerradas con `[not_executed]` antes del mensaje; el modelo recibe un historial válido | `test_new_turn_after_an_interrupted_read_closes_every_pending_call` |
| 21 | Turno nuevo tras una preparación de escritura interrumpida | T | Ídem | Cerrada con `[not_executed]`; 0 escrituras; sin `pending_write` | `test_new_turn_after_an_interrupted_write_preparation_closes_it_without_writing` |
| 22 | Turno nuevo sobre una escritura incierta | I | Bloqueado, pero sin la clave visible | Bloqueado; historial intacto; acción y clave conservadas; continuar resuelve con la misma clave | `test_uncertain_write_is_never_closed_and_keeps_action_and_key` |
| 23 | Grafo invocado directamente con una tool call sin respuesta | — | El historial inválido llegaba al modelo | `HistoryInconsistent`; el modelo no se invoca | `test_the_model_is_never_called_with_an_unanswered_tool_call` |
| 24 | Escritura confirmada con respuesta mal formada, vacía, de otra tarea o perdida | I | `WRITE_UNCERTAIN` y replay al continuar (sin verificar `replayed`) | Sin éxito hasta tener evidencia: `replayed: true`, 1 auditoría, una sola clave | `test_committed_write_with_bad_or_lost_response_is_reconciled_by_replay` (×4) |
| 25 | Respuesta mal formada persistente | I | — | Sigue en `WRITE_UNCERTAIN` tras 3 continuaciones; misma clave, visible para reconciliar | `test_persistently_bad_response_stays_uncertain_without_new_operations` |
| 26 | Grafo persistente sin `lock_dir` (al construirlo o en cualquiera de las 4 funciones del runtime) | — | Se ejecutaba sin exclusión mutua | `LockingRequired` antes de tocar checkpoints o base | `test_building_a_persistent_graph_without_lock_dir_fails`, `test_every_runtime_entry_point_refuses_to_run_without_a_lock` (×4) |
| 27 | Proceso con el lock terminado con kill | — | — | El otro proceso recibe `ThreadBusy` mientras vive; el lock se libera tras el kill | `test_lock_held_by_another_process_is_released_when_it_crashes` |
| 28 | Secretos en errores de transporte, protocolo o validación | — | `__context__` conservaba el texto; sin tipo de error | Ninguna superficie expone el secreto; `code`, `tool`, `error_type` y `trace_id` disponibles | `test_gateway_failures_carry_diagnostics_but_no_secret` (×7), `test_failed_turn_logs_trace_operation_and_type_without_secrets` |

## Inyección de fallos

`arl.agent.faults.FaultyClient` envuelve el `mcp.Client` real y aplica un plan determinístico
`(tool, n-ésima llamada | "*") → Fault`. Lanza las **mismas excepciones del SDK** que aparecen en
fallos reales, así que lo que se ejercita es la clasificación real del gateway. Para fallos reales
entre procesos, `tests/integration/faulty_mcp_server.py` altera el servidor stdio (demora o muerte
alrededor del commit) y `tests/integration/agent_worker.py` ejecuta agentes concurrentes con una
barrera.

## Chequeo de mutaciones

```bash
uv run python scripts/mutation_check.py            # 34 mutantes (unos 8 minutos)
uv run python scripts/mutation_check.py --list     # qué regresión reintroduce cada uno
uv run python scripts/mutation_check.py -k gateway --all-failures
uv run python scripts/mutation_check.py --baseline-only
```

- **Seguro:** copia el repo a un directorio temporal, muta solo la copia y verifica que `arl` se
  importe desde ella. El árbol de trabajo nunca se modifica.
- **Línea base obligatoria:** si la suite sin mutar falla en la copia, aborta (exit 2). Un test
  roto haría pasar por "detectados" mutantes que en realidad no detecta.
- Si un fragmento ya no coincide con el código, aborta en lugar de reportar un falso "KILLED".
- Estas propiedades tienen tests propios (`tests/unit/test_mutation_check.py`), que ejecutan el
  script real sobre un repositorio mínimo: una línea base rota da exit 2 sin aplicar mutantes; un
  fragmento desactualizado da exit 2; un sobreviviente da exit 1; el árbol original queda idéntico
  (hash).
- Resultado actual: **34/34 detectados**.

## Riesgos pendientes (no cubiertos)

- **Fallos del LLM** (timeouts, rate limits, respuestas inválidas del proveedor): sin probar, porque
  el LLM sigue siendo simulado.
- **Reconexión automática.** Si el servidor stdio muere, la conexión del proceso queda inutilizable;
  la recuperación exige otro proceso u otra conexión. El gateway no se reconecta solo.
- **Fallo del LLM a mitad de turno.** Si `call_model` falla, el thread queda con el mensaje del
  usuario sin respuesta, y un turno nuevo agrega otro mensaje de usuario consecutivo. No viola el
  contrato de tool calling, pero no se probó contra un proveedor real.
- **Acceso directo al grafo sobre una escritura incierta.** Fuera del runtime, una entrada nueva
  llega a `begin_turn`, que limpia `pending_write`. `call_model` se niega a continuar
  (`HistoryInconsistent`), pero la acción ya no está en el estado. La clave sigue siendo derivable
  (`write_identity(thread_id, tool_call_id)`) y el servidor la tiene en su ledger.
- **Timeouts en memoria.** Con el transporte en memoria, una tool síncrona lenta no puede cortarse
  por timeout (verificado). Los timeouts reales solo se prueban con stdio.
- **Lock solo local:** funciona entre procesos de una misma máquina y sobre un sistema de
  archivos local. En Windows, la liberación tras un crash es asíncrona (`ThreadBusy` durante unos
  milisegundos).
- **Disco y SQLite:** no se simularon disco lleno, corrupción ni bloqueos prolongados de SQLite
  (más allá de `busy_timeout`).
- **Reconciliación de `WRITE_UNCERTAIN`** ante respuestas inválidas persistentes: es manual
  (consultar el ledger por la clave). No hay un comando para cerrar el estado; llegará con la CLI.
- **Logs:** todavía no hay observabilidad propia (etapa 5). La verificación cubre los logs de
  LangGraph, del SDK y de `arl.agent` a nivel DEBUG en los tests, no un pipeline de logs. La
  redacción es heurística.
- **`aiosqlite` a nivel DEBUG vuelca los checkpoints completos**: mensajes, resultados de tools y
  metadatos como la ruta de `lock_dir`. Lo observamos en los tests. No contiene secretos porque el
  estado no los tiene, pero el DEBUG de `aiosqlite` no debe activarse fuera del desarrollo.
