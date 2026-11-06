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
uv run python scripts/mutation_check.py --check-fragments   # catálogo: sin pytest ni mutaciones
uv run python scripts/mutation_check.py --list --group P1   # qué regresión y qué tests
uv run python scripts/mutation_check.py --group P1 --timeout 120 --results <afuera>.jsonl
uv run python scripts/mutation_check.py --names-file <nombres>.txt --list    # selección exacta
uv run python scripts/mutation_check.py --group historical --results <afuera>.jsonl
uv run python scripts/mutation_check.py -k gateway --all-failures
uv run python scripts/mutation_check.py --baseline-only
```

- **Seguro:** copia el repo a un directorio temporal, muta solo la copia y verifica que `arl` se
  importe desde ella. El árbol de trabajo nunca se modifica. Cada mutación se revierte en un
  `finally` (también ante una excepción o Ctrl+C) y se verifica por hash; si no se puede
  restaurar, aborta.
- **La copia no lleva** secretos (`.env`, `.env.*` salvo `.env.example`, `*.pem`, `*.key`), bases
  SQLite (`*.db`, `*.db-*`, `*.sqlite`, `*.sqlite3`), reportes (`live-reports`) ni
  transcripciones (`live-transcripts`, `*.jsonl`). pytest no hereda variables con `API_KEY`,
  `SECRET`, `TOKEN` o `PASSWORD` en el nombre (los tests usan canarios propios).
- **Línea base obligatoria:** si los tests sin mutar fallan o superan `--baseline-timeout` en la
  copia, aborta (exit 2). La línea base corre la unión de los tests de los mutantes elegidos.
- **Catálogo verificado antes de copiar:** fragmento único, mutante que compila, nombres únicos y
  tests focalizados dentro de `TESTS`. Si algo falla, aborta (exit 2) en lugar de reportar un
  falso "KILLED". El chequeo de compilación se agregó en la etapa 5, después de que un mutante con
  un `SyntaxError` apareciera como "detectado".
- **Estados (6B.11), separados:** `KILLED` (pytest falla con un test identificado; también un
  error de colección causado por el mutante), `SURVIVED` (pytest pasa), `TIMEOUT` (superó
  `--timeout`; se mata el árbol de procesos de pytest, con `taskkill /T` en Windows o el grupo de
  procesos en POSIX) y `ERROR` (pytest terminó sin un test fallido identificado: error interno, de
  uso, sin tests). **Solo `KILLED` cuenta como detectado**; los otros tres dan exit 1 y quedan
  para revisar.
- **Resultados incrementales** (`--results`, siempre fuera del repositorio): una línea JSONL por
  evento, escrita y vaciada al terminar cada mutante: `run` (commit, `dirty`, versión de Python,
  selección, `TESTS`, tests de la línea base, `-x`, timeouts), `baseline`, `mutant` (nombre, grupo,
  archivo, estado, código de salida, duración, tests corridos y tests que lo detectaron) y
  `summary`. Una corrida cortada conserva lo hecho.
- **Tests focalizados:** un mutante puede declarar sus tests (`archivo::test`); si no, corre la
  lista `TESTS` completa, como los históricos.
- **Selección exacta** (`--names-file`): un nombre exacto del catálogo por línea (las vacías y
  las que empiezan con `#` se ignoran), nunca posiciones ni subcadenas; solo del grupo
  `historical`; aborta (exit 2) ante nombres desconocidos o repetidos, una selección vacía o
  combinada con `-k` o `--group`. El JSONL guarda el archivo, su SHA-256 y los nombres.
- Estas propiedades tienen tests propios (`tests/unit/test_mutation_check.py`), que ejecutan el
  script real sobre un repositorio mínimo (sin mutantes del proyecto): línea base rota, fragmento
  desactualizado y mutante que no compila dan exit 2; un sobreviviente da exit 1; un timeout da
  `TIMEOUT` (nunca `KILLED`) y no deja vivo ni a un nieto de pytest; resultados JSONL;
  `--results` dentro del repo rechazado; tests focalizados en lugar de la lista completa y fuera
  de `TESTS` rechazados; restauración ante una excepción; exclusiones de la copia; entorno sin
  credenciales; selección por nombres (válida e inválidas); y el catálogo real (110 históricos +
  16 P1, fragmentos únicos y compilables).

### Catálogo

| Grupo | Mutantes | Tests | Estado |
|---|---|---|---|
| `historical` | 110: 34 de etapas 3–4, 13 de observabilidad e integridad de trazas (incluido el parser de D1), 9 del evaluador, 9 del alcance del turno, 2 de la identidad de tool calls (R1), 39 de la evaluación con LLM real y 4 de la veracidad | `TESTS` completa (48 archivos desde 6B.11) | **22/22 detectados** en la selección de riesgo (ver abajo); **los otros 88, no revalidados** sobre el código actual |
| `P1` (6B.11) | 16: autorización de dominio (3), concurrencia (1), idempotencia (2), `RequestScope` por `resource`, `field` y `value` (3), respuesta visible autoritativa (2), redacción de la transcripción (1) y atribución propia y externa (4) | focalizados (18 tests en total) | **16/16 detectados** (`a057c33`) |

Mutantes P1 y sus tests focalizados (`--list --group P1` los muestra):

| Mutante | Defecto que reintroduce | Tests |
|---|---|---|
| un viewer puede modificar tareas (política) | carol (viewer) cerraría apo-2 | `test_policy.py` (matriz) y `test_viewer_cannot_update` (efecto en el servicio) |
| un no miembro lee el proyecto como viewer | dave leería Apollo | `test_foreign_and_missing_projects_are_indistinguishable` |
| `update_task` no verifica el rol de escritura | viewer escribe | `test_viewer_cannot_update` |
| sin control de versión ni compare-and-set | escritura con versión vieja (L13 sin conflicto) | `test_stale_version_is_rejected` y `test_concurrent_writers_with_same_version_have_one_winner` |
| un reintento no devuelve la respuesta guardada | el reintento de una escritura aplicada da conflicto | `test_retry_with_same_key_and_params_does_not_duplicate` |
| misma clave con otros parámetros hace replay | otra acción recibe la respuesta de la primera | `test_same_key_with_different_params_conflicts` |
| `covers` no verifica la tarea | L12: apo-4 pasaría el alcance | `test_grant_allows_only_its_tasks_fields_and_values` y el guion A de L12 |
| `covers` no verifica los campos | `null_padding` (desasignar) pasaría | ídem y el guion E de L12 |
| `covers` no verifica el valor | otro estado pasaría | `test_grant_allows_only_its_tasks_fields_and_values` y `test_unassigning_is_a_value_that_must_be_allowed` |
| con escrituras se muestra el texto del modelo | el usuario vería una afirmación falsa | `test_an_executed_write_is_reported_from_the_server_confirmation` y `test_user_reply_never_claims_the_write_the_server_refused` |
| `version_conflict` se informa como ejecutada | L13 diría "se aplicó" | `test_l13_keeps_bobs_change_and_never_applies_alices` y `test_bobs_change_is_never_attributed_to_the_agent` |
| la transcripción no redacta secretos | un `gsk_…` quedaría escrito | `test_transcript_never_copies_reasoning_headers_or_credentials` |
| `_own_audit` cuenta escrituras de otros actores | la escritura de bob sería del agente | `test_bobs_change_is_never_attributed_to_the_agent` y `test_l13_keeps_bobs_change_and_never_applies_alices` |
| `writes_executed` suma filas de otros actores | métrica de seguridad inflada | `test_bobs_concurrent_change_is_recorded_as_an_external_write` |
| `writes.effective` cuenta el ledger de otros actores | escritura efectiva inexistente | `test_bobs_ledger_entry_is_not_an_effective_write_of_the_agent` |
| toda fila de `audit_log` se marca como propia | `own: true` para bob | `test_bobs_concurrent_change_is_recorded_as_an_external_write` |

Dos tests en un mutante: uno prueba la regla en su capa (política, `covers`, servicio) y el otro
el efecto de punta a punta (agente o reporte), para que un cambio de redacción de un test no deje
pasar el defecto. El control de versión se muta en sus dos capas a la vez (chequeo explícito y
compare-and-set del repositorio): cada una sola es un mutante equivalente, porque la otra frena
la escritura vieja.

### Resultados sobre el código actual (6B.11, 2026-11-05)

Cada batería se corrió **una sola vez**, en una copia temporal, con restauración por hash; el
repositorio quedó sin cambios y sin procesos hijos vivos. Los JSONL y el log quedan en
`ARL_DATA_DIR/mutation/` (fuera del repo); no se versionan.

| Batería | Commit | Mutantes | Resultado | Duración | Evidencia |
|---|---|---|---|---|---|
| `P1` (`--group P1`, tests focalizados, `--timeout 120`) | `a057c33`, `dirty: false` | 16 | **16 KILLED**, 0 SURVIVED, 0 TIMEOUT, 0 ERROR | 72.4 s (línea base 8.4 s; 2.0–5.4 s por mutante) | `p1-1.jsonl` |
| Históricos de riesgo (`--names-file`, `TESTS` completa, `--timeout 900`) | `a057c33` + `--names-file` sin commit (`dirty: true`; los archivos mutados no cambiaron) | 22 | **22 KILLED**, 0 SURVIVED, 0 TIMEOUT, 0 ERROR | 686.2 s (línea base 181.9 s; 4.1–86.4 s por mutante) | `historical-core-22-1.jsonl`, `historical-core-22.txt`, `historical-core-22-1.log` |

Todas las detecciones fueron por tests fallidos identificados (código de salida 1); no hubo
errores de colección. Con `-x` se registra el primer test que falla, no todos los que lo
detectarían.

La selección de riesgo son los 22 históricos de `graph.py`, `runtime.py` y `state.py` (archivos
modificados desde la última corrida completa) que protegen identidad (6), aprobación ligada a la
acción (8), `RequestScope` en el grafo (4) y la ejecución de escrituras (4). Tests que los
detectaron:

| Mutante | Test |
|---|---|
| `runtime: _check_owner no verifica` | `test_thread_cannot_be_resumed_by_another_principal` |
| `runtime.resume: no compara action_id/args_hash` | `test_approval_for_another_action_is_refused` |
| `runtime.turn_result: sin chequeo de dueño` | `test_thread_cannot_be_resumed_by_another_principal` |
| `runtime.continue_after_failure: sin chequeo de dueño` | `test_other_principal_cannot_read_or_alter_a_paused_thread` |
| `graph.check_principal desactivado` | `test_thread_cannot_be_continued_by_another_principal` |
| `await_approval: no verifica el principal del proceso` | `test_thread_cannot_be_resumed_by_another_principal` |
| `await_approval: no verifica que la decisión sea de la acción` | `test_invalid_decision_sent_directly_to_graph_does_not_poison_the_thread` |
| `await_approval: lanza ante decisión inválida (envenena el thread)` | `test_invalid_decision_sent_directly_to_graph_does_not_poison_the_thread` |
| `execute_write: no exige aprobación válida` | `test_execute_write_refuses_an_approval_injected_into_the_state` |
| `reducer de principal lanza (envenena el thread)` | `test_principal_of_a_thread_cannot_change` |
| `reducer de principal acepta el nuevo principal` | `test_principal_of_a_thread_cannot_change` |
| `PendingWrite: sello sin verificar` | `test_pending_write_seal_detects_any_tampering` |
| `ApprovalDecision acepta campos extra` | `test_approval_decision_only_decides` |
| `idempotency_key ignora la tool call` | `test_two_approved_writes_in_one_thread_get_distinct_keys` |
| `runtime: run_turn descarta una escritura incierta` | `test_new_message_cannot_discard_an_unconfirmed_write` |
| `runtime: turn_result no informa WRITE_UNCERTAIN` | `test_state_of_a_thread_stopped_inside_execute_write_is_readable` |
| `continue_after_failure: acepta threads esperando aprobación` | `test_continue_requires_an_interrupted_run` |
| `WRITE_UNCERTAIN no expone la clave para reconciliar` | `test_unconfirmed_write_is_reported_as_uncertain_and_recovered_once` |
| `alcance: dispatch no verifica el alcance (F1 vuelve)` | `test_with_the_defense_out_of_scope_writes_are_blocked_before_approval` |
| `alcance: prepare_write reserva sin volver a verificar` | `test_two_actions_cannot_hold_a_single_use_grant_at_the_same_time` |
| `alcance: execute_write no exige la reserva` | `test_execute_write_refuses_an_approved_action_without_its_reservation` |
| `alcance: el rechazo humano consume el permiso` | `test_rejected_proposal_releases_its_reservation` |

**No revalidados en el código actual:** los otros 88 históricos (tolerancia a fallos y
reintentos, locks, trazas y observabilidad, R1, evaluador, veracidad, `scope.py`,
`tool_registry.py` y los 39 de la evaluación con LLM real). Su último resultado es el de abajo,
sobre otro código. 38 de 126 mutantes se ejecutaron sobre el código actual; los 126 no.

### Resultados históricos (no son el estado actual)

Corridas sobre el código de su momento; no son una afirmación sobre el código actual:

- 2026-11-04: 88/88 detectados (corrida completa, ~78 min, unos 53 s por mutante).
- 2026-11-05 (antes de 6B.7): los 22 agregados después, con `-k "live: "`: 39/39 (el del
  presupuesto de `reject-write` sobrevivió en la primera corrida y se detecta tras agregar el
  test del peor caso).

Costo estimado de los 88 no revalidados con la lista ampliada (línea base de unos 180 s): entre
1.5 y 3 h; peor caso, unas 5 h.

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
- **Logs:** la verificación cubre los logs de LangGraph, del SDK y de `arl.agent` a nivel DEBUG
  en los tests, no un pipeline de logs. La redacción es heurística. Las trazas de la etapa 5
  están en [observability.md](observability.md).
- **`aiosqlite` a nivel DEBUG vuelca los checkpoints completos**: mensajes, resultados de tools y
  metadatos como la ruta de `lock_dir`. Lo observamos en los tests. No contiene secretos porque el
  estado no los tiene. **Resuelto en la etapa 5:** `configure_logging` deja `aiosqlite` en INFO
  como mínimo (ver [observability.md](observability.md#logging)).
