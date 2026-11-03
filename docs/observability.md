# Observabilidad y análisis de trazas (etapa 5)

> Verificado con datos sintéticos, un LLM simulado y procesos locales. La redacción de logs es
> heurística y **no garantiza confidencialidad**; la protección principal es no producir el texto.

## Tres fuentes, tres funciones

| Fuente | Dónde | Para qué | Garantía |
|---|---|---|---|
| **`audit_log`** (+ `idempotency_keys`) | `app.db`, en el servidor MCP | **Fuente de verdad** de qué se escribió, quién y con qué clave | Se confirma en la misma transacción que la escritura; append-only (triggers) |
| **Trazas** (`arl.trace/1`) | `<trace_dir>/thread-<hash>/<run_id>.jsonl`, en el agente | Reconstruir y medir una ejecución: cronología, reintentos, errores, duración, estado final | **Best-effort**: pueden faltar eventos; nunca cambian el resultado |
| **Logs** | stderr (`arl.agent` y dependencias) | Diagnóstico humano en el momento | Best-effort; redactados; sin el DEBUG de `aiosqlite` |

Los checkpoints de LangGraph son **estado**, no observabilidad: no se exportan ni se registran.

**Sin duplicar.** Las trazas no copian el contenido del `audit_log` (valores `before`/`after`):
lo referencian por `idempotency_key` y `audit_id`. Ante una discrepancia, manda el `audit_log`.
Por eso una escritura sin `write_result` en las trazas es **incierta**, nunca "fallida".

## Qué se registra y qué no

| Se registra | Nunca se registra |
|---|---|
| Ids: `thread_id`, `run_id`, `tool_call_id`, `action_id`, `idempotency_key`, `audit_id`, `project_id`, `task_id` | Texto de mensajes (usuario, modelo, tools) y prompts |
| Nombres de los campos que se cambian (`changed_fields: ["status"]`) | Valores de los cambios (títulos, estados nuevos) |
| Códigos (`unavailable`, `version_conflict`), `error_type` (nombre del tipo), duraciones, intentos, conteos | Textos de error, tracebacks, rutas, checkpoints |
| Decisión del humano (`approve`/`reject`/`invalid`) y si hubo motivo (`with_reason`) | El texto del motivo de rechazo |

Cómo se garantiza:
- Cada evento se valida contra su modelo Pydantic con `extra="forbid"`, así que el esquema
  funciona como lista blanca.
- Los ids que vienen del LLM o del caller pasan por `safe_id`: si no tienen forma de id, se
  reemplazan por un hash.
- Un evento que no cumple el esquema se descarta, sin lanzar.

## Esquema `arl.trace/1`

Cada línea es un sobre:

```json
{"v": 1, "ts": "2026-11-03T03:30:58.946994+00:00", "seq": 3, "event": "mcp_call",
 "run_id": "304f81ab40db47b6", "thread_id": "demo-uncertain", "principal_id": "alice",
 "data": {"tool": "update_task", "outcome": "unavailable", "duration_ms": 20.549,
          "node": "execute_write", "tool_call_id": "call_1_0", "attempt": 1,
          "code": "unavailable", "error_type": "MCPError", "task_id": "apo-3",
          "idempotency_key": "arl-97c07d04eaee3e4dd6117edf3b5d35e4f83fa5f5"}}
```

Los campos `None` se omiten. Hay un JSON Schema completo, generado desde los modelos:
`uv run python -m arl.tracing schema`. **Versionado:** los lectores validan de forma estricta, así
que todo cambio de campos sube `v` (y `arl.trace/N`).

| Evento | Quién lo emite | Datos clave |
|---|---|---|
| `run_started` | runtime | `operation`: `run_turn` / `resume` / `continue_after_failure` |
| `recovery_started` | `continue_after_failure` | `prior_status` (`interrupted` / `write_uncertain`), `resume_node`, `action_id` |
| `history_closed` | `run_turn` | tool calls cerradas con `[not_executed]` |
| `model_call` | `call_model` | `model_turn`, duración, tool calls pedidas (id + nombre), `outcome` |
| `tool_call_routed` | `dispatch` (gate) | decisión: `execute` / `requires_approval` / `invalid_tool_call` / `tool_limit`; ids y `changed_fields` (lo **solicitado**) |
| `approval_requested` | `prepare_write` | acción sellada: `action_id`, `args_hash`, versión esperada, rol (lo que se **aprueba**) |
| `approval_decided` | `await_approval` | `approve` / `reject` / `invalid` (+ `invalid_reason`) |
| `mcp_call` | gateway | tool, `outcome` (`ok` / `domain_error` / `unavailable` / `invalid_response`), `code`, `error_type`, duración, `node` + `attempt`, `replayed` |
| `write_result` | `execute_write` | lo **ejecutado** según el servidor: `outcome`, `replayed`, `version`, `audit_id` |
| `write_uncertain` | runtime | escritura aprobada sin confirmar: `action_id`, `idempotency_key`, fallo |
| `run_finished` | runtime | `status` final, duración, `refusal` / `failure` (`code`, `operation`, `error_type`), conteos |

`run_finished.status` puede ser `completed`, `awaiting_approval`, `tool_limit`, `turn_limit`,
`interrupted`, `write_uncertain`, `busy` (otro proceso tiene el thread), `refused` (rechazo del
runtime: `ThreadNeedsRecovery`, `PrincipalMismatch`, …) o `error` (excepción no prevista). Si no
hay `run_finished`, la ejecución quedó **incompleta**: el proceso murió a mitad de camino.

**Correlación:**
- `thread_id` agrupa ejecuciones.
- `run_id` (también en `TurnResult.run_id`, en el log y en los metadatos del checkpoint) agrupa
  los eventos de una ejecución, y `seq` los ordena.
- `tool_call_id` liga lo solicitado con la acción, sus intentos y su resultado.
- `action_id` liga la solicitud de aprobación con la decisión.
- `idempotency_key` y `audit_id` ligan con el servidor.
- `attempt` cuenta los intentos de un nodo dentro de la ejecución (los reintentos de LangGraph).

## Best-effort y garantías

- **`emit` nunca lanza.** Si emitir lanzara después de un commit, LangGraph reintentaría
  `execute_write`, es decir, enviaría una escritura nueva. Un fallo al validar, serializar o
  escribir descarta el evento y deja un aviso en `arl.agent` (uno por ejecución, con
  `error_type`). Test: `test_trace_failure_right_after_the_commit_does_not_retry_the_write`.
- **Un evento descartado deja un hueco en `seq`.** El número se asigna antes de validar, así que
  el lector ve que falta algo en vez de recibir una secuencia que parece completa.
- **Un archivo por ejecución.** Lo escribe un solo proceso, así que no hay escrituras
  concurrentes en el mismo archivo (en Windows, el modo append no es atómico entre procesos).
- Cada evento se agrega y se cierra. Un crash pierde a lo sumo el evento en curso; una línea
  cortada se cuenta como `invalid_lines` y no impide leer el resto.
- Lo que **sí** es garantía está fuera de las trazas: el `audit_log` y el estado de los
  checkpoints (`turn_result`).

## Consultar trazas

```bash
uv run python -m arl.tracing runs   [--dir D]                      # ejecuciones recientes
uv run python -m arl.tracing show   <run_id|archivo.jsonl|reporte.json> [--dir D] [--json]
uv run python -m arl.tracing thread <thread_id> [--dir D] [--json] # todas las del thread
uv run python -m arl.tracing schema
```

- `--dir` toma por defecto `<ARL_DATA_DIR>/traces`.
- La consulta es de solo lectura (verificado por hash en los tests) y no importa código que
  pueda escribir: ni servicios, ni servidor MCP, ni runtime, ni SQLite
  (`test_query_cli_never_loads_code_that_can_write_business_data`).
- `--json` exporta un reporte `arl.run-report/2`. Incluye: `final_status`, `complete`, `runs`,
  `counts` (con `retries`), `retries`, `errors`, `writes` (solicitada → aprobada → ejecutada,
  con `state`, `basis` y `note`), `integrity`, `timeline` y `source` (los eventos validados).

Estados de una escritura en el reporte:
- `executed`: el servidor confirmó, puede ser con replay.
- `refused_by_server`: `[forbidden]`, `[version_conflict]`, etc.; no se aplicó.
- `uncertain`: aprobada o enviada al servidor, sin evidencia del resultado.
- `rejected`: el humano la rechazó.
- `awaiting_approval` o `not_prepared`: solo se deducen de eventos que **faltan**, por eso
  llevan `basis: "absence"` y una nota que lo dice. Los demás llevan `basis: "evidence"`.

## Integridad de las trazas

Los archivos de trazas pueden llegar duplicados, desordenados, cortados, de otra versión o
editados. La consulta los trata como datos no confiables:

| Situación | Qué hace la consulta |
|---|---|
| Línea que no es UTF-8, no es JSON o no cumple el esquema (incluye `ts` sin zona horaria o `seq < 1`) | La descarta y la cuenta en `integrity.invalid_lines` |
| Sobre con otra versión (`v != 1`) | La descarta y la cuenta aparte en `unsupported_version_lines` |
| Mismo `(run_id, seq)` dos veces, idéntico | Usa uno y cuenta `duplicate_events` |
| Mismo `(run_id, seq)` con contenido distinto | Descarta **las dos** versiones (`conflicting_events`): ninguna es evidencia |
| Líneas desordenadas | Ordena por `seq`; el reporte es el mismo |
| Huecos de `seq` | Los lista en `seq_gaps` |
| Eventos de varios threads en un archivo | Los lista en `thread_ids` |
| Ejecución sin `run_finished` | `incomplete_runs`; estado **desconocido**, no un error |
| Reporte `.json` editado | Lo reconstruye desde `source` (cada evento se valida de nuevo) y marca `report_mismatch`; nunca muestra lo calculado en el archivo |

`integrity.ok` es `false` si hay cualquiera de esos problemas, salvo ejecuciones incompletas, que
se informan aparte. La línea de estado lo indica (`[integridad dudosa: …]`).

Reglas de interpretación:
- La ausencia de un evento nunca prueba nada. Si hay envíos de `update_task` pero falta la
  decisión, la escritura es `uncertain`, no `not_prepared`.
- La fuente autoritativa de una escritura confirmada sigue siendo el `audit_log`, por
  `idempotency_key`.

Tests: `tests/integration/test_tracing_integrity.py`.

Trazas de ejemplo con datos sintéticos: `uv run python scripts/sample_trace.py --out <dir>`
(fuera de OneDrive).

## Logging

- `arl.tracing.logging_setup.configure_logging(level)` instala un único handler a stderr, con el
  filtro de redacción aplicado a todos los registros, también los de dependencias. El servidor
  MCP lo usa con `ARL_LOG_LEVEL` (por defecto `WARNING`).
- **`aiosqlite` nunca baja de INFO**, aunque se pida DEBUG. Lo verificamos con un canario en un
  turno real, por transporte en memoria y por stdio: fue el único logger que volcó contenido (32
  registros con los checkpoints completos). El SDK MCP y LangGraph no lo hicieron en las
  versiones fijadas. `allow_unsafe_debug=True` lo habilita a conciencia, solo para desarrollo.
- Los reintentos de LangGraph se registran en INFO con traceback. Es seguro porque las
  excepciones del gateway no conservan el texto original (etapa 4).

## Skill `analyze-run`

`.claude/skills/analyze-run/SKILL.md`: `/analyze-run <run_id | archivo.jsonl | reporte.json>`.
- Separa la evidencia (con citas `run #seq`) de las hipótesis.
- Nunca presenta una escritura incierta como fallida, y trata lo deducido por ausencia de
  eventos como hipótesis.
- Tests: `tests/devtools/test_analyze_run_skill.py`, que cubren la estructura, la seguridad y
  los comandos sobre trazas reales. La aceptación en vivo con `claude -p` es opcional:
  `ARL_LIVE_SKILL_TEST=1`.

### Modelo de confianza

El texto de `SKILL.md` es una instrucción para el modelo, **no una barrera de seguridad**. Las
barreras son estas, de afuera hacia adentro:

| Riesgo | Barrera | Verificación |
|---|---|---|
| Un argumento se ejecuta como comando | La Skill no usa `` !`…` ``: `$ARGUMENTS` es texto, nunca pasa por el shell antes de los permisos | `test_skill_only_grants_read_only_tools_…` |
| Comandos encadenados o redirecciones (`;`, `&&`, `\|`, `>`, `$(…)`) | `allowed-tools` preaprueba solo `show`, `thread` y `runs` de la CLI. Claude Code evalúa cada parte de un comando compuesto: el resto pide permiso, y en `claude -p` se deniega | Prueba en vivo (abajo) |
| Leer archivos crudos o secretos | No hay `Read` en `allowed-tools`. La CLI solo muestra lo que pasa el esquema (ids `[A-Za-z0-9_.:-]`, códigos, números); de un archivo ajeno solo informa cuántas líneas descartó | `test_malformed_report_files_…`, `test_report_without_valid_events_…` |
| Ruta o id manipulado | `run_id` exige `^[0-9a-f]{16}$` (sin `..`, comodines ni separadores); `thread_id` se usa solo como hash; archivos de más de 64 MiB se rechazan | `test_hostile_run_ids_never_resolve_to_a_file` |
| Reporte falsificado con instrucciones ("ignorá las reglas…") | Se reconstruye desde sus eventos validados; el texto libre del archivo nunca llega a la salida | `test_exported_report_is_rebuilt_from_…` |
| Operación de negocio | La CLI no importa servicios, servidor MCP, runtime ni SQLite. Fuera de la Skill, `update_task` está en `ask` (`.claude/settings.json`) y el servidor de `.mcp.json` usa `dev-viewer`, que la autorización del servidor rechaza para escribir | `test_query_cli_never_loads_…` y tests de autorización |

Prueba en vivo de permisos (`claude -p` en modo `default`, con los mismos permisos que la
Skill): se ejecutó `runs --dir …` y se denegaron `…; touch`, `… && touch`, `$(touch …)`,
`> archivo` y `| tee archivo`, con rutas dentro y fuera del proyecto. Ningún archivo canario se
creó.

Prueba en vivo de la Skill (`claude -p`, opcional con `ARL_LIVE_SKILL_TEST=1`):
- Un argumento con inyección (`x'; touch …; echo '`) fue rechazado sin ejecutar nada.
- Un reporte `.json` editado, con una instrucción en `thread_id`, se mostró reconstruido, con
  `report_mismatch`, y la instrucción se informó como hallazgo.
- En los dos casos, las trazas, checkpoints y bases quedaron idénticos (hash) y no se creó el
  canario.

**Límites:**
- `allowed-tools` **preaprueba** herramientas; no prohíbe las demás. Si el usuario aprueba
  manualmente otra acción, o corre en `bypassPermissions`, la Skill no lo impide.
- La preaprobación cubre la herramienta Bash. En Windows, el modelo a veces eligió PowerShell, y
  en `claude -p` una vez invocó la Skill por la tool `Skill` en vez de expandirla: en ambos casos
  la CLI pidió aprobación. La versión anterior de la Skill, sin las instrucciones actuales,
  terminó leyendo el `.json` crudo con `Read` (la instrucción inyectada igual se trató como
  dato). Ahora la Skill indica usar Bash y detenerse si un comando pide aprobación; con esa
  instrucción se detuvo. Es una instrucción, no una barrera: `Read` es de solo lectura y Claude
  Code lo permite por defecto.
- La validación de la entrada que pide `SKILL.md` es una defensa más, no una garantía: un modelo
  puede equivocarse. Por eso las barreras de arriba no dependen de ella.
- Los ids válidos pueden contener palabras (`ignorar.reglas`). El esquema impide texto libre,
  espacios y símbolos, pero no que un id "diga" algo. La Skill trata las trazas como datos.
- `uv run` puede sincronizar el entorno virtual (`.venv`) antes de ejecutar. No toca trazas,
  checkpoints ni bases.
- La CLI lee cualquier `.jsonl` o `.json` que se le indique. No revela su contenido salvo eventos
  válidos, pero sí confirma si el archivo existe.

## Alcance del turno (`arl.trace/2`, etapa 6A.1)

`run_started.scope`, la decisión `out_of_scope` con `scope_reason` y el evento
`scope_reservation` (estado de cada permiso). El reporte pasa a `arl.run-report/3`, con la
sección `scope`. Los lectores siguen aceptando las trazas v1. Detalle en [scope.md](scope.md).

## Limitaciones y riesgos pendientes

- **Best-effort real:** un disco lleno o un directorio inaccesible deja ejecuciones sin trazas.
  Solo queda el aviso en el log y el `audit_log` sigue siendo la verdad.
- **`attempt` es por ejecución.** Una recuperación posterior empieza de nuevo en 1. El reporte
  del thread suma los intentos de todas las ejecuciones.
- **Un crash pierde el evento en curso.** Por ejemplo, si el proceso muere entre el commit y
  `write_result`, la escritura figura como `uncertain` hasta la recuperación (probado entre
  procesos).
- **Códigos de dominio por texto.** El SDK MCP v2 entrega un `ToolError` como texto
  (`"Error executing tool <tool>: [codigo] …"`). `arl.agent.gateway.domain_code` acepta
  solo ese formato y códigos de `DomainError`; cualquier otro queda como `unknown`. Hasta la
  etapa 6A, todos quedaban como `unknown` (defecto D1, ver [evals.md](evals.md)).
- **Sin retención ni rotación:** los archivos crecen sin límite; borrarlos es manual.
- **Sin agregación entre máquinas** ni métricas en series de tiempo. No hay OpenTelemetry ni
  LangSmith (fuera del MVP).
- **La redacción de logs es heurística:** un secreto sin forma reconocible no se detecta.
- **Los ids y metadatos de las trazas son datos a proteger con datos reales.** `principal_id` y
  `thread_id` van en claro. `project_id`, `task_id`, `idempotency_key`, `audit_id`, nombres de
  campos, tiempos y conteos revelan quién trabajó sobre qué y cuándo. Hoy son identidades simuladas
  y datos sintéticos. Con datos reales, las trazas necesitan permisos de acceso, retención y
  probablemente seudonimización. Una `idempotency_key` no autoriza nada por sí sola: el servidor
  la liga al principal y al hash de los parámetros. Aun así, es un identificador interno que no
  debería exponerse fuera del equipo.
- **El LLM sigue simulado:** con un proveedor real habrá que medir sus errores y latencias, y no
  registrar sus respuestas.
