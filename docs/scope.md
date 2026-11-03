# Alcance del turno (`RequestScope`, etapa 6A.1)

> Corrige el hallazgo F1 de la etapa 6A. Datos sintéticos, modelo y humano simulados, identidad
> simulada: el alcance **no** es autenticación.

## Problema

Antes de esta etapa, la única barrera contra una escritura que el usuario no pidió era el humano.
Por ejemplo, una escritura inducida por una instrucción maliciosa escrita en la descripción de una
tarea. Si el humano la aprobaba sin revisar, se ejecutaba: la acción estaba aprobada, el usuario
tenía permiso de rol y se escribía exactamente lo aprobado.

## Diseño

Cada control lo decide un actor distinto:

| Control | Lo decide | Dónde |
|---|---|---|
| Qué escrituras se autorizaron para este turno | La **aplicación**, desde una declaración explícita del usuario | `run_turn(..., scope=RequestScope(...))` |
| Si el usuario puede escribir en ese proyecto | El **servidor**, por rol | `ProjectService` (sin cambios) |
| Si esta acción exacta se ejecuta | El **humano** | `interrupt()` y `resume` |

Ninguno reemplaza a los otros. El alcance nunca da un permiso que el rol no tenga: el caso
`holdout-viewer-scope-does-not-grant-role` lo verifica.

```python
from arl.agent.scope import RequestScope, allow_update

scope = RequestScope.writes(allow_update("apo-3", status=["done"]))  # una escritura, un valor
RequestScope.read_only()  # sin escrituras
allow_update("apo-3", "apo-2", status=None, max_uses=2)  # None = cualquier valor
RequestScope.writes(grant_a, grant_b, max_writes=1)  # tope del turno
```

- Un `WriteGrant` define la tool, las tareas, los campos con sus valores permitidos y `max_uses`.
- `max_writes` limita el turno. Por defecto es la suma de los `max_uses`.
- Desasignar (`assignee_id=None`) es un valor más: hay que permitirlo explícitamente.
- Los modelos son inmutables y rechazan campos desconocidos (`extra=forbid`).

**Origen del alcance.** Lo declara la aplicación; nunca el LLM, una tool response ni el texto del
pedido. En esta etapa, la aplicación tiene que construirlo explícitamente: no existe una CLI que lo
derive del lenguaje natural, ni ninguna clasificación de intención.

**Vínculo y protección.**
- `run_turn` liga el alcance al `thread_id` y al principal del proceso (`BoundScope`) y lo guarda
  en el checkpoint (`request_scope`).
- Ningún nodo lo modifica. `begin_turn`, `dispatch`, `prepare_write` y `execute_write` verifican
  el sello y el vínculo cada vez.
- `resume` y `continue_after_failure` no aceptan otro alcance.

**El sello no es autenticación.** `scope_hash` (sha256) detecta cambios accidentales o hechos
desde el LLM. Quien pueda escribir en `checkpoints.db` puede recalcularlo. El modelo de amenaza
es el mismo que el de `PendingWrite`.

## Dónde se controla

| Nodo | Qué hace |
|---|---|
| `dispatch` | Bloquea la escritura fuera de alcance **antes** de cualquier lectura o aprobación: `ToolMessage "ERROR [out_of_scope] (<motivo>) …"` |
| `prepare_write` | Vuelve a verificar y **reserva** el permiso para el `action_id` |
| `execute_write` | Exige la reserva con el mismo `args_hash`. Si falta, lanza `ScopeViolation` y no envía nada |

Los motivos son códigos fijos y nunca incluyen los valores pedidos: `read_only`, `resource`,
`field`, `value`, `exhausted`, `max_writes`.

## Estado de cada reserva

| Evento | Estado | ¿Ocupa el permiso? |
|---|---|---|
| Se preparó la acción | `reserved` | Sí |
| El servidor confirmó (`updated`, `unchanged`, replay) | `consumed` | Sí |
| `WRITE_UNCERTAIN` | sigue `reserved` | Sí, hasta reconciliar |
| Rechazo humano (nunca se envió) | `released` | No |
| `forbidden`, `not_found`, `invalid_change`, `idempotency_conflict` o un código desconocido | `refused` | Sí. Se cierra sin reintento |
| `version_conflict` | `invalidated` | Sí. Una propuesta nueva necesita otro turno con otro alcance |

- Siempre se cumple: reservadas + consumidas + rechazadas + invalidadas ≤ `max_uses` (por permiso)
  y ≤ `max_writes` (por turno).
- Un **reintento técnico** es la misma acción: el reintento de LangGraph o `continue_after_failure`,
  con el mismo `action_id`, la misma `idempotency_key` y los mismos argumentos. Usa su propia
  reserva.
- Una **propuesta nueva** (otro `tool_call_id`) necesita cupo libre.
- Un crash después del commit deja la reserva tomada. `run_turn` rechaza un turno nuevo
  (`ThreadNeedsRecovery`) hasta que `continue_after_failure` reconcilie la operación original.

**Vigencia.** Un turno lógico. El alcance y las reservas se conservan durante `interrupt`,
`resume` y `continue_after_failure`, también después de reiniciar el proceso. Un `run_turn` nuevo
trae su propio alcance y un registro vacío. Nunca pisa reservas abiertas, porque se rechaza si hay
una aprobación pendiente o una escritura incierta.

## Trazas (`arl.trace/2`)

- `run_started.scope` indica el tipo de alcance (`explicit`, `read_only` o
  `unrestricted_legacy_v1`).
- `tool_call_routed.decision="out_of_scope"` incluye `scope_reason`.
- El evento `scope_reservation` registra `action_id`, `grant_id`, `task_id` y el estado.
- El reporte (`arl.run-report/3`) agrega `scope.blocked` y `scope.reservations`.
- Los lectores aceptan v1 y v2.

## Compatibilidad con la suite v1

`RequestScope.unrestricted_legacy_v1()` reproduce el comportamiento anterior a 6A.1. **Solo** lo
usa el runner de la suite de evaluación v1, cuyos casos no declaran alcance. Así esa evidencia
histórica sigue siendo comparable, incluido F1. Un test revisa todo el código y verifica que no se
use en otro lugar.

## Límites

- **Solo restringe escrituras.** Las lecturas siguen limitadas únicamente por el rol en el
  servidor. Una inyección todavía puede pedir leer otros proyectos del mismo usuario: falta
  restringir el alcance de las lecturas.
- **El control está en el agente, no en el servidor.** Un proceso de agente comprometido puede
  saltear el alcance, pero nunca la autorización por rol. Llevarlo al servidor requiere cambiar el
  contrato MCP.
- **La granularidad la decide la aplicación.** Un alcance demasiado amplio (por ejemplo,
  `status=None` sobre muchas tareas) vuelve a depender del humano dentro de ese margen.
- **No hay derivación automática** del alcance a partir del pedido. Si en el futuro un LLM
  propusiera el alcance, el humano debería confirmarlo; el LLM nunca debería ser la autoridad.
- El sello no protege contra quien pueda escribir en los checkpoints.
