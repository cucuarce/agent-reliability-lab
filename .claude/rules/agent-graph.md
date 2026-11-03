---
paths:
  - "src/arl/agent/**/*.py"
---

# Grafo del agente (LangGraph)

Invariantes verificadas con la API real de LangGraph 1.2.x. Romper cualquiera tiene consecuencias
concretas.

- **Efectos solo en `execute_read`, `prepare_write` (lecturas) y `execute_write` (la única
  escritura).** `await_approval` contiene solo `interrupt()`: el nodo se re-ejecuta desde el
  inicio al reanudar.
- **Ningún nodo lanza excepciones después de `interrupt()` por una decisión inválida.** LangGraph
  conserva el valor de resume rechazado y lo vuelve a entregar: el thread quedaría envenenado.
  Registrá `approval_error` y volvé a `await_approval`. Test:
  `test_invalid_decision_sent_directly_to_graph_does_not_poison_the_thread`.
- **Los reducers nunca lanzan.** LangGraph persiste la entrada antes de aplicar reducers; un
  reducer que falla deja el thread ilegible. Test:
  `test_thread_cannot_be_continued_by_another_principal`.
- **`execute_write` usa solo la `PendingWrite` sellada y aprobada**, nunca argumentos del LLM.
  `expected_version` e `idempotency_key` los genera el agente (`write_identity`); el LLM no los ve.
- **La identidad es la del gateway**, ligada al grafo en `build_graph`. El runtime no acepta
  `principal_id` del caller.
- **`src/arl/agent/runtime.py` es la interfaz confiable.** Toda función pública nueva debe obtener
  `_lock_dir(graph)` (falla con `LockingRequired` si falta), tomar `thread_lock` y llamar a
  `_check_owner` antes de ejecutar el grafo. No agregues un modo sin lock. Tests:
  `test_other_principal_cannot_read_or_alter_a_paused_thread`,
  `test_every_runtime_entry_point_refuses_to_run_without_a_lock`.
- **Fallos del MCP:** el gateway los convierte en `ToolUnavailable` (reintentable) o
  `ToolResponseInvalid` con mensajes fijos, y los lanza **fuera** del `except`, para que no quede
  `__context__` con el texto original (LangGraph registra el traceback de cada reintento). Para
  diagnosticar quedan `code`, `tool` y `error_type`. Los errores de dominio nunca se reintentan.
  Test: `test_gateway_failures_carry_diagnostics_but_no_secret`.
- **Historial:** nunca cierres una escritura aprobada sin confirmar (`WRITE_UNCERTAIN`) ni le
  inventes una respuesta. Las lecturas y preparaciones interrumpidas se cierran con
  `[not_executed]` en `run_turn`, antes del mensaje nuevo. `call_model` verifica el contrato antes
  de invocar al LLM. Tests: `tests/integration/test_agent_history.py`.
- Si agregás un mutante o cambiás código mutado, actualizá `scripts/mutation_check.py`: aborta
  cuando un fragmento deja de coincidir.
- **No hay reglas de permisos en el agente:** la autorización por rol es exclusiva del servidor
  MCP. El **alcance del turno** (`arl.agent.scope`) es otra cosa: lo que la aplicación autorizó
  escribir. Se verifica en `dispatch` (antes de pedir aprobación), se reserva en
  `prepare_write` y se exige en `execute_write`. Ningún nodo modifica `request_scope`; solo los
  nodos deterministas escriben `scope_ledger`. Nunca liberes una reserva incierta ni consumida.
  Tests: `tests/integration/test_agent_scope.py`, `tests/unit/test_request_scope.py`.
- Toda tool nueva se registra en `src/arl/agent/tool_registry.py` con su `ToolKind`. Su
  contrato lo verifica `McpGateway.verify_contract`.
