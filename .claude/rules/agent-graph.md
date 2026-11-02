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
- **`src/arl/agent/runtime.py` es la interfaz confiable.** Toda función pública nueva debe llamar a
  `_check_owner` antes de leer o escribir el thread. Test:
  `test_other_principal_cannot_read_or_alter_a_paused_thread`.
- **No hay reglas de permisos en el agente:** la autorización es exclusiva del servidor MCP.
- Toda tool nueva se registra en `src/arl/agent/tool_registry.py` con su `ToolKind`. Su
  contrato lo verifica `McpGateway.verify_contract`.
