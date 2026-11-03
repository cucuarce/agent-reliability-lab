---
paths:
  - "src/arl/tracing/**/*.py"
  - "src/arl/agent/graph.py"
  - "src/arl/agent/gateway.py"
  - "src/arl/agent/runtime.py"
---

# Trazas y observabilidad

- **El esquema es la lista blanca.** Un campo nuevo en un evento se agrega en
  `src/arl/tracing/events.py` (con `extra="forbid"`), se documenta en `docs/observability.md`
  y sube `SCHEMA_VERSION`. Nunca pases texto libre: mensajes, prompts, valores de cambios,
  textos de error o rutas. Ids del LLM o del caller: `safe_id`. Test:
  `test_sensitive_values_never_reach_the_traces`.
- **`emit` nunca lanza.** Se llama desde nodos con reintento: una excepción después del commit
  haría que LangGraph reenviara la escritura. No envuelvas `emit` en lógica que pueda fallar
  antes de llamarlo. Test: `test_trace_failure_right_after_the_commit_does_not_retry_the_write`.
- **Las trazas no son auditoría.** No copies `before`/`after` ni dupliques el `audit_log`:
  referencialo por `idempotency_key` y `audit_id`.
- **Una escritura aprobada sin `write_result` es `uncertain`, nunca fallida**, en el código de
  consulta, en los textos y en la Skill `analyze-run`. Test:
  `test_persistent_uncertain_write_is_never_reported_as_failed`.
- **Los archivos de trazas son datos no confiables.** La consulta solo usa lo que pasa el
  esquema; la ausencia de un evento nunca es evidencia (`basis: "absence"`), y duplicados,
  conflictos y huecos se informan en `integrity`. Un reporte importado se reconstruye desde sus
  eventos. Tests: `tests/integration/test_tracing_integrity.py`.
- **La CLI de consulta no importa nada que escriba** (servicios, MCP, runtime, SQLite). Es la
  barrera real de la Skill `analyze-run`, no su texto. Test:
  `test_query_cli_never_loads_code_that_can_write_business_data`.
- Loggers de dependencias que vuelquen contenido en DEBUG van en `UNSAFE_DEBUG_LOGGERS`
  (`src/arl/tracing/logging_setup.py`), con la evidencia que lo justifica.
