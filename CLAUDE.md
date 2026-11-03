# Agent Reliability Lab

Portfolio: agente LangGraph que gestiona proyectos mediante tools MCP, con autorización
determinística, aprobación humana de escrituras y evaluaciones de confiabilidad.
Datos sintéticos. `--user` es una identidad simulada: no hay autenticación real y no hay que afirmarlo.

## Estado y alcance

- Etapa actual: 4 (tolerancia a fallos, ver `docs/reliability.md`). No avanzar sin aprobación.
- Cambios en identidad, aprobación o fallos: correr `uv run python scripts/mutation_check.py`.
- Fuera del MVP: RAG, multiagentes, frontend, despliegue cloud, LangSmith, OpenTelemetry.

## Dos mundos separados

- Producto: `src/arl/` y sus tests (`tests/` salvo `tests/devtools/`).
- Infraestructura de Claude Code: `CLAUDE.md`, `.claude/`, `tests/devtools/`.
  El producto nunca importa ni ejecuta nada de `.claude/`.

## Comandos

- `uv sync`: instala las versiones fijadas en `uv.lock`.
- `uv run pytest`: todos los tests. `uv run pytest tests/devtools` solo para la infraestructura.
- `uv run ruff check .` y `uv run ruff format --check .`
- Dependencias: `uv add <pkg>` / `uv add --dev <pkg>`. Nunca editar `uv.lock` a mano.

## Invariantes (no negociables)

1. La identidad nunca proviene de argumentos generados por el LLM.
2. La autorización la decide código determinístico del servidor, no el prompt ni el cliente.
3. Toda escritura requiere aprobación humana ligada a la acción exacta (args + versión).
4. Ningún efecto secundario antes de `interrupt()`; toda escritura lleva clave de idempotencia.
5. No modificar las expectativas de `evals/cases/` para que un caso pase.
6. No afirmar confiabilidad de producción a partir de datos sintéticos; reportar métricas reales.

## Convenciones

- Python 3.12, tipado completo, `sqlite3` de stdlib (sin ORM).
- Todo acceso al dominio pasa por `src/arl/services/project_service.py`; SQLite solo en `ARL_DATA_DIR`
  (nunca dentro de OneDrive).
- SDK MCP v2: `from mcp.server import MCPServer` y `from mcp import Client` (`FastMCP` ya no existe).
- LangGraph: `StateGraph` propio; `interrupt`/`Command` desde `langgraph.types`.
- Las APIs de librerías se verifican en `tests/contracts/test_pinned_sdks.py` antes de usarlas.
- Todo cambio de producto va con tests. Antes de dar algo por terminado: pytest y ruff en verde.

## Hooks activos

- `.claude/hooks/guard_paths.py` (PreToolUse): bloquea secretos y pide confirmación al modificar
  casos de evaluación existentes. Tests: `tests/devtools/test_guard_paths.py`.
