# Agent Reliability Lab

Proyecto de portfolio para practicar la ingeniería de agentes de IA confiables. Un agente construido con
Python y LangGraph gestiona proyectos mediante tools MCP (`list_projects`, `get_project`,
`update_task`). El proyecto incluye:

- autorización determinística por usuario y por proyecto, resuelta fuera del LLM;
- aprobación humana antes de cualquier escritura;
- escrituras idempotentes que no se duplican ante reintentos ni reanudaciones;
- trazabilidad en JSONL y auditoría en SQLite;
- dos suites de evaluación separadas: una determinística y otra conductual.

> **Alcance honesto.** Todos los datos son sintéticos. El parámetro `--user` representa una
> **identidad simulada**: el proyecto no implementa autenticación real. Por ahora el agente se
> prueba con un **LLM simulado** y determinístico; no hay un proveedor real integrado. Las métricas de las
> evaluaciones describen el comportamiento sobre estos escenarios sintéticos y no demuestran
> confiabilidad en producción.

## Estado

| Etapa | Contenido | Estado |
|---|---|---|
| 0. Bootstrap | Repo, dependencias fijadas, configuración de Claude Code, hook de protección, tests de contrato | ✅ |
| 1. Dominio | Modelos, política de autorización, SQLite, ProjectService, idempotencia y auditoría | ✅ |
| 2. Servidor MCP | 3 tools sobre `MCPServer` (SDK v2), `.mcp.json` de solo lectura ([docs/mcp.md](docs/mcp.md)) | ✅ |
| 3. Grafo | `StateGraph` con aprobación por `interrupt()` ([docs/agent.md](docs/agent.md)) | ✅ |
| 4. Robustez | Clasificación de fallos, reintentos acotados, `WRITE_UNCERTAIN`, historial consistente, lock por thread obligatorio, logs sin secretos, fault injection ([docs/reliability.md](docs/reliability.md)) | ✅ |
| 5. Observabilidad | Trazas JSONL versionadas, consulta y reconstrucción (`python -m arl.tracing`), logging seguro, Skill `/analyze-run` ([docs/observability.md](docs/observability.md)). La CLI de chat/aprobación queda pendiente | ✅ |
| 6A. Evaluación determinística | 37 casos declarativos, graders determinísticos, métricas funcionales y de seguridad separadas (`python -m arl.evals`, [docs/evals.md](docs/evals.md)) | ✅ |
| 6A.1. Alcance del turno | `RequestScope` declarado por la aplicación: bloquea escrituras fuera de lo autorizado antes de pedir aprobación; suite v2 con casos reservados ([docs/scope.md](docs/scope.md)) | ✅ |
| 6B.1. Integración con Groq | Ruta aislada con presupuesto estricto, dry-run sin inferencia, 14 escenarios y primera tanda (L01–L02, máx. 8 pedidos) preparada; todavía sin inferencia real ([docs/live-evals.md](docs/live-evals.md)) | ✅ |
| 7. Documentación | Arquitectura, threat model, ADRs | pendiente |

## Requisitos

- [uv](https://docs.astral.sh/uv/) ≥ 0.12
- Python 3.12. Si no está instalado, uv lo descarga (`.python-version`).

## Uso

```bash
uv sync                      # crea .venv e instala las versiones fijadas en uv.lock
uv run pytest                # todos los tests
uv run ruff check .          # lint
uv run ruff format --check . # formato
uv run pytest --cov=arl --cov-branch  # cobertura
uv run python scripts/mutation_check.py  # ¿los tests detectan las regresiones? (~10 min)
uv run python scripts/sample_trace.py --out <dir>  # trazas de ejemplo (datos sintéticos)
uv run python -m arl.evals                        # suite de evaluación -> evals/reports/<fecha>/
uv run python -m arl.tracing show <run_id> --dir <dir>/traces  # reconstruir una ejecución
```

### Datos

Las bases SQLite viven en `ARL_DATA_DIR`. Por defecto la ruta es `%LOCALAPPDATA%\arl` en Windows
y `$XDG_DATA_HOME/arl` o `~/.local/share/arl` en el resto. Una ruta dentro de OneDrive se rechaza:
la sincronización bloquea archivos y puede corromper el WAL de SQLite.

## Estructura

```
src/arl/domain/      Modelos Pydantic, errores, política de autorización (pura)
src/arl/storage/     SQLite: schema, repositorios, ledger de idempotencia + auditoría, seed
src/arl/services/    ProjectService: único punto de acceso al dominio
src/arl/mcp_server/  Servidor MCP stdio: adaptador fino sobre ProjectService
src/arl/agent/       Agente LangGraph: registro de tools, gateway MCP, grafo con aprobación
src/arl/tracing/     Trazas JSONL (arl.trace/1), consulta/reconstrucción y logging seguro
src/arl/evals/       Evaluación determinística: casos, runner aislado, graders y métricas
evals/cases/         Casos declarativos (JSON), uno por archivo
tests/unit/          Política, modelos, configuración
tests/integration/   Servicio contra SQLite real: autorización, versiones, idempotencia, concurrencia
tests/contracts/     Contratos de las librerías fijadas (SDK MCP v2, LangGraph)
tests/devtools/      Tests de la infraestructura de Claude Code
CLAUDE.md, .claude/  Infraestructura de desarrollo con Claude Code (no forma parte del producto)
```

## Dominio (etapa 1)

- **Identidad simulada.** `Principal` representa al actor y **no autentica**: solo valida el
  formato del identificador. Lo construye código de confianza (el entorno del servidor MCP, y más
  adelante la CLI), nunca un argumento generado por el LLM.
- **Autorización.** `domain/policy.py` es una tabla fija (rol, acción) → permitido: `viewer`
  solo lee; `editor` y `owner` pueden modificar tareas; quien no es miembro no ve nada. Un proyecto
  ajeno y uno inexistente devuelven el mismo `NotFound`.
- **Escrituras.** `ProjectService.update_task` verifica, dentro de una única transacción
  `BEGIN IMMEDIATE`: visibilidad → ledger de idempotencia → rol → reglas del cambio → versión
  esperada. Recién después escribe tarea, auditoría y ledger, que se confirman juntos.
- **Idempotencia.** La clave se aísla por principal. La misma clave con los mismos parámetros
  devuelve la respuesta original sin volver a escribir; con otros parámetros devuelve
  `IdempotencyConflict`. El replay exige visibilidad actual: quien perdió la membresía recibe
  `NotFound`, nunca la respuesta guardada.
- **Sin cambios efectivos.** Un patch igual al estado actual devuelve `outcome="unchanged"`, sin
  versión nueva ni auditoría. Igual exige permiso de escritura y versión vigente, y queda en el
  ledger para que los reintentos sean coherentes.

Los datos (`src/arl/storage/seed_data.json`) son sintéticos.

El producto (`src/arl/`) y la infraestructura de desarrollo de Claude Code (`CLAUDE.md`, `.claude/`)
están separados: el producto no importa ni ejecuta nada de `.claude/`.

### Hook de protección (Claude Code)

`.claude/hooks/guard_paths.py` se ejecuta antes de cada `Read`, `Edit`, `Write`, `Bash` y
`PowerShell` de Claude Code:

- **Secretos** (`.env`, `.env.*`, `*.pem`, `*.key`): bloquea la acción. `.env.example` sigue permitido.
- **Casos de evaluación** (`evals/cases/`): crear un caso nuevo está permitido. Modificar uno
  existente pide confirmación humana (`ask`), para que nadie "arregle" una evaluación cambiando lo
  que espera.

El hook es una protección *best-effort* del entorno de desarrollo, **no una barrera de seguridad
absoluta**. En comandos de shell busca por tokens, así que un comando ofuscado o una búsqueda
recursiva que lea secretos no se detectan. La frontera de seguridad del producto es la autorización
determinística del servidor, no este hook.

## Licencia

[MIT](LICENSE). Las dependencias usan licencias permisivas (MIT, BSD, Apache-2.0, PSF, MPL-2.0 a
nivel de archivo); el repositorio no incluye código de terceros.
