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
> **identidad simulada**: el proyecto no implementa autenticación real. Las métricas de las
> evaluaciones describen el comportamiento sobre estos escenarios sintéticos y no demuestran
> confiabilidad en producción.

## Estado

| Etapa | Contenido | Estado |
|---|---|---|
| 0. Bootstrap | Repo, dependencias fijadas, configuración de Claude Code, hook de protección, tests de contrato | ✅ |
| 1. Dominio | Modelos, política de autorización, SQLite, servicios | pendiente |
| 2. Servidor MCP | 3 tools sobre `MCPServer` (SDK v2) | pendiente |
| 3. Grafo | `StateGraph` con aprobación por `interrupt()` | pendiente |
| 4. Robustez | Idempotencia, conflictos de versión, crash/resume | pendiente |
| 5. CLI y observabilidad | `arl chat/pending/approve/trace` | pendiente |
| 6. Evaluaciones | ~30 casos, suite determinística + conductual | pendiente |
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
```

## Estructura

```
src/arl/             Código del producto (agente, servidor MCP, dominio)
tests/contracts/     Contratos de las librerías fijadas (SDK MCP v2, LangGraph)
tests/devtools/      Tests de la infraestructura de Claude Code
CLAUDE.md, .claude/  Infraestructura de desarrollo con Claude Code (no forma parte del producto)
```

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
