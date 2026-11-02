# Servidor MCP (`arl`)

Adaptador fino sobre `ProjectService`, construido con el SDK oficial MCP Python v2 (`MCPServer`).
Transporte: stdio.

> La identidad es **simulada**: `ARL_PRINCIPAL` la fija quien lanza el proceso. No hay
> autenticación real ni garantías de seguridad de producción.

## Tools

| Tool | Argumentos | Resultado | Hints (no son seguridad) |
|---|---|---|---|
| `list_projects` | — | `{"result": [{id, name, my_role}]}` | read-only |
| `get_project` | `project_id` | `{project, my_role, tasks[]}` (cada tarea con `version`) | read-only |
| `update_task` | `task_id`, `changes{title?, status?, assignee_id?}`, `expected_version`, `idempotency_key` | `{task, outcome, audit_id, replayed}` | escritura idempotente |

Ningún schema incluye identidad. Los argumentos de nivel superior no declarados los ignora el SDK.
Dentro de `changes`, un campo desconocido se rechaza (`extra_forbidden`).

Los errores de dominio llegan como `isError: true` con un código estable: `[not_found]`,
`[forbidden]`, `[version_conflict]`, `[idempotency_conflict]`, `[invalid_change]`. Cualquier otra
falla se ve solo como `Error executing tool <name>`.

## Configuración (entorno del proceso)

| Variable | Default | Uso |
|---|---|---|
| `ARL_PRINCIPAL` | — (obligatoria) | Identidad simulada; debe existir en la base |
| `ARL_DATA_DIR` | `%LOCALAPPDATA%\arl` / `~/.local/share/arl` | Directorio de bases; rechazado si está en OneDrive |
| `ARL_DB_NAME` | `app.db` | Nombre del archivo (sin rutas) |

El proceso valida todo al arrancar y termina con exit 2 y un mensaje en stderr si algo falla.

## Base de desarrollo

```bash
uv run python -m arl.storage.seed --db-name dev.db --read-only-user dev-viewer --reset
```

Crea `dev.db` con los datos sintéticos más `dev-viewer`, que es `viewer` en todos los proyectos.

## Claude Code

`.mcp.json` registra `arl` con `ARL_PRINCIPAL=dev-viewer` y `ARL_DB_NAME=dev.db`. Hay dos capas
independientes:

1. **Cliente:** `.claude/settings.json` pone `mcp__arl__update_task` en `ask`.
2. **Servidor (autoritativa):** aunque se confirme o se saltee la pregunta, `dev-viewer` no tiene
   permiso de escritura y el servidor responde `[forbidden]` sin tocar la base.

Pasos:

1. Crear `dev.db` (arriba).
2. Abrir `claude` en el repo y aprobar el servidor de proyecto `arl` cuando lo pida
   (`claude mcp list` muestra `Pending approval` hasta entonces).
3. Probar: "listá los proyectos con la tool de arl".

## MCP Inspector

Interfaz web:

```bash
npx @modelcontextprotocol/inspector -e ARL_PRINCIPAL=dev-viewer -e ARL_DB_NAME=dev.db uv run --quiet python -m arl.mcp_server
```

Modo CLI (útil para scripts):

```bash
npx @modelcontextprotocol/inspector --cli -e ARL_PRINCIPAL=dev-viewer -e ARL_DB_NAME=dev.db \
  uv run --quiet python -m arl.mcp_server --method tools/list
```

Para probar una escritura permitida, usá un principal con rol de escritura sobre una base
descartable, por ejemplo `-e ARL_PRINCIPAL=alice -e ARL_DB_NAME=scratch.db` después de
`python -m arl.storage.seed --db-name scratch.db`.

## Tests

- `tests/integration/test_mcp_server.py`: `Client(server)` en memoria. Cubre descubrimiento,
  schemas sin identidad, éxitos, permisos, errores, ocultamiento de fallas y aislamiento entre
  usuarios.
- `tests/integration/test_mcp_entrypoint.py`: validación del entorno al arrancar.
- `tests/devtools/test_mcp_config.py`: lanza el comando exacto de `.mcp.json` por stdio y verifica
  que `dev-viewer` no pueda escribir en ningún proyecto.
