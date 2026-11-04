---
paths:
  - "src/arl/**/*.py"
  - "src/arl/storage/schema.sql"
  - ".mcp.json"
---

# Seguridad del producto (complementa las invariantes de CLAUDE.md)

## Identidad

- Solo `src/arl/mcp_server/__main__.py` lee `ARL_PRINCIPAL`. Nada más lee identidad del entorno.
- Ninguna tool ni modelo de entrada declara campos de identidad (`user_id`, `principal`, `actor_id`,
  `role`, ...). Si agregás una tool, extendé `test_input_schemas_expose_no_identity_fields`.
- Los modelos de entrada usan `extra="forbid"`. No los relajes para "aceptar" campos del LLM.

## Autorización

- Las reglas viven en `src/arl/domain/policy.py` y se aplican en `ProjectService`. El servidor MCP
  no decide permisos ni filtra resultados.
- Inexistente y ajeno devuelven el mismo `NotFound`. `Forbidden` solo cuando el principal ya ve
  el proyecto.

## Escrituras

Toda operación de escritura nueva necesita, en una sola `transaction(write=True)`: autorización,
`expected_version`, clave de idempotencia con hash de parámetros, auditoría y ledger. También
necesita tests de reintento, conflicto de clave, rollback y acceso ajeno.

## Errores y datos

- Al cliente MCP solo llegan mensajes de `DomainError` (`[codigo] mensaje`). Nunca `str(exc)` de
  excepciones inesperadas, SQL, rutas ni trazas.
- SQL siempre parametrizado. Los nombres de columna salen de listas blancas fijas.
- En el servidor stdio, stdout es del protocolo MCP: diagnósticos a stderr, sin secretos.
- Las credenciales de proveedores de LLM (`*_API_KEY`) son del cliente del modelo: el servidor
  MCP no las recibe (`gateway.stdio_server_parameters`) ni las lee. Test:
  `test_stdio_server_process_never_receives_llm_provider_credentials`.

## Entorno de desarrollo

`.mcp.json` usa `dev-viewer` (solo lectura) sobre `dev.db`. No cambies el principal a uno con
escritura ni apuntes a `app.db`: el servidor debe rechazar escrituras de Claude Code por
autorización, no por confirmación del cliente.
