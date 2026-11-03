"""Hook PreToolUse de Claude Code: protege secretos y casos de evaluación.

Infraestructura de desarrollo, no forma parte del producto (src/arl/).

Contrato con Claude Code:
- stdin: JSON del evento PreToolUse (``tool_name``, ``tool_input``, ``cwd``, ...).
- stdout: ``{"hookSpecificOutput": {"permissionDecision": "deny" | "ask", ...}}`` cuando hay
  objeción; vacío cuando no la hay (el flujo normal de permisos decide, el hook nunca aprueba).
- exit 0 si se pudo decidir; exit 2 si el payload no se puede interpretar (fail-closed: el
  stderr llega a Claude como motivo del bloqueo).

Reglas:
1. Secretos (``.env``, ``.env.*``, ``*.pem``, ``*.key``; excepto plantillas como ``.env.example``):
   ``deny`` para Read/Edit/Write y para comandos Bash/PowerShell que mencionen esos archivos.
2. ``evals/cases/``, ``evals/cases-v2/`` y ``evals/baselines/``: crear archivos nuevos se
   permite; modificar archivos existentes -> ``ask``,
   así un humano confirma cada cambio de expectativas (mantenimiento legítimo sigue siendo posible).

Limitaciones: la detección en comandos de shell es best-effort por tokens (un comando ofuscado o
un ``grep -r`` sobre todo el repo no se detecta). No es una barrera de seguridad; la frontera real
para el producto es la autorización del servidor MCP.

Solo stdlib: se ejecuta con ``uv run --no-project``, sin depender del entorno del proyecto.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path, PurePath

FILE_TOOLS = frozenset({"Read", "Edit", "Write"})
SHELL_TOOLS = frozenset({"Bash", "PowerShell"})

SECRET_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template"})
SECRET_EXTENSIONS = (".pem", ".key")
# Casos de evaluación (v1 y v2) y evidencia histórica de sus resultados.
EVAL_PROTECTED_ROOTS = (("evals", "cases"), ("evals", "cases-v2"), ("evals", "baselines"))

# Separadores de tokens en comandos de shell (bash y PowerShell).
_SHELL_TOKEN_SPLIT = re.compile(r"[\s'\"`;|&<>(),=]+")


def _name(path: str) -> str:
    return PurePath(path.replace("\\", "/")).name.lower()


def _is_env_file(name: str) -> bool:
    if name == ".env":
        return True
    return name.startswith(".env.") and name.removeprefix(".env.") not in SECRET_TEMPLATE_SUFFIXES


def is_secret_path(path: str) -> bool:
    """True si el nombre de archivo corresponde a un secreto (independiente del directorio)."""
    name = _name(path)
    return _is_env_file(name) or name.endswith(SECRET_EXTENSIONS)


def mentions_secret(command: str, project_dir: str) -> str | None:
    """Devuelve el primer token del comando que apunta a un secreto, o None.

    ``.env*`` se bloquea siempre. ``*.pem``/``*.key`` solo si el archivo existe: como tokens de
    shell colisionan con texto legítimo (``jq .key``, ``grep "self.key"``).
    """
    for token in _SHELL_TOKEN_SPLIT.split(command):
        if not token:
            continue
        name = _name(token)
        if _is_env_file(name):
            return token
        if name.endswith(SECRET_EXTENSIONS) and Path(project_dir, token).is_file():
            return token
    return None


def is_eval_case(path: str, project_dir: str) -> bool:
    """True si ``path`` está dentro de ``evals/cases/``, ``evals/cases-v2/`` o
    ``evals/baselines/`` del proyecto (comparación por partes)."""
    target = Path(project_dir, path.replace("\\", "/")).resolve()
    root = Path(project_dir).resolve()
    try:
        rel_parts = target.relative_to(root).parts
    except ValueError:
        return False
    rel = tuple(os.path.normcase(part) for part in rel_parts)
    return any(rel[: len(root)] == root and len(rel) > len(root) for root in EVAL_PROTECTED_ROOTS)


def decide(payload: dict, project_dir: str) -> tuple[str, str] | None:
    """Devuelve ``(decision, motivo)`` o None si el hook no tiene objeciones."""
    tool = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool, str) or not isinstance(tool_input, dict):
        raise ValueError("payload sin 'tool_name' (str) o 'tool_input' (objeto)")

    if tool in FILE_TOOLS:
        path = tool_input.get("file_path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{tool} sin 'file_path'")
        if is_secret_path(path):
            return "deny", (
                f"guard_paths: '{path}' es un archivo de secretos; Claude no lo lee ni lo edita. "
                "Usá .env.example para documentar variables."
            )
        if tool != "Read" and is_eval_case(path, project_dir):
            if tool == "Write" and not Path(project_dir, path).exists():
                return None
            return "ask", (
                f"guard_paths: '{path}' es un caso de evaluación existente. Modificar expectativas "
                "requiere confirmación humana (invariante 5 de CLAUDE.md)."
            )
        return None

    if tool in SHELL_TOOLS:
        command = tool_input.get("command")
        if not isinstance(command, str):
            raise ValueError(f"{tool} sin 'command'")
        token = mentions_secret(command, project_dir)
        if token:
            return "deny", (
                f"guard_paths: el comando menciona el archivo de secretos '{token}'. "
                "Si es mantenimiento legítimo, ejecutalo manualmente fuera de Claude Code."
            )
    return None


def main() -> int:
    try:
        # utf-8-sig: PowerShell 5.1 antepone un BOM al canalizar texto hacia procesos nativos.
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("el payload no es un objeto JSON")
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
        result = decide(payload, project_dir)
    except (ValueError, UnicodeDecodeError) as exc:  # json.JSONDecodeError es ValueError
        print(f"guard_paths: payload inválido, se bloquea por seguridad: {exc}", file=sys.stderr)
        return 2

    if result is not None:
        decision, reason = result
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        }
        sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
