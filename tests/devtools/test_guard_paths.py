"""Tests del hook PreToolUse .claude/hooks/guard_paths.py.

Dos niveles: lógica de decisión (funciones puras) y contrato de proceso (stdin/stdout/exit code),
incluido el comando exacto registrado en .claude/settings.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "guard_paths.py"


def _payload(tool: str, **tool_input: str) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Proyecto mínimo con un caso de evaluación existente."""
    cases = tmp_path / "evals" / "cases"
    cases.mkdir(parents=True)
    (cases / "existing.yaml").write_text("id: existing\n", encoding="utf-8")
    return tmp_path


# --- Secretos ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "config/.env.production",
        r"C:\repo\config\.ENV",
        "certs/server.pem",
        "deploy/private.key",
    ],
)
@pytest.mark.parametrize("tool", ["Read", "Edit", "Write"])
def test_file_tools_deny_secrets(guard, project: Path, tool: str, path: str) -> None:
    decision, reason = guard.decide(_payload(tool, file_path=path), str(project))
    assert decision == "deny"
    assert "secretos" in reason


@pytest.mark.parametrize(
    "path", [".env.example", ".env.sample", "docs/environment.md", "src/arl/keys.py"]
)
def test_file_tools_allow_non_secrets(guard, project: Path, path: str) -> None:
    assert guard.decide(_payload("Read", file_path=path), str(project)) is None
    assert guard.decide(_payload("Edit", file_path=path), str(project)) is None


@pytest.mark.parametrize(
    ("tool", "command"),
    [
        ("Bash", "cat .env"),
        ("Bash", "grep API_KEY config/.env.local"),
        ("Bash", 'cp "secrets/.env" /tmp/x'),
        ("PowerShell", r"Get-Content .\.env"),
        ("PowerShell", "type 'config\\.env.production'"),
        ("Bash", "git add .env"),
        # Solo las plantillas exactas están permitidas, igual que en .gitignore.
        ("Bash", "cp .env.example .env.example.bak"),
    ],
)
def test_shell_tools_deny_commands_mentioning_secrets(guard, project, tool, command) -> None:
    decision, _ = guard.decide(_payload(tool, command=command), str(project))
    assert decision == "deny"


@pytest.mark.parametrize(
    ("tool", "command"),
    [
        ("Bash", "uv run pytest"),
        ("Bash", "cat .env.example"),
        ("Bash", "cp .env.example docs/environment.md"),
        ("PowerShell", "Get-ChildItem -Force"),
        ("Bash", "echo $ENV_VAR"),
        ("PowerShell", "$env:PATH"),
        # *.key / *.pem como texto, no como archivo existente: no debe bloquear.
        ("Bash", "jq .key data.json"),
        ("Bash", 'grep -rn "self.key" src'),
        ("Bash", "uv run python -c 'print(cfg.pem)'"),
    ],
)
def test_shell_tools_allow_regular_commands(guard, project, tool, command) -> None:
    assert guard.decide(_payload(tool, command=command), str(project)) is None


def test_shell_tools_deny_existing_key_files(guard, project: Path) -> None:
    (project / "certs").mkdir()
    (project / "certs" / "server.pem").write_text("-----BEGIN-----", encoding="utf-8")
    decision, _ = guard.decide(_payload("Bash", command="cat certs/server.pem"), str(project))
    assert decision == "deny"


# --- Casos de evaluación -----------------------------------------------------------------------


def test_creating_new_eval_case_is_allowed(guard, project: Path) -> None:
    payload = _payload("Write", file_path=str(project / "evals" / "cases" / "new.yaml"))
    assert guard.decide(payload, str(project)) is None


@pytest.mark.parametrize("tool", ["Edit", "Write"])
def test_modifying_existing_eval_case_asks_human(guard, project: Path, tool: str) -> None:
    payload = _payload(tool, file_path=str(project / "evals" / "cases" / "existing.yaml"))
    decision, reason = guard.decide(payload, str(project))
    assert decision == "ask"
    assert "confirmación humana" in reason


@pytest.mark.parametrize(
    "parts", [("cases-v2", "holdout", "h.json"), ("baselines", "6a-before-d1-fix", "report.json")]
)
def test_v2_cases_and_historical_baselines_are_protected_too(guard, project: Path, parts) -> None:
    """Etapa 6A.1: los casos v2 (incluidos los reservados) y la evidencia histórica."""
    target = project.joinpath("evals", *parts)
    target.parent.mkdir(parents=True)
    assert guard.decide(_payload("Write", file_path=str(target)), str(project)) is None  # nuevo
    target.write_text("{}", encoding="utf-8")
    decision, _ = guard.decide(_payload("Edit", file_path=str(target)), str(project))
    assert decision == "ask"


def test_eval_case_relative_and_windows_style_paths(guard, project: Path) -> None:
    for path in (
        "evals/cases/existing.yaml",
        r"evals\cases\existing.yaml",
        "EVALS/Cases/existing.yaml",
    ):
        if path.startswith("EVALS") and os.name != "nt":
            continue  # case-insensitive solo en Windows
        decision, _ = guard.decide(_payload("Edit", file_path=path), str(project))
        assert decision == "ask", path


def test_reading_eval_cases_and_editing_other_eval_files_is_allowed(guard, project: Path) -> None:
    case = str(project / "evals" / "cases" / "existing.yaml")
    assert guard.decide(_payload("Read", file_path=case), str(project)) is None
    runner = str(project / "evals" / "runner.py")
    assert guard.decide(_payload("Edit", file_path=runner), str(project)) is None


def test_eval_cases_outside_project_are_not_matched(guard, project: Path, tmp_path_factory) -> None:
    other = tmp_path_factory.mktemp("other") / "evals" / "cases" / "x.yaml"
    assert guard.decide(_payload("Edit", file_path=str(other)), str(project)) is None


def test_unrelated_tools_are_ignored(guard, project: Path) -> None:
    assert guard.decide(_payload("Glob", pattern="**/.env"), str(project)) is None


# --- Contrato de proceso ------------------------------------------------------------------------


def _run(args: list[str], stdin: bytes, project: Path) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project)}
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=60, check=False)


def _encode(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def test_process_emits_deny_json_and_exit_0(project: Path) -> None:
    proc = _run(
        [sys.executable, str(HOOK_PATH)], _encode(_payload("Read", file_path=".env")), project
    )
    assert proc.returncode == 0
    output = json.loads(proc.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"


def test_process_is_silent_when_allowed(project: Path) -> None:
    payload = _encode(_payload("Read", file_path="src/arl/módulo_ñ.py"))  # UTF-8 no ASCII
    proc = _run([sys.executable, str(HOOK_PATH)], payload, project)
    assert proc.returncode == 0
    assert proc.stdout == b""


def test_process_accepts_utf8_bom(project: Path) -> None:
    """PowerShell 5.1 antepone un BOM al canalizar texto; no debe bloquear todo por eso."""
    stdin = b"\xef\xbb\xbf" + _encode(_payload("Read", file_path=".env"))
    proc = _run([sys.executable, str(HOOK_PATH)], stdin, project)
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "stdin", [b"not json", b"[]", b'{"tool_name": "Read"}', _encode(_payload("Read"))]
)
def test_process_fails_closed_on_invalid_payload(project: Path, stdin: bytes) -> None:
    proc = _run([sys.executable, str(HOOK_PATH)], stdin, project)
    assert proc.returncode == 2
    assert b"payload" in proc.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv no está en PATH")
def test_command_registered_in_settings_runs_on_this_platform(project: Path) -> None:
    """Ejecuta el comando exacto de settings.json (forma exec, sin shell), como lo hace Claude Code.

    En Windows valida que `uv` resuelva a un .exe real y que la ruta con
    ${CLAUDE_PROJECT_DIR} funcione con separadores mixtos.
    """
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    handler = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    args = [
        arg.replace("${CLAUDE_PROJECT_DIR}", str(REPO_ROOT))
        for arg in [handler["command"], *handler["args"]]
    ]
    proc = _run(args, _encode(_payload("Bash", command="cat .env")), project)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
