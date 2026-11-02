"""Tests de la configuración de Claude Code: detectan deriva entre la configuración y el repo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ALLOWED_PERMISSION_PREFIXES = (
    "Bash(uv sync",
    "Bash(uv run pytest",
    "Bash(uv run ruff check",
    "Bash(uv run ruff format",
    "Bash(git status",
    "Bash(git diff",
    "Bash(git log",
)
MAX_CLAUDE_MD_LINES = 80


@pytest.fixture(scope="module")
def settings(repo_root: Path) -> dict:
    # json.loads es estricto: comentarios o comas finales fallan, igual que en Claude Code.
    return json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))


def test_settings_permissions_are_minimal(settings: dict) -> None:
    permissions = settings["permissions"]
    assert "defaultMode" not in permissions
    for rule in permissions["allow"]:
        assert rule.startswith(ALLOWED_PERMISSION_PREFIXES), f"permiso no previsto: {rule}"
    assert {"Read(**/.env)", "Edit(**/.env)"} <= set(permissions["deny"])


def test_hook_scripts_exist_and_cover_required_tools(settings: dict, repo_root: Path) -> None:
    pre_tool_use = settings["hooks"]["PreToolUse"]
    matchers = {tool for group in pre_tool_use for tool in group["matcher"].split("|")}
    assert {"Read", "Edit", "Write", "Bash", "PowerShell"} <= matchers

    for group in pre_tool_use:
        for handler in group["hooks"]:
            scripts = [a for a in handler["args"] if a.startswith("${CLAUDE_PROJECT_DIR}/")]
            assert scripts, "cada hook debe referenciar un script del repo"
            for script in scripts:
                path = repo_root / script.removeprefix("${CLAUDE_PROJECT_DIR}/")
                assert path.is_file(), f"script inexistente: {path}"


def test_claude_md_is_concise_and_states_simulated_identity(repo_root: Path) -> None:
    text = (repo_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= MAX_CLAUDE_MD_LINES
    assert "identidad simulada" in text


def test_claude_md_referenced_files_exist(repo_root: Path) -> None:
    """Cada archivo citado entre backticks en CLAUDE.md debe existir (evita instrucciones rotas)."""
    text = (repo_root / "CLAUDE.md").read_text(encoding="utf-8")
    referenced = re.findall(r"`([\w./-]+\.(?:py|md|json|toml|lock|yaml))`", text)
    assert referenced
    missing = [ref for ref in referenced if not (repo_root / ref).is_file()]
    assert not missing, f"CLAUDE.md referencia archivos inexistentes: {missing}"
