"""Arranque del servidor: la identidad y la base salen del entorno de ejecución y se validan."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.server import MCPServer

from arl.config import ConfigError
from arl.domain.models import Principal
from arl.mcp_server.__main__ import build_from_env
from arl.mcp_server.server import create_server
from arl.storage import seed


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    seed.main(
        ["--db-name", "dev.db", "--read-only-user", "dev-viewer"], {"ARL_DATA_DIR": str(tmp_path)}
    )
    return tmp_path


def _env(data_dir: Path, **extra: str) -> dict[str, str]:
    return {"ARL_DATA_DIR": str(data_dir), "ARL_DB_NAME": "dev.db"} | extra


def test_valid_environment_builds_server(data_dir: Path) -> None:
    assert isinstance(build_from_env(_env(data_dir, ARL_PRINCIPAL="dev-viewer")), MCPServer)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "falta ARL_PRINCIPAL"),
        ({"ARL_PRINCIPAL": ""}, "falta ARL_PRINCIPAL"),
        ({"ARL_PRINCIPAL": "Alice"}, "formato inválido"),
        ({"ARL_PRINCIPAL": "ghost"}, "no existe"),
        ({"ARL_PRINCIPAL": "alice", "ARL_DB_NAME": "missing.db"}, "no existe la base"),
        ({"ARL_PRINCIPAL": "alice", "ARL_DB_NAME": "../app.db"}, "inválido"),
    ],
)
def test_invalid_environment_fails_fast(data_dir: Path, overrides: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_from_env(_env(data_dir) | overrides)


def test_create_server_requires_trusted_principal(data_dir: Path) -> None:
    with pytest.raises(TypeError, match="Principal"):
        create_server(data_dir / "dev.db", "alice")  # type: ignore[arg-type]
    assert create_server(data_dir / "dev.db", Principal(user_id="alice"))


def test_read_only_user_is_viewer_everywhere() -> None:
    data = seed.with_read_only_user(seed.load_seed(), "dev-viewer")
    roles = {m.project_id: m.role for m in data.memberships if m.user_id == "dev-viewer"}
    assert roles == {"apollo": "viewer", "borealis": "viewer", "cassini": "viewer"}
    with pytest.raises(ValueError, match="ya existe"):
        seed.with_read_only_user(seed.load_seed(), "alice")
