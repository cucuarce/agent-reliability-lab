from __future__ import annotations

from pathlib import Path

import pytest

from arl.config import ConfigError, app_db_path, resolve_data_dir


def test_explicit_data_dir(tmp_path: Path) -> None:
    assert resolve_data_dir({"ARL_DATA_DIR": str(tmp_path)}) == tmp_path.resolve()
    assert app_db_path({"ARL_DATA_DIR": str(tmp_path)}) == tmp_path.resolve() / "app.db"


def test_default_uses_localappdata_on_windows(tmp_path: Path) -> None:
    assert resolve_data_dir({"LOCALAPPDATA": str(tmp_path)}) == (tmp_path / "arl").resolve()


def test_default_uses_xdg_data_home(tmp_path: Path) -> None:
    assert resolve_data_dir({"XDG_DATA_HOME": str(tmp_path)}) == (tmp_path / "arl").resolve()


@pytest.mark.parametrize(
    "data_dir",
    [
        "C:/Users/someone/OneDrive/Desktop/arl",
        "C:/Users/someone/OneDrive - Contoso/arl",
        "/home/someone/onedrive/arl",
    ],
)
def test_rejects_onedrive_folders(data_dir: str) -> None:
    with pytest.raises(ConfigError, match="OneDrive"):
        resolve_data_dir({"ARL_DATA_DIR": data_dir})


def test_repo_location_is_rejected_as_data_dir() -> None:
    """Este repo vive en OneDrive: usarlo como ARL_DATA_DIR debe fallar en esta máquina."""
    repo = Path(__file__).resolve().parents[2]
    if not any(p.casefold().startswith("onedrive") for p in repo.parts):
        pytest.skip("el repo no está en OneDrive en esta máquina")
    with pytest.raises(ConfigError):
        resolve_data_dir({"ARL_DATA_DIR": str(repo / "var")})
