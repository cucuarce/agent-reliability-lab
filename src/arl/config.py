"""Configuración de rutas de datos.

Las bases SQLite viven en ``ARL_DATA_DIR``. Por defecto se ubican fuera de cualquier carpeta
sincronizada: los clientes de sincronización (OneDrive) bloquean archivos y pueden corromper el
WAL de SQLite, así que una ruta dentro de OneDrive se rechaza explícitamente.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DATA_DIR_ENV = "ARL_DATA_DIR"
APP_DB_NAME = "app.db"


class ConfigError(ValueError):
    """Configuración inválida."""


def _default_data_dir(env: Mapping[str, str]) -> Path:
    if local_app_data := env.get("LOCALAPPDATA"):  # Windows
        return Path(local_app_data) / "arl"
    if xdg := env.get("XDG_DATA_HOME"):
        return Path(xdg) / "arl"
    return Path.home() / ".local" / "share" / "arl"


def _is_synced_folder(path: Path) -> bool:
    return any(part.casefold().startswith("onedrive") for part in path.parts)


def resolve_data_dir(env: Mapping[str, str] | None = None) -> Path:
    """Devuelve el directorio de datos absoluto, sin crearlo."""
    env = os.environ if env is None else env
    raw = env.get(DATA_DIR_ENV)
    path = Path(raw).expanduser() if raw else _default_data_dir(env)
    path = path.resolve()
    if _is_synced_folder(path):
        raise ConfigError(
            f"{DATA_DIR_ENV}={path} está dentro de una carpeta de OneDrive; "
            "SQLite necesita un directorio local no sincronizado."
        )
    return path


def app_db_path(env: Mapping[str, str] | None = None) -> Path:
    return resolve_data_dir(env) / APP_DB_NAME
