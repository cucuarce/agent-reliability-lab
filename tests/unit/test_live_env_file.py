"""``GROQ_API_KEY`` desde un ``.env`` local (etapa 6B.1), sin claves reales ni red.

Todos los archivos se crean en ``tmp_path``: ningún test lee el ``.env`` del repo.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

from arl.evals.live import __main__ as cli
from arl.evals.live.providers import API_KEY_ENV, load_env_file

ROOT = Path(__file__).resolve().parents[2]
CANARY = "gsk_canary_" + "e" * 40
OTHER = "gsk_other_" + "o" * 40


def _env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def no_key(monkeypatch):
    """Sin ``GROQ_API_KEY`` en el entorno; el teardown la quita aunque el test la cargue."""
    monkeypatch.setenv(API_KEY_ENV, "x")
    monkeypatch.delenv(API_KEY_ENV)


def test_loads_the_key_when_the_environment_does_not_define_it(tmp_path, caplog) -> None:
    environ: dict[str, str] = {}
    with caplog.at_level(logging.DEBUG):
        source = load_env_file(_env_file(tmp_path, f"{API_KEY_ENV}={CANARY}\n"), environ)
    assert source == "env_file"
    assert environ == {API_KEY_ENV: CANARY}
    assert CANARY not in caplog.text


@pytest.mark.parametrize(
    ("existing", "source"), [(OTHER, "environment"), ("", None)], ids=["definida", "vacia"]
)
def test_never_overwrites_an_existing_variable(tmp_path, existing, source) -> None:
    environ = {API_KEY_ENV: existing}
    assert load_env_file(_env_file(tmp_path, f"{API_KEY_ENV}={CANARY}\n"), environ) == source
    assert environ == {API_KEY_ENV: existing}


def test_takes_only_the_groq_key_from_the_file(tmp_path) -> None:
    text = (
        f"ARL_PRINCIPAL=admin\nARL_DATA_DIR={tmp_path}\nARL_LOG_LEVEL=DEBUG\n"
        f"OTRA_CLAVE=1\n{API_KEY_ENV}={CANARY}\n"
    )
    environ: dict[str, str] = {}
    load_env_file(_env_file(tmp_path, text), environ)
    assert environ == {API_KEY_ENV: CANARY}


def test_value_is_taken_literally_without_interpolation(tmp_path) -> None:
    environ: dict[str, str] = {}
    text = f'PREFIJO=abc\n{API_KEY_ENV}="${{PREFIJO}}-{CANARY}"\n'
    load_env_file(_env_file(tmp_path, text), environ)
    assert environ == {API_KEY_ENV: f"${{PREFIJO}}-{CANARY}"}


@pytest.mark.parametrize(
    "text", ["", f"{API_KEY_ENV}=\n", f"# {API_KEY_ENV}={CANARY}\n", "OTRA=1\n"]
)
def test_missing_or_empty_key_leaves_the_environment_untouched(tmp_path, text) -> None:
    environ: dict[str, str] = {}
    assert load_env_file(_env_file(tmp_path, text), environ) is None
    assert environ == {}


def test_missing_file_is_not_an_error(tmp_path) -> None:
    environ: dict[str, str] = {}
    assert load_env_file(tmp_path / "no-existe.env", environ) is None
    assert environ == {}


# --- CLI ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "env_value", "shown", "origin"),
    [
        (f"{API_KEY_ENV}={CANARY}\n", None, "definida", ".env"),
        (f"{API_KEY_ENV}={CANARY}\n", OTHER, "definida", "entorno"),
        (f"{API_KEY_ENV}=\n", None, "no definida", None),
    ],
    ids=["desde-env-file", "el-entorno-gana", "plantilla-vacia"],
)
def test_config_reports_presence_and_origin_but_never_the_value(
    tmp_path, monkeypatch, capsys, no_key, text, env_value, shown, origin
) -> None:
    if env_value is not None:
        monkeypatch.setenv(API_KEY_ENV, env_value)
    monkeypatch.setattr(cli, "ENV_FILE", _env_file(tmp_path, text))
    assert cli.main(["config"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["credential"] == {API_KEY_ENV: shown, "origen": origin}
    assert CANARY not in out.out + out.err and OTHER not in out.out + out.err


def test_env_file_is_loaded_before_get_models(tmp_path, monkeypatch, capsys, no_key) -> None:
    monkeypatch.setattr(cli, "ENV_FILE", _env_file(tmp_path, f"{API_KEY_ENV}={CANARY}\n"))
    seen: list[bool] = []

    def fake_list_models():
        seen.append(os.environ.get(API_KEY_ENV) == CANARY)
        return []

    monkeypatch.setattr(cli, "list_models", fake_list_models)
    assert cli.main(["models"]) == 0
    assert seen == [True]
    out = capsys.readouterr()
    assert CANARY not in out.out + out.err


def test_run_without_a_key_in_env_or_file_is_refused(tmp_path, monkeypatch, no_key) -> None:
    monkeypatch.setattr(cli, "ENV_FILE", _env_file(tmp_path, f"{API_KEY_ENV}=\n"))

    def forbidden(*args, **kwargs):
        raise AssertionError("no debía ejecutar nada")

    monkeypatch.setattr(cli, "run_live_suite", forbidden)
    assert cli.main(["run", "--only", "L01", "--confirm-real-calls", "--no-write"]) == 2


def test_server_side_code_never_reads_provider_credentials() -> None:
    """En las evaluaciones el servidor MCP corre en el mismo proceso que el cliente de Groq
    (en memoria): la clave está en ``os.environ``, pero el código del servidor no la lee."""
    server_side = ("mcp_server", "services", "domain", "storage")
    for package in server_side:
        for source in (ROOT / "src" / "arl" / package).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            assert "API_KEY" not in text and "GROQ" not in text.upper(), source


# --- Plantilla y exclusión de git -----------------------------------------------------------------


def test_template_documents_the_key_without_any_value() -> None:
    values = dotenv_values(ROOT / ".env.example")
    assert values == {API_KEY_ENV: ""}


@pytest.mark.parametrize(
    ("path", "ignored"),
    [(".env", True), (".env.local", True), ("certs/dev.pem", True), (".env.example", False)],
)
def test_git_ignores_secret_files_but_not_the_template(path, ignored) -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("sin git o fuera de un repositorio")
    proc = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=False)
    assert proc.returncode == (0 if ignored else 1)
