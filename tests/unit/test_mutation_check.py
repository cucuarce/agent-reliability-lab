"""``scripts/mutation_check.py``: exige una línea base limpia y nunca modifica el árbol de trabajo.

Se ejecuta el script real (copia, aislamiento, pytest en subproceso) sobre un repositorio mínimo
con un mutante propio, para que el test sea rápido.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mutation_check.py"


def _load(monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mutation_check_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # @dataclass lo busca al definirse
    spec.loader.exec_module(module)
    return module


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "arl").mkdir(parents=True)
    (root / "src" / "arl" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "arl" / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_target.py").write_text(
        "from arl.target import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    return root


def _run(
    monkeypatch,
    capsys,
    root: Path,
    old: str = "VALUE = 1",
    new: str = "VALUE = 2",
    *,
    argv: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
    suite: tuple[str, ...] = ("tests/test_target.py",),
) -> tuple[int, str]:
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "TESTS", list(suite))
    mutant = module.Mutant("valor", "src/arl/target.py", old, new, tests)
    monkeypatch.setattr(module, "MUTANTS", [mutant])
    monkeypatch.setattr(sys, "argv", ["mutation_check.py", *argv])
    code = module.main()
    return code, capsys.readouterr().out


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_mutant_is_killed_and_the_working_tree_is_untouched(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    before = _tree_hash(fake_repo)
    code, out = _run(monkeypatch, capsys, fake_repo)
    assert code == 0
    assert "Línea base OK" in out and "KILLED" in out
    assert _tree_hash(fake_repo) == before


def test_broken_baseline_aborts_before_applying_any_mutant(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    test = fake_repo / "tests" / "test_target.py"
    test.write_text(test.read_text(encoding="utf-8").replace("== 1", "== 99"), encoding="utf-8")
    before = _tree_hash(fake_repo)
    code, out = _run(monkeypatch, capsys, fake_repo)
    assert code == 2
    assert "ABORT" in out and "test_value" in out
    assert "KILLED" not in out and "SURVIVED" not in out  # ningún mutante llegó a ejecutarse
    assert _tree_hash(fake_repo) == before


def test_stale_fragment_aborts_instead_of_reporting_killed(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    code, out = _run(monkeypatch, capsys, fake_repo, old="VALUE = 7")
    assert code == 2
    assert "ABORT" in out and "KILLED" not in out


def test_surviving_mutant_fails_the_check(fake_repo: Path, monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, fake_repo, new="VALUE = 1  # sin cambio")
    assert code == 1
    assert "SURVIVED" in out


def test_mutant_that_does_not_compile_aborts_instead_of_counting_as_killed(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    code, out = _run(monkeypatch, capsys, fake_repo, new="VALUE = = 2")
    assert code == 2
    assert "no es Python válido" in out and "KILLED" not in out


# --- 6B.11: resultados incrementales, timeout, tests focalizados y estados ----------------------


def test_results_are_written_as_jsonl_with_the_run_configuration(
    fake_repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    results = tmp_path / "fuera" / "resultados.jsonl"
    code, _ = _run(monkeypatch, capsys, fake_repo, argv=("--results", str(results)))
    assert code == 0
    run, baseline, mutant, summary = _records(results)
    assert run["record"] == "run" and {"commit", "dirty", "python", "tests"} <= set(run)
    assert run["tests"] == ["tests/test_target.py"] and run["timeout_s"] == 900.0
    assert (baseline["record"], baseline["returncode"], baseline["timed_out"]) == (
        "baseline",
        0,
        False,
    )
    assert {k: mutant[k] for k in ("record", "name", "group", "status", "returncode")} == {
        "record": "mutant",
        "name": "valor",
        "group": "historical",
        "status": "KILLED",
        "returncode": 1,
    }
    assert mutant["killers"] == ["test_value"] and mutant["duration_s"] >= 0
    assert mutant["tests"] == ["tests/test_target.py"]
    assert (summary["record"], summary["KILLED"], summary["TIMEOUT"]) == ("summary", 1, 0)


def test_results_inside_the_repository_are_refused(fake_repo: Path, monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit, match="fuera del repositorio"):
        _run(monkeypatch, capsys, fake_repo, argv=("--results", str(fake_repo / "r.jsonl")))
    assert not (fake_repo / "r.jsonl").exists()


def test_a_timeout_is_never_detected_and_kills_the_whole_process_tree(
    fake_repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """El mutante cuelga la importación y deja un nieto vivo: al vencer el timeout se marca
    TIMEOUT (no KILLED) y no queda ningún proceso de esa corrida."""
    pid_file = tmp_path / "nieto.pid"
    hang = (
        "import pathlib, subprocess, sys, time\n"
        "_child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(_child.pid))\n"
        "time.sleep(120)\n"
        "VALUE = 1"
    )
    results = tmp_path / "r.jsonl"
    code, out = _run(
        monkeypatch, capsys, fake_repo, new=hang,
        argv=("--timeout", "8", "--results", str(results)),
    )  # fmt: skip
    assert code == 1
    assert "TIMEOUT" in out and "KILLED " not in out and "0/1 mutantes detectados" in out
    mutant = _records(results)[2]
    assert (mutant["status"], mutant["returncode"]) == ("TIMEOUT", None)
    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(pid)


@pytest.mark.parametrize(
    ("returncode", "killers", "status"),
    [
        (1, ["test_value"], "KILLED"),
        (2, ["test_value"], "KILLED"),  # error de colección causado por el mutante
        (0, [], "SURVIVED"),
        (None, [], "TIMEOUT"),
        (None, ["test_value"], "TIMEOUT"),  # un timeout nunca cuenta como detectado
        (1, [], "ERROR"),  # falla sin un test identificado
        (3, [], "ERROR"),  # error interno de pytest
        (4, [], "ERROR"),  # uso: p. ej. un test focalizado que no existe
        (5, [], "ERROR"),  # sin tests
    ],
)
def test_only_identified_test_failures_count_as_detection(
    monkeypatch, returncode, killers, status
) -> None:
    module = _load(monkeypatch)
    run = module.PytestRun(returncode=returncode, stdout="", duration_s=1.0)
    assert module.classify(run, killers) == status


def test_focused_tests_replace_the_full_list(
    fake_repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """``test_other`` está en ``TESTS`` y siempre falla: si la línea base o el mutante lo
    corrieran, abortaría. Con tests focalizados solo corre ``test_value``."""
    (fake_repo / "tests" / "test_other.py").write_text(
        "def test_other():\n    assert False\n", encoding="utf-8"
    )
    results = tmp_path / "r.jsonl"
    code, out = _run(
        monkeypatch, capsys, fake_repo,
        argv=("--results", str(results)),
        tests=("tests/test_target.py::test_value",),
        suite=("tests/test_target.py", "tests/test_other.py"),
    )  # fmt: skip
    assert code == 0 and "KILLED" in out
    run, _, mutant, _ = _records(results)
    assert run["baseline_tests"] == ["tests/test_target.py::test_value"]
    assert mutant["tests"] == ["tests/test_target.py::test_value"]


def test_focused_tests_outside_the_coverage_list_abort_before_copying(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    code, out = _run(monkeypatch, capsys, fake_repo, tests=("tests/test_nope.py",))
    assert code == 2
    assert "no está en TESTS" in out and "Línea base" not in out


def test_check_fragments_runs_no_pytest_and_applies_no_mutant(
    fake_repo: Path, monkeypatch, capsys
) -> None:
    module = _load(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("no debía correr pytest")

    monkeypatch.setattr(module, "run_pytest", forbidden)
    monkeypatch.setattr(module, "ROOT", fake_repo)
    monkeypatch.setattr(module, "TESTS", ["tests/test_target.py"])
    monkeypatch.setattr(
        module, "MUTANTS", [module.Mutant("valor", "src/arl/target.py", "VALUE = 1", "VALUE = 2")]
    )
    monkeypatch.setattr(sys, "argv", ["mutation_check.py", "--check-fragments"])
    before = _tree_hash(fake_repo)
    assert module.main() == 0
    assert "Catálogo OK: 1 mutantes" in capsys.readouterr().out
    assert _tree_hash(fake_repo) == before


def test_the_mutated_file_is_restored_even_if_the_run_raises(tmp_path: Path, monkeypatch) -> None:
    module = _load(monkeypatch)
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    mutant = module.Mutant("valor", "target.py", "VALUE = 1", "VALUE = 2")
    with pytest.raises(KeyboardInterrupt), module.mutated(target, mutant):
        assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
        raise KeyboardInterrupt
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_the_copy_excludes_secrets_reports_databases_and_transcripts(monkeypatch) -> None:
    module = _load(monkeypatch)
    names = [
        ".env", ".env.local", ".env.example", "key.pem", "tls.key", "app.db", "app.db-wal",
        "data.sqlite", "data.sqlite3", "transcript.jsonl", "live-reports", "live-transcripts",
        "main.py", "README.md",
    ]  # fmt: skip
    kept = set(names) - module.COPY_IGNORE("repo", names)
    assert kept == {".env.example", "main.py", "README.md"}


def test_pytest_never_inherits_credentials(monkeypatch, tmp_path: Path) -> None:
    module = _load(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_canary_" + "x" * 40)
    monkeypatch.setenv("SOME_TOKEN", "t")
    monkeypatch.setenv("ARL_PLAIN_SETTING", "ok")
    env = module.child_env(tmp_path)
    assert "GROQ_API_KEY" not in env and "SOME_TOKEN" not in env
    assert env["ARL_PLAIN_SETTING"] == "ok" and env["PYTHONPATH"] == str(tmp_path / "src")


def test_group_filter_lists_only_that_group(monkeypatch, capsys) -> None:
    module = _load(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["mutation_check.py", "--list", "--group", "P1"])
    assert module.main() == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("- ")]
    assert len(lines) == 16 and all(ln.startswith("- [P1] ") for ln in lines)


# --- El catálogo real (sin pytest ni mutaciones) -------------------------------------------------


def test_the_real_catalog_keeps_the_110_historical_mutants_and_adds_16_p1(monkeypatch) -> None:
    module = _load(monkeypatch)
    assert len(module.HISTORICAL) == 110 and len(module.P1) == 16
    assert module.MUTANTS == module.HISTORICAL + module.P1
    assert all(m.group == "historical" and m.tests == () for m in module.HISTORICAL)
    assert all(m.group == "P1" and m.tests for m in module.P1)


def test_every_fragment_is_unique_compiles_and_its_focused_tests_are_covered(monkeypatch) -> None:
    module = _load(monkeypatch)
    assert module.catalog_problems(module.ROOT, module.MUTANTS) == []
