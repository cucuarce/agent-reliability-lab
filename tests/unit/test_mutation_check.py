"""``scripts/mutation_check.py``: exige una línea base limpia y nunca modifica el árbol de trabajo.

Se ejecuta el script real (copia, aislamiento, pytest en subproceso) sobre un repositorio mínimo
con un mutante propio, para que el test sea rápido.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
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
    monkeypatch, capsys, root: Path, old: str = "VALUE = 1", new: str = "VALUE = 2"
) -> tuple[int, str]:
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "TESTS", ["tests/test_target.py"])
    monkeypatch.setattr(module, "MUTANTS", [module.Mutant("valor", "src/arl/target.py", old, new)])
    monkeypatch.setattr(sys, "argv", ["mutation_check.py"])
    code = module.main()
    return code, capsys.readouterr().out


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
