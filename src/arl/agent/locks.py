"""Lock por thread entre procesos: una sola ejecución a la vez por thread.

LangGraph no serializa ejecuciones concurrentes del mismo thread entre procesos: dos procesos que
reanudan a la vez bifurcan los checkpoints y pueden leer estados intermedios del otro (verificado
con procesos reales en ``test_agent_faults_process.py``). La idempotencia del servidor evita
escrituras duplicadas, pero el estado informado era incorrecto.

Se usa un lock de archivo del sistema operativo (``msvcrt`` en Windows, ``fcntl`` en POSIX):
es no bloqueante (un segundo proceso recibe ``ThreadBusy``) y el sistema lo libera solo si el
proceso muere, así que un crash o un kill nunca dejan un lock huérfano.

El lock es obligatorio: ``build_graph`` y cada función del runtime fallan con
``LockingRequired`` si falta ``lock_dir``, en vez de ejecutar sin exclusión mutua. Es local a una
máquina: procesos en máquinas distintas (o un sistema de archivos de red) no se excluyen.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class ThreadBusy(RuntimeError):
    """Otro proceso (o tarea) está ejecutando este thread en este momento."""


class LockingRequired(ValueError):
    """Un runtime con checkpoints persistentes no puede ejecutarse sin locks por thread."""


if os.name == "nt":
    import msvcrt

    def _try_lock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lock_path(lock_dir: Path, thread_id: str) -> Path:
    """Nombre derivado por hash: el thread_id nunca se usa como ruta."""
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:32]
    return lock_dir / f"thread-{digest}.lock"


@contextmanager
def thread_lock(lock_dir: Path, thread_id: str) -> Iterator[None]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_path(lock_dir, thread_id), "a+b") as handle:
        try:
            _try_lock(handle)
        except OSError:
            raise ThreadBusy("el thread se está ejecutando en otro proceso") from None
        try:
            yield
        finally:
            _unlock(handle)
