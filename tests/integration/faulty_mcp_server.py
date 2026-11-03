"""Servidor MCP real (stdio) con un fallo inyectado del lado del servidor. Auxiliar de tests.

Lanza ``arl.mcp_server`` sin cambios, salvo que la PRIMERA llamada a
``ProjectService.update_task`` se altera según ``ARL_TEST_FAULT``:

- ``delay_before_write``: duerme ``ARL_TEST_DELAY_S`` antes de escribir (el cliente expira y la
  escritura todavía no ocurrió).
- ``delay_after_write``: escribe, confirma y luego duerme (el cliente expira, ya escrito).
- ``crash_after_write``: escribe, confirma y el proceso muere con ``os._exit`` (desconexión).
"""

from __future__ import annotations

import os
import sys
import time

from arl.mcp_server.__main__ import main
from arl.services.project_service import ProjectService

FAULT = os.environ["ARL_TEST_FAULT"]
DELAY_S = float(os.environ.get("ARL_TEST_DELAY_S", "2"))
_original = ProjectService.update_task
_calls = 0


def _faulty_update_task(self, *args, **kwargs):
    global _calls
    _calls += 1
    if _calls > 1:
        return _original(self, *args, **kwargs)
    if FAULT == "delay_before_write":
        time.sleep(DELAY_S)
        return _original(self, *args, **kwargs)
    result = _original(self, *args, **kwargs)
    if FAULT == "delay_after_write":
        time.sleep(DELAY_S)
    elif FAULT == "crash_after_write":
        os._exit(70)
    return result


ProjectService.update_task = _faulty_update_task  # type: ignore[method-assign]

if __name__ == "__main__":
    sys.exit(main())
