from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Backend de los tests ``@pytest.mark.anyio`` (el SDK MCP usa anyio)."""
    return "asyncio"
