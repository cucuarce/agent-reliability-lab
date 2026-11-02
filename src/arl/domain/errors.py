"""Errores del dominio. Sus mensajes son seguros para mostrar al LLM: no revelan datos ajenos."""

from __future__ import annotations


class DomainError(Exception):
    code = "domain_error"


class NotFound(DomainError):
    """El recurso no existe **o** el principal no tiene acceso: ambos casos son indistinguibles."""

    code = "not_found"

    def __init__(self, kind: str, resource_id: str) -> None:
        super().__init__(f"{kind} '{resource_id}' no encontrado")


class Forbidden(DomainError):
    """El principal ve el recurso pero su rol no permite la acción."""

    code = "forbidden"


class InvalidChange(DomainError):
    code = "invalid_change"


class VersionConflict(DomainError):
    code = "version_conflict"

    def __init__(self, task_id: str, expected: int, current: int) -> None:
        super().__init__(
            f"la tarea '{task_id}' está en la versión {current}, no en la {expected} esperada"
        )
        self.expected = expected
        self.current = current


class IdempotencyConflict(DomainError):
    code = "idempotency_conflict"

    def __init__(self, key: str) -> None:
        super().__init__(f"la clave de idempotencia '{key}' ya se usó con otros parámetros")
