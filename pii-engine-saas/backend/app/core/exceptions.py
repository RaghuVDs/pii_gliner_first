"""
Custom exception hierarchy for the PII Engine SaaS platform.

All domain-specific exceptions inherit from PIIEngineException so they
can be caught uniformly by the global exception handlers in main.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PIIEngineException(Exception):
    """Base exception for all PII Engine errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred."
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detail = detail or self.__class__.detail
        self.error_code = error_code or self.__class__.error_code
        self.extra = extra or {}
        super().__init__(self.detail)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the exception for JSON responses."""
        payload: Dict[str, Any] = {
            "error_code": self.error_code,
            "detail": self.detail,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


# ── 4xx Client Errors ────────────────────────────────────────────────────


class NotFoundError(PIIEngineException):
    """Requested resource does not exist."""

    status_code = 404
    detail = "Resource not found."
    error_code = "NOT_FOUND"


class ForbiddenError(PIIEngineException):
    """Caller lacks permission for the requested action."""

    status_code = 403
    detail = "You do not have permission to perform this action."
    error_code = "FORBIDDEN"


class UnauthorizedError(PIIEngineException):
    """Authentication credentials are missing or invalid."""

    status_code = 401
    detail = "Authentication required."
    error_code = "UNAUTHORIZED"


class ValidationError(PIIEngineException):
    """Request payload failed domain validation."""

    status_code = 422
    detail = "Validation error."
    error_code = "VALIDATION_ERROR"


class ConflictError(PIIEngineException):
    """Operation conflicts with existing state (e.g. duplicate resource)."""

    status_code = 409
    detail = "Resource conflict."
    error_code = "CONFLICT"


class RateLimitExceeded(PIIEngineException):
    """Caller has exceeded their allowed request rate."""

    status_code = 429
    detail = "Rate limit exceeded. Please retry later."
    error_code = "RATE_LIMIT_EXCEEDED"


class TenantQuotaExceeded(PIIEngineException):
    """Tenant has exhausted their usage quota for the billing period."""

    status_code = 429
    detail = "Tenant quota exceeded for the current billing period."
    error_code = "TENANT_QUOTA_EXCEEDED"
