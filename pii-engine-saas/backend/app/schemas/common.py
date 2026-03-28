"""Common reusable schemas for the PII detection platform."""

from __future__ import annotations

import math
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic paginated wrapper
# ---------------------------------------------------------------------------
class PaginatedResponse(BaseModel, Generic[T]):
    """Envelope for any list endpoint that supports pagination."""

    items: list[T]
    total: int = Field(..., ge=0, description="Total number of records matching the query")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=200, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Computed total pages")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "items": [],
                "total": 0,
                "page": 1,
                "page_size": 20,
                "total_pages": 0,
            }
        },
    )

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Factory that auto-computes *total_pages*."""
        total_pages = math.ceil(total / page_size) if page_size else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


# ---------------------------------------------------------------------------
# Standard error / success envelopes
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Standard error payload returned by the API."""

    detail: str = Field(..., description="Human-readable error message")
    error_code: str = Field(
        ...,
        description="Machine-readable error code (e.g. 'INVALID_CREDENTIALS')",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid email or password.",
                "error_code": "INVALID_CREDENTIALS",
            }
        },
    )


class SuccessResponse(BaseModel):
    """Generic success acknowledgement."""

    message: str = Field(..., description="Human-readable success message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "Operation completed successfully."}
        },
    )


# ---------------------------------------------------------------------------
# UUID base model
# ---------------------------------------------------------------------------
class UUIDModel(BaseModel):
    """Base model that carries a UUID primary key."""

    id: UUID

    model_config = ConfigDict(from_attributes=True)
