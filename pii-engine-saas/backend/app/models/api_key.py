"""
API access request, API key, and API usage log models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .tenant import Tenant
    from .user import User


class APIAccessRequest(Base):
    __tablename__ = "api_access_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    use_case: Mapped[str] = mapped_column(Text, nullable=False)
    scopes_requested: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[\"detect\",\"redact\"]'::jsonb"),
    )
    rate_limit_requested: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- relationships --------------------------------------------------------
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="api_access_requests")
    requester: Mapped["User"] = relationship(
        "User", foreign_keys=[requested_by], lazy="selectin",
    )
    reviewer: Mapped["User | None"] = relationship(
        "User", foreign_keys=[reviewed_by], lazy="selectin",
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey", back_populates="access_request", lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<APIAccessRequest status={self.status!r}>"


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    access_request_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_access_requests.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    scopes: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[\"detect\",\"redact\"]'::jsonb"),
    )
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("60"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_requests: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # -- relationships --------------------------------------------------------
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="api_keys")
    user: Mapped["User"] = relationship("User", lazy="selectin")
    access_request: Mapped["APIAccessRequest | None"] = relationship(
        "APIAccessRequest", back_populates="api_keys",
    )
    usage_logs: Mapped[List["APIUsageLog"]] = relationship(
        "APIUsageLog", back_populates="api_key", lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<APIKey prefix={self.key_prefix!r} name={self.name!r}>"


class APIUsageLog(Base):
    """High-volume append-only log. Uses BigInteger auto-increment PK."""

    __tablename__ = "api_usage_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pii_types_found: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    input_char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    # Override: usage logs only need created_at, no updated_at from Base.
    # We keep them because Base provides both; they are harmless for append-only rows.

    # -- relationships --------------------------------------------------------
    api_key: Mapped["APIKey"] = relationship("APIKey", back_populates="usage_logs")

    def __repr__(self) -> str:
        return f"<APIUsageLog endpoint={self.endpoint!r} status={self.status_code}>"
