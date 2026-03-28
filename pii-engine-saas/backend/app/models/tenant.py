"""
Tenant model -- represents an organisation / workspace in the multi-tenant
PII detection platform.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, SoftDeleteMixin

if TYPE_CHECKING:
    from .api_key import APIAccessRequest, APIKey
    from .user import User
    from .pii_type import TenantPIIConfig


class Tenant(SoftDeleteMixin, Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    plan: Mapped[str] = mapped_column(String(50), nullable=False, server_default="free")
    settings: Mapped[dict] = mapped_column(JSON, nullable=False, server_default=text("'{}'::jsonb"))
    max_users: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("5"))
    max_api_keys: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("3"))
    max_monthly_scans: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1000"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    onboarded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # -- relationships --------------------------------------------------------
    users: Mapped[List["User"]] = relationship(
        "User", back_populates="tenant", lazy="selectin",
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey", back_populates="tenant", lazy="selectin",
    )
    api_access_requests: Mapped[List["APIAccessRequest"]] = relationship(
        "APIAccessRequest", back_populates="tenant", lazy="selectin",
    )
    pii_configs: Mapped[List["TenantPIIConfig"]] = relationship(
        "TenantPIIConfig", back_populates="tenant", lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Tenant slug={self.slug!r} plan={self.plan!r}>"
