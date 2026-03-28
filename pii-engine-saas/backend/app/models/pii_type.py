"""
PII category, PII type, tenant PII config, and custom PII type models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import IntBase, SoftDeleteMixin

if TYPE_CHECKING:
    from .tenant import Tenant
    from .user import User


class PIICategory(IntBase):
    __tablename__ = "pii_categories"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    # -- relationships --------------------------------------------------------
    pii_types: Mapped[List["PIIType"]] = relationship(
        "PIIType", back_populates="category", lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<PIICategory name={self.name!r} tier={self.tier}>"


class PIIType(IntBase):
    __tablename__ = "pii_types"

    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("pii_categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    gliner_aliases: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[]'::jsonb"),
    )
    default_threshold: Mapped[float] = mapped_column(
        Numeric(4, 3), nullable=False, server_default=text("0.400"),
    )
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    is_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    compliance_tags: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[]'::jsonb"),
    )

    # -- relationships --------------------------------------------------------
    category: Mapped["PIICategory"] = relationship("PIICategory", back_populates="pii_types")
    tenant_configs: Mapped[List["TenantPIIConfig"]] = relationship(
        "TenantPIIConfig", back_populates="pii_type", lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<PIIType name={self.name!r}>"


class TenantPIIConfig(IntBase):
    __tablename__ = "tenant_pii_configs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "pii_type_id", name="uq_tenant_pii_config"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pii_type_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("pii_types.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    custom_threshold: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    custom_aliases: Mapped[list | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="pii_configs")
    pii_type: Mapped["PIIType"] = relationship("PIIType", back_populates="tenant_configs")
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return f"<TenantPIIConfig tenant_id={self.tenant_id!r} pii_type_id={self.pii_type_id}>"


class CustomPIIType(SoftDeleteMixin, IntBase):
    __tablename__ = "custom_pii_types"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_custom_pii_type_tenant_name"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("pii_categories.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    gliner_aliases: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[]'::jsonb"),
    )
    default_threshold: Mapped[float] = mapped_column(
        Numeric(4, 3), nullable=False, server_default=text("0.400"),
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    compliance_tags: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[]'::jsonb"),
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    category: Mapped["PIICategory | None"] = relationship("PIICategory", lazy="selectin")
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return f"<CustomPIIType name={self.name!r} tenant_id={self.tenant_id!r}>"
