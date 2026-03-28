"""
Masking strategy, system masking rule, and tenant masking rule models.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, List

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import IntBase

if TYPE_CHECKING:
    from .user import User


class MaskingStrategy(IntBase):
    __tablename__ = "masking_strategies"

    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    implementation_key: Mapped[str] = mapped_column(String(100), nullable=False)

    # -- relationships --------------------------------------------------------
    masking_rules: Mapped[List["MaskingRule"]] = relationship(
        "MaskingRule", back_populates="strategy", lazy="selectin",
    )
    tenant_masking_rules: Mapped[List["TenantMaskingRule"]] = relationship(
        "TenantMaskingRule", back_populates="strategy", lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<MaskingStrategy name={self.name!r}>"


class MaskingRule(IntBase):
    __tablename__ = "masking_rules"

    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    strategy_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("masking_strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))

    # -- relationships --------------------------------------------------------
    strategy: Mapped["MaskingStrategy"] = relationship(
        "MaskingStrategy", back_populates="masking_rules",
    )

    def __repr__(self) -> str:
        return f"<MaskingRule pii_type_name={self.pii_type_name!r} strategy_id={self.strategy_id}>"


class TenantMaskingRule(IntBase):
    __tablename__ = "tenant_masking_rules"
    __table_args__ = (
        UniqueConstraint("tenant_id", "pii_type_name", name="uq_tenant_masking_rule"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    strategy_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("masking_strategies.id", ondelete="CASCADE"),
        nullable=False,
    )
    custom_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    strategy: Mapped["MaskingStrategy"] = relationship(
        "MaskingStrategy", back_populates="tenant_masking_rules",
    )
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return (
            f"<TenantMaskingRule tenant_id={self.tenant_id!r} "
            f"pii_type_name={self.pii_type_name!r}>"
        )
