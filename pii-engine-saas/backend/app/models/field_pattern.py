"""
System and tenant field pattern models.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    ForeignKey,
    SmallInteger,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import IntBase, SoftDeleteMixin

if TYPE_CHECKING:
    from .user import User


class FieldPattern(IntBase):
    __tablename__ = "field_patterns"

    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    label_pattern: Mapped[str] = mapped_column(String(255), nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))

    def __repr__(self) -> str:
        return f"<FieldPattern pii_type_name={self.pii_type_name!r} label_pattern={self.label_pattern!r}>"


class TenantFieldPattern(SoftDeleteMixin, IntBase):
    __tablename__ = "tenant_field_patterns"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "pii_type_name", "label_pattern",
            name="uq_tenant_field_pattern",
        ),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    label_pattern: Mapped[str] = mapped_column(String(255), nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return (
            f"<TenantFieldPattern tenant_id={self.tenant_id!r} "
            f"pii_type_name={self.pii_type_name!r}>"
        )
