"""
System and tenant context rule models.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import IntBase, SoftDeleteMixin

if TYPE_CHECKING:
    from .user import User


class ContextRule(IntBase):
    __tablename__ = "context_rules"

    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    keyword_pattern: Mapped[str] = mapped_column(Text, nullable=False)
    is_negative: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    source: Mapped[str] = mapped_column(String(30), nullable=False, server_default="manual")

    def __repr__(self) -> str:
        return f"<ContextRule pii_type_name={self.pii_type_name!r} is_negative={self.is_negative}>"


class TenantContextRule(SoftDeleteMixin, IntBase):
    __tablename__ = "tenant_context_rules"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    keyword_pattern: Mapped[str] = mapped_column(Text, nullable=False)
    is_negative: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    source: Mapped[str] = mapped_column(String(30), nullable=False, server_default="manual")
    promoted_from_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return (
            f"<TenantContextRule tenant_id={self.tenant_id!r} "
            f"pii_type_name={self.pii_type_name!r}>"
        )
