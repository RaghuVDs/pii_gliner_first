"""
System and tenant regex rule models.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import IntBase, SoftDeleteMixin

if TYPE_CHECKING:
    from .user import User


class RegexRule(IntBase):
    __tablename__ = "regex_rules"

    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    # -- relationships --------------------------------------------------------
    tenant_overrides: Mapped[list["TenantRegexRule"]] = relationship(
        "TenantRegexRule", back_populates="system_rule", lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<RegexRule pii_type_name={self.pii_type_name!r}>"


class TenantRegexRule(SoftDeleteMixin, IntBase):
    __tablename__ = "tenant_regex_rules"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pii_type_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    is_system_override: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    system_rule_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("regex_rules.id", ondelete="SET NULL"),
        nullable=True,
    )
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -- relationships --------------------------------------------------------
    system_rule: Mapped["RegexRule | None"] = relationship(
        "RegexRule", back_populates="tenant_overrides",
    )
    creator: Mapped["User | None"] = relationship("User", lazy="selectin")

    def __repr__(self) -> str:
        return f"<TenantRegexRule tenant_id={self.tenant_id!r} pii_type_name={self.pii_type_name!r}>"
