"""
Base model class and mixins for all SQLAlchemy ORM models.

Two base classes are provided:

- ``Base``     -- UUID primary key (gen_random_uuid)
- ``IntBase``  -- Auto-increment integer primary key

Both share the **same** MetaData and registry so that ForeignKey
references across UUID-keyed and integer-keyed tables resolve correctly
(e.g. ``TenantPIIConfig.tenant_id -> tenants.id``).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, MetaData, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
    registry,
)

# Shared convention for constraint naming (helps Alembic auto-generate).
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Single shared MetaData instance.
shared_metadata = MetaData(naming_convention=convention)

# Single shared registry so relationships across bases work.
shared_registry = registry()


class Base(DeclarativeBase):
    """Declarative base for models with UUID primary keys."""

    metadata = shared_metadata
    registry = shared_registry

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class IntBase(DeclarativeBase):
    """Declarative base for models with auto-increment integer primary keys."""

    metadata = shared_metadata
    registry = shared_registry

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SoftDeleteMixin:
    """Mixin that adds a ``deleted_at`` column for soft-delete support."""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
