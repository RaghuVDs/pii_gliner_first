"""
ML model version tracking model.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import IntBase


class MLModelVersion(IntBase):
    __tablename__ = "ml_model_versions"
    __table_args__ = (
        UniqueConstraint("tenant_id", "version", name="uq_ml_model_tenant_version"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    model_bucket_path: Mapped[str] = mapped_column(String(500), nullable=False)
    vocab_bucket_path: Mapped[str] = mapped_column(String(500), nullable=False)
    num_labels: Mapped[int] = mapped_column(Integer, nullable=False)
    num_examples: Mapped[int] = mapped_column(Integer, nullable=False)
    train_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    val_weighted_f1: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    metrics_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    training_trigger: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="auto",
    )
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<MLModelVersion tenant_id={self.tenant_id!r} "
            f"version={self.version} is_active={self.is_active}>"
        )
