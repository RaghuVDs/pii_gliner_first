"""
SQLAlchemy 2.0 async database setup for PostgreSQL.

Provides the async engine, session factory, declarative base,
and FastAPI-compatible dependency generator.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Globals (initialised lazily via init_db) ─────────────────────────────

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ── Lifecycle ────────────────────────────────────────────────────────────


async def init_db() -> None:
    """Create the async engine and session factory.

    Should be called once during application startup.
    """
    global _engine, _async_session_factory

    settings = get_settings()
    logger.info("Initialising PostgreSQL async engine: %s", settings.DATABASE_URL[:40] + "...")

    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info("PostgreSQL async engine initialised successfully.")


async def close_db() -> None:
    """Dispose of the engine and release all connections.

    Should be called during application shutdown.
    """
    global _engine, _async_session_factory

    if _engine is not None:
        logger.info("Closing PostgreSQL async engine...")
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("PostgreSQL async engine closed.")


# ── Dependency ───────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async SQLAlchemy session.

    The session is automatically closed when the request finishes.

    Raises:
        RuntimeError: If the database has not been initialised yet.
    """
    if _async_session_factory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during application startup."
        )

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_engine() -> AsyncEngine:
    """Return the current async engine (useful for Alembic migrations).

    Raises:
        RuntimeError: If the engine has not been created yet.
    """
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialised. Call init_db() first."
        )
    return _engine
