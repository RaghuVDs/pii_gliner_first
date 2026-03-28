"""
Alembic async environment configuration.

Imports all SQLAlchemy ORM models so that autogenerate can detect
schema changes, and runs migrations inside an async database context.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Ensure the backend directory is on sys.path so app.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import every model so that Base.metadata is fully populated.
# ---------------------------------------------------------------------------
from app.models import (  # noqa: F401 -- side-effect imports
    APIAccessRequest,
    APIKey,
    APIUsageLog,
    ContextRule,
    CustomPIIType,
    FieldPattern,
    MaskingRule,
    MaskingStrategy,
    MLModelVersion,
    Notification,
    PIICategory,
    PIIType,
    RegexRule,
    TenantContextRule,
    TenantFieldPattern,
    TenantMaskingRule,
    TenantPIIConfig,
    TenantRegexRule,
    Tenant,
    User,
    UserSession,
)
from app.models.base import shared_metadata

# ---------------------------------------------------------------------------
# Alembic Config object -- provides access to alembic.ini values.
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging (unless we are called
# programmatically without a config file).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the shared MetaData so that autogenerate sees all models.
target_metadata = shared_metadata

# Override the sqlalchemy.url from the environment variable if available.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Alembic needs the asyncpg driver prefix for async migrations.
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    config.set_main_option("sqlalchemy.url", DATABASE_URL)


# ---------------------------------------------------------------------------
# Offline mode -- generate SQL script without a live database connection.
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.  Calls to
    context.execute() emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online (async) mode -- run migrations against a live database.
# ---------------------------------------------------------------------------


def do_run_migrations(connection) -> None:
    """Helper that configures the context and runs migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine, connect, and run migrations."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
