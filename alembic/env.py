"""Alembic environment configuration with async support"""

import asyncio
import os
import logging
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your models
from db.models import Base

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Set target metadata
target_metadata = Base.metadata


def get_database_url() -> str:
    """Get database URL from environment with asyncpg driver"""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    # Use DIRECT URL for migrations (avoids pooler issues)
    db_url = os.getenv("DIRECT_DATABASE_URL")

    if not db_url:
        raise ValueError("DIRECT_DATABASE_URL not found in .env")

    # Parse URL to handle query parameters properly
    parsed = urlparse(db_url)

    # Ensure asyncpg driver is specified for async operations
    if parsed.scheme == "postgresql":
        # Replace scheme with asyncpg driver
        parsed = parsed._replace(scheme="postgresql+asyncpg")
    elif parsed.scheme != "postgresql+asyncpg":
        raise ValueError("Database URL must use postgresql:// scheme")

    # Remove sslmode from query parameters (asyncpg handles SSL differently)
    if parsed.query:
        query_params = parse_qs(parsed.query)
        query_params.pop("sslmode", None)  # Remove sslmode
        query_params.pop("channel_binding", None)  # Remove channel_binding
        new_query = urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)

    # Reconstruct URL
    final_url = urlunparse(parsed)

    logger.info(f"🔗 Original URL scheme: {parsed.scheme}")
    logger.info(f"🔗 Final URL: {final_url[:60]}...")

    # Verify the URL has asyncpg driver
    if "+asyncpg" not in final_url:
        raise ValueError(f"Failed to add asyncpg driver to URL: {final_url}")

    return final_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a connection"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect default value changes
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode with proper timeouts for Neon scale-to-zero"""

    # Get database URL
    database_url = get_database_url()

    # Create config for async engine with extended timeouts
    configuration = {
        "sqlalchemy.url": database_url,
    }

    logger.info("🏗️ Creating engine with extended timeouts for Neon cold starts...")

    # Create async engine with extended timeouts for Neon cold starts
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # ============================================================
        # CRITICAL: Extended timeouts for Neon scale-to-zero wakeup
        # ============================================================
        connect_args={
            "timeout": 60,  # Overall connection timeout (seconds)
            "command_timeout": 60,  # Command execution timeout (seconds)
            # SSL configuration for Neon (required in production)
            "ssl": "require",  # Require SSL connection
            "server_settings": {
                "statement_timeout": "60000",  # PostgreSQL statement timeout (ms)
            },
        },
    )

    logger.info("� Attemptinng connection (may take up to 60s for Neon cold start)...")

    async with connectable.connect() as connection:
        logger.info("✅ Connected to database, running migrations...")
        await connection.run_sync(do_run_migrations)
        logger.info("✅ Migrations completed successfully")

    await connectable.dispose()
    logger.info("🔚 Engine disposed")


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    logger.info("🔧 Starting async migrations...")
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
