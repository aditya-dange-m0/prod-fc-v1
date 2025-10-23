# db/session.py - Production-Ready Database Session Management
"""
Production database connection and SQLAlchemy session factory management
with comprehensive error handling, monitoring, and resilience patterns.

NOTE: This file is for DATABASE SESSIONS (SQLAlchemy), 
NOT for application session models (those are now part of Project).
"""
import asyncio
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from sqlalchemy.pool import NullPool, Pool
from sqlalchemy.exc import OperationalError, TimeoutError as SQLAlchemyTimeoutError
from .config import get_db_settings

# Configure logging
logger = logging.getLogger(__name__)


# Global engine and session factory (singleton pattern)
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None
_init_lock = asyncio.Lock()


async def init_db(use_direct: bool = False) -> AsyncEngine:
    """
    Initialize database engine and session factory with production settings.
    Does NOT create tables - use Alembic migrations instead.
    
    Args:
        use_direct: If True, use direct connection (port 5432, for migrations).
                   If False, use pooled connection (port 6543 via Supavisor, default).
    
    Returns:
        AsyncEngine: Initialized database engine
    
    Raises:
        ValueError: If database configuration is invalid
        OperationalError: If connection to database fails
    """
    global _engine, _session_factory
    
    # Thread-safe initialization with async lock
    async with _init_lock:
        # Return existing engine if already initialized (singleton)
        if _engine is not None:
            logger.info("✓ Database already initialized, reusing existing engine")
            return _engine
        
        logger.info("INFO: Initializing database connection...")
        settings = get_db_settings()
        
        # Log masked configuration for debugging (never log passwords!)
        masked_config = settings.mask_sensitive_data()
        logger.info(f"Environment: {masked_config['ENV']}")
        logger.info(f"Connection mode: {'direct (port 5432)' if use_direct else 'pooled via Supavisor (port 6543)'}")
        
        try:
            # Build connection URL with SSL and timeout parameters
            connection_url = settings.get_connection_url(use_direct=use_direct)
            logger.info("✓ Connection URL built successfully")
        except ValueError as e:
            logger.error(f"Failed to build connection URL: {e}")
            raise
        
        # Supabase/asyncpg-specific connection arguments
        # CRITICAL: These settings prevent prepared statement conflicts with Supavisor
        connect_args = {
            # Disable prepared statement caching
            # Supavisor recycles connections across clients, causing statement name collisions
            # Without this, you'll get "prepared statement already exists" errors
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            
            # Generate unique prepared statement names to prevent conflicts
            # Each statement gets a UUID, ensuring no collisions even with connection reuse
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex[:16]}__",
            
            # Server settings (passed to PostgreSQL during connection)
            "server_settings": {
                # Set application name for connection tracking in pg_stat_activity
                "application_name": f"app_{settings.ENV}",
                # Set statement timeout at connection level (milliseconds)
                "statement_timeout": str(settings.COMMAND_TIMEOUT * 1000),
            },
        }
        
        # Connection pool strategy:
        # - For Supavisor (pooled): Use NullPool - no SQLAlchemy pooling since Supavisor pools
        # - For direct: Use default QueuePool with configured pool settings
        poolclass = NullPool if not use_direct else None
        
        # Build engine configuration
        engine_kwargs = {
            "echo": settings.ECHO_SQL,  # Log SQL queries (disable in production)
            "echo_pool": settings.ECHO_POOL,  # Log pool activity (debugging only)
            "pool_pre_ping": settings.POOL_PRE_PING,  # Validate connections before use
            "poolclass": poolclass,  # NullPool for Supavisor, QueuePool for direct
            "connect_args": connect_args,  # asyncpg-specific arguments
        }
        
        # Add pool configuration ONLY for direct connections
        # Supavisor manages pooling, so SQLAlchemy pooling is disabled (NullPool)
        if use_direct:
            engine_kwargs.update({
                "pool_size": settings.POOL_SIZE,  # Connections to maintain
                "max_overflow": settings.MAX_OVERFLOW,  # Additional connections on demand
                "pool_timeout": settings.POOL_TIMEOUT,  # Wait time for connection from pool
                "pool_recycle": settings.POOL_RECYCLE,  # Recycle connections (prevent stale)
            })
        
        try:
            # Create async engine
            _engine = create_async_engine(connection_url, **engine_kwargs)
            logger.info("✓ Database engine created successfully")
            
            # Test connection
            from sqlalchemy import text
            async with _engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("✓ Database connection test successful")
            
        except OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Check your connection parameters, network, and database availability")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during engine creation: {e}")
            raise
        
        # Create session factory with optimized settings
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects accessible after commit (no re-query)
            autocommit=False,  # Manual transaction control
            autoflush=False,  # Manual flush control (performance optimization)
        )
        logger.info("✓ Session factory created successfully")
        
        # Reminder for developers
        if settings.ENV == "development":
            logger.warning("⚠️  Remember to run migrations: alembic upgrade head")
        
        return _engine


async def get_engine() -> AsyncEngine:
    """
    Get or lazily initialize database engine.
    
    Returns:
        AsyncEngine: Database engine instance
    """
    if _engine is None:
        await init_db()
    return _engine


def get_session_factory() -> async_sessionmaker:
    """
    Get session factory (engine must be initialized first).
    
    Returns:
        async_sessionmaker: Session factory for creating database sessions
    
    Raises:
        RuntimeError: If database not initialized
    """
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialized. Call init_db() first, "
            "or use get_db_session() which initializes automatically."
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Production context manager for DATABASE sessions (SQLAlchemy).
    NOT related to application sessions (those are now Project records).
    
    Provides automatic transaction management:
    - Commits on success
    - Rolls back on exceptions
    - Always closes session to release connections
    
    Use this in your tools for database operations.
    
    Example:
        async with get_db_session() as session:
            repo = FileRepository(session)
            file = await repo.save_project_file(...)
            # Automatically commits here if no exception
    
    Yields:
        AsyncSession: Database session for queries and transactions
    
    Raises:
        OperationalError: If database connection fails
        SQLAlchemyTimeoutError: If query exceeds timeout
    """
    # Lazy initialization - ensure database is ready
    if _session_factory is None:
        await init_db()
    
    # Create new session from factory
    async with _session_factory() as session:
        try:
            # Yield session for use in 'async with' block
            yield session
            
            # If no exception occurred, commit all pending changes
            await session.commit()
            logger.debug("Database session committed successfully")
            
        except (OperationalError, SQLAlchemyTimeoutError) as e:
            # Database-specific errors (connection, timeout, etc.)
            await session.rollback()
            logger.error(f"Database error, transaction rolled back: {type(e).__name__}: {e}")
            raise
            
        except Exception as e:
            # Any other exception - rollback to maintain data integrity
            await session.rollback()
            logger.error(f"Unexpected error, transaction rolled back: {type(e).__name__}: {e}")
            raise
            
        finally:
            # Always close session to release connection back to pool
            # Critical for preventing connection leaks
            await session.close()
            logger.debug("Database session closed")


async def health_check() -> bool:
    """
    Check database connectivity for health monitoring.
    
    Use this in your application startup or health check endpoints
    to verify database is accessible.
    
    Returns:
        bool: True if database is healthy, False otherwise
    
    Example:
        if not await health_check():
            logger.error("Database health check failed!")
    """
    try:
        if _engine is None:
            await init_db()
        
        # Simple connectivity test
        from sqlalchemy import text
        async with _engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
        
        logger.info("✓ Database health check passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database health check failed: {e}")
        return False


async def close_db():
    """
    Gracefully close database connections and cleanup resources.
    
    Call this during application shutdown to ensure clean termination.
    
    Example:
        @app.on_event("shutdown")
        async def shutdown():
            await close_db()
    """
    global _engine, _session_factory
    
    if _engine:
        logger.info("Closing database connections...")
        
        try:
            # Dispose of engine - closes all pooled connections
            await _engine.dispose()
            logger.info("✓ Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
        finally:
            # Reset globals to allow re-initialization if needed
            _engine = None
            _session_factory = None


async def get_pool_status() -> dict:
    """
    Get connection pool status for monitoring.
    
    Returns:
        dict: Pool statistics including size, checked_out connections, etc.
        Returns empty dict if using NullPool (Supavisor mode)
    
    Example:
        status = await get_pool_status()
        logger.info(f"Pool size: {status.get('size', 'N/A')}")
    """
    if _engine is None:
        return {"status": "not_initialized"}
    
    pool: Pool = _engine.pool
    
    # NullPool doesn't have these attributes
    if isinstance(pool, NullPool):
        return {
            "type": "NullPool",
            "note": "Using Supavisor connection pooling, SQLAlchemy pool disabled"
        }
    
    try:
        return {
            "type": pool.__class__.__name__,
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
        }
    except AttributeError:
        return {"type": pool.__class__.__name__, "status": "metrics_unavailable"}
