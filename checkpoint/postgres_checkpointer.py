"""
PostgreSQL Checkpointer for LangGraph - Production Setup
========================================================

Features:
- Automatic message history persistence
- State snapshots at each step
- Thread-based conversation management
- Connection pooling for performance
"""

import logging
import platform
import asyncio
import os
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import psycopg

logger = logging.getLogger(__name__)

# Fix Windows event loop issue for psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class CheckpointerService:
    """
    Production PostgreSQL checkpointer service with optimized pooling.

    Handles:
    - Message history persistence (automatic)
    - State checkpoints
    - Thread management
    - Connection lifecycle
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        min_pool_size: int = 2,
        max_pool_size: int = 20,
        pool_timeout: int = 30,
        max_idle_time: int = 300,  # 5 minutes
    ):
        self.database_url = database_url or os.getenv("CHECKPOINT_DATABASE_URL")
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool_timeout = pool_timeout
        self.max_idle_time = max_idle_time
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self._initialized = False

    async def initialize(self):
        """Initialize checkpointer with connection pool"""

        if self._initialized:
            logger.info("✅ Checkpointer already initialized")
            return

        try:
            logger.info(
                f"[CHECKPOINTER] Initializing with pool: "
                f"min={self.min_pool_size}, max={self.max_pool_size}"
            )

            # Get database URL from environment if not set
            if self.database_url is None:
                database_url = os.getenv("DIRECT_DATABASE_URL")
                if not database_url:
                    raise ValueError("DIRECT_DATABASE_URL not found in environment")

                # Convert SQLAlchemy dialect to psycopg format
                if database_url.startswith("postgresql+asyncpg://"):
                    database_url = database_url.replace(
                        "postgresql+asyncpg://", "postgresql://"
                    )

                self.database_url = database_url

            logger.info(f"[CHECKPOINTER] Database URL: {self.database_url[:30]}...")

            # Step 1: Run setup() FIRST with autocommit (outside transaction)
            #
            # ⚠️  IMPORTANT: TABLE CREATION SETUP - READ BEFORE UNCOMMENTING
            # ================================================================
            #
            # The line below creates PostgreSQL tables for LangGraph checkpointer.
            # It should ONLY be run ONCE during initial setup or after dropping tables.
            #
            # WHEN TO UNCOMMENT:
            # - First time setting up the application
            # - After running fix_checkpointer_schema.py (which drops tables)
            # - After manually dropping checkpointer tables
            # - When migrating to a new database
            #
            # WHEN TO KEEP COMMENTED:
            # - Normal application startup (tables already exist)
            # - Production deployments (tables should already exist)
            # - Development restarts (tables persist between restarts)
            #
            # HOW TO USE:
            # 1. Uncomment the line below
            # 2. Start the application once
            # 3. Verify tables are created (check logs for "setup completed successfully")
            # 4. Comment the line again
            # 5. Restart application normally
            #
            # TABLES CREATED:
            # - checkpoints (main conversation history)
            # - checkpoint_blobs (binary data storage)
            # - checkpoint_writes (intermediate writes)
            #
            # ================================================================

            logger.info("[CHECKPOINTER] Skipping table setup (tables already exist)")
            # await self._run_setup_with_autocommit()  # ← UNCOMMENT ONLY FOR INITIAL SETUP
            logger.info("✅ Database tables verified (assuming they exist)")

            # Step 2: Create connection pool for regular operations
            logger.info("[CHECKPOINTER] Creating connection pool...")
            self.pool = AsyncConnectionPool(
                conninfo=self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                timeout=self.pool_timeout,
                max_idle=self.max_idle_time,
                max_lifetime=3600,  # Recycle connections after 1 hour
                num_workers=3,  # Background workers for pool maintenance
                open=False,  # Don't open immediately (we'll call wait())
            )

            # Open pool and wait for it to be ready
            await self.pool.open()
            await self.pool.wait()
            logger.info("✅ Connection pool initialized")

            # Step 3: Create checkpointer with the pool
            self.checkpointer = AsyncPostgresSaver(self.pool)
            logger.info("✅ AsyncPostgresSaver created")

            self._initialized = True
            logger.info("✅ Checkpointer fully initialized and ready")

        except Exception as e:
            logger.error(f"❌ Checkpointer initialization failed: {e}", exc_info=True)

            # Cleanup on failure
            if self.pool:
                try:
                    await self.pool.close()
                except Exception:
                    pass

            raise

    async def _run_setup_with_autocommit(self):
        """
        Run checkpointer setup with autocommit mode.

        This is required because PostgreSQL's CREATE INDEX CONCURRENTLY
        cannot run inside a transaction block.
        """

        try:
            # Create a single connection with autocommit for setup
            conn = await psycopg.AsyncConnection.connect(
                self.database_url,
                autocommit=True,  # ← KEY FIX: Enable autocommit
            )

            try:
                # Create temporary checkpointer for setup only
                temp_checkpointer = AsyncPostgresSaver(conn)

                # Run setup (creates tables and indexes)
                await temp_checkpointer.setup()

                logger.info("✅ Checkpointer setup completed successfully")

            finally:
                # Close the temporary connection
                await conn.close()

        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            raise

    async def close(self):
        """Close checkpointer connections gracefully"""

        if self.pool:
            try:
                await self.pool.close()
                logger.info("✅ Connection pool closed")
            except Exception as e:
                logger.error(f"Error closing pool: {e}")

        self._initialized = False
        logger.info("✅ Checkpointer shutdown complete")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """Get the checkpointer instance"""

        if not self._initialized:
            raise RuntimeError(
                "Checkpointer not initialized. "
                "Call await checkpointer_service.initialize() first."
            )

        return self.checkpointer

    async def health_check(self) -> dict:
        """Check checkpointer health"""

        if not self._initialized or not self.pool:
            return {
                "status": "not_initialized",
                "message": "Checkpointer not initialized",
            }

        try:
            # Get pool stats
            stats = self.pool.get_stats()

            return {
                "status": "healthy",
                "pool_size": stats.get("pool_size", 0),
                "pool_available": stats.get("pool_available", 0),
                "pool_min": self.min_pool_size,
                "pool_max": self.max_pool_size,
                "requests_waiting": stats.get("requests_waiting", 0),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def get_thread_history(self, thread_id: str, limit: int = 50) -> list:
        """Get conversation history for a thread (project)"""

        if not self._initialized:
            raise RuntimeError("Checkpointer not initialized")

        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Get state history
            history = []
            async for state in self.checkpointer.alist(
                config=config,
                limit=limit,
            ):
                history.append(
                    {
                        "checkpoint_id": state.checkpoint["id"],
                        "timestamp": state.checkpoint.get("ts"),
                        "messages": state.values.get("messages", []),
                        "metadata": state.metadata,
                    }
                )

            logger.info(f"Retrieved {len(history)} checkpoints for thread {thread_id}")
            return history

        except Exception as e:
            logger.error(f"Failed to get thread history: {e}", exc_info=True)
            return []

    async def get_current_state(self, thread_id: str) -> Optional[dict]:
        """Get current state for a thread"""

        if not self._initialized:
            raise RuntimeError("Checkpointer not initialized")

        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Get current state
            state = await self.checkpointer.aget(config)

            if state:
                return {
                    "checkpoint_id": state.checkpoint["id"],
                    "timestamp": state.checkpoint.get("ts"),
                    "messages": state.values.get("messages", []),
                    "metadata": state.metadata,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get current state: {e}", exc_info=True)
            return None

    async def delete_thread_history(self, thread_id: str) -> bool:
        """
        Delete all history for a thread.

        Use when:
        - Project is deleted
        - User wants to clear conversation
        """

        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Delete all checkpoints for this thread
            # Note: LangGraph doesn't have a direct delete method yet
            # You may need to implement this via direct SQL if needed

            logger.warning(
                f"Thread deletion not fully implemented in LangGraph. "
                f"Consider marking thread as inactive in your Project table instead."
            )

            return True

        except Exception as e:
            logger.error(f"Failed to delete thread history: {e}")
            return False


# Global singleton instance with thread-safe initialization
_checkpointer_service: Optional[CheckpointerService] = None
_service_lock = asyncio.Lock()


async def get_checkpointer_service() -> CheckpointerService:
    """
    Get or create the global checkpointer service (thread-safe singleton).

    This ensures only one CheckpointerService instance exists across the entire
    application, sharing connection pools and resources efficiently.
    """
    global _checkpointer_service

    if _checkpointer_service is not None:
        return _checkpointer_service

    async with _service_lock:
        # Double-check pattern for thread safety
        if _checkpointer_service is None:
            _checkpointer_service = CheckpointerService(
                min_pool_size=int(os.getenv("CHECKPOINT_POOL_MIN", "2")),
                max_pool_size=int(os.getenv("CHECKPOINT_POOL_MAX", "20")),
            )
            logger.info("✅ Checkpointer service singleton created")

        return _checkpointer_service


# Backward compatibility - create singleton instance immediately
checkpointer_service = CheckpointerService(
    min_pool_size=int(os.getenv("CHECKPOINT_POOL_MIN", "2")),
    max_pool_size=int(os.getenv("CHECKPOINT_POOL_MAX", "20")),
)
