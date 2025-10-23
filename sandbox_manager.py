# multi_tenant_sandbox_manager.py
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from e2b import AsyncSandbox
from e2b.exceptions import (
    SandboxException,
    AuthenticationException,
    RateLimitException,
    TimeoutException,
)


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation"""

    template: Optional[str] = "next-fast-mongo-pre-v2"   # "next-fast-mongo-pre-v2"
    timeout: int = 500
    auto_pause: bool = True
    allow_internet_access: bool = True
    secure: bool = True
    api_key: Optional[str] = None

    # Pool limits
    max_sandboxes_per_user: int = 2
    max_total_sandboxes: int = 10

    # Cleanup settings
    idle_timeout: int = 500  # 10 minutes
    max_sandbox_age: int = 900  # 1 hour

    # Retry configuration
    max_retries: int = 2
    retry_delay: float = 1.0


@dataclass
class SandboxInfo:
    """Information about a sandbox instance"""

    sandbox: AsyncSandbox
    sandbox_id: str
    user_id: str
    project_id: str
    created_at: float
    last_activity: float
    request_count: int = 0

    def is_idle(self, timeout: int) -> bool:
        """Check if sandbox has been idle too long"""
        return (time.time() - self.last_activity) > timeout

    def is_expired(self, max_age: int) -> bool:
        """Check if sandbox has exceeded max age"""
        return (time.time() - self.created_at) > max_age

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        self.request_count += 1


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    return logging.getLogger("MultiTenantSandboxManager")


class MultiTenantSandboxManager:
    """
    Production-grade multi-tenant sandbox manager.

    Each (user_id, project_id) combination gets its own isolated sandbox.
    Implements resource limits, automatic cleanup, and proper isolation.
    """

    _instance: Optional["MultiTenantSandboxManager"] = None
    _instance_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = setup_logging()

        # Sandbox pool: key = (user_id, project_id), value = SandboxInfo
        self._sandbox_pool: Dict[Tuple[str, str], SandboxInfo] = {}
        self._pool_lock = asyncio.Lock()

        # Per-user sandbox creation locks to prevent race conditions
        self._user_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._user_locks_lock = asyncio.Lock()

        # Configuration
        self._config: Optional[SandboxConfig] = None

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_sandboxes_created": 0,
            "total_requests": 0,
            "active_sandboxes": 0,
            "cleaned_up_sandboxes": 0,
            "rejected_requests": 0,
        }

        self._initialized = True
        self.logger.info("=" * 80)
        self.logger.info("MultiTenantSandboxManager initialized")
        self.logger.info("Each user+project gets isolated sandbox")
        self.logger.info("=" * 80)

    async def initialize(
        self, config: Optional[SandboxConfig] = None
    ) -> "MultiTenantSandboxManager":
        """Initialize the manager with configuration"""
        async with self._instance_lock:
            if config is None:
                config = SandboxConfig()

            if config.api_key is None:
                config.api_key = os.getenv("E2B_API_KEY")
                if not config.api_key:
                    raise ValueError("E2B_API_KEY not set")

            # Use template from environment if not explicitly set
            if config.template is None:
                config.template = os.getenv("E2B_TEMPLATE_ID")
                if config.template:
                    self.logger.info(
                        f"Using E2B template from environment: {config.template}"
                    )

            self._config = config
            self.logger.info(
                f"Configuration: template={config.template or 'default'}, "
                f"max_per_user={config.max_sandboxes_per_user}, "
                f"max_total={config.max_total_sandboxes}"
            )

            # Start cleanup task
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            return self

    async def _get_user_lock(self, user_id: str, project_id: str) -> asyncio.Lock:
        """Get or create lock for specific user+project"""
        key = (user_id, project_id)
        async with self._user_locks_lock:
            if key not in self._user_locks:
                self._user_locks[key] = asyncio.Lock()
            return self._user_locks[key]

    async def get_sandbox(
        self,
        user_id: str,
        project_id: str,
        metadata: Optional[Dict[str, str]] = None,
        envs: Optional[Dict[str, str]] = None,
    ) -> AsyncSandbox:
        """
        Get or create sandbox for specific user and project.

        Args:
            user_id: Unique user identifier
            project_id: Unique project identifier
            metadata: Additional metadata for sandbox
            envs: Environment variables for sandbox

        Returns:
            AsyncSandbox instance isolated for this user+project
        """
        if not self._config:
            raise ValueError("Manager not initialized. Call initialize() first.")

        key = (user_id, project_id)
        self._stats["total_requests"] += 1

        self.logger.info(f"[{user_id}/{project_id}] Sandbox request")

        # Get user-specific lock to prevent race conditions for this user+project
        user_lock = await self._get_user_lock(user_id, project_id)

        async with user_lock:
            # Check if sandbox already exists for this user+project
            if key in self._sandbox_pool:
                sandbox_info = self._sandbox_pool[key]

                # Verify sandbox is still healthy
                try:
                    await self._verify_sandbox_health(sandbox_info.sandbox)
                    sandbox_info.update_activity()
                    self.logger.info(
                        f"[{user_id}/{project_id}] Returning existing sandbox: "
                        f"{sandbox_info.sandbox_id} (requests: {sandbox_info.request_count})"
                    )
                    return sandbox_info.sandbox
                except Exception as e:
                    self.logger.warning(
                        f"[{user_id}/{project_id}] Health check failed: {e}. "
                        "Creating new sandbox."
                    )
                    await self._remove_sandbox(key)

            # Check resource limits before creating new sandbox
            await self._enforce_resource_limits(user_id, project_id)

            # Create new sandbox
            sandbox = await self._create_sandbox_for_user(
                user_id, project_id, metadata, envs
            )

            return sandbox

    async def _enforce_resource_limits(self, user_id: str, project_id: str):
        """Enforce resource limits before creating new sandbox"""
        # Check total sandbox limit
        if len(self._sandbox_pool) >= self._config.max_total_sandboxes:
            self._stats["rejected_requests"] += 1
            self.logger.error(
                f"[{user_id}/{project_id}] REJECTED: Total sandbox limit reached "
                f"({self._config.max_total_sandboxes})"
            )
            raise RuntimeError(
                f"Maximum total sandboxes ({self._config.max_total_sandboxes}) reached. "
                "Try again later."
            )

        # Check per-user limit
        user_sandboxes = [key for key in self._sandbox_pool.keys() if key[0] == user_id]

        if len(user_sandboxes) >= self._config.max_sandboxes_per_user:
            self._stats["rejected_requests"] += 1
            self.logger.error(
                f"[{user_id}/{project_id}] REJECTED: User sandbox limit reached "
                f"({self._config.max_sandboxes_per_user})"
            )
            raise RuntimeError(
                f"User {user_id} has reached maximum sandboxes "
                f"({self._config.max_sandboxes_per_user}). "
                "Close existing sandboxes first."
            )

    async def _create_sandbox_for_user(
        self,
        user_id: str,
        project_id: str,
        metadata: Optional[Dict[str, str]] = None,
        envs: Optional[Dict[str, str]] = None,
    ) -> AsyncSandbox:
        """Create new sandbox for user+project with retry logic"""
        key = (user_id, project_id)

        self.logger.info(f"[{user_id}/{project_id}] Creating NEW sandbox...")

        # Prepare metadata
        full_metadata = {
            "user_id": user_id,
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        }

        # Retry logic
        for attempt in range(1, self._config.max_retries + 1):
            try:
                sandbox = await AsyncSandbox.create(
                    template=self._config.template,
                    timeout=self._config.timeout,
                    allow_internet_access=self._config.allow_internet_access,
                    metadata=full_metadata,
                    envs=envs or {},
                    secure=self._config.secure,
                    api_key=self._config.api_key,
                    debug=os.getenv("E2B_DEBUG", "false").lower() == "true",
                )

                # Store in pool
                sandbox_info = SandboxInfo(
                    sandbox=sandbox,
                    sandbox_id=sandbox.sandbox_id,
                    user_id=user_id,
                    project_id=project_id,
                    created_at=time.time(),
                    last_activity=time.time(),
                )

                async with self._pool_lock:
                    self._sandbox_pool[key] = sandbox_info
                    self._stats["total_sandboxes_created"] += 1
                    self._stats["active_sandboxes"] = len(self._sandbox_pool)

                self.logger.info("=" * 80)
                self.logger.info(f"[{user_id}/{project_id}] ✅ Sandbox created!")
                self.logger.info(f"   Sandbox ID: {sandbox.sandbox_id}")
                self.logger.info(f"   Active sandboxes: {len(self._sandbox_pool)}")
                self.logger.info(
                    f"   Total created: {self._stats['total_sandboxes_created']}"
                )
                self.logger.info("=" * 80)

                return sandbox

            except (
                SandboxException,
                AuthenticationException,
                RateLimitException,
                TimeoutException,
            ) as e:
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay * (2 ** (attempt - 1))
                    self.logger.warning(
                        f"[{user_id}/{project_id}] Creation attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"[{user_id}/{project_id}] Creation failed: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"[{user_id}/{project_id}] Unexpected error: {e}")
                raise

        raise RuntimeError("Failed to create sandbox after retries")

    async def _verify_sandbox_health(self, sandbox: AsyncSandbox):
        """Quick health check"""
        try:
            await asyncio.wait_for(sandbox.files.list("."), timeout=3.0)
        except Exception as e:
            raise Exception(f"Health check failed: {e}")

    async def close_sandbox(self, user_id: str, project_id: str):
        """Close sandbox for specific user+project"""
        key = (user_id, project_id)
        await self._remove_sandbox(key)

    async def close_all_user_sandboxes(self, user_id: str):
        """Close all sandboxes for a specific user"""
        user_keys = [key for key in self._sandbox_pool.keys() if key[0] == user_id]

        self.logger.info(f"Closing {len(user_keys)} sandboxes for user {user_id}")

        for key in user_keys:
            await self._remove_sandbox(key)

    async def _remove_sandbox(self, key: Tuple[str, str]):
        """Remove and close sandbox from pool"""
        async with self._pool_lock:
            if key in self._sandbox_pool:
                sandbox_info = self._sandbox_pool[key]

                try:
                    await sandbox_info.sandbox.kill()
                    self.logger.info(
                        f"[{key[0]}/{key[1]}] Sandbox closed: {sandbox_info.sandbox_id}"
                    )
                except Exception as e:
                    self.logger.warning(f"Error closing sandbox: {e}")

                del self._sandbox_pool[key]
                self._stats["active_sandboxes"] = len(self._sandbox_pool)
                self._stats["cleaned_up_sandboxes"] += 1

    async def _cleanup_loop(self):
        """Background task to cleanup idle/expired sandboxes"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self._config:
                    continue

                keys_to_remove = []

                async with self._pool_lock:
                    for key, sandbox_info in self._sandbox_pool.items():
                        # Check if idle
                        if sandbox_info.is_idle(self._config.idle_timeout):
                            self.logger.info(
                                f"[{key[0]}/{key[1]}] Cleanup: Idle timeout "
                                f"({self._config.idle_timeout}s)"
                            )
                            keys_to_remove.append(key)
                        # Check if expired
                        elif sandbox_info.is_expired(self._config.max_sandbox_age):
                            self.logger.info(
                                f"[{key[0]}/{key[1]}] Cleanup: Max age exceeded "
                                f"({self._config.max_sandbox_age}s)"
                            )
                            keys_to_remove.append(key)

                # Remove identified sandboxes
                for key in keys_to_remove:
                    await self._remove_sandbox(key)

                if keys_to_remove:
                    self.logger.info(
                        f"Cleanup: Removed {len(keys_to_remove)} sandboxes"
                    )

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        user_distribution = {}
        for user_id, project_id in self._sandbox_pool.keys():
            user_distribution[user_id] = user_distribution.get(user_id, 0) + 1

        return {
            "total_sandboxes_created": self._stats["total_sandboxes_created"],
            "active_sandboxes": len(self._sandbox_pool),
            "total_requests": self._stats["total_requests"],
            "cleaned_up_sandboxes": self._stats["cleaned_up_sandboxes"],
            "rejected_requests": self._stats["rejected_requests"],
            "unique_users": len(set(key[0] for key in self._sandbox_pool.keys())),
            "unique_projects": len(self._sandbox_pool),
            "user_distribution": user_distribution,
            "sandbox_details": [
                {
                    "user_id": info.user_id,
                    "project_id": info.project_id,
                    "sandbox_id": info.sandbox_id,
                    "age_seconds": time.time() - info.created_at,
                    "idle_seconds": time.time() - info.last_activity,
                    "request_count": info.request_count,
                }
                for info in self._sandbox_pool.values()
            ],
        }

    async def shutdown(self):
        """Shutdown manager and close all sandboxes"""
        self.logger.info("Shutting down MultiTenantSandboxManager...")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all sandboxes
        keys = list(self._sandbox_pool.keys())
        for key in keys:
            await self._remove_sandbox(key)

        # Print final stats
        stats = self.get_stats()
        self.logger.info("=" * 80)
        self.logger.info("FINAL STATISTICS:")
        self.logger.info(f"  Total created: {stats['total_sandboxes_created']}")
        self.logger.info(f"  Total requests: {stats['total_requests']}")
        self.logger.info(f"  Cleaned up: {stats['cleaned_up_sandboxes']}")
        self.logger.info(f"  Rejected: {stats['rejected_requests']}")
        self.logger.info("=" * 80)


# Global instance
_multi_tenant_manager: Optional[MultiTenantSandboxManager] = None
_manager_lock = asyncio.Lock()


async def get_multi_tenant_manager() -> MultiTenantSandboxManager:
    """Get the global multi-tenant manager"""
    global _multi_tenant_manager

    async with _manager_lock:
        if _multi_tenant_manager is None:
            _multi_tenant_manager = MultiTenantSandboxManager()
            await _multi_tenant_manager.initialize()

        return _multi_tenant_manager


async def get_user_sandbox(user_id: str, project_id: str, **kwargs) -> AsyncSandbox:
    """Convenience function to get sandbox for user+project"""
    manager = await get_multi_tenant_manager()
    return await manager.get_sandbox(user_id, project_id, **kwargs)


async def cleanup_multi_tenant_manager():
    """Cleanup on shutdown"""
    global _multi_tenant_manager
    if _multi_tenant_manager:
        await _multi_tenant_manager.shutdown()
        _multi_tenant_manager = None


# multi_tenant_sandbox_manager.py - WITH REDIS CACHING
# sandbox_manager.py - FIXED VERSION

# import asyncio
# import logging
# import time
# from typing import Optional, Dict, Any, Tuple
# from dataclasses import dataclass
# from datetime import datetime
# import os

# from e2b import AsyncSandbox
# from e2b.exceptions import (
#     SandboxException,
#     AuthenticationException,
#     RateLimitException,
#     TimeoutException,
# )

# from redis_client import get_redis, close_redis


# @dataclass
# class SandboxConfig:
#     """Configuration for sandbox creation"""

#     template: Optional[str] = "next-fast-mongo-pre-v2"
#     timeout: int = 500
#     auto_pause: bool = True
#     allow_internet_access: bool = True
#     secure: bool = True
#     api_key: Optional[str] = None

#     # Pool limits
#     max_sandboxes_per_user: int = 2
#     max_total_sandboxes: int = 10

#     # Cleanup settings (matches Redis TTL!)
#     idle_timeout: int = 500  # 500 seconds
#     max_sandbox_age: int = 900  # 900 seconds (15 min)

#     # Retry configuration
#     max_retries: int = 2
#     retry_delay: float = 1.0

#     # Redis caching
#     enable_redis: bool = True


# @dataclass
# class SandboxInfo:
#     """Information about a sandbox instance"""

#     sandbox: AsyncSandbox
#     sandbox_id: str
#     user_id: str
#     project_id: str
#     created_at: float
#     last_activity: float
#     request_count: int = 0

#     def is_idle(self, timeout: int) -> bool:
#         """Check if sandbox has been idle too long"""
#         return (time.time() - self.last_activity) > timeout

#     def is_expired(self, max_age: int) -> bool:
#         """Check if sandbox has exceeded max age"""
#         return (time.time() - self.created_at) > max_age

#     def update_activity(self):
#         """Update last activity timestamp"""
#         self.last_activity = time.time()
#         self.request_count += 1


# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
#     )
#     return logging.getLogger("MultiTenantSandboxManager")


# class MultiTenantSandboxManager:
#     """
#     Production-grade multi-tenant sandbox manager with Redis caching.

#     Features:
#     - Redis caching of sandbox IDs (TTL = sandbox uptime)
#     - Each (user_id, project_id) gets isolated sandbox
#     - Automatic cleanup and resource limits
#     - Graceful fallback if Redis unavailable
#     """

#     _instance: Optional["MultiTenantSandboxManager"] = None
#     _instance_lock = asyncio.Lock()

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return

#         self.logger = setup_logging()

#         # Sandbox pool: key = (user_id, project_id), value = SandboxInfo
#         self._sandbox_pool: Dict[Tuple[str, str], SandboxInfo] = {}
#         self._pool_lock = asyncio.Lock()

#         # Per-user sandbox creation locks
#         self._user_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
#         self._user_locks_lock = asyncio.Lock()

#         # Configuration
#         self._config: Optional[SandboxConfig] = None

#         # Redis client
#         self._redis: Optional[any] = None

#         # Background tasks
#         self._cleanup_task: Optional[asyncio.Task] = None

#         # Statistics
#         self._stats = {
#             "total_sandboxes_created": 0,
#             "total_requests": 0,
#             "active_sandboxes": 0,
#             "cleaned_up_sandboxes": 0,
#             "rejected_requests": 0,
#             "redis_cache_hits": 0,
#             "redis_cache_misses": 0,
#         }

#         self._initialized = True
#         self.logger.info("=" * 80)
#         self.logger.info("MultiTenantSandboxManager initialized")
#         self.logger.info("Each user+project gets isolated sandbox")
#         self.logger.info("Redis caching enabled for persistence")
#         self.logger.info("=" * 80)

#     async def initialize(
#         self, config: Optional[SandboxConfig] = None
#     ) -> "MultiTenantSandboxManager":
#         """Initialize the manager with configuration"""
#         async with self._instance_lock:
#             if config is None:
#                 config = SandboxConfig()

#             if config.api_key is None:
#                 config.api_key = os.getenv("E2B_API_KEY")
#                 if not config.api_key:
#                     raise ValueError("E2B_API_KEY not set")

#             if config.template is None:
#                 config.template = os.getenv("E2B_TEMPLATE_ID")

#             self._config = config

#             # Initialize Redis (SYNC CALL - NO AWAIT!)
#             if config.enable_redis:
#                 try:
#                     self._redis = get_redis()  # ← SYNC!
#                     if self._redis:
#                         self.logger.info("✅ Redis caching enabled")
#                     else:
#                         self.logger.warning("⚠️ Redis unavailable - caching disabled")
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Redis init failed: {e} - caching disabled")
#                     self._redis = None

#             self.logger.info(
#                 f"Configuration: template={config.template or 'default'}, "
#                 f"max_per_user={config.max_sandboxes_per_user}, "
#                 f"max_total={config.max_total_sandboxes}, "
#                 f"redis_enabled={self._redis is not None}"
#             )

#             # Start cleanup task
#             if self._cleanup_task is None:
#                 self._cleanup_task = asyncio.create_task(self._cleanup_loop())

#             return self

#     # =========================================================================
#     # REDIS CACHE OPERATIONS (ALL SYNC - NO ASYNC!)
#     # =========================================================================

#     def _get_redis_key(self, user_id: str, project_id: str) -> str:
#         """Generate Redis key for user+project"""
#         return f"sandbox:{user_id}:{project_id}"

#     def _get_cached_sandbox_id(self, user_id: str, project_id: str) -> Optional[str]:
#         """Get sandbox ID from Redis cache (SYNC)"""
#         if not self._redis:
#             return None

#         try:
#             key = self._get_redis_key(user_id, project_id)
#             sandbox_id = self._redis.get(key)  # ← SYNC!

#             if sandbox_id:
#                 self._stats["redis_cache_hits"] += 1
#                 self.logger.debug(
#                     f"[{user_id}/{project_id}] Redis cache HIT: {sandbox_id}"
#                 )
#                 return sandbox_id
#             else:
#                 self._stats["redis_cache_misses"] += 1
#                 self.logger.debug(f"[{user_id}/{project_id}] Redis cache MISS")
#                 return None

#         except Exception as e:
#             self.logger.warning(f"Redis get error: {e}")
#             return None

#     def _cache_sandbox_id(
#         self, user_id: str, project_id: str, sandbox_id: str, ttl: int
#     ):
#         """Cache sandbox ID in Redis with TTL (SYNC)"""
#         if not self._redis:
#             return

#         try:
#             key = self._get_redis_key(user_id, project_id)
#             self._redis.setex(key, ttl, sandbox_id)  # ← SYNC!
#             self.logger.debug(
#                 f"[{user_id}/{project_id}] Cached in Redis: {sandbox_id} (TTL={ttl}s)"
#             )
#         except Exception as e:
#             self.logger.warning(f"Redis set error: {e}")

#     def _remove_cached_sandbox_id(self, user_id: str, project_id: str):
#         """Remove sandbox ID from Redis cache (SYNC)"""
#         if not self._redis:
#             return

#         try:
#             key = self._get_redis_key(user_id, project_id)
#             self._redis.delete(key)  # ← SYNC!
#             self.logger.debug(f"[{user_id}/{project_id}] Removed from Redis cache")
#         except Exception as e:
#             self.logger.warning(f"Redis delete error: {e}")

#     # =========================================================================
#     # SANDBOX RETRIEVAL WITH REDIS
#     # =========================================================================
#     async def get_sandbox(
#         self,
#         user_id: str,
#         project_id: str,
#         metadata: Optional[Dict[str, str]] = None,
#         envs: Optional[Dict[str, str]] = None,
#     ) -> AsyncSandbox:
#         """Get or create sandbox for specific user and project."""

#         if not self._config:
#             raise ValueError("Manager not initialized. Call initialize() first.")

#         key = (user_id, project_id)
#         self._stats["total_requests"] += 1

#         self.logger.info(f"[{user_id}/{project_id}] Sandbox request")

#         user_lock = await self._get_user_lock(user_id, project_id)

#         async with user_lock:
#             # Step 1: Check memory pool
#             if key in self._sandbox_pool:
#                 sandbox_info = self._sandbox_pool[key]

#                 # Smart health check: Only verify if sandbox has been idle
#                 idle_seconds = time.time() - sandbox_info.last_activity

#                 # Skip health check if recently used (< 30s)
#                 if idle_seconds < 30:
#                     sandbox_info.update_activity()
#                     self.logger.info(
#                         f"[{user_id}/{project_id}] Memory pool HIT (fresh): "
#                         f"{sandbox_info.sandbox_id}"
#                     )
#                     return sandbox_info.sandbox

#                 # Health check for idle sandboxes only
#                 try:
#                     await self._verify_sandbox_health(sandbox_info.sandbox)
#                     sandbox_info.update_activity()
#                     self.logger.info(
#                         f"[{user_id}/{project_id}] Memory pool HIT (verified): "
#                         f"{sandbox_info.sandbox_id}"
#                     )
#                     return sandbox_info.sandbox
#                 except Exception as e:
#                     self.logger.warning(
#                         f"[{user_id}/{project_id}] Health check failed: {e}"
#                     )
#                     await self._remove_sandbox(key)

#             # Step 2: Check Redis cache
#             cached_sandbox_id = self._get_cached_sandbox_id(user_id, project_id)

#             if cached_sandbox_id:
#                 try:
#                     sandbox = await self._reconnect_to_sandbox(
#                         cached_sandbox_id, user_id, project_id
#                     )
#                     return sandbox
#                 except Exception as e:
#                     self.logger.warning(
#                         f"[{user_id}/{project_id}] Reconnect failed for "
#                         f"{cached_sandbox_id}: {e}"
#                     )
#                     self._remove_cached_sandbox_id(user_id, project_id)

#             # Step 3: Create new sandbox
#             await self._enforce_resource_limits(user_id, project_id)
#             sandbox = await self._create_sandbox_for_user(
#                 user_id, project_id, metadata, envs
#             )

#             return sandbox

#     async def _reconnect_to_sandbox(
#         self, sandbox_id: str, user_id: str, project_id: str
#     ) -> AsyncSandbox:
#         """
#         Reconnect using Method 3 (direct constructor).
#         Avoids the broken connect() methods.
#         """
#         key = (user_id, project_id)

#         self.logger.info(f"[{user_id}/{project_id}] Reconnecting: {sandbox_id}")

#         try:
#             # Import dependencies
#             from e2b.api.client.types import Unset
#             from e2b.connection_config import ConnectionConfig
#             from packaging.version import Version

#             # Step 1: Get sandbox info (doesn't call resume!)
#             try:
#                 response = await asyncio.wait_for(
#                     AsyncSandbox.get_info(
#                         sandbox_id,
#                         api_key=self._config.api_key,
#                     ),
#                     timeout=5.0,
#                 )
#             except asyncio.TimeoutError:
#                 raise Exception("get_info timed out - sandbox may be deleted")

#             # Step 2: Build connection headers
#             sandbox_headers = {}
#             envd_access_token = response._envd_access_token

#             if envd_access_token is not None and not isinstance(
#                 envd_access_token, Unset
#             ):
#                 sandbox_headers["X-Access-Token"] = envd_access_token

#             # Step 3: Create connection config
#             connection_config = ConnectionConfig(
#                 extra_sandbox_headers=sandbox_headers,
#                 api_key=self._config.api_key,
#             )

#             # Step 4: Use direct constructor (Method 3 - WORKS!)
#             sandbox = AsyncSandbox(
#                 sandbox_id=response.sandbox_id,
#                 sandbox_domain=response.sandbox_domain,
#                 envd_version=Version(response.envd_version),
#                 envd_access_token=envd_access_token,
#                 connection_config=connection_config,
#             )

#             # Step 5: Verify it's alive
#             try:
#                 await asyncio.wait_for(
#                     self._verify_sandbox_health(sandbox), timeout=5.0
#                 )
#             except asyncio.TimeoutError:
#                 raise Exception("Sandbox not responding")

#             # Step 6: Add to pool
#             sandbox_info = SandboxInfo(
#                 sandbox=sandbox,
#                 sandbox_id=sandbox_id,
#                 user_id=user_id,
#                 project_id=project_id,
#                 created_at=time.time(),
#                 last_activity=time.time(),
#             )

#             async with self._pool_lock:
#                 self._sandbox_pool[key] = sandbox_info
#                 self._stats["active_sandboxes"] = len(self._sandbox_pool)

#             self.logger.info(f"[{user_id}/{project_id}] ✅ Reconnected (Method 3)")

#             return sandbox

#         except Exception as e:
#             self.logger.warning(f"[{user_id}/{project_id}] Reconnect failed: {e}")
#             raise

#     async def _get_user_lock(self, user_id: str, project_id: str) -> asyncio.Lock:
#         """Get or create lock for specific user+project"""
#         key = (user_id, project_id)
#         async with self._user_locks_lock:
#             if key not in self._user_locks:
#                 self._user_locks[key] = asyncio.Lock()
#             return self._user_locks[key]

#     async def _enforce_resource_limits(self, user_id: str, project_id: str):
#         """Enforce resource limits before creating new sandbox"""
#         if len(self._sandbox_pool) >= self._config.max_total_sandboxes:
#             self._stats["rejected_requests"] += 1
#             raise RuntimeError(
#                 f"Maximum total sandboxes ({self._config.max_total_sandboxes}) reached"
#             )

#         user_sandboxes = [key for key in self._sandbox_pool.keys() if key[0] == user_id]

#         if len(user_sandboxes) >= self._config.max_sandboxes_per_user:
#             self._stats["rejected_requests"] += 1
#             raise RuntimeError(
#                 f"User {user_id} reached max sandboxes "
#                 f"({self._config.max_sandboxes_per_user})"
#             )

#     async def _create_sandbox_for_user(
#         self,
#         user_id: str,
#         project_id: str,
#         metadata: Optional[Dict[str, str]] = None,
#         envs: Optional[Dict[str, str]] = None,
#     ) -> AsyncSandbox:
#         """Create new sandbox and cache in Redis"""
#         key = (user_id, project_id)

#         self.logger.info(f"[{user_id}/{project_id}] Creating NEW sandbox...")

#         full_metadata = {
#             "user_id": user_id,
#             "project_id": project_id,
#             "created_at": datetime.now().isoformat(),
#             **(metadata or {}),
#         }

#         for attempt in range(1, self._config.max_retries + 1):
#             try:
#                 sandbox = await AsyncSandbox.create(
#                     template=self._config.template,
#                     timeout=self._config.timeout,
#                     allow_internet_access=self._config.allow_internet_access,
#                     metadata=full_metadata,
#                     envs=envs or {},
#                     secure=self._config.secure,
#                     api_key=self._config.api_key,
#                 )

#                 # Store in memory pool
#                 sandbox_info = SandboxInfo(
#                     sandbox=sandbox,
#                     sandbox_id=sandbox.sandbox_id,
#                     user_id=user_id,
#                     project_id=project_id,
#                     created_at=time.time(),
#                     last_activity=time.time(),
#                 )

#                 async with self._pool_lock:
#                     self._sandbox_pool[key] = sandbox_info
#                     self._stats["total_sandboxes_created"] += 1
#                     self._stats["active_sandboxes"] = len(self._sandbox_pool)

#                 # Cache in Redis (SYNC CALL!)
#                 self._cache_sandbox_id(  # ← NO AWAIT!
#                     user_id,
#                     project_id,
#                     sandbox.sandbox_id,
#                     ttl=self._config.max_sandbox_age,
#                 )

#                 self.logger.info("=" * 80)
#                 self.logger.info(f"[{user_id}/{project_id}] ✅ Sandbox created!")
#                 self.logger.info(f"   Sandbox ID: {sandbox.sandbox_id}")
#                 self.logger.info(f"   Redis TTL: {self._config.max_sandbox_age}s")
#                 self.logger.info(f"   Active: {len(self._sandbox_pool)}")
#                 self.logger.info("=" * 80)

#                 return sandbox

#             except Exception as e:
#                 if attempt < self._config.max_retries:
#                     delay = self._config.retry_delay * (2 ** (attempt - 1))
#                     self.logger.warning(
#                         f"[{user_id}/{project_id}] Attempt {attempt} failed: {e}. "
#                         f"Retrying in {delay:.2f}s..."
#                     )
#                     await asyncio.sleep(delay)
#                 else:
#                     raise

#         raise RuntimeError("Failed to create sandbox after retries")

#     async def _verify_sandbox_health(self, sandbox: AsyncSandbox):
#         """Quick health check"""
#         try:
#             await asyncio.wait_for(sandbox.files.list("."), timeout=3.0)
#         except Exception as e:
#             raise Exception(f"Health check failed: {e}")

#     async def close_sandbox(self, user_id: str, project_id: str):
#         """Close sandbox and remove from Redis"""
#         key = (user_id, project_id)
#         await self._remove_sandbox(key)
#         self._remove_cached_sandbox_id(user_id, project_id)  # ← SYNC CALL!

#     async def _remove_sandbox(self, key: Tuple[str, str]):
#         """Remove and close sandbox from pool"""
#         async with self._pool_lock:
#             if key in self._sandbox_pool:
#                 sandbox_info = self._sandbox_pool[key]

#                 try:
#                     await sandbox_info.sandbox.kill()
#                     self.logger.info(
#                         f"[{key[0]}/{key[1]}] Closed: {sandbox_info.sandbox_id}"
#                     )
#                 except Exception as e:
#                     self.logger.warning(f"Error closing sandbox: {e}")

#                 # Remove from Redis cache (SYNC CALL!)
#                 self._remove_cached_sandbox_id(key[0], key[1])  # ← NO AWAIT!

#                 del self._sandbox_pool[key]
#                 self._stats["active_sandboxes"] = len(self._sandbox_pool)
#                 self._stats["cleaned_up_sandboxes"] += 1

#     async def _cleanup_loop(self):
#         """Background cleanup task"""
#         while True:
#             try:
#                 await asyncio.sleep(30)

#                 if not self._config:
#                     continue

#                 keys_to_remove = []

#                 async with self._pool_lock:
#                     for key, sandbox_info in self._sandbox_pool.items():
#                         if sandbox_info.is_idle(self._config.idle_timeout):
#                             self.logger.info(f"[{key[0]}/{key[1]}] Cleanup: Idle")
#                             keys_to_remove.append(key)
#                         elif sandbox_info.is_expired(self._config.max_sandbox_age):
#                             self.logger.info(f"[{key[0]}/{key[1]}] Cleanup: Expired")
#                             keys_to_remove.append(key)

#                 for key in keys_to_remove:
#                     await self._remove_sandbox(key)

#             except Exception as e:
#                 self.logger.error(f"Cleanup error: {e}")

#     def get_stats(self) -> Dict[str, Any]:
#         """Get manager statistics"""
#         return {
#             **self._stats,
#             "active_sandboxes": len(self._sandbox_pool),
#             "redis_enabled": self._redis is not None,
#             "cache_hit_rate": (
#                 self._stats["redis_cache_hits"]
#                 / (self._stats["redis_cache_hits"] + self._stats["redis_cache_misses"])
#                 if (self._stats["redis_cache_hits"] + self._stats["redis_cache_misses"])
#                 > 0
#                 else 0
#             ),
#         }

#     async def shutdown(self):
#         """Shutdown manager"""
#         self.logger.info("Shutting down...")

#         if self._cleanup_task:
#             self._cleanup_task.cancel()
#             try:
#                 await self._cleanup_task
#             except asyncio.CancelledError:
#                 pass

#         keys = list(self._sandbox_pool.keys())
#         for key in keys:
#             await self._remove_sandbox(key)

#         # Close Redis (SYNC CALL!)
#         close_redis()  # ← NO AWAIT!

#         stats = self.get_stats()
#         self.logger.info("=" * 80)
#         self.logger.info("FINAL STATS:")
#         self.logger.info(f"  Total created: {stats['total_sandboxes_created']}")
#         self.logger.info(f"  Redis cache hits: {stats['redis_cache_hits']}")
#         self.logger.info(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
#         self.logger.info("=" * 80)


# # Global instance
# _multi_tenant_manager: Optional[MultiTenantSandboxManager] = None
# _manager_lock = asyncio.Lock()


# async def get_multi_tenant_manager() -> MultiTenantSandboxManager:
#     """Get the global multi-tenant manager"""
#     global _multi_tenant_manager

#     async with _manager_lock:
#         if _multi_tenant_manager is None:
#             _multi_tenant_manager = MultiTenantSandboxManager()
#             await _multi_tenant_manager.initialize()

#         return _multi_tenant_manager


# async def get_user_sandbox(user_id: str, project_id: str, **kwargs) -> AsyncSandbox:
#     """Convenience function to get sandbox for user+project"""
#     manager = await get_multi_tenant_manager()
#     return await manager.get_sandbox(user_id, project_id, **kwargs)


# async def cleanup_multi_tenant_manager():
#     """Cleanup on shutdown"""
#     global _multi_tenant_manager
#     if _multi_tenant_manager:
#         await _multi_tenant_manager.shutdown()
#         _multi_tenant_manager = None
