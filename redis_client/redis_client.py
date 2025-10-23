# redis/redis_client.py - PRODUCTION VERSION

"""
Production Redis client for sandbox ID caching.
- Thread-safe with connection pooling
- Automatic reconnection on failure
- Graceful degradation if Redis unavailable
"""

import os
import logging
from typing import Optional
import threading
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Redis package not installed - caching disabled")
    logger.warning("   Install with: pip install redis")
    redis = None
    ConnectionPool = None
    REDIS_AVAILABLE = False


class RedisClient:
    """
    Production Redis client with connection pooling.
    
    Features:
    - Thread-safe connection pool
    - Automatic retry on connection errors
    - Graceful fallback if unavailable
    - Singleton pattern for efficient resource usage
    
    Best practices from redis-py documentation:
    - Uses connection pooling (recommended for production)
    - Thread-safe: Client instances can be shared between threads
    - Connection pool handles connection reuse automatically
    """
    
    _instance: Optional['redis.Redis'] = None
    _pool: Optional['ConnectionPool'] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> Optional['redis.Redis']:
        """
        Get or create Redis client instance (thread-safe).
        
        Returns:
            Redis client with connection pool, or None if unavailable
        """
        if not REDIS_AVAILABLE:
            return None
        
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                    
                    try:
                        # Create connection pool (best practice)
                        cls._pool = ConnectionPool.from_url(
                            redis_url,
                            decode_responses=True,
                            max_connections=10,  # Pool size
                            socket_connect_timeout=5,
                            socket_timeout=5,
                            socket_keepalive=True,
                            socket_keepalive_options={
                                # Keep-alive options for better connection stability
                            },
                            retry_on_timeout=True,  # Auto-retry on timeout
                            health_check_interval=30,  # Check connection health
                        )
                        
                        # Create Redis client with connection pool
                        cls._instance = redis.Redis(
                            connection_pool=cls._pool,
                            # Retry on connection errors (sync default behavior)
                            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                        )
                        
                        # Test connection
                        cls._instance.ping()
                        logger.info(f"✅ Redis connected: {redis_url}")
                        logger.info(f"   Connection pool: max={cls._pool.max_connections}")
                        
                    except redis.ConnectionError as e:
                        logger.error(f"❌ Redis connection failed: {e}")
                        logger.warning("⚠️ Falling back to no caching")
                        cls._instance = None
                        cls._pool = None
                        
                    except Exception as e:
                        logger.error(f"❌ Redis setup error: {e}")
                        logger.warning("⚠️ Falling back to no caching")
                        cls._instance = None
                        cls._pool = None
        
        return cls._instance
    
    @classmethod
    def close(cls):
        """Close Redis connection pool"""
        if cls._pool:
            cls._pool.disconnect()
            logger.info("Redis connection pool closed")
        
        cls._instance = None
        cls._pool = None
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get connection pool statistics"""
        if not cls._pool:
            return {"available": False}
        
        try:
            return {
                "available": True,
                "max_connections": cls._pool.max_connections,
                "available_connections": len(cls._pool._available_connections),
                "in_use_connections": len(cls._pool._in_use_connections),
            }
        except:
            return {"available": True, "stats_unavailable": True}


def get_redis() -> Optional['redis.Redis']:
    """
    Get Redis client (thread-safe, with connection pooling).
    
    Safe to call from async code - uses sync Redis with connection pool.
    The connection pool handles thread safety automatically.
    
    Returns:
        Redis client instance, or None if Redis is unavailable
    """
    return RedisClient.get_instance()


def close_redis():
    """Close Redis connection pool"""
    RedisClient.close()


def get_redis_stats() -> dict:
    """Get Redis connection pool statistics"""
    return RedisClient.get_stats()


# Wrapper functions for common operations (with error handling)

def safe_get(key: str) -> Optional[str]:
    """
    Safely get value from Redis (returns None on error).
    
    Args:
        key: Redis key
        
    Returns:
        Value or None if not found/error
    """
    client = get_redis()
    if not client:
        return None
    
    try:
        return client.get(key)
    except Exception as e:
        logger.warning(f"Redis GET error for {key}: {e}")
        return None


def safe_set(key: str, value: str, ex: int = None) -> bool:
    """
    Safely set value in Redis (returns False on error).
    
    Args:
        key: Redis key
        value: Value to set
        ex: Expiration time in seconds (optional)
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis()
    if not client:
        return False
    
    try:
        if ex:
            client.setex(key, ex, value)
        else:
            client.set(key, value)
        return True
    except Exception as e:
        logger.warning(f"Redis SET error for {key}: {e}")
        return False


def safe_delete(key: str) -> bool:
    """
    Safely delete key from Redis (returns False on error).
    
    Args:
        key: Redis key
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis()
    if not client:
        return False
    
    try:
        client.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Redis DELETE error for {key}: {e}")
        return False
