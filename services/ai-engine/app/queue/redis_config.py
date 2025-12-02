"""
Redis connection configuration and management.
Handles connection pooling and configuration for job queue and caching.
"""

import os
import logging
from redis import Redis
from redis.connection import ConnectionPool

logger = logging.getLogger(__name__)


class RedisConfig:
    """Redis configuration and connection management."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis configuration.
        
        Args:
            host: Redis host (default from env or localhost)
            port: Redis port (default from env or 6379)
            db: Redis database number (default from env or 0)
            password: Redis password (default from env)
            decode_responses: Whether to decode responses as strings
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db or int(os.getenv("REDIS_DB", 0))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.decode_responses = decode_responses
        
        self._connection_pool = None
        self._client = None
    
    def get_connection_pool(self) -> ConnectionPool:
        """
        Get or create Redis connection pool.
        
        Returns:
            ConnectionPool instance
        """
        if self._connection_pool is None:
            self._connection_pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
                max_connections=10,
            )
            logger.info(
                f"Redis connection pool created: {self.host}:{self.port}/db{self.db}"
            )
        
        return self._connection_pool
    
    def get_client(self) -> Redis:
        """
        Get or create Redis client.
        
        Returns:
            Redis client instance
        """
        if self._client is None:
            self._client = Redis(
                connection_pool=self.get_connection_pool()
            )
            # Test connection
            try:
                self._client.ping()
                logger.info("Redis connection successful")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                raise
        
        return self._client
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
        if self._connection_pool:
            self._connection_pool.disconnect()
            self._connection_pool = None
        logger.info("Redis connection closed")


# Global Redis configuration instance
_redis_config = None


def get_redis_config() -> RedisConfig:
    """Get or create global Redis configuration."""
    global _redis_config
    if _redis_config is None:
        _redis_config = RedisConfig()
    return _redis_config


def get_redis_client() -> Redis:
    """Get Redis client from global configuration."""
    return get_redis_config().get_client()
