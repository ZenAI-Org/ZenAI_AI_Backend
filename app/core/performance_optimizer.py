"""
Performance optimization module for caching, query optimization, and metrics.
Handles context caching, pgvector query optimization, and performance monitoring.
"""

import logging
import time
import json
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib

from app.queue.redis_config import get_redis_client

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of frequently accessed context using Redis."""
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.redis = redis_client or get_redis_client()
        self.default_ttl = default_ttl
    
    def _make_key(self, prefix: str, *args) -> str:
        """
        Create a cache key from prefix and arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Arguments to include in key
            
        Returns:
            Cache key string
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            value = self.redis.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ttl = ttl or self.default_ttl
            self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        try:
            deleted = self.redis.delete(key) > 0
            if deleted:
                logger.debug(f"Cache deleted: {key}")
            return deleted
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "context:project_*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis.keys(pattern)
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cache cleared {deleted} keys matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern failed for {pattern}: {e}")
            return 0
    
    def get_context(
        self,
        project_id: str,
        query: str,
        limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached context for a project query.
        
        Args:
            project_id: Project identifier
            query: Query text
            limit: Result limit
            
        Returns:
            Cached context or None
        """
        # Create hash of query for cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        key = self._make_key("context", project_id, query_hash, limit)
        return self.get(key)
    
    def set_context(
        self,
        project_id: str,
        query: str,
        context: Dict[str, Any],
        limit: int = 5,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache context for a project query.
        
        Args:
            project_id: Project identifier
            query: Query text
            context: Context data to cache
            limit: Result limit
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        key = self._make_key("context", project_id, query_hash, limit)
        return self.set(key, context, ttl)
    
    def get_suggestions(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached suggestions for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Cached suggestions or None
        """
        key = self._make_key("suggestions", project_id)
        return self.get(key)
    
    def set_suggestions(
        self,
        project_id: str,
        suggestions: Dict[str, Any],
        ttl: int = 21600  # 6 hours
    ) -> bool:
        """
        Cache suggestions for a project.
        
        Args:
            project_id: Project identifier
            suggestions: Suggestions data
            ttl: Time-to-live in seconds (default 6 hours)
            
        Returns:
            True if successful
        """
        key = self._make_key("suggestions", project_id)
        return self.set(key, suggestions, ttl)
    
    def get_conversation_history(
        self,
        user_id: str,
        project_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached conversation history.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            
        Returns:
            Cached conversation history or None
        """
        key = self._make_key("chat_history", user_id, project_id)
        return self.get(key)
    
    def set_conversation_history(
        self,
        user_id: str,
        project_id: str,
        history: List[Dict[str, Any]],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """
        Cache conversation history.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            history: Conversation history
            ttl: Time-to-live in seconds (default 24 hours)
            
        Returns:
            True if successful
        """
        key = self._make_key("chat_history", user_id, project_id)
        return self.set(key, history, ttl)


class PerformanceMetrics:
    """Tracks performance metrics for monitoring and optimization."""
    
    def __init__(self, redis_client=None):
        """
        Initialize performance metrics tracker.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client or get_redis_client()
        self.metrics = {}
    
    def record_latency(
        self,
        operation: str,
        duration: float,
        project_id: Optional[str] = None
    ) -> None:
        """
        Record operation latency.
        
        Args:
            operation: Operation name (e.g., "semantic_search", "chat_response")
            duration: Duration in seconds
            project_id: Optional project identifier
        """
        try:
            key = f"metrics:latency:{operation}"
            timestamp = datetime.now().isoformat()
            
            # Store in Redis for aggregation
            self.redis.lpush(
                key,
                json.dumps({
                    "duration": duration,
                    "timestamp": timestamp,
                    "project_id": project_id
                })
            )
            
            # Keep only last 1000 entries
            self.redis.ltrim(key, 0, 999)
            
            # Set expiration to 24 hours
            self.redis.expire(key, 86400)
            
            logger.debug(f"Recorded latency: {operation} = {duration:.3f}s")
        except Exception as e:
            logger.warning(f"Failed to record latency: {e}")
    
    def record_cache_hit(self, cache_type: str) -> None:
        """
        Record cache hit.
        
        Args:
            cache_type: Type of cache (e.g., "context", "suggestions")
        """
        try:
            key = f"metrics:cache_hits:{cache_type}"
            self.redis.incr(key)
            self.redis.expire(key, 86400)
        except Exception as e:
            logger.warning(f"Failed to record cache hit: {e}")
    
    def record_cache_miss(self, cache_type: str) -> None:
        """
        Record cache miss.
        
        Args:
            cache_type: Type of cache
        """
        try:
            key = f"metrics:cache_misses:{cache_type}"
            self.redis.incr(key)
            self.redis.expire(key, 86400)
        except Exception as e:
            logger.warning(f"Failed to record cache miss: {e}")
    
    def get_cache_stats(self, cache_type: str) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            Dictionary with cache statistics
        """
        try:
            hits_key = f"metrics:cache_hits:{cache_type}"
            misses_key = f"metrics:cache_misses:{cache_type}"
            
            hits = int(self.redis.get(hits_key) or 0)
            misses = int(self.redis.get(misses_key) or 0)
            total = hits + misses
            
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            return {
                "cache_type": cache_type,
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hit_rate
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}
    
    def get_latency_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get latency statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with latency statistics
        """
        try:
            key = f"metrics:latency:{operation}"
            entries = self.redis.lrange(key, 0, -1)
            
            if not entries:
                return {
                    "operation": operation,
                    "count": 0,
                    "avg": 0,
                    "min": 0,
                    "max": 0
                }
            
            durations = []
            for entry in entries:
                data = json.loads(entry)
                durations.append(data["duration"])
            
            return {
                "operation": operation,
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p95": sorted(durations)[int(len(durations) * 0.95)]
            }
        except Exception as e:
            logger.warning(f"Failed to get latency stats: {e}")
            return {}


class QueryOptimizer:
    """Optimizes pgvector queries for better performance."""
    
    def __init__(self, db_connection):
        """
        Initialize query optimizer.
        
        Args:
            db_connection: PostgreSQL connection
        """
        self.db = db_connection
    
    def ensure_indexes(self) -> bool:
        """
        Ensure all necessary indexes exist on pgvector table.
        
        Returns:
            True if successful
        """
        try:
            cursor = self.db.cursor()
            
            # Create IVFFlat index for vector similarity search
            create_ivf_index = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_vector_ivf
            ON project_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            
            # Create index on project_id for filtering
            create_project_index = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_project_id
            ON project_embeddings (project_id);
            """
            
            # Create index on content_type for filtering
            create_type_index = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_content_type
            ON project_embeddings (content_type);
            """
            
            # Create composite index for common queries
            create_composite_index = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_project_type
            ON project_embeddings (project_id, content_type);
            """
            
            # Create index on created_at for time-based queries
            create_time_index = """
            CREATE INDEX IF NOT EXISTS idx_project_embeddings_created_at
            ON project_embeddings (created_at DESC);
            """
            
            for index_sql in [
                create_ivf_index,
                create_project_index,
                create_type_index,
                create_composite_index,
                create_time_index
            ]:
                try:
                    cursor.execute(index_sql)
                    self.db.commit()
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
                    self.db.rollback()
            
            cursor.close()
            logger.info("pgvector indexes ensured")
            return True
        
        except Exception as e:
            logger.error(f"Failed to ensure indexes: {e}")
            return False
    
    def optimize_search_query(
        self,
        project_id: str,
        query_embedding: List[float],
        content_type: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> str:
        """
        Generate optimized search query.
        
        Args:
            project_id: Project identifier
            query_embedding: Query embedding vector
            content_type: Optional content type filter
            limit: Result limit
            similarity_threshold: Minimum similarity
            
        Returns:
            Optimized SQL query string
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        if content_type:
            query = f"""
            SELECT 
                id,
                project_id,
                content_type,
                content_id,
                metadata,
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM project_embeddings
            WHERE project_id = '{project_id}'
            AND content_type = '{content_type}'
            AND 1 - (embedding <=> '{embedding_str}'::vector) > {similarity_threshold}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {limit};
            """
        else:
            query = f"""
            SELECT 
                id,
                project_id,
                content_type,
                content_id,
                metadata,
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM project_embeddings
            WHERE project_id = '{project_id}'
            AND 1 - (embedding <=> '{embedding_str}'::vector) > {similarity_threshold}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {limit};
            """
        
        return query


def cached_operation(
    cache_manager: CacheManager,
    cache_key_prefix: str,
    ttl: Optional[int] = None
):
    """
    Decorator for caching operation results.
    
    Args:
        cache_manager: CacheManager instance
        cache_key_prefix: Prefix for cache key
        ttl: Time-to-live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = f"{cache_key_prefix}:{':'.join(str(arg) for arg in args)}"
            
            # Try to get from cache
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


def timed_operation(
    metrics: PerformanceMetrics,
    operation_name: str
):
    """
    Decorator for timing operation execution.
    
    Args:
        metrics: PerformanceMetrics instance
        operation_name: Name of operation for metrics
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_latency(operation_name, duration)
                logger.debug(f"{operation_name} took {duration:.3f}s")
        
        return wrapper
    
    return decorator


# Global instances
_cache_manager = None
_performance_metrics = None
_query_optimizer = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_performance_metrics() -> PerformanceMetrics:
    """Get or create global performance metrics tracker."""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetrics()
    return _performance_metrics


def get_query_optimizer(db_connection) -> QueryOptimizer:
    """Get or create query optimizer."""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer(db_connection)
    return _query_optimizer
