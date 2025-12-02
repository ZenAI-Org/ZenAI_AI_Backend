"""
Tests for performance optimization, caching, and batch processing.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from app.core.performance_optimizer import (
    CacheManager,
    PerformanceMetrics,
    QueryOptimizer,
    cached_operation,
    timed_operation,
    get_cache_manager,
    get_performance_metrics
)
from app.queue.batch_processor import (
    BatchProcessor,
    BatchJob,
    BatchResult,
    get_batch_processor
)
from app.core.context_retriever import ContextRetriever


class TestCacheManager:
    """Test suite for cache management."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = MagicMock()
        mock_client.get = Mock(return_value=None)
        mock_client.setex = Mock(return_value=True)
        mock_client.delete = Mock(return_value=1)
        mock_client.keys = Mock(return_value=[])
        return mock_client
    
    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create cache manager with mock Redis."""
        return CacheManager(redis_client=mock_redis, default_ttl=3600)
    
    def test_cache_set_and_get(self, cache_manager, mock_redis):
        """Test setting and getting cache values."""
        # Arrange
        key = "test_key"
        value = {"data": "test_value"}
        mock_redis.get.return_value = json.dumps(value).encode()
        
        # Act
        cache_manager.set(key, value)
        result = cache_manager.get(key)
        
        # Assert
        mock_redis.setex.assert_called_once()
        assert result == value
    
    def test_cache_miss(self, cache_manager, mock_redis):
        """Test cache miss returns None."""
        # Arrange
        key = "nonexistent_key"
        mock_redis.get.return_value = None
        
        # Act
        result = cache_manager.get(key)
        
        # Assert
        assert result is None
    
    def test_cache_delete(self, cache_manager, mock_redis):
        """Test deleting cache values."""
        # Arrange
        key = "test_key"
        mock_redis.delete.return_value = 1
        
        # Act
        deleted = cache_manager.delete(key)
        
        # Assert
        assert deleted is True
        mock_redis.delete.assert_called_once_with(key)
    
    def test_cache_clear_pattern(self, cache_manager, mock_redis):
        """Test clearing cache by pattern."""
        # Arrange
        pattern = "context:project_*"
        mock_redis.keys.return_value = ["context:project_1", "context:project_2"]
        mock_redis.delete.return_value = 2
        
        # Act
        deleted_count = cache_manager.clear_pattern(pattern)
        
        # Assert
        assert deleted_count == 2
        mock_redis.keys.assert_called_once_with(pattern)
    
    def test_get_context_cache(self, cache_manager, mock_redis):
        """Test getting cached context."""
        # Arrange
        project_id = "proj_1"
        query = "test query"
        context = {"summaries": [], "decisions": []}
        mock_redis.get.return_value = json.dumps(context).encode()
        
        # Act
        result = cache_manager.get_context(project_id, query)
        
        # Assert
        assert result == context
    
    def test_set_context_cache(self, cache_manager, mock_redis):
        """Test setting cached context."""
        # Arrange
        project_id = "proj_1"
        query = "test query"
        context = {"summaries": [], "decisions": []}
        
        # Act
        success = cache_manager.set_context(project_id, query, context)
        
        # Assert
        assert success is True
        mock_redis.setex.assert_called_once()
    
    def test_get_suggestions_cache(self, cache_manager, mock_redis):
        """Test getting cached suggestions."""
        # Arrange
        project_id = "proj_1"
        suggestions = {"pending_tasks": [], "blockers": []}
        mock_redis.get.return_value = json.dumps(suggestions).encode()
        
        # Act
        result = cache_manager.get_suggestions(project_id)
        
        # Assert
        assert result == suggestions
    
    def test_set_suggestions_cache(self, cache_manager, mock_redis):
        """Test setting cached suggestions."""
        # Arrange
        project_id = "proj_1"
        suggestions = {"pending_tasks": [], "blockers": []}
        
        # Act
        success = cache_manager.set_suggestions(project_id, suggestions)
        
        # Assert
        assert success is True
        mock_redis.setex.assert_called_once()
    
    def test_get_conversation_history(self, cache_manager, mock_redis):
        """Test getting cached conversation history."""
        # Arrange
        user_id = "user_1"
        project_id = "proj_1"
        history = [{"role": "user", "content": "Hello"}]
        mock_redis.get.return_value = json.dumps(history).encode()
        
        # Act
        result = cache_manager.get_conversation_history(user_id, project_id)
        
        # Assert
        assert result == history
    
    def test_set_conversation_history(self, cache_manager, mock_redis):
        """Test setting cached conversation history."""
        # Arrange
        user_id = "user_1"
        project_id = "proj_1"
        history = [{"role": "user", "content": "Hello"}]
        
        # Act
        success = cache_manager.set_conversation_history(user_id, project_id, history)
        
        # Assert
        assert success is True
        mock_redis.setex.assert_called_once()


class TestPerformanceMetrics:
    """Test suite for performance metrics tracking."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = MagicMock()
        mock_client.lpush = Mock(return_value=1)
        mock_client.ltrim = Mock(return_value=True)
        mock_client.expire = Mock(return_value=True)
        mock_client.incr = Mock(return_value=1)
        mock_client.get = Mock(return_value=b"5")
        mock_client.lrange = Mock(return_value=[])
        return mock_client
    
    @pytest.fixture
    def metrics(self, mock_redis):
        """Create performance metrics with mock Redis."""
        return PerformanceMetrics(redis_client=mock_redis)
    
    def test_record_latency(self, metrics, mock_redis):
        """Test recording operation latency."""
        # Arrange
        operation = "semantic_search"
        duration = 0.5
        
        # Act
        metrics.record_latency(operation, duration)
        
        # Assert
        mock_redis.lpush.assert_called_once()
        mock_redis.ltrim.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    def test_record_cache_hit(self, metrics, mock_redis):
        """Test recording cache hit."""
        # Arrange
        cache_type = "context"
        
        # Act
        metrics.record_cache_hit(cache_type)
        
        # Assert
        mock_redis.incr.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    def test_record_cache_miss(self, metrics, mock_redis):
        """Test recording cache miss."""
        # Arrange
        cache_type = "context"
        
        # Act
        metrics.record_cache_miss(cache_type)
        
        # Assert
        mock_redis.incr.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    def test_get_cache_stats(self, metrics, mock_redis):
        """Test getting cache statistics."""
        # Arrange
        cache_type = "context"
        mock_redis.get.side_effect = [b"10", b"2"]  # 10 hits, 2 misses
        
        # Act
        stats = metrics.get_cache_stats(cache_type)
        
        # Assert
        assert stats["cache_type"] == cache_type
        assert stats["hits"] == 10
        assert stats["misses"] == 2
        assert stats["total"] == 12
        assert stats["hit_rate"] == pytest.approx(83.33, rel=0.1)
    
    def test_get_latency_stats(self, metrics, mock_redis):
        """Test getting latency statistics."""
        # Arrange
        operation = "semantic_search"
        latency_entries = [
            json.dumps({"duration": 0.5, "timestamp": "2024-01-01T00:00:00"}),
            json.dumps({"duration": 0.6, "timestamp": "2024-01-01T00:00:01"}),
            json.dumps({"duration": 0.4, "timestamp": "2024-01-01T00:00:02"})
        ]
        mock_redis.lrange.return_value = latency_entries
        
        # Act
        stats = metrics.get_latency_stats(operation)
        
        # Assert
        assert stats["operation"] == operation
        assert stats["count"] == 3
        assert stats["avg"] == pytest.approx(0.5, rel=0.1)
        assert stats["min"] == 0.4
        assert stats["max"] == 0.6


class TestQueryOptimizer:
    """Test suite for pgvector query optimization."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.commit = Mock()
        mock_conn.rollback = Mock()
        return mock_conn
    
    @pytest.fixture
    def query_optimizer(self, mock_db):
        """Create query optimizer with mock database."""
        return QueryOptimizer(mock_db)
    
    def test_ensure_indexes(self, query_optimizer, mock_db):
        """Test ensuring pgvector indexes exist."""
        # Arrange
        mock_cursor = mock_db.cursor.return_value
        
        # Act
        success = query_optimizer.ensure_indexes()
        
        # Assert
        assert success is True
        mock_cursor.execute.assert_called()
        mock_db.commit.assert_called()
    
    def test_optimize_search_query_with_content_type(self, query_optimizer):
        """Test generating optimized search query with content type filter."""
        # Arrange
        project_id = "proj_1"
        query_embedding = [0.1, 0.2, 0.3]
        content_type = "summary"
        
        # Act
        query = query_optimizer.optimize_search_query(
            project_id=project_id,
            query_embedding=query_embedding,
            content_type=content_type,
            limit=5
        )
        
        # Assert
        assert "project_id" in query
        assert "content_type" in query
        assert "similarity" in query
        assert "ORDER BY" in query
    
    def test_optimize_search_query_without_content_type(self, query_optimizer):
        """Test generating optimized search query without content type filter."""
        # Arrange
        project_id = "proj_1"
        query_embedding = [0.1, 0.2, 0.3]
        
        # Act
        query = query_optimizer.optimize_search_query(
            project_id=project_id,
            query_embedding=query_embedding,
            limit=5
        )
        
        # Assert
        assert "project_id" in query
        assert "similarity" in query
        assert "ORDER BY" in query


class TestBatchProcessor:
    """Test suite for batch processing."""
    
    @pytest.fixture
    def mock_job_queue(self):
        """Create mock job queue."""
        mock_queue = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job_123"
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {"text": "transcript"}
        mock_queue.enqueue_job = Mock(return_value=mock_job)
        mock_queue.get_job = Mock(return_value=mock_job)
        return mock_queue
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics."""
        mock_m = MagicMock()
        mock_m.record_latency = Mock()
        return mock_m
    
    @pytest.fixture
    def batch_processor(self, mock_job_queue, mock_metrics):
        """Create batch processor with mocks."""
        return BatchProcessor(
            job_queue_manager=mock_job_queue,
            metrics=mock_metrics,
            batch_size=5,
            max_concurrent=3
        )
    
    @pytest.mark.asyncio
    async def test_process_batch(self, batch_processor):
        """Test processing a batch of meetings."""
        # Arrange
        meetings = [
            {"meeting_id": "m1", "s3_key": "s3://bucket/m1.mp3"},
            {"meeting_id": "m2", "s3_key": "s3://bucket/m2.mp3"},
            {"meeting_id": "m3", "s3_key": "s3://bucket/m3.mp3"}
        ]
        project_id = "proj_1"
        
        # Act
        result = await batch_processor.process_batch(
            meetings=meetings,
            project_id=project_id
        )
        
        # Assert
        assert result.batch_id is not None
        assert result.total_jobs == 3
        assert result.completed_jobs >= 0
        assert result.failed_jobs >= 0
        assert result.duration_seconds is not None
    
    @pytest.mark.asyncio
    async def test_process_batch_with_pipeline(self, batch_processor):
        """Test processing batch through pipeline."""
        # Arrange
        meetings = [
            {"meeting_id": "m1", "s3_key": "s3://bucket/m1.mp3"},
            {"meeting_id": "m2", "s3_key": "s3://bucket/m2.mp3"}
        ]
        pipeline = ["transcription", "summarization", "task_extraction"]
        project_id = "proj_1"
        
        # Act
        result = await batch_processor.process_batch_with_pipeline(
            meetings=meetings,
            pipeline=pipeline,
            project_id=project_id
        )
        
        # Assert
        assert result.batch_id is not None
        assert result.total_jobs == 2
        assert result.duration_seconds is not None
    
    def test_batch_job_creation(self):
        """Test creating batch jobs."""
        # Arrange
        job_id = "job_123"
        meeting_id = "m1"
        s3_key = "s3://bucket/m1.mp3"
        
        # Act
        batch_job = BatchJob(
            job_id=job_id,
            meeting_id=meeting_id,
            s3_key=s3_key
        )
        
        # Assert
        assert batch_job.job_id == job_id
        assert batch_job.meeting_id == meeting_id
        assert batch_job.s3_key == s3_key
        assert batch_job.status == "queued"
    
    def test_batch_result_success_rate(self):
        """Test calculating batch result success rate."""
        # Arrange
        batch_jobs = [
            BatchJob(job_id="j1", meeting_id="m1", s3_key="s3://m1", status="completed"),
            BatchJob(job_id="j2", meeting_id="m2", s3_key="s3://m2", status="completed"),
            BatchJob(job_id="j3", meeting_id="m3", s3_key="s3://m3", status="error")
        ]
        
        batch_result = BatchResult(
            batch_id="batch_1",
            total_jobs=3,
            completed_jobs=2,
            failed_jobs=1,
            jobs=batch_jobs,
            started_at=datetime.now()
        )
        
        # Act
        success_rate = batch_result.success_rate()
        
        # Assert
        assert success_rate == pytest.approx(66.67, rel=0.1)


class TestContextRetrieverWithCaching:
    """Test context retriever with caching."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        return MagicMock()
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        mock_cache = MagicMock()
        mock_cache.get_context = Mock(return_value=None)
        mock_cache.set_context = Mock(return_value=True)
        return mock_cache
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics."""
        mock_m = MagicMock()
        mock_m.record_cache_hit = Mock()
        mock_m.record_cache_miss = Mock()
        return mock_m
    
    @pytest.fixture
    def context_retriever(self, mock_db, mock_cache_manager, mock_metrics):
        """Create context retriever with mocks."""
        with patch('app.core.context_retriever.EmbeddingStore'):
            retriever = ContextRetriever(
                db_connection=mock_db,
                cache_manager=mock_cache_manager,
                metrics=mock_metrics
            )
            retriever.embedding_store = MagicMock()
            retriever.embedding_store.semantic_search = Mock(return_value=[])
            return retriever
    
    def test_retrieve_context_cache_hit(self, context_retriever, mock_cache_manager, mock_metrics):
        """Test context retrieval with cache hit."""
        # Arrange
        project_id = "proj_1"
        query = "test query"
        cached_context = {"summaries": [], "decisions": [], "blockers": []}
        mock_cache_manager.get_context.return_value = cached_context
        
        # Act
        result = context_retriever.retrieve_meeting_context(
            project_id=project_id,
            query=query,
            use_cache=True
        )
        
        # Assert
        assert result == cached_context
        mock_metrics.record_cache_hit.assert_called_once_with("context")
        mock_cache_manager.get_context.assert_called_once()
    
    def test_retrieve_context_cache_miss(self, context_retriever, mock_cache_manager, mock_metrics):
        """Test context retrieval with cache miss."""
        # Arrange
        project_id = "proj_1"
        query = "test query"
        mock_cache_manager.get_context.return_value = None
        
        # Act
        result = context_retriever.retrieve_meeting_context(
            project_id=project_id,
            query=query,
            use_cache=True
        )
        
        # Assert
        assert "summaries" in result
        assert "decisions" in result
        assert "blockers" in result
        mock_metrics.record_cache_miss.assert_called_once_with("context")
        mock_cache_manager.set_context.assert_called_once()
    
    def test_retrieve_context_without_cache(self, context_retriever, mock_cache_manager):
        """Test context retrieval without caching."""
        # Arrange
        project_id = "proj_1"
        query = "test query"
        
        # Act
        result = context_retriever.retrieve_meeting_context(
            project_id=project_id,
            query=query,
            use_cache=False
        )
        
        # Assert
        assert "summaries" in result
        mock_cache_manager.get_context.assert_not_called()
        mock_cache_manager.set_context.assert_not_called()


class TestCachedOperationDecorator:
    """Test cached operation decorator."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        mock_cache = MagicMock()
        mock_cache.get = Mock(return_value=None)
        mock_cache.set = Mock(return_value=True)
        return mock_cache
    
    def test_cached_operation_decorator(self, mock_cache_manager):
        """Test cached operation decorator functionality."""
        # Arrange
        call_count = 0
        
        @cached_operation(mock_cache_manager, "test_op")
        def expensive_operation(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Act - First call (cache miss)
        result1 = expensive_operation(1, 2)
        
        # Assert
        assert result1 == 3
        assert call_count == 1
        mock_cache_manager.set.assert_called_once()
        
        # Act - Second call (cache hit)
        mock_cache_manager.get.return_value = 3
        result2 = expensive_operation(1, 2)
        
        # Assert
        assert result2 == 3
        assert call_count == 1  # Should not increment
        mock_cache_manager.get.assert_called()


class TestTimedOperationDecorator:
    """Test timed operation decorator."""
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics."""
        mock_m = MagicMock()
        mock_m.record_latency = Mock()
        return mock_m
    
    def test_timed_operation_decorator(self, mock_metrics):
        """Test timed operation decorator functionality."""
        # Arrange
        @timed_operation(mock_metrics, "test_op")
        def slow_operation():
            return "result"
        
        # Act
        result = slow_operation()
        
        # Assert
        assert result == "result"
        mock_metrics.record_latency.assert_called_once()
        call_args = mock_metrics.record_latency.call_args
        assert call_args[0][0] == "test_op"
        assert call_args[0][1] >= 0  # Duration should be non-negative
