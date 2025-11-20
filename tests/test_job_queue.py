"""
Tests for job queue and Redis integration.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from app.queue.redis_config import RedisConfig, get_redis_config, get_redis_client
from app.queue.job_queue import JobQueueManager, JobType, get_job_queue_manager
from app.queue.job_listeners import (
    JobEventListener,
    JobEventType,
    RetryPolicy,
    get_job_event_listener,
    get_retry_policy,
)
from app.queue.orchestration_engine import (
    AIOrchestrationEngine,
    get_orchestration_engine,
)


class TestRedisConfig:
    """Tests for Redis configuration."""
    
    def test_redis_config_initialization(self):
        """Test Redis config initialization with defaults."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.decode_responses is True
    
    def test_redis_config_custom_values(self):
        """Test Redis config with custom values."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
    
    @patch("app.queue.redis_config.Redis")
    def test_get_connection_pool(self, mock_redis):
        """Test connection pool creation."""
        config = RedisConfig()
        pool = config.get_connection_pool()
        assert pool is not None
        # Verify pool is cached
        pool2 = config.get_connection_pool()
        assert pool is pool2
    
    @patch("app.queue.redis_config.Redis")
    def test_get_client(self, mock_redis):
        """Test Redis client creation."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        config = RedisConfig()
        client = config.get_client()
        assert client is not None
        # Verify client is cached
        client2 = config.get_client()
        assert client is client2


class TestJobQueueManager:
    """Tests for job queue manager."""
    
    @patch("app.queue.job_queue.get_redis_client")
    def test_job_queue_initialization(self, mock_redis):
        """Test job queue manager initialization."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        manager = JobQueueManager()
        assert manager.queue_name == "ai_workflows"
        assert manager.redis_client is not None
    
    @patch("app.queue.job_queue.Queue")
    @patch("app.queue.job_queue.get_redis_client")
    def test_enqueue_job(self, mock_redis, mock_queue_class):
        """Test job enqueueing."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_queue.enqueue.return_value = mock_job
        
        manager = JobQueueManager()
        
        def dummy_func(x):
            return x * 2
        
        job = manager.enqueue_job(
            job_type=JobType.TRANSCRIPTION,
            func=dummy_func,
            args=(5,),
            job_id="job-123",
        )
        
        assert job.id == "job-123"
        mock_queue.enqueue.assert_called_once()
    
    @patch("app.queue.job_queue.Job")
    @patch("app.queue.job_queue.get_redis_client")
    def test_get_job(self, mock_redis, mock_job_class):
        """Test retrieving a job."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job_class.fetch.return_value = mock_job
        
        manager = JobQueueManager()
        job = manager.get_job("job-123")
        
        assert job.id == "job-123"
        mock_job_class.fetch.assert_called_once()
    
    @patch("app.queue.job_queue.get_redis_client")
    def test_get_queue_stats(self, mock_redis):
        """Test getting queue statistics."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        manager = JobQueueManager()
        manager.queue = MagicMock()
        manager.queue.__len__ = MagicMock(return_value=5)
        manager.queue.started_job_registry = MagicMock()
        manager.queue.started_job_registry.__len__ = MagicMock(return_value=2)
        manager.queue.finished_job_registry = MagicMock()
        manager.queue.finished_job_registry.__len__ = MagicMock(return_value=10)
        manager.queue.failed_job_registry = MagicMock()
        manager.queue.failed_job_registry.__len__ = MagicMock(return_value=1)
        manager.queue.scheduled_job_registry = MagicMock()
        manager.queue.scheduled_job_registry.__len__ = MagicMock(return_value=0)
        
        stats = manager.get_queue_stats()
        
        assert stats["queue_name"] == "ai_workflows"
        assert stats["job_count"] == 5
        assert stats["started_job_registry_count"] == 2
        assert stats["finished_job_registry_count"] == 10
        assert stats["failed_job_registry_count"] == 1


class TestJobEventListener:
    """Tests for job event listener."""
    
    def test_event_listener_initialization(self):
        """Test event listener initialization."""
        listener = JobEventListener()
        assert listener is not None
        assert len(listener._listeners) == len(JobEventType)
    
    def test_register_listener(self):
        """Test registering a listener."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.COMPLETED, callback)
        
        assert callback in listener._listeners[JobEventType.COMPLETED]
    
    def test_emit_event(self):
        """Test emitting an event."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.COMPLETED, callback)
        listener.emit_event(
            JobEventType.COMPLETED,
            "job-123",
            {"result": "success"},
        )
        
        callback.assert_called_once_with("job-123", {"result": "success"})
    
    def test_on_job_queued(self):
        """Test job queued event."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.QUEUED, callback)
        listener.on_job_queued("job-123")
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "job-123"
        assert args[1]["status"] == "queued"
    
    def test_on_job_completed(self):
        """Test job completed event."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.COMPLETED, callback)
        listener.on_job_completed("job-123", result={"data": "test"})
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "job-123"
        assert args[1]["status"] == "success"
    
    def test_on_job_failed(self):
        """Test job failed event."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.FAILED, callback)
        listener.on_job_failed("job-123", error="Test error")
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "job-123"
        assert args[1]["status"] == "error"
        assert args[1]["error"] == "Test error"
    
    def test_on_job_progress(self):
        """Test job progress event."""
        listener = JobEventListener()
        callback = Mock()
        
        listener.register_listener(JobEventType.PROGRESS, callback)
        listener.on_job_progress("job-123", progress_percent=50, message="Half done")
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "job-123"
        assert args[1]["progress"] == 50
        assert args[1]["message"] == "Half done"


class TestRetryPolicy:
    """Tests for retry policy."""
    
    def test_retry_policy_initialization(self):
        """Test retry policy initialization."""
        policy = RetryPolicy(max_retries=3, backoff_factor=2.0, initial_delay=1)
        assert policy.max_retries == 3
        assert policy.backoff_factor == 2.0
        assert policy.initial_delay == 1
    
    def test_get_retry_delay(self):
        """Test calculating retry delay with exponential backoff."""
        policy = RetryPolicy(max_retries=3, backoff_factor=2.0, initial_delay=1)
        
        # First retry: 1 * 2^0 = 1 second
        assert policy.get_retry_delay(0) == 1
        
        # Second retry: 1 * 2^1 = 2 seconds
        assert policy.get_retry_delay(1) == 2
        
        # Third retry: 1 * 2^2 = 4 seconds
        assert policy.get_retry_delay(2) == 4
        
        # Beyond max retries
        assert policy.get_retry_delay(3) is None
    
    def test_should_retry(self):
        """Test should_retry logic."""
        policy = RetryPolicy(max_retries=3)
        
        assert policy.should_retry(0) is True
        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False


class TestAIOrchestrationEngine:
    """Tests for AI orchestration engine."""
    
    @patch("app.queue.orchestration_engine.get_job_queue_manager")
    @patch("app.queue.orchestration_engine.get_job_event_listener")
    @patch("app.queue.orchestration_engine.get_retry_policy")
    def test_orchestration_engine_initialization(
        self,
        mock_retry_policy,
        mock_event_listener,
        mock_job_queue,
    ):
        """Test orchestration engine initialization."""
        engine = AIOrchestrationEngine()
        assert engine is not None
        assert engine.job_queue is not None
        assert engine.event_listener is not None
        assert engine.retry_policy is not None
    
    @patch("app.queue.orchestration_engine.get_job_queue_manager")
    @patch("app.queue.orchestration_engine.get_job_event_listener")
    @patch("app.queue.orchestration_engine.get_retry_policy")
    def test_enqueue_transcription_job(
        self,
        mock_retry_policy,
        mock_event_listener,
        mock_job_queue,
    ):
        """Test enqueueing a transcription job."""
        mock_queue_manager = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_queue_manager.enqueue_job.return_value = mock_job
        mock_job_queue.return_value = mock_queue_manager
        
        mock_listener = MagicMock()
        mock_event_listener.return_value = mock_listener
        
        engine = AIOrchestrationEngine()
        job_id = engine.enqueue_transcription_job(
            meeting_id="meeting-123",
            s3_key="s3://bucket/audio.mp3",
        )
        
        assert job_id == "job-123"
        mock_queue_manager.enqueue_job.assert_called_once()
        mock_listener.on_job_queued.assert_called_once_with("job-123")
    
    @patch("app.queue.orchestration_engine.get_job_queue_manager")
    @patch("app.queue.orchestration_engine.get_job_event_listener")
    @patch("app.queue.orchestration_engine.get_retry_policy")
    def test_get_job_status(
        self,
        mock_retry_policy,
        mock_event_listener,
        mock_job_queue,
    ):
        """Test getting job status."""
        mock_queue_manager = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.get_status.return_value = "completed"
        mock_job.created_at = None
        mock_job.started_at = None
        mock_job.ended_at = None
        mock_job.result = {"data": "test"}
        mock_job.is_failed = False
        mock_job.exc_info = None
        mock_job.meta = {"job_type": "transcription"}
        mock_queue_manager.get_job.return_value = mock_job
        mock_job_queue.return_value = mock_queue_manager
        
        engine = AIOrchestrationEngine()
        status = engine.get_job_status("job-123")
        
        assert status["job_id"] == "job-123"
        assert status["status"] == "completed"
        assert status["result"] == {"data": "test"}
    
    @patch("app.queue.orchestration_engine.get_job_queue_manager")
    @patch("app.queue.orchestration_engine.get_job_event_listener")
    @patch("app.queue.orchestration_engine.get_retry_policy")
    def test_get_queue_stats(
        self,
        mock_retry_policy,
        mock_event_listener,
        mock_job_queue,
    ):
        """Test getting queue statistics."""
        mock_queue_manager = MagicMock()
        mock_queue_manager.get_queue_stats.return_value = {
            "queue_name": "ai_workflows",
            "job_count": 5,
        }
        mock_queue_manager.get_worker_count.return_value = 2
        mock_job_queue.return_value = mock_queue_manager
        
        engine = AIOrchestrationEngine()
        stats = engine.get_queue_stats()
        
        assert stats["queue_name"] == "ai_workflows"
        assert stats["job_count"] == 5
        assert stats["worker_count"] == 2
        assert "timestamp" in stats
