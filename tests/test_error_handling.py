"""
Tests for comprehensive error handling and recovery system.
Validates error tracking, notifications, graceful degradation, and recovery.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from app.core.error_dashboard import ErrorMetrics, get_error_metrics
from app.core.error_notifications import ErrorNotificationManager
from app.core.graceful_degradation import (
    GracefulDegradationHandler,
    DegradationStrategy,
    TranscriptionDegradation,
    SummarizationDegradation,
    TaskExtractionDegradation,
)
from app.core.api_retry import APIRetryHandler, RetryConfig


class TestErrorMetrics:
    """Test error metrics tracking and alerting."""
    
    def test_record_error(self):
        """Test recording an error."""
        metrics = ErrorMetrics()
        
        metrics.record_error(
            error_type="timeout",
            api_name="whisper",
            severity="high",
            message="Request timed out",
        )
        
        assert len(metrics.errors) == 1
        assert metrics.error_counts_by_type["timeout"] == 1
        assert metrics.error_counts_by_api["whisper"] == 1
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = ErrorMetrics()
        
        # Record multiple errors
        for i in range(5):
            metrics.record_error(
                error_type="api_error",
                api_name="openai",
                severity="high" if i < 3 else "low",
                message=f"Error {i}",
            )
        
        error_rate = metrics.get_error_rate()
        assert error_rate > 0
        assert error_rate <= 100
    
    def test_should_alert_threshold(self):
        """Test alert threshold checking."""
        metrics = ErrorMetrics()
        
        # Record critical errors to exceed threshold
        for i in range(10):
            metrics.record_error(
                error_type="critical_error",
                api_name="whisper",
                severity="critical",
                message=f"Critical error {i}",
            )
        
        assert metrics.should_alert(threshold=5.0)
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        metrics = ErrorMetrics()
        
        metrics.record_error(
            error_type="timeout",
            api_name="whisper",
            severity="high",
            message="Timeout",
        )
        
        summary = metrics.get_metrics_summary()
        
        assert summary["total_errors"] == 1
        assert summary["high_errors"] == 1
        assert "error_rate" in summary
        assert "errors_by_type" in summary
        assert "errors_by_api" in summary
    
    def test_get_recent_errors(self):
        """Test retrieving recent errors."""
        metrics = ErrorMetrics()
        
        for i in range(5):
            metrics.record_error(
                error_type="api_error",
                api_name="openai",
                severity="medium",
                message=f"Error {i}",
            )
        
        recent = metrics.get_recent_errors(limit=3)
        assert len(recent) == 3
    
    def test_get_errors_by_api(self):
        """Test retrieving errors by API."""
        metrics = ErrorMetrics()
        
        metrics.record_error(
            error_type="timeout",
            api_name="whisper",
            severity="high",
            message="Whisper timeout",
        )
        metrics.record_error(
            error_type="rate_limit",
            api_name="openai",
            severity="medium",
            message="OpenAI rate limit",
        )
        
        whisper_errors = metrics.get_errors_by_api("whisper")
        assert len(whisper_errors) == 1
        assert whisper_errors[0]["api_name"] == "whisper"
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        metrics = ErrorMetrics()
        
        metrics.record_error(
            error_type="error",
            api_name="api",
            severity="high",
            message="Error",
        )
        
        assert len(metrics.errors) == 1
        
        metrics.reset()
        
        assert len(metrics.errors) == 0
        assert len(metrics.error_counts_by_type) == 0


class TestErrorNotificationManager:
    """Test error notification system."""
    
    def test_get_user_friendly_message(self):
        """Test user-friendly error message generation."""
        manager = ErrorNotificationManager()
        
        message = manager.get_user_friendly_message("whisper_timeout")
        assert "transcription" in message.lower()
        assert "longer" in message.lower()
    
    def test_unknown_error_message(self):
        """Test unknown error message fallback."""
        manager = ErrorNotificationManager()
        
        message = manager.get_user_friendly_message("unknown_error_key")
        assert "unexpected" in message.lower()
    
    def test_notify_error(self):
        """Test error notification."""
        manager = ErrorNotificationManager()
        mock_socketio = Mock()
        manager.set_socketio_manager(mock_socketio)
        
        manager.notify_error(
            project_id="proj_123",
            error_type="timeout",
            api_name="whisper",
            message="Request timed out",
            severity="high",
        )
        
        # Verify error was recorded in metrics
        metrics = get_error_metrics()
        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] > 0
    
    def test_notify_recovery(self):
        """Test recovery notification."""
        manager = ErrorNotificationManager()
        mock_socketio = Mock()
        manager.set_socketio_manager(mock_socketio)
        
        manager.notify_recovery(
            project_id="proj_123",
            api_name="whisper",
            message="Whisper service recovered",
        )
        
        # Should not raise exception
        assert True
    
    def test_notify_alert(self):
        """Test alert notification."""
        manager = ErrorNotificationManager()
        mock_socketio = Mock()
        manager.set_socketio_manager(mock_socketio)
        
        manager.notify_alert(
            project_id="proj_123",
            alert_type="high_error_rate",
            message="Error rate exceeded threshold",
            severity="high",
        )
        
        # Should not raise exception
        assert True
    
    def test_check_and_alert_error_rate(self):
        """Test error rate alert checking."""
        manager = ErrorNotificationManager()
        mock_socketio = Mock()
        manager.set_socketio_manager(mock_socketio)
        
        # Record errors to trigger alert
        metrics = get_error_metrics()
        for i in range(10):
            metrics.record_error(
                error_type="error",
                api_name="api",
                severity="critical",
                message=f"Error {i}",
            )
        
        # Check if alert is triggered
        should_alert = manager.check_and_alert_error_rate("proj_123", threshold=5.0)
        assert isinstance(should_alert, bool)


class TestGracefulDegradation:
    """Test graceful degradation strategies."""
    
    def test_register_fallback(self):
        """Test registering fallback function."""
        handler = GracefulDegradationHandler()
        
        def fallback_func(context):
            return {"fallback": True}
        
        handler.register_fallback("component", fallback_func)
        assert "component" in handler.fallback_functions
    
    def test_register_strategy(self):
        """Test registering degradation strategy."""
        handler = GracefulDegradationHandler()
        
        handler.register_strategy("component", DegradationStrategy.SKIP_COMPONENT)
        assert handler.degradation_strategies["component"] == DegradationStrategy.SKIP_COMPONENT
    
    def test_handle_skip_component(self):
        """Test skip component degradation."""
        handler = GracefulDegradationHandler()
        handler.register_strategy("component", DegradationStrategy.SKIP_COMPONENT)
        
        error = Exception("Component failed")
        result = handler.handle_component_failure("component", error)
        
        assert result["status"] == "degraded"
        assert result["strategy"] == "skip_component"
    
    def test_handle_use_fallback(self):
        """Test fallback degradation."""
        handler = GracefulDegradationHandler()
        
        def fallback_func(context):
            return {"fallback_data": "available"}
        
        handler.register_fallback("component", fallback_func)
        handler.register_strategy("component", DegradationStrategy.USE_FALLBACK)
        
        error = Exception("Component failed")
        result = handler.handle_component_failure("component", error)
        
        assert result["status"] == "degraded"
        assert result["strategy"] == "use_fallback"
        assert result["result"]["fallback_data"] == "available"
    
    def test_handle_partial_result(self):
        """Test partial result degradation."""
        handler = GracefulDegradationHandler()
        handler.register_strategy("component", DegradationStrategy.PARTIAL_RESULT)
        
        error = Exception("Component failed")
        context = {"partial_data": {"key": "value"}}
        result = handler.handle_component_failure("component", error, context)
        
        assert result["status"] == "degraded"
        assert result["strategy"] == "partial_result"
    
    def test_transcription_degradation(self):
        """Test transcription degradation."""
        error = Exception("Transcription failed")
        result = TranscriptionDegradation.handle_transcription_failure(
            meeting_id="meeting_123",
            error=error,
            partial_transcript="Partial transcript...",
        )
        
        assert result["status"] == "degraded"
        assert result["component"] == "transcription"
        assert result["partial_transcript"] == "Partial transcript..."
    
    def test_summarization_degradation(self):
        """Test summarization degradation."""
        error = Exception("Summarization failed")
        result = SummarizationDegradation.handle_summarization_failure(
            meeting_id="meeting_123",
            transcript="Full transcript available",
            error=error,
        )
        
        assert result["status"] == "degraded"
        assert result["component"] == "summarization"
        assert result["available_data"]["transcript"] == "Full transcript available"
    
    def test_task_extraction_degradation(self):
        """Test task extraction degradation."""
        error = Exception("Task extraction failed")
        result = TaskExtractionDegradation.handle_extraction_failure(
            meeting_id="meeting_123",
            transcript="Full transcript available",
            error=error,
            partial_tasks=[{"title": "Task 1"}],
        )
        
        assert result["status"] == "degraded"
        assert result["component"] == "task_extraction"
        assert len(result["available_data"]["tasks"]) == 1


class TestAPIRetryHandler:
    """Test API retry handler with exponential backoff."""
    
    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
        )
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
    
    def test_get_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(base_delay=1.0, max_delay=60.0)
        
        delay_0 = config.get_delay(0)
        delay_1 = config.get_delay(1)
        delay_2 = config.get_delay(2)
        
        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0)
        
        delay = config.get_delay(10)  # Would be 1024 without cap
        assert delay == 10.0
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test successful async retry."""
        handler = APIRetryHandler()
        
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await handler.retry_async(
            test_func,
            api_name="test",
            endpoint="test_endpoint",
        )
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_sync_success(self):
        """Test successful sync retry."""
        handler = APIRetryHandler()
        
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = handler.retry_sync(
            test_func,
            api_name="test",
            endpoint="test_endpoint",
        )
        
        assert result == "success"
        assert call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
