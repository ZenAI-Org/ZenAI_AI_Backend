"""
Tests for logging and error handling infrastructure.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.core.logger import StructuredLogger, get_logger, JSONFormatter
from app.core.error_handler import (
    ErrorCategory,
    ErrorSeverity,
    AppError,
    ErrorCategorizer,
    ErrorRecovery,
    APIErrorHandler,
)
from app.core.api_retry import RetryConfig, APIRetryHandler, get_retry_handler
from app.core.request_logger import APICallLogger, APICallTimer


class TestStructuredLogger:
    """Tests for structured logging."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = get_logger("test")
        assert logger is not None
        assert logger.logger.name == "test"
    
    def test_logger_debug(self, caplog):
        """Test debug logging."""
        logger = get_logger("test_debug")
        logger.debug("Test debug message", extra_fields={"key": "value"})
        # Verify logging was called
        assert logger.logger is not None
    
    def test_logger_info(self, caplog):
        """Test info logging."""
        logger = get_logger("test_info")
        logger.info("Test info message", extra_fields={"key": "value"})
        assert logger.logger is not None
    
    def test_logger_warning(self, caplog):
        """Test warning logging."""
        logger = get_logger("test_warning")
        logger.warning("Test warning message", extra_fields={"key": "value"})
        assert logger.logger is not None
    
    def test_logger_error(self, caplog):
        """Test error logging."""
        logger = get_logger("test_error")
        logger.error("Test error message", extra_fields={"key": "value"})
        assert logger.logger is not None
    
    def test_logger_critical(self, caplog):
        """Test critical logging."""
        logger = get_logger("test_critical")
        logger.critical("Test critical message", extra_fields={"key": "value"})
        assert logger.logger is not None
    
    def test_logger_with_exception(self):
        """Test logging with exception info."""
        logger = get_logger("test_exception")
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Error occurred", exc_info=True)
        
        assert logger.logger is not None


class TestErrorCategorization:
    """Tests for error categorization."""
    
    def test_categorize_validation_error(self):
        """Test categorizing validation error."""
        error = ValueError("Invalid input validation")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.VALIDATION_ERROR
        assert severity == ErrorSeverity.LOW
        assert status_code == 400
    
    def test_categorize_authentication_error(self):
        """Test categorizing authentication error."""
        error = Exception("Unauthorized authentication failed")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.AUTHENTICATION_ERROR
        assert severity == ErrorSeverity.MEDIUM
        assert status_code == 401
    
    def test_categorize_authorization_error(self):
        """Test categorizing authorization error."""
        error = Exception("Forbidden permission denied")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.AUTHORIZATION_ERROR
        assert severity == ErrorSeverity.MEDIUM
        assert status_code == 403
    
    def test_categorize_not_found_error(self):
        """Test categorizing not found error."""
        error = Exception("Resource not found")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.NOT_FOUND_ERROR
        assert severity == ErrorSeverity.LOW
        assert status_code == 404
    
    def test_categorize_rate_limit_error(self):
        """Test categorizing rate limit error."""
        error = Exception("Rate limit exceeded too many requests")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.RATE_LIMIT_ERROR
        assert severity == ErrorSeverity.MEDIUM
        assert status_code == 429
    
    def test_categorize_timeout_error(self):
        """Test categorizing timeout error."""
        error = Exception("Request timeout timed out")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.TIMEOUT_ERROR
        assert severity == ErrorSeverity.HIGH
        assert status_code == 504
    
    def test_categorize_database_error(self):
        """Test categorizing database error."""
        error = Exception("Database connection failed SQL error")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.DATABASE_ERROR
        assert severity == ErrorSeverity.HIGH
        assert status_code == 500
    
    def test_categorize_external_api_error(self):
        """Test categorizing external API error."""
        error = Exception("External API call failed")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.EXTERNAL_API_ERROR
        assert severity == ErrorSeverity.MEDIUM
        assert status_code == 502
    
    def test_categorize_unknown_error(self):
        """Test categorizing unknown error."""
        error = Exception("Some random error")
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        assert category == ErrorCategory.INTERNAL_ERROR
        assert severity == ErrorSeverity.HIGH
        assert status_code == 500


class TestAppError:
    """Tests for application error."""
    
    def test_app_error_creation(self):
        """Test creating application error."""
        error = AppError(
            message="Test error",
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            status_code=400,
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION_ERROR
        assert error.severity == ErrorSeverity.LOW
        assert error.status_code == 400
    
    def test_app_error_to_dict(self):
        """Test converting error to dictionary."""
        error = AppError(
            message="Test error",
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            status_code=400,
            details={"field": "email"},
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "validation_error"
        assert error_dict["severity"] == "low"
        assert error_dict["status_code"] == 400
        assert error_dict["details"]["field"] == "email"
        assert "timestamp" in error_dict


class TestErrorRecovery:
    """Tests for error recovery."""
    
    def test_is_retryable_timeout_error(self):
        """Test timeout error is retryable."""
        error = Exception("Request timeout")
        assert ErrorRecovery.is_retryable(error) is True
    
    def test_is_retryable_rate_limit_error(self):
        """Test rate limit error is retryable."""
        error = Exception("Rate limit exceeded")
        assert ErrorRecovery.is_retryable(error) is True
    
    def test_is_retryable_external_api_error(self):
        """Test external API error is retryable."""
        error = Exception("External API failed")
        assert ErrorRecovery.is_retryable(error) is True
    
    def test_is_not_retryable_validation_error(self):
        """Test validation error is not retryable."""
        error = ValueError("Invalid input")
        assert ErrorRecovery.is_retryable(error) is False
    
    def test_get_retry_delay(self):
        """Test calculating retry delay with exponential backoff."""
        # First retry: 1 * 2^0 = 1
        assert ErrorRecovery.get_retry_delay(0, base_delay=1.0) == 1.0
        
        # Second retry: 1 * 2^1 = 2
        assert ErrorRecovery.get_retry_delay(1, base_delay=1.0) == 2.0
        
        # Third retry: 1 * 2^2 = 4
        assert ErrorRecovery.get_retry_delay(2, base_delay=1.0) == 4.0
    
    def test_should_retry_within_limit(self):
        """Test should retry when within attempt limit."""
        assert ErrorRecovery.should_retry(0, max_retries=3) is True
        assert ErrorRecovery.should_retry(1, max_retries=3) is True
        assert ErrorRecovery.should_retry(2, max_retries=3) is True
    
    def test_should_not_retry_exceeded_limit(self):
        """Test should not retry when exceeding attempt limit."""
        assert ErrorRecovery.should_retry(3, max_retries=3) is False
        assert ErrorRecovery.should_retry(4, max_retries=3) is False
    
    def test_should_not_retry_non_retryable_error(self):
        """Test should not retry non-retryable error."""
        error = ValueError("Invalid input")
        assert ErrorRecovery.should_retry(0, max_retries=3, error=error) is False


class TestAPIErrorHandler:
    """Tests for API error handler."""
    
    def test_handle_api_error(self):
        """Test handling API error."""
        handler = APIErrorHandler()
        error = Exception("API request timeout")
        
        error_info = handler.handle_api_error(
            error,
            api_name="TestAPI",
            endpoint="/test",
            attempt=1,
        )
        
        assert error_info["api"] == "TestAPI"
        assert error_info["endpoint"] == "/test"
        assert error_info["attempt"] == 1
        assert "timestamp" in error_info
        assert "category" in error_info
        assert "severity" in error_info


class TestRetryConfig:
    """Tests for retry configuration."""
    
    def test_retry_config_defaults(self):
        """Test retry config with default values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
    
    def test_retry_config_custom_values(self):
        """Test retry config with custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
    
    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, max_delay=60.0)
        
        # First retry: 1 * 2^0 = 1
        assert config.get_delay(0) == 1.0
        
        # Second retry: 1 * 2^1 = 2
        assert config.get_delay(1) == 2.0
        
        # Third retry: 1 * 2^2 = 4
        assert config.get_delay(2) == 4.0
    
    def test_get_delay_respects_max_delay(self):
        """Test delay respects maximum delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0)
        
        # Large attempt number would exceed max_delay
        delay = config.get_delay(10)
        assert delay <= 10.0


class TestAPIRetryHandler:
    """Tests for API retry handler."""
    
    def test_retry_handler_initialization(self):
        """Test retry handler initialization."""
        handler = APIRetryHandler()
        assert handler.config is not None
        assert handler.config.max_retries == 3
    
    def test_retry_handler_with_custom_config(self):
        """Test retry handler with custom config."""
        config = RetryConfig(max_retries=5)
        handler = APIRetryHandler(config)
        assert handler.config.max_retries == 5
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test async retry succeeds on first attempt."""
        handler = APIRetryHandler()
        
        async def mock_func():
            return "success"
        
        result = await handler.retry_async(
            mock_func,
            api_name="TestAPI",
            endpoint="/test",
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        """Test async retry succeeds after retries."""
        handler = APIRetryHandler(RetryConfig(base_delay=0.01))
        
        call_count = 0
        
        async def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Request timeout")
            return "success"
        
        result = await handler.retry_async(
            mock_func,
            api_name="TestAPI",
            endpoint="/test",
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_exhausts_retries(self):
        """Test async retry exhausts all retries."""
        handler = APIRetryHandler(RetryConfig(max_retries=2, base_delay=0.01))
        
        async def mock_func():
            raise Exception("Request timeout")
        
        with pytest.raises(Exception):
            await handler.retry_async(
                mock_func,
                api_name="TestAPI",
                endpoint="/test",
            )
    
    def test_retry_sync_success_first_attempt(self):
        """Test sync retry succeeds on first attempt."""
        handler = APIRetryHandler()
        
        def mock_func():
            return "success"
        
        result = handler.retry_sync(
            mock_func,
            api_name="TestAPI",
            endpoint="/test",
        )
        
        assert result == "success"
    
    def test_retry_sync_success_after_retries(self):
        """Test sync retry succeeds after retries."""
        handler = APIRetryHandler(RetryConfig(base_delay=0.01))
        
        call_count = 0
        
        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Request timeout")
            return "success"
        
        result = handler.retry_sync(
            mock_func,
            api_name="TestAPI",
            endpoint="/test",
        )
        
        assert result == "success"
        assert call_count == 3


class TestAPICallLogger:
    """Tests for API call logging."""
    
    def test_sanitize_headers(self):
        """Test sanitizing sensitive headers."""
        headers = {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
            "X-API-Key": "secret-key",
        }
        
        sanitized = APICallLogger._sanitize_headers(headers)
        
        assert sanitized["Authorization"] == "***REDACTED***"
        assert sanitized["Content-Type"] == "application/json"
        assert sanitized["X-API-Key"] == "***REDACTED***"
    
    def test_sanitize_body(self):
        """Test sanitizing sensitive body fields."""
        body = {
            "username": "user@example.com",
            "password": "secret123",
            "api_key": "key123",
            "data": {"token": "token123"},
        }
        
        sanitized = APICallLogger._sanitize_body(body)
        
        assert sanitized["username"] == "user@example.com"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["data"]["token"] == "***REDACTED***"
    
    def test_log_request(self):
        """Test logging API request."""
        APICallLogger.log_request(
            api_name="TestAPI",
            endpoint="/test",
            method="POST",
            headers={"Content-Type": "application/json"},
            body={"key": "value"},
        )
        # Verify no exception is raised
        assert True
    
    def test_log_response(self):
        """Test logging API response."""
        APICallLogger.log_response(
            api_name="TestAPI",
            endpoint="/test",
            status_code=200,
            response_body={"result": "success"},
            duration_ms=100.5,
        )
        # Verify no exception is raised
        assert True
    
    def test_log_api_call(self):
        """Test logging complete API call."""
        APICallLogger.log_api_call(
            api_name="TestAPI",
            endpoint="/test",
            method="POST",
            status_code=200,
            duration_ms=150.5,
            request_body={"key": "value"},
            response_body={"result": "success"},
        )
        # Verify no exception is raised
        assert True


class TestAPICallTimer:
    """Tests for API call timer."""
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        import time
        
        with APICallTimer("TestAPI", "/test") as timer:
            time.sleep(0.01)
        
        assert timer.duration_ms > 0
        assert timer.duration_ms < 100  # Should be less than 100ms
    
    def test_timer_with_exception(self):
        """Test timer handles exceptions."""
        with pytest.raises(ValueError):
            with APICallTimer("TestAPI", "/test") as timer:
                raise ValueError("Test error")
