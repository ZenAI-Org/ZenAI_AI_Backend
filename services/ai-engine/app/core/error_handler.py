"""
Error handling and categorization module.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from app.core.logger import get_logger

logger = get_logger(__name__)


class ErrorCategory(Enum):
    """Categories for different types of errors."""
    
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    EXTERNAL_API_ERROR = "external_api_error"
    DATABASE_ERROR = "database_error"
    INTERNAL_ERROR = "internal_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AppError(Exception):
    """Base application error."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize application error."""
        self.message = message
        self.category = category
        self.severity = severity
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class ErrorCategorizer:
    """Categorize exceptions into error types."""
    
    @staticmethod
    def categorize(exception: Exception) -> tuple[ErrorCategory, ErrorSeverity, int]:
        """Categorize an exception and return category, severity, and status code."""
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()
        
        # Validation errors
        if "validation" in exception_message or "invalid" in exception_message:
            return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.LOW, 400
        
        # Authentication errors
        if "unauthorized" in exception_message or "authentication" in exception_message:
            return ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.MEDIUM, 401
        
        # Authorization errors
        if "forbidden" in exception_message or "permission" in exception_message:
            return ErrorCategory.AUTHORIZATION_ERROR, ErrorSeverity.MEDIUM, 403
        
        # Not found errors
        if "not found" in exception_message or "notfound" in exception_message:
            return ErrorCategory.NOT_FOUND_ERROR, ErrorSeverity.LOW, 404
        
        # Rate limit errors
        if "rate limit" in exception_message or "too many requests" in exception_message:
            return ErrorCategory.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM, 429
        
        # Timeout errors
        if "timeout" in exception_message or "timed out" in exception_message:
            return ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.HIGH, 504
        
        # Database errors
        if "database" in exception_message or "sql" in exception_message:
            return ErrorCategory.DATABASE_ERROR, ErrorSeverity.HIGH, 500
        
        # External API errors
        if "api" in exception_message or "external" in exception_message:
            return ErrorCategory.EXTERNAL_API_ERROR, ErrorSeverity.MEDIUM, 502
        
        # Default to internal error
        return ErrorCategory.INTERNAL_ERROR, ErrorSeverity.HIGH, 500


class ErrorRecovery:
    """Error recovery utilities."""
    
    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Determine if an error is retryable."""
        category, _, _ = ErrorCategorizer.categorize(error)
        
        retryable_categories = {
            ErrorCategory.TIMEOUT_ERROR,
            ErrorCategory.RATE_LIMIT_ERROR,
            ErrorCategory.EXTERNAL_API_ERROR,
        }
        
        return category in retryable_categories
    
    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay."""
        return base_delay * (2 ** attempt)
    
    @staticmethod
    def should_retry(
        attempt: int,
        max_retries: int = 3,
        error: Optional[Exception] = None,
    ) -> bool:
        """Determine if should retry based on attempt count and error type."""
        if attempt >= max_retries:
            return False
        
        if error and not ErrorRecovery.is_retryable(error):
            return False
        
        return True


class APIErrorHandler:
    """Handle errors from external API calls."""
    
    def __init__(self):
        """Initialize API error handler."""
        self.logger = get_logger(__name__)
    
    def handle_api_error(
        self,
        error: Exception,
        api_name: str,
        endpoint: str,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """Handle and log API error."""
        category, severity, status_code = ErrorCategorizer.categorize(error)
        
        error_info = {
            "api": api_name,
            "endpoint": endpoint,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "severity": severity.value,
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Critical API error from {api_name}",
                extra_fields=error_info,
            )
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"High severity API error from {api_name}",
                extra_fields=error_info,
            )
        else:
            self.logger.warning(
                f"API error from {api_name}",
                extra_fields=error_info,
            )
        
        return error_info
