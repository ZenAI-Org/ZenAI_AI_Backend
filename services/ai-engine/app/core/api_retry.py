"""
API retry and recovery utilities with exponential backoff.
"""

import asyncio
from typing import Callable, Any, Optional, TypeVar
from app.core.logger import get_logger
from app.core.error_handler import ErrorRecovery, APIErrorHandler

logger = get_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        """Initialize retry configuration."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt with exponential backoff."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


class APIRetryHandler:
    """Handle retries for API calls with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry handler."""
        self.config = config or RetryConfig()
        self.error_handler = APIErrorHandler()
        self.logger = get_logger(__name__)
    
    async def retry_async(
        self,
        func: Callable[..., Any],
        api_name: str,
        endpoint: str,
        *args,
        **kwargs,
    ) -> Any:
        """Retry an async function with exponential backoff."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(
                    f"Attempting API call to {api_name}",
                    extra_fields={
                        "api": api_name,
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                    },
                )
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(
                        f"API call succeeded after {attempt} retries",
                        extra_fields={
                            "api": api_name,
                            "endpoint": endpoint,
                            "attempts": attempt + 1,
                        },
                    )
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Log the error
                self.error_handler.handle_api_error(
                    e,
                    api_name,
                    endpoint,
                    attempt + 1,
                )
                
                # Check if should retry
                if not ErrorRecovery.should_retry(attempt, self.config.max_retries, e):
                    self.logger.error(
                        f"API call failed and not retryable: {api_name}",
                        extra_fields={
                            "api": api_name,
                            "endpoint": endpoint,
                            "error": str(e),
                        },
                    )
                    raise
                
                # Calculate delay
                delay = self.config.get_delay(attempt)
                self.logger.info(
                    f"Retrying API call after {delay}s",
                    extra_fields={
                        "api": api_name,
                        "endpoint": endpoint,
                        "delay_seconds": delay,
                        "attempt": attempt + 1,
                    },
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_error
    
    def retry_sync(
        self,
        func: Callable[..., Any],
        api_name: str,
        endpoint: str,
        *args,
        **kwargs,
    ) -> Any:
        """Retry a sync function with exponential backoff."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(
                    f"Attempting API call to {api_name}",
                    extra_fields={
                        "api": api_name,
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                    },
                )
                
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(
                        f"API call succeeded after {attempt} retries",
                        extra_fields={
                            "api": api_name,
                            "endpoint": endpoint,
                            "attempts": attempt + 1,
                        },
                    )
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Log the error
                self.error_handler.handle_api_error(
                    e,
                    api_name,
                    endpoint,
                    attempt + 1,
                )
                
                # Check if should retry
                if not ErrorRecovery.should_retry(attempt, self.config.max_retries, e):
                    self.logger.error(
                        f"API call failed and not retryable: {api_name}",
                        extra_fields={
                            "api": api_name,
                            "endpoint": endpoint,
                            "error": str(e),
                        },
                    )
                    raise
                
                # Calculate delay
                delay = self.config.get_delay(attempt)
                self.logger.info(
                    f"Retrying API call after {delay}s",
                    extra_fields={
                        "api": api_name,
                        "endpoint": endpoint,
                        "delay_seconds": delay,
                        "attempt": attempt + 1,
                    },
                )
                
                # Wait before retry
                import time
                time.sleep(delay)
        
        # All retries exhausted
        raise last_error


def get_retry_handler(config: Optional[RetryConfig] = None) -> APIRetryHandler:
    """Get or create API retry handler."""
    return APIRetryHandler(config)
