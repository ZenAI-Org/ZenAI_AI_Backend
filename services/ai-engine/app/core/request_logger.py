"""
Request and response logging for API calls.
"""

import json
import time
from typing import Any, Dict, Optional
from datetime import datetime
from app.core.logger import get_logger

logger = get_logger(__name__)


class APICallLogger:
    """Log API calls with request/response details."""
    
    @staticmethod
    def log_request(
        api_name: str,
        endpoint: str,
        method: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log outgoing API request."""
        log_data = {
            "api": api_name,
            "endpoint": endpoint,
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if headers:
            # Sanitize sensitive headers
            sanitized_headers = APICallLogger._sanitize_headers(headers)
            log_data["headers"] = sanitized_headers
        
        if query_params:
            log_data["query_params"] = query_params
        
        if body:
            log_data["body"] = APICallLogger._sanitize_body(body)
        
        logger.debug(
            f"API request: {method} {endpoint}",
            extra_fields=log_data,
        )
    
    @staticmethod
    def log_response(
        api_name: str,
        endpoint: str,
        status_code: int,
        response_body: Optional[Any] = None,
        duration_ms: float = 0,
    ) -> None:
        """Log API response."""
        log_data = {
            "api": api_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if response_body:
            log_data["response"] = APICallLogger._sanitize_body(response_body)
        
        # Determine log level based on status code
        if 200 <= status_code < 300:
            logger.debug(
                f"API response: {status_code}",
                extra_fields=log_data,
            )
        elif 400 <= status_code < 500:
            logger.warning(
                f"API response: {status_code}",
                extra_fields=log_data,
            )
        else:
            logger.error(
                f"API response: {status_code}",
                extra_fields=log_data,
            )
    
    @staticmethod
    def log_api_call(
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        request_body: Optional[Any] = None,
        response_body: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log complete API call with request and response."""
        log_data = {
            "api": api_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if request_body:
            log_data["request"] = APICallLogger._sanitize_body(request_body)
        
        if response_body:
            log_data["response"] = APICallLogger._sanitize_body(response_body)
        
        if error:
            log_data["error"] = error
        
        # Determine log level
        if error or status_code >= 400:
            logger.error(
                f"API call failed: {method} {endpoint}",
                extra_fields=log_data,
            )
        elif 200 <= status_code < 300:
            logger.info(
                f"API call succeeded: {method} {endpoint}",
                extra_fields=log_data,
            )
        else:
            logger.warning(
                f"API call: {method} {endpoint}",
                extra_fields=log_data,
            )
    
    @staticmethod
    def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_keys = {
            "authorization",
            "x-api-key",
            "x-auth-token",
            "cookie",
            "x-csrf-token",
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def _sanitize_body(body: Any) -> Any:
        """Remove sensitive information from request/response body."""
        if isinstance(body, dict):
            sensitive_keys = {
                "password",
                "token",
                "api_key",
                "secret",
                "authorization",
            }
            
            sanitized = {}
            for key, value in body.items():
                if key.lower() in sensitive_keys:
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = APICallLogger._sanitize_body(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        APICallLogger._sanitize_body(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value
            
            return sanitized
        
        return body


class APICallTimer:
    """Context manager for timing API calls."""
    
    def __init__(self, api_name: str, endpoint: str):
        """Initialize timer."""
        self.api_name = api_name
        self.endpoint = endpoint
        self.start_time = None
        self.duration_ms = 0
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log."""
        self.duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type:
            logger.error(
                f"API call error: {self.api_name}",
                extra_fields={
                    "api": self.api_name,
                    "endpoint": self.endpoint,
                    "duration_ms": round(self.duration_ms, 2),
                    "error": str(exc_val),
                },
            )
        else:
            logger.debug(
                f"API call completed: {self.api_name}",
                extra_fields={
                    "api": self.api_name,
                    "endpoint": self.endpoint,
                    "duration_ms": round(self.duration_ms, 2),
                },
            )
