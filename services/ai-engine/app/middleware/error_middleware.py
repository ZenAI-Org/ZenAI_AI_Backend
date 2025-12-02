"""
Error handling middleware for FastAPI.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
from app.core.logger import get_logger
from app.core.error_handler import (
    AppError,
    ErrorCategorizer,
    ErrorSeverity,
)

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors and logging requests/responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Process request and handle errors."""
        request_id = request.headers.get("X-Request-ID", "unknown")
        start_time = time.time()
        
        try:
            # Log incoming request
            logger.info(
                f"Incoming request: {request.method} {request.url.path}",
                extra_fields={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else "unknown",
                },
            )
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code}",
                extra_fields={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            
            return response
        
        except AppError as e:
            # Handle application errors
            duration = time.time() - start_time
            
            logger.warning(
                f"Application error: {e.message}",
                extra_fields={
                    "request_id": request_id,
                    "error_category": e.category.value,
                    "error_severity": e.severity.value,
                    "status_code": e.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "details": e.details,
                },
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict(),
            )
        
        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time
            category, severity, status_code = ErrorCategorizer.categorize(e)
            
            error_data = {
                "message": str(e),
                "category": category.value,
                "severity": severity.value,
                "status_code": status_code,
            }
            
            log_level = "critical" if severity == ErrorSeverity.CRITICAL else "error"
            
            if log_level == "critical":
                logger.critical(
                    f"Unexpected error: {str(e)}",
                    extra_fields={
                        "request_id": request_id,
                        "error_type": type(e).__name__,
                        "duration_ms": round(duration * 1000, 2),
                    },
                    exc_info=True,
                )
            else:
                logger.error(
                    f"Unexpected error: {str(e)}",
                    extra_fields={
                        "request_id": request_id,
                        "error_type": type(e).__name__,
                        "duration_ms": round(duration * 1000, 2),
                    },
                    exc_info=True,
                )
            
            return JSONResponse(
                status_code=status_code,
                content=error_data,
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Log request and response details."""
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Log request headers and body for debugging
        logger.debug(
            f"Request details: {request.method} {request.url.path}",
            extra_fields={
                "request_id": request_id,
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
            },
        )
        
        response = await call_next(request)
        return response
