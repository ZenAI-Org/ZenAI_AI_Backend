"""
Middleware for tracking API request metrics.
Records request latency and status codes to Redis for monitoring.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from app.core.performance_optimizer import get_performance_metrics

logger = logging.getLogger(__name__)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = get_performance_metrics()
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            # Record latency
            duration = time.time() - start_time
            operation = f"api:{method}:{path}"
            
            # Group dynamic paths
            if "/meetings/" in path and "/process" in path:
                operation = f"api:{method}:/meetings/:id/process"
            elif "/projects/" in path:
                if "/aipm" in path:
                    operation = f"api:{method}:/projects/:id/aipm"
                elif "/suggestions" in path:
                    operation = f"api:{method}:/projects/:id/suggestions"
            
            self.metrics.record_latency(operation, duration)
            
            return response
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            logger.error(f"Request failed: {e}")
            raise e
