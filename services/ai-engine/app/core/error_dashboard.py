"""
Error dashboard and monitoring for tracking error rates and patterns.
Provides real-time error metrics and alerting capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
import threading
from app.core.logger import get_logger

logger = get_logger(__name__)


class ErrorMetrics:
    """Track error metrics for monitoring and alerting."""
    
    def __init__(self, window_size_minutes: int = 60):
        """
        Initialize error metrics tracker.
        
        Args:
            window_size_minutes: Time window for error rate calculation (default 60 minutes)
        """
        self.window_size = timedelta(minutes=window_size_minutes)
        self.errors: List[Dict[str, Any]] = []
        self.error_counts_by_type: Dict[str, int] = defaultdict(int)
        self.error_counts_by_api: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_error(
        self,
        error_type: str,
        api_name: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error (e.g., "timeout", "rate_limit", "api_error")
            api_name: Name of the API that failed (e.g., "whisper", "openai", "notion")
            severity: Severity level ("low", "medium", "high", "critical")
            message: Error message
            details: Additional error details
        """
        with self.lock:
            error_record = {
                "timestamp": datetime.utcnow(),
                "error_type": error_type,
                "api_name": api_name,
                "severity": severity,
                "message": message,
                "details": details or {},
            }
            
            self.errors.append(error_record)
            self.error_counts_by_type[error_type] += 1
            self.error_counts_by_api[api_name] += 1
            
            logger.debug(
                f"Error recorded: {error_type} from {api_name}",
                extra_fields={
                    "error_type": error_type,
                    "api_name": api_name,
                    "severity": severity,
                },
            )
    
    def get_error_rate(self) -> float:
        """
        Get error rate as percentage in the current time window.
        
        Returns:
            Error rate as percentage (0-100)
        """
        with self.lock:
            # Clean up old errors outside the window
            cutoff_time = datetime.utcnow() - self.window_size
            self.errors = [e for e in self.errors if e["timestamp"] > cutoff_time]
            
            if not self.errors:
                return 0.0
            
            # Count errors by severity
            critical_count = sum(1 for e in self.errors if e["severity"] == "critical")
            high_count = sum(1 for e in self.errors if e["severity"] == "high")
            
            # Weight critical errors more heavily
            weighted_errors = (critical_count * 2) + high_count
            total_errors = len(self.errors)
            
            # Calculate rate as percentage
            error_rate = (weighted_errors / max(total_errors, 1)) * 100
            
            return min(error_rate, 100.0)
    
    def should_alert(self, threshold: float = 5.0) -> bool:
        """
        Determine if error rate exceeds alert threshold.
        
        Args:
            threshold: Alert threshold as percentage (default 5%)
            
        Returns:
            True if error rate exceeds threshold
        """
        return self.get_error_rate() > threshold
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of error metrics.
        
        Returns:
            Dictionary with error metrics
        """
        with self.lock:
            # Clean up old errors
            cutoff_time = datetime.utcnow() - self.window_size
            self.errors = [e for e in self.errors if e["timestamp"] > cutoff_time]
            
            total_errors = len(self.errors)
            critical_errors = sum(1 for e in self.errors if e["severity"] == "critical")
            high_errors = sum(1 for e in self.errors if e["severity"] == "high")
            
            return {
                "total_errors": total_errors,
                "critical_errors": critical_errors,
                "high_errors": high_errors,
                "error_rate": self.get_error_rate(),
                "errors_by_type": dict(self.error_counts_by_type),
                "errors_by_api": dict(self.error_counts_by_api),
                "window_minutes": self.window_size.total_seconds() / 60,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error records
        """
        with self.lock:
            # Sort by timestamp descending and return most recent
            sorted_errors = sorted(
                self.errors,
                key=lambda e: e["timestamp"],
                reverse=True,
            )
            
            return [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "error_type": e["error_type"],
                    "api_name": e["api_name"],
                    "severity": e["severity"],
                    "message": e["message"],
                    "details": e["details"],
                }
                for e in sorted_errors[:limit]
            ]
    
    def get_errors_by_api(self, api_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors for a specific API.
        
        Args:
            api_name: Name of the API
            limit: Maximum number of errors to return
            
        Returns:
            List of error records for the API
        """
        with self.lock:
            api_errors = [e for e in self.errors if e["api_name"] == api_name]
            sorted_errors = sorted(
                api_errors,
                key=lambda e: e["timestamp"],
                reverse=True,
            )
            
            return [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "error_type": e["error_type"],
                    "severity": e["severity"],
                    "message": e["message"],
                    "details": e["details"],
                }
                for e in sorted_errors[:limit]
            ]
    
    def reset(self) -> None:
        """Reset all error metrics."""
        with self.lock:
            self.errors.clear()
            self.error_counts_by_type.clear()
            self.error_counts_by_api.clear()
            logger.info("Error metrics reset")


# Global error metrics instance
_error_metrics = None


def get_error_metrics() -> ErrorMetrics:
    """Get or create global error metrics instance."""
    global _error_metrics
    if _error_metrics is None:
        _error_metrics = ErrorMetrics()
    return _error_metrics
