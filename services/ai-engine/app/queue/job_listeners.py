"""
Job event listeners for monitoring and handling job state changes.
Implements callbacks for job status transitions and progress updates.
"""

import logging
from typing import Callable, Dict, Any, Optional
from enum import Enum

from rq.job import Job, JobStatus

logger = logging.getLogger(__name__)


class JobEventType(str, Enum):
    """Types of job events."""
    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PROGRESS = "progress"


class JobEventListener:
    """Handles job event callbacks and notifications."""
    
    def __init__(self):
        """Initialize job event listener."""
        self._listeners: Dict[JobEventType, list[Callable]] = {
            event_type: [] for event_type in JobEventType
        }
        logger.info("Job event listener initialized")
    
    def register_listener(
        self,
        event_type: JobEventType,
        callback: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """
        Register a callback for a job event.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function(job_id, event_data)
        """
        self._listeners[event_type].append(callback)
        logger.info(f"Listener registered for event: {event_type.value}")
    
    def unregister_listener(
        self,
        event_type: JobEventType,
        callback: Callable,
    ) -> None:
        """
        Unregister a callback for a job event.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)
            logger.info(f"Listener unregistered for event: {event_type.value}")
    
    def emit_event(
        self,
        event_type: JobEventType,
        job_id: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a job event to all registered listeners.
        
        Args:
            event_type: Type of event
            job_id: Job ID
            event_data: Optional event data
        """
        event_data = event_data or {}
        
        for callback in self._listeners[event_type]:
            try:
                callback(job_id, event_data)
            except Exception as e:
                logger.error(
                    f"Error in event listener for {event_type.value}: {e}"
                )
    
    def on_job_queued(self, job_id: str) -> None:
        """
        Handle job queued event.
        
        Args:
            job_id: Job ID
        """
        self.emit_event(
            JobEventType.QUEUED,
            job_id,
            {"status": "queued"},
        )
        logger.info(f"Job queued: {job_id}")
    
    def on_job_active(self, job_id: str) -> None:
        """
        Handle job active event.
        
        Args:
            job_id: Job ID
        """
        self.emit_event(
            JobEventType.ACTIVE,
            job_id,
            {"status": "running"},
        )
        logger.info(f"Job active: {job_id}")
    
    def on_job_completed(self, job_id: str, result: Any = None) -> None:
        """
        Handle job completed event.
        
        Args:
            job_id: Job ID
            result: Job result
        """
        self.emit_event(
            JobEventType.COMPLETED,
            job_id,
            {"status": "success", "result": result},
        )
        logger.info(f"Job completed: {job_id}")
    
    def on_job_failed(self, job_id: str, error: str = None) -> None:
        """
        Handle job failed event.
        
        Args:
            job_id: Job ID
            error: Error message
        """
        self.emit_event(
            JobEventType.FAILED,
            job_id,
            {"status": "error", "error": error},
        )
        logger.error(f"Job failed: {job_id} - {error}")
    
    def on_job_progress(
        self,
        job_id: str,
        progress_percent: int,
        message: str = None,
    ) -> None:
        """
        Handle job progress event.
        
        Args:
            job_id: Job ID
            progress_percent: Progress percentage (0-100)
            message: Optional progress message
        """
        self.emit_event(
            JobEventType.PROGRESS,
            job_id,
            {
                "progress": progress_percent,
                "message": message,
            },
        )
        logger.debug(f"Job progress: {job_id} - {progress_percent}%")


class RetryPolicy:
    """Retry policy for failed jobs."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: int = 1,
    ):
        """
        Initialize retry policy.
        
        Args:
            max_retries: Maximum number of retries (default 3)
            backoff_factor: Exponential backoff factor (default 2.0)
            initial_delay: Initial delay in seconds (default 1)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
    
    def get_retry_delay(self, attempt: int) -> int:
        """
        Calculate retry delay with exponential backoff.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if attempt >= self.max_retries:
            return None
        
        delay = int(self.initial_delay * (self.backoff_factor ** attempt))
        return delay
    
    def should_retry(self, attempt: int) -> bool:
        """
        Check if job should be retried.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            True if should retry, False otherwise
        """
        return attempt < self.max_retries


# Global job event listener instance
_job_event_listener = None


def get_job_event_listener() -> JobEventListener:
    """Get or create global job event listener."""
    global _job_event_listener
    if _job_event_listener is None:
        _job_event_listener = JobEventListener()
    return _job_event_listener


# Global retry policy instance
_retry_policy = None


def get_retry_policy() -> RetryPolicy:
    """Get or create global retry policy."""
    global _retry_policy
    if _retry_policy is None:
        _retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1,
        )
    return _retry_policy
