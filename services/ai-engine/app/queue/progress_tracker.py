"""
Progress tracking utility for agents to emit progress updates.
Allows agents to report progress during long-running operations.
"""

import logging
from typing import Optional
from datetime import datetime

from app.queue.job_listeners import get_job_event_listener, JobEventType

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks and emits progress updates for a job.
    Used by agents to report progress during execution.
    """
    
    def __init__(self, job_id: str):
        """
        Initialize progress tracker.
        
        Args:
            job_id: Job ID to track progress for
        """
        self.job_id = job_id
        self.event_listener = get_job_event_listener()
        self.current_progress = 0
        self.last_update_time = datetime.now()
        self.min_update_interval = 5  # Minimum seconds between updates
    
    def update_progress(
        self,
        progress_percent: int,
        message: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Update job progress.
        
        Args:
            progress_percent: Progress percentage (0-100)
            message: Optional progress message
            force: Force update even if within min_update_interval
        """
        # Validate progress
        if progress_percent < 0 or progress_percent > 100:
            logger.warning(f"Invalid progress percentage: {progress_percent}")
            return
        
        # Check if enough time has passed since last update
        elapsed = (datetime.now() - self.last_update_time).total_seconds()
        if not force and elapsed < self.min_update_interval:
            # Still update internal state, just don't emit event
            if progress_percent != self.current_progress:
                self.current_progress = progress_percent
            return
        
        # Only emit if progress changed
        if progress_percent == self.current_progress and not force:
            return
        
        self.current_progress = progress_percent
        self.last_update_time = datetime.now()
        
        # Emit progress event
        self.event_listener.on_job_progress(
            self.job_id,
            progress_percent,
            message,
        )
        
        logger.debug(
            f"Progress updated for job {self.job_id}: {progress_percent}% - {message}"
        )
    
    def set_progress(self, progress_percent: int, message: Optional[str] = None) -> None:
        """
        Set progress to a specific percentage.
        
        Args:
            progress_percent: Progress percentage (0-100)
            message: Optional progress message
        """
        self.update_progress(progress_percent, message, force=True)
    
    def increment_progress(
        self,
        increment: int = 10,
        message: Optional[str] = None,
    ) -> None:
        """
        Increment progress by a percentage.
        
        Args:
            increment: Percentage to increment by (default 10)
            message: Optional progress message
        """
        new_progress = min(self.current_progress + increment, 100)
        self.update_progress(new_progress, message)
    
    def mark_complete(self, message: Optional[str] = None) -> None:
        """
        Mark job as complete (100% progress).
        
        Args:
            message: Optional completion message
        """
        self.set_progress(100, message or "Complete")


def get_progress_tracker(job_id: str) -> ProgressTracker:
    """
    Get a progress tracker for a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(job_id)
