"""
Job queue management using RQ (Redis Queue).
Handles enqueueing, processing, and monitoring of background AI workflows.
"""

import logging
from typing import Any, Callable, Dict, Optional
from datetime import timedelta
from enum import Enum

from rq import Queue, Worker
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError

from app.queue.redis_config import get_redis_client

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    """Types of AI workflow jobs."""
    TRANSCRIPTION = "transcription"
    SUMMARIZATION = "summarization"
    TASK_EXTRACTION = "task_extraction"
    AIPM_ANALYSIS = "aipm_analysis"
    CHAT = "chat"
    SUGGESTIONS = "suggestions"
    NOTION_SYNC = "notion_sync"
    WEEKLY_REPORT = "weekly_report"


class JobQueueManager:
    """Manages job enqueueing, processing, and monitoring."""
    
    def __init__(self, queue_name: str = "ai_workflows"):
        """
        Initialize job queue manager.
        
        Args:
            queue_name: Name of the RQ queue
        """
        self.queue_name = queue_name
        self.redis_client = get_redis_client()
        self.queue = Queue(queue_name, connection=self.redis_client)
        logger.info(f"Job queue initialized: {queue_name}")
    
    def enqueue_job(
        self,
        job_type: JobType,
        func: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        job_id: Optional[str] = None,
        timeout: int = 3600,
        result_ttl: int = 86400,
        failure_ttl: int = 604800,
    ) -> Job:
        """
        Enqueue a job for background processing.
        
        Args:
            job_type: Type of job (from JobType enum)
            func: Callable function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            job_id: Optional custom job ID
            timeout: Job timeout in seconds (default 1 hour)
            result_ttl: Result time-to-live in seconds (default 24 hours)
            failure_ttl: Failed job time-to-live in seconds (default 7 days)
            
        Returns:
            Job instance
        """
        kwargs = kwargs or {}
        
        try:
            job = self.queue.enqueue(
                func,
                *args,
                job_id=job_id,
                job_timeout=timeout,
                result_ttl=result_ttl,
                failure_ttl=failure_ttl,
                meta={
                    "job_type": job_type.value,
                    "status": "queued",
                },
                **kwargs,
            )
            
            logger.info(
                f"Job enqueued: {job.id} (type: {job_type.value}, "
                f"timeout: {timeout}s)"
            )
            
            return job
        
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job instance or None if not found
        """
        try:
            return Job.fetch(job_id, connection=self.redis_client)
        except NoSuchJobError:
            logger.warning(f"Job not found: {job_id}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status string or None if not found
        """
        job = self.get_job(job_id)
        if job:
            return job.get_status()
        return None
    
    def get_job_result(self, job_id: str) -> Any:
        """
        Get job result.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job result or None if not available
        """
        job = self.get_job(job_id)
        if job:
            return job.result
        return None
    
    def get_job_error(self, job_id: str) -> Optional[str]:
        """
        Get job error message.
        
        Args:
            job_id: Job ID
            
        Returns:
            Error message or None if no error
        """
        job = self.get_job(job_id)
        if job and job.is_failed:
            return job.exc_info
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled, False if not found
        """
        job = self.get_job(job_id)
        if job:
            job.cancel()
            logger.info(f"Job cancelled: {job_id}")
            return True
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue stats
        """
        return {
            "queue_name": self.queue_name,
            "job_count": len(self.queue),
            "started_job_registry_count": len(self.queue.started_job_registry),
            "finished_job_registry_count": len(self.queue.finished_job_registry),
            "failed_job_registry_count": len(self.queue.failed_job_registry),
            "scheduled_job_registry_count": len(self.queue.scheduled_job_registry),
        }
    
    def get_worker_count(self) -> int:
        """
        Get number of active workers.
        
        Returns:
            Number of workers
        """
        workers = Worker.all(connection=self.redis_client)
        return len(workers)
    
    def start_worker(self, worker_name: str = None, job_monitoring_interval: int = 30):
        """
        Start a worker to process jobs.
        
        Args:
            worker_name: Optional worker name
            job_monitoring_interval: Interval for job monitoring in seconds
            
        Returns:
            Worker instance
        """
        worker = Worker(
            [self.queue],
            connection=self.redis_client,
            name=worker_name,
            job_monitoring_interval=job_monitoring_interval,
        )
        
        logger.info(f"Worker started: {worker.name}")
        return worker


# Global job queue manager instance
_job_queue_manager = None


def get_job_queue_manager() -> JobQueueManager:
    """Get or create global job queue manager."""
    global _job_queue_manager
    if _job_queue_manager is None:
        _job_queue_manager = JobQueueManager()
    return _job_queue_manager
