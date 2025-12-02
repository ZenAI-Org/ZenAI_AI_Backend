"""
Batch processing module for efficient handling of multiple meetings.
Implements batch job enqueueing, parallel processing, and result aggregation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from app.queue.job_queue import JobQueueManager, JobType, get_job_queue_manager
from app.core.performance_optimizer import (
    get_performance_metrics,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a single job in a batch."""
    
    job_id: str
    meeting_id: str
    s3_key: str
    project_id: Optional[str] = None
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    jobs: List[BatchJob]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100


class BatchProcessor:
    """Processes multiple meetings in batches for efficiency."""
    
    def __init__(
        self,
        job_queue_manager: Optional[JobQueueManager] = None,
        metrics: Optional[PerformanceMetrics] = None,
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """
        Initialize batch processor.
        
        Args:
            job_queue_manager: Job queue manager instance
            metrics: Performance metrics tracker
            batch_size: Number of jobs to process per batch
            max_concurrent: Maximum concurrent jobs
        """
        self.job_queue = job_queue_manager or get_job_queue_manager()
        self.metrics = metrics or get_performance_metrics()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.active_batches: Dict[str, BatchResult] = {}
    
    async def process_batch(
        self,
        meetings: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> BatchResult:
        """
        Process a batch of meetings.
        
        Args:
            meetings: List of meeting dicts with 'meeting_id' and 's3_key'
            batch_id: Optional batch identifier
            project_id: Optional project identifier for all meetings
            
        Returns:
            BatchResult with job statuses and results
        """
        batch_id = batch_id or f"batch_{datetime.now().timestamp()}"
        start_time = datetime.now()
        
        logger.info(
            f"Starting batch processing: batch_id={batch_id}, "
            f"meetings={len(meetings)}, project_id={project_id}"
        )
        
        # Create batch jobs
        batch_jobs = []
        for meeting in meetings:
            job_id = f"batch_{batch_id}_{meeting['meeting_id']}"
            batch_job = BatchJob(
                job_id=job_id,
                meeting_id=meeting["meeting_id"],
                s3_key=meeting["s3_key"],
                project_id=project_id or meeting.get("project_id"),
                created_at=datetime.now()
            )
            batch_jobs.append(batch_job)
        
        # Enqueue jobs in batches
        for i in range(0, len(batch_jobs), self.batch_size):
            batch_chunk = batch_jobs[i:i + self.batch_size]
            
            for batch_job in batch_chunk:
                try:
                    job = self.job_queue.enqueue_job(
                        job_type=JobType.TRANSCRIPTION,
                        func=self._transcription_job_wrapper,
                        kwargs={
                            "meeting_id": batch_job.meeting_id,
                            "s3_key": batch_job.s3_key,
                            "batch_id": batch_id,
                        },
                        job_id=batch_job.job_id,
                        timeout=3600,
                    )
                    batch_job.job_id = job.id
                    batch_job.status = "queued"
                    logger.debug(f"Enqueued batch job: {batch_job.job_id}")
                except Exception as e:
                    batch_job.status = "error"
                    batch_job.error = str(e)
                    logger.error(f"Failed to enqueue batch job: {e}")
        
        # Wait for batch completion
        batch_result = await self._wait_for_batch(batch_id, batch_jobs)
        
        # Record metrics
        self.metrics.record_latency(
            "batch_processing",
            batch_result.duration_seconds or 0,
            project_id
        )
        
        logger.info(
            f"Batch processing completed: batch_id={batch_id}, "
            f"success_rate={batch_result.success_rate():.1f}%"
        )
        
        return batch_result
    
    async def process_batch_with_pipeline(
        self,
        meetings: List[Dict[str, Any]],
        pipeline: List[str],
        batch_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> BatchResult:
        """
        Process a batch of meetings through a pipeline of operations.
        
        Args:
            meetings: List of meeting dicts
            pipeline: List of operations ('transcription', 'summarization', 'task_extraction')
            batch_id: Optional batch identifier
            project_id: Optional project identifier
            
        Returns:
            BatchResult with pipeline results
        """
        batch_id = batch_id or f"batch_pipeline_{datetime.now().timestamp()}"
        start_time = datetime.now()
        
        logger.info(
            f"Starting batch pipeline: batch_id={batch_id}, "
            f"meetings={len(meetings)}, pipeline={pipeline}"
        )
        
        # Create batch jobs for first stage
        batch_jobs = []
        for meeting in meetings:
            job_id = f"batch_{batch_id}_{meeting['meeting_id']}"
            batch_job = BatchJob(
                job_id=job_id,
                meeting_id=meeting["meeting_id"],
                s3_key=meeting["s3_key"],
                project_id=project_id or meeting.get("project_id"),
                created_at=datetime.now()
            )
            batch_jobs.append(batch_job)
        
        # Process through pipeline stages
        current_jobs = batch_jobs
        
        for stage_idx, stage in enumerate(pipeline):
            logger.info(f"Processing batch stage {stage_idx + 1}/{len(pipeline)}: {stage}")
            
            # Enqueue jobs for this stage
            for batch_job in current_jobs:
                if batch_job.status == "error":
                    continue  # Skip failed jobs
                
                try:
                    if stage == "transcription":
                        job = self.job_queue.enqueue_job(
                            job_type=JobType.TRANSCRIPTION,
                            func=self._transcription_job_wrapper,
                            kwargs={
                                "meeting_id": batch_job.meeting_id,
                                "s3_key": batch_job.s3_key,
                                "batch_id": batch_id,
                            },
                            job_id=f"{batch_job.job_id}_transcription",
                            timeout=3600,
                        )
                    elif stage == "summarization":
                        job = self.job_queue.enqueue_job(
                            job_type=JobType.SUMMARIZATION,
                            func=self._summarization_job_wrapper,
                            kwargs={
                                "meeting_id": batch_job.meeting_id,
                                "project_id": batch_job.project_id,
                                "batch_id": batch_id,
                            },
                            job_id=f"{batch_job.job_id}_summarization",
                            timeout=600,
                        )
                    elif stage == "task_extraction":
                        job = self.job_queue.enqueue_job(
                            job_type=JobType.TASK_EXTRACTION,
                            func=self._task_extraction_job_wrapper,
                            kwargs={
                                "meeting_id": batch_job.meeting_id,
                                "project_id": batch_job.project_id,
                                "batch_id": batch_id,
                            },
                            job_id=f"{batch_job.job_id}_task_extraction",
                            timeout=600,
                        )
                    else:
                        logger.warning(f"Unknown pipeline stage: {stage}")
                        continue
                    
                    batch_job.status = "processing"
                    logger.debug(f"Enqueued {stage} job: {job.id}")
                except Exception as e:
                    batch_job.status = "error"
                    batch_job.error = f"Failed to enqueue {stage} job: {str(e)}"
                    logger.error(batch_job.error)
            
            # Wait for stage completion
            await self._wait_for_stage_completion(current_jobs)
        
        # Create final batch result
        completed = sum(1 for j in current_jobs if j.status == "completed")
        failed = sum(1 for j in current_jobs if j.status == "error")
        
        batch_result = BatchResult(
            batch_id=batch_id,
            total_jobs=len(current_jobs),
            completed_jobs=completed,
            failed_jobs=failed,
            jobs=current_jobs,
            started_at=start_time,
            completed_at=datetime.now(),
            duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        logger.info(
            f"Batch pipeline completed: batch_id={batch_id}, "
            f"success_rate={batch_result.success_rate():.1f}%"
        )
        
        return batch_result
    
    async def _wait_for_batch(
        self,
        batch_id: str,
        batch_jobs: List[BatchJob],
        timeout: int = 3600
    ) -> BatchResult:
        """
        Wait for batch jobs to complete.
        
        Args:
            batch_id: Batch identifier
            batch_jobs: List of batch jobs
            timeout: Timeout in seconds
            
        Returns:
            BatchResult with completion status
        """
        start_time = datetime.now()
        
        while True:
            # Check job statuses
            completed = 0
            failed = 0
            
            for batch_job in batch_jobs:
                if batch_job.status in ["completed", "error"]:
                    if batch_job.status == "completed":
                        completed += 1
                    else:
                        failed += 1
                    continue
                
                try:
                    job = self.job_queue.get_job(batch_job.job_id)
                    
                    if not job:
                        batch_job.status = "error"
                        batch_job.error = "Job not found"
                        failed += 1
                        continue
                    
                    if job.is_finished:
                        if job.is_failed:
                            batch_job.status = "error"
                            batch_job.error = job.exc_info
                            failed += 1
                        else:
                            batch_job.status = "completed"
                            batch_job.result = job.result
                            batch_job.completed_at = datetime.now()
                            completed += 1
                except Exception as e:
                    batch_job.status = "error"
                    batch_job.error = str(e)
                    failed += 1
            
            # Check if all jobs completed
            all_done = all(j.status in ["completed", "error"] for j in batch_jobs)
            
            if all_done:
                break
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Batch {batch_id} timed out after {elapsed}s")
                break
            
            # Wait before checking again
            await asyncio.sleep(2)
        
        # Create batch result
        completed = sum(1 for j in batch_jobs if j.status == "completed")
        failed = sum(1 for j in batch_jobs if j.status == "error")
        
        batch_result = BatchResult(
            batch_id=batch_id,
            total_jobs=len(batch_jobs),
            completed_jobs=completed,
            failed_jobs=failed,
            jobs=batch_jobs,
            started_at=start_time,
            completed_at=datetime.now(),
            duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        return batch_result
    
    async def _wait_for_stage_completion(
        self,
        batch_jobs: List[BatchJob],
        timeout: int = 600
    ) -> None:
        """
        Wait for all jobs in a pipeline stage to complete.
        
        Args:
            batch_jobs: List of batch jobs
            timeout: Timeout in seconds
        """
        start_time = datetime.now()
        
        while True:
            all_done = all(j.status in ["completed", "error"] for j in batch_jobs)
            
            if all_done:
                break
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Stage timed out after {elapsed}s")
                break
            
            # Wait before checking again
            await asyncio.sleep(1)
    
    def _transcription_job_wrapper(
        self,
        meeting_id: str,
        s3_key: str,
        batch_id: str
    ) -> Dict[str, Any]:
        """Wrapper for transcription job in batch context."""
        from app.agents.transcription_agent import transcribe_audio_job
        
        logger.debug(f"Batch transcription job: {meeting_id} (batch: {batch_id})")
        return transcribe_audio_job(meeting_id, s3_key)
    
    def _summarization_job_wrapper(
        self,
        meeting_id: str,
        project_id: Optional[str],
        batch_id: str
    ) -> Dict[str, Any]:
        """Wrapper for summarization job in batch context."""
        from app.agents.summarization_agent import summarize_meeting_job
        
        logger.debug(f"Batch summarization job: {meeting_id} (batch: {batch_id})")
        return summarize_meeting_job(meeting_id, "", project_id)
    
    def _task_extraction_job_wrapper(
        self,
        meeting_id: str,
        project_id: Optional[str],
        batch_id: str
    ) -> Dict[str, Any]:
        """Wrapper for task extraction job in batch context."""
        from app.agents.task_extraction_agent import extract_tasks_job
        
        logger.debug(f"Batch task extraction job: {meeting_id} (batch: {batch_id})")
        return extract_tasks_job(meeting_id, "", project_id)
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchResult]:
        """
        Get status of a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            BatchResult or None if not found
        """
        return self.active_batches.get(batch_id)


# Global batch processor instance
_batch_processor = None


def get_batch_processor() -> BatchProcessor:
    """Get or create global batch processor."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
