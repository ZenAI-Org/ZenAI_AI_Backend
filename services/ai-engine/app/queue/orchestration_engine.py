"""
AI Orchestration Engine that coordinates job enqueueing and agent execution.
Manages the workflow of AI tasks through the job queue system.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime

from app.queue.job_queue import JobQueueManager, JobType, get_job_queue_manager
from app.queue.job_listeners import (
    JobEventListener,
    RetryPolicy,
    get_job_event_listener,
    get_retry_policy,
)
from app.core.context_retriever import ContextRetriever
from app.agents.base_agent import AgentConfig
from app.core.error_notifications import get_error_notification_manager
from app.core.error_dashboard import get_error_metrics

logger = logging.getLogger(__name__)

# Try to import Socket.io manager
try:
    from app.queue.socketio_manager import get_socketio_manager
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False


class AIOrchestrationEngine:
    """
    Central orchestration engine for AI workflows.
    Coordinates job enqueueing, monitoring, and agent execution.
    """
    
    def __init__(
        self,
        job_queue_manager: Optional[JobQueueManager] = None,
        event_listener: Optional[JobEventListener] = None,
        retry_policy: Optional[RetryPolicy] = None,
        db_connection: Optional[Any] = None,
    ):
        """
        Initialize orchestration engine.
        
        Args:
            job_queue_manager: Job queue manager instance
            event_listener: Job event listener instance
            retry_policy: Retry policy for failed jobs
            db_connection: PostgreSQL connection for context retrieval
        """
        self.job_queue = job_queue_manager or get_job_queue_manager()
        self.event_listener = event_listener or get_job_event_listener()
        self.retry_policy = retry_policy or get_retry_policy()
        self.db_connection = db_connection
        self._context_retriever = None
        self._context_retriever_initialized = False
        
        # Error handling components
        self.error_notifier = get_error_notification_manager()
        self.error_metrics = get_error_metrics()
        
        # Socket.io manager for real-time updates
        self.socketio_manager = None
        if SOCKETIO_AVAILABLE:
            try:
                self.socketio_manager = get_socketio_manager()
                # Set Socket.io manager for error notifications
                self.error_notifier.set_socketio_manager(self.socketio_manager)
            except Exception as e:
                logger.warning(f"Failed to initialize Socket.io manager: {e}")
        
        logger.info("AI Orchestration Engine initialized")
    
    @property
    def context_retriever(self) -> Optional[ContextRetriever]:
        """
        Lazy-load context retriever on first access.
        
        Returns:
            ContextRetriever instance or None if db_connection not available
        """
        if not self._context_retriever_initialized:
            try:
                if self.db_connection:
                    self._context_retriever = ContextRetriever(self.db_connection)
                self._context_retriever_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize context retriever: {e}")
                self._context_retriever_initialized = True
        
        return self._context_retriever
    
    @context_retriever.setter
    def context_retriever(self, value: Optional[ContextRetriever]) -> None:
        """
        Set context retriever (for testing).
        
        Args:
            value: ContextRetriever instance or None
        """
        self._context_retriever = value
        self._context_retriever_initialized = True
    
    def enqueue_transcription_job(
        self,
        meeting_id: str,
        s3_key: str,
        project_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a transcription job.
        
        Args:
            meeting_id: Meeting ID
            s3_key: S3 key for audio file
            project_id: Optional project ID for tracking
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.transcription_agent import transcribe_audio_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.TRANSCRIPTION,
            func=transcribe_audio_job,
            kwargs={
                "meeting_id": meeting_id,
                "s3_key": s3_key,
            },
            job_id=job_id,
            timeout=3600,  # 1 hour for transcription
        )
        
        # Track job for project if provided
        if project_id and self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Transcription job enqueued: {job.id} "
            f"(meeting: {meeting_id}, s3_key: {s3_key}, project: {project_id})"
        )
        
        return job.id
    
    def enqueue_summarization_job(
        self,
        meeting_id: str,
        transcript: str,
        project_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a summarization job.
        
        Args:
            meeting_id: Meeting ID
            transcript: Meeting transcript
            project_id: Optional project ID for context
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.summarization_agent import summarize_meeting_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.SUMMARIZATION,
            func=summarize_meeting_job,
            kwargs={
                "meeting_id": meeting_id,
                "transcript": transcript,
                "project_id": project_id,
            },
            job_id=job_id,
            timeout=600,  # 10 minutes for summarization
        )
        
        # Track job for project if provided
        if project_id and self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Summarization job enqueued: {job.id} (meeting: {meeting_id}, project: {project_id})"
        )
        
        return job.id
    
    def enqueue_task_extraction_job(
        self,
        meeting_id: str,
        transcript: str,
        project_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a task extraction job.
        
        Args:
            meeting_id: Meeting ID
            transcript: Meeting transcript
            project_id: Optional project ID
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.task_extraction_agent import extract_tasks_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.TASK_EXTRACTION,
            func=extract_tasks_job,
            kwargs={
                "meeting_id": meeting_id,
                "transcript": transcript,
                "project_id": project_id,
            },
            job_id=job_id,
            timeout=600,  # 10 minutes for task extraction
        )
        
        # Track job for project if provided
        if project_id and self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Task extraction job enqueued: {job.id} (meeting: {meeting_id}, project: {project_id})"
        )
        
        return job.id
    
    def enqueue_aipm_analysis_job(
        self,
        project_id: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue an AIPM analysis job.
        
        Args:
            project_id: Project ID
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.aipm_agent import analyze_project_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.AIPM_ANALYSIS,
            func=analyze_project_job,
            kwargs={
                "project_id": project_id,
            },
            job_id=job_id,
            timeout=900,  # 15 minutes for AIPM analysis
        )
        
        # Track job for project
        if self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(f"AIPM analysis job enqueued: {job.id} (project: {project_id})")
        
        return job.id
    
    def enqueue_chat_job(
        self,
        user_id: str,
        project_id: str,
        message: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a chat job.
        
        Args:
            user_id: User ID
            project_id: Project ID
            message: Chat message
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.chat_agent import process_chat_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.CHAT,
            func=process_chat_job,
            kwargs={
                "user_id": user_id,
                "project_id": project_id,
                "message": message,
            },
            job_id=job_id,
            timeout=300,  # 5 minutes for chat
        )
        
        # Track job for project
        if self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Chat job enqueued: {job.id} "
            f"(user: {user_id}, project: {project_id})"
        )
        
        return job.id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and details.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary with job status and details
        """
        job = self.job_queue.get_job(job_id)
        
        if not job:
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": "Job not found",
            }
        
        status_data = {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "result": job.result,
            "error": job.exc_info if job.is_failed else None,
        }
        
        if job.meta:
            status_data["meta"] = job.meta
        
        return status_data
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled, False if not found
        """
        success = self.job_queue.cancel_job(job_id)
        
        if success:
            logger.info(f"Job cancelled: {job_id}")
        
        return success
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue stats
        """
        stats = self.job_queue.get_queue_stats()
        stats["worker_count"] = self.job_queue.get_worker_count()
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    async def process_meeting(
        self,
        meeting_id: str,
        s3_key: str,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate complete meeting processing workflow.
        
        Coordinates: transcription → summarization → task extraction → Notion sync
        
        Args:
            meeting_id: Meeting ID
            s3_key: S3 key for audio file
            project_id: Optional project ID for context
            
        Returns:
            Dictionary with workflow results and job IDs
        """
        try:
            logger.info(
                f"Starting meeting processing workflow: "
                f"meeting_id={meeting_id}, s3_key={s3_key}, project_id={project_id}"
            )
            
            workflow_result = {
                "meeting_id": meeting_id,
                "project_id": project_id,
                "status": "initiated",
                "jobs": {},
                "errors": [],
                "started_at": datetime.now().isoformat(),
            }
            
            # Step 1: Enqueue transcription job
            try:
                transcription_job_id = self.enqueue_transcription_job(
                    meeting_id=meeting_id,
                    s3_key=s3_key,
                    project_id=project_id,
                    job_id=f"transcription_{meeting_id}",
                )
                workflow_result["jobs"]["transcription"] = transcription_job_id
                logger.info(f"Transcription job enqueued: {transcription_job_id}")
            except Exception as e:
                error_msg = f"Failed to enqueue transcription job: {str(e)}"
                logger.error(error_msg)
                workflow_result["errors"].append(error_msg)
                workflow_result["status"] = "failed"
                return workflow_result
            
            # Step 2: Wait for transcription to complete and get transcript
            # (In production, this would use job completion callbacks)
            transcript = await self._wait_for_transcription(transcription_job_id)
            
            if not transcript:
                error_msg = "Transcription failed or timed out"
                logger.error(error_msg)
                workflow_result["errors"].append(error_msg)
                workflow_result["status"] = "failed"
                return workflow_result
            
            # Step 3: Enqueue summarization job
            try:
                summarization_job_id = self.enqueue_summarization_job(
                    meeting_id=meeting_id,
                    transcript=transcript,
                    project_id=project_id,
                    job_id=f"summarization_{meeting_id}",
                )
                workflow_result["jobs"]["summarization"] = summarization_job_id
                logger.info(f"Summarization job enqueued: {summarization_job_id}")
            except Exception as e:
                error_msg = f"Failed to enqueue summarization job: {str(e)}"
                logger.error(error_msg)
                workflow_result["errors"].append(error_msg)
                # Continue with task extraction even if summarization fails
            
            # Step 4: Enqueue task extraction job
            try:
                task_extraction_job_id = self.enqueue_task_extraction_job(
                    meeting_id=meeting_id,
                    transcript=transcript,
                    project_id=project_id,
                    job_id=f"task_extraction_{meeting_id}",
                )
                workflow_result["jobs"]["task_extraction"] = task_extraction_job_id
                logger.info(f"Task extraction job enqueued: {task_extraction_job_id}")
            except Exception as e:
                error_msg = f"Failed to enqueue task extraction job: {str(e)}"
                logger.error(error_msg)
                workflow_result["errors"].append(error_msg)
                # Continue with Notion sync even if task extraction fails
            
            # Step 5: Enqueue Notion sync job if project_id provided
            if project_id:
                try:
                    notion_sync_job_id = self.enqueue_notion_sync_job(
                        meeting_id=meeting_id,
                        project_id=project_id,
                        job_id=f"notion_sync_{meeting_id}",
                    )
                    workflow_result["jobs"]["notion_sync"] = notion_sync_job_id
                    logger.info(f"Notion sync job enqueued: {notion_sync_job_id}")
                except Exception as e:
                    error_msg = f"Failed to enqueue Notion sync job: {str(e)}"
                    logger.error(error_msg)
                    workflow_result["errors"].append(error_msg)
                    # Notion sync failure is not critical
            
            workflow_result["status"] = "processing"
            workflow_result["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Meeting processing workflow initiated successfully: {workflow_result}")
            return workflow_result
        
        except Exception as e:
            error_msg = f"Meeting processing workflow failed: {str(e)}"
            logger.error(error_msg)
            return {
                "meeting_id": meeting_id,
                "project_id": project_id,
                "status": "failed",
                "error": error_msg,
                "started_at": datetime.now().isoformat(),
            }
    
    async def process_meetings_batch(
        self,
        meetings: List[Dict[str, Any]],
        concurrency: int = 3
    ) -> Dict[str, Any]:
        """
        Process a batch of meetings with concurrency control.
        
        Args:
            meetings: List of meeting dicts with keys: meeting_id, s3_key, project_id
            concurrency: Maximum number of concurrent processing tasks
            
        Returns:
            Dictionary with batch results
        """
        logger.info(f"Starting batch processing for {len(meetings)} meetings (concurrency={concurrency})")
        
        results = {
            "total": len(meetings),
            "successful": 0,
            "failed": 0,
            "results": [],
            "started_at": datetime.now().isoformat()
        }
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(meeting):
            async with semaphore:
                return await self.process_meeting(
                    meeting_id=meeting["meeting_id"],
                    s3_key=meeting["s3_key"],
                    project_id=meeting.get("project_id")
                )
        
        # Create tasks
        tasks = [process_with_semaphore(meeting) for meeting in meetings]
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed with exception: {result}")
                results["results"].append({
                    "meeting_id": meetings[i]["meeting_id"],
                    "status": "failed",
                    "error": str(result)
                })
                results["failed"] += 1
            else:
                results["results"].append(result)
                if result.get("status") == "processing" or result.get("status") == "initiated":
                    results["successful"] += 1
                else:
                    results["failed"] += 1
        
        results["completed_at"] = datetime.now().isoformat()
        logger.info(f"Batch processing completed: {results['successful']}/{results['total']} successful")
        
        return results
    
    async def chat(
        self,
        user_id: str,
        project_id: str,
        message: str,
        user_role: str = "member",
    ) -> Dict[str, Any]:
        """
        Route chat request to Chat Agent.
        
        Args:
            user_id: User ID
            project_id: Project ID
            message: Chat message
            user_role: User role for permission validation
            
        Returns:
            Dictionary with chat response
        """
        try:
            logger.info(
                f"Chat request: user_id={user_id}, project_id={project_id}, "
                f"message_length={len(message)}"
            )
            
            # Enqueue chat job
            chat_job_id = self.enqueue_chat_job(
                user_id=user_id,
                project_id=project_id,
                message=message,
                job_id=f"chat_{user_id}_{project_id}_{datetime.now().timestamp()}",
            )
            
            # Wait for chat job to complete
            chat_result = await self._wait_for_job_completion(chat_job_id, timeout=30)
            
            if not chat_result:
                return {
                    "user_id": user_id,
                    "project_id": project_id,
                    "status": "error",
                    "error": "Chat request timed out",
                    "job_id": chat_job_id,
                }
            
            logger.info(f"Chat request completed: {chat_job_id}")
            return {
                "user_id": user_id,
                "project_id": project_id,
                "status": "success",
                "result": chat_result,
                "job_id": chat_job_id,
            }
        
        except Exception as e:
            error_msg = f"Chat request failed: {str(e)}"
            logger.error(error_msg)
            return {
                "user_id": user_id,
                "project_id": project_id,
                "status": "error",
                "error": error_msg,
            }
    
    async def analyze_project(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Route project analysis request to AIPM Agent.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with AIPM analysis results
        """
        try:
            logger.info(f"AIPM analysis request: project_id={project_id}")
            
            # Enqueue AIPM analysis job
            aipm_job_id = self.enqueue_aipm_analysis_job(
                project_id=project_id,
                job_id=f"aipm_{project_id}_{datetime.now().timestamp()}",
            )
            
            # Wait for AIPM job to complete
            aipm_result = await self._wait_for_job_completion(aipm_job_id, timeout=60)
            
            if not aipm_result:
                return {
                    "project_id": project_id,
                    "status": "error",
                    "error": "AIPM analysis timed out",
                    "job_id": aipm_job_id,
                }
            
            logger.info(f"AIPM analysis completed: {aipm_job_id}")
            return {
                "project_id": project_id,
                "status": "success",
                "result": aipm_result,
                "job_id": aipm_job_id,
            }
        
        except Exception as e:
            error_msg = f"AIPM analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "project_id": project_id,
                "status": "error",
                "error": error_msg,
            }
    
    async def generate_suggestions(
        self,
        project_id: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Route suggestion generation request to Suggestions Agent.
        
        Args:
            project_id: Project ID
            force_refresh: Force refresh even if cached
            
        Returns:
            Dictionary with dashboard suggestions
        """
        try:
            logger.info(
                f"Suggestion generation request: project_id={project_id}, "
                f"force_refresh={force_refresh}"
            )
            
            # Enqueue suggestions job
            suggestions_job_id = self.enqueue_suggestions_job(
                project_id=project_id,
                force_refresh=force_refresh,
                job_id=f"suggestions_{project_id}_{datetime.now().timestamp()}",
            )
            
            # Wait for suggestions job to complete
            suggestions_result = await self._wait_for_job_completion(
                suggestions_job_id, timeout=45
            )
            
            if not suggestions_result:
                return {
                    "project_id": project_id,
                    "status": "error",
                    "error": "Suggestion generation timed out",
                    "job_id": suggestions_job_id,
                }
            
            logger.info(f"Suggestion generation completed: {suggestions_job_id}")
            return {
                "project_id": project_id,
                "status": "success",
                "result": suggestions_result,
                "job_id": suggestions_job_id,
            }
        
        except Exception as e:
            error_msg = f"Suggestion generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "project_id": project_id,
                "status": "error",
                "error": error_msg,
            }
    
    async def retrieve_context(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve semantic search context for a project.
        
        Args:
            project_id: Project ID
            query: Search query
            limit: Maximum results per content type
            
        Returns:
            Dictionary with retrieved context
        """
        try:
            logger.info(
                f"Context retrieval request: project_id={project_id}, "
                f"query={query}, limit={limit}"
            )
            
            if not self.context_retriever:
                return {
                    "project_id": project_id,
                    "status": "error",
                    "error": "Context retriever not initialized",
                }
            
            # Retrieve context using semantic search
            context = self.context_retriever.retrieve_meeting_context(
                project_id=project_id,
                query=query,
                limit=limit,
            )
            
            logger.info(
                f"Context retrieved: {len(context.get('summaries', []))} summaries, "
                f"{len(context.get('decisions', []))} decisions, "
                f"{len(context.get('blockers', []))} blockers"
            )
            
            return {
                "project_id": project_id,
                "status": "success",
                "query": query,
                "context": context,
                "retrieved_at": datetime.now().isoformat(),
            }
        
        except Exception as e:
            error_msg = f"Context retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                "project_id": project_id,
                "status": "error",
                "error": error_msg,
                "query": query,
            }
    
    def enqueue_suggestions_job(
        self,
        project_id: str,
        force_refresh: bool = False,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a suggestions generation job.
        
        Args:
            project_id: Project ID
            force_refresh: Force refresh even if cached
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.suggestions_agent import generate_suggestions_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.SUGGESTIONS,
            func=generate_suggestions_job,
            kwargs={
                "project_id": project_id,
                "force_refresh": force_refresh,
            },
            job_id=job_id,
            timeout=600,  # 10 minutes for suggestions
        )
        
        # Track job for project
        if self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Suggestions job enqueued: {job.id} "
            f"(project: {project_id}, force_refresh: {force_refresh})"
        )
        
        return job.id
    
    def enqueue_notion_sync_job(
        self,
        meeting_id: str,
        project_id: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Enqueue a Notion sync job.
        
        Args:
            meeting_id: Meeting ID
            project_id: Project ID
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        from app.agents.notion_integration_agent import sync_tasks_to_notion_job
        
        job = self.job_queue.enqueue_job(
            job_type=JobType.NOTION_SYNC,
            func=sync_tasks_to_notion_job,
            kwargs={
                "meeting_id": meeting_id,
                "project_id": project_id,
            },
            job_id=job_id,
            timeout=300,  # 5 minutes for Notion sync
        )
        
        # Track job for project
        if self.socketio_manager:
            self.socketio_manager.track_job_for_project(job.id, project_id)
        
        self.event_listener.on_job_queued(job.id)
        
        logger.info(
            f"Notion sync job enqueued: {job.id} "
            f"(meeting: {meeting_id}, project: {project_id})"
        )
        
        return job.id
    
    async def _wait_for_transcription(
        self,
        job_id: str,
        timeout: int = 3600,
    ) -> Optional[str]:
        """
        Wait for transcription job to complete and retrieve transcript.
        
        Args:
            job_id: Job ID
            timeout: Timeout in seconds
            
        Returns:
            Transcript text or None if failed/timed out
        """
        try:
            start_time = datetime.now()
            
            while True:
                job = self.job_queue.get_job(job_id)
                
                if not job:
                    logger.error(f"Transcription job not found: {job_id}")
                    return None
                
                if job.is_finished:
                    if job.is_failed:
                        logger.error(f"Transcription job failed: {job_id}")
                        return None
                    
                    # Extract transcript from result
                    result = job.result
                    if isinstance(result, dict) and "text" in result:
                        return result["text"]
                    elif isinstance(result, str):
                        return result
                    
                    logger.error(f"Invalid transcription result format: {result}")
                    return None
                
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    logger.error(f"Transcription job timed out: {job_id}")
                    return None
                
                # Wait before checking again
                await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Error waiting for transcription: {str(e)}")
            return None
    
    async def _wait_for_job_completion(
        self,
        job_id: str,
        timeout: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a job to complete and retrieve result.
        
        Args:
            job_id: Job ID
            timeout: Timeout in seconds
            
        Returns:
            Job result or None if failed/timed out
        """
        try:
            start_time = datetime.now()
            
            while True:
                job = self.job_queue.get_job(job_id)
                
                if not job:
                    logger.error(f"Job not found: {job_id}")
                    return None
                
                if job.is_finished:
                    if job.is_failed:
                        logger.error(f"Job failed: {job_id}")
                        return None
                    
                    return job.result
                
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    logger.error(f"Job timed out: {job_id}")
                    return None
                
                # Wait before checking again
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error waiting for job completion: {str(e)}")
            return None


# Global orchestration engine instance
_orchestration_engine = None


def get_orchestration_engine() -> AIOrchestrationEngine:
    """Get or create global orchestration engine."""
    global _orchestration_engine
    if _orchestration_engine is None:
        _orchestration_engine = AIOrchestrationEngine()
    return _orchestration_engine
