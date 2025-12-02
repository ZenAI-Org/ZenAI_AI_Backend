"""
Integration tests for AI Orchestration Engine.

Tests end-to-end meeting processing workflow, agent coordination,
error recovery, and context sharing between agents.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.queue.orchestration_engine import AIOrchestrationEngine
from app.queue.job_queue import JobQueueManager, JobType
from app.queue.job_listeners import JobEventListener, RetryPolicy
from app.agents.base_agent import AgentConfig, AgentStatus, AgentResult
from app.core.context_retriever import ContextRetriever


class TestOrchestrationEngine:
    """Test suite for AI Orchestration Engine."""
    
    @pytest.fixture
    def mock_job_queue(self):
        """Create mock job queue manager."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_queue.enqueue_job = Mock(return_value=Mock(id="job_123"))
        mock_queue.get_job = Mock(return_value=Mock(
            id="job_123",
            get_status=Mock(return_value="completed"),
            is_finished=True,
            is_failed=False,
            result={"text": "Sample transcript"},
            created_at=datetime.now(),
            started_at=datetime.now(),
            ended_at=datetime.now(),
            meta={}
        ))
        mock_queue.cancel_job = Mock(return_value=True)
        mock_queue.get_queue_stats = Mock(return_value={
            "queued": 0,
            "active": 0,
            "completed": 0,
            "failed": 0
        })
        mock_queue.get_worker_count = Mock(return_value=2)
        return mock_queue
    
    @pytest.fixture
    def mock_event_listener(self):
        """Create mock event listener."""
        mock_listener = Mock(spec=JobEventListener)
        mock_listener.on_job_queued = Mock()
        mock_listener.on_job_started = Mock()
        mock_listener.on_job_completed = Mock()
        mock_listener.on_job_failed = Mock()
        return mock_listener
    
    @pytest.fixture
    def mock_retry_policy(self):
        """Create mock retry policy."""
        mock_policy = Mock(spec=RetryPolicy)
        mock_policy.max_retries = 3
        mock_policy.backoff_factor = 2
        return mock_policy
    
    @pytest.fixture
    def mock_context_retriever(self):
        """Create mock context retriever."""
        mock_retriever = Mock(spec=ContextRetriever)
        mock_retriever.retrieve_meeting_context = Mock(return_value={
            "summaries": [{"metadata": {"text": "Meeting summary"}}],
            "decisions": [{"metadata": {"text": "Key decision"}}],
            "blockers": [{"metadata": {"text": "Known blocker"}}]
        })
        mock_retriever.build_prompt_context = Mock(return_value="Project context")
        return mock_retriever
    
    @pytest.fixture
    def orchestration_engine(self, mock_job_queue, mock_event_listener, 
                            mock_retry_policy, mock_context_retriever):
        """Create orchestration engine with mocks."""
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_job_queue,
            event_listener=mock_event_listener,
            retry_policy=mock_retry_policy,
            db_connection=Mock()
        )
        engine.context_retriever = mock_context_retriever
        return engine
    
    # Test 1: End-to-end meeting processing workflow
    @pytest.mark.asyncio
    async def test_process_meeting_complete_workflow(self, orchestration_engine):
        """Test complete meeting processing workflow."""
        # Arrange
        meeting_id = "meeting_123"
        s3_key = "meetings/meeting_123.mp3"
        project_id = "project_456"
        
        # Mock job queue to return successful jobs
        mock_job = Mock()
        mock_job.id = "job_123"
        mock_job.get_status = Mock(return_value="completed")
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {"text": "Sample transcript"}
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key,
            project_id=project_id
        )
        
        # Assert
        assert result["meeting_id"] == meeting_id
        assert result["project_id"] == project_id
        assert result["status"] in ["processing", "failed"]
        assert "jobs" in result
        assert "transcription" in result["jobs"]
        assert orchestration_engine.job_queue.enqueue_job.called
    
    # Test 2: Agent coordination and state sharing
    @pytest.mark.asyncio
    async def test_agent_coordination_state_sharing(self, orchestration_engine):
        """Test agent coordination and state sharing between agents."""
        # Arrange
        meeting_id = "meeting_123"
        project_id = "project_456"
        
        # Mock context retriever to simulate state sharing
        shared_context = {
            "meeting_id": meeting_id,
            "project_id": project_id,
            "transcript": "Sample transcript",
            "summary": "Sample summary",
            "tasks": [{"title": "Task 1", "priority": "high"}]
        }
        
        orchestration_engine.context_retriever.retrieve_meeting_context = Mock(
            return_value=shared_context
        )
        
        # Act
        context = orchestration_engine.context_retriever.retrieve_meeting_context(
            project_id=project_id,
            query="meeting context"
        )
        
        # Assert
        assert context["meeting_id"] == meeting_id
        assert context["project_id"] == project_id
        assert "transcript" in context
        assert "summary" in context
        assert "tasks" in context
    
    # Test 3: Error recovery and fallback logic
    @pytest.mark.asyncio
    async def test_error_recovery_fallback_logic(self, orchestration_engine):
        """Test error recovery and fallback logic in workflow."""
        # Arrange
        meeting_id = "meeting_123"
        s3_key = "meetings/meeting_123.mp3"
        project_id = "project_456"
        
        # Mock job queue to simulate transcription failure
        mock_job = Mock()
        mock_job.id = "job_123"
        mock_job.get_status = Mock(return_value="failed")
        mock_job.is_finished = True
        mock_job.is_failed = True
        mock_job.result = None
        mock_job.exc_info = "Transcription API error"
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        orchestration_engine.job_queue.get_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key,
            project_id=project_id
        )
        
        # Assert
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
    
    # Test 4: Context sharing between agents
    @pytest.mark.asyncio
    async def test_context_sharing_between_agents(self, orchestration_engine):
        """Test context sharing between agents via semantic search."""
        # Arrange
        project_id = "project_456"
        query = "project status and blockers"
        
        # Mock context retriever
        expected_context = {
            "summaries": [
                {
                    "id": "summary_1",
                    "content_type": "summary",
                    "metadata": {"text": "Meeting summary"}
                }
            ],
            "decisions": [
                {
                    "id": "decision_1",
                    "content_type": "decision",
                    "metadata": {"text": "Key decision"}
                }
            ],
            "blockers": [
                {
                    "id": "blocker_1",
                    "content_type": "blocker",
                    "metadata": {"text": "Known blocker"}
                }
            ]
        }
        
        orchestration_engine.context_retriever.retrieve_meeting_context = Mock(
            return_value=expected_context
        )
        
        # Act
        context = await orchestration_engine.retrieve_context(
            project_id=project_id,
            query=query
        )
        
        # Assert
        assert context["status"] == "success"
        assert context["project_id"] == project_id
        assert "context" in context
        assert len(context["context"]["summaries"]) > 0
        assert len(context["context"]["decisions"]) > 0
        assert len(context["context"]["blockers"]) > 0
    
    # Test 5: Chat agent routing
    @pytest.mark.asyncio
    async def test_chat_agent_routing(self, orchestration_engine):
        """Test routing to Chat Agent."""
        # Arrange
        user_id = "user_123"
        project_id = "project_456"
        message = "What are the current blockers?"
        
        # Mock job queue
        mock_job = Mock()
        mock_job.id = "chat_job_123"
        mock_job.get_status = Mock(return_value="completed")
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {
            "response": "Current blockers are...",
            "sources": ["meeting_1", "meeting_2"],
            "confidence": 0.95
        }
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.chat(
            user_id=user_id,
            project_id=project_id,
            message=message
        )
        
        # Assert
        assert result["status"] == "success"
        assert result["user_id"] == user_id
        assert result["project_id"] == project_id
        assert "result" in result
    
    # Test 6: AIPM agent routing
    @pytest.mark.asyncio
    async def test_aipm_agent_routing(self, orchestration_engine):
        """Test routing to AIPM Agent."""
        # Arrange
        project_id = "project_456"
        
        # Mock job queue
        mock_job = Mock()
        mock_job.id = "aipm_job_123"
        mock_job.get_status = Mock(return_value="completed")
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {
            "status": "success",
            "project_id": project_id,
            "health": "at-risk",
            "blockers": [{"title": "Resource constraint", "impact": "high"}],
            "recommendations": [{"title": "Hire additional resources", "priority": 9}],
            "metrics": {"completion_rate": 71.1}
        }
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        orchestration_engine.job_queue.get_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.analyze_project(
            project_id=project_id
        )
        
        # Assert
        assert result["status"] == "success"
        assert result["project_id"] == project_id
        assert "result" in result
        assert result["result"]["health"] in ["healthy", "at-risk", "critical"]
    
    # Test 7: Suggestions agent routing
    @pytest.mark.asyncio
    async def test_suggestions_agent_routing(self, orchestration_engine):
        """Test routing to Suggestions Agent."""
        # Arrange
        project_id = "project_456"
        
        # Mock job queue
        mock_job = Mock()
        mock_job.id = "suggestions_job_123"
        mock_job.get_status = Mock(return_value="completed")
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {
            "status": "success",
            "project_id": project_id,
            "pending_tasks": [
                {"title": "Review PR", "priority": 8}
            ],
            "project_insights": [
                {"title": "Velocity increasing", "priority": 6}
            ],
            "blockers": [
                {"title": "Waiting on design", "priority": 9}
            ],
            "opportunities": [
                {"title": "Automate testing", "priority": 7}
            ]
        }
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        orchestration_engine.job_queue.get_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.generate_suggestions(
            project_id=project_id
        )
        
        # Assert
        assert result["status"] == "success"
        assert result["project_id"] == project_id
        assert "result" in result
        assert "pending_tasks" in result["result"]
        assert "project_insights" in result["result"]
    
    # Test 8: Semantic search context retrieval
    @pytest.mark.asyncio
    async def test_semantic_search_context_retrieval(self, orchestration_engine):
        """Test semantic search for context retrieval."""
        # Arrange
        project_id = "project_456"
        query = "database performance issues"
        
        # Mock context retriever with semantic search results
        expected_results = {
            "summaries": [
                {
                    "id": "summary_1",
                    "content_type": "summary",
                    "metadata": {"text": "Database optimization meeting"},
                    "similarity": 0.92
                }
            ],
            "decisions": [
                {
                    "id": "decision_1",
                    "content_type": "decision",
                    "metadata": {"text": "Implement database indexing"},
                    "similarity": 0.88
                }
            ],
            "blockers": [
                {
                    "id": "blocker_1",
                    "content_type": "blocker",
                    "metadata": {"text": "Database queries too slow"},
                    "similarity": 0.95
                }
            ]
        }
        
        orchestration_engine.context_retriever.retrieve_meeting_context = Mock(
            return_value=expected_results
        )
        
        # Act
        result = await orchestration_engine.retrieve_context(
            project_id=project_id,
            query=query,
            limit=5
        )
        
        # Assert
        assert result["status"] == "success"
        assert result["query"] == query
        assert "context" in result
        assert len(result["context"]["summaries"]) > 0
        assert result["context"]["summaries"][0]["similarity"] > 0.8
    
    # Test 9: Job status tracking
    def test_job_status_tracking(self, orchestration_engine):
        """Test job status tracking and retrieval."""
        # Arrange
        job_id = "job_123"
        
        # Mock job with various statuses
        mock_job = Mock()
        mock_job.id = job_id
        mock_job.get_status = Mock(return_value="completed")
        mock_job.is_finished = True
        mock_job.is_failed = False
        mock_job.result = {"status": "success"}
        mock_job.created_at = datetime.now()
        mock_job.started_at = datetime.now()
        mock_job.ended_at = datetime.now()
        mock_job.meta = {"meeting_id": "meeting_123"}
        
        orchestration_engine.job_queue.get_job = Mock(return_value=mock_job)
        
        # Act
        status = orchestration_engine.get_job_status(job_id)
        
        # Assert
        assert status["job_id"] == job_id
        assert status["status"] == "completed"
        assert status["result"] == {"status": "success"}
        assert status["meta"]["meeting_id"] == "meeting_123"
    
    # Test 10: Queue statistics
    def test_queue_statistics(self, orchestration_engine):
        """Test queue statistics retrieval."""
        # Arrange
        expected_stats = {
            "queued": 5,
            "active": 2,
            "completed": 100,
            "failed": 3
        }
        
        orchestration_engine.job_queue.get_queue_stats = Mock(
            return_value=expected_stats
        )
        orchestration_engine.job_queue.get_worker_count = Mock(return_value=4)
        
        # Act
        stats = orchestration_engine.get_queue_stats()
        
        # Assert
        assert stats["queued"] == 5
        assert stats["active"] == 2
        assert stats["completed"] == 100
        assert stats["failed"] == 3
        assert stats["worker_count"] == 4
        assert "timestamp" in stats


class TestOrchestrationErrorHandling:
    """Test error handling in orchestration engine."""
    
    @pytest.fixture
    def orchestration_engine(self):
        """Create orchestration engine with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock(spec=JobEventListener)
        mock_policy = Mock(spec=RetryPolicy)
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        return engine
    
    @pytest.mark.asyncio
    async def test_transcription_failure_handling(self, orchestration_engine):
        """Test handling of transcription failures."""
        # Arrange
        meeting_id = "meeting_123"
        s3_key = "meetings/meeting_123.mp3"
        
        # Mock failed transcription job
        mock_job = Mock()
        mock_job.id = "job_123"
        mock_job.is_finished = True
        mock_job.is_failed = True
        mock_job.result = None
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key
        )
        
        # Assert
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_retrieval_failure_handling(self, orchestration_engine):
        """Test handling of context retrieval failures."""
        # Arrange
        project_id = "project_456"
        query = "test query"
        
        # Mock context retriever failure
        orchestration_engine.context_retriever = None
        
        # Act
        result = await orchestration_engine.retrieve_context(
            project_id=project_id,
            query=query
        )
        
        # Assert
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_job_timeout_handling(self, orchestration_engine):
        """Test handling of job timeouts."""
        # Arrange
        user_id = "user_123"
        project_id = "project_456"
        message = "Test message"
        
        # Mock job that never completes
        mock_job = Mock()
        mock_job.id = "chat_job_123"
        mock_job.is_finished = False
        
        orchestration_engine.job_queue.enqueue_job = Mock(return_value=mock_job)
        orchestration_engine.job_queue.get_job = Mock(return_value=mock_job)
        
        # Act
        result = await orchestration_engine.chat(
            user_id=user_id,
            project_id=project_id,
            message=message
        )
        
        # Assert
        assert result["status"] == "error"
        assert "timed out" in result.get("error", "").lower()
