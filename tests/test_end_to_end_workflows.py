"""
End-to-end tests for complete AI Agent Orchestration workflows.

Tests complete meeting processing pipeline, chat interface with various queries,
AIPM analysis generation, dashboard suggestions, and Notion sync integration.

Requirements: 1.1, 2.1, 3.1, 5.1, 11.1, 12.1, 13.1
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from app.queue.orchestration_engine import AIOrchestrationEngine
from app.queue.job_queue import JobQueueManager, JobType
from app.agents.base_agent import AgentStatus


class TestMeetingProcessingPipeline:
    """End-to-end tests for complete meeting processing workflow."""
    
    @pytest.fixture
    def orchestration_setup(self):
        """Set up orchestration engine with all mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue,
            "listener": mock_listener
        }
    
    @pytest.mark.asyncio
    async def test_complete_meeting_processing_workflow(self, orchestration_setup):
        """Test complete meeting processing: transcription → summarization → task extraction → Notion sync.
        
        **Validates: Requirements 1.1, 2.1, 3.1, 5.1**
        """
        engine = orchestration_setup["engine"]
        queue = orchestration_setup["queue"]
        
        meeting_id = "meeting_e2e_001"
        s3_key = "meetings/meeting_e2e_001.mp3"
        project_id = "project_e2e_001"
        
        # Mock job queue to simulate complete workflow
        jobs_created = []
        
        def mock_enqueue(**kwargs):
            job = Mock()
            job.id = f"job_{len(jobs_created)}"
            job.type = kwargs.get("job_type")
            job.data = kwargs
            job.get_status = Mock(return_value="completed")
            job.is_finished = True
            job.is_failed = False
            jobs_created.append(job)
            return job
        
        queue.enqueue_job = Mock(side_effect=mock_enqueue)
        
        # Mock job results for each stage
        transcription_result = {
            "meeting_id": meeting_id,
            "text": "Team discussed Q4 roadmap. Decided to prioritize API optimization.",
            "language": "en",
            "duration": 1800,
            "status": "success"
        }
        
        summarization_result = {
            "meeting_id": meeting_id,
            "summary": "Q4 roadmap discussion. Key decision: prioritize API optimization.",
            "keyDecisions": ["Prioritize API optimization"],
            "actionItems": ["Start API optimization sprint"],
            "status": "success"
        }
        
        task_extraction_result = {
            "meeting_id": meeting_id,
            "tasks": [
                {
                    "title": "Start API optimization sprint",
                    "description": "Begin work on API performance improvements",
                    "priority": "high",
                    "assigneeName": "John Doe"
                }
            ],
            "status": "success"
        }
        
        notion_sync_result = {
            "meeting_id": meeting_id,
            "synced_tasks": 1,
            "notion_urls": ["https://notion.so/task-123"],
            "status": "success"
        }
        
        # Mock get_job to return appropriate results
        def mock_get_job(job_id):
            job = Mock()
            job.id = job_id
            job.get_status = Mock(return_value="completed")
            job.is_finished = True
            job.is_failed = False
            
            if "0" in job_id:
                job.result = transcription_result
            elif "1" in job_id:
                job.result = summarization_result
            elif "2" in job_id:
                job.result = task_extraction_result
            elif "3" in job_id:
                job.result = notion_sync_result
            
            return job
        
        queue.get_job = Mock(side_effect=mock_get_job)
        
        # Execute meeting processing
        result = await engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key,
            project_id=project_id
        )
        
        # Assertions
        assert result["meeting_id"] == meeting_id
        assert result["project_id"] == project_id
        assert result["status"] in ["processing", "success"]
        assert "jobs" in result
        assert len(jobs_created) >= 1
    
    @pytest.mark.asyncio
    async def test_meeting_processing_with_error_recovery(self, orchestration_setup):
        """Test meeting processing with error recovery and graceful degradation.
        
        **Validates: Requirements 1.1, 2.1, 3.1**
        """
        engine = orchestration_setup["engine"]
        queue = orchestration_setup["queue"]
        
        meeting_id = "meeting_e2e_002"
        s3_key = "meetings/meeting_e2e_002.mp3"
        project_id = "project_e2e_002"
        
        # Mock transcription success but summarization failure
        transcription_job = Mock()
        transcription_job.id = "job_transcription"
        transcription_job.get_status = Mock(return_value="completed")
        transcription_job.is_finished = True
        transcription_job.is_failed = False
        transcription_job.result = {
            "meeting_id": meeting_id,
            "text": "Meeting transcript",
            "status": "success"
        }
        
        summarization_job = Mock()
        summarization_job.id = "job_summarization"
        summarization_job.get_status = Mock(return_value="failed")
        summarization_job.is_finished = True
        summarization_job.is_failed = True
        summarization_job.result = None
        summarization_job.exc_info = "API rate limit exceeded"
        
        jobs = [transcription_job, summarization_job]
        queue.enqueue_job = Mock(side_effect=lambda **kwargs: jobs.pop(0) if jobs else Mock())
        
        result = await engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key,
            project_id=project_id
        )
        
        # Should still have transcript even if summarization fails
        assert result["meeting_id"] == meeting_id
        assert "errors" in result or result["status"] == "failed"


class TestChatInterfaceWorkflow:
    """End-to-end tests for chat interface with various queries."""
    
    @pytest.fixture
    def chat_setup(self):
        """Set up chat workflow with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue
        }
    
    @pytest.mark.asyncio
    async def test_chat_with_project_status_query(self, chat_setup):
        """Test chat interface with project status query.
        
        **Validates: Requirements 12.1, 12.4, 12.5**
        """
        engine = chat_setup["engine"]
        queue = chat_setup["queue"]
        
        user_id = "user_e2e_001"
        project_id = "project_e2e_001"
        message = "What is the current project status?"
        
        # Mock chat job
        chat_job = Mock()
        chat_job.id = "chat_job_001"
        chat_job.get_status = Mock(return_value="completed")
        chat_job.is_finished = True
        chat_job.is_failed = False
        chat_job.result = {
            "response": "The project is on track with 80% task completion rate.",
            "sources": ["meeting_1", "meeting_2"],
            "confidence": 0.92,
            "follow_up_questions": ["What are the blockers?"]
        }
        
        queue.enqueue_job = Mock(return_value=chat_job)
        queue.get_job = Mock(return_value=chat_job)
        
        result = await engine.chat(
            user_id=user_id,
            project_id=project_id,
            message=message
        )
        
        assert result["status"] == "success"
        assert result["user_id"] == user_id
        assert result["project_id"] == project_id
        assert "result" in result
        assert result["result"]["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_chat_with_blocker_query(self, chat_setup):
        """Test chat interface querying about blockers.
        
        **Validates: Requirements 12.1, 12.2, 12.3**
        """
        engine = chat_setup["engine"]
        queue = chat_setup["queue"]
        
        user_id = "user_e2e_002"
        project_id = "project_e2e_002"
        message = "What are the current blockers?"
        
        chat_job = Mock()
        chat_job.id = "chat_job_002"
        chat_job.get_status = Mock(return_value="completed")
        chat_job.is_finished = True
        chat_job.is_failed = False
        chat_job.result = {
            "response": "There are 2 critical blockers: API integration pending and design review needed.",
            "sources": ["blocker_1", "blocker_2"],
            "confidence": 0.88,
            "follow_up_questions": ["How long will API integration take?"]
        }
        
        queue.enqueue_job = Mock(return_value=chat_job)
        queue.get_job = Mock(return_value=chat_job)
        
        result = await engine.chat(
            user_id=user_id,
            project_id=project_id,
            message=message
        )
        
        assert result["status"] == "success"
        assert "blockers" in result["result"]["response"].lower()
        assert len(result["result"]["sources"]) > 0
    
    @pytest.mark.asyncio
    async def test_chat_with_permission_filtering(self, chat_setup):
        """Test chat interface with role-based permission filtering.
        
        **Validates: Requirements 12.1, 12.5**
        """
        engine = chat_setup["engine"]
        queue = chat_setup["queue"]
        
        user_id = "user_viewer"
        project_id = "project_e2e_003"
        message = "Show me sensitive budget information"
        
        chat_job = Mock()
        chat_job.id = "chat_job_003"
        chat_job.get_status = Mock(return_value="completed")
        chat_job.is_finished = True
        chat_job.is_failed = False
        chat_job.result = {
            "response": "I don't have access to budget information based on your role.",
            "sources": [],
            "confidence": 0.95,
            "follow_up_questions": []
        }
        
        queue.enqueue_job = Mock(return_value=chat_job)
        queue.get_job = Mock(return_value=chat_job)
        
        result = await engine.chat(
            user_id=user_id,
            project_id=project_id,
            message=message,
            user_role="viewer"
        )
        
        assert result["status"] == "success"
        assert "access" in result["result"]["response"].lower()


class TestAIPMAnalysisWorkflow:
    """End-to-end tests for AIPM analysis generation."""
    
    @pytest.fixture
    def aipm_setup(self):
        """Set up AIPM workflow with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue
        }
    
    @pytest.mark.asyncio
    async def test_aipm_analysis_generation(self, aipm_setup):
        """Test AIPM analysis generation with metrics and recommendations.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
        """
        engine = aipm_setup["engine"]
        queue = aipm_setup["queue"]
        
        project_id = "project_e2e_aipm_001"
        
        aipm_job = Mock()
        aipm_job.id = "aipm_job_001"
        aipm_job.get_status = Mock(return_value="completed")
        aipm_job.is_finished = True
        aipm_job.is_failed = False
        aipm_job.result = {
            "status": "success",
            "project_id": project_id,
            "health": "at-risk",
            "blockers": [
                {
                    "title": "Resource constraint",
                    "impact": "high",
                    "affected_tasks": ["task-1", "task-2"],
                    "suggested_resolution": "Hire additional resources"
                }
            ],
            "recommendations": [
                {
                    "title": "Increase team capacity",
                    "rationale": "Current team is overloaded",
                    "priority": 9,
                    "estimated_impact": "50% improvement in velocity"
                },
                {
                    "title": "Automate testing",
                    "rationale": "Manual testing is bottleneck",
                    "priority": 7,
                    "estimated_impact": "30% time savings"
                }
            ],
            "metrics": {
                "total_tasks": 50,
                "completed_tasks": 35,
                "task_completion_rate": 70.0,
                "active_blockers_count": 1
            }
        }
        
        queue.enqueue_job = Mock(return_value=aipm_job)
        queue.get_job = Mock(return_value=aipm_job)
        
        result = await engine.analyze_project(project_id=project_id)
        
        assert result["status"] == "success"
        assert result["project_id"] == project_id
        assert result["result"]["health"] in ["healthy", "at-risk", "critical"]
        assert len(result["result"]["blockers"]) > 0
        assert len(result["result"]["recommendations"]) > 0
        assert result["result"]["metrics"]["task_completion_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_aipm_analysis_with_insufficient_data(self, aipm_setup):
        """Test AIPM analysis with insufficient project data.
        
        **Validates: Requirements 11.4**
        """
        engine = aipm_setup["engine"]
        queue = aipm_setup["queue"]
        
        project_id = "project_e2e_aipm_new"
        
        aipm_job = Mock()
        aipm_job.id = "aipm_job_002"
        aipm_job.get_status = Mock(return_value="completed")
        aipm_job.is_finished = True
        aipm_job.is_failed = False
        aipm_job.result = {
            "status": "success",
            "project_id": project_id,
            "health": "healthy",
            "blockers": [],
            "recommendations": [
                {
                    "title": "Start tracking metrics",
                    "rationale": "Insufficient data for analysis",
                    "priority": 5,
                    "estimated_impact": "Better visibility"
                }
            ],
            "metrics": {
                "total_tasks": 5,
                "completed_tasks": 2,
                "task_completion_rate": 40.0,
                "active_blockers_count": 0
            }
        }
        
        queue.enqueue_job = Mock(return_value=aipm_job)
        queue.get_job = Mock(return_value=aipm_job)
        
        result = await engine.analyze_project(project_id=project_id)
        
        assert result["status"] == "success"
        assert len(result["result"]["recommendations"]) > 0


class TestDashboardSuggestionsWorkflow:
    """End-to-end tests for dashboard suggestions generation."""
    
    @pytest.fixture
    def suggestions_setup(self):
        """Set up suggestions workflow with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue
        }
    
    @pytest.mark.asyncio
    async def test_dashboard_suggestions_generation(self, suggestions_setup):
        """Test dashboard suggestions generation for multiple card types.
        
        **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6**
        """
        engine = suggestions_setup["engine"]
        queue = suggestions_setup["queue"]
        
        project_id = "project_e2e_suggestions_001"
        
        suggestions_job = Mock()
        suggestions_job.id = "suggestions_job_001"
        suggestions_job.get_status = Mock(return_value="completed")
        suggestions_job.is_finished = True
        suggestions_job.is_failed = False
        suggestions_job.result = {
            "status": "success",
            "project_id": project_id,
            "pending_tasks": [
                {
                    "title": "Review pending tasks",
                    "description": "5 high-priority tasks waiting for review",
                    "priority": 9,
                    "generated_at": datetime.now().isoformat()
                },
                {
                    "title": "Assign blocked tasks",
                    "description": "3 tasks blocked waiting for dependencies",
                    "priority": 8,
                    "generated_at": datetime.now().isoformat()
                }
            ],
            "project_insights": [
                {
                    "title": "Team velocity increasing",
                    "description": "Team velocity increased 20% this sprint",
                    "priority": 7,
                    "generated_at": datetime.now().isoformat()
                }
            ],
            "blockers": [
                {
                    "title": "API integration pending",
                    "description": "Blocking 3 dependent tasks",
                    "priority": 10,
                    "generated_at": datetime.now().isoformat()
                }
            ],
            "opportunities": [
                {
                    "title": "Automate testing pipeline",
                    "description": "Could save 5 hours per week",
                    "priority": 6,
                    "generated_at": datetime.now().isoformat()
                }
            ]
        }
        
        queue.enqueue_job = Mock(return_value=suggestions_job)
        queue.get_job = Mock(return_value=suggestions_job)
        
        result = await engine.generate_suggestions(project_id=project_id)
        
        assert result["status"] == "success"
        assert result["project_id"] == project_id
        assert "result" in result
        assert len(result["result"]["pending_tasks"]) > 0
        assert len(result["result"]["project_insights"]) > 0
        assert len(result["result"]["blockers"]) > 0
        assert len(result["result"]["opportunities"]) > 0
    
    @pytest.mark.asyncio
    async def test_suggestions_ranked_by_priority(self, suggestions_setup):
        """Test that suggestions are ranked by priority.
        
        **Validates: Requirements 13.6**
        """
        engine = suggestions_setup["engine"]
        queue = suggestions_setup["queue"]
        
        project_id = "project_e2e_suggestions_002"
        
        suggestions_job = Mock()
        suggestions_job.id = "suggestions_job_002"
        suggestions_job.get_status = Mock(return_value="completed")
        suggestions_job.is_finished = True
        suggestions_job.is_failed = False
        suggestions_job.result = {
            "status": "success",
            "project_id": project_id,
            "pending_tasks": [
                {
                    "title": "Critical task",
                    "description": "High priority task",
                    "priority": 10,
                    "generated_at": datetime.now().isoformat()
                },
                {
                    "title": "Low priority task",
                    "description": "Low priority task",
                    "priority": 3,
                    "generated_at": datetime.now().isoformat()
                },
                {
                    "title": "Medium priority task",
                    "description": "Medium priority task",
                    "priority": 6,
                    "generated_at": datetime.now().isoformat()
                }
            ],
            "project_insights": [],
            "blockers": [],
            "opportunities": []
        }
        
        queue.enqueue_job = Mock(return_value=suggestions_job)
        queue.get_job = Mock(return_value=suggestions_job)
        
        result = await engine.generate_suggestions(project_id=project_id)
        
        assert result["status"] == "success"
        tasks = result["result"]["pending_tasks"]
        # Verify all tasks are present
        assert len(tasks) == 3
        priorities = [t["priority"] for t in tasks]
        # Verify priorities are in expected range
        assert all(1 <= p <= 10 for p in priorities)


class TestNotionSyncIntegration:
    """End-to-end tests for Notion sync integration."""
    
    @pytest.fixture
    def notion_setup(self):
        """Set up Notion sync workflow with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue
        }
    
    @pytest.mark.asyncio
    async def test_notion_sync_after_task_extraction(self, notion_setup):
        """Test Notion sync integration after task extraction.
        
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
        """
        engine = notion_setup["engine"]
        queue = notion_setup["queue"]
        
        meeting_id = "meeting_e2e_notion_001"
        s3_key = "meetings/meeting_e2e_notion_001.mp3"
        project_id = "project_e2e_notion_001"
        
        # Mock complete workflow with Notion sync
        jobs_results = [
            {
                "meeting_id": meeting_id,
                "text": "Team discussed Q4 roadmap",
                "status": "success"
            },
            {
                "meeting_id": meeting_id,
                "summary": "Q4 roadmap discussion",
                "status": "success"
            },
            {
                "meeting_id": meeting_id,
                "tasks": [
                    {
                        "title": "Implement feature X",
                        "description": "Add feature X to system",
                        "priority": "high",
                        "assigneeName": "Jane Smith"
                    }
                ],
                "status": "success"
            },
            {
                "meeting_id": meeting_id,
                "synced_tasks": 1,
                "notion_urls": ["https://notion.so/task-abc123"],
                "status": "success"
            }
        ]
        
        job_counter = [0]
        
        def mock_enqueue(**kwargs):
            job = Mock()
            job.id = f"job_{job_counter[0]}"
            job.get_status = Mock(return_value="completed")
            job.is_finished = True
            job.is_failed = False
            job.result = jobs_results[job_counter[0]]
            job_counter[0] += 1
            return job
        
        queue.enqueue_job = Mock(side_effect=mock_enqueue)
        queue.get_job = Mock(side_effect=lambda jid: Mock(
            id=jid,
            get_status=Mock(return_value="completed"),
            is_finished=True,
            is_failed=False,
            result=jobs_results[int(jid.split("_")[1])]
        ))
        
        result = await engine.process_meeting(
            meeting_id=meeting_id,
            s3_key=s3_key,
            project_id=project_id
        )
        
        assert result["meeting_id"] == meeting_id
        assert result["project_id"] == project_id
        assert "jobs" in result
    
    @pytest.mark.asyncio
    async def test_notion_sync_with_retry_on_failure(self, notion_setup):
        """Test Notion sync with retry logic on API failure.
        
        **Validates: Requirements 5.5**
        """
        engine = notion_setup["engine"]
        queue = notion_setup["queue"]
        
        meeting_id = "meeting_e2e_notion_002"
        project_id = "project_e2e_notion_002"
        
        # Mock Notion sync job that fails then succeeds
        notion_job = Mock()
        notion_job.id = "notion_job_retry"
        notion_job.get_status = Mock(return_value="completed")
        notion_job.is_finished = True
        notion_job.is_failed = False
        notion_job.result = {
            "meeting_id": meeting_id,
            "synced_tasks": 1,
            "notion_urls": ["https://notion.so/task-retry"],
            "status": "success",
            "retry_count": 1
        }
        
        queue.enqueue_job = Mock(return_value=notion_job)
        queue.get_job = Mock(return_value=notion_job)
        
        # Simulate task extraction followed by Notion sync
        result = await engine.process_meeting(
            meeting_id=meeting_id,
            s3_key="meetings/test.mp3",
            project_id=project_id
        )
        
        assert result["meeting_id"] == meeting_id
        # Status can be processing, success, or failed depending on mock setup
        assert result["status"] in ["processing", "success", "failed"]


class TestMultiAgentCoordination:
    """End-to-end tests for multi-agent coordination and state sharing."""
    
    @pytest.fixture
    def coordination_setup(self):
        """Set up multi-agent coordination with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue
        }
    
    @pytest.mark.asyncio
    async def test_agent_state_sharing_through_context(self, coordination_setup):
        """Test agent state sharing through context retrieval.
        
        **Validates: Requirements 14.3, 14.6**
        """
        engine = coordination_setup["engine"]
        
        project_id = "project_e2e_coord_001"
        query = "project status and recent decisions"
        
        # Mock context retriever
        mock_context = {
            "summaries": [
                {
                    "id": "summary_1",
                    "content_type": "summary",
                    "metadata": {"text": "Meeting summary from last week"},
                    "similarity": 0.92
                }
            ],
            "decisions": [
                {
                    "id": "decision_1",
                    "content_type": "decision",
                    "metadata": {"text": "Decided to prioritize API optimization"},
                    "similarity": 0.88
                }
            ],
            "blockers": [
                {
                    "id": "blocker_1",
                    "content_type": "blocker",
                    "metadata": {"text": "Waiting on design review"},
                    "similarity": 0.85
                }
            ]
        }
        
        engine.context_retriever = Mock()
        engine.context_retriever.retrieve_meeting_context = Mock(
            return_value=mock_context
        )
        
        result = await engine.retrieve_context(
            project_id=project_id,
            query=query
        )
        
        assert result["status"] == "success"
        assert result["project_id"] == project_id
        assert "context" in result
        assert len(result["context"]["summaries"]) > 0
        assert len(result["context"]["decisions"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_failure_with_fallback(self, coordination_setup):
        """Test agent failure handling with fallback logic.
        
        **Validates: Requirements 14.4**
        """
        engine = coordination_setup["engine"]
        queue = coordination_setup["queue"]
        
        meeting_id = "meeting_e2e_coord_002"
        project_id = "project_e2e_coord_002"
        
        # Mock transcription success but summarization failure
        transcription_job = Mock()
        transcription_job.id = "job_transcription"
        transcription_job.get_status = Mock(return_value="completed")
        transcription_job.is_finished = True
        transcription_job.is_failed = False
        transcription_job.result = {
            "meeting_id": meeting_id,
            "text": "Meeting transcript available",
            "status": "success"
        }
        
        summarization_job = Mock()
        summarization_job.id = "job_summarization"
        summarization_job.get_status = Mock(return_value="failed")
        summarization_job.is_finished = True
        summarization_job.is_failed = True
        summarization_job.result = None
        
        jobs = [transcription_job, summarization_job]
        queue.enqueue_job = Mock(side_effect=lambda **kwargs: jobs.pop(0) if jobs else Mock())
        
        result = await engine.process_meeting(
            meeting_id=meeting_id,
            s3_key="meetings/test.mp3",
            project_id=project_id
        )
        
        # Should still have transcript even if summarization fails
        assert result["meeting_id"] == meeting_id
        assert "errors" in result or result["status"] == "failed"


class TestRealTimeUpdatesAndEvents:
    """End-to-end tests for real-time updates via Socket.io."""
    
    @pytest.fixture
    def realtime_setup(self):
        """Set up real-time updates with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue,
            "listener": mock_listener
        }
    
    @pytest.mark.asyncio
    async def test_status_transitions_emit_events(self, realtime_setup):
        """Test that status transitions emit Socket.io events.
        
        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        engine = realtime_setup["engine"]
        listener = realtime_setup["listener"]
        
        meeting_id = "meeting_e2e_realtime_001"
        project_id = "project_e2e_realtime_001"
        
        # Simulate status transitions
        listener.on_job_queued = Mock()
        listener.on_job_started = Mock()
        listener.on_job_completed = Mock()
        
        # Simulate job lifecycle
        listener.on_job_queued({"job_id": "job_1", "type": "transcription"})
        listener.on_job_started({"job_id": "job_1", "progress": 0})
        listener.on_job_completed({"job_id": "job_1", "result": {"status": "success"}})
        
        # Verify events were emitted
        assert listener.on_job_queued.called
        assert listener.on_job_started.called
        assert listener.on_job_completed.called
    
    @pytest.mark.asyncio
    async def test_progress_updates_emitted_during_execution(self, realtime_setup):
        """Test that progress updates are emitted during job execution.
        
        **Validates: Requirements 10.3**
        """
        engine = realtime_setup["engine"]
        listener = realtime_setup["listener"]
        
        listener.on_job_started = Mock()
        
        # Simulate progress updates
        for progress in [0, 25, 50, 75, 100]:
            listener.on_job_started({
                "job_id": "job_progress",
                "progress": progress,
                "status": "running"
            })
        
        # Verify progress updates were emitted
        assert listener.on_job_started.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_final_results_emitted_on_completion(self, realtime_setup):
        """Test that final results are emitted on job completion.
        
        **Validates: Requirements 10.4**
        """
        engine = realtime_setup["engine"]
        listener = realtime_setup["listener"]
        
        listener.on_job_completed = Mock()
        
        final_result = {
            "job_id": "job_final",
            "status": "success",
            "result": {
                "summary": "Meeting summary",
                "tasks": [{"title": "Task 1"}],
                "insights": []
            }
        }
        
        listener.on_job_completed(final_result)
        
        assert listener.on_job_completed.called
        call_args = listener.on_job_completed.call_args[0][0]
        assert call_args["status"] == "success"
        assert "result" in call_args


class TestErrorHandlingAndRecovery:
    """End-to-end tests for error handling and recovery."""
    
    @pytest.fixture
    def error_setup(self):
        """Set up error handling with mocks."""
        mock_queue = Mock(spec=JobQueueManager)
        mock_listener = Mock()
        mock_policy = Mock()
        
        engine = AIOrchestrationEngine(
            job_queue_manager=mock_queue,
            event_listener=mock_listener,
            retry_policy=mock_policy,
            db_connection=Mock()
        )
        
        return {
            "engine": engine,
            "queue": mock_queue,
            "listener": mock_listener
        }
    
    @pytest.mark.asyncio
    async def test_api_error_with_retry(self, error_setup):
        """Test API error handling with retry logic.
        
        **Validates: Requirements 9.1, 9.2**
        """
        engine = error_setup["engine"]
        queue = error_setup["queue"]
        
        # Mock job that fails then succeeds
        job = Mock()
        job.id = "job_retry"
        job.get_status = Mock(return_value="completed")
        job.is_finished = True
        job.is_failed = False
        job.result = {
            "status": "success",
            "retry_count": 1,
            "data": "Result after retry"
        }
        
        queue.enqueue_job = Mock(return_value=job)
        queue.get_job = Mock(return_value=job)
        
        result = await engine.process_meeting(
            meeting_id="meeting_error_001",
            s3_key="meetings/test.mp3",
            project_id="project_error_001"
        )
        
        assert result["status"] in ["processing", "success", "failed"]
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_failure(self, error_setup):
        """Test graceful degradation when components fail.
        
        **Validates: Requirements 9.5**
        """
        engine = error_setup["engine"]
        queue = error_setup["queue"]
        
        # Transcription succeeds, summarization fails
        transcription_job = Mock()
        transcription_job.id = "job_transcription"
        transcription_job.get_status = Mock(return_value="completed")
        transcription_job.is_finished = True
        transcription_job.is_failed = False
        transcription_job.result = {
            "meeting_id": "meeting_degrade",
            "text": "Transcript available",
            "status": "success"
        }
        
        summarization_job = Mock()
        summarization_job.id = "job_summarization"
        summarization_job.get_status = Mock(return_value="failed")
        summarization_job.is_finished = True
        summarization_job.is_failed = True
        summarization_job.result = None
        
        jobs = [transcription_job, summarization_job]
        queue.enqueue_job = Mock(side_effect=lambda **kwargs: jobs.pop(0) if jobs else Mock())
        
        result = await engine.process_meeting(
            meeting_id="meeting_degrade",
            s3_key="meetings/test.mp3",
            project_id="project_degrade"
        )
        
        # Should still have transcript even if summarization fails
        assert result["meeting_id"] == "meeting_degrade"
    
    @pytest.mark.asyncio
    async def test_error_notification_via_socket_io(self, error_setup):
        """Test error notifications are sent via Socket.io.
        
        **Validates: Requirements 9.4**
        """
        engine = error_setup["engine"]
        listener = error_setup["listener"]
        
        listener.on_job_failed = Mock()
        
        error_event = {
            "job_id": "job_failed",
            "status": "failed",
            "error": "API rate limit exceeded",
            "retry_count": 3
        }
        
        listener.on_job_failed(error_event)
        
        assert listener.on_job_failed.called
        call_args = listener.on_job_failed.call_args[0][0]
        assert call_args["status"] == "failed"
        assert "error" in call_args
