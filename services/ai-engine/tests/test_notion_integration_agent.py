"""
Unit tests for Notion Integration Agent.

Tests task creation and update logic, error handling with retry logic,
and sync status updates via Socket.io.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from app.agents.notion_integration_agent import NotionIntegrationAgent
from app.agents.base_agent import AgentConfig, AgentStatus


class TestNotionIntegrationAgentInitialization:
    """Tests for NotionIntegrationAgent initialization."""
    
    def test_agent_initialization(self):
        """Test NotionIntegrationAgent initialization."""
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = NotionIntegrationAgent(config)
        
        assert agent.config == config
        assert agent.socket_io_client is None
        assert agent.db_connection is None
        assert agent.notion_client is None
        assert agent.max_retries == 1
    
    def test_agent_initialization_with_clients(self):
        """Test NotionIntegrationAgent initialization with Socket.io and DB."""
        config = AgentConfig()
        mock_socket_io = MagicMock()
        mock_db = MagicMock()
        
        agent = NotionIntegrationAgent(
            config,
            socket_io_client=mock_socket_io,
            db_connection=mock_db
        )
        
        assert agent.socket_io_client == mock_socket_io
        assert agent.db_connection == mock_db


class TestTaskCreationWithRetry:
    """Tests for task creation with retry logic."""
    
    @pytest.mark.asyncio
    async def test_create_task_success_first_attempt(self):
        """Test successful task creation on first attempt."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock Notion client
        mock_notion = MagicMock()
        mock_notion.create_task.return_value = {
            "success": True,
            "task_id": "notion-123",
            "url": "https://notion.so/task-123"
        }
        agent.notion_client = mock_notion
        
        task = {
            "title": "Implement feature",
            "description": "Add new feature to the system",
            "priority": "high",
            "assignee_name": "John Doe",
            "due_date": "2024-12-31"
        }
        
        result = await agent._create_task_with_retry(task, "meeting-123")
        
        assert result["success"] is True
        assert result["notion_task_id"] == "notion-123"
        assert result["notion_url"] == "https://notion.so/task-123"
        mock_notion.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_task_failure_then_success(self):
        """Test task creation that fails then succeeds on retry."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock Notion client with failure then success
        mock_notion = MagicMock()
        mock_notion.create_task.side_effect = [
            Exception("API rate limit"),
            {
                "success": True,
                "task_id": "notion-456",
                "url": "https://notion.so/task-456"
            }
        ]
        agent.notion_client = mock_notion
        
        task = {
            "title": "Fix bug",
            "description": "Resolve authentication issue",
            "priority": "medium"
        }
        
        result = await agent._create_task_with_retry(task, "meeting-123")
        
        assert result["success"] is True
        assert result["notion_task_id"] == "notion-456"
        assert mock_notion.create_task.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_task_failure_all_retries(self):
        """Test task creation that fails all retry attempts."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock Notion client with persistent failure
        mock_notion = MagicMock()
        mock_notion.create_task.side_effect = Exception("API error")
        agent.notion_client = mock_notion
        
        task = {
            "title": "Deploy to production",
            "description": "Deploy the new version",
            "priority": "high"
        }
        
        result = await agent._create_task_with_retry(task, "meeting-123")
        
        assert result["success"] is False
        assert "error" in result
        # Should attempt initial + 1 retry = 2 calls
        assert mock_notion.create_task.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_task_with_minimal_data(self):
        """Test task creation with minimal required data."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        mock_notion = MagicMock()
        mock_notion.create_task.return_value = {
            "success": True,
            "task_id": "notion-789",
            "url": "https://notion.so/task-789"
        }
        agent.notion_client = mock_notion
        
        task = {
            "title": "Simple task"
        }
        
        result = await agent._create_task_with_retry(task, "meeting-123")
        
        assert result["success"] is True
        mock_notion.create_task.assert_called_once()
        
        # Verify defaults were used
        call_args = mock_notion.create_task.call_args
        assert call_args[1]["description"] == ""
        assert call_args[1]["priority"] == "Medium"


class TestTaskUpdateWithRetry:
    """Tests for task update with retry logic."""
    
    @pytest.mark.asyncio
    @patch("app.agents.notion_integration_agent.requests")
    async def test_update_task_success(self, mock_requests):
        """Test successful task update."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock Notion client
        mock_notion = MagicMock()
        mock_notion.api_key = "test-key"
        agent.notion_client = mock_notion
        
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.patch.return_value = mock_response
        
        task_updates = {
            "title": "Updated task title",
            "status": "In Progress",
            "priority": "high"
        }
        
        result = await agent._update_task_with_retry("notion-123", task_updates)
        
        assert result["success"] is True
        assert result["notion_task_id"] == "notion-123"
        mock_requests.patch.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.agents.notion_integration_agent.requests")
    async def test_update_task_failure_then_success(self, mock_requests):
        """Test task update that fails then succeeds on retry."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        mock_notion = MagicMock()
        mock_notion.api_key = "test-key"
        agent.notion_client = mock_notion
        
        # Mock responses: failure then success
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 429
        mock_response_fail.json.return_value = {"message": "Rate limited"}
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        
        mock_requests.patch.side_effect = [
            mock_response_fail,
            mock_response_success
        ]
        
        task_updates = {"status": "Done"}
        
        result = await agent._update_task_with_retry("notion-123", task_updates)
        
        assert result["success"] is True
        assert mock_requests.patch.call_count == 2
    
    @pytest.mark.asyncio
    @patch("app.agents.notion_integration_agent.requests")
    async def test_update_task_failure_all_retries(self, mock_requests):
        """Test task update that fails all retry attempts."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        mock_notion = MagicMock()
        mock_notion.api_key = "test-key"
        agent.notion_client = mock_notion
        
        # Mock persistent failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Server error"}
        mock_requests.patch.return_value = mock_response
        
        task_updates = {"status": "Done"}
        
        result = await agent._update_task_with_retry("notion-123", task_updates)
        
        assert result["success"] is False
        assert "error" in result
        assert mock_requests.patch.call_count == 2
    
    @pytest.mark.asyncio
    async def test_update_task_no_notion_client(self):
        """Test update task when Notion client not initialized."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        agent.notion_client = None
        
        result = await agent.update_task_in_notion(
            "notion-123",
            {"status": "Done"}
        )
        
        assert result["success"] is False
        assert "not initialized" in result["error"]


class TestExecuteMethod:
    """Tests for the main execute method."""
    
    @pytest.mark.asyncio
    async def test_execute_missing_required_params(self):
        """Test execute with missing required parameters."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        result = await agent.execute(
            meeting_id="meeting-123"
            # Missing project_id and organization_id
        )
        
        assert result.status == AgentStatus.ERROR
        assert "Missing required parameters" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_no_tasks(self):
        """Test execute with no tasks to sync."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            project_id="project-456",
            organization_id="org-789",
            tasks=[]
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["synced_count"] == 0
        assert result.data["failed_count"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_no_notion_integration(self):
        """Test execute when no Notion integration exists."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock _get_notion_token to return None
        agent._get_notion_token = AsyncMock(return_value=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            project_id="project-456",
            organization_id="org-789",
            tasks=[
                {
                    "title": "Task 1",
                    "description": "Description 1"
                }
            ]
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["synced_count"] == 0
        assert result.metadata["notion_integration_active"] is False
    
    @pytest.mark.asyncio
    async def test_execute_successful_sync(self):
        """Test successful task sync."""
        config = AgentConfig()
        mock_socket_io = MagicMock()
        agent = NotionIntegrationAgent(config, socket_io_client=mock_socket_io)
        
        # Mock dependencies
        agent._get_notion_token = AsyncMock(return_value="test-token")
        
        mock_notion = MagicMock()
        mock_notion.create_task.return_value = {
            "success": True,
            "task_id": "notion-123",
            "url": "https://notion.so/task-123"
        }
        
        with patch.object(agent, "notion_client", mock_notion):
            result = await agent.execute(
                meeting_id="meeting-123",
                project_id="project-456",
                organization_id="org-789",
                tasks=[
                    {
                        "title": "Task 1",
                        "description": "Description 1",
                        "priority": "high"
                    },
                    {
                        "title": "Task 2",
                        "description": "Description 2",
                        "priority": "medium"
                    }
                ],
                agent_run_id="run-123"
            )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["synced_count"] == 2
        assert result.data["failed_count"] == 0
        assert len(result.data["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_execute_partial_sync_failure(self):
        """Test sync with some tasks failing."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        agent._get_notion_token = AsyncMock(return_value="test-token")
        
        # Mock the _create_task_with_retry method directly
        async def mock_create_task(task, meeting_id):
            if task["title"] == "Task 2":
                return {
                    "success": False,
                    "task_title": task["title"],
                    "error": "API error"
                }
            return {
                "success": True,
                "task_title": task["title"],
                "notion_task_id": f"notion-{task['title']}",
                "notion_url": f"https://notion.so/{task['title']}"
            }
        
        with patch.object(agent, "_create_task_with_retry", side_effect=mock_create_task):
            result = await agent.execute(
                meeting_id="meeting-123",
                project_id="project-456",
                organization_id="org-789",
                tasks=[
                    {"title": "Task 1", "description": "Desc 1"},
                    {"title": "Task 2", "description": "Desc 2"},
                    {"title": "Task 3", "description": "Desc 3"}
                ]
            )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["synced_count"] == 2
        assert result.data["failed_count"] == 1


class TestSocketIOEventEmission:
    """Tests for Socket.io event emission."""
    
    def test_emit_socket_event_success(self):
        """Test successful Socket.io event emission."""
        config = AgentConfig()
        mock_socket_io = MagicMock()
        agent = NotionIntegrationAgent(config, socket_io_client=mock_socket_io)
        
        agent._emit_socket_event(
            "run-123",
            "sync_status",
            {"status": "running", "message": "Syncing..."}
        )
        
        mock_socket_io.emit.assert_called_once()
        call_args = mock_socket_io.emit.call_args
        assert call_args[0][0] == "sync_status"
        assert call_args[1]["room"] == "agent_run_run-123"
    
    def test_emit_socket_event_no_client(self):
        """Test Socket.io emission when client not available."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config, socket_io_client=None)
        
        # Should not raise error
        agent._emit_socket_event(
            "run-123",
            "sync_status",
            {"status": "running"}
        )
    
    def test_emit_socket_event_no_agent_run_id(self):
        """Test Socket.io emission when agent_run_id not provided."""
        config = AgentConfig()
        mock_socket_io = MagicMock()
        agent = NotionIntegrationAgent(config, socket_io_client=mock_socket_io)
        
        # Should not emit when agent_run_id is None
        agent._emit_socket_event(
            None,
            "sync_status",
            {"status": "running"}
        )
        
        mock_socket_io.emit.assert_not_called()


class TestPriorityMapping:
    """Tests for priority mapping."""
    
    def test_map_priority_low(self):
        """Test mapping low priority."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        assert agent._map_priority("low") == "Low"
        assert agent._map_priority("LOW") == "Low"
    
    def test_map_priority_medium(self):
        """Test mapping medium priority."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        assert agent._map_priority("medium") == "Medium"
        assert agent._map_priority("MEDIUM") == "Medium"
    
    def test_map_priority_high(self):
        """Test mapping high priority."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        assert agent._map_priority("high") == "High"
        assert agent._map_priority("HIGH") == "High"
    
    def test_map_priority_default(self):
        """Test mapping unknown priority defaults to Medium."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        assert agent._map_priority("urgent") == "Medium"
        assert agent._map_priority("") == "Medium"


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        """Test exception handling in execute method."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        # Mock _get_notion_token to raise exception
        agent._get_notion_token = AsyncMock(side_effect=Exception("DB error"))
        
        result = await agent.execute(
            meeting_id="meeting-123",
            project_id="project-456",
            organization_id="org-789",
            tasks=[{"title": "Task 1", "description": "Desc 1"}]
        )
        
        assert result.status == AgentStatus.ERROR
        assert "Notion sync failed" in result.error
    
    @pytest.mark.asyncio
    async def test_sync_task_exception_handling(self):
        """Test exception handling during task sync."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        agent._get_notion_token = AsyncMock(return_value="test-token")
        
        # Mock _create_task_with_retry to raise exception
        async def mock_create_task(task, meeting_id):
            raise Exception("Unexpected error")
        
        with patch.object(agent, "_create_task_with_retry", side_effect=mock_create_task):
            result = await agent.execute(
                meeting_id="meeting-123",
                project_id="project-456",
                organization_id="org-789",
                tasks=[{"title": "Task 1", "description": "Desc 1"}]
            )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["synced_count"] == 0
        assert result.data["failed_count"] == 1


class TestUpdateTaskInNotion:
    """Tests for update_task_in_notion method."""
    
    @pytest.mark.asyncio
    @patch("app.agents.notion_integration_agent.requests")
    async def test_update_task_in_notion_success(self, mock_requests):
        """Test successful task update via public method."""
        config = AgentConfig()
        mock_socket_io = MagicMock()
        agent = NotionIntegrationAgent(config, socket_io_client=mock_socket_io)
        
        mock_notion = MagicMock()
        mock_notion.api_key = "test-key"
        agent.notion_client = mock_notion
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.patch.return_value = mock_response
        
        result = await agent.update_task_in_notion(
            "notion-123",
            {"status": "Done", "priority": "high"},
            agent_run_id="run-123"
        )
        
        assert result["success"] is True
        mock_socket_io.emit.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_task_in_notion_no_client(self):
        """Test update when Notion client not initialized."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        agent.notion_client = None
        
        result = await agent.update_task_in_notion(
            "notion-123",
            {"status": "Done"}
        )
        
        assert result["success"] is False


class TestTaskDataHandling:
    """Tests for task data handling and transformation."""
    
    @pytest.mark.asyncio
    async def test_create_task_with_all_fields(self):
        """Test task creation with all optional fields."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        mock_notion = MagicMock()
        mock_notion.create_task.return_value = {
            "success": True,
            "task_id": "notion-123",
            "url": "https://notion.so/task-123"
        }
        agent.notion_client = mock_notion
        
        task = {
            "title": "Complete project",
            "description": "Finish all deliverables",
            "priority": "high",
            "assignee_name": "John Doe",
            "due_date": "2024-12-31",
            "blocked": True,
            "blocker_description": "Waiting for approval"
        }
        
        result = await agent._create_task_with_retry(task, "meeting-123")
        
        assert result["success"] is True
        mock_notion.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_multiple_tasks_preserves_order(self):
        """Test that syncing multiple tasks preserves order."""
        config = AgentConfig()
        agent = NotionIntegrationAgent(config)
        
        agent._get_notion_token = AsyncMock(return_value="test-token")
        
        # Mock _create_task_with_retry to return predictable results
        async def mock_create_task(task, meeting_id):
            # Extract task number from title
            task_num = task["title"].split()[-1]
            return {
                "success": True,
                "task_title": task["title"],
                "notion_task_id": f"notion-{task_num}",
                "notion_url": f"https://notion.so/task-{task_num}"
            }
        
        tasks = [
            {"title": f"Task {i}", "description": f"Desc {i}"}
            for i in range(3)
        ]
        
        with patch.object(agent, "_create_task_with_retry", side_effect=mock_create_task):
            result = await agent.execute(
                meeting_id="meeting-123",
                project_id="project-456",
                organization_id="org-789",
                tasks=tasks
            )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["results"]) == 3
        for i, sync_result in enumerate(result.data["results"]):
            assert sync_result["notion_task_id"] == f"notion-{i}"
