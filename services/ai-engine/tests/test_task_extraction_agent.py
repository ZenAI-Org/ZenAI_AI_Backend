"""
Unit tests for Task Extraction Agent.
Tests prompt template formatting, Pydantic schema validation, LangChain chain execution,
and assignee matching logic.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import ValidationError

from app.agents.task_extraction_agent import (
    TaskExtractionAgent,
    ExtractedTask,
    TaskExtractionOutput,
)
from app.agents.base_agent import AgentConfig, AgentStatus


class TestTaskExtractionAgentInitialization:
    """Tests for TaskExtractionAgent initialization."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_agent_initialization(self, mock_langchain_init):
        """Test TaskExtractionAgent initialization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = TaskExtractionAgent(config, db_connection=None)
        
        assert agent.config == config
        assert agent.llm is not None
        assert agent.context_retriever is None  # No DB connection
        assert agent.embedding_store is None
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @patch("app.agents.task_extraction_agent.ContextRetriever")
    @patch("app.agents.task_extraction_agent.EmbeddingStore")
    def test_agent_initialization_with_db(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test TaskExtractionAgent initialization with database connection."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=mock_db)
        
        assert agent.db_connection == mock_db
        assert agent.context_retriever is not None
        assert agent.embedding_store is not None
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_prompt_templates_initialized(self, mock_langchain_init):
        """Test prompt templates are initialized."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        assert agent.extraction_template is not None
        assert "transcript" in agent.extraction_template.input_variables
        assert "context" in agent.extraction_template.input_variables


class TestExtractedTaskValidation:
    """Tests for ExtractedTask Pydantic schema validation."""
    
    def test_valid_extracted_task(self):
        """Test valid extracted task passes validation."""
        data = {
            "title": "Implement user authentication",
            "description": "Add OAuth2 authentication to the application",
            "priority": "high",
            "assignee_name": "John Doe",
            "due_date": "2024-12-31",
            "blocked": False
        }
        
        task = ExtractedTask(**data)
        
        assert task.title == data["title"]
        assert task.description == data["description"]
        assert task.priority == data["priority"]
        assert task.assignee_name == data["assignee_name"]
        assert task.due_date == data["due_date"]
        assert task.blocked is False
    
    def test_extracted_task_minimal(self):
        """Test minimal valid extracted task."""
        data = {
            "title": "Fix bug in login",
            "description": "Resolve authentication issue"
        }
        
        task = ExtractedTask(**data)
        
        assert task.title == data["title"]
        assert task.description == data["description"]
        assert task.priority == "medium"  # Default
        assert task.assignee_name is None
        assert task.due_date is None
        assert task.blocked is False
    
    def test_task_title_too_short(self):
        """Test task title that is too short fails validation."""
        data = {
            "title": "Fix",  # Less than 5 chars
            "description": "Fix something"
        }
        
        with pytest.raises(ValidationError):
            ExtractedTask(**data)
    
    def test_task_title_too_long(self):
        """Test task title that is too long fails validation."""
        data = {
            "title": "x" * 201,  # More than 200 chars
            "description": "Fix something"
        }
        
        with pytest.raises(ValidationError):
            ExtractedTask(**data)
    
    def test_task_description_too_short(self):
        """Test task description that is too short fails validation."""
        data = {
            "title": "Fix bug",
            "description": "Fix it"  # Less than 10 chars
        }
        
        with pytest.raises(ValidationError):
            ExtractedTask(**data)
    
    def test_invalid_priority(self):
        """Test invalid priority value fails validation."""
        data = {
            "title": "Fix bug in login",
            "description": "Resolve authentication issue",
            "priority": "urgent"  # Invalid, must be low/medium/high
        }
        
        with pytest.raises(ValidationError):
            ExtractedTask(**data)
    
    def test_valid_priorities(self):
        """Test all valid priority values."""
        for priority in ["low", "medium", "high"]:
            data = {
                "title": "Fix bug in login",
                "description": "Resolve authentication issue",
                "priority": priority
            }
            
            task = ExtractedTask(**data)
            assert task.priority == priority
    
    def test_blocked_task_with_blocker_description(self):
        """Test blocked task with blocker description."""
        data = {
            "title": "Deploy to production",
            "description": "Deploy the new version",
            "blocked": True,
            "blocker_description": "Waiting for security review"
        }
        
        task = ExtractedTask(**data)
        
        assert task.blocked is True
        assert task.blocker_description == "Waiting for security review"
    
    def test_whitespace_stripping(self):
        """Test whitespace is stripped from fields."""
        data = {
            "title": "  Fix bug in login  ",
            "description": "  Resolve authentication issue  ",
            "assignee_name": "  John Doe  "
        }
        
        task = ExtractedTask(**data)
        
        assert task.title == "Fix bug in login"
        assert task.description == "Resolve authentication issue"
        assert task.assignee_name == "John Doe"


class TestTaskExtractionOutputValidation:
    """Tests for TaskExtractionOutput schema validation."""
    
    def test_valid_extraction_output(self):
        """Test valid extraction output passes validation."""
        data = {
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "priority": "high",
                    "assignee_name": "John"
                }
            ],
            "blockers": ["Waiting for API", "Missing requirements"],
            "summary": "Extracted 1 task with 2 blockers"
        }
        
        output = TaskExtractionOutput(**data)
        
        assert len(output.tasks) == 1
        assert len(output.blockers) == 2
        assert output.summary == data["summary"]
    
    def test_extraction_output_empty(self):
        """Test empty extraction output."""
        data = {
            "tasks": [],
            "blockers": []
        }
        
        output = TaskExtractionOutput(**data)
        
        assert len(output.tasks) == 0
        assert len(output.blockers) == 0
        assert output.summary is None
    
    def test_tasks_max_items(self):
        """Test tasks respects max items limit."""
        data = {
            "tasks": [
                {
                    "title": f"Task {i}",
                    "description": f"Description for task {i}"
                }
                for i in range(51)  # 51 items, max is 50
            ]
        }
        
        with pytest.raises(ValidationError):
            TaskExtractionOutput(**data)
    
    def test_blockers_max_items(self):
        """Test blockers respects max items limit."""
        data = {
            "blockers": [f"Blocker {i}" for i in range(21)]  # 21 items, max is 20
        }
        
        with pytest.raises(ValidationError):
            TaskExtractionOutput(**data)


class TestPromptTemplateFormatting:
    """Tests for prompt template formatting."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_prompt_formatting_with_context(self, mock_langchain_init):
        """Test prompt template formatting with context."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        transcript = "Meeting transcript content"
        context = "Previous project context"
        
        formatted = agent.extraction_template.format(
            transcript=transcript,
            context=context
        )
        
        assert transcript in formatted
        assert context in formatted
        assert "JSON format" in formatted
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_prompt_formatting_without_context(self, mock_langchain_init):
        """Test prompt template formatting without context."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        transcript = "Meeting transcript content"
        
        formatted = agent.extraction_template.format(
            transcript=transcript,
            context=""
        )
        
        assert transcript in formatted
        assert "JSON format" in formatted


class TestContextRetrieval:
    """Tests for context retrieval and filtering."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @patch("app.agents.task_extraction_agent.ContextRetriever")
    @patch("app.agents.task_extraction_agent.EmbeddingStore")
    @pytest.mark.asyncio
    async def test_context_retrieval_success(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test successful context retrieval."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_context_retriever.return_value = mock_retriever_instance
        
        mock_retriever_instance.build_prompt_context.return_value = "Previous context"
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=mock_db)
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "priority": "high",
                    "assignee_name": "John"
                }
            ],
            "blockers": ["Blocker 1"],
            "summary": "Extracted 1 task"
        })
        mock_llm.invoke.return_value = mock_response
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            project_id="project-456"
        )
        
        assert result.status == AgentStatus.SUCCESS
        mock_retriever_instance.build_prompt_context.assert_called_once()
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @patch("app.agents.task_extraction_agent.ContextRetriever")
    @patch("app.agents.task_extraction_agent.EmbeddingStore")
    @pytest.mark.asyncio
    async def test_context_retrieval_failure_continues(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test that context retrieval failure doesn't stop extraction."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_context_retriever.return_value = mock_retriever_instance
        
        # Simulate context retrieval failure
        mock_retriever_instance.build_prompt_context.side_effect = Exception("DB error")
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=mock_db)
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system"
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            project_id="project-456"
        )
        
        # Should still succeed despite context retrieval failure
        assert result.status == AgentStatus.SUCCESS
        assert result.metadata["context_used"] is False


class TestLangChainChainExecution:
    """Tests for LangChain chain execution."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_success(self, mock_langchain_init):
        """Test successful LLM invocation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement authentication",
                    "description": "Add OAuth2 authentication to the system",
                    "priority": "high",
                    "assignee_name": "John Doe"
                },
                {
                    "title": "Fix login bug",
                    "description": "Resolve authentication issue in login flow",
                    "priority": "medium",
                    "assignee_name": "Jane Smith"
                }
            ],
            "blockers": ["Waiting for API documentation"],
            "summary": "Extracted 2 tasks with 1 blocker"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript content"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 2
        assert len(result.data["blockers"]) == 1
        assert result.data["task_count"] == 2
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_with_markdown_json(self, mock_langchain_init):
        """Test LLM response with markdown code blocks."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "tasks": [
        {
            "title": "Implement feature",
            "description": "Add new feature to the system",
            "priority": "high"
        }
    ],
    "blockers": [],
    "summary": "Extracted 1 task"
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 1
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_failure(self, mock_langchain_init):
        """Test LLM invocation failure handling."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_llm.invoke.side_effect = Exception("API error")
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "Task extraction failed" in result.error


class TestPydanticValidation:
    """Tests for Pydantic validation of extraction output."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_langchain_init):
        """Test handling of invalid JSON response."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "This is not JSON"
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "failed" in result.error.lower()
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_missing_required_task_field(self, mock_langchain_init):
        """Test handling of missing required task field."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    # Missing required "title" field
                    "description": "Task description"
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        # Should handle gracefully by skipping invalid task
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 0
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_invalid_priority_value(self, mock_langchain_init):
        """Test handling of invalid priority value."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "priority": "urgent"  # Invalid priority
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        # Should handle gracefully by correcting invalid priority to default
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 1
        assert result.data["tasks"][0]["priority"] == "medium"  # Corrected to default
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_validate_extraction_method(self, mock_langchain_init):
        """Test validate_extraction method."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        # Valid data
        valid_data = {
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system"
                }
            ],
            "blockers": []
        }
        assert agent.validate_extraction(valid_data) is True
        
        # Invalid data (tasks not a list)
        invalid_data = {
            "tasks": "not a list",
            "blockers": []
        }
        assert agent.validate_extraction(invalid_data) is False


class TestAssigneeMatching:
    """Tests for assignee matching to organization members."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_assignee_exact_match(self, mock_langchain_init):
        """Test exact assignee name matching."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "assignee_name": "john doe"
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        org_members = [
            {"id": "user-1", "name": "John Doe"},
            {"id": "user-2", "name": "Jane Smith"}
        ]
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            org_members=org_members
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 1
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_assignee_partial_match(self, mock_langchain_init):
        """Test partial assignee name matching."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "assignee_name": "John"
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        org_members = [
            {"id": "user-1", "name": "John Doe"},
            {"id": "user-2", "name": "Jane Smith"}
        ]
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            org_members=org_members
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 1
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_assignee_no_match(self, mock_langchain_init):
        """Test assignee with no matching org member."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "tasks": [
                {
                    "title": "Implement feature",
                    "description": "Add new feature to the system",
                    "assignee_name": "Unknown Person"
                }
            ],
            "blockers": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        org_members = [
            {"id": "user-1", "name": "John Doe"},
            {"id": "user-2", "name": "Jane Smith"}
        ]
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            org_members=org_members
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["tasks"]) == 1
        # Assignee name should still be preserved
        assert result.data["tasks"][0]["assignee_name"] == "Unknown Person"


class TestInputValidation:
    """Tests for input validation."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_empty_transcript(self, mock_langchain_init):
        """Test handling of empty transcript."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript=""
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty or too short" in result.error.lower()
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_very_short_transcript(self, mock_langchain_init):
        """Test handling of very short transcript."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Hi"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty or too short" in result.error.lower()


class TestExtractionQualityScore:
    """Tests for extraction quality scoring."""
    
    @patch("app.agents.task_extraction_agent.LangChainInitializer")
    def test_quality_score_calculation(self, mock_langchain_init):
        """Test quality score calculation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = TaskExtractionAgent(config, db_connection=None)
        
        # High quality extraction
        high_quality = TaskExtractionOutput(
            tasks=[
                ExtractedTask(
                    title="Implement feature",
                    description="Add new feature to the system with detailed requirements",
                    priority="high",
                    assignee_name="John Doe"
                ),
                ExtractedTask(
                    title="Fix bug",
                    description="Resolve critical issue in authentication flow",
                    priority="high",
                    assignee_name="Jane Smith"
                )
            ],
            blockers=["Waiting for API documentation", "Missing requirements"],
            summary="Extracted 2 high-priority tasks with 2 blockers"
        )
        
        score = agent.get_extraction_quality_score(high_quality)
        assert score > 0.7
        
        # Low quality extraction
        low_quality = TaskExtractionOutput(
            tasks=[],
            blockers=[]
        )
        
        score = agent.get_extraction_quality_score(low_quality)
        assert score == 0.0
