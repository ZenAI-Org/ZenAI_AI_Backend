"""
Unit tests for Suggestions Agent.
Tests suggestion generation for each card type, ranking, caching, and API endpoint.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import ValidationError
from datetime import datetime

from app.agents.suggestions_agent import (
    SuggestionsAgent,
    Suggestion,
    DashboardSuggestions,
    SUGGESTIONS_CACHE_TTL
)
from app.agents.base_agent import AgentConfig, AgentStatus


class TestSuggestionValidation:
    """Tests for Suggestion model validation."""
    
    def test_valid_suggestion(self):
        """Test valid suggestion passes validation."""
        data = {
            "title": "Complete pending tasks",
            "description": "There are 5 high-priority tasks waiting for review. Consider assigning them to available team members.",
            "action_url": "/tasks?priority=high",
            "priority": 8
        }
        
        suggestion = Suggestion(**data)
        
        assert suggestion.title == data["title"]
        assert suggestion.description == data["description"]
        assert suggestion.action_url == data["action_url"]
        assert suggestion.priority == data["priority"]
    
    def test_suggestion_minimal(self):
        """Test minimal valid suggestion."""
        data = {
            "title": "Review pending tasks",
            "description": "There are pending tasks that need attention and review."
        }
        
        suggestion = Suggestion(**data)
        
        assert suggestion.title == data["title"]
        assert suggestion.description == data["description"]
        assert suggestion.action_url is None
        assert suggestion.priority == 1  # Default priority
    
    def test_suggestion_title_too_short(self):
        """Test suggestion with title too short."""
        data = {
            "title": "Task",  # Less than 5 chars
            "description": "This is a valid description with sufficient length."
        }
        
        with pytest.raises(ValidationError):
            Suggestion(**data)
    
    def test_suggestion_title_too_long(self):
        """Test suggestion with title too long."""
        data = {
            "title": "x" * 201,  # More than 200 chars
            "description": "This is a valid description with sufficient length."
        }
        
        with pytest.raises(ValidationError):
            Suggestion(**data)
    
    def test_suggestion_description_too_short(self):
        """Test suggestion with description too short."""
        data = {
            "title": "Valid Title",
            "description": "Short"  # Less than 10 chars
        }
        
        with pytest.raises(ValidationError):
            Suggestion(**data)
    
    def test_suggestion_priority_out_of_range(self):
        """Test suggestion with priority out of range."""
        data = {
            "title": "Valid Title",
            "description": "This is a valid description with sufficient length.",
            "priority": 15  # Max is 10
        }
        
        with pytest.raises(ValidationError):
            Suggestion(**data)
    
    def test_suggestion_priority_zero(self):
        """Test suggestion with priority zero."""
        data = {
            "title": "Valid Title",
            "description": "This is a valid description with sufficient length.",
            "priority": 0  # Min is 1
        }
        
        with pytest.raises(ValidationError):
            Suggestion(**data)


class TestDashboardSuggestionsValidation:
    """Tests for DashboardSuggestions model validation."""
    
    def test_valid_dashboard_suggestions(self):
        """Test valid dashboard suggestions."""
        data = {
            "project_id": "project-123",
            "pending_tasks": [
                {
                    "title": "Review pending tasks",
                    "description": "There are pending tasks that need attention.",
                    "priority": 8
                }
            ],
            "project_insights": [
                {
                    "title": "Team velocity increasing",
                    "description": "Team velocity has increased by 20% this sprint.",
                    "priority": 7
                }
            ],
            "blockers": [],
            "opportunities": []
        }
        
        suggestions = DashboardSuggestions(**data)
        
        assert suggestions.project_id == "project-123"
        assert len(suggestions.pending_tasks) == 1
        assert len(suggestions.project_insights) == 1
        assert len(suggestions.blockers) == 0
        assert len(suggestions.opportunities) == 0
    
    def test_dashboard_suggestions_max_items(self):
        """Test dashboard suggestions respects max items per category."""
        data = {
            "project_id": "project-123",
            "pending_tasks": [
                {
                    "title": f"Task {i}",
                    "description": f"This is a valid description for task {i}.",
                    "priority": 5
                }
                for i in range(4)  # 4 items, max is 3
            ]
        }
        
        with pytest.raises(ValidationError):
            DashboardSuggestions(**data)
    
    def test_dashboard_suggestions_generated_at(self):
        """Test dashboard suggestions includes generated_at timestamp."""
        data = {
            "project_id": "project-123"
        }
        
        suggestions = DashboardSuggestions(**data)
        
        assert suggestions.generated_at is not None
        assert isinstance(suggestions.generated_at, str)


class TestSuggestionsAgentInitialization:
    """Tests for SuggestionsAgent initialization."""
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_agent_initialization(self, mock_redis, mock_langchain_init):
        """Test SuggestionsAgent initialization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = SuggestionsAgent(config, db_connection=None)
        
        assert agent.config == config
        assert agent.llm is not None
        assert agent.redis_client is not None
        assert agent.context_retriever is None  # No DB connection
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.ContextRetriever")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_agent_initialization_with_db(self, mock_redis, mock_context_retriever, mock_langchain_init):
        """Test SuggestionsAgent initialization with database connection."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        mock_db = MagicMock()
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=mock_db)
        
        assert agent.db_connection == mock_db
        assert agent.context_retriever is not None
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_prompt_templates_initialized(self, mock_redis, mock_langchain_init):
        """Test prompt templates are initialized."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        assert agent.pending_tasks_template is not None
        assert agent.insights_template is not None
        assert agent.blockers_template is not None
        assert agent.opportunities_template is not None


class TestSuggestionGeneration:
    """Tests for suggestion generation for each card type."""
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_pending_tasks_suggestions_generation(self, mock_redis, mock_langchain_init):
        """Test pending tasks suggestions generation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # No cache
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Review pending tasks",
                    "description": "There are 5 high-priority tasks waiting for review.",
                    "priority": 8
                },
                {
                    "title": "Assign blocked tasks",
                    "description": "3 tasks are blocked waiting for dependencies.",
                    "priority": 9
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["pending_tasks"]) > 0
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_insights_suggestions_generation(self, mock_redis, mock_langchain_init):
        """Test project insights suggestions generation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # No cache
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Team velocity increasing",
                    "description": "Team velocity has increased by 20% this sprint.",
                    "priority": 7
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["project_insights"]) > 0
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_blockers_suggestions_generation(self, mock_redis, mock_langchain_init):
        """Test blockers suggestions generation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # No cache
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Critical blocker: API integration",
                    "description": "API integration is blocking 3 dependent tasks.",
                    "priority": 10
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["blockers"]) > 0
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_opportunities_suggestions_generation(self, mock_redis, mock_langchain_init):
        """Test opportunities suggestions generation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # No cache
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Automate testing pipeline",
                    "description": "Implementing automated testing could save 5 hours per week.",
                    "priority": 6
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["opportunities"]) > 0


class TestSuggestionRanking:
    """Tests for suggestion ranking and prioritization logic."""
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_rank_suggestions_by_priority(self, mock_redis, mock_langchain_init):
        """Test suggestions are ranked by priority."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        suggestions = [
            Suggestion(title="Low priority", description="This is a low priority suggestion.", priority=3),
            Suggestion(title="High priority", description="This is a high priority suggestion.", priority=9),
            Suggestion(title="Medium priority", description="This is a medium priority suggestion.", priority=6),
        ]
        
        ranked = agent._rank_suggestions(suggestions)
        
        # Should be sorted by priority descending
        assert ranked[0].priority == 9
        assert ranked[1].priority == 6
        assert ranked[2].priority == 3
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_rank_suggestions_limits_to_three(self, mock_redis, mock_langchain_init):
        """Test ranking limits suggestions to top 3."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        suggestions = [
            Suggestion(title=f"Suggestion {i}", description=f"Description {i}.", priority=i)
            for i in range(1, 6)
        ]
        
        ranked = agent._rank_suggestions(suggestions)
        
        assert len(ranked) == 3


class TestCaching:
    """Tests for caching and refresh logic."""
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_suggestions_cached_with_ttl(self, mock_redis, mock_langchain_init):
        """Test suggestions are cached with correct TTL."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # No cache initially
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Test suggestion",
                    "description": "This is a test suggestion.",
                    "priority": 5
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        # Verify setex was called with correct TTL
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == SUGGESTIONS_CACHE_TTL
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_cached_suggestions_retrieved(self, mock_redis, mock_langchain_init):
        """Test cached suggestions are retrieved without regeneration."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        # Return cached data
        cached_data = {
            "project_id": "project-123",
            "pending_tasks": [
                {
                    "title": "Cached suggestion",
                    "description": "This is a cached suggestion.",
                    "priority": 5,
                    "generated_at": datetime.now().isoformat()
                }
            ],
            "project_insights": [],
            "blockers": [],
            "opportunities": [],
            "generated_at": datetime.now().isoformat()
        }
        mock_redis_client.get.return_value = json.dumps(cached_data)
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123", force_refresh=False)
        
        assert result.status == AgentStatus.SUCCESS
        assert result.metadata["source"] == "cache"
        # LLM should not be called when using cache
        mock_llm.invoke.assert_not_called()
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, mock_redis, mock_langchain_init):
        """Test force_refresh bypasses cache."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        # Return cached data
        cached_data = {
            "project_id": "project-123",
            "pending_tasks": [],
            "project_insights": [],
            "blockers": [],
            "opportunities": [],
            "generated_at": datetime.now().isoformat()
        }
        mock_redis_client.get.return_value = json.dumps(cached_data)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "title": "Fresh suggestion",
                    "description": "This is a fresh suggestion.",
                    "priority": 5
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123", force_refresh=True)
        
        assert result.status == AgentStatus.SUCCESS
        assert result.metadata["cached"] is False
        # LLM should be called when force_refresh is True
        mock_llm.invoke.assert_called()
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_refresh_suggestions_clears_cache(self, mock_redis, mock_langchain_init):
        """Test refresh_suggestions clears cache."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        success = agent.refresh_suggestions("project-123")
        
        assert success is True
        mock_redis_client.delete.assert_called_once_with("suggestions:project-123")


class TestResponseParsing:
    """Tests for parsing and validating LLM responses."""
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_parse_suggestions_with_markdown_json(self, mock_redis, mock_langchain_init):
        """Test parsing suggestions from markdown JSON."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        response_text = """```json
{
    "suggestions": [
        {
            "title": "Test suggestion",
            "description": "This is a test suggestion.",
            "priority": 5
        }
    ]
}
```"""
        
        suggestions = agent._parse_suggestions_response(response_text)
        
        assert len(suggestions) == 1
        assert suggestions[0].title == "Test suggestion"
        assert suggestions[0].priority == 5
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_parse_suggestions_plain_json(self, mock_redis, mock_langchain_init):
        """Test parsing plain JSON suggestions."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        response_text = json.dumps({
            "suggestions": [
                {
                    "title": "Test suggestion",
                    "description": "This is a test suggestion.",
                    "priority": 5
                }
            ]
        })
        
        suggestions = agent._parse_suggestions_response(response_text)
        
        assert len(suggestions) == 1
        assert suggestions[0].title == "Test suggestion"
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_parse_suggestions_invalid_json(self, mock_redis, mock_langchain_init):
        """Test parsing invalid JSON raises error."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        response_text = "This is not JSON"
        
        with pytest.raises(ValueError):
            agent._parse_suggestions_response(response_text)
    
    @patch("app.agents.suggestions_agent.LangChainInitializer")
    @patch("app.agents.suggestions_agent.get_redis_client")
    def test_parse_suggestions_missing_suggestions_key(self, mock_redis, mock_langchain_init):
        """Test parsing JSON without suggestions key."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        config = AgentConfig()
        agent = SuggestionsAgent(config, db_connection=None)
        
        response_text = json.dumps({"data": []})
        
        suggestions = agent._parse_suggestions_response(response_text)
        
        # Should return empty list when suggestions key is missing
        assert len(suggestions) == 0


class TestAPIEndpoint:
    """Tests for REST API endpoint."""
    
    @patch("app.agents.suggestions_agent.SuggestionsAgent")
    @pytest.mark.asyncio
    async def test_get_suggestions_endpoint_success(self, mock_agent_class):
        """Test GET /api/projects/{project_id}/suggestions endpoint success."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        mock_result = MagicMock()
        mock_result.status.value = "success"
        mock_result.data = {
            "project_id": "project-123",
            "pending_tasks": [],
            "project_insights": [],
            "blockers": [],
            "opportunities": [],
            "generated_at": datetime.now().isoformat()
        }
        mock_result.metadata = {"agent": "SuggestionsAgent"}
        
        mock_agent.execute = AsyncMock(return_value=mock_result)
        
        client = TestClient(app)
        response = client.get("/api/projects/project-123/suggestions")
        
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    @patch("app.agents.suggestions_agent.SuggestionsAgent")
    @pytest.mark.asyncio
    async def test_get_suggestions_endpoint_with_force_refresh(self, mock_agent_class):
        """Test GET endpoint with force_refresh parameter."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        mock_result = MagicMock()
        mock_result.status.value = "success"
        mock_result.data = {
            "project_id": "project-123",
            "pending_tasks": [],
            "project_insights": [],
            "blockers": [],
            "opportunities": [],
            "generated_at": datetime.now().isoformat()
        }
        mock_result.metadata = {"agent": "SuggestionsAgent"}
        
        mock_agent.execute = AsyncMock(return_value=mock_result)
        
        client = TestClient(app)
        response = client.get("/api/projects/project-123/suggestions?force_refresh=true")
        
        assert response.status_code == 200
        # Verify force_refresh was passed to execute
        mock_agent.execute.assert_called_once()
        call_kwargs = mock_agent.execute.call_args[1]
        assert call_kwargs["force_refresh"] is True
