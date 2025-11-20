"""
Unit tests for AI Product Manager (AIPM) Agent.
Tests metrics aggregation logic, analysis chain execution, recommendation generation,
and Pydantic schema validation.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import ValidationError

from app.agents.aipm_agent import (
    AIProductManagerAgent,
    AIProductManagerInsights,
    ProjectMetrics,
    Blocker,
    Recommendation,
    HealthStatus,
    ImpactLevel
)
from app.agents.base_agent import AgentConfig, AgentStatus


class TestAIPMAgentInitialization:
    """Tests for AIProductManagerAgent initialization."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_agent_initialization(self, mock_langchain_init):
        """Test AIProductManagerAgent initialization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = AIProductManagerAgent(config, db_connection=None)
        
        assert agent.config == config
        assert agent.llm is not None
        assert agent.context_retriever is None
        assert agent.embedding_store is None
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @patch("app.agents.aipm_agent.ContextRetriever")
    @patch("app.agents.aipm_agent.EmbeddingStore")
    def test_agent_initialization_with_db(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test AIProductManagerAgent initialization with database connection."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=mock_db)
        
        assert agent.db_connection == mock_db
        assert agent.context_retriever is not None
        assert agent.embedding_store is not None
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_prompt_templates_initialized(self, mock_langchain_init):
        """Test prompt templates are initialized."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        assert agent.metrics_analysis_template is not None
        assert "metrics" in agent.metrics_analysis_template.input_variables
        assert "context" in agent.metrics_analysis_template.input_variables
        assert "meeting_insights" in agent.metrics_analysis_template.input_variables


class TestProjectMetricsValidation:
    """Tests for ProjectMetrics Pydantic schema validation."""
    
    def test_valid_metrics(self):
        """Test valid project metrics."""
        metrics = ProjectMetrics(
            total_tasks=50,
            completed_tasks=40,
            overdue_tasks=2,
            task_completion_rate=80.0,
            average_task_velocity=10.0,
            team_capacity_utilization=85.0,
            active_blockers_count=1
        )
        
        assert metrics.total_tasks == 50
        assert metrics.completed_tasks == 40
        assert metrics.task_completion_rate == 80.0
    
    def test_metrics_defaults(self):
        """Test metrics with default values."""
        metrics = ProjectMetrics()
        
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.task_completion_rate == 0.0
    
    def test_completion_rate_bounds(self):
        """Test completion rate validation bounds."""
        # Valid: 0-100
        metrics = ProjectMetrics(task_completion_rate=50.0)
        assert metrics.task_completion_rate == 50.0
        
        # Invalid: > 100
        with pytest.raises(ValidationError):
            ProjectMetrics(task_completion_rate=101.0)
        
        # Invalid: < 0
        with pytest.raises(ValidationError):
            ProjectMetrics(task_completion_rate=-1.0)
    
    def test_capacity_utilization_bounds(self):
        """Test capacity utilization validation bounds."""
        # Valid: 0-100
        metrics = ProjectMetrics(team_capacity_utilization=75.0)
        assert metrics.team_capacity_utilization == 75.0
        
        # Invalid: > 100
        with pytest.raises(ValidationError):
            ProjectMetrics(team_capacity_utilization=101.0)
    
    def test_negative_task_counts(self):
        """Test that negative task counts are rejected."""
        with pytest.raises(ValidationError):
            ProjectMetrics(total_tasks=-1)
        
        with pytest.raises(ValidationError):
            ProjectMetrics(completed_tasks=-1)
        
        with pytest.raises(ValidationError):
            ProjectMetrics(overdue_tasks=-1)


class TestBlockerValidation:
    """Tests for Blocker Pydantic schema validation."""
    
    def test_valid_blocker(self):
        """Test valid blocker."""
        blocker = Blocker(
            title="Database connection timeout",
            impact=ImpactLevel.HIGH,
            affected_tasks=["task-1", "task-2"],
            suggested_resolution="Increase connection pool size and add retry logic"
        )
        
        assert blocker.title == "Database connection timeout"
        assert blocker.impact == ImpactLevel.HIGH
        assert len(blocker.affected_tasks) == 2
    
    def test_blocker_title_length(self):
        """Test blocker title length validation."""
        # Too short (less than 5 chars)
        with pytest.raises(ValidationError):
            Blocker(
                title="Bad",
                impact=ImpactLevel.MEDIUM,
                suggested_resolution="This is a valid resolution with sufficient length"
            )
        
        # Too long
        with pytest.raises(ValidationError):
            Blocker(
                title="x" * 201,
                impact=ImpactLevel.MEDIUM,
                suggested_resolution="This is a valid resolution with sufficient length"
            )
    
    def test_blocker_resolution_length(self):
        """Test blocker resolution length validation."""
        # Too short
        with pytest.raises(ValidationError):
            Blocker(
                title="Valid blocker title",
                impact=ImpactLevel.MEDIUM,
                suggested_resolution="Short"
            )
        
        # Too long
        with pytest.raises(ValidationError):
            Blocker(
                title="Valid blocker title",
                impact=ImpactLevel.MEDIUM,
                suggested_resolution="x" * 501
            )
    
    def test_blocker_affected_tasks_max(self):
        """Test blocker affected tasks max items."""
        with pytest.raises(ValidationError):
            Blocker(
                title="Valid blocker title",
                impact=ImpactLevel.MEDIUM,
                affected_tasks=[f"task-{i}" for i in range(21)],
                suggested_resolution="This is a valid resolution with sufficient length"
            )


class TestRecommendationValidation:
    """Tests for Recommendation Pydantic schema validation."""
    
    def test_valid_recommendation(self):
        """Test valid recommendation."""
        rec = Recommendation(
            title="Implement caching layer",
            rationale="Reduce database load and improve response times",
            priority=8,
            estimated_impact="30% reduction in API latency"
        )
        
        assert rec.title == "Implement caching layer"
        assert rec.priority == 8
    
    def test_recommendation_priority_bounds(self):
        """Test recommendation priority bounds (1-10)."""
        # Valid
        rec = Recommendation(
            title="Valid recommendation",
            rationale="This is a valid rationale with sufficient length",
            priority=5,
            estimated_impact="Positive impact expected"
        )
        assert rec.priority == 5
        
        # Invalid: < 1
        with pytest.raises(ValidationError):
            Recommendation(
                title="Valid recommendation",
                rationale="This is a valid rationale with sufficient length",
                priority=0,
                estimated_impact="Positive impact expected"
            )
        
        # Invalid: > 10
        with pytest.raises(ValidationError):
            Recommendation(
                title="Valid recommendation",
                rationale="This is a valid rationale with sufficient length",
                priority=11,
                estimated_impact="Positive impact expected"
            )
    
    def test_recommendation_title_length(self):
        """Test recommendation title length validation."""
        # Too short (less than 5 chars)
        with pytest.raises(ValidationError):
            Recommendation(
                title="Bad",
                rationale="This is a valid rationale with sufficient length",
                priority=5,
                estimated_impact="Positive impact expected"
            )


class TestAIPMInsightsValidation:
    """Tests for AIProductManagerInsights Pydantic schema validation."""
    
    def test_valid_insights(self):
        """Test valid AIPM insights."""
        metrics = ProjectMetrics(
            total_tasks=50,
            completed_tasks=40,
            task_completion_rate=80.0
        )
        
        blocker = Blocker(
            title="Database connection timeout",
            impact=ImpactLevel.HIGH,
            suggested_resolution="Increase connection pool size and add retry logic"
        )
        
        rec = Recommendation(
            title="Implement caching layer",
            rationale="Reduce database load and improve response times",
            priority=8,
            estimated_impact="30% reduction in API latency"
        )
        
        insights = AIProductManagerInsights(
            project_id="project-123",
            health=HealthStatus.HEALTHY,
            blockers=[blocker],
            recommendations=[rec],
            metrics=metrics,
            summary="Project is performing well with good completion rates"
        )
        
        assert insights.project_id == "project-123"
        assert insights.health == HealthStatus.HEALTHY
        assert len(insights.blockers) == 1
        assert len(insights.recommendations) == 1
    
    def test_insights_defaults(self):
        """Test AIPM insights with defaults."""
        insights = AIProductManagerInsights(project_id="project-123")
        
        assert insights.project_id == "project-123"
        assert insights.health == HealthStatus.HEALTHY
        assert insights.blockers == []
        assert insights.recommendations == []
        assert insights.summary == ""
    
    def test_insights_blockers_max(self):
        """Test insights blockers max items."""
        blockers = [
            Blocker(
                title=f"Blocker {i}",
                impact=ImpactLevel.MEDIUM,
                suggested_resolution="This is a valid resolution with sufficient length"
            )
            for i in range(11)
        ]
        
        with pytest.raises(ValidationError):
            AIProductManagerInsights(
                project_id="project-123",
                blockers=blockers
            )
    
    def test_insights_recommendations_max(self):
        """Test insights recommendations max items."""
        recommendations = [
            Recommendation(
                title=f"Recommendation {i}",
                rationale="This is a valid rationale with sufficient length",
                priority=5,
                estimated_impact="Positive impact expected"
            )
            for i in range(11)
        ]
        
        with pytest.raises(ValidationError):
            AIProductManagerInsights(
                project_id="project-123",
                recommendations=recommendations
            )


class TestMetricsAggregation:
    """Tests for metrics aggregation logic."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_aggregate_metrics_success(self, mock_langchain_init):
        """Test successful metrics aggregation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        metrics = await agent._aggregate_metrics("project-123")
        
        assert isinstance(metrics, ProjectMetrics)
        assert metrics.total_tasks > 0
        assert metrics.completed_tasks >= 0
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_aggregate_metrics_returns_defaults_on_error(self, mock_langchain_init):
        """Test metrics aggregation returns defaults on error."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        # Even if there's an error, should return default metrics
        metrics = await agent._aggregate_metrics("project-123")
        
        assert isinstance(metrics, ProjectMetrics)


class TestAnalysisChainExecution:
    """Tests for LangChain analysis chain execution."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_success(self, mock_langchain_init):
        """Test successful LLM invocation for analysis."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "health": "healthy",
            "blockers": [
                {
                    "title": "Database connection timeout",
                    "impact": "high",
                    "affected_tasks": ["task-1"],
                    "suggested_resolution": "Increase connection pool size"
                }
            ],
            "recommendations": [
                {
                    "title": "Implement caching",
                    "rationale": "Reduce database load",
                    "priority": 8,
                    "estimated_impact": "30% latency reduction"
                }
            ],
            "summary": "Project is healthy with good metrics"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["health"] == "healthy"
        assert len(result.data["blockers"]) == 1
        assert len(result.data["recommendations"]) == 1
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_with_markdown_json(self, mock_langchain_init):
        """Test LLM response with markdown code blocks."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "health": "at-risk",
    "blockers": [],
    "recommendations": [],
    "summary": "Project needs attention"
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["health"] == "at-risk"
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_failure(self, mock_langchain_init):
        """Test LLM invocation failure handling."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_llm.invoke.side_effect = Exception("API error")
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.ERROR
        assert "AIPM analysis failed" in result.error


class TestRecommendationGeneration:
    """Tests for recommendation generation logic."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_recommendations_generated(self, mock_langchain_init):
        """Test that recommendations are generated."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "health": "healthy",
            "blockers": [],
            "recommendations": [
                {
                    "title": "Recommendation 1",
                    "rationale": "This is a valid rationale with sufficient length",
                    "priority": 8,
                    "estimated_impact": "Positive impact expected"
                },
                {
                    "title": "Recommendation 2",
                    "rationale": "This is another valid rationale with sufficient length",
                    "priority": 5,
                    "estimated_impact": "Moderate impact expected"
                }
            ],
            "summary": "Project analysis complete"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert len(result.data["recommendations"]) == 2
        assert result.data["recommendations"][0]["priority"] == 8
        assert result.data["recommendations"][1]["priority"] == 5
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_recommendations_priority_clamping(self, mock_langchain_init):
        """Test that recommendation priorities are clamped to 1-10."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "health": "healthy",
            "blockers": [],
            "recommendations": [
                {
                    "title": "Recommendation 1",
                    "rationale": "This is a valid rationale with sufficient length",
                    "priority": 15,  # Should be clamped to 10
                    "estimated_impact": "Positive impact expected"
                },
                {
                    "title": "Recommendation 2",
                    "rationale": "This is another valid rationale with sufficient length",
                    "priority": -5,  # Should be clamped to 1
                    "estimated_impact": "Moderate impact expected"
                }
            ],
            "summary": "Project analysis complete"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["recommendations"][0]["priority"] == 10
        assert result.data["recommendations"][1]["priority"] == 1


class TestPydanticSchemaValidation:
    """Tests for Pydantic schema validation."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_langchain_init):
        """Test handling of invalid JSON response."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "This is not JSON"
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="project-123")
        
        assert result.status == AgentStatus.ERROR
        assert "failed" in result.error.lower()
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_validate_insights_method(self, mock_langchain_init):
        """Test validate_insights method."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        # Valid data
        valid_data = {
            "project_id": "project-123",
            "health": "healthy",
            "blockers": [],
            "recommendations": [],
            "metrics": {
                "total_tasks": 50,
                "completed_tasks": 40,
                "overdue_tasks": 0,
                "task_completion_rate": 80.0,
                "average_task_velocity": 10.0,
                "team_capacity_utilization": 85.0,
                "active_blockers_count": 0
            }
        }
        assert agent.validate_insights(valid_data) is True
        
        # Invalid data (missing project_id)
        invalid_data = {
            "health": "healthy"
        }
        assert agent.validate_insights(invalid_data) is False


class TestHealthScoreCalculation:
    """Tests for health score calculation."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_health_score_high_completion(self, mock_langchain_init):
        """Test health score with high completion rate."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        metrics = ProjectMetrics(
            total_tasks=50,
            completed_tasks=45,
            task_completion_rate=90.0,
            overdue_tasks=0,
            active_blockers_count=0,
            team_capacity_utilization=80.0
        )
        
        score = agent.calculate_health_score(metrics)
        assert score > 0.7
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_health_score_low_completion(self, mock_langchain_init):
        """Test health score with low completion rate."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        metrics = ProjectMetrics(
            total_tasks=50,
            completed_tasks=10,
            task_completion_rate=20.0,
            overdue_tasks=10,
            active_blockers_count=5,
            team_capacity_utilization=90.0
        )
        
        score = agent.calculate_health_score(metrics)
        assert score < 0.5
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    def test_health_status_determination(self, mock_langchain_init):
        """Test health status determination from score."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        # Healthy
        assert agent.determine_health_status(0.8) == HealthStatus.HEALTHY
        
        # At-risk
        assert agent.determine_health_status(0.5) == HealthStatus.AT_RISK
        
        # Critical
        assert agent.determine_health_status(0.2) == HealthStatus.CRITICAL


class TestInputValidation:
    """Tests for input validation."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_empty_project_id(self, mock_langchain_init):
        """Test handling of empty project ID."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id="")
        
        assert result.status == AgentStatus.ERROR
        assert "Project ID is required" in result.error
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_none_project_id(self, mock_langchain_init):
        """Test handling of None project ID."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        result = await agent.execute(project_id=None)
        
        assert result.status == AgentStatus.ERROR


class TestMeetingInsightsRetrieval:
    """Tests for meeting insights retrieval."""
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @patch("app.agents.aipm_agent.ContextRetriever")
    @pytest.mark.asyncio
    async def test_meeting_insights_retrieval_success(self, mock_context_retriever, mock_langchain_init):
        """Test successful meeting insights retrieval."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_context_retriever.return_value = mock_retriever_instance
        mock_retriever_instance.build_prompt_context.return_value = "Recent meeting insights"
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        agent.context_retriever = mock_retriever_instance
        
        insights = await agent._retrieve_meeting_insights("project-123")
        
        assert insights == "Recent meeting insights"
    
    @patch("app.agents.aipm_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_meeting_insights_retrieval_no_context_retriever(self, mock_langchain_init):
        """Test meeting insights retrieval without context retriever."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = AIProductManagerAgent(config, db_connection=None)
        
        insights = await agent._retrieve_meeting_insights("project-123")
        
        assert insights == ""
