"""
Unit tests for Summarization Agent.
Tests prompt template formatting, context retrieval, LangChain chain execution,
and Zod validation for summary output.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import ValidationError

from app.agents.summarization_agent import SummarizationAgent, SummaryOutput
from app.agents.base_agent import AgentConfig, AgentStatus


class TestSummarizationAgentInitialization:
    """Tests for SummarizationAgent initialization."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_agent_initialization(self, mock_langchain_init):
        """Test SummarizationAgent initialization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = SummarizationAgent(config, db_connection=None)
        
        assert agent.config == config
        assert agent.llm is not None
        assert agent.context_retriever is None  # No DB connection
        assert agent.embedding_store is None
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @patch("app.agents.summarization_agent.ContextRetriever")
    @patch("app.agents.summarization_agent.EmbeddingStore")
    def test_agent_initialization_with_db(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test SummarizationAgent initialization with database connection."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=mock_db)
        
        assert agent.db_connection == mock_db
        assert agent.context_retriever is not None
        assert agent.embedding_store is not None
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_prompt_templates_initialized(self, mock_langchain_init):
        """Test prompt templates are initialized."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        assert agent.summarization_template is not None
        assert "transcript" in agent.summarization_template.input_variables
        assert "context" in agent.summarization_template.input_variables


class TestSummaryOutputValidation:
    """Tests for SummaryOutput Zod-like schema validation."""
    
    def test_valid_summary_output(self):
        """Test valid summary output passes validation."""
        data = {
            "summary": "This is a comprehensive test summary of the meeting with sufficient detail and length.",
            "key_decisions": ["Decision 1", "Decision 2"],
            "action_items": ["Action 1: John", "Action 2: Jane"],
            "participants": ["John", "Jane", "Bob"],
            "next_steps": "Follow up next week"
        }
        
        output = SummaryOutput(**data)
        
        assert output.summary == data["summary"]
        assert output.key_decisions == data["key_decisions"]
        assert output.action_items == data["action_items"]
        assert output.participants == data["participants"]
        assert output.next_steps == data["next_steps"]
    
    def test_summary_output_minimal(self):
        """Test minimal valid summary output."""
        data = {
            "summary": "This is a minimal but valid meeting summary with sufficient length."
        }
        
        output = SummaryOutput(**data)
        
        assert output.summary == data["summary"]
        assert output.key_decisions == []
        assert output.action_items == []
        assert output.participants == []
        assert output.next_steps is None
    
    def test_summary_too_short(self):
        """Test summary that is too short fails validation."""
        data = {
            "summary": "Too short"  # Less than 50 chars
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_summary_too_long(self):
        """Test summary that is too long fails validation."""
        data = {
            "summary": "x" * 501  # More than 500 chars
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_missing_summary_field(self):
        """Test missing summary field fails validation."""
        data = {
            "key_decisions": ["Decision 1"]
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_key_decisions_max_items(self):
        """Test key_decisions respects max items limit."""
        data = {
            "summary": "This is a valid summary with enough content.",
            "key_decisions": [f"Decision {i}" for i in range(11)]  # 11 items, max is 10
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_action_items_max_items(self):
        """Test action_items respects max items limit."""
        data = {
            "summary": "This is a valid summary with enough content.",
            "action_items": [f"Action {i}" for i in range(11)]  # 11 items, max is 10
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_participants_max_items(self):
        """Test participants respects max items limit."""
        data = {
            "summary": "This is a valid summary with enough content.",
            "participants": [f"Person {i}" for i in range(21)]  # 21 items, max is 20
        }
        
        with pytest.raises(ValidationError):
            SummaryOutput(**data)
    
    def test_whitespace_stripping(self):
        """Test whitespace is stripped from fields."""
        data = {
            "summary": "  This is a comprehensive summary with spaces and sufficient length.  ",
            "key_decisions": ["  Decision 1  ", "  Decision 2  "]
        }
        
        output = SummaryOutput(**data)
        
        assert output.summary == "This is a comprehensive summary with spaces and sufficient length."
        assert output.key_decisions[0] == "Decision 1"
        assert output.key_decisions[1] == "Decision 2"


class TestPromptTemplateFormatting:
    """Tests for prompt template formatting."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_prompt_formatting_with_context(self, mock_langchain_init):
        """Test prompt template formatting with context."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        transcript = "Meeting transcript content"
        context = "Previous meeting summary"
        
        formatted = agent.summarization_template.format(
            transcript=transcript,
            context=context
        )
        
        assert transcript in formatted
        assert context in formatted
        assert "JSON format" in formatted
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_prompt_formatting_without_context(self, mock_langchain_init):
        """Test prompt template formatting without context."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        transcript = "Meeting transcript content"
        
        formatted = agent.summarization_template.format(
            transcript=transcript,
            context=""
        )
        
        assert transcript in formatted
        assert "JSON format" in formatted


class TestContextRetrieval:
    """Tests for context retrieval and filtering."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @patch("app.agents.summarization_agent.ContextRetriever")
    @patch("app.agents.summarization_agent.EmbeddingStore")
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
        agent = SummarizationAgent(config, db_connection=mock_db)
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"],
            "participants": ["John"],
            "next_steps": "Follow up"
        })
        mock_llm.invoke.return_value = mock_response
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            project_id="project-456"
        )
        
        assert result.status == AgentStatus.SUCCESS
        mock_retriever_instance.build_prompt_context.assert_called_once()
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @patch("app.agents.summarization_agent.ContextRetriever")
    @patch("app.agents.summarization_agent.EmbeddingStore")
    @pytest.mark.asyncio
    async def test_context_retrieval_failure_continues(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test that context retrieval failure doesn't stop summarization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_retriever_instance = MagicMock()
        mock_context_retriever.return_value = mock_retriever_instance
        
        # Simulate context retrieval failure
        mock_retriever_instance.build_prompt_context.side_effect = Exception("DB error")
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=mock_db)
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"],
            "participants": ["John"],
            "next_steps": "Follow up"
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
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_success(self, mock_langchain_init):
        """Test successful LLM invocation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1", "Decision 2"],
            "action_items": ["Action 1: John", "Action 2: Jane"],
            "participants": ["John", "Jane"],
            "next_steps": "Follow up next week"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript content"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["summary"] == "This is a comprehensive meeting summary with sufficient length."
        assert len(result.data["key_decisions"]) == 2
        assert len(result.data["action_items"]) == 2
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_with_markdown_json(self, mock_langchain_init):
        """Test LLM response with markdown code blocks."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "summary": "This is a comprehensive meeting summary with sufficient length.",
    "key_decisions": ["Decision 1"],
    "action_items": ["Action 1"],
    "participants": ["John"],
    "next_steps": "Follow up"
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert "comprehensive meeting summary" in result.data["summary"]
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_llm_invocation_failure(self, mock_langchain_init):
        """Test LLM invocation failure handling."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_llm.invoke.side_effect = Exception("API error")
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "Summarization failed" in result.error


class TestZodValidation:
    """Tests for Zod-like validation of summary output."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_langchain_init):
        """Test handling of invalid JSON response."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "This is not JSON"
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "failed" in result.error.lower()
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_missing_required_field(self, mock_langchain_init):
        """Test handling of missing required field."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"]
            # Missing required "summary" field
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "failed" in result.error.lower()
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_invalid_field_type(self, mock_langchain_init):
        """Test handling of invalid field type."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": "Not a list",  # Should be a list
            "action_items": ["Action 1"],
            "participants": ["John"]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript"
        )
        
        # Should handle gracefully by converting to list
        assert result.status == AgentStatus.SUCCESS
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_validate_summary_method(self, mock_langchain_init):
        """Test validate_summary method."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        # Valid data
        valid_data = {
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"],
            "participants": ["John"]
        }
        assert agent.validate_summary(valid_data) is True
        
        # Invalid data (missing summary)
        invalid_data = {
            "key_decisions": ["Decision 1"]
        }
        assert agent.validate_summary(invalid_data) is False


class TestEmbeddingStorage:
    """Tests for embedding storage in pgvector."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @patch("app.agents.summarization_agent.ContextRetriever")
    @patch("app.agents.summarization_agent.EmbeddingStore")
    @pytest.mark.asyncio
    async def test_embedding_storage_success(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test successful embedding storage."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_embedding_instance = MagicMock()
        mock_embedding_store.return_value = mock_embedding_instance
        mock_embedding_instance.store_embedding.return_value = 123
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"],
            "participants": ["John"],
            "next_steps": "Follow up"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=mock_db)
        agent.embedding_store = mock_embedding_instance
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            project_id="project-456"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["embedding_id"] == 123
        mock_embedding_instance.store_embedding.assert_called_once()
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @patch("app.agents.summarization_agent.ContextRetriever")
    @patch("app.agents.summarization_agent.EmbeddingStore")
    @pytest.mark.asyncio
    async def test_embedding_storage_failure_continues(self, mock_embedding_store, mock_context_retriever, mock_langchain_init):
        """Test that embedding storage failure doesn't stop summarization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        mock_db = MagicMock()
        mock_embedding_instance = MagicMock()
        mock_embedding_store.return_value = mock_embedding_instance
        mock_embedding_instance.store_embedding.side_effect = Exception("Storage error")
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This is a comprehensive meeting summary with sufficient length.",
            "key_decisions": ["Decision 1"],
            "action_items": ["Action 1"],
            "participants": ["John"]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=mock_db)
        agent.embedding_store = mock_embedding_instance
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Meeting transcript",
            project_id="project-456"
        )
        
        # Should still succeed despite embedding storage failure
        assert result.status == AgentStatus.SUCCESS
        assert result.data["embedding_id"] is None


class TestInputValidation:
    """Tests for input validation."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_empty_transcript(self, mock_langchain_init):
        """Test handling of empty transcript."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript=""
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty or too short" in result.error.lower()
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    @pytest.mark.asyncio
    async def test_very_short_transcript(self, mock_langchain_init):
        """Test handling of very short transcript."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        result = await agent.execute(
            meeting_id="meeting-123",
            transcript="Hi"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty or too short" in result.error.lower()


class TestSummaryQualityScore:
    """Tests for summary quality scoring."""
    
    @patch("app.agents.summarization_agent.LangChainInitializer")
    def test_quality_score_calculation(self, mock_langchain_init):
        """Test quality score calculation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        
        config = AgentConfig()
        agent = SummarizationAgent(config, db_connection=None)
        
        # High quality summary
        high_quality = SummaryOutput(
            summary="This is a comprehensive meeting summary with sufficient length and detail.",
            key_decisions=["Decision 1", "Decision 2"],
            action_items=["Action 1", "Action 2"],
            participants=["John", "Jane"],
            next_steps="Follow up next week"
        )
        
        score = agent.get_summary_quality_score(high_quality)
        assert score > 0.7
        
        # Low quality summary
        low_quality = SummaryOutput(
            summary="This is a comprehensive meeting summary with sufficient length."
        )
        
        score = agent.get_summary_quality_score(low_quality)
        assert score < 0.5
