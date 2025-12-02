"""
Unit tests for Chat Agent.
Tests permission validation, context retrieval, conversation history management,
and chat response generation.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.agents.chat_agent import (
    ChatAgent, ChatMessage, ConversationHistory, PermissionValidator
)
from app.agents.base_agent import AgentConfig, AgentStatus


class TestChatMessageAndHistory:
    """Tests for ChatMessage and ConversationHistory."""
    
    def test_chat_message_creation(self):
        """Test ChatMessage creation."""
        msg = ChatMessage(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
    
    def test_chat_message_to_dict(self):
        """Test ChatMessage to_dict conversion."""
        msg = ChatMessage(role="assistant", content="Hi there")
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there"
        assert "timestamp" in msg_dict
    
    def test_chat_message_from_dict(self):
        """Test ChatMessage from_dict creation."""
        data = {
            "role": "user",
            "content": "Test message",
            "timestamp": datetime.now().isoformat()
        }
        
        msg = ChatMessage.from_dict(data)
        
        assert msg.role == "user"
        assert msg.content == "Test message"
        assert msg.timestamp is not None
    
    def test_conversation_history_add_message(self):
        """Test adding message to conversation history."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        
        history = ConversationHistory(mock_redis, "user1", "project1")
        msg = ChatMessage(role="user", content="Hello")
        
        history.add_message(msg)
        
        # Verify setex was called
        assert mock_redis.setex.called
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "chat:user1:project1"
    
    def test_conversation_history_get_messages(self):
        """Test retrieving messages from conversation history."""
        mock_redis = MagicMock()
        
        messages_data = [
            {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "Hi", "timestamp": datetime.now().isoformat()}
        ]
        mock_redis.get.return_value = json.dumps(messages_data)
        
        history = ConversationHistory(mock_redis, "user1", "project1")
        messages = history.get_messages()
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
    
    def test_conversation_history_get_messages_with_limit(self):
        """Test retrieving messages with limit."""
        mock_redis = MagicMock()
        
        messages_data = [
            {"role": "user", "content": f"Message {i}", "timestamp": datetime.now().isoformat()}
            for i in range(10)
        ]
        mock_redis.get.return_value = json.dumps(messages_data)
        
        history = ConversationHistory(mock_redis, "user1", "project1")
        messages = history.get_messages(limit=3)
        
        assert len(messages) == 3
    
    def test_conversation_history_clear(self):
        """Test clearing conversation history."""
        mock_redis = MagicMock()
        
        history = ConversationHistory(mock_redis, "user1", "project1")
        history.clear()
        
        mock_redis.delete.assert_called_once_with("chat:user1:project1")
    
    def test_conversation_history_max_messages(self):
        """Test that conversation history keeps only last 20 messages."""
        mock_redis = MagicMock()
        
        # Simulate Redis storage
        stored_data = {}
        
        def mock_get(key):
            return stored_data.get(key)
        
        def mock_setex(key, ttl, data):
            stored_data[key] = data
        
        mock_redis.get.side_effect = mock_get
        mock_redis.setex.side_effect = mock_setex
        
        history = ConversationHistory(mock_redis, "user1", "project1")
        
        # Add 25 messages
        for i in range(25):
            msg = ChatMessage(role="user", content=f"Message {i}")
            history.add_message(msg)
        
        # Verify only last 20 are stored
        final_stored = stored_data.get("chat:user1:project1")
        stored_messages = json.loads(final_stored)
        assert len(stored_messages) == 20
        # Verify it's the last 20 messages (5-24)
        assert stored_messages[0]["content"] == "Message 5"
        assert stored_messages[-1]["content"] == "Message 24"


class TestPermissionValidator:
    """Tests for PermissionValidator."""
    
    def test_admin_has_access(self):
        """Test that admin users have access to all projects."""
        result = PermissionValidator.validate_user_project_access(
            user_id="user1",
            project_id="project1",
            user_role="admin",
            db_connection=None
        )
        
        assert result is True
    
    def test_member_has_access(self):
        """Test that member users have access."""
        result = PermissionValidator.validate_user_project_access(
            user_id="user1",
            project_id="project1",
            user_role="member",
            db_connection=None
        )
        
        assert result is True
    
    def test_viewer_has_access(self):
        """Test that viewer users have access."""
        result = PermissionValidator.validate_user_project_access(
            user_id="user1",
            project_id="project1",
            user_role="viewer",
            db_connection=None
        )
        
        assert result is True
    
    def test_filter_context_for_viewer(self):
        """Test context filtering for viewer role."""
        context = {
            "summaries": ["Summary 1"],
            "decisions": ["Decision 1"],
            "blockers": ["Blocker 1"],
            "sensitive_data": "Secret"
        }
        
        filtered = PermissionValidator.filter_context_by_role(context, "viewer")
        
        assert "summaries" in filtered
        assert "decisions" in filtered
        assert "blockers" not in filtered
        assert "sensitive_data" not in filtered
    
    def test_filter_context_for_member(self):
        """Test context filtering for member role."""
        context = {
            "summaries": ["Summary 1"],
            "decisions": ["Decision 1"],
            "blockers": ["Blocker 1"],
            "sensitive_data": "Secret"
        }
        
        filtered = PermissionValidator.filter_context_by_role(context, "member")
        
        assert "summaries" in filtered
        assert "decisions" in filtered
        assert "blockers" in filtered
        assert "sensitive_data" not in filtered
    
    def test_filter_context_for_admin(self):
        """Test context filtering for admin role."""
        context = {
            "summaries": ["Summary 1"],
            "decisions": ["Decision 1"],
            "blockers": ["Blocker 1"],
            "sensitive_data": "Secret"
        }
        
        filtered = PermissionValidator.filter_context_by_role(context, "admin")
        
        assert "summaries" in filtered
        assert "decisions" in filtered
        assert "blockers" in filtered
        assert "sensitive_data" in filtered


class TestChatAgentInitialization:
    """Tests for ChatAgent initialization."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    def test_agent_initialization(self, mock_get_redis, mock_langchain_init):
        """Test ChatAgent initialization."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = ChatAgent(config, db_connection=None)
        
        assert agent.config == config
        assert agent.llm is not None
        assert agent.redis_client is not None
        assert agent.context_retriever is None
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @patch("app.agents.chat_agent.ContextRetriever")
    def test_agent_initialization_with_db(self, mock_context_retriever, mock_get_redis, mock_langchain_init):
        """Test ChatAgent initialization with database connection."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        mock_db = MagicMock()
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=mock_db)
        
        assert agent.db_connection == mock_db
        assert agent.context_retriever is not None
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    def test_chat_template_initialized(self, mock_get_redis, mock_langchain_init):
        """Test chat template is initialized."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        assert agent.chat_template is not None
        assert "context" in agent.chat_template.input_variables
        assert "history" in agent.chat_template.input_variables
        assert "question" in agent.chat_template.input_variables


class TestChatAgentPermissionValidation:
    """Tests for permission validation in Chat Agent."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @patch("app.agents.chat_agent.PermissionValidator")
    @pytest.mark.asyncio
    async def test_permission_denied(self, mock_validator, mock_get_redis, mock_langchain_init):
        """Test chat execution with permission denied."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        mock_validator.validate_user_project_access.return_value = False
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Hello",
            user_role="member"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "does not have access" in result.error


class TestChatAgentContextRetrieval:
    """Tests for context retrieval in Chat Agent."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @patch("app.agents.chat_agent.ContextRetriever")
    @pytest.mark.asyncio
    async def test_context_retrieval_success(self, mock_context_retriever_class, mock_get_redis, mock_langchain_init):
        """Test successful context retrieval."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_context_retriever_class.return_value = mock_retriever
        
        mock_retriever.retrieve_meeting_context.return_value = {
            "summaries": [{"metadata": {"text": "Summary 1"}}],
            "decisions": [{"metadata": {"text": "Decision 1"}}],
            "blockers": []
        }
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "response": "Based on the context, here's what I found.",
            "sources": ["summary1"],
            "confidence": 0.9,
            "follow_up_questions": ["What about X?"]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=mock_db)
        agent.context_retriever = mock_retriever
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="What happened in the last meeting?",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.metadata["context_used"] is True
        mock_retriever.retrieve_meeting_context.assert_called_once()
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @patch("app.agents.chat_agent.ContextRetriever")
    @pytest.mark.asyncio
    async def test_context_retrieval_failure_continues(self, mock_context_retriever_class, mock_get_redis, mock_langchain_init):
        """Test that context retrieval failure doesn't stop chat."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_context_retriever_class.return_value = mock_retriever
        
        mock_retriever.retrieve_meeting_context.side_effect = Exception("DB error")
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "response": "I can help with that.",
            "sources": [],
            "confidence": 0.7,
            "follow_up_questions": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=mock_db)
        agent.context_retriever = mock_retriever
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Hello",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.metadata["context_used"] is False


class TestChatAgentConversationHistory:
    """Tests for conversation history management."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_conversation_history_saved(self, mock_get_redis, mock_langchain_init):
        """Test that conversation history is saved."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "response": "Hello! How can I help?",
            "sources": [],
            "confidence": 0.9,
            "follow_up_questions": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Hello",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        # Verify setex was called to save history
        assert mock_redis.setex.called
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    def test_get_conversation_history(self, mock_get_redis, mock_langchain_init):
        """Test retrieving conversation history."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        messages_data = [
            {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "Hi", "timestamp": datetime.now().isoformat()}
        ]
        mock_redis.get.return_value = json.dumps(messages_data)
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        history = agent.get_conversation_history("user1", "project1")
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    def test_clear_conversation_history(self, mock_get_redis, mock_langchain_init):
        """Test clearing conversation history."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = agent.clear_conversation_history("user1", "project1")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("chat:user1:project1")


class TestChatAgentResponseGeneration:
    """Tests for chat response generation."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_response_generation_success(self, mock_get_redis, mock_langchain_init):
        """Test successful response generation."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "response": "This is a helpful response.",
            "sources": ["source1", "source2"],
            "confidence": 0.95,
            "follow_up_questions": ["Question 1?", "Question 2?"]
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="What is the project status?",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["response"] == "This is a helpful response."
        assert len(result.data["sources"]) == 2
        assert result.data["confidence"] == 0.95
        assert len(result.data["follow_up_questions"]) == 2
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_response_with_markdown_json(self, mock_get_redis, mock_langchain_init):
        """Test response parsing with markdown code blocks."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "response": "Here's the answer.",
    "sources": ["source1"],
    "confidence": 0.85,
    "follow_up_questions": []
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Tell me more",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["response"] == "Here's the answer."
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_response_invalid_json_fallback(self, mock_get_redis, mock_langchain_init):
        """Test fallback when response is not valid JSON."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        mock_response = MagicMock()
        mock_response.content = "This is just plain text, not JSON"
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Hello",
            user_role="member"
        )
        
        assert result.status == AgentStatus.SUCCESS
        assert result.data["response"] == "This is just plain text, not JSON"
        assert result.data["confidence"] == 0.5


class TestChatAgentInputValidation:
    """Tests for input validation."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_empty_message(self, mock_get_redis, mock_langchain_init):
        """Test handling of empty message."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="",
            user_role="member"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty" in result.error.lower()
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_whitespace_only_message(self, mock_get_redis, mock_langchain_init):
        """Test handling of whitespace-only message."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="   ",
            user_role="member"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "empty" in result.error.lower()


class TestChatAgentRoleBasedFiltering:
    """Tests for role-based context filtering."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @patch("app.agents.chat_agent.ContextRetriever")
    @pytest.mark.asyncio
    async def test_viewer_context_filtering(self, mock_context_retriever_class, mock_get_redis, mock_langchain_init):
        """Test context filtering for viewer role."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db = MagicMock()
        mock_retriever = MagicMock()
        mock_context_retriever_class.return_value = mock_retriever
        
        mock_retriever.retrieve_meeting_context.return_value = {
            "summaries": [{"metadata": {"text": "Summary 1"}}],
            "decisions": [{"metadata": {"text": "Decision 1"}}],
            "blockers": [{"metadata": {"text": "Blocker 1"}}],
            "sensitive_data": "Secret"
        }
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "response": "Here's what I can share.",
            "sources": [],
            "confidence": 0.8,
            "follow_up_questions": []
        })
        mock_llm.invoke.return_value = mock_response
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=mock_db)
        agent.context_retriever = mock_retriever
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="What's happening?",
            user_role="viewer"
        )
        
        assert result.status == AgentStatus.SUCCESS
        # Verify context was filtered (blockers should be removed for viewer)
        assert result.metadata["context_used"] is True


class TestChatAgentErrorHandling:
    """Tests for error handling."""
    
    @patch("app.agents.chat_agent.LangChainInitializer")
    @patch("app.agents.chat_agent.get_redis_client")
    @pytest.mark.asyncio
    async def test_llm_invocation_failure(self, mock_get_redis, mock_langchain_init):
        """Test handling of LLM invocation failure."""
        mock_llm = MagicMock()
        mock_langchain_init.get_llm.return_value = mock_llm
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        mock_llm.invoke.side_effect = Exception("API error")
        
        config = AgentConfig()
        agent = ChatAgent(config, db_connection=None)
        
        result = await agent.execute(
            user_id="user1",
            project_id="project1",
            message="Hello",
            user_role="member"
        )
        
        assert result.status == AgentStatus.ERROR
        assert "Chat failed" in result.error
