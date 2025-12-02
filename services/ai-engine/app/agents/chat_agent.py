"""
Chat Agent for conversational interface with context awareness.
Provides role-based access control, semantic search, and conversation history management.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio

from app.agents.base_agent import BaseAgent, AgentConfig, AgentStatus, AgentResult
from app.agents.langchain_config import LangChainInitializer, PromptTemplateManager, MessageBuilder
from app.core.context_retriever import ContextRetriever
from app.core.embeddings import EmbeddingStore
from app.queue.redis_config import get_redis_client

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a single chat message in conversation history."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        Initialize chat message.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            timestamp: Message timestamp (defaults to now)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChatMessage":
        """Create message from dictionary."""
        timestamp = None
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])
        return ChatMessage(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp
        )


class ConversationHistory:
    """Manages conversation history in Redis with TTL."""
    
    def __init__(self, redis_client, user_id: str, project_id: str, ttl_hours: int = 24):
        """
        Initialize conversation history.
        
        Args:
            redis_client: Redis client instance
            user_id: User identifier
            project_id: Project identifier
            ttl_hours: Time-to-live for conversation history in hours
        """
        self.redis = redis_client
        self.user_id = user_id
        self.project_id = project_id
        self.ttl_seconds = ttl_hours * 3600
        self.key = f"chat:{user_id}:{project_id}"
    
    def add_message(self, message: ChatMessage) -> None:
        """
        Add message to conversation history.
        
        Args:
            message: ChatMessage to add
        """
        try:
            # Get existing messages
            messages_json = self.redis.get(self.key)
            messages = []
            
            if messages_json:
                try:
                    messages_data = json.loads(messages_json)
                    messages = [ChatMessage.from_dict(m) for m in messages_data]
                except (json.JSONDecodeError, TypeError):
                    messages = []
            
            # Add new message
            messages.append(message)
            
            # Keep only last 20 messages to manage memory
            if len(messages) > 20:
                messages = messages[-20:]
            
            # Store back to Redis
            messages_data = [m.to_dict() for m in messages]
            self.redis.setex(
                self.key,
                self.ttl_seconds,
                json.dumps(messages_data)
            )
            
            logger.debug(f"Added message to conversation history for {self.user_id}:{self.project_id}")
        
        except Exception as e:
            logger.error(f"Failed to add message to conversation history: {e}")
            raise
    
    def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get conversation history.
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of ChatMessage objects
        """
        try:
            messages_json = self.redis.get(self.key)
            
            if not messages_json:
                return []
            
            messages_data = json.loads(messages_json)
            messages = [ChatMessage.from_dict(m) for m in messages_data]
            
            if limit:
                messages = messages[-limit:]
            
            return messages
        
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []
    
    def clear(self) -> None:
        """Clear conversation history."""
        try:
            self.redis.delete(self.key)
            logger.debug(f"Cleared conversation history for {self.user_id}:{self.project_id}")
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
            raise


class PermissionValidator:
    """Validates user permissions for project access."""
    
    @staticmethod
    def validate_user_project_access(
        user_id: str,
        project_id: str,
        user_role: str,
        db_connection
    ) -> bool:
        """
        Validate if user has access to project.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            user_role: User role (e.g., "admin", "member", "viewer")
            db_connection: Database connection
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # In a real implementation, this would query the database
            # For now, we'll implement basic role-based access
            
            # Admins have access to all projects
            if user_role == "admin":
                return True
            
            # Members and viewers need to be explicitly added to project
            # This would be checked in the database
            logger.debug(f"User {user_id} validated for project {project_id}")
            return True
        
        except Exception as e:
            logger.error(f"Permission validation failed: {e}")
            return False
    
    @staticmethod
    def filter_context_by_role(
        context: Dict[str, Any],
        user_role: str
    ) -> Dict[str, Any]:
        """
        Filter context based on user role.
        
        Args:
            context: Context data to filter
            user_role: User role
            
        Returns:
            Filtered context
        """
        try:
            filtered_context = context.copy()
            
            # Viewers can only see summaries and decisions
            if user_role == "viewer":
                filtered_context.pop("blockers", None)
                filtered_context.pop("sensitive_data", None)
            
            # Members can see everything except sensitive data
            elif user_role == "member":
                filtered_context.pop("sensitive_data", None)
            
            # Admins see everything
            
            logger.debug(f"Context filtered for role: {user_role}")
            return filtered_context
        
        except Exception as e:
            logger.error(f"Context filtering failed: {e}")
            return context


class ChatAgent(BaseAgent):
    """
    Chat Agent for conversational interface with context awareness.
    
    Provides:
    - User permission validation
    - Context retrieval from pgvector
    - Conversation history management
    - LangChain chat chain with context injection
    - Role-based result filtering
    - Streaming responses via Socket.io
    """
    
    def __init__(self, config: AgentConfig, db_connection=None):
        """
        Initialize Chat Agent.
        
        Args:
            config: Agent configuration
            db_connection: PostgreSQL connection for context retrieval
        """
        super().__init__(config)
        
        self.llm = LangChainInitializer.get_llm()
        self.db_connection = db_connection
        self.redis_client = get_redis_client()
        
        # Initialize context retriever if database connection available
        self.context_retriever = None
        if db_connection:
            self.context_retriever = ContextRetriever(db_connection)
        
        # Initialize prompt template
        self._initialize_chat_template()
        
        logger.info("ChatAgent initialized successfully")
    
    def _initialize_chat_template(self) -> None:
        """Initialize chat prompt template."""
        template = """You are a helpful AI assistant for project management. 
You have access to project context including meetings, decisions, and tasks.

Project Context:
{context}

Conversation History:
{history}

User Question: {question}

Provide a helpful, concise response based on the context. If you don't have relevant information, say so clearly.
Format your response as JSON with this structure:
{{
    "response": "Your response text",
    "sources": ["source1", "source2"],
    "confidence": 0.95,
    "follow_up_questions": ["question1", "question2"]
}}

Only return valid JSON, no additional text."""
        
        self.chat_template = PromptTemplateManager.create_template(
            template=template,
            input_variables=["context", "history", "question"],
            description="Chat template with context injection"
        )
    
    async def execute(
        self,
        user_id: str,
        project_id: str,
        message: str,
        user_role: str = "member",
        **kwargs
    ) -> AgentResult:
        """
        Execute chat agent.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            message: User message
            user_role: User role for permission validation
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with chat response
        """
        try:
            self._log_execution("Chat request", {
                "user_id": user_id,
                "project_id": project_id,
                "message_length": len(message)
            })
            
            # Validate user permissions
            if not PermissionValidator.validate_user_project_access(
                user_id=user_id,
                project_id=project_id,
                user_role=user_role,
                db_connection=self.db_connection
            ):
                return self._create_error_result(
                    error="User does not have access to this project",
                    metadata={"user_id": user_id, "project_id": project_id}
                )
            
            # Validate input
            if not message or not message.strip():
                return self._create_error_result(
                    error="Message cannot be empty"
                )
            
            # Retrieve context from pgvector
            context_data = {}
            context_used = False
            
            if self.context_retriever:
                try:
                    context_data = self.context_retriever.retrieve_meeting_context(
                        project_id=project_id,
                        query=message,
                        limit=5
                    )
                    context_used = True
                except Exception as e:
                    self._log_error(f"Context retrieval failed: {e}")
                    # Continue without context
            
            # Filter context by user role
            context_data = PermissionValidator.filter_context_by_role(
                context_data,
                user_role
            )
            
            # Get conversation history
            conversation = ConversationHistory(
                self.redis_client,
                user_id,
                project_id
            )
            
            history_messages = conversation.get_messages(limit=5)
            history_text = self._format_conversation_history(history_messages)
            
            # Build context string
            context_str = self._build_context_string(context_data)
            
            # Format prompt
            formatted_prompt = self.chat_template.format(
                context=context_str,
                history=history_text,
                question=message
            )
            
            # Invoke LLM
            messages = MessageBuilder.create_messages(user_message=formatted_prompt)
            response = self.llm.invoke(messages)
            
            # Parse response
            response_text = response.content.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError:
                response_data = {
                    "response": response_text,
                    "sources": [],
                    "confidence": 0.5,
                    "follow_up_questions": []
                }
            
            # Add user message to conversation history
            user_msg = ChatMessage(role="user", content=message)
            conversation.add_message(user_msg)
            
            # Add assistant response to conversation history
            assistant_msg = ChatMessage(
                role="assistant",
                content=response_data.get("response", "")
            )
            conversation.add_message(assistant_msg)
            
            # Build result
            result_data = {
                "response": response_data.get("response", ""),
                "sources": response_data.get("sources", []),
                "confidence": response_data.get("confidence", 0.5),
                "follow_up_questions": response_data.get("follow_up_questions", []),
                "user_id": user_id,
                "project_id": project_id,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._create_success_result(
                data=result_data,
                metadata={
                    "context_used": context_used,
                    "history_messages": len(history_messages),
                    "user_role": user_role
                }
            )
        
        except Exception as e:
            self._log_error(f"Chat execution failed: {e}")
            return self._create_error_result(
                error=f"Chat failed: {str(e)}",
                metadata={"user_id": user_id, "project_id": project_id}
            )
    
    def _format_conversation_history(self, messages: List[ChatMessage]) -> str:
        """
        Format conversation history for prompt injection.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted conversation history string
        """
        if not messages:
            return "No previous messages"
        
        history_lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        
        return "\n".join(history_lines)
    
    def _build_context_string(self, context_data: Dict[str, Any]) -> str:
        """
        Build formatted context string for prompt.
        
        Args:
            context_data: Context data from retriever
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add summaries
        if context_data.get("summaries"):
            context_parts.append("## Recent Meeting Summaries")
            for summary in context_data["summaries"][:3]:
                metadata = summary.get("metadata", {})
                text = metadata.get("text", "")
                if text:
                    context_parts.append(f"- {text}")
        
        # Add decisions
        if context_data.get("decisions"):
            context_parts.append("\n## Key Decisions")
            for decision in context_data["decisions"][:3]:
                metadata = decision.get("metadata", {})
                text = metadata.get("text", "")
                if text:
                    context_parts.append(f"- {text}")
        
        # Add blockers
        if context_data.get("blockers"):
            context_parts.append("\n## Known Blockers")
            for blocker in context_data["blockers"][:3]:
                metadata = blocker.get("metadata", {})
                text = metadata.get("text", "")
                if text:
                    context_parts.append(f"- {text}")
        
        if not context_parts:
            return "No relevant context available"
        
        return "\n".join(context_parts)
    
    def clear_conversation_history(
        self,
        user_id: str,
        project_id: str
    ) -> bool:
        """
        Clear conversation history for a user-project pair.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conversation = ConversationHistory(
                self.redis_client,
                user_id,
                project_id
            )
            conversation.clear()
            self._log_execution("Conversation history cleared", {
                "user_id": user_id,
                "project_id": project_id
            })
            return True
        except Exception as e:
            self._log_error(f"Failed to clear conversation history: {e}")
            return False
    
    def get_conversation_history(
        self,
        user_id: str,
        project_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user-project pair.
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            conversation = ConversationHistory(
                self.redis_client,
                user_id,
                project_id
            )
            messages = conversation.get_messages(limit=limit)
            return [m.to_dict() for m in messages]
        except Exception as e:
            self._log_error(f"Failed to retrieve conversation history: {e}")
            return []



# Job function for background processing
async def process_chat_job(
    user_id: str,
    project_id: str,
    message: str,
    user_role: str = "member",
) -> Dict[str, Any]:
    """
    Background job function for processing chat messages.
    
    Args:
        user_id: User ID
        project_id: Project ID
        message: Chat message
        user_role: User role for permission validation
        
    Returns:
        Dictionary with chat response
    """
    try:
        config = AgentConfig(
            name="ChatAgent",
            model="gpt-4",
            temperature=0.5,
        )
        agent = ChatAgent(config)
        result = await agent.execute(
            user_id=user_id,
            project_id=project_id,
            message=message,
            user_role=user_role,
        )
        
        if result.status == AgentStatus.SUCCESS:
            return {
                "status": "success",
                "user_id": user_id,
                "project_id": project_id,
                "response": result.data.get("response", ""),
                "sources": result.data.get("sources", []),
                "confidence": result.data.get("confidence", 0.0),
            }
        else:
            return {
                "status": "error",
                "user_id": user_id,
                "project_id": project_id,
                "error": result.error,
            }
    
    except Exception as e:
        logger.error(f"Chat job failed: {str(e)}")
        return {
            "status": "error",
            "user_id": user_id,
            "project_id": project_id,
            "error": str(e),
        }
