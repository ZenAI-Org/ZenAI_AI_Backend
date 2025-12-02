"""
Summarization Agent for intelligent meeting summarization.
Uses LangChain with OpenAI to generate concise, actionable meeting summaries.
"""

import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage

from app.agents.base_agent import BaseAgent, AgentConfig, AgentStatus, AgentResult
from app.agents.langchain_config import LangChainInitializer, PromptTemplateManager, MessageBuilder
from app.core.context_retriever import ContextRetriever
from app.core.embeddings import EmbeddingStore
from app.core.performance_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


class SummaryOutput(BaseModel):
    """Zod-like schema for summary output validation."""
    
    summary: str = Field(
        ...,
        description="Concise meeting summary (150-300 words)",
        min_length=50,
        max_length=500
    )
    key_decisions: List[str] = Field(
        default_factory=list,
        description="List of key decisions made during the meeting",
        max_items=10
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="List of action items identified in the meeting",
        max_items=10
    )
    participants: List[str] = Field(
        default_factory=list,
        description="List of meeting participants mentioned",
        max_items=20
    )
    next_steps: Optional[str] = Field(
        default=None,
        description="Recommended next steps or follow-up actions"
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class SummarizationAgent(BaseAgent):
    """
    Agent for generating intelligent meeting summaries.
    
    Responsibilities:
    - Retrieve transcript and project context
    - Use LangChain summarization chain with custom prompts
    - Embed summary in pgvector for future retrieval
    - Validate summary quality with Zod schema
    - Store summary in AgentRun output
    """
    
    def __init__(self, config: AgentConfig, db_connection=None):
        """
        Initialize Summarization Agent.
        
        Args:
            config: Agent configuration
            db_connection: PostgreSQL connection for context retrieval and embedding storage
        """
        super().__init__(config)
        self.db_connection = db_connection
        self.llm = LangChainInitializer.get_llm()
        self.context_retriever = ContextRetriever(db_connection) if db_connection else None
        self.embedding_store = EmbeddingStore(db_connection) if db_connection else None
        self.cache_manager = get_cache_manager()
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize LangChain prompt templates for summarization."""
        
        # Main summarization prompt template
        self.summarization_template = PromptTemplateManager.create_template(
            template="""You are an expert meeting summarizer. Your task is to create a concise, actionable summary of a meeting transcript.

Meeting Transcript:
{transcript}

{context}

Please provide a comprehensive summary in the following JSON format:
{{
    "summary": "A concise 150-300 word summary of the meeting highlighting key points and outcomes",
    "key_decisions": ["Decision 1", "Decision 2", ...],
    "action_items": ["Action 1: Owner", "Action 2: Owner", ...],
    "participants": ["Participant 1", "Participant 2", ...],
    "next_steps": "Recommended next steps or follow-up actions"
}}

Ensure the summary is:
- Concise and focused on key outcomes
- Includes specific decisions and action items
- Identifies who is responsible for each action
- Highlights any blockers or risks mentioned
- References previous context when relevant""",
            input_variables=["transcript", "context"],
            description="Template for meeting summarization"
        )
        
        self._log_execution("Prompt templates initialized")
    
    async def execute(
        self,
        meeting_id: str,
        transcript: str,
        project_id: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute summarization workflow.
        
        Args:
            meeting_id: Meeting identifier
            transcript: Meeting transcript text
            project_id: Project identifier for context retrieval
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with summary data
        """
        try:
            self._log_execution(
                "Starting summarization",
                {"meeting_id": meeting_id, "transcript_length": len(transcript)}
            )
            
            # Validate inputs
            if not transcript or len(transcript.strip()) < 10:
                return self._create_error_result(
                    "Transcript is empty or too short for summarization"
                )
            
            # Check cache
            cache_key = self.cache_manager._make_key("summary", meeting_id)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self._log_execution("Returning cached summary", {"meeting_id": meeting_id})
                return self._create_success_result(cached_result["data"], cached_result["metadata"])
            
            # Retrieve project context if available
            context_str = ""
            if project_id and self.context_retriever:
                try:
                    context_str = self.context_retriever.build_prompt_context(
                        project_id=project_id,
                        query=transcript[:500],  # Use first 500 chars as query
                        max_tokens=1000
                    )
                    self._log_execution("Retrieved project context", {"project_id": project_id})
                except Exception as e:
                    self._log_error("Failed to retrieve context", {"error": str(e)})
                    context_str = ""
            
            # Format prompt with transcript and context
            prompt_text = self.summarization_template.format(
                transcript=transcript,
                context=f"\nPrevious Context:\n{context_str}" if context_str else ""
            )
            
            # Create messages for LLM
            messages = MessageBuilder.create_messages(
                system_prompt="You are an expert meeting summarizer. Provide summaries in valid JSON format.",
                user_message=prompt_text
            )
            
            # Call LLM
            self._log_execution("Calling LLM for summarization")
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse and validate response
            summary_data = self._parse_and_validate_response(response_text)
            
            # Create summary output
            summary_output = SummaryOutput(**summary_data)
            
            # Store summary embedding if database connection available
            embedding_id = None
            if project_id and self.embedding_store:
                try:
                    embedding_id = self.embedding_store.store_embedding(
                        project_id=project_id,
                        content_type="summary",
                        content_id=meeting_id,
                        text=summary_output.summary,
                        metadata={
                            "meeting_id": meeting_id,
                            "key_decisions": summary_output.key_decisions,
                            "action_items": summary_output.action_items,
                            "participants": summary_output.participants
                        }
                    )
                    self._log_execution("Stored summary embedding", {"embedding_id": embedding_id})
                except Exception as e:
                    self._log_error("Failed to store embedding", {"error": str(e)})
            
            # Prepare result data
            result_data = {
                "meeting_id": meeting_id,
                "summary": summary_output.summary,
                "key_decisions": summary_output.key_decisions,
                "action_items": summary_output.action_items,
                "participants": summary_output.participants,
                "next_steps": summary_output.next_steps,
                "embedding_id": embedding_id
            }
            
            # Prepare metadata
            metadata = {
                "agent": "SummarizationAgent",
                "model": self.config.model_name,
                "transcript_length": len(transcript),
                "summary_length": len(summary_output.summary),
                "context_used": bool(context_str),
                "embedding_stored": embedding_id is not None
            }
            
            self._log_execution("Summarization completed successfully", metadata)
            
            # Cache result
            self.cache_manager.set(cache_key, {"data": result_data, "metadata": metadata})
            
            return self._create_success_result(result_data, metadata)
        
        except ValidationError as e:
            error_msg = f"Summary validation failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
        
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
    
    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed and validated summary data
            
        Raises:
            ValueError: If response cannot be parsed or validated
        """
        import json
        
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps response in markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            if "summary" not in parsed:
                raise ValueError("Missing 'summary' field in response")
            
            # Ensure lists are properly formatted
            if "key_decisions" not in parsed:
                parsed["key_decisions"] = []
            if "action_items" not in parsed:
                parsed["action_items"] = []
            if "participants" not in parsed:
                parsed["participants"] = []
            
            # Ensure lists contain strings
            parsed["key_decisions"] = [str(item) for item in parsed.get("key_decisions", [])]
            parsed["action_items"] = [str(item) for item in parsed.get("action_items", [])]
            parsed["participants"] = [str(item) for item in parsed.get("participants", [])]
            
            return parsed
        
        except json.JSONDecodeError as e:
            self._log_error("Failed to parse JSON response", {"error": str(e)})
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            self._log_error("Failed to parse response", {"error": str(e)})
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def validate_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Validate summary data against schema.
        
        Args:
            summary_data: Summary data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            SummaryOutput(**summary_data)
            return True
        except ValidationError as e:
            self._log_error("Summary validation failed", {"error": str(e)})
            return False
    
    def get_summary_quality_score(self, summary_output: SummaryOutput) -> float:
        """
        Calculate quality score for summary (0-1).
        
        Args:
            summary_output: Validated summary output
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Check summary length (150-300 words optimal)
        word_count = len(summary_output.summary.split())
        if 150 <= word_count <= 300:
            score += 0.3
        elif 100 <= word_count <= 400:
            score += 0.2
        
        # Check for key decisions
        if len(summary_output.key_decisions) > 0:
            score += 0.2
        
        # Check for action items
        if len(summary_output.action_items) > 0:
            score += 0.2
        
        # Check for participants
        if len(summary_output.participants) > 0:
            score += 0.15
        
        # Check for next steps
        if summary_output.next_steps:
            score += 0.15
        
        return min(score, 1.0)



# Job function for background processing
async def summarize_meeting_job(
    meeting_id: str,
    transcript: str,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job function for summarizing meetings.
    
    Args:
        meeting_id: Meeting ID
        transcript: Meeting transcript
        project_id: Optional project ID for context
        
    Returns:
        Dictionary with summarization result
    """
    try:
        config = AgentConfig(
            name="SummarizationAgent",
            model="gpt-4",
            temperature=0.3,
        )
        agent = SummarizationAgent(config)
        result = await agent.execute(
            meeting_id=meeting_id,
            transcript=transcript,
            project_id=project_id,
        )
        
        if result.status == AgentStatus.SUCCESS:
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "summary": result.data.get("summary", ""),
                "key_decisions": result.data.get("key_decisions", []),
                "action_items": result.data.get("action_items", []),
                "participants": result.data.get("participants", []),
                "next_steps": result.data.get("next_steps"),
            }
        else:
            return {
                "status": "error",
                "meeting_id": meeting_id,
                "error": result.error,
            }
    
    except Exception as e:
        logger.error(f"Summarization job failed: {str(e)}")
        return {
            "status": "error",
            "meeting_id": meeting_id,
            "error": str(e),
        }
