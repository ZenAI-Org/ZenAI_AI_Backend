"""
Task Extraction Agent - Extracts actionable tasks from meeting transcripts.
Uses LangChain with structured output validation to identify tasks, assignees, and priorities.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from app.agents.base_agent import BaseAgent, AgentConfig, AgentStatus, AgentResult
from app.agents.langchain_config import LangChainInitializer, PromptTemplateManager, MessageBuilder
from app.core.context_retriever import ContextRetriever
from app.core.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)


class ExtractedTask(BaseModel):
    """Schema for a single extracted task."""
    
    title: str = Field(
        ...,
        description="Task title (concise, action-oriented)",
        min_length=5,
        max_length=200
    )
    description: str = Field(
        ...,
        description="Detailed task description",
        min_length=10,
        max_length=1000
    )
    priority: str = Field(
        default="medium",
        description="Task priority level",
        pattern="^(low|medium|high)$"
    )
    assignee_name: Optional[str] = Field(
        default=None,
        description="Name of person assigned to task"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Due date in YYYY-MM-DD format"
    )
    blocked: bool = Field(
        default=False,
        description="Whether task is blocked by dependencies"
    )
    blocker_description: Optional[str] = Field(
        default=None,
        description="Description of blocker if task is blocked"
    )
    assignment_type: str = Field(
        default="unassigned",
        description="Type of assignment: explicit, inferred, or unassigned",
        pattern="^(explicit|inferred|unassigned)$"
    )
    confidence_score: float = Field(
        default=1.0,
        description="Confidence score of the assignment (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    assignment_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning key for why this person was assigned (especially for inferred)"
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class TaskExtractionOutput(BaseModel):
    """Schema for task extraction output."""
    
    tasks: List[ExtractedTask] = Field(
        default_factory=list,
        description="List of extracted tasks",
        max_items=50
    )
    blockers: List[str] = Field(
        default_factory=list,
        description="List of identified blockers or risks",
        max_items=20
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief summary of task extraction results"
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class TaskExtractionAgent(BaseAgent):
    """
    Agent for extracting actionable tasks from meeting transcripts.
    
    Responsibilities:
    - Parse transcript using LangChain extraction chain
    - Validate extracted tasks against Pydantic schema
    - Match assignees to OrgMembers
    - Create Task records in database
    - Handle validation failures gracefully
    """
    
    def __init__(self, config: AgentConfig, db_connection=None):
        """
        Initialize Task Extraction Agent.
        
        Args:
            config: Agent configuration
            db_connection: PostgreSQL connection for context retrieval and task storage
        """
        super().__init__(config)
        self.db_connection = db_connection
        self.llm = LangChainInitializer.get_llm()
        self.context_retriever = ContextRetriever(db_connection) if db_connection else None
        self.embedding_store = EmbeddingStore(db_connection) if db_connection else None
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize LangChain prompt templates for task extraction."""
        
        # Main task extraction prompt template
        self.extraction_template = PromptTemplateManager.create_template(
            template="""You are an expert task extraction specialist. Your task is to identify and extract actionable tasks from a meeting transcript.

Meeting Transcript:
{transcript}

{context}

Please extract all actionable tasks and provide them in the following JSON format:
{{
    "tasks": [
        {{
            "title": "Concise task title (action-oriented)",
            "description": "Detailed description of what needs to be done",
            "priority": "low|medium|high",
            "assignee_name": "Person name if mentioned, or null",
            "assignment_type": "explicit|inferred|unassigned",
            "confidence_score": 0.0 to 1.0,
            "assignment_reasoning": "Reasoning for assignment, especially if inferred",
            "due_date": "YYYY-MM-DD format if mentioned, or null",
            "blocked": false,
            "blocker_description": null
        }}
    ],
    "blockers": ["Blocker 1", "Blocker 2"],
    "summary": "Brief summary of extracted tasks"
}}

Guidelines:
- Extract ONLY actionable tasks (not discussions or decisions without action)
- Identify specific people mentioned as assignees.
- **Assignment Logic**:
    - **Explicit**: User clearly says "I will do this" or "Bob, you handle this".
    - **Inferred**: Context implies responsibility (e.g., "Bob is the frontend lead" -> Assign frontend task to Bob). Provide reasoning.
    - **Unassigned**: No specific person found.
- **Confidence**: Provide a score (0.0-1.0) for the assignment. Low confidence (<0.7) should be flagged.
- Infer priority from context (urgent/critical = high, routine = low, default = medium)
- Flag tasks that are blocked by dependencies or external factors
- Include due dates if mentioned or implied
- Ensure each task has a clear, specific title and description
- Group related subtasks under a single parent task when appropriate
- Return empty arrays if no tasks or blockers are identified""",
            input_variables=["transcript", "context"],
            description="Template for task extraction from transcripts"
        )
        
        self._log_execution("Prompt templates initialized")
    
    async def execute(
        self,
        meeting_id: str,
        transcript: str,
        project_id: Optional[str] = None,
        org_members: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute task extraction workflow.
        
        Args:
            meeting_id: Meeting identifier
            transcript: Meeting transcript text
            project_id: Project identifier for context retrieval
            org_members: List of organization members for assignee matching
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with extracted tasks
        """
        try:
            self._log_execution(
                "Starting task extraction",
                {"meeting_id": meeting_id, "transcript_length": len(transcript)}
            )
            
            # Validate inputs
            if not transcript or len(transcript.strip()) < 10:
                return self._create_error_result(
                    "Transcript is empty or too short for task extraction"
                )
            
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
            prompt_text = self.extraction_template.format(
                transcript=transcript,
                context=f"\nPrevious Context:\n{context_str}" if context_str else ""
            )
            
            # Create messages for LLM
            messages = MessageBuilder.create_messages(
                system_prompt="You are an expert task extraction specialist. Provide task extraction results in valid JSON format.",
                user_message=prompt_text
            )
            
            # Call LLM
            self._log_execution("Calling LLM for task extraction")
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse and validate response
            extraction_data = self._parse_and_validate_response(response_text)
            
            # Create extraction output
            extraction_output = TaskExtractionOutput(**extraction_data)
            
            # Match assignees to org members if provided
            if org_members:
                extraction_output = self._match_assignees(extraction_output, org_members)
            
            # Store task embeddings if database connection available
            task_embeddings = []
            if project_id and self.embedding_store:
                try:
                    for task in extraction_output.tasks:
                        embedding_id = self.embedding_store.store_embedding(
                            project_id=project_id,
                            content_type="task",
                            content_id=f"{meeting_id}_{task.title}",
                            text=f"{task.title}: {task.description}",
                            metadata={
                                "meeting_id": meeting_id,
                                "priority": task.priority,
                                "assignee": task.assignee_name,
                                "blocked": task.blocked
                            }
                        )
                        task_embeddings.append(embedding_id)
                    self._log_execution("Stored task embeddings", {"count": len(task_embeddings)})
                except Exception as e:
                    self._log_error("Failed to store task embeddings", {"error": str(e)})
            
            # Prepare result data
            result_data = {
                "meeting_id": meeting_id,
                "tasks": [task.dict() for task in extraction_output.tasks],
                "blockers": extraction_output.blockers,
                "summary": extraction_output.summary,
                "task_count": len(extraction_output.tasks),
                "blocker_count": len(extraction_output.blockers),
                "task_embeddings": task_embeddings
            }
            
            # Prepare metadata
            metadata = {
                "agent": "TaskExtractionAgent",
                "model": self.config.model_name,
                "transcript_length": len(transcript),
                "tasks_extracted": len(extraction_output.tasks),
                "blockers_identified": len(extraction_output.blockers),
                "context_used": bool(context_str),
                "embeddings_stored": len(task_embeddings)
            }
            
            self._log_execution("Task extraction completed successfully", metadata)
            
            return self._create_success_result(result_data, metadata)
        
        except ValidationError as e:
            error_msg = f"Task extraction validation failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
        
        except Exception as e:
            error_msg = f"Task extraction failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
    
    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed and validated extraction data
            
        Raises:
            ValueError: If response cannot be parsed or validated
        """
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
            if "tasks" not in parsed:
                parsed["tasks"] = []
            if "blockers" not in parsed:
                parsed["blockers"] = []
            
            # Ensure tasks is a list
            if not isinstance(parsed["tasks"], list):
                parsed["tasks"] = []
            
            # Ensure blockers is a list
            if not isinstance(parsed["blockers"], list):
                parsed["blockers"] = []
            
            # Ensure blockers contain strings
            parsed["blockers"] = [str(item) for item in parsed.get("blockers", [])]
            
            # Validate and clean each task
            validated_tasks = []
            for task_data in parsed.get("tasks", []):
                try:
                    # Ensure required fields exist
                    if "title" not in task_data or not task_data["title"]:
                        continue
                    if "description" not in task_data or not task_data["description"]:
                        task_data["description"] = task_data.get("title", "")
                    
                    # Ensure priority is valid
                    if "priority" not in task_data or task_data["priority"] not in ["low", "medium", "high"]:
                        task_data["priority"] = "medium"
                    
                    # Ensure boolean fields
                    if "blocked" not in task_data:
                        task_data["blocked"] = False
                    
                    # Validate new ownership fields
                    if "assignment_type" not in task_data or task_data["assignment_type"] not in ["explicit", "inferred", "unassigned"]:
                         task_data["assignment_type"] = "unassigned"
                    
                    if "confidence_score" not in task_data:
                        task_data["confidence_score"] = 1.0
                    
                    # Create ExtractedTask to validate
                    task = ExtractedTask(**task_data)
                    validated_tasks.append(task.dict())
                except ValidationError as e:
                    self._log_error(f"Task validation failed, skipping: {str(e)}")
                    continue
            
            parsed["tasks"] = validated_tasks
            
            return parsed
        
        except json.JSONDecodeError as e:
            self._log_error("Failed to parse JSON response", {"error": str(e)})
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            self._log_error("Failed to parse response", {"error": str(e)})
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def _match_assignees(
        self,
        extraction_output: TaskExtractionOutput,
        org_members: List[Dict[str, Any]]
    ) -> TaskExtractionOutput:
        """
        Match extracted assignee names to organization members.
        
        Args:
            extraction_output: Extraction output with tasks
            org_members: List of organization members with name and id
            
        Returns:
            Updated extraction output with matched assignee IDs
        """
        try:
            # Create a mapping of member names to IDs
            member_map = {}
            for member in org_members:
                name = member.get("name", "").lower()
                if name:
                    member_map[name] = member.get("id")
            
            # Update tasks with matched assignee IDs
            for task in extraction_output.tasks:
                if task.assignee_name:
                    assignee_lower = task.assignee_name.lower()
                    
                    # Try exact match first
                    if assignee_lower in member_map:
                        task.assignee_id = member_map[assignee_lower]
                    else:
                        # Try partial match (first name or last name)
                        for member_name, member_id in member_map.items():
                            if assignee_lower in member_name or member_name in assignee_lower:
                                task.assignee_id = member_id
                                break
            
            self._log_execution("Matched assignees to org members")
            return extraction_output
        
        except Exception as e:
            self._log_error("Failed to match assignees", {"error": str(e)})
            return extraction_output
    
    def validate_extraction(self, extraction_data: Dict[str, Any]) -> bool:
        """
        Validate extraction data against schema.
        
        Args:
            extraction_data: Extraction data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            TaskExtractionOutput(**extraction_data)
            return True
        except ValidationError as e:
            self._log_error("Extraction validation failed", {"error": str(e)})
            return False
    
    def get_extraction_quality_score(self, extraction_output: TaskExtractionOutput) -> float:
        """
        Calculate quality score for extraction (0-1).
        
        Args:
            extraction_output: Validated extraction output
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Check for tasks extracted
        if len(extraction_output.tasks) > 0:
            score += 0.3
        
        # Check for task details
        for task in extraction_output.tasks:
            if task.description and len(task.description) > 20:
                score += 0.1
                break
        
        # Check for priority assignments
        if any(task.priority in ["high", "low"] for task in extraction_output.tasks):
            score += 0.2
        
        # Check for assignee assignments
        if any(task.assignee_name for task in extraction_output.tasks):
            score += 0.2
        
        # Check for blocker identification
        if len(extraction_output.blockers) > 0:
            score += 0.2
        
        return min(score, 1.0)



# Job function for background processing
async def extract_tasks_job(
    meeting_id: str,
    transcript: str,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job function for extracting tasks from meetings.
    
    Args:
        meeting_id: Meeting ID
        transcript: Meeting transcript
        project_id: Optional project ID
        
    Returns:
        Dictionary with task extraction result
    """
    try:
        config = AgentConfig(
            name="TaskExtractionAgent",
            model="gpt-4",
            temperature=0.2,
        )
        agent = TaskExtractionAgent(config)
        result = await agent.execute(
            meeting_id=meeting_id,
            transcript=transcript,
            project_id=project_id,
        )
        
        if result.status == AgentStatus.SUCCESS:
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "tasks": result.data.get("tasks", []),
                "blockers": result.data.get("blockers", []),
            }
        else:
            return {
                "status": "error",
                "meeting_id": meeting_id,
                "error": result.error,
            }
    
    except Exception as e:
        logger.error(f"Task extraction job failed: {str(e)}")
        return {
            "status": "error",
            "meeting_id": meeting_id,
            "error": str(e),
        }
