"""
Suggestions Agent for generating AI-powered dashboard suggestions.
Analyzes project data and generates actionable insights for dashboard cards.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.agents.base_agent import BaseAgent, AgentConfig, AgentStatus, AgentResult
from app.agents.langchain_config import LangChainInitializer, PromptTemplateManager, MessageBuilder
from app.core.context_retriever import ContextRetriever
from app.queue.redis_config import get_redis_client

logger = logging.getLogger(__name__)

# Redis cache TTL for suggestions (6 hours)
SUGGESTIONS_CACHE_TTL = 6 * 60 * 60


class Suggestion(BaseModel):
    """Individual suggestion model."""
    
    title: str = Field(
        ...,
        description="Suggestion title",
        min_length=5,
        max_length=200
    )
    description: str = Field(
        ...,
        description="Detailed suggestion description",
        min_length=10,
        max_length=500
    )
    action_url: Optional[str] = Field(
        default=None,
        description="URL for action if applicable"
    )
    priority: int = Field(
        default=1,
        description="Priority score (1-10, higher is more important)",
        ge=1,
        le=10
    )
    generated_at: Optional[str] = Field(
        default=None,
        description="Timestamp when suggestion was generated"
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class DashboardSuggestions(BaseModel):
    """Dashboard suggestions grouped by card type."""
    
    project_id: str = Field(..., description="Project ID")
    pending_tasks: List[Suggestion] = Field(
        default_factory=list,
        description="Suggestions for pending tasks card",
        max_items=3
    )
    project_insights: List[Suggestion] = Field(
        default_factory=list,
        description="Suggestions for project insights card",
        max_items=3
    )
    blockers: List[Suggestion] = Field(
        default_factory=list,
        description="Suggestions for blockers card",
        max_items=3
    )
    opportunities: List[Suggestion] = Field(
        default_factory=list,
        description="Suggestions for opportunities card",
        max_items=3
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when suggestions were generated"
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class SuggestionsAgent(BaseAgent):
    """
    Agent for generating AI-powered dashboard suggestions.
    
    Responsibilities:
    - Analyze project data for insights
    - Generate suggestions for multiple card types
    - Rank suggestions by relevance
    - Cache suggestions in Redis with TTL
    - Implement suggestion refresh on project changes
    """
    
    def __init__(self, config: AgentConfig, db_connection=None):
        """
        Initialize Suggestions Agent.
        
        Args:
            config: Agent configuration
            db_connection: PostgreSQL connection for context retrieval
        """
        super().__init__(config)
        self.db_connection = db_connection
        self.llm = LangChainInitializer.get_llm()
        self.context_retriever = ContextRetriever(db_connection) if db_connection else None
        self.redis_client = get_redis_client()
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize LangChain prompt templates for suggestion generation."""
        
        # Pending tasks suggestions template
        self.pending_tasks_template = PromptTemplateManager.create_template(
            template="""You are an AI project assistant. Analyze the project data and generate 2-3 actionable suggestions for pending tasks.

Project Data:
{project_data}

Generate suggestions in JSON format:
{{
    "suggestions": [
        {{
            "title": "Suggestion title",
            "description": "Detailed description of the suggestion",
            "action_url": "URL if applicable or null",
            "priority": 8
        }}
    ]
}}

Focus on:
- High-priority tasks that need attention
- Tasks that are blocking other work
- Tasks approaching deadlines
- Tasks with unclear ownership

Ensure suggestions are specific and actionable.""",
            input_variables=["project_data"],
            description="Template for pending tasks suggestions"
        )
        
        # Project insights suggestions template
        self.insights_template = PromptTemplateManager.create_template(
            template="""You are an AI project analyst. Analyze the project data and generate 2-3 insights about project health and progress.

Project Data:
{project_data}

Generate suggestions in JSON format:
{{
    "suggestions": [
        {{
            "title": "Insight title",
            "description": "Detailed insight description",
            "action_url": "URL if applicable or null",
            "priority": 7
        }}
    ]
}}

Focus on:
- Project velocity and completion trends
- Team capacity and workload distribution
- Risk factors and dependencies
- Opportunities for improvement

Ensure insights are data-driven and actionable.""",
            input_variables=["project_data"],
            description="Template for project insights suggestions"
        )
        
        # Blockers suggestions template
        self.blockers_template = PromptTemplateManager.create_template(
            template="""You are an AI risk analyst. Analyze the project data and identify 2-3 critical blockers or risks.

Project Data:
{project_data}

Generate suggestions in JSON format:
{{
    "suggestions": [
        {{
            "title": "Blocker title",
            "description": "Detailed description of the blocker and impact",
            "action_url": "URL if applicable or null",
            "priority": 9
        }}
    ]
}}

Focus on:
- Tasks blocked by dependencies
- Resource constraints
- Technical risks
- Communication gaps
- Overdue items

Ensure blockers are clearly identified with suggested resolutions.""",
            input_variables=["project_data"],
            description="Template for blockers suggestions"
        )
        
        # Opportunities suggestions template
        self.opportunities_template = PromptTemplateManager.create_template(
            template="""You are an AI opportunity scout. Analyze the project data and identify 2-3 opportunities for improvement or acceleration.

Project Data:
{project_data}

Generate suggestions in JSON format:
{{
    "suggestions": [
        {{
            "title": "Opportunity title",
            "description": "Detailed description of the opportunity and potential impact",
            "action_url": "URL if applicable or null",
            "priority": 6
        }}
    ]
}}

Focus on:
- Quick wins that can boost momentum
- Process improvements
- Team collaboration opportunities
- Knowledge sharing possibilities
- Automation opportunities

Ensure opportunities are realistic and achievable.""",
            input_variables=["project_data"],
            description="Template for opportunities suggestions"
        )
        
        self._log_execution("Prompt templates initialized")
    
    async def execute(
        self,
        project_id: str,
        force_refresh: bool = False,
        **kwargs
    ) -> AgentResult:
        """
        Execute suggestion generation workflow.
        
        Args:
            project_id: Project identifier
            force_refresh: Force refresh even if cached
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with suggestions data
        """
        try:
            self._log_execution(
                "Starting suggestion generation",
                {"project_id": project_id, "force_refresh": force_refresh}
            )
            
            # Check cache first if not forcing refresh
            if not force_refresh:
                cached_suggestions = self._get_cached_suggestions(project_id)
                if cached_suggestions:
                    self._log_execution("Retrieved suggestions from cache", {"project_id": project_id})
                    return self._create_success_result(
                        cached_suggestions,
                        {"source": "cache", "project_id": project_id}
                    )
            
            # Retrieve project data
            project_data = self._retrieve_project_data(project_id)
            if not project_data:
                return self._create_error_result(
                    f"Unable to retrieve project data for project {project_id}"
                )
            
            # Generate suggestions for each card type
            pending_tasks_suggestions = await self._generate_pending_tasks_suggestions(project_data)
            insights_suggestions = await self._generate_insights_suggestions(project_data)
            blockers_suggestions = await self._generate_blockers_suggestions(project_data)
            opportunities_suggestions = await self._generate_opportunities_suggestions(project_data)
            
            # Create dashboard suggestions object
            dashboard_suggestions = DashboardSuggestions(
                project_id=project_id,
                pending_tasks=pending_tasks_suggestions,
                project_insights=insights_suggestions,
                blockers=blockers_suggestions,
                opportunities=opportunities_suggestions
            )
            
            # Cache suggestions
            self._cache_suggestions(project_id, dashboard_suggestions)
            
            # Prepare result data
            result_data = dashboard_suggestions.dict()
            
            # Prepare metadata
            metadata = {
                "agent": "SuggestionsAgent",
                "model": self.config.model_name,
                "project_id": project_id,
                "total_suggestions": (
                    len(pending_tasks_suggestions) +
                    len(insights_suggestions) +
                    len(blockers_suggestions) +
                    len(opportunities_suggestions)
                ),
                "cached": False,
                "generated_at": datetime.now().isoformat()
            }
            
            self._log_execution("Suggestion generation completed successfully", metadata)
            
            return self._create_success_result(result_data, metadata)
        
        except ValidationError as e:
            error_msg = f"Suggestion validation failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
        
        except Exception as e:
            error_msg = f"Suggestion generation failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
    
    async def _generate_pending_tasks_suggestions(
        self,
        project_data: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate suggestions for pending tasks card."""
        try:
            prompt_text = self.pending_tasks_template.format(
                project_data=json.dumps(project_data, indent=2)
            )
            
            messages = MessageBuilder.create_messages(
                system_prompt="You are an AI project assistant. Generate actionable suggestions in valid JSON format.",
                user_message=prompt_text
            )
            
            response = self.llm.invoke(messages)
            suggestions = self._parse_suggestions_response(response.content)
            
            return self._rank_suggestions(suggestions)
        
        except Exception as e:
            self._log_error("Failed to generate pending tasks suggestions", {"error": str(e)})
            return []
    
    async def _generate_insights_suggestions(
        self,
        project_data: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate suggestions for project insights card."""
        try:
            prompt_text = self.insights_template.format(
                project_data=json.dumps(project_data, indent=2)
            )
            
            messages = MessageBuilder.create_messages(
                system_prompt="You are an AI project analyst. Generate data-driven insights in valid JSON format.",
                user_message=prompt_text
            )
            
            response = self.llm.invoke(messages)
            suggestions = self._parse_suggestions_response(response.content)
            
            return self._rank_suggestions(suggestions)
        
        except Exception as e:
            self._log_error("Failed to generate insights suggestions", {"error": str(e)})
            return []
    
    async def _generate_blockers_suggestions(
        self,
        project_data: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate suggestions for blockers card."""
        try:
            prompt_text = self.blockers_template.format(
                project_data=json.dumps(project_data, indent=2)
            )
            
            messages = MessageBuilder.create_messages(
                system_prompt="You are an AI risk analyst. Identify critical blockers in valid JSON format.",
                user_message=prompt_text
            )
            
            response = self.llm.invoke(messages)
            suggestions = self._parse_suggestions_response(response.content)
            
            return self._rank_suggestions(suggestions)
        
        except Exception as e:
            self._log_error("Failed to generate blockers suggestions", {"error": str(e)})
            return []
    
    async def _generate_opportunities_suggestions(
        self,
        project_data: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate suggestions for opportunities card."""
        try:
            prompt_text = self.opportunities_template.format(
                project_data=json.dumps(project_data, indent=2)
            )
            
            messages = MessageBuilder.create_messages(
                system_prompt="You are an AI opportunity scout. Identify opportunities in valid JSON format.",
                user_message=prompt_text
            )
            
            response = self.llm.invoke(messages)
            suggestions = self._parse_suggestions_response(response.content)
            
            return self._rank_suggestions(suggestions)
        
        except Exception as e:
            self._log_error("Failed to generate opportunities suggestions", {"error": str(e)})
            return []
    
    def _parse_suggestions_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse and validate LLM response for suggestions.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List of parsed suggestions
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Extract suggestions list
            suggestions_list = parsed.get("suggestions", [])
            
            # Validate and convert to Suggestion objects
            validated_suggestions = []
            for suggestion_data in suggestions_list:
                try:
                    suggestion = Suggestion(
                        title=suggestion_data.get("title", ""),
                        description=suggestion_data.get("description", ""),
                        action_url=suggestion_data.get("action_url"),
                        priority=suggestion_data.get("priority", 5),
                        generated_at=datetime.now().isoformat()
                    )
                    validated_suggestions.append(suggestion)
                except ValidationError as e:
                    self._log_error("Failed to validate suggestion", {"error": str(e)})
                    continue
            
            return validated_suggestions
        
        except json.JSONDecodeError as e:
            self._log_error("Failed to parse JSON response", {"error": str(e)})
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            self._log_error("Failed to parse response", {"error": str(e)})
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def _rank_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """
        Rank suggestions by priority and relevance.
        
        Args:
            suggestions: List of suggestions to rank
            
        Returns:
            Ranked list of suggestions (highest priority first)
        """
        # Sort by priority descending, then by generated_at descending
        ranked = sorted(
            suggestions,
            key=lambda s: (-s.priority, s.generated_at or ""),
            reverse=False
        )
        
        # Limit to top 3 suggestions
        return ranked[:3]
    
    def _retrieve_project_data(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve project data for analysis.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dictionary with project data or None if not found
        """
        try:
            # This would typically query the database for:
            # - Project metadata
            # - Task statistics (total, completed, overdue, in progress)
            # - Team members and workload
            # - Recent meetings and decisions
            # - Blockers and risks
            # - Velocity metrics
            
            # For now, return a sample structure
            project_data = {
                "project_id": project_id,
                "name": f"Project {project_id}",
                "total_tasks": 0,
                "completed_tasks": 0,
                "in_progress_tasks": 0,
                "overdue_tasks": 0,
                "team_members": [],
                "recent_meetings": [],
                "blockers": [],
                "velocity": 0,
                "completion_rate": 0.0
            }
            
            # If context retriever is available, use it to get real data
            if self.context_retriever:
                try:
                    context = self.context_retriever.build_prompt_context(
                        project_id=project_id,
                        query="project metrics and status",
                        max_tokens=2000
                    )
                    if context:
                        project_data["context"] = context
                except Exception as e:
                    self._log_error("Failed to retrieve context", {"error": str(e)})
            
            return project_data
        
        except Exception as e:
            self._log_error("Failed to retrieve project data", {"error": str(e)})
            return None
    
    def _get_cached_suggestions(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached suggestions from Redis.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Cached suggestions or None if not found/expired
        """
        try:
            cache_key = f"suggestions:{project_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
        
        except Exception as e:
            self._log_error("Failed to retrieve cached suggestions", {"error": str(e)})
            return None
    
    def _cache_suggestions(
        self,
        project_id: str,
        suggestions: DashboardSuggestions
    ) -> bool:
        """
        Cache suggestions in Redis with TTL.
        
        Args:
            project_id: Project identifier
            suggestions: Suggestions to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = f"suggestions:{project_id}"
            cache_data = json.dumps(suggestions.dict())
            
            self.redis_client.setex(
                cache_key,
                SUGGESTIONS_CACHE_TTL,
                cache_data
            )
            
            self._log_execution("Suggestions cached", {"project_id": project_id, "ttl": SUGGESTIONS_CACHE_TTL})
            
            return True
        
        except Exception as e:
            self._log_error("Failed to cache suggestions", {"error": str(e)})
            return False
    
    def refresh_suggestions(self, project_id: str) -> bool:
        """
        Refresh suggestions for a project (clear cache).
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if cache cleared successfully
        """
        try:
            cache_key = f"suggestions:{project_id}"
            self.redis_client.delete(cache_key)
            
            self._log_execution("Suggestions cache cleared", {"project_id": project_id})
            
            return True
        
        except Exception as e:
            self._log_error("Failed to clear suggestions cache", {"error": str(e)})
            return False
    
    def get_cached_or_generate(
        self,
        project_id: str,
        force_refresh: bool = False
    ) -> Optional[DashboardSuggestions]:
        """
        Get cached suggestions or generate new ones.
        
        Args:
            project_id: Project identifier
            force_refresh: Force generation even if cached
            
        Returns:
            DashboardSuggestions or None if generation fails
        """
        try:
            # Check cache first
            if not force_refresh:
                cached = self._get_cached_suggestions(project_id)
                if cached:
                    return DashboardSuggestions(**cached)
            
            # If not cached or force refresh, would need to call execute()
            # This is a helper method for synchronous access
            return None
        
        except Exception as e:
            self._log_error("Failed to get or generate suggestions", {"error": str(e)})
            return None



# Job function for background processing
async def generate_suggestions_job(project_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Background job function for generating suggestions.
    
    Args:
        project_id: Project identifier
        force_refresh: Force refresh even if cached
        
    Returns:
        Dictionary with suggestions data
    """
    try:
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.1,
            timeout=600
        )
        
        agent = SuggestionsAgent(config=config)
        result = await agent.execute(
            project_id=project_id,
            force_refresh=force_refresh
        )
        
        return result.to_dict()
    
    except Exception as e:
        logger.error(f"Suggestions job failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "project_id": project_id
        }
