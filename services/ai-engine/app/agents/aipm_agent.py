"""
AI Product Manager (AIPM) Agent for project analysis and strategic recommendations.
Analyzes project health, identifies blockers, and generates prioritized recommendations.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import json

from app.agents.base_agent import BaseAgent, AgentConfig, AgentStatus, AgentResult
from app.agents.langchain_config import LangChainInitializer, PromptTemplateManager, MessageBuilder
from app.core.context_retriever import ContextRetriever
from app.core.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Project health status enum."""
    HEALTHY = "healthy"
    AT_RISK = "at-risk"
    CRITICAL = "critical"


class ImpactLevel(str, Enum):
    """Impact level for blockers and recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Blocker(BaseModel):
    """Schema for project blockers."""
    
    title: str = Field(
        ...,
        description="Title of the blocker",
        min_length=5,
        max_length=200
    )
    impact: ImpactLevel = Field(
        ...,
        description="Impact level of the blocker"
    )
    affected_tasks: List[str] = Field(
        default_factory=list,
        description="List of affected task IDs",
        max_length=20
    )
    suggested_resolution: str = Field(
        ...,
        description="Suggested resolution for the blocker",
        min_length=10,
        max_length=500
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True
        validate_assignment = True


class Recommendation(BaseModel):
    """Schema for strategic recommendations."""
    
    title: str = Field(
        ...,
        description="Title of the recommendation",
        min_length=5,
        max_length=200
    )
    rationale: str = Field(
        ...,
        description="Rationale for the recommendation",
        min_length=10,
        max_length=500
    )
    priority: int = Field(
        ...,
        description="Priority score (1-10, higher is more important)",
        ge=1,
        le=10
    )
    estimated_impact: str = Field(
        ...,
        description="Estimated impact of implementing the recommendation",
        min_length=5,
        max_length=300
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True
        validate_assignment = True


class ProjectMetrics(BaseModel):
    """Schema for project metrics."""
    
    total_tasks: int = Field(
        default=0,
        description="Total number of tasks in the project",
        ge=0
    )
    completed_tasks: int = Field(
        default=0,
        description="Number of completed tasks",
        ge=0
    )
    overdue_tasks: int = Field(
        default=0,
        description="Number of overdue tasks",
        ge=0
    )
    task_completion_rate: float = Field(
        default=0.0,
        description="Task completion rate as percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    average_task_velocity: float = Field(
        default=0.0,
        description="Average tasks completed per week",
        ge=0.0
    )
    team_capacity_utilization: float = Field(
        default=0.0,
        description="Team capacity utilization percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    active_blockers_count: int = Field(
        default=0,
        description="Number of active blockers",
        ge=0
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class AIProductManagerInsights(BaseModel):
    """Schema for AIPM analysis output."""
    
    project_id: str = Field(
        ...,
        description="Project identifier"
    )
    health: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Overall project health status"
    )
    blockers: List[Blocker] = Field(
        default_factory=list,
        description="List of identified blockers",
        max_length=10
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="List of strategic recommendations",
        max_length=10
    )
    metrics: ProjectMetrics = Field(
        default_factory=ProjectMetrics,
        description="Project metrics"
    )
    summary: str = Field(
        default="",
        description="Executive summary of project analysis",
        max_length=1000
    )
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class AIProductManagerAgent(BaseAgent):
    """
    AI Product Manager Agent for project analysis and strategic recommendations.
    
    Responsibilities:
    - Aggregate project metrics (task velocity, completion rate, blockers)
    - Analyze meeting trends and decisions from pgvector
    - Identify risks and opportunities using LangChain analysis chain
    - Generate prioritized recommendations
    - Create Pydantic schema for AIPM insights
    - Store insights in AgentRun output
    """
    
    def __init__(self, config: AgentConfig, db_connection=None):
        """
        Initialize AIPM Agent.
        
        Args:
            config: Agent configuration
            db_connection: PostgreSQL connection for data retrieval
        """
        super().__init__(config)
        self.db_connection = db_connection
        self.llm = LangChainInitializer.get_llm()
        self.context_retriever = ContextRetriever(db_connection) if db_connection else None
        self.embedding_store = EmbeddingStore(db_connection) if db_connection else None
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize LangChain prompt templates for AIPM analysis."""
        
        # Metrics analysis template
        self.metrics_analysis_template = PromptTemplateManager.create_template(
            template="""You are an expert AI Product Manager. Analyze the following project metrics and context to identify health status, blockers, and recommendations.

Project Metrics:
{metrics}

Project Context:
{context}

Recent Meeting Insights:
{meeting_insights}

Based on this analysis, provide insights in the following JSON format:
{{
    "health": "healthy|at-risk|critical",
    "blockers": [
        {{
            "title": "Blocker title",
            "impact": "high|medium|low",
            "affected_tasks": ["task_id_1", "task_id_2"],
            "suggested_resolution": "How to resolve this blocker"
        }}
    ],
    "recommendations": [
        {{
            "title": "Recommendation title",
            "rationale": "Why this recommendation",
            "priority": 8,
            "estimated_impact": "Expected impact of this recommendation"
        }}
    ],
    "summary": "Executive summary of the analysis"
}}

Guidelines:
- Identify blockers that are preventing progress
- Prioritize recommendations by impact and feasibility
- Consider team capacity and velocity trends
- Flag risks early based on metrics trends
- Provide actionable recommendations""",
            input_variables=["metrics", "context", "meeting_insights"],
            description="Template for AIPM metrics analysis"
        )
        
        self._log_execution("Prompt templates initialized")
    
    async def execute(
        self,
        project_id: str,
        **kwargs
    ) -> AgentResult:
        """
        Execute AIPM analysis workflow.
        
        Args:
            project_id: Project identifier
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with AIPM insights
        """
        try:
            self._log_execution(
                "Starting AIPM analysis",
                {"project_id": project_id}
            )
            
            # Validate input
            if not project_id or len(project_id.strip()) == 0:
                return self._create_error_result("Project ID is required")
            
            # Aggregate project metrics
            metrics = await self._aggregate_metrics(project_id)
            self._log_execution("Aggregated project metrics", {"project_id": project_id})
            
            # Retrieve project context
            context_str = ""
            if self.context_retriever:
                try:
                    context_str = self.context_retriever.build_prompt_context(
                        project_id=project_id,
                        query="project overview status health",
                        max_tokens=1500
                    )
                    self._log_execution("Retrieved project context", {"project_id": project_id})
                except Exception as e:
                    self._log_error("Failed to retrieve context", {"error": str(e)})
                    context_str = ""
            
            # Retrieve meeting insights
            meeting_insights = await self._retrieve_meeting_insights(project_id)
            self._log_execution("Retrieved meeting insights", {"project_id": project_id})
            
            # Format metrics for prompt
            metrics_json = json.dumps({
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "overdue_tasks": metrics.overdue_tasks,
                "completion_rate": f"{metrics.task_completion_rate:.1f}%",
                "velocity": f"{metrics.average_task_velocity:.1f} tasks/week",
                "capacity_utilization": f"{metrics.team_capacity_utilization:.1f}%",
                "active_blockers": metrics.active_blockers_count
            })
            
            # Format prompt
            prompt_text = self.metrics_analysis_template.format(
                metrics=metrics_json,
                context=context_str if context_str else "No previous context available",
                meeting_insights=meeting_insights if meeting_insights else "No recent meeting insights"
            )
            
            # Create messages for LLM
            messages = MessageBuilder.create_messages(
                system_prompt="You are an expert AI Product Manager. Provide analysis in valid JSON format.",
                user_message=prompt_text
            )
            
            # Call LLM
            self._log_execution("Calling LLM for AIPM analysis")
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse and validate response
            analysis_data = self._parse_and_validate_response(response_text)
            
            # Create AIPM insights output
            aipm_insights = AIProductManagerInsights(
                project_id=project_id,
                health=analysis_data.get("health", HealthStatus.HEALTHY),
                blockers=[Blocker(**b) for b in analysis_data.get("blockers", [])],
                recommendations=[Recommendation(**r) for r in analysis_data.get("recommendations", [])],
                metrics=metrics,
                summary=analysis_data.get("summary", "")
            )
            
            # Store insights embedding if database connection available
            embedding_id = None
            if self.embedding_store:
                try:
                    embedding_id = self.embedding_store.store_embedding(
                        project_id=project_id,
                        content_type="aipm_analysis",
                        content_id=f"aipm_{project_id}_{datetime.now().isoformat()}",
                        text=aipm_insights.summary,
                        metadata={
                            "health": aipm_insights.health.value,
                            "blockers_count": len(aipm_insights.blockers),
                            "recommendations_count": len(aipm_insights.recommendations),
                            "completion_rate": aipm_insights.metrics.task_completion_rate
                        }
                    )
                    self._log_execution("Stored AIPM insights embedding", {"embedding_id": embedding_id})
                except Exception as e:
                    self._log_error("Failed to store embedding", {"error": str(e)})
            
            # Prepare result data
            result_data = {
                "project_id": project_id,
                "health": aipm_insights.health.value,
                "blockers": [b.dict() for b in aipm_insights.blockers],
                "recommendations": [r.dict() for r in aipm_insights.recommendations],
                "metrics": aipm_insights.metrics.dict(),
                "summary": aipm_insights.summary,
                "embedding_id": embedding_id
            }
            
            # Prepare metadata
            metadata = {
                "agent": "AIProductManagerAgent",
                "model": self.config.model_name,
                "blockers_identified": len(aipm_insights.blockers),
                "recommendations_generated": len(aipm_insights.recommendations),
                "health_status": aipm_insights.health.value,
                "embedding_stored": embedding_id is not None
            }
            
            self._log_execution("AIPM analysis completed successfully", metadata)
            
            return self._create_success_result(result_data, metadata)
        
        except ValidationError as e:
            error_msg = f"AIPM insights validation failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
        
        except Exception as e:
            error_msg = f"AIPM analysis failed: {str(e)}"
            self._log_error(error_msg)
            return self._create_error_result(error_msg)
    
    async def _aggregate_metrics(self, project_id: str) -> ProjectMetrics:
        """
        Aggregate project metrics from database.
        
        Args:
            project_id: Project identifier
            
        Returns:
            ProjectMetrics object
        """
        try:
            # In a real implementation, this would query the database
            # For now, return mock metrics
            metrics = ProjectMetrics(
                total_tasks=45,
                completed_tasks=32,
                overdue_tasks=3,
                task_completion_rate=71.1,
                average_task_velocity=8.5,
                team_capacity_utilization=85.0,
                active_blockers_count=2
            )
            return metrics
        except Exception as e:
            self._log_error("Failed to aggregate metrics", {"error": str(e)})
            return ProjectMetrics()
    
    async def _retrieve_meeting_insights(self, project_id: str) -> str:
        """
        Retrieve recent meeting insights from pgvector.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Formatted meeting insights string
        """
        try:
            if not self.context_retriever:
                return ""
            
            # Retrieve recent meeting summaries
            insights = self.context_retriever.build_prompt_context(
                project_id=project_id,
                query="recent meetings decisions blockers risks",
                max_tokens=1000
            )
            return insights
        except Exception as e:
            self._log_error("Failed to retrieve meeting insights", {"error": str(e)})
            return ""
    
    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed and validated analysis data
            
        Raises:
            ValueError: If response cannot be parsed or validated
        """
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            if "health" not in parsed:
                parsed["health"] = HealthStatus.HEALTHY
            
            # Ensure lists are properly formatted
            if "blockers" not in parsed:
                parsed["blockers"] = []
            if "recommendations" not in parsed:
                parsed["recommendations"] = []
            
            # Normalize blocker data
            normalized_blockers = []
            for blocker in parsed.get("blockers", []):
                normalized_blockers.append({
                    "title": str(blocker.get("title", "Unknown blocker")),
                    "impact": blocker.get("impact", "medium"),
                    "affected_tasks": blocker.get("affected_tasks", []),
                    "suggested_resolution": str(blocker.get("suggested_resolution", "To be determined"))
                })
            parsed["blockers"] = normalized_blockers
            
            # Normalize recommendation data
            normalized_recommendations = []
            for rec in parsed.get("recommendations", []):
                try:
                    priority = int(rec.get("priority", 5))
                    priority = max(1, min(10, priority))  # Clamp between 1-10
                except (ValueError, TypeError):
                    priority = 5
                
                normalized_recommendations.append({
                    "title": str(rec.get("title", "Unknown recommendation")),
                    "rationale": str(rec.get("rationale", "To be determined")),
                    "priority": priority,
                    "estimated_impact": str(rec.get("estimated_impact", "To be determined"))
                })
            parsed["recommendations"] = normalized_recommendations
            
            return parsed
        
        except json.JSONDecodeError as e:
            self._log_error("Failed to parse JSON response", {"error": str(e)})
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            self._log_error("Failed to parse response", {"error": str(e)})
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def validate_insights(self, insights_data: Dict[str, Any]) -> bool:
        """
        Validate AIPM insights data against schema.
        
        Args:
            insights_data: Insights data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            AIProductManagerInsights(**insights_data)
            return True
        except ValidationError as e:
            self._log_error("Insights validation failed", {"error": str(e)})
            return False
    
    def calculate_health_score(self, metrics: ProjectMetrics) -> float:
        """
        Calculate overall health score based on metrics (0-1).
        
        Args:
            metrics: Project metrics
            
        Returns:
            Health score between 0 and 1
        """
        score = 0.0
        
        # Completion rate (0-0.3)
        if metrics.task_completion_rate >= 80:
            score += 0.3
        elif metrics.task_completion_rate >= 60:
            score += 0.2
        elif metrics.task_completion_rate >= 40:
            score += 0.1
        
        # Overdue tasks (0-0.3)
        if metrics.overdue_tasks == 0:
            score += 0.3
        elif metrics.overdue_tasks <= 2:
            score += 0.2
        elif metrics.overdue_tasks <= 5:
            score += 0.1
        
        # Blockers (0-0.2)
        if metrics.active_blockers_count == 0:
            score += 0.2
        elif metrics.active_blockers_count <= 2:
            score += 0.1
        
        # Capacity utilization (0-0.2)
        if 70 <= metrics.team_capacity_utilization <= 90:
            score += 0.2
        elif 50 <= metrics.team_capacity_utilization <= 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def determine_health_status(self, health_score: float) -> HealthStatus:
        """
        Determine health status based on health score.
        
        Args:
            health_score: Health score (0-1)
            
        Returns:
            HealthStatus enum value
        """
        if health_score >= 0.7:
            return HealthStatus.HEALTHY
        elif health_score >= 0.4:
            return HealthStatus.AT_RISK
        else:
            return HealthStatus.CRITICAL



# Job function for background processing
async def analyze_project_job(
    project_id: str,
) -> Dict[str, Any]:
    """
    Background job function for AIPM analysis.
    
    Args:
        project_id: Project ID
        
    Returns:
        Dictionary with AIPM analysis result
    """
    try:
        config = AgentConfig(
            name="AIProductManagerAgent",
            model="gpt-4",
            temperature=0.4,
        )
        agent = AIProductManagerAgent(config)
        result = await agent.execute(project_id=project_id)
        
        if result.status == AgentStatus.SUCCESS:
            return {
                "status": "success",
                "project_id": project_id,
                "health": result.data.get("health", "unknown"),
                "blockers": result.data.get("blockers", []),
                "recommendations": result.data.get("recommendations", []),
                "metrics": result.data.get("metrics", {}),
            }
        else:
            return {
                "status": "error",
                "project_id": project_id,
                "error": result.error,
            }
    
    except Exception as e:
        logger.error(f"AIPM analysis job failed: {str(e)}")
        return {
            "status": "error",
            "project_id": project_id,
            "error": str(e),
        }
