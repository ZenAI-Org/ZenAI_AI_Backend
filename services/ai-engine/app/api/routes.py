"""
FastAPI routes for AI orchestration workflows.

Provides endpoints for:
- Meeting processing (transcription, summarization, task extraction)
- Chat interface with context awareness
- AIPM analysis
- Dashboard suggestions
- Agent run status tracking
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from app.queue.orchestration_engine import get_orchestration_engine
from app.core.logger import get_logger
from app.core.error_dashboard import get_error_metrics
from app.core.error_notifications import get_error_notification_manager

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ai-workflows"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ProcessMeetingRequest(BaseModel):
    """Request model for meeting processing."""
    s3_key: str = Field(..., description="S3 key for audio file")
    project_id: Optional[str] = Field(None, description="Optional project ID for context")
    
    @validator("s3_key")
    def validate_s3_key(cls, v):
        if not v or not v.strip():
            raise ValueError("s3_key cannot be empty")
        return v


class ChatMessage(BaseModel):
    """Request model for chat messages."""
    message: str = Field(..., description="Chat message from user")
    project_id: str = Field(..., description="Project ID for context")
    user_role: Optional[str] = Field("member", description="User role for permission validation")
    
    @validator("message")
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("message cannot be empty")
        return v
    
    @validator("project_id")
    def validate_project_id(cls, v):
        if not v or not v.strip():
            raise ValueError("project_id cannot be empty")
        return v


class ChatResponse(BaseModel):
    """Response model for chat messages."""
    message: str
    sources: List[str] = []
    confidence: float
    follow_up_questions: Optional[List[str]] = None
    job_id: str


class AIProductManagerInsights(BaseModel):
    """Response model for AIPM analysis."""
    project_id: str
    health: str  # "healthy", "at-risk", "critical"
    blockers: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    job_id: str


class DashboardSuggestion(BaseModel):
    """Individual dashboard suggestion."""
    title: str
    description: str
    action_url: Optional[str] = None
    priority: int
    generated_at: datetime


class DashboardSuggestions(BaseModel):
    """Response model for dashboard suggestions."""
    project_id: str
    pending_tasks: List[DashboardSuggestion] = []
    project_insights: List[DashboardSuggestion] = []
    blockers: List[DashboardSuggestion] = []
    opportunities: List[DashboardSuggestion] = []
    job_id: str


class AgentRunStatus(BaseModel):
    """Response model for agent run status."""
    job_id: str
    status: str  # "queued", "running", "success", "error"
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ProcessMeetingResponse(BaseModel):
    """Response model for meeting processing."""
    meeting_id: str
    project_id: Optional[str]
    status: str
    jobs: Dict[str, str]
    errors: List[str] = []
    started_at: datetime


class ContextRetrievalRequest(BaseModel):
    """Request model for context retrieval."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Maximum results per content type", ge=1, le=20)
    
    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v


class ContextData(BaseModel):
    """Response model for context retrieval."""
    project_id: str
    query: str
    summaries: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    blockers: List[Dict[str, Any]] = []
    retrieved_at: datetime


# ============================================================================
# Dependency Injection
# ============================================================================

def get_orchestration_engine_dep():
    """Dependency for orchestration engine."""
    return get_orchestration_engine()


# ============================================================================
# Meeting Processing Endpoints
# ============================================================================

@router.post(
    "/meetings/{meeting_id}/process",
    response_model=ProcessMeetingResponse,
    summary="Trigger meeting processing workflow",
    description="Enqueue meeting for transcription, summarization, and task extraction",
)
async def process_meeting(
    meeting_id: str,
    request: ProcessMeetingRequest,
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Trigger complete meeting processing workflow.
    
    Orchestrates:
    1. Transcription (Whisper API)
    2. Summarization (LangChain)
    3. Task Extraction (LangChain)
    4. Notion Sync (if integration active)
    
    Args:
        meeting_id: Meeting ID
        request: ProcessMeetingRequest with s3_key and optional project_id
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        ProcessMeetingResponse with job IDs and status
        
    Raises:
        HTTPException: If meeting_id is invalid or processing fails
    """
    try:
        if not meeting_id or not meeting_id.strip():
            raise HTTPException(status_code=400, detail="meeting_id cannot be empty")
        
        logger.info(
            f"Processing meeting: meeting_id={meeting_id}, "
            f"s3_key={request.s3_key}, project_id={request.project_id}"
        )
        
        # Orchestrate meeting processing
        result = await orchestrator.process_meeting(
            meeting_id=meeting_id,
            s3_key=request.s3_key,
            project_id=request.project_id,
        )
        
        if result.get("status") == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Meeting processing failed: {result.get('error', 'Unknown error')}",
            )
        
        return ProcessMeetingResponse(
            meeting_id=meeting_id,
            project_id=request.project_id,
            status=result.get("status", "processing"),
            jobs=result.get("jobs", {}),
            errors=result.get("errors", []),
            started_at=datetime.fromisoformat(result.get("started_at", datetime.now().isoformat())),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing meeting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send chat message with context awareness",
    description="Query project data conversationally with role-based access control",
)
async def chat(
    request: ChatMessage,
    user_id: str = Query(..., description="User ID"),
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Send chat message to AI Chat Agent.
    
    Features:
    - Context-aware responses using semantic search
    - Role-based access control
    - Conversation history management
    - Real-time streaming via Socket.io
    
    Args:
        request: ChatMessage with message, project_id, and optional user_role
        user_id: User ID (from query parameter)
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        ChatResponse with message, sources, and confidence
        
    Raises:
        HTTPException: If user lacks permissions or chat fails
    """
    try:
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required")
        
        logger.info(
            f"Chat request: user_id={user_id}, project_id={request.project_id}, "
            f"message_length={len(request.message)}"
        )
        
        # Route to Chat Agent
        result = await orchestrator.chat(
            user_id=user_id,
            project_id=request.project_id,
            message=request.message,
            user_role=request.user_role,
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Chat failed: {result.get('error', 'Unknown error')}",
            )
        
        chat_result = result.get("result", {})
        
        return ChatResponse(
            message=chat_result.get("message", ""),
            sources=chat_result.get("sources", []),
            confidence=chat_result.get("confidence", 0.0),
            follow_up_questions=chat_result.get("follow_up_questions"),
            job_id=result.get("job_id", ""),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ============================================================================
# AIPM Analysis Endpoints
# ============================================================================

@router.get(
    "/projects/{project_id}/aipm",
    response_model=AIProductManagerInsights,
    summary="Get AIPM analysis for project",
    description="Analyze project health and generate strategic recommendations",
)
async def get_aipm_analysis(
    project_id: str,
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Get AI Product Manager analysis for a project.
    
    Analyzes:
    - Task velocity and completion rates
    - Blockers and risks
    - Team capacity and workload
    - Meeting trends and decisions
    
    Args:
        project_id: Project ID
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        AIProductManagerInsights with health status and recommendations
        
    Raises:
        HTTPException: If project_id is invalid or analysis fails
    """
    try:
        if not project_id or not project_id.strip():
            raise HTTPException(status_code=400, detail="project_id cannot be empty")
        
        logger.info(f"AIPM analysis request: project_id={project_id}")
        
        # Route to AIPM Agent
        result = await orchestrator.analyze_project(project_id=project_id)
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"AIPM analysis failed: {result.get('error', 'Unknown error')}",
            )
        
        aipm_result = result.get("result", {})
        
        return AIProductManagerInsights(
            project_id=project_id,
            health=aipm_result.get("health", "unknown"),
            blockers=aipm_result.get("blockers", []),
            recommendations=aipm_result.get("recommendations", []),
            metrics=aipm_result.get("metrics", {}),
            job_id=result.get("job_id", ""),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AIPM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AIPM analysis failed: {str(e)}")


# ============================================================================
# Dashboard Suggestions Endpoints
# ============================================================================

@router.get(
    "/projects/{project_id}/suggestions",
    response_model=DashboardSuggestions,
    summary="Get dashboard suggestions for project",
    description="Generate AI-powered suggestions for dashboard cards",
)
async def get_suggestions(
    project_id: str,
    force_refresh: bool = Query(False, description="Force refresh even if cached"),
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Get dashboard suggestions for a project.
    
    Generates suggestions for:
    - Pending tasks
    - Project insights
    - Blockers
    - Opportunities
    
    Args:
        project_id: Project ID
        force_refresh: Force refresh even if cached
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        DashboardSuggestions with suggestions grouped by type
        
    Raises:
        HTTPException: If project_id is invalid or generation fails
    """
    try:
        if not project_id or not project_id.strip():
            raise HTTPException(status_code=400, detail="project_id cannot be empty")
        
        logger.info(
            f"Suggestions request: project_id={project_id}, "
            f"force_refresh={force_refresh}"
        )
        
        # Route to Suggestions Agent
        result = await orchestrator.generate_suggestions(
            project_id=project_id,
            force_refresh=force_refresh,
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Suggestions generation failed: {result.get('error', 'Unknown error')}",
            )
        
        suggestions_result = result.get("result", {})
        
        # Parse suggestions into DashboardSuggestion objects
        def parse_suggestions(suggestions_list):
            return [
                DashboardSuggestion(
                    title=s.get("title", ""),
                    description=s.get("description", ""),
                    action_url=s.get("action_url"),
                    priority=s.get("priority", 0),
                    generated_at=datetime.fromisoformat(
                        s.get("generated_at", datetime.now().isoformat())
                    ),
                )
                for s in suggestions_list
            ]
        
        return DashboardSuggestions(
            project_id=project_id,
            pending_tasks=parse_suggestions(suggestions_result.get("pending_tasks", [])),
            project_insights=parse_suggestions(suggestions_result.get("project_insights", [])),
            blockers=parse_suggestions(suggestions_result.get("blockers", [])),
            opportunities=parse_suggestions(suggestions_result.get("opportunities", [])),
            job_id=result.get("job_id", ""),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestions generation failed: {str(e)}")


# ============================================================================
# Agent Run Status Endpoints
# ============================================================================

@router.get(
    "/agent-runs/{job_id}",
    response_model=AgentRunStatus,
    summary="Get agent run status",
    description="Poll status of an agent run job",
)
async def get_agent_run_status(
    job_id: str,
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Get status of an agent run job.
    
    Args:
        job_id: Job ID
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        AgentRunStatus with current status and results
        
    Raises:
        HTTPException: If job_id is invalid or not found
    """
    try:
        if not job_id or not job_id.strip():
            raise HTTPException(status_code=400, detail="job_id cannot be empty")
        
        logger.info(f"Status request: job_id={job_id}")
        
        # Get job status
        status_data = orchestrator.get_job_status(job_id)
        
        if status_data.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Parse timestamps
        created_at = None
        started_at = None
        ended_at = None
        
        if status_data.get("created_at"):
            created_at = datetime.fromisoformat(status_data["created_at"])
        if status_data.get("started_at"):
            started_at = datetime.fromisoformat(status_data["started_at"])
        if status_data.get("ended_at"):
            ended_at = datetime.fromisoformat(status_data["ended_at"])
        
        return AgentRunStatus(
            job_id=job_id,
            status=status_data.get("status", "unknown"),
            created_at=created_at,
            started_at=started_at,
            ended_at=ended_at,
            result=status_data.get("result"),
            error=status_data.get("error"),
            metadata=status_data.get("meta", {}),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent run status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# ============================================================================
# Context Retrieval Endpoints
# ============================================================================

@router.post(
    "/projects/{project_id}/context",
    response_model=ContextData,
    summary="Retrieve semantic search context",
    description="Retrieve relevant project context using semantic search",
)
async def retrieve_context(
    project_id: str,
    request: ContextRetrievalRequest,
    orchestrator=Depends(get_orchestration_engine_dep),
):
    """
    Retrieve semantic search context for a project.
    
    Args:
        project_id: Project ID
        request: ContextRetrievalRequest with query and limit
        orchestrator: AIOrchestrationEngine instance
        
    Returns:
        ContextData with retrieved summaries, decisions, and blockers
        
    Raises:
        HTTPException: If project_id is invalid or retrieval fails
    """
    try:
        if not project_id or not project_id.strip():
            raise HTTPException(status_code=400, detail="project_id cannot be empty")
        
        logger.info(
            f"Context retrieval request: project_id={project_id}, "
            f"query={request.query}, limit={request.limit}"
        )
        
        # Retrieve context
        result = await orchestrator.retrieve_context(
            project_id=project_id,
            query=request.query,
            limit=request.limit,
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Context retrieval failed: {result.get('error', 'Unknown error')}",
            )
        
        context = result.get("context", {})
        
        return ContextData(
            project_id=project_id,
            query=request.query,
            summaries=context.get("summaries", []),
            decisions=context.get("decisions", []),
            blockers=context.get("blockers", []),
            retrieved_at=datetime.fromisoformat(
                result.get("retrieved_at", datetime.now().isoformat())
            ),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


# ============================================================================
# Error Dashboard Endpoints
# ============================================================================

@router.get(
    "/errors/metrics",
    summary="Get error metrics and dashboard data",
    description="Retrieve error metrics for monitoring and alerting",
)
async def get_error_metrics_endpoint():
    """
    Get error metrics for the error dashboard.
    
    Returns:
        Dictionary with error metrics including error rate, counts, and recent errors
    """
    try:
        metrics = get_error_metrics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.get_metrics_summary(),
            "recent_errors": metrics.get_recent_errors(limit=10),
        }
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.get(
    "/errors/by-api/{api_name}",
    summary="Get errors for a specific API",
    description="Retrieve recent errors for a specific API (whisper, openai, notion)",
)
async def get_errors_by_api(
    api_name: str,
    limit: int = Query(10, description="Maximum number of errors to return", ge=1, le=50),
):
    """
    Get recent errors for a specific API.
    
    Args:
        api_name: Name of the API (whisper, openai, notion)
        limit: Maximum number of errors to return
        
    Returns:
        List of error records for the API
    """
    try:
        if not api_name or not api_name.strip():
            raise HTTPException(status_code=400, detail="api_name cannot be empty")
        
        metrics = get_error_metrics()
        errors = metrics.get_errors_by_api(api_name, limit=limit)
        
        return {
            "status": "success",
            "api_name": api_name,
            "error_count": len(errors),
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving API errors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve errors: {str(e)}")


@router.get(
    "/errors/alert-status",
    summary="Check if error rate exceeds alert threshold",
    description="Determine if error rate is above the alert threshold",
)
async def check_alert_status(
    threshold: float = Query(5.0, description="Alert threshold as percentage", ge=0.1, le=100),
):
    """
    Check if error rate exceeds alert threshold.
    
    Args:
        threshold: Alert threshold as percentage (default 5%)
        
    Returns:
        Dictionary with alert status and current error rate
    """
    try:
        metrics = get_error_metrics()
        should_alert = metrics.should_alert(threshold)
        error_rate = metrics.get_error_rate()
        
        return {
            "status": "success",
            "should_alert": should_alert,
            "error_rate": error_rate,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error checking alert status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check alert status: {str(e)}")


# ============================================================================
# Health Check Endpoints
# ============================================================================

@router.get(
    "/health",
    summary="Health check for AI workflows",
    description="Check if AI orchestration engine is operational",
)
async def health_check(orchestrator=Depends(get_orchestration_engine_dep)):
    """
    Health check endpoint for AI workflows.
    
    Returns:
        Dictionary with health status and queue statistics
    """
    try:
        stats = orchestrator.get_queue_stats()
        metrics = get_error_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "queue_stats": stats,
            "error_metrics": metrics.get_metrics_summary(),
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
