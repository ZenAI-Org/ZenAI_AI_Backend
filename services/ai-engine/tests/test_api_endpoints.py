"""
Tests for FastAPI endpoints for AI workflows.

Tests cover:
- Request validation with Pydantic
- Response model validation
- Endpoint routing
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from app.main import app
from app.api.routes import (
    ProcessMeetingRequest,
    ChatMessage,
    ProcessMeetingResponse,
    ChatResponse,
    AIProductManagerInsights,
    DashboardSuggestions,
    AgentRunStatus,
    ContextData,
    ContextRetrievalRequest,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authorization headers."""
    return {"Authorization": "Bearer test_token_123"}


# ============================================================================
# Meeting Processing Endpoint Tests
# ============================================================================

class TestProcessMeetingEndpoint:
    """Tests for POST /api/meetings/{meeting_id}/process endpoint."""
    
    def test_process_meeting_missing_s3_key(self, client, auth_headers):
        """Test meeting processing with missing s3_key."""
        response = client.post(
            "/api/meetings/meeting_123/process",
            json={"project_id": "project_456"},
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_meeting_empty_s3_key(self, client, auth_headers):
        """Test meeting processing with empty s3_key."""
        response = client.post(
            "/api/meetings/meeting_123/process",
            json={
                "s3_key": "",
                "project_id": "project_456",
            },
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_meeting_invalid_meeting_id(self, client, auth_headers):
        """Test meeting processing with invalid meeting_id."""
        response = client.post(
            "/api/meetings//process",
            json={
                "s3_key": "meetings/audio_123.mp3",
                "project_id": "project_456",
            },
            headers=auth_headers,
        )
        
        assert response.status_code == 404  # Not found


# ============================================================================
# Chat Endpoint Tests
# ============================================================================

class TestChatEndpoint:
    """Tests for POST /api/chat endpoint."""
    
    def test_chat_missing_message(self, client, auth_headers):
        """Test chat with missing message."""
        response = client.post(
            "/api/chat",
            json={
                "project_id": "project_456",
                "user_role": "member",
            },
            params={"user_id": "user_123"},
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_empty_message(self, client, auth_headers):
        """Test chat with empty message."""
        response = client.post(
            "/api/chat",
            json={
                "message": "",
                "project_id": "project_456",
                "user_role": "member",
            },
            params={"user_id": "user_123"},
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_missing_user_id(self, client, auth_headers):
        """Test chat with missing user_id."""
        response = client.post(
            "/api/chat",
            json={
                "message": "What is the project status?",
                "project_id": "project_456",
                "user_role": "member",
            },
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error (missing required query param)


# ============================================================================
# AIPM Analysis Endpoint Tests
# ============================================================================

class TestAIPMAnalysisEndpoint:
    """Tests for GET /api/projects/{project_id}/aipm endpoint."""
    
    def test_aipm_analysis_invalid_project_id(self, client, auth_headers):
        """Test AIPM analysis with invalid project_id."""
        response = client.get(
            "/api/projects//aipm",
            headers=auth_headers,
        )
        
        assert response.status_code == 404  # Not found


# ============================================================================
# Dashboard Suggestions Endpoint Tests
# ============================================================================

class TestSuggestionsEndpoint:
    """Tests for GET /api/projects/{project_id}/suggestions endpoint."""
    
    def test_suggestions_invalid_project_id(self, client, auth_headers):
        """Test suggestions with invalid project_id."""
        response = client.get(
            "/api/projects//suggestions",
            headers=auth_headers,
        )
        
        assert response.status_code == 404  # Not found


# ============================================================================
# Agent Run Status Endpoint Tests
# ============================================================================

class TestAgentRunStatusEndpoint:
    """Tests for GET /api/agent-runs/{job_id} endpoint."""
    
    def test_agent_run_status_invalid_job_id(self, client, auth_headers):
        """Test agent run status with invalid job_id."""
        response = client.get(
            "/api/agent-runs/",
            headers=auth_headers,
        )
        
        assert response.status_code == 404  # Not found


# ============================================================================
# Context Retrieval Endpoint Tests
# ============================================================================

class TestContextRetrievalEndpoint:
    """Tests for POST /api/projects/{project_id}/context endpoint."""
    
    def test_context_retrieval_missing_query(self, client, auth_headers):
        """Test context retrieval with missing query."""
        response = client.post(
            "/api/projects/project_456/context",
            json={"limit": 5},
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_context_retrieval_empty_query(self, client, auth_headers):
        """Test context retrieval with empty query."""
        response = client.post(
            "/api/projects/project_456/context",
            json={
                "query": "",
                "limit": 5,
            },
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_context_retrieval_invalid_limit(self, client, auth_headers):
        """Test context retrieval with invalid limit."""
        response = client.post(
            "/api/projects/project_456/context",
            json={
                "query": "project blockers",
                "limit": 100,  # Exceeds max of 20
            },
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error


# ============================================================================
# Health Check Endpoint Tests
# ============================================================================

class TestHealthCheckEndpoint:
    """Tests for GET /api/health endpoint."""
    
    def test_health_check_endpoint_exists(self, client):
        """Test that health check endpoint exists."""
        response = client.get("/api/health")
        
        # Should return either 200 or 503, but endpoint should exist
        assert response.status_code in [200, 503]


# ============================================================================
# Request Validation Tests
# ============================================================================

class TestRequestValidation:
    """Tests for Pydantic request validation."""
    
    def test_process_meeting_request_validation(self):
        """Test ProcessMeetingRequest validation."""
        # Valid request
        valid_request = ProcessMeetingRequest(
            s3_key="meetings/audio.mp3",
            project_id="project_123",
        )
        assert valid_request.s3_key == "meetings/audio.mp3"
        
        # Invalid request - empty s3_key
        with pytest.raises(ValueError):
            ProcessMeetingRequest(s3_key="", project_id="project_123")
    
    def test_chat_message_validation(self):
        """Test ChatMessage validation."""
        # Valid message
        valid_message = ChatMessage(
            message="What is the project status?",
            project_id="project_123",
            user_role="member",
        )
        assert valid_message.message == "What is the project status?"
        
        # Invalid message - empty
        with pytest.raises(ValueError):
            ChatMessage(message="", project_id="project_123")
        
        # Invalid project_id - empty
        with pytest.raises(ValueError):
            ChatMessage(message="What is the status?", project_id="")
    
    def test_context_retrieval_request_validation(self):
        """Test ContextRetrievalRequest validation."""
        from app.api.routes import ContextRetrievalRequest
        
        # Valid request
        valid_request = ContextRetrievalRequest(
            query="project blockers",
            limit=5,
        )
        assert valid_request.query == "project blockers"
        assert valid_request.limit == 5
        
        # Invalid query - empty
        with pytest.raises(ValueError):
            ContextRetrievalRequest(query="", limit=5)
        
        # Invalid limit - too high
        with pytest.raises(ValueError):
            ContextRetrievalRequest(query="blockers", limit=100)


# ============================================================================
# Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def test_api_endpoints_are_registered(self):
        """Test that all API endpoints are registered."""
        from app.main import app
        
        # Get all registered routes
        routes = [route.path for route in app.routes if "/api" in route.path]
        
        # Verify all required endpoints are registered
        assert "/api/meetings/{meeting_id}/process" in routes
        assert "/api/chat" in routes
        assert "/api/projects/{project_id}/aipm" in routes
        assert "/api/projects/{project_id}/suggestions" in routes
        assert "/api/agent-runs/{job_id}" in routes
        assert "/api/projects/{project_id}/context" in routes
        assert "/api/health" in routes
