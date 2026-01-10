import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch
from app.agents.task_extraction_agent import TaskExtractionAgent, ExtractedTask, AgentConfig, AgentStatus
from langchain.schema import AIMessage

# Mock responses for different scenarios
MOCK_RESPONSE_EXPLICIT = """
```json
{
    "tasks": [
        {
            "title": "Integrate API",
            "description": "Integrate the new payment API",
            "priority": "high",
            "assignee_name": "Alice",
            "assignment_type": "explicit",
            "confidence_score": 0.95,
            "assignment_reasoning": "User explicitly stated 'I will handle this'",
            "due_date": "2023-12-25"
        }
    ],
    "blockers": [],
    "summary": "Alice took the API task."
}
```
"""

MOCK_RESPONSE_INFERRED = """
```json
{
    "tasks": [
        {
            "title": "Fix Frontend Bug",
            "description": "Fix the alignment issue on the dashboard",
            "priority": "medium",
            "assignee_name": "Bob",
            "assignment_type": "inferred",
            "confidence_score": 0.8,
            "assignment_reasoning": "Bob is contextually known as the frontend lead",
            "due_date": null
        }
    ],
    "blockers": [],
    "summary": "Bob assigned frontend bug."
}
```
"""

MOCK_RESPONSE_UNASSIGNED = """
```json
{
    "tasks": [
        {
            "title": "Update Documentation",
            "description": "Update the API docs with new endpoints",
            "priority": "low",
            "assignee_name": null,
            "assignment_type": "unassigned",
            "confidence_score": 1.0,
            "assignment_reasoning": "No person mentioned or implied",
            "due_date": null
        }
    ],
    "blockers": [],
    "summary": "Docs need update."
}
```
"""

@pytest.fixture(autouse=True)
def mock_env_setup():
    """Mock environment variables and config for all tests."""
    with patch("app.agents.langchain_config.LangChainConfig.validate_api_key", return_value="sk-test-key"):
        with patch("app.agents.langchain_config.LangChainInitializer._initialized", False): # Reset init state
             # We also need to mock ChatOpenAI init to avoid actual connection attempts or validation if strict
             with patch("app.agents.langchain_config.ChatOpenAI"):
                 # Initialize the singleton with our mock
                 from app.agents.langchain_config import LangChainInitializer
                 LangChainInitializer.initialize()
                 yield

@pytest.mark.asyncio
async def test_explicit_assignment():
    """Test explicit assignment extraction."""
    config = AgentConfig()
    
    # We need to ensure get_llm returns our mock or a mock compatible object
    # The fixture above initializes it with a mock ChatOpenAI.
    # But TaskExtractionAgent calls get_llm().
    
    agent = TaskExtractionAgent(config)
    
    # Now patch the invoke method on the instance's llm
    agent.llm.invoke = MagicMock(return_value=AIMessage(content=MOCK_RESPONSE_EXPLICIT))
        
    result = await agent.execute(
        meeting_id="test-1",
        transcript="Alice: I'll handle the API integration by Christmas."
    )
    
    assert result.status == AgentStatus.SUCCESS
    task = result.data["tasks"][0]
    assert task["title"] == "Integrate API"
    assert task["assignee_name"] == "Alice"
    assert task["assignment_type"] == "explicit"
    assert task["confidence_score"] > 0.9

@pytest.mark.asyncio
async def test_inferred_assignment():
    """Test inferred assignment extraction."""
    config = AgentConfig()
    agent = TaskExtractionAgent(config)
    
    agent.llm.invoke = MagicMock(return_value=AIMessage(content=MOCK_RESPONSE_INFERRED))
        
    result = await agent.execute(
        meeting_id="test-2",
        transcript="Alice: We need to fix the dashboard. Bob, that's your domain."
    )
    
    assert result.status == AgentStatus.SUCCESS
    task = result.data["tasks"][0]
    assert task["assignee_name"] == "Bob"
    assert task["assignment_type"] == "inferred"
    assert task["confidence_score"] == 0.8
    assert "frontend lead" in task["assignment_reasoning"]

@pytest.mark.asyncio
async def test_unassigned_task():
    """Test unassigned task extraction."""
    config = AgentConfig()
    agent = TaskExtractionAgent(config)
    
    agent.llm.invoke = MagicMock(return_value=AIMessage(content=MOCK_RESPONSE_UNASSIGNED))
        
    result = await agent.execute(
        meeting_id="test-3",
        transcript="Someone really needs to update the docs."
    )
    
    assert result.status == AgentStatus.SUCCESS
    task = result.data["tasks"][0]
    assert task["assignee_name"] is None
    assert task["assignment_type"] == "unassigned"
