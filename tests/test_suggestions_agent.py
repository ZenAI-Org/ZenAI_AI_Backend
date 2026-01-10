import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from app.agents.suggestions_agent import SuggestionsAgent, Suggestion, AgentConfig, AgentStatus
from langchain.schema import AIMessage

MOCK_RESPONSE = """
```json
{
    "suggestions": [
        {
            "title": "Address Blocking Task",
            "description": "The 'API Integration' task is blocking 3 others.",
            "action_url": null,
            "priority": 9,
            "reasoning": "Because this task is blocking 3 other items and the deadline is in 2 days.",
            "confidence": 0.95
        },
        {
            "title": "Review Overdue Item",
            "description": "The 'Design Review' is 5 days overdue.",
            "action_url": "/tasks/123",
            "priority": 7,
            "reasoning": "Because this item is significantly overdue and impacts the design phase.",
            "confidence": 0.8
        }
    ]
}
```
"""

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("app.agents.langchain_config.LangChainConfig.validate_api_key", return_value="sk-test-key"):
        with patch("app.agents.langchain_config.LangChainInitializer._initialized", False):
            with patch("app.agents.langchain_config.ChatOpenAI"):
                with patch("app.agents.suggestions_agent.get_redis_client") as mock_redis:
                    mock_redis.return_value = MagicMock()
                    from app.agents.langchain_config import LangChainInitializer
                    LangChainInitializer.initialize()
                    yield

@pytest.mark.asyncio
async def test_parse_suggestions_with_reasoning():
    """Test that reasoning and confidence are correctly parsed."""
    config = AgentConfig()
    
    with patch("app.agents.suggestions_agent.get_redis_client", return_value=MagicMock()):
        agent = SuggestionsAgent(config)
    
    # Call the parsing method directly
    suggestions = agent._parse_suggestions_response(MOCK_RESPONSE)
    
    assert len(suggestions) == 2
    
    # Check first suggestion
    first = suggestions[0]
    assert first.title == "Address Blocking Task"
    assert first.reasoning == "Because this task is blocking 3 other items and the deadline is in 2 days."
    assert first.confidence == 0.95
    assert first.priority == 9
    
    # Check second suggestion
    second = suggestions[1]
    assert second.reasoning == "Because this item is significantly overdue and impacts the design phase."
    assert second.confidence == 0.8

def test_suggestion_model_defaults():
    """Test Suggestion model has correct defaults for new fields."""
    suggestion = Suggestion(
        title="Test Title",
        description="Test Description"
    )
    
    assert suggestion.reasoning is None
    assert suggestion.confidence == 0.8  # Default
    assert suggestion.priority == 1  # Default
