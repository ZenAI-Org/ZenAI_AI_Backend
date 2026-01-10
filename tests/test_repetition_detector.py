import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from app.services.repetition_detector import RepetitionDetector

@pytest.fixture
def mock_embedding_store():
    store = MagicMock()
    return store

@pytest.fixture
def mock_notion():
    notion = MagicMock()
    notion.query_all_tasks_with_emails.return_value = [
        {"title": "Fix Login Bug", "status": "Done"},
        {"title": "Update API", "status": "In Progress"}
    ]
    return notion

def test_detect_recurring_unsolved(mock_embedding_store, mock_notion):
    with patch("app.services.repetition_detector.EmbeddingStore", return_value=mock_embedding_store):
        detector = RepetitionDetector(db_connection=None, notion_integration=mock_notion)
    
    # Mock search results for "API Latency"
    # 2 previous meetings in last 30 days
    now = datetime.now()
    mock_embedding_store.semantic_search.return_value = [
        # Match 1: 5 days ago
        {
            "metadata": {"meeting_id": "m1", "date": (now - timedelta(days=5)).isoformat()},
            "created_at": (now - timedelta(days=5)).isoformat()
        },
        # Match 2: 10 days ago
        {
            "metadata": {"meeting_id": "m2", "date": (now - timedelta(days=10)).isoformat()},
            "created_at": (now - timedelta(days=10)).isoformat()
        }
    ]
    
    issues = detector.detect_recurring_topics(
        project_id="p1", 
        current_topics=["API Latency"]
    )
    
    assert len(issues) == 1
    assert issues[0]["topic"] == "API Latency"
    assert issues[0]["status"] == "recurring_unsolved"
    assert issues[0]["recurrence_count"] == 2

def test_detect_recurring_solved(mock_embedding_store, mock_notion):
    with patch("app.services.repetition_detector.EmbeddingStore", return_value=mock_embedding_store):
        detector = RepetitionDetector(db_connection=None, notion_integration=mock_notion)
    
    # Mock search results for "Fix Login Bug" (which is Done in Notion)
    now = datetime.now()
    mock_embedding_store.semantic_search.return_value = [
        {
            "metadata": {"meeting_id": "m1", "date": (now - timedelta(days=5)).isoformat()},
            "created_at": (now - timedelta(days=5)).isoformat()
        },
        {
            "metadata": {"meeting_id": "m2", "date": (now - timedelta(days=10)).isoformat()},
            "created_at": (now - timedelta(days=10)).isoformat()
        }
    ]
    
    issues = detector.detect_recurring_topics(
        project_id="p1", 
        current_topics=["Fix Login Bug"]
    )
    
    # Should be empty because it matched a "Done" task
    assert len(issues) == 0

def test_no_recurrence(mock_embedding_store, mock_notion):
    with patch("app.services.repetition_detector.EmbeddingStore", return_value=mock_embedding_store):
        detector = RepetitionDetector(db_connection=None, notion_integration=mock_notion)
    
    # No past matches
    mock_embedding_store.semantic_search.return_value = []
    
    issues = detector.detect_recurring_topics(
        project_id="p1", 
        current_topics=["New Feature"]
    )
    
    assert len(issues) == 0
