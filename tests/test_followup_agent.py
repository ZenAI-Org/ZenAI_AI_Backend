import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.getcwd())

from app.agents.followup_agent import FollowUpAgent

async def test_followup_agent():
    print("=== Testing FollowUpAgent ===")
    
    # Mock dependencies
    agent = FollowUpAgent()
    
    # Mock Notion Integration
    agent.notion = MagicMock()
    
    # Create sample tasks
    now = datetime.utcnow()
    tasks = [
        {
            "id": "1",
            "title": "Active Task",
            "status": "In Progress",
            "last_edited_time": (now - timedelta(days=1)).isoformat() + "Z", # Active
            "assignee_email": "active@example.com",
            "assignee_name": "Active User"
        },
        {
            "id": "2",
            "title": "Inactive Task",
            "status": "In Progress",
            "last_edited_time": (now - timedelta(days=5)).isoformat() + "Z", # Inactive > 3 days
            "assignee_email": "inactive@example.com",
            "assignee_name": "Inactive User"
        },
        {
            "id": "3",
            "title": "Blocked Task",
            "status": "In Progress",
            "last_edited_time": (now - timedelta(days=10)).isoformat() + "Z", # Inactive
            "assignee_email": "blocked@example.com",
            "assignee_name": "Blocked User"
        }
    ]
    
    agent.notion.query_all_tasks_with_emails.return_value = tasks
    
    # Mock Context Retriever
    agent.context_retriever = AsyncMock()
    # Return context finding a "blocker" for task 3
    async def mock_search(project_id, query, limit):
        if "Blocked Task" in query:
            return {
                "chunks": [{"text": "We are facing a blocker on the Blocked Task due to API limits."}]
            }
        return {"chunks": []}
        
    agent.context_retriever.search_context.side_effect = mock_search
    
    # Mock Email Service
    agent.email_service = MagicMock()
    agent.email_service.send_email = MagicMock(return_value=True)
    agent.email_service.send_daily_report = MagicMock(return_value=True)
    agent.email_service.from_email = "admin@example.com"

    # Run the agent
    print("\nRunning daily followup...")
    result = await agent.run_daily_followup(project_id="test-project")
    
    print("\n=== Results ===")
    print(f"Status: {result.status}")
    if result.status == "success":
        data = result.data
        print(f"Processed Tasks: {len(tasks)}")
        print(f"Inactive Tasks Detected: {len(data['inactive_tasks'])}")
        print(f"Inactive Task Titles: {data['inactive_tasks']}")
        print(f"Nudges Sent: {data['nudges_sent']}")
        print(f"Report Sent: {data['report_sent']}")
        
        # Validation
        assert len(data['inactive_tasks']) == 2, "Should detect 2 inactive tasks"
        assert "Inactive Task" in data['inactive_tasks']
        assert "Blocked Task" in data['inactive_tasks']
        assert data['nudges_sent'] == 2, "Should send 2 nudges"
        assert data['report_sent'] == True, "Should send report"
        
        # Verify contextual Nudge
        # Check if email service called with blocker text for task 3
        calls = agent.email_service.send_email.call_args_list
        blocked_call = None
        for call in calls:
            args = call[0] # (to, subject, body)
            if "Blockers" in args[1]: # Subject check
                blocked_call = args
        
        if blocked_call:
            print("\n[SUCCESS] Contextual blocker nudge detected!")
            print(f"Subject: {blocked_call[1]}")
            print(f"Body snippet: {blocked_call[2][:100]}...")
        else:
            print("\n[FAIL] Contextual blocker nudge NOT detected")
            
    else:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_followup_agent())
