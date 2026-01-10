import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.core.embeddings import EmbeddingStore
from app.integrations.notion_integration import NotionIntegration

logger = logging.getLogger(__name__)

class RepetitionDetector:
    """
    Detects recurring topics across meetings and flags potential stall points.
    """
    
    def __init__(self, db_connection, notion_integration: Optional[NotionIntegration] = None):
        self.db = db_connection
        self.embedding_store = EmbeddingStore(db_connection)
        self.notion = notion_integration
        
    def detect_recurring_topics(
        self, 
        project_id: str, 
        current_topics: List[str], 
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Identify which of the current topics have been discussed frequently 
        in the recent past without resolution.
        
        Args:
            project_id: The project identifier.
            current_topics: List of topic strings from the current meeting.
            lookback_days: How far back to check (default 30 days).
            
        Returns:
            List of recurring topic insights.
        """
        if not current_topics:
            return []
            
        recurring_issues = []
        
        # 1. Fetch completed tasks only once if notion is available
        # We'll use this to check if a topic might have been "done"
        completed_tasks_text = []
        if self.notion:
            try:
                # Optimized: ideally we'd have a better query, but for now filtering in python
                all_tasks = self.notion.query_all_tasks_with_emails()
                completed_tasks_text = [
                    f"{t['title']} {t.get('description', '')}".lower() 
                    for t in all_tasks 
                    if t.get('status') == 'Done'
                ]
            except Exception as e:
                logger.warning(f"Could not fetch Notion tasks for repetition check: {e}")
        
        # 2. Check each topic
        for topic in current_topics:
            try:
                # Search for similar topics in vector DB
                # content_type="topic"
                similar_topics = self.embedding_store.semantic_search(
                    project_id=project_id,
                    query=topic,
                    content_type="topic",
                    limit=20, # Fetch enough to filter by date
                    similarity_threshold=0.85
                )
                
                # Filter by date and uniqueness
                recent_meetings = set()
                earliest_date = None
                
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                
                for match in similar_topics:
                    created_at_str = match.get("created_at")
                    if not created_at_str:
                        continue
                        
                    try:
                        # created_at is likely ISO string from previous step
                        created_date = datetime.fromisoformat(created_at_str)
                        
                        if created_date > cutoff_date:
                            # Use metadata to get meeting_id to ensure we count unique meetings
                            meeting_id = match.get("metadata", {}).get("meeting_id")
                            if meeting_id:
                                recent_meetings.add(meeting_id)
                                
                                if earliest_date is None or created_date < earliest_date:
                                    earliest_date = created_date
                    except ValueError:
                        continue

                # 3. Decision Logic
                # If discussed in at least 2 PREVIOUS meetings (so 3 total including now)
                if len(recent_meetings) >= 2:
                    
                    # Check if solved
                    is_solved = False
                    for task_text in completed_tasks_text:
                        # Heuristic: simple substring match
                        if topic.lower() in task_text:
                            is_solved = True
                            break
                    
                    if not is_solved:
                        days_since_first = (datetime.now() - earliest_date).days if earliest_date else 0
                        
                        recurring_issues.append({
                            "topic": topic,
                            "recurrence_count": len(recent_meetings),
                            "first_discussed_days_ago": days_since_first,
                            "status": "recurring_unsolved",
                            "insight": f"Topic '{topic}' has been discussed in {len(recent_meetings)} recent meetings but no tasks are closed."
                        })
                        
            except Exception as e:
                logger.error(f"Error checking repetition for topic '{topic}': {e}")
                continue
                
        return recurring_issues

    def store_topics(self, project_id: str, topics: List[str], meeting_id: str):
        """
        Store extracted topics into the vector database for future reference.
        """
        if not topics:
            return

        try:
            items = []
            for i, topic in enumerate(topics):
                items.append({
                    "content_id": f"{meeting_id}_topic_{i}",
                    "text": topic,
                    "metadata": {
                        "meeting_id": meeting_id,
                        "date": datetime.now().isoformat(),
                        "type": "topic"
                    }
                })
            
            # Use batch storage
            self.embedding_store.store_embeddings_batch(
                project_id=project_id,
                content_type="topic",
                items=items
            )
            logger.info(f"Stored {len(topics)} topics for meeting {meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to store topics: {e}")
