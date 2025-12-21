from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TaskContext(BaseModel):
    """Context analysis for a task."""
    is_blocked: bool = Field(default=False, description="Whether the task appears blocked based on context")
    notes: List[str] = Field(default_factory=list, description="Relevant meeting notes or snippets")

class InactiveTask(BaseModel):
    """Model for a detected inactive task."""
    task_id: str
    title: str
    assignee: Optional[str] = None
    days_inactive: int
    context: TaskContext
    status: str

class FollowUpReport(BaseModel):
    """Model for the daily follow-up report."""
    project_id: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    total_tasks_checked: int
    inactive_tasks_found: int
    nudges_sent: int
    inactive_task_details: List[InactiveTask] = Field(default_factory=list)
