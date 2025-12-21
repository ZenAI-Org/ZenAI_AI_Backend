from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio

from app.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from app.integrations.notion_integration import NotionIntegration
from app.services.email_service import EmailService
from app.core.logger import get_logger
from app.models.followup import FollowUpReport, InactiveTask, TaskContext

# Import context retriever if available, otherwise mock or optional
try:
    from app.core.context_retriever import ContextRetriever
except ImportError:
    ContextRetriever = None

logger = get_logger(__name__)

class FollowUpAgent(BaseAgent):
    """
    Agent responsible for monitoring project tasks, checking for inactivity,
    analyzing context for stalled tasks, and sending automated nudges.
    """
    
    def __init__(self, config: AgentConfig = None, db_connection=None):
        # Allow init without config for simpler usage if needed, or create default
        if config is None:
            config = AgentConfig()
            
        super().__init__(config)
        
        self.notion = NotionIntegration()
        self.email_service = EmailService()
        self.db_connection = db_connection
        self.context_retriever = ContextRetriever(db_connection) if (ContextRetriever and db_connection) else None
        
        # Thresholds
        self.inactivity_days_threshold = 3
        
    async def execute(self, **kwargs) -> AgentResult:
        """
        Main execution method for the agent.
        Expected kwargs: project_id, check_type (default: daily_followup)
        """
        project_id = kwargs.get("project_id")
        check_type = kwargs.get("check_type", "daily_followup")
        
        logger.info(f"Starting FollowUpAgent run. Type: {check_type}")
        
        if check_type == "daily_followup":
            return await self.run_daily_followup(project_id)
        else:
            return AgentResult(status="error", error=f"Unknown check_type: {check_type}")

    async def run_daily_followup(self, project_id: Optional[str] = None) -> AgentResult:
        """
        Orchestrate the daily follow-up process:
        1. Detect inactive tasks
        2. Analyze context
        3. Send nudges
        4. Generate report
        """
        # Track metrics for report
        total_tasks_checked = 0
        nudges_sent = 0
        detected_inactive = []
        
        try:
            # 1. Fetch all tasks
            tasks = await asyncio.to_thread(self.notion.query_all_tasks_with_emails)
            total_tasks_checked = len(tasks)
            logger.info(f"Fetched {len(tasks)} tasks from Notion")
            
            # 2. Detect inactive tasks (returns list of dicts for now, we convert to model later)
            raw_inactive_tasks = self.detect_inactive_tasks(tasks)
            logger.info(f"Detected {len(raw_inactive_tasks)} inactive tasks")
            
            # 3. Process each inactive task
            for task in raw_inactive_tasks:
                # Analyze context
                context_dict = await self.analyze_task_context(task, project_id)
                
                # Convert to Pydantic models
                context_model = TaskContext(
                    is_blocked=context_dict.get("is_blocked", False),
                    notes=context_dict.get("notes", [])
                )
                
                inactive_model = InactiveTask(
                    task_id=task.get("id"),
                    title=task.get("title"),
                    assignee=task.get("assignee_name"),
                    days_inactive=task.get("days_inactive"),
                    context=context_model,
                    status=task.get("status")
                )
                
                detected_inactive.append(inactive_model)
                
                # Send nudge if appropriate
                if task.get('assignee_email'):
                    sent = await self.send_nudge(task, context_dict)
                    if sent:
                        nudges_sent += 1
            
            # 4. Generate Report object
            report = FollowUpReport(
                project_id=project_id or "default",
                total_tasks_checked=total_tasks_checked,
                inactive_tasks_found=len(detected_inactive),
                nudges_sent=nudges_sent,
                inactive_task_details=detected_inactive
            )
            
            # Email report
            admin_email = self.email_service.from_email 
            if admin_email:
                report_md = self.generate_report_markdown_from_model(report)
                sent = await asyncio.to_thread(
                    self.email_service.send_daily_report, 
                    report_md, 
                    admin_email
                )
            
            return AgentResult(status="success", data=report.dict())
            
        except Exception as e:
            logger.error(f"FollowUpAgent run failed: {e}")
            return AgentResult(status="error", error=str(e))
            
        except Exception as e:
            logger.error(f"FollowUpAgent run failed: {e}")
            return AgentResult(status="error", error=str(e))

    def detect_inactive_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """
        Identify tasks that haven't been edited in X days and are not 'Done'.
        """
        inactive = []
        now = datetime.utcnow()
        
        for task in tasks:
            status = task.get("status", "").lower()
            if status in ["done", "completed", "verified"]:
                continue
                
            last_edited_str = task.get("last_edited_time")
            if not last_edited_str:
                continue
                
            # Time format from Notion: 2023-10-25T12:00:00.000Z
            # Clean up Z for simpler parsing if needed, or usage isoformat
            try:
                last_edited_str = last_edited_str.replace("Z", "+00:00")
                last_edited = datetime.fromisoformat(last_edited_str)
                
                # Make naive for comparison if needed, or ensure both aware
                last_edited = last_edited.replace(tzinfo=None)
                
                diff = now - last_edited
                if diff.days >= self.inactivity_days_threshold:
                    task['days_inactive'] = diff.days
                    inactive.append(task)
            except ValueError as e:
                logger.warning(f"Could not parse date {last_edited_str}: {e}")
                
        return inactive

    async def analyze_task_context(self, task: Dict, project_id: Optional[str]) -> Dict:
        """
        Use ContextRetriever (pgvector) to finding mentions of this task in recent meetings.
        Returns a dict with flags like 'is_blocked', 'blockers_mentioned', etc.
        """
        if not self.context_retriever or not project_id:
            return {"context_found": False}
        
        query = f"blockers risks delays regarding {task.get('title')}"
        
        results = await self.context_retriever.search_context(
            project_id=project_id,
            query=query,
            limit=3
        )
        
        # Simple keyword analysis on retrieved chunks
        is_blocked = False
        blocker_notes = []
        
        for chunk in results.get('chunks', []):
            text = chunk.get('text', '').lower()
            if 'block' in text or 'risk' in text or 'delay' in text:
                is_blocked = True
                blocker_notes.append(chunk.get('text'))
                
        return {
            "context_found": True,
            "is_blocked": is_blocked,
            "notes": blocker_notes[:2] # Limit context
        }

    async def send_nudge(self, task: Dict, context: Dict) -> bool:
        """
        Send an email nudge. Content depends on whether task seems blocked.
        """
        assignee = task.get("assignee_name", "Team Member")
        email = task.get("assignee_email")
        title = task.get("title")
        days = task.get("days_inactive")
        
        if not email:
            return False
            
        if context.get("is_blocked"):
            # Send "Need Help?" email
            subject = f"Check-in: Blockers on '{title}'?"
            body_text = (
                f"Hi {assignee},\n\n"
                f"I noticed '{title}' hasn't been updated in {days} days.\n"
                f"Recent meetings mentioned potential blockers: {context.get('notes')}\n\n"
                f"Do you need any support to move this forward?\n\n"
                f"ZenAI Project Manager"
            )
            # In a real app we'd use a nice HTML template similar to EmailService types
            # Here we reuse simple text or call a specific EmailService method if we added one.
            # actually let's use the generic send_email for now.
            return await asyncio.to_thread(
                self.email_service.send_email,
                [email],
                subject,
                body_text
            )
        else:
            # Send standard nudge
            subject = f"Update reminder: '{title}'"
            body_text = (
                f"Hi {assignee},\n\n"
                f"Just a friendly nudge! '{title}' hasn't been updated in {days} days.\n"
                f"Please update the status or let us know if you're stuck.\n\n"
                f"ZenAI Project Manager"
            )
            return await asyncio.to_thread(
                self.email_service.send_email,
                [email],
                subject,
                body_text
            )

    def generate_report_markdown_from_model(self, report: FollowUpReport) -> str:
        """
        Generate a daily health report from the FollowUpReport model.
        """
        lines = [
            "# Daily Project Health Report",
            f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d')}",
            "",
            "## Overview",
            f"- **Total Tasks Checked**: {report.total_tasks_checked}",
            f"- **Inactive Tasks**: {report.inactive_tasks_found} (stalled > 3 days)",
            f"- **Nudges Sent**: {report.nudges_sent}",
            "",
            "## Attention Required",
        ]
        
        if report.inactive_task_details:
            for t in report.inactive_task_details:
                risk_label = "🔴 BLOCKED" if t.context.is_blocked else "🟡 STALLED"
                lines.append(f"- {risk_label} **{t.title}** ({t.assignee or 'Unassigned'}) - {t.days_inactive} days inactive")
        else:
            lines.append("No critical issues detected. Keep up the momentum!")
            
        return "\n".join(lines)
