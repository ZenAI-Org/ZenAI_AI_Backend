import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.agents.followup_agent import FollowUpAgent
from app.integrations.notion_integration import NotionIntegration

logger = logging.getLogger(__name__)

class SchedulerService:
    """
    Manages background scheduled tasks using APScheduler.
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    def start(self):
        """Start the scheduler if not already running."""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("SchedulerService started.")
            
            # Register jobs
            self.schedule_daily_followup()
    
    def shutdown(self):
        """Shutdown the scheduler."""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("SchedulerService shut down.")

    def schedule_daily_followup(self):
        """
        Schedule the Follow-up Agent to run daily at 9:00 AM.
        """
        # We can configure the time via env vars potentially, defaulting to 9 AM
        trigger = CronTrigger(hour=9, minute=0)
        
        self.scheduler.add_job(
            self.run_followup_job,
            trigger=trigger,
            id="daily_followup",
            replace_existing=True,
            name="Daily Project Follow-up"
        )
        logger.info("Scheduled 'Daily Project Follow-up' for 09:00 AM daily.")

    async def run_followup_job(self):
        """
        The actual job function to be executed.
        """
        logger.info("Executing scheduled job: Daily Project Follow-up")
        try:
            # We need to know WHICH project to run for. 
            # In a real SaaS, we'd iterate over all active projects/organizations.
            # For this single-user/MVP setup, we might iterate all projects in the Notion DB 
            # or just run for a default one if configured.
            
            # Option 1: Query Notion for all unique projects (if we had that structure)
            # Option 2: Just run the agent which queries ALL tasks from the connected DB.
            # The FollowUpAgent currently takes a project_id but its `detect_inactive_tasks` 
            # actually fetches ALL tasks from the `notion_integration` which is tied to one DB.
            # So `project_id` is mostly for context retrieval scope and reporting.
            
            # Let's assume a default project ID or iterate if we can.
            # For now, we'll use a placeholder "default-project" which matches the single Notion DB context.
            
            agent = FollowUpAgent()
            result = await agent.execute(project_id="default-project")
            
            logger.info(f"Scheduled job completed: {result.status}")
            if result.status == "error":
                logger.error(f"Job failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error in scheduled job: {e}")
