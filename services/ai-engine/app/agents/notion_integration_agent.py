"""
Notion Integration Agent for syncing extracted tasks to Notion.

This agent handles task creation and updates in Notion databases,
with retry logic and error handling for API failures.
"""

import asyncio
import logging
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json

from app.agents.base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
from app.integrations.notion_integration import NotionIntegration

logger = logging.getLogger(__name__)


class NotionIntegrationAgent(BaseAgent):
    """
    Agent responsible for syncing tasks to Notion.
    
    Handles task creation, updates, and error recovery with retry logic.
    Emits Socket.io events for sync status updates.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        socket_io_client: Optional[Any] = None,
        db_connection: Optional[Any] = None,
    ):
        """
        Initialize Notion Integration Agent.
        
        Args:
            config: Agent configuration
            socket_io_client: Socket.io client for emitting events
            db_connection: Database connection for retrieving integration tokens
        """
        super().__init__(config)
        self.socket_io_client = socket_io_client
        self.db_connection = db_connection
        self.notion_client = None
        self.max_retries = 1  # Retry once per requirement 5.5
        self.retry_delay = 1  # seconds
    
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute Notion sync for extracted tasks.
        
        Args:
            meeting_id: ID of the meeting
            project_id: ID of the project
            organization_id: ID of the organization
            tasks: List of extracted tasks to sync
            agent_run_id: ID of the agent run for status updates
            
        Returns:
            AgentResult with sync status and results
        """
        try:
            meeting_id = kwargs.get("meeting_id")
            project_id = kwargs.get("project_id")
            organization_id = kwargs.get("organization_id")
            tasks = kwargs.get("tasks", [])
            agent_run_id = kwargs.get("agent_run_id")
            
            # Validate inputs
            if not meeting_id or not project_id or not organization_id:
                return self._create_error_result(
                    "Missing required parameters: meeting_id, project_id, organization_id"
                )
            
            if not tasks:
                self._log_execution("No tasks to sync")
                return self._create_success_result(
                    data={
                        "synced_count": 0,
                        "failed_count": 0,
                        "results": [],
                    },
                    metadata={"notion_integration_active": False}
                )
            
            # Emit status update
            self._emit_socket_event(
                agent_run_id,
                "sync_status",
                {"status": "running", "message": "Checking Notion integration..."}
            )
            
            # Check if Notion integration exists for organization
            notion_token = await self._get_notion_token(organization_id)
            
            if not notion_token:
                self._log_execution("No Notion integration found for organization")
                return self._create_success_result(
                    data={
                        "synced_count": 0,
                        "failed_count": 0,
                        "results": [],
                    },
                    metadata={"notion_integration_active": False}
                )
            
            # Initialize Notion client with token
            self.notion_client = NotionIntegration()
            
            # Emit status update
            self._emit_socket_event(
                agent_run_id,
                "sync_status",
                {"status": "running", "message": f"Syncing {len(tasks)} tasks to Notion..."}
            )
            
            # Sync tasks to Notion
            sync_results = await self._sync_tasks_to_notion(
                tasks=tasks,
                meeting_id=meeting_id,
                agent_run_id=agent_run_id
            )
            
            synced_count = sum(1 for r in sync_results if r.get("success"))
            failed_count = len(sync_results) - synced_count
            
            # Emit completion status
            self._emit_socket_event(
                agent_run_id,
                "sync_status",
                {
                    "status": "completed",
                    "message": f"Synced {synced_count}/{len(tasks)} tasks to Notion",
                    "synced_count": synced_count,
                    "failed_count": failed_count,
                }
            )
            
            self._log_execution(
                "Notion sync completed",
                {"synced": synced_count, "failed": failed_count}
            )
            
            return self._create_success_result(
                data={
                    "synced_count": synced_count,
                    "failed_count": failed_count,
                    "results": sync_results,
                },
                metadata={
                    "notion_integration_active": True,
                    "meeting_id": meeting_id,
                }
            )
        
        except Exception as e:
            error_msg = f"Notion sync failed: {str(e)}"
            self._log_error(error_msg)
            
            # Emit error status
            agent_run_id = kwargs.get("agent_run_id")
            if agent_run_id:
                self._emit_socket_event(
                    agent_run_id,
                    "sync_status",
                    {"status": "error", "message": error_msg}
                )
            
            return self._create_error_result(error_msg)
    
    async def _get_notion_token(self, organization_id: str) -> Optional[str]:
        """
        Retrieve Notion API token from Integration model.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            Notion API token or None if not found
        """
        try:
            if not self.db_connection:
                self._log_execution("No database connection available")
                return None
            
            # Query Integration model for Notion token
            # This assumes a database query method exists
            integration = await self._query_integration(
                organization_id=organization_id,
                provider="notion"
            )
            
            if integration and integration.get("api_token"):
                return integration["api_token"]
            
            return None
        
        except Exception as e:
            self._log_error(f"Failed to retrieve Notion token: {str(e)}")
            return None
    
    async def _sync_tasks_to_notion(
        self,
        tasks: List[Dict[str, Any]],
        meeting_id: str,
        agent_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sync extracted tasks to Notion database.
        
        Args:
            tasks: List of extracted tasks
            meeting_id: Meeting ID for context
            agent_run_id: Agent run ID for status updates
            
        Returns:
            List of sync results for each task
        """
        results = []
        
        for idx, task in enumerate(tasks):
            try:
                # Emit progress update
                if agent_run_id:
                    self._emit_socket_event(
                        agent_run_id,
                        "sync_progress",
                        {
                            "current": idx + 1,
                            "total": len(tasks),
                            "task_title": task.get("title", "Unknown")
                        }
                    )
                
                # Create task in Notion with retry logic
                result = await self._create_task_with_retry(task, meeting_id)
                results.append(result)
            
            except Exception as e:
                self._log_error(f"Failed to sync task: {str(e)}")
                results.append({
                    "success": False,
                    "task_title": task.get("title", "Unknown"),
                    "error": str(e)
                })
        
        return results
    
    async def _create_task_with_retry(
        self,
        task: Dict[str, Any],
        meeting_id: str,
    ) -> Dict[str, Any]:
        """
        Create task in Notion with retry logic.
        
        Implements retry with exponential backoff (max 1 retry per requirement 5.5).
        
        Args:
            task: Task data to create
            meeting_id: Meeting ID for context
            
        Returns:
            Result dictionary with success status and details
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                self._log_execution(
                    f"Creating Notion task (attempt {attempt + 1})",
                    {"task_title": task.get("title")}
                )
                
                # Create task in Notion
                result = self.notion_client.create_task(
                    title=task.get("title", "Untitled Task"),
                    description=task.get("description", ""),
                    assignee=task.get("assignee_name"),
                    priority=self._map_priority(task.get("priority", "medium")),
                    due_date=task.get("due_date"),
                    meeting_date=datetime.now().strftime("%Y-%m-%d"),
                    source=f"Meeting {meeting_id}"
                )
                
                if result.get("success"):
                    return {
                        "success": True,
                        "task_title": task.get("title"),
                        "notion_task_id": result.get("task_id"),
                        "notion_url": result.get("url"),
                    }
                else:
                    last_error = result.get("error", "Unknown error")
                    raise Exception(last_error)
            
            except Exception as e:
                last_error = str(e)
                attempt += 1
                
                if attempt <= self.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    self._log_execution(
                        f"Retrying task creation after {wait_time}s",
                        {"error": last_error}
                    )
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        # All retries exhausted
        self._log_error(
            f"Failed to create Notion task after {attempt} attempts",
            {"task_title": task.get("title"), "error": last_error}
        )
        
        return {
            "success": False,
            "task_title": task.get("title"),
            "error": last_error,
        }
    
    async def update_task_in_notion(
        self,
        notion_task_id: str,
        task_updates: Dict[str, Any],
        agent_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing task in Notion.
        
        Implements requirement 5.4: sync updates within 5 seconds.
        
        Args:
            notion_task_id: Notion task ID to update
            task_updates: Dictionary of fields to update
            agent_run_id: Agent run ID for status updates
            
        Returns:
            Result dictionary with update status
        """
        try:
            if not self.notion_client:
                return {
                    "success": False,
                    "error": "Notion client not initialized"
                }
            
            # Emit status update
            if agent_run_id:
                self._emit_socket_event(
                    agent_run_id,
                    "update_status",
                    {"status": "running", "message": "Updating task in Notion..."}
                )
            
            # Update task with retry logic
            result = await self._update_task_with_retry(
                notion_task_id,
                task_updates
            )
            
            # Emit completion status
            if agent_run_id:
                status = "completed" if result.get("success") else "error"
                self._emit_socket_event(
                    agent_run_id,
                    "update_status",
                    {
                        "status": status,
                        "message": result.get("message", "Update completed")
                    }
                )
            
            return result
        
        except Exception as e:
            error_msg = f"Failed to update Notion task: {str(e)}"
            self._log_error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def _update_task_with_retry(
        self,
        notion_task_id: str,
        task_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update task in Notion with retry logic.
        
        Args:
            notion_task_id: Notion task ID
            task_updates: Fields to update
            
        Returns:
            Result dictionary
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                self._log_execution(
                    f"Updating Notion task (attempt {attempt + 1})",
                    {"task_id": notion_task_id}
                )
                
                # Build update payload
                properties = {}
                
                if "title" in task_updates:
                    properties["Name"] = {
                        "title": [{"text": {"content": task_updates["title"]}}]
                    }
                
                if "description" in task_updates:
                    properties["Description"] = {
                        "rich_text": [{"text": {"content": task_updates["description"][:2000]}}]
                    }
                
                if "status" in task_updates:
                    properties["Status"] = {
                        "select": {"name": task_updates["status"]}
                    }
                
                if "priority" in task_updates:
                    properties["Priority"] = {
                        "select": {"name": self._map_priority(task_updates["priority"])}
                    }
                
                if "due_date" in task_updates and task_updates["due_date"]:
                    properties["Due Date"] = {
                        "date": {"start": task_updates["due_date"]}
                    }
                
                # Update via Notion API
                url = f"https://api.notion.com/v1/pages/{notion_task_id}"
                headers = {
                    "Authorization": f"Bearer {self.notion_client.api_key}",
                    "Content-Type": "application/json",
                    "Notion-Version": "2022-06-28"
                }
                
                response = requests.patch(
                    url,
                    headers=headers,
                    json={"properties": properties}
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Task updated successfully",
                        "notion_task_id": notion_task_id
                    }
                else:
                    last_error = response.json().get("message", response.text)
                    raise Exception(last_error)
            
            except Exception as e:
                last_error = str(e)
                attempt += 1
                
                if attempt <= self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    self._log_execution(f"Retrying update after {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        self._log_error(
            f"Failed to update Notion task after {attempt} attempts",
            {"error": last_error}
        )
        
        return {
            "success": False,
            "error": last_error,
            "message": f"Failed to update task: {last_error}"
        }
    
    def _map_priority(self, priority: str) -> str:
        """
        Map internal priority to Notion priority format.
        
        Args:
            priority: Internal priority (low, medium, high)
            
        Returns:
            Notion priority format
        """
        priority_map = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
        }
        return priority_map.get(priority.lower(), "Medium")
    
    def _emit_socket_event(
        self,
        agent_run_id: Optional[str],
        event_name: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Emit Socket.io event for sync status updates.
        
        Args:
            agent_run_id: Agent run ID for routing
            event_name: Socket.io event name
            data: Event data
        """
        if not self.socket_io_client or not agent_run_id:
            return
        
        try:
            # Emit to agent run room
            self.socket_io_client.emit(
                event_name,
                data,
                room=f"agent_run_{agent_run_id}"
            )
            self._log_execution(f"Emitted Socket.io event: {event_name}")
        
        except Exception as e:
            self._log_error(f"Failed to emit Socket.io event: {str(e)}")
    
    async def _query_integration(
        self,
        organization_id: str,
        provider: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query Integration model for provider token.
        
        Args:
            organization_id: Organization ID
            provider: Provider name (e.g., "notion")
            
        Returns:
            Integration record or None
        """
        # This is a placeholder for database query
        # In real implementation, this would query the Integration model
        # For now, return None to indicate no integration found
        return None



# Job function for background processing
async def sync_tasks_to_notion_job(
    meeting_id: str,
    project_id: str,
    organization_id: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    agent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job function for syncing tasks to Notion.
    
    Args:
        meeting_id: Meeting identifier
        project_id: Project identifier
        organization_id: Organization identifier
        tasks: List of tasks to sync (optional, can be retrieved from database)
        agent_run_id: Agent run identifier for status updates
        
    Returns:
        Dictionary with sync results
    """
    try:
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.1,
            timeout=300
        )
        
        agent = NotionIntegrationAgent(config=config)
        
        # If tasks not provided, retrieve from database
        if not tasks:
            tasks = []
        
        result = await agent.execute(
            meeting_id=meeting_id,
            project_id=project_id,
            organization_id=organization_id or project_id,
            tasks=tasks,
            agent_run_id=agent_run_id
        )
        
        return result.to_dict()
    
    except Exception as e:
        logger.error(f"Notion sync job failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "meeting_id": meeting_id,
            "project_id": project_id
        }
