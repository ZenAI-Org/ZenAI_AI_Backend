"""
Socket.io manager for real-time updates on agent status changes.
Handles event emission for job status transitions and progress updates.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

try:
    import socketio
except ImportError:
    socketio = None

from app.queue.job_listeners import JobEventType, JobEventListener

logger = logging.getLogger(__name__)


class SocketIOManager:
    """
    Manages Socket.io connections and real-time event emission.
    Integrates with job event listener to emit real-time updates.
    """
    
    def __init__(self, sio: Optional['socketio.AsyncServer'] = None):
        """
        Initialize Socket.io manager.
        
        Args:
            sio: Optional socketio.AsyncServer instance
        """
        self.sio = sio
        self._connected_clients: Dict[str, set] = {}  # project_id -> set of client sids
        self._job_to_project: Dict[str, str] = {}  # job_id -> project_id
        
        logger.info("Socket.io manager initialized")
    
    def set_socketio_server(self, sio: 'socketio.AsyncServer') -> None:
        """
        Set the Socket.io server instance.
        
        Args:
            sio: socketio.AsyncServer instance
        """
        self.sio = sio
        logger.info("Socket.io server set")
    
    def register_job_listeners(self, event_listener: JobEventListener) -> None:
        """
        Register Socket.io callbacks with job event listener.
        
        Args:
            event_listener: JobEventListener instance
        """
        if not self.sio:
            logger.warning("Socket.io server not set, skipping listener registration")
            return
        
        # Register callbacks for each event type
        event_listener.register_listener(
            JobEventType.QUEUED,
            self._on_job_queued,
        )
        event_listener.register_listener(
            JobEventType.ACTIVE,
            self._on_job_active,
        )
        event_listener.register_listener(
            JobEventType.COMPLETED,
            self._on_job_completed,
        )
        event_listener.register_listener(
            JobEventType.FAILED,
            self._on_job_failed,
        )
        event_listener.register_listener(
            JobEventType.PROGRESS,
            self._on_job_progress,
        )
        
        logger.info("Socket.io listeners registered with job event listener")
    
    def track_job_for_project(self, job_id: str, project_id: str) -> None:
        """
        Track a job for a specific project.
        
        Args:
            job_id: Job ID
            project_id: Project ID
        """
        self._job_to_project[job_id] = project_id
        logger.debug(f"Tracking job {job_id} for project {project_id}")
    
    def register_client_for_project(self, project_id: str, sid: str) -> None:
        """
        Register a client connection for a project.
        
        Args:
            project_id: Project ID
            sid: Socket.io session ID
        """
        if project_id not in self._connected_clients:
            self._connected_clients[project_id] = set()
        
        self._connected_clients[project_id].add(sid)
        logger.debug(f"Client {sid} registered for project {project_id}")
    
    def unregister_client_for_project(self, project_id: str, sid: str) -> None:
        """
        Unregister a client connection for a project.
        
        Args:
            project_id: Project ID
            sid: Socket.io session ID
        """
        if project_id in self._connected_clients:
            self._connected_clients[project_id].discard(sid)
            
            if not self._connected_clients[project_id]:
                del self._connected_clients[project_id]
        
        logger.debug(f"Client {sid} unregistered for project {project_id}")
    
    def _on_job_queued(self, job_id: str, event_data: Dict[str, Any]) -> None:
        """
        Handle job queued event.
        
        Args:
            job_id: Job ID
            event_data: Event data
        """
        if not self.sio:
            return
        
        project_id = self._job_to_project.get(job_id)
        if not project_id:
            logger.debug(f"No project tracked for job {job_id}")
            return
        
        event_payload = {
            "job_id": job_id,
            "status": "queued",
            "timestamp": datetime.now().isoformat(),
            "event": "status_change",
        }
        
        self._emit_to_project(project_id, "agent_status", event_payload)
        logger.info(f"Emitted queued event for job {job_id} to project {project_id}")
    
    def _on_job_active(self, job_id: str, event_data: Dict[str, Any]) -> None:
        """
        Handle job active event.
        
        Args:
            job_id: Job ID
            event_data: Event data
        """
        if not self.sio:
            return
        
        project_id = self._job_to_project.get(job_id)
        if not project_id:
            logger.debug(f"No project tracked for job {job_id}")
            return
        
        event_payload = {
            "job_id": job_id,
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "event": "status_change",
        }
        
        self._emit_to_project(project_id, "agent_status", event_payload)
        logger.info(f"Emitted active event for job {job_id} to project {project_id}")
    
    def _on_job_completed(self, job_id: str, event_data: Dict[str, Any]) -> None:
        """
        Handle job completed event.
        
        Args:
            job_id: Job ID
            event_data: Event data
        """
        if not self.sio:
            return
        
        project_id = self._job_to_project.get(job_id)
        if not project_id:
            logger.debug(f"No project tracked for job {job_id}")
            return
        
        result = event_data.get("result")
        
        event_payload = {
            "job_id": job_id,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "event": "status_change",
            "result": result,
        }
        
        self._emit_to_project(project_id, "agent_status", event_payload)
        logger.info(f"Emitted completed event for job {job_id} to project {project_id}")
    
    def _on_job_failed(self, job_id: str, event_data: Dict[str, Any]) -> None:
        """
        Handle job failed event.
        
        Args:
            job_id: Job ID
            event_data: Event data
        """
        if not self.sio:
            return
        
        project_id = self._job_to_project.get(job_id)
        if not project_id:
            logger.debug(f"No project tracked for job {job_id}")
            return
        
        error = event_data.get("error")
        
        event_payload = {
            "job_id": job_id,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "event": "status_change",
            "error": error,
        }
        
        self._emit_to_project(project_id, "agent_status", event_payload)
        logger.error(f"Emitted failed event for job {job_id} to project {project_id}: {error}")
    
    def _on_job_progress(self, job_id: str, event_data: Dict[str, Any]) -> None:
        """
        Handle job progress event.
        
        Args:
            job_id: Job ID
            event_data: Event data
        """
        if not self.sio:
            return
        
        project_id = self._job_to_project.get(job_id)
        if not project_id:
            logger.debug(f"No project tracked for job {job_id}")
            return
        
        progress = event_data.get("progress", 0)
        message = event_data.get("message")
        
        event_payload = {
            "job_id": job_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
            "event": "progress_update",
            "message": message,
        }
        
        self._emit_to_project(project_id, "agent_progress", event_payload)
        logger.debug(f"Emitted progress event for job {job_id}: {progress}%")
    
    def _emit_to_project(
        self,
        project_id: str,
        event_name: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Emit an event to all clients connected to a project.
        
        Args:
            project_id: Project ID
            event_name: Socket.io event name
            data: Event data
        """
        if not self.sio:
            return
        
        # Get all connected clients for this project
        client_sids = self._connected_clients.get(project_id, set())
        
        if not client_sids:
            logger.debug(f"No connected clients for project {project_id}")
            return
        
        # Emit to each client
        for sid in client_sids:
            try:
                # Use asyncio to emit without blocking
                asyncio.create_task(
                    self.sio.emit(
                        event_name,
                        data,
                        to=sid,
                        namespace=f"/projects/{project_id}",
                    )
                )
            except Exception as e:
                logger.error(f"Error emitting to client {sid}: {e}")
    
    async def emit_to_project_async(
        self,
        project_id: str,
        event_name: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Asynchronously emit an event to all clients connected to a project.
        
        Args:
            project_id: Project ID
            event_name: Socket.io event name
            data: Event data
        """
        if not self.sio:
            return
        
        # Get all connected clients for this project
        client_sids = self._connected_clients.get(project_id, set())
        
        if not client_sids:
            logger.debug(f"No connected clients for project {project_id}")
            return
        
        # Emit to each client
        for sid in client_sids:
            try:
                await self.sio.emit(
                    event_name,
                    data,
                    to=sid,
                    namespace=f"/projects/{project_id}",
                )
            except Exception as e:
                logger.error(f"Error emitting to client {sid}: {e}")


# Global Socket.io manager instance
_socketio_manager = None


def get_socketio_manager() -> SocketIOManager:
    """Get or create global Socket.io manager."""
    global _socketio_manager
    if _socketio_manager is None:
        _socketio_manager = SocketIOManager()
    return _socketio_manager
