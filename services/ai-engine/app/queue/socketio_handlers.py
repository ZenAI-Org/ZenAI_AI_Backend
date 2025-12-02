"""
Socket.io event handlers for connection management and authentication.
Handles client connections, disconnections, and project subscriptions.
"""

import logging
from typing import Optional, Dict, Any

try:
    import socketio
except ImportError:
    socketio = None

from app.queue.socketio_manager import get_socketio_manager
from app.core.logger import get_logger

logger = get_logger(__name__)


class SocketIOEventHandlers:
    """
    Manages Socket.io event handlers for real-time communication.
    Handles connection lifecycle and project subscriptions.
    """
    
    def __init__(self, sio: Optional['socketio.AsyncServer'] = None):
        """
        Initialize Socket.io event handlers.
        
        Args:
            sio: Optional socketio.AsyncServer instance
        """
        self.sio = sio
        self.socketio_manager = get_socketio_manager()
        
        if self.sio:
            self._register_handlers()
        
        logger.info("Socket.io event handlers initialized")
    
    def set_socketio_server(self, sio: 'socketio.AsyncServer') -> None:
        """
        Set the Socket.io server instance and register handlers.
        
        Args:
            sio: socketio.AsyncServer instance
        """
        self.sio = sio
        self.socketio_manager.set_socketio_server(sio)
        self._register_handlers()
        logger.info("Socket.io server set and handlers registered")
    
    def _register_handlers(self) -> None:
        """Register all Socket.io event handlers."""
        if not self.sio:
            return
        
        # Connection handlers
        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        
        # Project subscription handlers
        self.sio.on("subscribe_project", self._on_subscribe_project)
        self.sio.on("unsubscribe_project", self._on_unsubscribe_project)
        
        logger.info("Socket.io event handlers registered")
    
    async def _on_connect(self, sid: str, environ: Dict[str, Any]) -> None:
        """
        Handle client connection.
        
        Args:
            sid: Socket.io session ID
            environ: WSGI environ dict
        """
        try:
            # Extract authentication token from query parameters
            query_string = environ.get("QUERY_STRING", "")
            token = self._extract_token_from_query(query_string)
            
            if not token:
                logger.warning(f"Connection attempt without token: {sid}")
                # Optionally reject connection without token
                # await self.sio.disconnect(sid)
                # return
            
            # Validate token (implement your auth logic here)
            user_id = await self._validate_token(token) if token else None
            
            if token and not user_id:
                logger.warning(f"Invalid token for connection: {sid}")
                await self.sio.disconnect(sid)
                return
            
            # Store user info in session
            await self.sio.save_session(sid, {"user_id": user_id, "token": token})
            
            logger.info(f"Client connected: {sid} (user: {user_id})")
        
        except Exception as e:
            logger.error(f"Error in connect handler: {e}")
            try:
                await self.sio.disconnect(sid)
            except Exception as disconnect_error:
                logger.error(f"Error disconnecting client: {disconnect_error}")
    
    async def _on_disconnect(self, sid: str) -> None:
        """
        Handle client disconnection.
        
        Args:
            sid: Socket.io session ID
        """
        try:
            # Get session data
            session = await self.sio.get_session(sid)
            user_id = session.get("user_id") if session else None
            
            # Unsubscribe from all projects
            # (In a real implementation, you'd track which projects the client was subscribed to)
            
            logger.info(f"Client disconnected: {sid} (user: {user_id})")
        
        except Exception as e:
            logger.error(f"Error in disconnect handler: {e}")
    
    async def _on_subscribe_project(
        self,
        sid: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Handle project subscription request.
        
        Args:
            sid: Socket.io session ID
            data: Subscription data with project_id
        """
        try:
            project_id = data.get("project_id")
            
            if not project_id:
                await self.sio.emit(
                    "error",
                    {"message": "project_id is required"},
                    to=sid,
                )
                return
            
            # Get session data
            session = await self.sio.get_session(sid)
            user_id = session.get("user_id") if session else None
            
            # Validate user has access to project
            # (Implement your authorization logic here)
            has_access = await self._validate_project_access(user_id, project_id)
            
            if not has_access:
                await self.sio.emit(
                    "error",
                    {"message": f"Access denied to project {project_id}"},
                    to=sid,
                )
                logger.warning(
                    f"Access denied for user {user_id} to project {project_id}"
                )
                return
            
            # Register client for project
            self.socketio_manager.register_client_for_project(project_id, sid)
            
            # Join Socket.io room for project
            await self.sio.enter_room(sid, f"project_{project_id}")
            
            # Send confirmation
            await self.sio.emit(
                "subscribed",
                {"project_id": project_id},
                to=sid,
            )
            
            logger.info(f"Client {sid} subscribed to project {project_id}")
        
        except Exception as e:
            logger.error(f"Error in subscribe_project handler: {e}")
            await self.sio.emit(
                "error",
                {"message": f"Subscription failed: {str(e)}"},
                to=sid,
            )
    
    async def _on_unsubscribe_project(
        self,
        sid: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Handle project unsubscription request.
        
        Args:
            sid: Socket.io session ID
            data: Unsubscription data with project_id
        """
        try:
            project_id = data.get("project_id")
            
            if not project_id:
                await self.sio.emit(
                    "error",
                    {"message": "project_id is required"},
                    to=sid,
                )
                return
            
            # Unregister client for project
            self.socketio_manager.unregister_client_for_project(project_id, sid)
            
            # Leave Socket.io room for project
            await self.sio.leave_room(sid, f"project_{project_id}")
            
            # Send confirmation
            await self.sio.emit(
                "unsubscribed",
                {"project_id": project_id},
                to=sid,
            )
            
            logger.info(f"Client {sid} unsubscribed from project {project_id}")
        
        except Exception as e:
            logger.error(f"Error in unsubscribe_project handler: {e}")
            await self.sio.emit(
                "error",
                {"message": f"Unsubscription failed: {str(e)}"},
                to=sid,
            )
    
    def _extract_token_from_query(self, query_string: str) -> Optional[str]:
        """
        Extract authentication token from query string.
        
        Args:
            query_string: WSGI query string
            
        Returns:
            Token or None
        """
        try:
            # Parse query string for token parameter
            # Example: "token=abc123&other=value"
            params = {}
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
            
            return params.get("token")
        
        except Exception as e:
            logger.error(f"Error extracting token: {e}")
            return None
    
    async def _validate_token(self, token: str) -> Optional[str]:
        """
        Validate authentication token.
        
        Args:
            token: Authentication token
            
        Returns:
            User ID if valid, None otherwise
        """
        try:
            # Implement your token validation logic here
            # This is a placeholder that accepts any non-empty token
            # In production, validate against JWT, session store, etc.
            
            if not token or not token.strip():
                return None
            
            # TODO: Implement actual token validation
            # For now, extract user_id from token or session
            # Example: decode JWT token and extract user_id
            
            # Placeholder: assume token format is "user_id:token_hash"
            if ":" in token:
                user_id = token.split(":")[0]
                return user_id if user_id else None
            
            return None
        
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None
    
    async def _validate_project_access(
        self,
        user_id: Optional[str],
        project_id: str,
    ) -> bool:
        """
        Validate user has access to project.
        
        Args:
            user_id: User ID
            project_id: Project ID
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Implement your authorization logic here
            # This is a placeholder that allows all authenticated users
            
            if not user_id:
                # Allow unauthenticated access for now
                # In production, require authentication
                return True
            
            # TODO: Implement actual authorization check
            # Check if user is member of project, has role, etc.
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating project access: {e}")
            return False


# Global Socket.io event handlers instance
_socketio_handlers = None


def get_socketio_handlers() -> SocketIOEventHandlers:
    """Get or create global Socket.io event handlers."""
    global _socketio_handlers
    if _socketio_handlers is None:
        _socketio_handlers = SocketIOEventHandlers()
    return _socketio_handlers
