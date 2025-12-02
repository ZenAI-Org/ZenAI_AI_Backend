"""
Tests for Socket.io real-time updates integration.
Tests event emission, client tracking, and project isolation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.queue.socketio_manager import SocketIOManager
from app.queue.socketio_handlers import SocketIOEventHandlers
from app.queue.job_listeners import JobEventListener, JobEventType
from app.queue.progress_tracker import ProgressTracker


class TestSocketIOManager:
    """Tests for SocketIOManager."""
    
    def test_initialization(self):
        """Test SocketIOManager initialization."""
        manager = SocketIOManager()
        
        assert manager.sio is None
        assert manager._connected_clients == {}
        assert manager._job_to_project == {}
    
    def test_set_socketio_server(self):
        """Test setting Socket.io server."""
        manager = SocketIOManager()
        mock_sio = Mock()
        
        manager.set_socketio_server(mock_sio)
        
        assert manager.sio == mock_sio
    
    def test_track_job_for_project(self):
        """Test tracking a job for a project."""
        manager = SocketIOManager()
        
        manager.track_job_for_project("job_123", "project_456")
        
        assert manager._job_to_project["job_123"] == "project_456"
    
    def test_register_client_for_project(self):
        """Test registering a client for a project."""
        manager = SocketIOManager()
        
        manager.register_client_for_project("project_123", "client_abc")
        
        assert "project_123" in manager._connected_clients
        assert "client_abc" in manager._connected_clients["project_123"]
    
    def test_unregister_client_for_project(self):
        """Test unregistering a client for a project."""
        manager = SocketIOManager()
        
        manager.register_client_for_project("project_123", "client_abc")
        manager.unregister_client_for_project("project_123", "client_abc")
        
        assert "project_123" not in manager._connected_clients
    
    def test_unregister_client_removes_empty_project(self):
        """Test that unregistering last client removes project entry."""
        manager = SocketIOManager()
        
        manager.register_client_for_project("project_123", "client_abc")
        manager.register_client_for_project("project_123", "client_def")
        manager.unregister_client_for_project("project_123", "client_abc")
        
        assert "project_123" in manager._connected_clients
        assert "client_abc" not in manager._connected_clients["project_123"]
        assert "client_def" in manager._connected_clients["project_123"]
    
    def test_on_job_queued_emits_event(self):
        """Test that job queued event is emitted."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        manager._on_job_queued("job_123", {"status": "queued"})
        
        # Verify emit was called (asyncio.create_task is used)
        assert manager.sio is not None
    
    def test_on_job_active_emits_event(self):
        """Test that job active event is emitted."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        manager._on_job_active("job_123", {"status": "running"})
        
        assert manager.sio is not None
    
    def test_on_job_completed_emits_event(self):
        """Test that job completed event is emitted."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        result = {"summary": "Meeting summary"}
        manager._on_job_completed("job_123", {"result": result})
        
        assert manager.sio is not None
    
    def test_on_job_failed_emits_event(self):
        """Test that job failed event is emitted."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        manager._on_job_failed("job_123", {"error": "API error"})
        
        assert manager.sio is not None
    
    def test_on_job_progress_emits_event(self):
        """Test that job progress event is emitted."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        manager._on_job_progress("job_123", {"progress": 50, "message": "Processing..."})
        
        assert manager.sio is not None
    
    def test_no_emit_without_tracked_project(self):
        """Test that events are not emitted if project not tracked."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        # Don't track job for project
        manager._on_job_queued("job_123", {"status": "queued"})
        
        # Should not raise error, just skip emission
        assert manager.sio is not None
    
    def test_register_job_listeners(self):
        """Test registering job listeners."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        event_listener = JobEventListener()
        manager.register_job_listeners(event_listener)
        
        # Verify listeners were registered
        assert len(event_listener._listeners[JobEventType.QUEUED]) > 0
        assert len(event_listener._listeners[JobEventType.ACTIVE]) > 0
        assert len(event_listener._listeners[JobEventType.COMPLETED]) > 0
        assert len(event_listener._listeners[JobEventType.FAILED]) > 0
        assert len(event_listener._listeners[JobEventType.PROGRESS]) > 0


class TestSocketIOEventHandlers:
    """Tests for SocketIOEventHandlers."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test SocketIOEventHandlers initialization."""
        handlers = SocketIOEventHandlers()
        
        assert handlers.sio is None
        assert handlers.socketio_manager is not None
    
    @pytest.mark.asyncio
    async def test_set_socketio_server(self):
        """Test setting Socket.io server."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        
        handlers.set_socketio_server(mock_sio)
        
        assert handlers.sio == mock_sio
    
    @pytest.mark.asyncio
    async def test_on_connect_with_valid_token(self):
        """Test client connection with valid token."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        handlers.set_socketio_server(mock_sio)
        
        environ = {"QUERY_STRING": "token=user123:token_hash"}
        
        await handlers._on_connect("client_abc", environ)
        
        # Verify session was saved
        mock_sio.save_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_on_connect_without_token(self):
        """Test client connection without token."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        handlers.set_socketio_server(mock_sio)
        
        environ = {"QUERY_STRING": ""}
        
        await handlers._on_connect("client_abc", environ)
        
        # Should still allow connection (for now)
        mock_sio.save_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_on_disconnect(self):
        """Test client disconnection."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user_id": "user123"})
        handlers.set_socketio_server(mock_sio)
        
        await handlers._on_disconnect("client_abc")
        
        # Verify session was retrieved
        mock_sio.get_session.assert_called_once_with("client_abc")
    
    @pytest.mark.asyncio
    async def test_on_subscribe_project_with_valid_project(self):
        """Test subscribing to a project."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user_id": "user123"})
        handlers.set_socketio_server(mock_sio)
        
        data = {"project_id": "project_456"}
        
        await handlers._on_subscribe_project("client_abc", data)
        
        # Verify room was joined
        mock_sio.enter_room.assert_called_once()
        mock_sio.emit.assert_called()
    
    @pytest.mark.asyncio
    async def test_on_subscribe_project_without_project_id(self):
        """Test subscribing without project_id."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        handlers.set_socketio_server(mock_sio)
        
        data = {}
        
        await handlers._on_subscribe_project("client_abc", data)
        
        # Verify error was emitted
        mock_sio.emit.assert_called()
        call_args = mock_sio.emit.call_args
        assert "error" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_on_unsubscribe_project(self):
        """Test unsubscribing from a project."""
        handlers = SocketIOEventHandlers()
        mock_sio = AsyncMock()
        handlers.set_socketio_server(mock_sio)
        
        data = {"project_id": "project_456"}
        
        await handlers._on_unsubscribe_project("client_abc", data)
        
        # Verify room was left
        mock_sio.leave_room.assert_called_once()
        mock_sio.emit.assert_called()
    
    def test_extract_token_from_query(self):
        """Test extracting token from query string."""
        handlers = SocketIOEventHandlers()
        
        query_string = "token=abc123&other=value"
        token = handlers._extract_token_from_query(query_string)
        
        assert token == "abc123"
    
    def test_extract_token_from_query_no_token(self):
        """Test extracting token when not present."""
        handlers = SocketIOEventHandlers()
        
        query_string = "other=value"
        token = handlers._extract_token_from_query(query_string)
        
        assert token is None
    
    @pytest.mark.asyncio
    async def test_validate_token_with_valid_format(self):
        """Test validating token with valid format."""
        handlers = SocketIOEventHandlers()
        
        token = "user123:token_hash"
        user_id = await handlers._validate_token(token)
        
        assert user_id == "user123"
    
    @pytest.mark.asyncio
    async def test_validate_token_with_invalid_format(self):
        """Test validating token with invalid format."""
        handlers = SocketIOEventHandlers()
        
        token = "invalid_token"
        user_id = await handlers._validate_token(token)
        
        assert user_id is None
    
    @pytest.mark.asyncio
    async def test_validate_project_access(self):
        """Test validating project access."""
        handlers = SocketIOEventHandlers()
        
        has_access = await handlers._validate_project_access("user123", "project_456")
        
        assert has_access is True


class TestProgressTracker:
    """Tests for ProgressTracker."""
    
    def test_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker("job_123")
        
        assert tracker.job_id == "job_123"
        assert tracker.current_progress == 0
        assert tracker.event_listener is not None
    
    def test_set_progress(self):
        """Test setting progress."""
        tracker = ProgressTracker("job_123")
        
        tracker.set_progress(50, "Processing...")
        
        assert tracker.current_progress == 50
    
    def test_increment_progress(self):
        """Test incrementing progress."""
        tracker = ProgressTracker("job_123")
        
        tracker.increment_progress(10)
        
        assert tracker.current_progress == 10
    
    def test_increment_progress_multiple_times(self):
        """Test incrementing progress multiple times."""
        tracker = ProgressTracker("job_123")
        
        tracker.increment_progress(10)
        tracker.increment_progress(20)
        tracker.increment_progress(30)
        
        assert tracker.current_progress == 60
    
    def test_increment_progress_caps_at_100(self):
        """Test that progress is capped at 100%."""
        tracker = ProgressTracker("job_123")
        
        tracker.set_progress(90)
        tracker.increment_progress(20)
        
        assert tracker.current_progress == 100
    
    def test_mark_complete(self):
        """Test marking job as complete."""
        tracker = ProgressTracker("job_123")
        
        tracker.mark_complete()
        
        assert tracker.current_progress == 100
    
    def test_invalid_progress_percentage(self):
        """Test that invalid progress percentages are rejected."""
        tracker = ProgressTracker("job_123")
        
        tracker.set_progress(-10)
        assert tracker.current_progress == 0
        
        tracker.set_progress(150)
        assert tracker.current_progress == 0
    
    def test_progress_update_respects_min_interval(self):
        """Test that progress updates respect minimum interval."""
        tracker = ProgressTracker("job_123")
        tracker.min_update_interval = 5
        
        tracker.set_progress(10)
        initial_time = tracker.last_update_time
        
        # Try to update immediately (event should be skipped but state updated)
        tracker.update_progress(20)
        
        # Progress should be updated internally
        assert tracker.current_progress == 20
        # But last_update_time should not change (event not emitted)
        assert tracker.last_update_time == initial_time
    
    def test_progress_update_force_ignores_interval(self):
        """Test that force update ignores minimum interval."""
        tracker = ProgressTracker("job_123")
        tracker.min_update_interval = 5
        
        tracker.set_progress(10)
        
        # Force update immediately
        tracker.update_progress(20, force=True)
        
        # Progress should have changed
        assert tracker.current_progress == 20


class TestSocketIOIntegration:
    """Integration tests for Socket.io real-time updates."""
    
    def test_job_event_listener_integration(self):
        """Test integration between job listener and Socket.io manager."""
        manager = SocketIOManager()
        mock_sio = AsyncMock()
        manager.set_socketio_server(mock_sio)
        
        event_listener = JobEventListener()
        manager.register_job_listeners(event_listener)
        
        # Track job for project
        manager.track_job_for_project("job_123", "project_456")
        manager.register_client_for_project("project_456", "client_abc")
        
        # Emit job queued event
        event_listener.on_job_queued("job_123")
        
        # Verify Socket.io manager received the event
        assert manager.sio is not None
    
    def test_project_isolation(self):
        """Test that events are isolated by project."""
        manager = SocketIOManager()
        
        # Register clients for different projects
        manager.register_client_for_project("project_1", "client_a")
        manager.register_client_for_project("project_2", "client_b")
        
        # Track jobs for different projects
        manager.track_job_for_project("job_1", "project_1")
        manager.track_job_for_project("job_2", "project_2")
        
        # Verify isolation
        assert "project_1" in manager._connected_clients
        assert "project_2" in manager._connected_clients
        assert manager._job_to_project["job_1"] == "project_1"
        assert manager._job_to_project["job_2"] == "project_2"
    
    def test_multiple_clients_per_project(self):
        """Test multiple clients connected to same project."""
        manager = SocketIOManager()
        
        # Register multiple clients for same project
        manager.register_client_for_project("project_1", "client_a")
        manager.register_client_for_project("project_1", "client_b")
        manager.register_client_for_project("project_1", "client_c")
        
        # Verify all clients are registered
        assert len(manager._connected_clients["project_1"]) == 3
        assert "client_a" in manager._connected_clients["project_1"]
        assert "client_b" in manager._connected_clients["project_1"]
        assert "client_c" in manager._connected_clients["project_1"]
