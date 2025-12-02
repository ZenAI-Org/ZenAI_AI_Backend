# Socket.io Real-Time Updates

This module implements real-time event emission for AI agent status changes using Socket.io. It provides project-level isolation, authentication, and progress tracking for long-running jobs.

## Overview

The Socket.io integration consists of three main components:

1. **SocketIOManager** - Manages Socket.io connections and event emission
2. **SocketIOEventHandlers** - Handles client connections and project subscriptions
3. **ProgressTracker** - Allows agents to report progress during execution

## Architecture

### Event Flow

```
Job Event Listener
    ↓
SocketIOManager (tracks job → project mapping)
    ↓
Socket.io Server
    ↓
Connected Clients (filtered by project)
```

### Status Transitions

Jobs emit events for the following status transitions:

- **queued** → Job is waiting in queue
- **running** → Job is actively processing
- **success** → Job completed successfully
- **error** → Job failed with error
- **progress** → Progress update (0-100%)

## Usage

### Client-Side (JavaScript/TypeScript)

```javascript
import io from 'socket.io-client';

// Connect with authentication token
const socket = io('http://localhost:8080', {
  query: {
    token: 'user123:token_hash'
  }
});

// Subscribe to project updates
socket.emit('subscribe_project', { project_id: 'project_456' });

// Listen for agent status changes
socket.on('agent_status', (data) => {
  console.log(`Job ${data.job_id} status: ${data.status}`);
  if (data.result) {
    console.log('Result:', data.result);
  }
});

// Listen for progress updates
socket.on('agent_progress', (data) => {
  console.log(`Job ${data.job_id} progress: ${data.progress}%`);
  if (data.message) {
    console.log('Message:', data.message);
  }
});

// Handle errors
socket.on('error', (data) => {
  console.error('Error:', data.message);
});

// Unsubscribe when done
socket.emit('unsubscribe_project', { project_id: 'project_456' });
```

### Server-Side (Python)

#### Tracking Jobs for Projects

```python
from app.queue.orchestration_engine import get_orchestration_engine

orchestrator = get_orchestration_engine()

# Enqueue a job (automatically tracked if project_id provided)
job_id = orchestrator.enqueue_transcription_job(
    meeting_id="meeting_123",
    s3_key="s3://bucket/audio.mp3",
    project_id="project_456"  # Job will be tracked for this project
)
```

#### Emitting Progress Updates

```python
from app.queue.progress_tracker import get_progress_tracker

# In your agent code
tracker = get_progress_tracker(job_id)

# Update progress
tracker.set_progress(25, "Downloading audio file...")
tracker.increment_progress(25, "Transcribing audio...")
tracker.set_progress(75, "Processing transcript...")
tracker.mark_complete("Transcription complete!")
```

#### Manual Event Emission

```python
from app.queue.socketio_manager import get_socketio_manager

socketio_manager = get_socketio_manager()

# Track a job for a project
socketio_manager.track_job_for_project("job_123", "project_456")

# Register a client for a project
socketio_manager.register_client_for_project("project_456", "client_sid")

# Emit custom event
await socketio_manager.emit_to_project_async(
    "project_456",
    "custom_event",
    {"data": "value"}
)
```

## Features

### Project-Level Isolation

Events are automatically routed only to clients connected to the relevant project:

```python
# Job for project_1 only reaches clients subscribed to project_1
socketio_manager.track_job_for_project("job_1", "project_1")
socketio_manager.register_client_for_project("project_1", "client_a")

# client_b (subscribed to project_2) won't receive events for job_1
socketio_manager.register_client_for_project("project_2", "client_b")
```

### Authentication

Clients must provide an authentication token when connecting:

```javascript
const socket = io('http://localhost:8080', {
  query: {
    token: 'user123:token_hash'
  }
});
```

Token validation is performed in `SocketIOEventHandlers._validate_token()`. Customize this method to integrate with your authentication system (JWT, session store, etc.).

### Progress Tracking

Agents can report progress during long-running operations:

```python
tracker = get_progress_tracker(job_id)

# Set progress to specific percentage
tracker.set_progress(50, "Halfway done")

# Increment by percentage
tracker.increment_progress(10, "Processing step 2")

# Mark as complete
tracker.mark_complete("All done!")
```

Progress updates are throttled to prevent excessive event emission (default: 5 second minimum interval).

### Authorization

Project access is validated in `SocketIOEventHandlers._validate_project_access()`. Customize this method to implement role-based access control:

```python
async def _validate_project_access(self, user_id, project_id):
    # Check if user is member of project
    user = await get_user(user_id)
    project = await get_project(project_id)
    return user in project.members
```

## Event Payloads

### Status Change Event

```json
{
  "job_id": "job_123",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event": "status_change",
  "result": null,
  "error": null
}
```

### Progress Update Event

```json
{
  "job_id": "job_123",
  "progress": 50,
  "timestamp": "2024-01-15T10:30:05.000Z",
  "event": "progress_update",
  "message": "Processing transcript..."
}
```

### Completion Event

```json
{
  "job_id": "job_123",
  "status": "success",
  "timestamp": "2024-01-15T10:30:30.000Z",
  "event": "status_change",
  "result": {
    "summary": "Meeting summary text",
    "tasks": [...]
  }
}
```

### Error Event

```json
{
  "job_id": "job_123",
  "status": "error",
  "timestamp": "2024-01-15T10:30:30.000Z",
  "event": "status_change",
  "error": "API rate limit exceeded"
}
```

## Configuration

### Socket.io Server Options

Edit `app/main.py` to customize Socket.io configuration:

```python
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # Restrict in production
    ping_timeout=60,           # Seconds before disconnecting idle clients
    ping_interval=25,          # Seconds between ping/pong
)
```

### Progress Tracker Options

Customize progress tracking behavior:

```python
tracker = get_progress_tracker(job_id)
tracker.min_update_interval = 5  # Minimum seconds between event emissions
```

## Integration with Job Queue

The Socket.io manager automatically integrates with the job event listener:

```python
# In app/main.py
job_event_listener = get_job_event_listener()
socketio_manager.register_job_listeners(job_event_listener)

# Now all job events are automatically emitted via Socket.io
```

## Testing

Run the Socket.io integration tests:

```bash
pytest tests/test_socketio_integration.py -v
```

Tests cover:
- Client connection/disconnection
- Project subscription/unsubscription
- Event emission and routing
- Progress tracking
- Project isolation
- Authentication and authorization

## Troubleshooting

### Events Not Received

1. Verify client is subscribed to correct project:
   ```javascript
   socket.emit('subscribe_project', { project_id: 'project_456' });
   ```

2. Check that job is tracked for project:
   ```python
   socketio_manager.track_job_for_project(job_id, project_id)
   ```

3. Verify Socket.io server is initialized:
   ```python
   socketio_manager = get_socketio_manager()
   assert socketio_manager.sio is not None
   ```

### Authentication Failures

1. Verify token format in query string:
   ```javascript
   const socket = io('http://localhost:8080', {
     query: { token: 'user123:token_hash' }
   });
   ```

2. Check token validation logic in `_validate_token()`

3. Verify authorization check in `_validate_project_access()`

### Progress Updates Not Emitted

1. Verify progress tracker is created with correct job_id:
   ```python
   tracker = get_progress_tracker(job_id)
   ```

2. Check minimum update interval:
   ```python
   tracker.min_update_interval = 1  # Reduce for testing
   ```

3. Verify job is tracked for project before updating progress

## Performance Considerations

- **Event Throttling**: Progress updates are throttled to 1 per 5 seconds by default
- **Client Isolation**: Events are only sent to clients subscribed to relevant project
- **Async Emission**: Events are emitted asynchronously to avoid blocking job execution
- **Memory**: Connected clients are tracked in memory; consider Redis for distributed deployments

## Future Enhancements

- [ ] Redis adapter for distributed deployments
- [ ] Event persistence for offline clients
- [ ] Batch event emission for high-throughput scenarios
- [ ] Metrics collection for event emission performance
- [ ] Client-side reconnection with state recovery
