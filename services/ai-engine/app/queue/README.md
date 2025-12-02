# Job Queue & Redis Integration

This module provides background job queue management for AI workflows using RQ (Redis Queue) and Redis.

## Overview

The job queue system enables asynchronous processing of long-running AI tasks:
- Meeting transcription
- Meeting summarization
- Task extraction
- AIPM analysis
- Chat processing
- Notion synchronization

## Architecture

### Components

1. **RedisConfig** (`redis_config.py`)
   - Manages Redis connection pooling
   - Configurable via environment variables
   - Singleton pattern for global access

2. **JobQueueManager** (`job_queue.py`)
   - Enqueues jobs to RQ
   - Monitors job status and results
   - Manages worker lifecycle
   - Provides queue statistics

3. **JobEventListener** (`job_listeners.py`)
   - Emits events for job state changes
   - Supports callbacks for: queued, active, completed, failed, progress
   - Implements retry policy with exponential backoff

4. **AIOrchestrationEngine** (`orchestration_engine.py`)
   - Central coordinator for all AI workflows
   - Routes jobs to appropriate agents
   - Manages job lifecycle and status tracking

## Configuration

### Environment Variables

```bash
REDIS_HOST=localhost          # Redis server host
REDIS_PORT=6379              # Redis server port
REDIS_DB=0                   # Redis database number
REDIS_PASSWORD=              # Redis password (optional)
```

### Example .env

```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

## Usage

### Basic Job Enqueueing

```python
from app.queue.orchestration_engine import get_orchestration_engine

engine = get_orchestration_engine()

# Enqueue a transcription job
job_id = engine.enqueue_transcription_job(
    meeting_id="meeting-123",
    s3_key="s3://bucket/audio.mp3"
)

# Check job status
status = engine.get_job_status(job_id)
print(status)  # {'job_id': 'job-123', 'status': 'queued', ...}
```

### Job Types

```python
from app.queue.job_queue import JobType

# Available job types
JobType.TRANSCRIPTION      # Audio transcription
JobType.SUMMARIZATION      # Meeting summarization
JobType.TASK_EXTRACTION    # Task extraction
JobType.AIPM_ANALYSIS      # AIPM analysis
JobType.CHAT               # Chat processing
JobType.NOTION_SYNC        # Notion synchronization
JobType.WEEKLY_REPORT      # Weekly report generation
```

### Event Listeners

```python
from app.queue.job_listeners import get_job_event_listener, JobEventType

listener = get_job_event_listener()

def on_job_completed(job_id, event_data):
    print(f"Job {job_id} completed: {event_data}")

# Register callback
listener.register_listener(JobEventType.COMPLETED, on_job_completed)

# Emit event
listener.on_job_completed("job-123", result={"summary": "..."})
```

### Retry Policy

```python
from app.queue.job_listeners import RetryPolicy

policy = RetryPolicy(
    max_retries=3,           # Maximum 3 retries
    backoff_factor=2.0,      # Exponential backoff: 1s, 2s, 4s
    initial_delay=1          # Start with 1 second delay
)

# Get retry delay for attempt 0: 1 second
delay = policy.get_retry_delay(0)

# Check if should retry
should_retry = policy.should_retry(0)  # True
```

### Queue Statistics

```python
engine = get_orchestration_engine()

stats = engine.get_queue_stats()
print(stats)
# {
#     'queue_name': 'ai_workflows',
#     'job_count': 5,
#     'started_job_registry_count': 2,
#     'finished_job_registry_count': 10,
#     'failed_job_registry_count': 1,
#     'worker_count': 2,
#     'timestamp': '2024-01-15T10:30:00'
# }
```

## Job Lifecycle

```
┌─────────────┐
│   Queued    │  Job enqueued, waiting for worker
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Active    │  Worker processing job
└──────┬──────┘
       │
       ├─────────────────┬──────────────────┐
       ▼                 ▼                  ▼
   ┌────────┐      ┌────────┐        ┌──────────┐
   │ Success│      │ Failed │        │ Progress │
   └────────┘      └────────┘        └──────────┘
       │                │                  │
       └────────────────┴──────────────────┘
              ▼
        ┌──────────────┐
        │   Completed  │  Job finished (success or error)
        └──────────────┘
```

## Error Handling

### Job Failures

Failed jobs are automatically retried with exponential backoff:
- Attempt 1: 1 second delay
- Attempt 2: 2 seconds delay
- Attempt 3: 4 seconds delay
- After 3 failures: Job marked as failed

### Error Information

```python
engine = get_orchestration_engine()

status = engine.get_job_status("job-123")
if status['status'] == 'error':
    print(f"Error: {status['error']}")
```

## Worker Management

### Starting a Worker

```python
from app.queue.job_queue import get_job_queue_manager

manager = get_job_queue_manager()
worker = manager.start_worker(
    worker_name="worker-1",
    job_monitoring_interval=30
)

# Worker processes jobs from queue
worker.work()
```

### Multiple Workers

For production, run multiple workers:

```bash
# Terminal 1
python -c "from app.queue.job_queue import get_job_queue_manager; \
           manager = get_job_queue_manager(); \
           worker = manager.start_worker('worker-1'); \
           worker.work()"

# Terminal 2
python -c "from app.queue.job_queue import get_job_queue_manager; \
           manager = get_job_queue_manager(); \
           worker = manager.start_worker('worker-2'); \
           worker.work()"
```

## Integration with FastAPI

### API Endpoint Example

```python
from fastapi import FastAPI
from app.queue.orchestration_engine import get_orchestration_engine

app = FastAPI()
engine = get_orchestration_engine()

@app.post("/meetings/{meeting_id}/process")
async def process_meeting(meeting_id: str, s3_key: str):
    """Enqueue meeting for processing."""
    job_id = engine.enqueue_transcription_job(meeting_id, s3_key)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job status."""
    return engine.get_job_status(job_id)

@app.get("/queue/stats")
async def get_queue_stats():
    """Get queue statistics."""
    return engine.get_queue_stats()
```

## Testing

Run tests with:

```bash
pytest tests/test_job_queue.py -v
```

Tests cover:
- Redis configuration and connection pooling
- Job enqueueing and retrieval
- Event listener registration and emission
- Retry policy calculations
- Orchestration engine job management

## Requirements

- Redis server (local or remote)
- Python 3.8+
- Dependencies: `redis`, `rq`

## Troubleshooting

### Redis Connection Failed

```
Error: Redis connection failed: Connection refused
```

**Solution**: Ensure Redis is running:
```bash
redis-server
```

### No Workers Available

```
Job stuck in 'queued' status
```

**Solution**: Start a worker process:
```bash
python -c "from app.queue.job_queue import get_job_queue_manager; \
           manager = get_job_queue_manager(); \
           worker = manager.start_worker(); \
           worker.work()"
```

### Job Timeout

```
Job failed: Job exceeded timeout
```

**Solution**: Increase timeout when enqueueing:
```python
job = manager.enqueue_job(
    job_type=JobType.TRANSCRIPTION,
    func=transcribe_audio_job,
    kwargs={"meeting_id": "123", "s3_key": "s3://..."},
    timeout=7200  # 2 hours instead of default 1 hour
)
```

## Next Steps

This job queue infrastructure is ready for:
1. Task 5: Implement Transcription Agent
2. Task 6: Implement Summarization Agent
3. Task 7: Implement Task Extraction Agent
4. Task 8: Implement Notion Integration Agent
5. Task 9: Implement AIPM Agent
6. Task 10: Implement Chat Agent
7. Task 11: Implement Suggestions Agent

Each agent will implement its job function and be integrated with the orchestration engine.
