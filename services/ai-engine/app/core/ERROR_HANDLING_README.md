# Comprehensive Error Handling & Recovery System

## Overview

This document describes the comprehensive error handling and recovery system implemented for the AI Agent Orchestration platform. The system provides:

1. **Error Tracking & Metrics** - Real-time error monitoring and alerting
2. **Error Notifications** - User-friendly error messages via Socket.io
3. **Graceful Degradation** - Continued operation with reduced functionality
4. **API Retry Logic** - Exponential backoff for transient failures
5. **Error Dashboard** - REST API endpoints for monitoring

## Architecture

### Components

#### 1. Error Dashboard (`error_dashboard.py`)

Tracks error metrics and provides monitoring capabilities.

**Key Classes:**
- `ErrorMetrics` - Tracks errors within a time window
- `get_error_metrics()` - Global error metrics instance

**Features:**
- Records errors with type, API name, severity, and details
- Calculates error rate as percentage
- Alerts when error rate exceeds threshold
- Provides recent error history
- Filters errors by API

**Usage:**
```python
from app.core.error_dashboard import get_error_metrics

metrics = get_error_metrics()

# Record an error
metrics.record_error(
    error_type="timeout",
    api_name="whisper",
    severity="high",
    message="Request timed out",
    details={"file_size": 25000000}
)

# Check error rate
if metrics.should_alert(threshold=5.0):
    print("Error rate exceeded 5%")

# Get metrics summary
summary = metrics.get_metrics_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Error rate: {summary['error_rate']}%")

# Get recent errors
recent = metrics.get_recent_errors(limit=10)

# Get errors by API
whisper_errors = metrics.get_errors_by_api("whisper")
```

#### 2. Error Notifications (`error_notifications.py`)

Manages error notifications and user-friendly messages.

**Key Classes:**
- `ErrorNotificationManager` - Handles error notifications
- `get_error_notification_manager()` - Global notification manager instance

**Features:**
- User-friendly error messages for common errors
- Socket.io event emission for real-time notifications
- Error rate alerting
- Service recovery notifications

**Error Message Mapping:**
- `whisper_timeout` → "Audio transcription is taking longer than expected..."
- `whisper_api_error` → "Unable to transcribe audio..."
- `openai_timeout` → "AI service is taking longer than expected..."
- `notion_api_error` → "Unable to sync with Notion..."
- And more...

**Usage:**
```python
from app.core.error_notifications import get_error_notification_manager

manager = get_error_notification_manager()

# Notify error
manager.notify_error(
    project_id="proj_123",
    error_type="timeout",
    api_name="whisper",
    message="Request timed out after 3600 seconds",
    severity="high",
    user_message="Audio transcription is taking longer than expected. Please try again."
)

# Notify recovery
manager.notify_recovery(
    project_id="proj_123",
    api_name="whisper",
    message="Whisper service recovered"
)

# Notify alert
manager.notify_alert(
    project_id="proj_123",
    alert_type="high_error_rate",
    message="Error rate is 7.5% (threshold: 5%)",
    severity="high"
)

# Check and alert on error rate
if manager.check_and_alert_error_rate("proj_123", threshold=5.0):
    print("Alert sent for high error rate")
```

#### 3. Graceful Degradation (`graceful_degradation.py`)

Handles component failures with graceful degradation strategies.

**Key Classes:**
- `GracefulDegradationHandler` - Main degradation handler
- `TranscriptionDegradation` - Transcription-specific degradation
- `SummarizationDegradation` - Summarization-specific degradation
- `TaskExtractionDegradation` - Task extraction-specific degradation

**Degradation Strategies:**
- `SKIP_COMPONENT` - Skip the failed component and continue
- `USE_FALLBACK` - Use fallback function/data
- `PARTIAL_RESULT` - Return partial results
- `RETRY_LATER` - Queue for retry
- `NOTIFY_USER` - Notify user of degradation

**Usage:**
```python
from app.core.graceful_degradation import (
    get_degradation_handler,
    TranscriptionDegradation,
    DegradationStrategy
)

# Register fallback and strategy
handler = get_degradation_handler()

def fallback_summarize(context):
    return {"summary": "Unable to generate summary"}

handler.register_fallback("summarization", fallback_summarize)
handler.register_strategy("summarization", DegradationStrategy.USE_FALLBACK)

# Handle component failure
error = Exception("Summarization API failed")
result = handler.handle_component_failure(
    "summarization",
    error,
    context={"transcript": "..."}
)

# Transcription-specific degradation
result = TranscriptionDegradation.handle_transcription_failure(
    meeting_id="meeting_123",
    error=Exception("Transcription failed"),
    partial_transcript="Partial transcript..."
)

# Summarization-specific degradation
result = SummarizationDegradation.handle_summarization_failure(
    meeting_id="meeting_123",
    transcript="Full transcript available",
    error=Exception("Summarization failed")
)

# Task extraction-specific degradation
result = TaskExtractionDegradation.handle_extraction_failure(
    meeting_id="meeting_123",
    transcript="Full transcript available",
    error=Exception("Task extraction failed"),
    partial_tasks=[{"title": "Task 1"}]
)
```

#### 4. API Retry Handler (`api_retry.py`)

Implements exponential backoff retry logic for API calls.

**Key Classes:**
- `RetryConfig` - Configuration for retry behavior
- `APIRetryHandler` - Handles retries with exponential backoff

**Features:**
- Configurable max retries (default 3)
- Exponential backoff with configurable base and max delay
- Automatic error categorization
- Detailed logging of retry attempts

**Usage:**
```python
from app.core.api_retry import APIRetryHandler, RetryConfig

# Create retry handler with custom config
config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)
handler = APIRetryHandler(config)

# Retry async function
async def call_whisper_api():
    # API call here
    pass

result = await handler.retry_async(
    call_whisper_api,
    api_name="whisper",
    endpoint="transcriptions.create"
)

# Retry sync function
def call_openai_api():
    # API call here
    pass

result = handler.retry_sync(
    call_openai_api,
    api_name="openai",
    endpoint="chat.completions.create"
)
```

## Integration with Agents

### Transcription Agent

The Transcription Agent includes comprehensive error handling:

```python
from app.agents.transcription_agent import TranscriptionAgent, AgentConfig

config = AgentConfig(
    name="TranscriptionAgent",
    model="whisper-1",
    temperature=0.0,
)
agent = TranscriptionAgent(config)

# Execute with error handling
result = await agent.execute(
    meeting_id="meeting_123",
    s3_key="meetings/audio.mp3"
)

# Result includes error handling
if result.status == AgentStatus.SUCCESS:
    transcript = result.data.get("transcript")
else:
    error = result.error
```

**Error Handling Features:**
- Automatic retry with exponential backoff
- Error metrics recording
- User-friendly error notifications
- Graceful degradation on failure

## REST API Endpoints

### Error Dashboard Endpoints

#### Get Error Metrics
```
GET /api/errors/metrics
```

Returns error metrics summary and recent errors.

**Response:**
```json
{
  "status": "success",
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "total_errors": 5,
    "critical_errors": 0,
    "high_errors": 2,
    "error_rate": 3.5,
    "errors_by_type": {
      "timeout": 2,
      "api_error": 3
    },
    "errors_by_api": {
      "whisper": 3,
      "openai": 2
    },
    "window_minutes": 60
  },
  "recent_errors": [...]
}
```

#### Get Errors by API
```
GET /api/errors/by-api/{api_name}?limit=10
```

Returns recent errors for a specific API.

**Parameters:**
- `api_name` - API name (whisper, openai, notion)
- `limit` - Maximum errors to return (1-50, default 10)

**Response:**
```json
{
  "status": "success",
  "api_name": "whisper",
  "error_count": 3,
  "errors": [
    {
      "timestamp": "2024-01-15T10:25:00Z",
      "error_type": "timeout",
      "severity": "high",
      "message": "Request timed out",
      "details": {}
    }
  ]
}
```

#### Check Alert Status
```
GET /api/errors/alert-status?threshold=5.0
```

Check if error rate exceeds alert threshold.

**Parameters:**
- `threshold` - Alert threshold as percentage (0.1-100, default 5.0)

**Response:**
```json
{
  "status": "success",
  "should_alert": false,
  "error_rate": 3.5,
  "threshold": 5.0,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Health Check
```
GET /api/health
```

Returns health status including error metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "queue_stats": {...},
  "error_metrics": {
    "total_errors": 5,
    "error_rate": 3.5,
    ...
  }
}
```

## Socket.io Events

### Error Notifications

**Event:** `error_notification`

Emitted when an error occurs.

**Payload:**
```json
{
  "type": "error",
  "error_type": "timeout",
  "api_name": "whisper",
  "severity": "high",
  "message": "Audio transcription is taking longer than expected. Please try again.",
  "timestamp": "2024-01-15T10:25:00Z",
  "details": {}
}
```

### Service Recovery

**Event:** `service_recovery`

Emitted when a service recovers.

**Payload:**
```json
{
  "type": "recovery",
  "api_name": "whisper",
  "message": "Service recovered",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### System Alerts

**Event:** `system_alert`

Emitted for system-level alerts (e.g., high error rate).

**Payload:**
```json
{
  "type": "alert",
  "alert_type": "high_error_rate",
  "message": "Error rate is 7.5% (threshold: 5%)",
  "severity": "high",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling Strategies by Component

### Transcription Failures

**Strategy:** Graceful Degradation with Notification

1. Attempt transcription with retry logic (3 attempts)
2. If all retries fail:
   - Record error in metrics
   - Notify user via Socket.io
   - Return error status
3. User can retry or manually upload audio

### Summarization Failures

**Strategy:** Graceful Degradation - Transcript Available

1. Attempt summarization with retry logic
2. If summarization fails:
   - Record error in metrics
   - Notify user that transcript is available
   - Return transcript without summary
3. User can still access transcript for manual review

### Task Extraction Failures

**Strategy:** Graceful Degradation - Partial Results

1. Attempt task extraction with retry logic
2. If extraction fails:
   - Record error in metrics
   - Return partial tasks if available
   - Notify user to review transcript manually
3. User can manually create tasks from transcript

### Notion Sync Failures

**Strategy:** Graceful Degradation - Tasks Available Locally

1. Attempt Notion sync with retry logic (1 retry)
2. If sync fails:
   - Record error in metrics
   - Notify user of sync failure
   - Tasks remain available in ZenAI
3. User can retry sync or manually update Notion

## Monitoring & Alerting

### Error Rate Monitoring

The system tracks error rates in 1-hour windows:

- **Low:** < 2% error rate
- **Medium:** 2-5% error rate
- **High:** 5-10% error rate
- **Critical:** > 10% error rate

### Alert Thresholds

Default alert threshold is 5% error rate. Alerts are sent via:

1. Socket.io events to connected clients
2. Error dashboard API
3. System logs

### Error Metrics Dashboard

Access error metrics at:
- `GET /api/errors/metrics` - Overall metrics
- `GET /api/errors/by-api/{api_name}` - API-specific errors
- `GET /api/errors/alert-status` - Alert status

## Best Practices

### For Developers

1. **Always use retry handler for API calls:**
   ```python
   result = await handler.retry_async(
       api_call_func,
       api_name="whisper",
       endpoint="transcriptions.create"
   )
   ```

2. **Record errors in metrics:**
   ```python
   error_metrics.record_error(
       error_type="timeout",
       api_name="whisper",
       severity="high",
       message=str(error)
   )
   ```

3. **Notify users of failures:**
   ```python
   error_notifier.notify_error(
       project_id=project_id,
       error_type="api_error",
       api_name="whisper",
       message=str(error),
       severity="high"
   )
   ```

4. **Implement graceful degradation:**
   ```python
   result = handler.handle_component_failure(
       component_name,
       error,
       context=context
   )
   ```

### For Operations

1. **Monitor error dashboard regularly:**
   - Check `/api/errors/metrics` for trends
   - Review `/api/errors/by-api/{api_name}` for API-specific issues
   - Set up alerts for error rate > 5%

2. **Respond to alerts:**
   - Check Socket.io events for error notifications
   - Review error details in dashboard
   - Take corrective action (restart services, etc.)

3. **Track error patterns:**
   - Identify recurring errors
   - Correlate with external service issues
   - Adjust retry policies as needed

## Testing

Run error handling tests:

```bash
pytest tests/test_error_handling.py -v
```

Tests cover:
- Error metrics tracking
- Error notifications
- Graceful degradation strategies
- API retry logic
- Error rate calculations
- Alert thresholds

## Future Enhancements

1. **Persistent Error Storage** - Store errors in database for long-term analysis
2. **Error Analytics** - Analyze error patterns and trends
3. **Automated Recovery** - Automatically restart failed services
4. **Error Prediction** - Predict failures before they occur
5. **Custom Alert Rules** - Allow users to define custom alert rules
6. **Error Correlation** - Correlate errors across components
7. **Performance Impact Analysis** - Measure impact of errors on performance
