# AI Agent Orchestration API Documentation

## Overview

This document provides comprehensive API documentation for the ZenAI AI Agent Orchestration system. All endpoints require authentication via JWT token in the `Authorization` header.

## Base URL

```
http://localhost:3000/api
```

## Authentication

All API requests require a valid JWT token in the Authorization header:

```
Authorization: Bearer <jwt_token>
```

## Endpoints

### Meeting Processing

#### POST /meetings/:id/process

Trigger meeting processing workflow (transcription → summarization → task extraction → Notion sync).

**Parameters:**
- `id` (path, required): Meeting ID
- `s3Key` (body, required): S3 key for the audio file

**Request Body:**
```json
{
  "s3Key": "meetings/2024-01-15/meeting-123.mp3"
}
```

**Response (202 Accepted):**
```json
{
  "agentRunId": "run_abc123",
  "status": "queued",
  "meetingId": "meeting_123",
  "createdAt": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `202 Accepted`: Job queued successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Meeting not found
- `500 Internal Server Error`: Server error

**Real-time Updates:**
The system emits Socket.io events for status changes:
- `agent:status` - Status transition (queued → running → success/error)
- `agent:progress` - Progress updates every 5 seconds
- `agent:complete` - Final results on completion

---

### Chat Interface

#### POST /chat

Send a chat message and receive contextual response based on project data.

**Request Body:**
```json
{
  "projectId": "project_123",
  "message": "What are the blockers for the current sprint?",
  "conversationId": "conv_456" // optional, for conversation continuity
}
```

**Response (200 OK):**
```json
{
  "conversationId": "conv_456",
  "message": "Based on recent meetings, the main blockers are...",
  "sources": [
    "meeting_summary_123",
    "task_456",
    "decision_789"
  ],
  "confidence": 0.92,
  "followUpQuestions": [
    "What's the timeline for resolving these blockers?",
    "Who is responsible for each blocker?"
  ]
}
```

**Status Codes:**
- `200 OK`: Response generated successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: User lacks permission to access project
- `500 Internal Server Error`: Server error

**Real-time Streaming:**
For streaming responses, use Socket.io connection:
```javascript
socket.emit('chat:message', {
  projectId: 'project_123',
  message: 'What are the blockers?'
});

socket.on('chat:response', (chunk) => {
  console.log('Streaming response:', chunk);
});
```

---

### AIPM Analysis

#### GET /projects/:id/aipm

Get AI Product Manager analysis for a project.

**Parameters:**
- `id` (path, required): Project ID

**Query Parameters:**
- `includeMetrics` (optional, default: true): Include detailed metrics
- `timeRange` (optional, default: "7d"): Analysis time range (7d, 30d, 90d)

**Response (200 OK):**
```json
{
  "projectId": "project_123",
  "health": "at-risk",
  "metrics": {
    "taskVelocity": 12,
    "completionRate": 0.75,
    "averageCycleTime": 3.2,
    "teamCapacity": 0.85
  },
  "blockers": [
    {
      "title": "API rate limiting issues",
      "impact": "high",
      "affectedTasks": ["task_123", "task_456"],
      "suggestedResolution": "Implement request batching and caching"
    }
  ],
  "recommendations": [
    {
      "title": "Increase sprint capacity",
      "rationale": "Team is operating at 85% capacity with increasing task volume",
      "priority": 1,
      "estimatedImpact": "Reduce cycle time by 20%"
    }
  ],
  "generatedAt": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Analysis generated successfully
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: User lacks permission to access project
- `404 Not Found`: Project not found
- `500 Internal Server Error`: Server error

---

### Dashboard Suggestions

#### GET /projects/:id/suggestions

Get AI-powered suggestions for dashboard cards.

**Parameters:**
- `id` (path, required): Project ID

**Query Parameters:**
- `cardTypes` (optional): Comma-separated list of card types (pending_tasks, project_insights, blockers, opportunities)
- `limit` (optional, default: 3): Number of suggestions per card type

**Response (200 OK):**
```json
{
  "projectId": "project_123",
  "pendingTasks": [
    {
      "title": "Review API design document",
      "description": "The API design needs review before implementation starts",
      "actionUrl": "/tasks/task_123",
      "priority": 1,
      "generatedAt": "2024-01-15T10:30:00Z"
    }
  ],
  "projectInsights": [
    {
      "title": "Sprint velocity trending up",
      "description": "Team velocity increased 15% over the last 3 sprints",
      "priority": 2,
      "generatedAt": "2024-01-15T10:30:00Z"
    }
  ],
  "blockers": [
    {
      "title": "Waiting on external API approval",
      "description": "Integration with payment provider is blocked pending approval",
      "actionUrl": "/tasks/task_456",
      "priority": 1,
      "generatedAt": "2024-01-15T10:30:00Z"
    }
  ],
  "opportunities": [
    {
      "title": "Automate deployment process",
      "description": "Manual deployment steps could be automated to save 2 hours per release",
      "priority": 3,
      "generatedAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Status Codes:**
- `200 OK`: Suggestions retrieved successfully
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: User lacks permission to access project
- `404 Not Found`: Project not found
- `500 Internal Server Error`: Server error

---

### Agent Run Status

#### GET /agent-runs/:id

Get status and results of an agent run.

**Parameters:**
- `id` (path, required): Agent Run ID

**Response (200 OK):**
```json
{
  "id": "run_abc123",
  "projectId": "project_123",
  "type": "meeting_summarize",
  "inputRef": "meeting_123",
  "status": "success",
  "output": {
    "summary": "Meeting focused on Q1 planning...",
    "keyDecisions": ["Prioritize API redesign", "Hire 2 engineers"],
    "actionItems": ["task_123", "task_456"],
    "tasks": [
      {
        "title": "Implement new API design",
        "priority": "high",
        "assignee": "john@example.com"
      }
    ]
  },
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T10:45:00Z",
  "completedAt": "2024-01-15T10:45:00Z"
}
```

**Status Codes:**
- `200 OK`: Agent run retrieved successfully
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Agent run not found
- `500 Internal Server Error`: Server error

---

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request parameters",
    "details": {
      "field": "s3Key",
      "reason": "S3 key is required"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | User lacks permission |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Standard endpoints**: 100 requests per minute per user
- **Chat endpoint**: 30 requests per minute per user
- **Meeting processing**: 10 concurrent jobs per project

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705329000
```

---

## Socket.io Events

### Client → Server

**chat:message**
```javascript
socket.emit('chat:message', {
  projectId: 'project_123',
  message: 'What are the blockers?',
  conversationId: 'conv_456' // optional
});
```

**agent:subscribe**
```javascript
socket.emit('agent:subscribe', {
  agentRunId: 'run_abc123'
});
```

### Server → Client

**agent:status**
```javascript
socket.on('agent:status', (data) => {
  console.log(data);
  // {
  //   agentRunId: 'run_abc123',
  //   status: 'running',
  //   previousStatus: 'queued',
  //   timestamp: '2024-01-15T10:30:00Z'
  // }
});
```

**agent:progress**
```javascript
socket.on('agent:progress', (data) => {
  console.log(data);
  // {
  //   agentRunId: 'run_abc123',
  //   percentage: 45,
  //   currentStep: 'Summarizing transcript...',
  //   timestamp: '2024-01-15T10:30:05Z'
  // }
});
```

**agent:complete**
```javascript
socket.on('agent:complete', (data) => {
  console.log(data);
  // {
  //   agentRunId: 'run_abc123',
  //   status: 'success',
  //   output: { ... },
  //   timestamp: '2024-01-15T10:30:45Z'
  // }
});
```

**chat:response**
```javascript
socket.on('chat:response', (chunk) => {
  console.log(chunk);
  // { text: 'Based on recent meetings...' }
});
```

**error**
```javascript
socket.on('error', (error) => {
  console.error(error);
  // {
  //   code: 'AGENT_FAILED',
  //   message: 'Agent execution failed',
  //   details: { ... }
  // }
});
```

---

## Examples

### Example 1: Process a Meeting

```bash
curl -X POST http://localhost:3000/api/meetings/meeting_123/process \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "s3Key": "meetings/2024-01-15/meeting-123.mp3"
  }'
```

### Example 2: Chat with Context

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "projectId": "project_123",
    "message": "What are the top blockers?"
  }'
```

### Example 3: Get AIPM Analysis

```bash
curl -X GET "http://localhost:3000/api/projects/project_123/aipm?timeRange=30d" \
  -H "Authorization: Bearer <jwt_token>"
```

### Example 4: Get Dashboard Suggestions

```bash
curl -X GET "http://localhost:3000/api/projects/project_123/suggestions?limit=5" \
  -H "Authorization: Bearer <jwt_token>"
```

---

## Pagination

Endpoints that return lists support pagination:

**Query Parameters:**
- `page` (optional, default: 1): Page number
- `limit` (optional, default: 20): Items per page

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

---

## Versioning

The API uses URL versioning. Current version is v1:

```
http://localhost:3000/api/v1/...
```

Future versions will be available at `/api/v2/`, etc.

---

## Support

For API support and issues, contact: api-support@zenai.com
