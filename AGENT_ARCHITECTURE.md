# AI Agent Architecture & Design Decisions

## Overview

This document describes the architecture of the AI Agent Orchestration system, including design decisions, component interactions, and implementation patterns.

## Architecture Principles

### 1. Multi-Agent Pattern

The system uses a multi-agent architecture where specialized agents handle different aspects of meeting intelligence:

- **Transcription Agent**: Converts audio to text
- **Summarization Agent**: Generates meeting summaries
- **Task Extraction Agent**: Extracts actionable tasks
- **AIPM Agent**: Analyzes project health
- **Chat Agent**: Provides conversational interface
- **Suggestions Agent**: Generates dashboard insights

**Benefits:**
- Separation of concerns: Each agent has a single responsibility
- Scalability: Agents can be scaled independently
- Resilience: Failure of one agent doesn't block others
- Testability: Each agent can be tested in isolation

### 2. Orchestration Layer

The AI Orchestration Engine coordinates agent execution and manages:

- **Agent Lifecycle**: Initialization, execution, cleanup
- **State Management**: Shared context between agents
- **Error Recovery**: Fallback logic and retry strategies
- **Event Emission**: Real-time updates via Socket.io

**Key Responsibilities:**
```
Request → Route to Agent → Execute → Update State → Emit Events → Response
```

### 3. Background Job Queue

BullMQ manages long-running workflows:

- **Async Processing**: Prevents API blocking
- **Retry Logic**: Exponential backoff for failed jobs
- **Concurrency Control**: Manages API rate limits
- **Job Persistence**: Tracks job state in Redis

**Job Types:**
- `meeting_summarize`: Full meeting processing pipeline
- `task_extract`: Task extraction from transcript
- `weekly_report`: Consolidated weekly report
- `aipm_analysis`: Project health analysis
- `chat`: Chat message processing

### 4. Memory & Context Management

The system maintains context across agent executions:

**pgvector (Semantic Search)**
- Stores embeddings of summaries, decisions, and insights
- Enables semantic search for context retrieval
- Maintains project-specific knowledge base

**Redis (Conversation History)**
- Stores conversation history with 24-hour TTL
- Caches frequently accessed context
- Manages session state for chat interface

**Database (Persistent Storage)**
- Stores transcripts, summaries, tasks, and agent runs
- Maintains audit trail of all operations
- Enables historical analysis and reporting

## Component Architecture

### Agent Base Class

All agents extend a common base class:

```python
class BaseAgent:
    def __init__(self, project_id: str, config: AgentConfig):
        self.project_id = project_id
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.context_retriever = ContextRetriever()
        
    async def execute(self, input_data: Dict) -> AgentResult:
        """Execute agent workflow"""
        try:
            self.logger.info(f"Starting {self.__class__.__name__}")
            result = await self._run(input_data)
            self.logger.info(f"Completed {self.__class__.__name__}")
            return result
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            return AgentResult(status="error", error=str(e))
    
    async def _run(self, input_data: Dict) -> AgentResult:
        """Override in subclass"""
        raise NotImplementedError
    
    async def retrieve_context(self, query: str) -> ContextData:
        """Retrieve relevant context from pgvector"""
        return await self.context_retriever.search(
            project_id=self.project_id,
            query=query
        )
```

**Benefits:**
- Consistent error handling across all agents
- Standardized logging and monitoring
- Shared context retrieval interface
- Simplified testing and mocking

### Transcription Agent

**Workflow:**
1. Download audio from S3
2. Check file size and chunk if necessary
3. Call Whisper API
4. Store transcript in database
5. Update AgentRun status

**Key Design Decisions:**

- **File Chunking**: Large files (>25MB) are split into 25MB chunks to respect API limits
- **Retry Logic**: Failed transcriptions retry up to 3 times with exponential backoff
- **Language Detection**: Whisper automatically detects language; stored for reference
- **Error Handling**: Partial transcripts are stored with error flag for manual review

**Implementation:**
```python
class TranscriptionAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        meeting_id = input_data['meeting_id']
        s3_key = input_data['s3_key']
        
        # Download audio
        audio_data = await self.download_from_s3(s3_key)
        
        # Chunk if necessary
        chunks = self.chunk_audio(audio_data, max_size=25*1024*1024)
        
        # Transcribe each chunk
        transcripts = []
        for chunk in chunks:
            transcript = await self.call_whisper_api(chunk)
            transcripts.append(transcript)
        
        # Combine and store
        full_transcript = ' '.join(transcripts)
        await self.store_transcript(meeting_id, full_transcript)
        
        return AgentResult(
            status="success",
            data={"transcript": full_transcript}
        )
```

### Summarization Agent

**Workflow:**
1. Retrieve transcript and project context
2. Build LangChain summarization chain
3. Execute chain with custom prompt
4. Validate summary with Pydantic schema
5. Embed summary in pgvector
6. Store in database

**Key Design Decisions:**

- **Context Injection**: Previous meeting summaries and decisions are injected into prompt
- **Prompt Template**: Reusable template ensures consistent summarization style
- **Validation**: Pydantic schema validates summary structure (key decisions, action items)
- **Embedding**: Summary is embedded and stored for future semantic search

**Prompt Template:**
```
You are an expert meeting summarizer. Summarize the following meeting transcript.

Project Context:
{project_context}

Previous Meetings:
{previous_summaries}

Meeting Transcript:
{transcript}

Generate a summary that includes:
1. Key decisions made
2. Action items with owners
3. Blockers or risks identified
4. Next steps

Format as JSON with keys: summary, keyDecisions, actionItems, blockers
```

**Implementation:**
```python
class SummarizationAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        meeting_id = input_data['meeting_id']
        transcript = input_data['transcript']
        
        # Retrieve context
        context = await self.retrieve_context(
            query=f"meeting {meeting_id} context"
        )
        
        # Build chain
        chain = self.build_summarization_chain(context)
        
        # Execute
        result = await chain.arun(transcript=transcript)
        
        # Validate
        summary = SummarySchema.parse_obj(result)
        
        # Embed and store
        embedding = await self.embed_text(summary.summary)
        await self.store_summary(meeting_id, summary, embedding)
        
        return AgentResult(status="success", data=summary.dict())
```

### Task Extraction Agent

**Workflow:**
1. Retrieve transcript and project context
2. Build LangChain extraction chain
3. Execute chain with structured output
4. Validate tasks with Zod schema
5. Match assignees to OrgMembers
6. Create Task records in database

**Key Design Decisions:**

- **Structured Output**: LangChain's structured output ensures consistent JSON format
- **Assignee Matching**: Fuzzy matching against OrgMembers for flexible name matching
- **Validation**: Zod schema validates task structure before database insertion
- **Error Handling**: Invalid tasks are logged but don't block other tasks

**Task Schema:**
```typescript
const ExtractedTaskSchema = z.object({
  title: z.string().min(5).max(200),
  description: z.string().max(1000),
  priority: z.enum(['low', 'medium', 'high']),
  assigneeName: z.string().optional(),
  dueDate: z.string().datetime().optional(),
  blockedBy: z.array(z.string()).optional(),
});
```

**Implementation:**
```python
class TaskExtractionAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        meeting_id = input_data['meeting_id']
        transcript = input_data['transcript']
        
        # Build extraction chain
        chain = self.build_extraction_chain()
        
        # Execute
        result = await chain.arun(transcript=transcript)
        
        # Validate and process
        tasks = []
        for task_data in result['tasks']:
            try:
                task = ExtractedTaskSchema.parse_obj(task_data)
                
                # Match assignee
                if task.assigneeName:
                    assignee = await self.match_assignee(task.assigneeName)
                    task.assigneeId = assignee.id if assignee else None
                
                # Create in database
                created_task = await self.create_task(task)
                tasks.append(created_task)
            except ValidationError as e:
                self.logger.warning(f"Invalid task: {e}")
        
        return AgentResult(status="success", data={"tasks": tasks})
```

### AIPM Agent

**Workflow:**
1. Aggregate project metrics (velocity, completion rate, blockers)
2. Retrieve recent meetings and decisions from pgvector
3. Analyze trends and patterns
4. Identify risks and opportunities
5. Generate prioritized recommendations

**Key Design Decisions:**

- **Multi-Metric Analysis**: Combines multiple data sources for holistic view
- **Trend Analysis**: Compares current metrics against historical baselines
- **Risk Scoring**: Quantifies risk impact for prioritization
- **Recommendation Ranking**: Ranks recommendations by estimated impact

**Metrics Calculated:**
- Task velocity: Tasks completed per sprint
- Completion rate: Percentage of tasks completed on time
- Cycle time: Average time from task creation to completion
- Team capacity: Percentage of team capacity utilized
- Blocker frequency: Number of blockers per sprint

**Implementation:**
```python
class AIProductManagerAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        project_id = input_data['project_id']
        
        # Aggregate metrics
        metrics = await self.aggregate_metrics(project_id)
        
        # Retrieve context
        context = await self.retrieve_context(
            query=f"project {project_id} decisions and blockers"
        )
        
        # Analyze
        analysis = await self.analyze_project(metrics, context)
        
        # Generate recommendations
        recommendations = await self.generate_recommendations(analysis)
        
        return AgentResult(
            status="success",
            data={
                "health": analysis.health,
                "metrics": metrics,
                "blockers": analysis.blockers,
                "recommendations": recommendations
            }
        )
```

### Chat Agent

**Workflow:**
1. Validate user permissions
2. Retrieve conversation history from Redis
3. Retrieve relevant context from pgvector
4. Build LangChain chat chain
5. Generate response with context
6. Store conversation in Redis

**Key Design Decisions:**

- **Permission Validation**: Checks user role and project access before processing
- **Context Filtering**: Filters results based on user permissions
- **Conversation History**: Maintains context across multiple messages
- **Streaming Responses**: Emits response chunks via Socket.io for real-time feedback

**Implementation:**
```python
class ChatAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        user_id = input_data['user_id']
        project_id = input_data['project_id']
        message = input_data['message']
        conversation_id = input_data.get('conversation_id')
        
        # Validate permissions
        if not await self.validate_access(user_id, project_id):
            raise PermissionError("User lacks access to project")
        
        # Retrieve conversation history
        history = await self.get_conversation_history(
            conversation_id or f"{user_id}:{project_id}"
        )
        
        # Retrieve context
        context = await self.retrieve_context(message)
        
        # Build chain
        chain = self.build_chat_chain(context, history)
        
        # Generate response
        response = await chain.arun(message=message)
        
        # Store in history
        await self.store_conversation(conversation_id, message, response)
        
        return AgentResult(status="success", data=response)
```

### Suggestions Agent

**Workflow:**
1. Analyze project data (tasks, meetings, metrics)
2. Generate suggestions for each card type
3. Rank suggestions by relevance
4. Cache suggestions in Redis (6-hour TTL)
5. Return suggestions via API

**Key Design Decisions:**

- **Card-Type Specific**: Different suggestion logic for each dashboard card
- **Ranking Algorithm**: Combines recency, impact, and relevance scores
- **Caching**: Reduces computation by caching suggestions
- **Refresh Triggers**: Updates suggestions on significant project changes

**Suggestion Types:**
- **Pending Tasks**: High-priority tasks due soon
- **Project Insights**: Trends and patterns in project data
- **Blockers**: Current blockers and their impact
- **Opportunities**: Potential improvements and optimizations

**Implementation:**
```python
class SuggestionsAgent(BaseAgent):
    async def _run(self, input_data: Dict) -> AgentResult:
        project_id = input_data['project_id']
        
        # Check cache
        cached = await self.get_cached_suggestions(project_id)
        if cached:
            return AgentResult(status="success", data=cached)
        
        # Generate suggestions
        suggestions = {
            "pending_tasks": await self.generate_pending_tasks_suggestions(),
            "project_insights": await self.generate_insights_suggestions(),
            "blockers": await self.generate_blockers_suggestions(),
            "opportunities": await self.generate_opportunities_suggestions()
        }
        
        # Rank and cache
        ranked = self.rank_suggestions(suggestions)
        await self.cache_suggestions(project_id, ranked, ttl=6*3600)
        
        return AgentResult(status="success", data=ranked)
```

## Error Handling Strategy

### Error Categories

1. **Transient Errors** (retry with backoff)
   - Network timeouts
   - Rate limit errors (429)
   - Temporary service unavailability (503)

2. **Validation Errors** (log and continue)
   - Invalid input format
   - Schema validation failures
   - Malformed API responses

3. **Permanent Errors** (fail and notify)
   - Authentication failures
   - Permission denied
   - Resource not found

### Retry Strategy

```python
async def retry_with_backoff(
    func,
    max_attempts=3,
    base_delay=1,
    max_delay=60
):
    for attempt in range(max_attempts):
        try:
            return await func()
        except TransientError as e:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
        except PermanentError:
            raise
```

### Graceful Degradation

- If summarization fails: Return transcript alone
- If task extraction fails: User can manually create tasks
- If Notion sync fails: Tasks still exist in ZenAI
- If suggestions fail: Show cached suggestions

## State Management

### Agent Run Lifecycle

```
┌─────────┐
│ queued  │ ← Job enqueued in BullMQ
└────┬────┘
     │
┌────▼────┐
│ running │ ← Agent executing
└────┬────┘
     │
     ├─────────────────┐
     │                 │
┌────▼────┐      ┌─────▼──┐
│ success │      │ error  │
└─────────┘      └────────┘
```

### Context Sharing

Agents share context through:

1. **Shared Redis Store**
   ```python
   context = {
       "project_id": "...",
       "meeting_id": "...",
       "transcript": "...",
       "summary": "...",
       "tasks": [...]
   }
   await redis.set(f"context:{agent_run_id}", json.dumps(context))
   ```

2. **pgvector Embeddings**
   ```python
   embedding = await embed_text(content)
   await db.project_embeddings.create({
       "project_id": project_id,
       "content_type": "summary",
       "embedding": embedding,
       "metadata": {"meeting_id": meeting_id}
   })
   ```

3. **Database Records**
   ```python
   await db.agent_run.create({
       "project_id": project_id,
       "type": "meeting_summarize",
       "output_json": result
   })
   ```

## Performance Considerations

### Optimization Strategies

1. **Caching**
   - Cache embeddings for frequently accessed content
   - Cache suggestions with 6-hour TTL
   - Cache conversation history with 24-hour TTL

2. **Batch Processing**
   - Process multiple meetings concurrently
   - Batch API calls to reduce latency
   - Use connection pooling for database

3. **Async/Await**
   - All I/O operations are async
   - Agents execute concurrently in job queue
   - Socket.io events emitted asynchronously

### Latency Targets

- Transcription: < 2x audio duration
- Summarization: < 30 seconds
- Task extraction: < 20 seconds
- Chat response: < 5 seconds
- Suggestions generation: < 10 seconds

## Monitoring & Observability

### Metrics Tracked

- Agent execution time
- Success/failure rates
- API call latency
- Error rates by type
- Queue depth and processing time

### Logging

All agents log:
- Agent start/completion
- API calls and responses
- Errors and exceptions
- Performance metrics

### Alerting

Alerts triggered for:
- Error rate > 5% in 1-hour window
- Agent execution time > 2x target
- Queue depth > 100 jobs
- API rate limit approaching

## Security Considerations

1. **API Key Management**
   - OpenAI API key in environment variables
   - Notion tokens encrypted in database
   - Token rotation for long-lived integrations

2. **Access Control**
   - User permissions validated before processing
   - Results filtered based on user role
   - Project-level access control enforced

3. **Data Privacy**
   - Sensitive data encrypted in transit and at rest
   - Data retention policies enforced
   - Audit logging for sensitive operations

## Future Enhancements

1. **Agent Learning**
   - Track suggestion accuracy and feedback
   - Improve recommendations over time
   - A/B test prompt variations

2. **Advanced Coordination**
   - Agents negotiate and collaborate on complex tasks
   - Dynamic agent selection based on task type
   - Agent specialization and fine-tuning

3. **Extended Integrations**
   - Slack integration for notifications
   - Jira integration for task sync
   - GitHub integration for code context

4. **Performance Optimization**
   - Implement agent caching and memoization
   - Optimize pgvector queries
   - Implement distributed agent execution
