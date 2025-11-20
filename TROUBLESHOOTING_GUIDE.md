# Troubleshooting Guide

## Common Issues & Solutions

This guide provides solutions for common issues encountered when running the AI Agent Orchestration system.

---

## Table of Contents

1. [Startup Issues](#startup-issues)
2. [Database Issues](#database-issues)
3. [Redis Issues](#redis-issues)
4. [API Issues](#api-issues)
5. [Agent Execution Issues](#agent-execution-issues)
6. [Performance Issues](#performance-issues)
7. [Integration Issues](#integration-issues)
8. [Logging & Debugging](#logging--debugging)

---

## Startup Issues

### Issue: Application fails to start

**Error Message:**
```
Error: Cannot find module 'express'
```

**Solution:**

```bash
# Install dependencies
npm install

# Or for Python
pip install -r requirements.txt

# Verify installation
npm list express
```

---

### Issue: Port already in use

**Error Message:**
```
Error: listen EADDRINUSE: address already in use :::3000
```

**Solution:**

```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>

# Or use different port
PORT=3001 npm start
```

---

### Issue: Environment variables not loaded

**Error Message:**
```
Error: OPENAI_API_KEY is not defined
```

**Solution:**

```bash
# Check if .env file exists
ls -la .env

# Create .env file if missing
cp .env.example .env

# Edit .env with your values
nano .env

# Verify variables are loaded
node -e "console.log(process.env.OPENAI_API_KEY)"
```

---

## Database Issues

### Issue: Cannot connect to PostgreSQL

**Error Message:**
```
Error: connect ECONNREFUSED 127.0.0.1:5432
```

**Solution:**

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL if not running
sudo systemctl start postgresql

# Test connection
psql -U zenai_user -d zenai_prod -c "SELECT 1;"

# Check connection string
echo $DATABASE_URL
```

---

### Issue: pgvector extension not found

**Error Message:**
```
Error: type "vector" does not exist
```

**Solution:**

```bash
# Connect to database
psql -U zenai_user -d zenai_prod

# Create extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';

# Exit psql
\q
```

---

### Issue: Database migrations fail

**Error Message:**
```
Error: Migration failed: syntax error in migration
```

**Solution:**

```bash
# Check migration status
npx prisma migrate status

# View migration details
cat prisma/migrations/<migration-name>/migration.sql

# Reset database (development only)
npx prisma migrate reset

# Or rollback to previous migration
npx prisma migrate resolve --rolled-back <migration-name>
```

---

### Issue: Connection pool exhausted

**Error Message:**
```
Error: remaining connection slots are reserved for non-replication superuser connections
```

**Solution:**

```bash
# Check active connections
psql -U zenai_user -d zenai_prod -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
psql -U zenai_user -d zenai_prod -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND query_start < now() - interval '10 minutes';"

# Increase max connections
sudo nano /etc/postgresql/14/main/postgresql.conf
# Change: max_connections = 200
sudo systemctl restart postgresql
```

---

### Issue: Slow database queries

**Error Message:**
```
Query took 5000ms (expected < 100ms)
```

**Solution:**

```bash
# Analyze query performance
psql -U zenai_user -d zenai_prod -c "
EXPLAIN ANALYZE 
SELECT * FROM agent_runs WHERE project_id = 'proj_123';"

# Create missing indexes
psql -U zenai_user -d zenai_prod -c "
CREATE INDEX idx_agent_runs_project_id ON agent_runs(project_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);"

# Vacuum and analyze
psql -U zenai_user -d zenai_prod -c "VACUUM ANALYZE;"

# Check index usage
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM pg_stat_user_indexes 
WHERE idx_scan = 0;"
```

---

## Redis Issues

### Issue: Cannot connect to Redis

**Error Message:**
```
Error: connect ECONNREFUSED 127.0.0.1:6379
```

**Solution:**

```bash
# Check if Redis is running
sudo systemctl status redis-server

# Start Redis if not running
sudo systemctl start redis-server

# Test connection
redis-cli ping

# Check connection string
echo $REDIS_URL
```

---

### Issue: Redis memory full

**Error Message:**
```
Error: OOM command not allowed when used memory > 'maxmemory'
```

**Solution:**

```bash
# Check memory usage
redis-cli INFO memory

# Set max memory policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Clear expired keys
redis-cli FLUSHDB ASYNC

# Or increase max memory
redis-cli CONFIG SET maxmemory 2gb
```

---

### Issue: Redis connection timeout

**Error Message:**
```
Error: Redis connection timeout
```

**Solution:**

```bash
# Check Redis responsiveness
redis-cli --latency

# Check Redis configuration
redis-cli CONFIG GET timeout

# Increase timeout
redis-cli CONFIG SET timeout 300

# Restart Redis
sudo systemctl restart redis-server
```

---

### Issue: Job queue not processing

**Error Message:**
```
Jobs stuck in queue, not being processed
```

**Solution:**

```bash
# Check queue depth
redis-cli LLEN bullmq:queue:meeting_summarize

# Check failed jobs
redis-cli LLEN bullmq:queue:meeting_summarize:failed

# Check worker status
redis-cli HGETALL bullmq:worker:worker_id

# Restart workers
sudo systemctl restart zenai-workers

# Clear failed jobs (if safe)
redis-cli DEL bullmq:queue:meeting_summarize:failed
```

---

## API Issues

### Issue: 401 Unauthorized

**Error Message:**
```
Error: 401 Unauthorized - Invalid or missing token
```

**Solution:**

```bash
# Check if token is provided
curl -H "Authorization: Bearer <token>" http://localhost:3000/api/health

# Verify JWT secret
echo $JWT_SECRET

# Generate new token
node -e "
const jwt = require('jsonwebtoken');
const token = jwt.sign(
  { userId: 'test' },
  process.env.JWT_SECRET,
  { expiresIn: '24h' }
);
console.log(token);
"

# Test with new token
curl -H "Authorization: Bearer <new-token>" http://localhost:3000/api/health
```

---

### Issue: 403 Forbidden

**Error Message:**
```
Error: 403 Forbidden - User lacks permission
```

**Solution:**

```bash
# Check user permissions
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM user_permissions 
WHERE user_id = 'user_123';"

# Check project access
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM project_members 
WHERE user_id = 'user_123' AND project_id = 'proj_123';"

# Grant permission if needed
psql -U zenai_user -d zenai_prod -c "
INSERT INTO project_members (user_id, project_id, role) 
VALUES ('user_123', 'proj_123', 'member');"
```

---

### Issue: 404 Not Found

**Error Message:**
```
Error: 404 Not Found - Resource not found
```

**Solution:**

```bash
# Check if resource exists
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM meetings WHERE id = 'meeting_123';"

# Check if endpoint is correct
curl -X GET http://localhost:3000/api/meetings/meeting_123

# Check API documentation
# See API_DOCUMENTATION.md for correct endpoints
```

---

### Issue: 500 Internal Server Error

**Error Message:**
```
Error: 500 Internal Server Error
```

**Solution:**

```bash
# Check application logs
tail -f /var/log/zenai/error.log

# Check for specific error
grep "500" /var/log/zenai/app.log | tail -20

# Check system resources
top -b -n 1 | head -20

# Restart application
sudo systemctl restart zenai-orchestration

# Check Sentry for error details
# Visit https://sentry.io/organizations/zenai/
```

---

### Issue: Rate limiting

**Error Message:**
```
Error: 429 Too Many Requests
```

**Solution:**

```bash
# Check rate limit headers
curl -i http://localhost:3000/api/health | grep X-RateLimit

# Wait before retrying
sleep 60

# Implement exponential backoff in client
# See API_DOCUMENTATION.md for rate limits

# Increase rate limit if needed
# Update OPENAI_RATE_LIMIT in .env
```

---

## Agent Execution Issues

### Issue: Agent execution timeout

**Error Message:**
```
Error: Agent execution timeout after 60000ms
```

**Solution:**

```bash
# Check agent logs
grep "Agent timeout" /var/log/zenai/app.log

# Check if API is responding
curl -X GET https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Increase timeout
# Update OPENAI_TIMEOUT in .env
OPENAI_TIMEOUT=60000

# Restart application
sudo systemctl restart zenai-orchestration
```

---

### Issue: Agent execution fails

**Error Message:**
```
Error: Agent execution failed - Invalid response format
```

**Solution:**

```bash
# Check agent run status
curl -X GET http://localhost:3000/api/agent-runs/run_123 \
  -H "Authorization: Bearer <token>"

# Check error details
psql -U zenai_user -d zenai_prod -c "
SELECT error_msg FROM agent_runs WHERE id = 'run_123';"

# Check agent logs
grep "Agent failed" /var/log/zenai/app.log

# Retry agent execution
curl -X POST http://localhost:3000/api/meetings/meeting_123/process \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"s3Key": "meetings/meeting.mp3"}'
```

---

### Issue: Transcription fails

**Error Message:**
```
Error: Whisper API error - Invalid audio format
```

**Solution:**

```bash
# Check audio file
file /path/to/audio.mp3

# Verify audio format
ffprobe -v error -select_streams a:0 -show_entries stream=codec_type,codec_name -of default=noprint_wrappers=1:nokey=1:nokey=1 /path/to/audio.mp3

# Convert to supported format
ffmpeg -i input.wav -acodec libmp3lame -ab 192k output.mp3

# Check file size
ls -lh /path/to/audio.mp3

# If > 25MB, it will be chunked automatically
# Check chunking logs
grep "Chunking audio" /var/log/zenai/app.log
```

---

### Issue: Summarization produces poor quality

**Error Message:**
```
Summary is too short or missing key points
```

**Solution:**

```bash
# Check prompt template
grep -A 10 "SUMMARIZATION_PROMPT" app/agents/summarization_agent.py

# Adjust temperature for more consistent output
OPENAI_TEMPERATURE=0.5

# Increase max tokens
OPENAI_MAX_TOKENS=3000

# Check context retrieval
grep "Context retrieved" /var/log/zenai/app.log

# Verify transcript quality
psql -U zenai_user -d zenai_prod -c "
SELECT text FROM transcripts WHERE meeting_id = 'meeting_123';"
```

---

### Issue: Task extraction misses tasks

**Error Message:**
```
Extracted tasks don't match expected tasks
```

**Solution:**

```bash
# Check extraction prompt
grep -A 10 "EXTRACTION_PROMPT" app/agents/task_extraction_agent.py

# Review extracted tasks
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM tasks WHERE meeting_id = 'meeting_123';"

# Check validation errors
grep "Validation error" /var/log/zenai/app.log

# Adjust prompt or schema if needed
# See AGENT_ARCHITECTURE.md for prompt templates
```

---

## Performance Issues

### Issue: Slow API response

**Error Message:**
```
API response time > 5 seconds
```

**Solution:**

```bash
# Check response time
time curl http://localhost:3000/api/health

# Check database query time
psql -U zenai_user -d zenai_prod -c "
SELECT query, mean_exec_time FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;"

# Check Redis latency
redis-cli --latency

# Check system resources
top -b -n 1 | head -20

# Enable query logging
psql -U zenai_user -d zenai_prod -c "
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();"
```

---

### Issue: High CPU usage

**Error Message:**
```
CPU usage > 80%
```

**Solution:**

```bash
# Check top processes
top -b -n 1 | head -20

# Check if agents are running
ps aux | grep agent

# Check job queue depth
redis-cli LLEN bullmq:queue:meeting_summarize

# Reduce concurrency
BULLMQ_CONCURRENCY=5

# Restart application
sudo systemctl restart zenai-orchestration
```

---

### Issue: High memory usage

**Error Message:**
```
Memory usage > 80%
```

**Solution:**

```bash
# Check memory usage
free -h

# Check process memory
ps aux --sort=-%mem | head -10

# Check Redis memory
redis-cli INFO memory

# Clear Redis cache
redis-cli FLUSHDB ASYNC

# Increase system memory or scale horizontally
```

---

### Issue: Slow pgvector queries

**Error Message:**
```
Semantic search taking > 5 seconds
```

**Solution:**

```bash
# Check index status
psql -U zenai_user -d zenai_prod -c "
SELECT * FROM pg_stat_user_indexes 
WHERE relname = 'project_embeddings';"

# Recreate index if needed
psql -U zenai_user -d zenai_prod -c "
DROP INDEX IF EXISTS project_embeddings_embedding_idx;
CREATE INDEX ON project_embeddings USING ivfflat (embedding vector_cosine_ops);"

# Check query performance
psql -U zenai_user -d zenai_prod -c "
EXPLAIN ANALYZE 
SELECT * FROM project_embeddings 
ORDER BY embedding <-> '[...]' LIMIT 10;"

# Adjust IVFFlat parameters
# See pgvector documentation for tuning
```

---

## Integration Issues

### Issue: Notion sync fails

**Error Message:**
```
Error: Notion API error - Invalid database ID
```

**Solution:**

```bash
# Check Notion API key
echo $NOTION_API_KEY

# Verify database ID
# Get from Notion URL: https://notion.so/workspace/database_id

# Test Notion API
curl -X GET https://api.notion.com/v1/databases/database_id \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2024-01-01"

# Check sync logs
grep "Notion sync" /var/log/zenai/app.log

# Retry sync
curl -X POST http://localhost:3000/api/tasks/task_123/sync-notion \
  -H "Authorization: Bearer <token>"
```

---

### Issue: S3 upload fails

**Error Message:**
```
Error: AWS S3 error - Access Denied
```

**Solution:**

```bash
# Check AWS credentials
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Verify S3 bucket exists
aws s3 ls s3://$AWS_S3_BUCKET

# Check bucket permissions
aws s3api get-bucket-acl --bucket $AWS_S3_BUCKET

# Test upload
aws s3 cp test.txt s3://$AWS_S3_BUCKET/test.txt

# Check IAM policy
# Ensure user has s3:PutObject permission
```

---

### Issue: OpenAI API rate limit

**Error Message:**
```
Error: OpenAI API error - Rate limit exceeded
```

**Solution:**

```bash
# Check current usage
# Visit https://platform.openai.com/account/usage/overview

# Implement exponential backoff
# See AGENT_ARCHITECTURE.md for retry strategy

# Reduce concurrency
BULLMQ_CONCURRENCY=3

# Upgrade API plan if needed
# Visit https://platform.openai.com/account/billing/overview
```

---

## Logging & Debugging

### Enable Debug Logging

```bash
# Set log level to debug
LOG_LEVEL=debug npm start

# Or in .env
LOG_LEVEL=debug
```

### View Logs

```bash
# View application logs
tail -f /var/log/zenai/app.log

# View error logs
tail -f /var/log/zenai/error.log

# Search for specific error
grep "ERROR" /var/log/zenai/app.log | tail -20

# View logs with timestamps
tail -f /var/log/zenai/app.log | grep "2024-01-15"
```

### Enable Request Logging

```bash
# Add to .env
LOG_REQUESTS=true
LOG_RESPONSES=true

# Restart application
sudo systemctl restart zenai-orchestration

# View request logs
grep "POST /api/chat" /var/log/zenai/app.log
```

### Debug Agent Execution

```bash
# Add debug logging to agent
# In agent code:
logger.debug("Agent state", {
  agent_type: "summarization",
  project_id: project_id,
  transcript_length: transcript.length
});

# View debug logs
grep "Agent state" /var/log/zenai/app.log
```

### Use Sentry for Error Tracking

```bash
# View errors in Sentry
# Visit https://sentry.io/organizations/zenai/

# Search for specific error
# Filter by environment, timestamp, etc.

# View error details
# Click on error to see stack trace and context
```

---

## Getting Help

If you can't resolve an issue:

1. **Check logs**: `tail -f /var/log/zenai/app.log`
2. **Check documentation**: See API_DOCUMENTATION.md, AGENT_ARCHITECTURE.md
3. **Search issues**: Check GitHub issues for similar problems
4. **Contact support**: 
   - Email: support@zenai.com
   - Slack: #zenai-support
   - On-call: Check PagerDuty

---

## Reporting Issues

When reporting an issue, include:

1. Error message and stack trace
2. Steps to reproduce
3. Environment (development/staging/production)
4. Relevant logs (last 50 lines)
5. System information (OS, Node version, etc.)

Example:

```
Title: Agent execution timeout on large meetings

Description:
When processing meetings > 1 hour, the summarization agent times out.

Steps to reproduce:
1. Upload 2-hour meeting audio
2. Trigger meeting processing
3. Wait for summarization agent

Error:
Error: Agent execution timeout after 60000ms

Environment: production
Logs: [attached]
System: Ubuntu 20.04, Node 18.12.0
```
