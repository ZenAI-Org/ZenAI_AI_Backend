# Deployment Guide & Runbooks

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Setup](#environment-setup)
3. [Database Setup](#database-setup)
4. [Deployment Steps](#deployment-steps)
5. [Post-Deployment Verification](#post-deployment-verification)
6. [Scaling & Performance](#scaling--performance)
7. [Monitoring & Alerting](#monitoring--alerting)
8. [Troubleshooting Runbooks](#troubleshooting-runbooks)
9. [Rollback Procedures](#rollback-procedures)

---

## Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] All tests passing (unit, integration, end-to-end)
- [ ] Code reviewed and approved
- [ ] Environment variables configured
- [ ] Database migrations tested
- [ ] API keys and secrets secured
- [ ] SSL certificates valid
- [ ] Monitoring and alerting configured
- [ ] Backup strategy in place
- [ ] Incident response plan documented
- [ ] Team trained on deployment process

---

## Environment Setup

### 1. System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 50GB SSD
- Network: 100 Mbps connection

**Recommended for Production:**
- CPU: 8+ cores
- Memory: 16GB+ RAM
- Storage: 200GB+ SSD
- Network: 1 Gbps connection
- Load balancer for high availability

### 2. Required Services

Ensure the following services are running:

- **PostgreSQL 14+** with pgvector extension
- **Redis 7+** for caching and job queue
- **Node.js 18+** or **Python 3.10+** (depending on implementation)
- **Docker** (optional, for containerized deployment)

### 3. Install Dependencies

```bash
# Clone repository
git clone https://github.com/zenai/ai-orchestration.git
cd ai-orchestration

# Install Python dependencies
pip install -r requirements.txt

# Or install Node.js dependencies
npm install
```

### 4. Configure Environment Variables

Create `.env` file in project root:

```bash
# API Configuration
NODE_ENV=production
PORT=3000
API_BASE_URL=https://api.zenai.com

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/zenai_prod
REDIS_URL=redis://localhost:6379

# OpenAI API
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Notion Integration
NOTION_API_KEY=secret_...

# AWS S3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=zenai-meetings

# Authentication
JWT_SECRET=your-secret-key
JWT_EXPIRY=24h

# Logging
LOG_LEVEL=info
SENTRY_DSN=https://...

# Socket.io
SOCKET_IO_CORS_ORIGIN=https://app.zenai.com

# BullMQ
BULLMQ_CONCURRENCY=5
BULLMQ_MAX_ATTEMPTS=3
```

---

## Database Setup

### 1. Create PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE zenai_prod;

# Create user
CREATE USER zenai_user WITH PASSWORD 'secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE zenai_prod TO zenai_user;
```

### 2. Install pgvector Extension

```bash
# Connect to database
psql -U zenai_user -d zenai_prod

# Create extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Run Database Migrations

```bash
# Using Prisma
npx prisma migrate deploy

# Or using Alembic (Python)
alembic upgrade head
```

### 4. Create Indexes

```sql
-- Create index for pgvector similarity search
CREATE INDEX ON project_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Create indexes for common queries
CREATE INDEX idx_agent_runs_project_id ON agent_runs(project_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);
CREATE INDEX idx_transcripts_meeting_id ON transcripts(meeting_id);
CREATE INDEX idx_tasks_project_id ON tasks(project_id);
```

### 5. Verify Database Connection

```bash
# Test connection
psql -U zenai_user -d zenai_prod -c "SELECT version();"

# Test pgvector
psql -U zenai_user -d zenai_prod -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

---

## Deployment Steps

### Option 1: Manual Deployment

```bash
# 1. Pull latest code
git pull origin main

# 2. Install dependencies
npm install
# or
pip install -r requirements.txt

# 3. Run database migrations
npx prisma migrate deploy
# or
alembic upgrade head

# 4. Build application
npm run build
# or
python -m py_compile app/

# 5. Start application
npm start
# or
python -m app.main

# 6. Verify deployment
curl http://localhost:3000/health
```

### Option 2: Docker Deployment

```bash
# 1. Build Docker image
docker build -t zenai-orchestration:latest .

# 2. Push to registry
docker push your-registry/zenai-orchestration:latest

# 3. Pull on production server
docker pull your-registry/zenai-orchestration:latest

# 4. Run container
docker run -d \
  --name zenai-orchestration \
  --env-file .env \
  -p 3000:3000 \
  -v /data/logs:/app/logs \
  your-registry/zenai-orchestration:latest

# 5. Verify deployment
docker logs zenai-orchestration
curl http://localhost:3000/health
```

### Option 3: Kubernetes Deployment

```bash
# 1. Create namespace
kubectl create namespace zenai

# 2. Create secrets
kubectl create secret generic zenai-secrets \
  --from-env-file=.env \
  -n zenai

# 3. Apply deployment
kubectl apply -f k8s/deployment.yaml -n zenai

# 4. Verify deployment
kubectl get pods -n zenai
kubectl logs -f deployment/zenai-orchestration -n zenai

# 5. Expose service
kubectl expose deployment zenai-orchestration \
  --type=LoadBalancer \
  --port=80 \
  --target-port=3000 \
  -n zenai
```

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# Check API health
curl http://localhost:3000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-15T10:30:00Z",
#   "services": {
#     "database": "connected",
#     "redis": "connected",
#     "openai": "connected"
#   }
# }
```

### 2. Database Verification

```bash
# Check database connection
psql -U zenai_user -d zenai_prod -c "SELECT COUNT(*) FROM agent_runs;"

# Check pgvector
psql -U zenai_user -d zenai_prod -c "SELECT COUNT(*) FROM project_embeddings;"
```

### 3. Redis Verification

```bash
# Check Redis connection
redis-cli ping

# Expected response: PONG

# Check Redis memory
redis-cli info memory
```

### 4. API Endpoint Testing

```bash
# Test authentication
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password"}'

# Test meeting processing endpoint
curl -X POST http://localhost:3000/api/meetings/test/process \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"s3Key": "test/meeting.mp3"}'

# Test chat endpoint
curl -X POST http://localhost:3000/api/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"projectId": "test", "message": "Hello"}'
```

### 5. Log Verification

```bash
# Check application logs
tail -f /var/log/zenai/app.log

# Check error logs
tail -f /var/log/zenai/error.log

# Search for errors
grep ERROR /var/log/zenai/app.log | tail -20
```

---

## Scaling & Performance

### 1. Horizontal Scaling

**Add More Application Instances:**

```bash
# Using Docker Compose
docker-compose up -d --scale app=3

# Using Kubernetes
kubectl scale deployment zenai-orchestration --replicas=3 -n zenai
```

**Configure Load Balancer:**

```nginx
upstream zenai_backend {
    server app1:3000;
    server app2:3000;
    server app3:3000;
}

server {
    listen 80;
    server_name api.zenai.com;
    
    location / {
        proxy_pass http://zenai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Database Optimization

**Connection Pooling:**

```python
# Using PgBouncer
# /etc/pgbouncer/pgbouncer.ini
[databases]
zenai_prod = host=localhost port=5432 dbname=zenai_prod

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

**Query Optimization:**

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM agent_runs WHERE project_id = 'proj_123';

-- Vacuum and analyze
VACUUM ANALYZE;
```

### 3. Redis Optimization

**Memory Management:**

```bash
# Set max memory policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Monitor memory usage
redis-cli INFO memory
```

**Persistence:**

```bash
# Enable AOF (Append-Only File)
redis-cli CONFIG SET appendonly yes

# Set fsync policy
redis-cli CONFIG SET appendfsync everysec
```

### 4. Job Queue Optimization

**Concurrency Settings:**

```python
# app/queue/job_queue.py
BULLMQ_CONFIG = {
    'concurrency': 10,  # Increase for more parallel jobs
    'max_attempts': 3,
    'backoff': {
        'type': 'exponential',
        'delay': 2000
    }
}
```

---

## Monitoring & Alerting

### 1. Application Monitoring

**Using Prometheus:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'zenai-orchestration'
    static_configs:
      - targets: ['localhost:3000']
```

**Key Metrics to Monitor:**

- Request latency (p50, p95, p99)
- Error rate (errors per minute)
- Agent execution time
- Queue depth
- Database connection pool usage
- Redis memory usage

### 2. Error Tracking

**Using Sentry:**

```python
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    traces_sample_rate=0.1,
    environment=os.getenv('NODE_ENV')
)
```

### 3. Alerting Rules

**Alert Conditions:**

```yaml
# Alert if error rate > 5%
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"

# Alert if agent execution time > 2x target
- alert: SlowAgentExecution
  expr: agent_execution_time_seconds > 60
  for: 5m
  annotations:
    summary: "Agent execution time exceeds threshold"

# Alert if queue depth > 100
- alert: HighQueueDepth
  expr: bullmq_queue_depth > 100
  for: 5m
  annotations:
    summary: "Job queue depth is high"
```

### 4. Logging

**Structured Logging:**

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("Agent started", extra={
    "agent_type": "summarization",
    "project_id": "proj_123",
    "meeting_id": "meet_456"
})
```

---

## Troubleshooting Runbooks

### Runbook 1: High Error Rate

**Symptoms:**
- Error rate > 5% in monitoring dashboard
- Users reporting failures
- Sentry alerts triggered

**Investigation Steps:**

```bash
# 1. Check application logs
tail -f /var/log/zenai/error.log

# 2. Check error types
grep ERROR /var/log/zenai/app.log | cut -d' ' -f5 | sort | uniq -c

# 3. Check database connection
psql -U zenai_user -d zenai_prod -c "SELECT count(*) FROM pg_stat_activity;"

# 4. Check Redis connection
redis-cli ping

# 5. Check API key validity
curl -X GET https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Resolution Steps:**

1. If database connection issue:
   ```bash
   # Restart database
   sudo systemctl restart postgresql
   
   # Check connection pool
   psql -U zenai_user -d zenai_prod -c "SELECT * FROM pg_stat_activity;"
   ```

2. If Redis issue:
   ```bash
   # Restart Redis
   sudo systemctl restart redis-server
   
   # Check memory
   redis-cli INFO memory
   ```

3. If API key issue:
   ```bash
   # Update API key
   export OPENAI_API_KEY=sk-new-key
   
   # Restart application
   sudo systemctl restart zenai-orchestration
   ```

### Runbook 2: Slow Agent Execution

**Symptoms:**
- Agent execution time > 2x target
- Users experiencing delays
- Queue depth increasing

**Investigation Steps:**

```bash
# 1. Check agent execution times
grep "Agent completed" /var/log/zenai/app.log | tail -20

# 2. Check database query performance
psql -U zenai_user -d zenai_prod -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# 3. Check Redis latency
redis-cli --latency

# 4. Check system resources
top -b -n 1 | head -20
```

**Resolution Steps:**

1. If database slow:
   ```bash
   # Analyze and vacuum
   psql -U zenai_user -d zenai_prod -c "VACUUM ANALYZE;"
   
   # Check indexes
   psql -U zenai_user -d zenai_prod -c "SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;"
   ```

2. If Redis slow:
   ```bash
   # Check memory usage
   redis-cli INFO memory
   
   # Clear expired keys
   redis-cli FLUSHDB ASYNC
   ```

3. If system resources low:
   ```bash
   # Scale up application
   kubectl scale deployment zenai-orchestration --replicas=5 -n zenai
   
   # Or increase instance size
   ```

### Runbook 3: Job Queue Backlog

**Symptoms:**
- Queue depth > 100 jobs
- Jobs not processing
- Users unable to submit new jobs

**Investigation Steps:**

```bash
# 1. Check queue depth
redis-cli LLEN bullmq:queue:meeting_summarize

# 2. Check failed jobs
redis-cli LLEN bullmq:queue:meeting_summarize:failed

# 3. Check job details
redis-cli HGETALL bullmq:job:job_id

# 4. Check worker status
redis-cli HGETALL bullmq:worker:worker_id
```

**Resolution Steps:**

1. Increase concurrency:
   ```python
   # Update config
   BULLMQ_CONFIG['concurrency'] = 20
   
   # Restart workers
   sudo systemctl restart zenai-workers
   ```

2. Scale up workers:
   ```bash
   # Add more worker instances
   kubectl scale deployment zenai-workers --replicas=5 -n zenai
   ```

3. Clear failed jobs:
   ```bash
   # Move failed jobs back to queue
   redis-cli LRANGE bullmq:queue:meeting_summarize:failed 0 -1 | \
     xargs -I {} redis-cli RPUSH bullmq:queue:meeting_summarize {}
   ```

### Runbook 4: Database Connection Pool Exhausted

**Symptoms:**
- "Too many connections" errors
- New requests failing
- Existing connections hanging

**Investigation Steps:**

```bash
# 1. Check active connections
psql -U zenai_user -d zenai_prod -c "SELECT count(*) FROM pg_stat_activity;"

# 2. Check connection details
psql -U zenai_user -d zenai_prod -c "SELECT pid, usename, application_name, state FROM pg_stat_activity;"

# 3. Check pool configuration
grep "max_connections" /etc/postgresql/14/main/postgresql.conf
```

**Resolution Steps:**

1. Increase max connections:
   ```bash
   # Edit postgresql.conf
   sudo nano /etc/postgresql/14/main/postgresql.conf
   
   # Change max_connections
   max_connections = 200
   
   # Restart PostgreSQL
   sudo systemctl restart postgresql
   ```

2. Kill idle connections:
   ```sql
   -- Kill idle connections
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' 
   AND query_start < now() - interval '10 minutes';
   ```

3. Implement connection pooling:
   ```bash
   # Install PgBouncer
   sudo apt-get install pgbouncer
   
   # Configure and start
   sudo systemctl start pgbouncer
   ```

---

## Rollback Procedures

### Rollback to Previous Version

```bash
# 1. Identify previous version
git log --oneline | head -5

# 2. Checkout previous version
git checkout <commit-hash>

# 3. Rebuild application
npm run build

# 4. Restart application
sudo systemctl restart zenai-orchestration

# 5. Verify deployment
curl http://localhost:3000/health
```

### Database Rollback

```bash
# 1. List migrations
npx prisma migrate status

# 2. Rollback to previous migration
npx prisma migrate resolve --rolled-back <migration-name>

# 3. Revert migration
npx prisma migrate deploy

# 4. Verify database
psql -U zenai_user -d zenai_prod -c "SELECT * FROM _prisma_migrations ORDER BY finished_at DESC LIMIT 5;"
```

### Emergency Rollback

```bash
# 1. Stop current deployment
kubectl delete deployment zenai-orchestration -n zenai

# 2. Restore from backup
kubectl apply -f k8s/deployment-backup.yaml -n zenai

# 3. Verify services
kubectl get pods -n zenai
kubectl logs -f deployment/zenai-orchestration -n zenai

# 4. Notify team
# Send incident notification
```

---

## Maintenance Tasks

### Daily Tasks

- Monitor error rates and alerts
- Check database disk usage
- Review application logs for warnings

### Weekly Tasks

- Analyze performance metrics
- Review slow queries
- Check backup integrity

### Monthly Tasks

- Update dependencies
- Review security patches
- Capacity planning analysis
- Performance optimization review

### Quarterly Tasks

- Major version upgrades
- Disaster recovery testing
- Security audit
- Architecture review

---

## Support & Escalation

**For deployment issues:**
- Contact: devops@zenai.com
- Slack: #zenai-deployment
- On-call: Check PagerDuty

**For performance issues:**
- Contact: platform@zenai.com
- Slack: #zenai-performance

**For security issues:**
- Contact: security@zenai.com
- Slack: #zenai-security (private)
