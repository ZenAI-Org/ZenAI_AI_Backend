# Configuration Guide

## Environment Variables

This document describes all environment variables required for the AI Agent Orchestration system.

### Quick Start

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

---

## Environment Variables Reference

### Application Configuration

#### NODE_ENV
- **Type:** String
- **Required:** Yes
- **Values:** `development`, `staging`, `production`
- **Default:** `development`
- **Description:** Application environment
- **Example:** `NODE_ENV=production`

#### PORT
- **Type:** Number
- **Required:** No
- **Default:** `3000`
- **Description:** Port for Express server
- **Example:** `PORT=3000`

#### API_BASE_URL
- **Type:** String
- **Required:** Yes
- **Description:** Base URL for API endpoints
- **Example:** `API_BASE_URL=https://api.zenai.com`

#### LOG_LEVEL
- **Type:** String
- **Required:** No
- **Values:** `debug`, `info`, `warn`, `error`
- **Default:** `info`
- **Description:** Logging level
- **Example:** `LOG_LEVEL=info`

---

### Database Configuration

#### DATABASE_URL
- **Type:** String
- **Required:** Yes
- **Format:** `postgresql://user:password@host:port/database`
- **Description:** PostgreSQL connection string
- **Example:** `DATABASE_URL=postgresql://zenai_user:password@localhost:5432/zenai_prod`
- **Notes:**
  - Must include pgvector extension
  - Use connection pooling for production
  - Ensure SSL is enabled for remote connections

#### DATABASE_POOL_SIZE
- **Type:** Number
- **Required:** No
- **Default:** `20`
- **Description:** Maximum database connections
- **Example:** `DATABASE_POOL_SIZE=50`

#### DATABASE_POOL_TIMEOUT
- **Type:** Number
- **Required:** No
- **Default:** `30000` (milliseconds)
- **Description:** Connection pool timeout
- **Example:** `DATABASE_POOL_TIMEOUT=30000`

#### DATABASE_SSL
- **Type:** Boolean
- **Required:** No
- **Default:** `true` (production), `false` (development)
- **Description:** Enable SSL for database connections
- **Example:** `DATABASE_SSL=true`

---

### Redis Configuration

#### REDIS_URL
- **Type:** String
- **Required:** Yes
- **Format:** `redis://[:password@]host:port[/db]`
- **Description:** Redis connection string
- **Example:** `REDIS_URL=redis://localhost:6379`
- **Notes:**
  - Use Redis 7+ for best performance
  - Enable persistence for production
  - Use Sentinel or Cluster for high availability

#### REDIS_PASSWORD
- **Type:** String
- **Required:** No
- **Description:** Redis password (if not in URL)
- **Example:** `REDIS_PASSWORD=secure_password`

#### REDIS_DB
- **Type:** Number
- **Required:** No
- **Default:** `0`
- **Description:** Redis database number
- **Example:** `REDIS_DB=0`

#### REDIS_CACHE_TTL
- **Type:** Number
- **Required:** No
- **Default:** `3600` (seconds)
- **Description:** Default cache TTL
- **Example:** `REDIS_CACHE_TTL=3600`

---

### OpenAI Configuration

#### OPENAI_API_KEY
- **Type:** String
- **Required:** Yes
- **Description:** OpenAI API key
- **Example:** `OPENAI_API_KEY=sk-...`
- **Notes:**
  - Keep this secret, never commit to version control
  - Rotate regularly for security
  - Monitor usage to avoid unexpected charges

#### OPENAI_MODEL
- **Type:** String
- **Required:** No
- **Default:** `gpt-4`
- **Values:** `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Description:** Default OpenAI model
- **Example:** `OPENAI_MODEL=gpt-4`

#### OPENAI_TEMPERATURE
- **Type:** Number
- **Required:** No
- **Default:** `0.7`
- **Range:** `0.0` - `2.0`
- **Description:** Model temperature (creativity)
- **Example:** `OPENAI_TEMPERATURE=0.7`

#### OPENAI_MAX_TOKENS
- **Type:** Number
- **Required:** No
- **Default:** `2000`
- **Description:** Maximum tokens per request
- **Example:** `OPENAI_MAX_TOKENS=2000`

#### OPENAI_TIMEOUT
- **Type:** Number
- **Required:** No
- **Default:** `30000` (milliseconds)
- **Description:** API request timeout
- **Example:** `OPENAI_TIMEOUT=30000`

#### OPENAI_RATE_LIMIT
- **Type:** Number
- **Required:** No
- **Default:** `3500` (requests per minute)
- **Description:** Rate limit for API calls
- **Example:** `OPENAI_RATE_LIMIT=3500`

---

### AWS Configuration

#### AWS_ACCESS_KEY_ID
- **Type:** String
- **Required:** Yes
- **Description:** AWS access key
- **Example:** `AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE`
- **Notes:**
  - Use IAM user with minimal permissions
  - Rotate keys regularly

#### AWS_SECRET_ACCESS_KEY
- **Type:** String
- **Required:** Yes
- **Description:** AWS secret access key
- **Example:** `AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
- **Notes:**
  - Keep this secret, never commit to version control
  - Use environment variables or AWS credentials file

#### AWS_REGION
- **Type:** String
- **Required:** No
- **Default:** `us-east-1`
- **Description:** AWS region
- **Example:** `AWS_REGION=us-east-1`

#### AWS_S3_BUCKET
- **Type:** String
- **Required:** Yes
- **Description:** S3 bucket for meeting audio
- **Example:** `AWS_S3_BUCKET=zenai-meetings`
- **Notes:**
  - Bucket must exist and be accessible
  - Enable versioning for backup
  - Set lifecycle policies for old files

#### AWS_S3_REGION
- **Type:** String
- **Required:** No
- **Default:** Same as AWS_REGION
- **Description:** S3 bucket region
- **Example:** `AWS_S3_REGION=us-east-1`

#### AWS_S3_PREFIX
- **Type:** String
- **Required:** No
- **Default:** `meetings/`
- **Description:** S3 key prefix for uploads
- **Example:** `AWS_S3_PREFIX=meetings/`

---

### Notion Integration

#### NOTION_API_KEY
- **Type:** String
- **Required:** No
- **Description:** Notion API key (if using Notion integration)
- **Example:** `NOTION_API_KEY=secret_...`
- **Notes:**
  - Only required if Notion integration is enabled
  - Create integration in Notion workspace settings
  - Grant necessary permissions to databases

#### NOTION_API_VERSION
- **Type:** String
- **Required:** No
- **Default:** `2024-01-01`
- **Description:** Notion API version
- **Example:** `NOTION_API_VERSION=2024-01-01`

#### NOTION_TIMEOUT
- **Type:** Number
- **Required:** No
- **Default:** `30000` (milliseconds)
- **Description:** Notion API timeout
- **Example:** `NOTION_TIMEOUT=30000`

---

### Authentication & Security

#### JWT_SECRET
- **Type:** String
- **Required:** Yes
- **Description:** Secret key for JWT signing
- **Example:** `JWT_SECRET=your-super-secret-key`
- **Notes:**
  - Use strong, random value (min 32 characters)
  - Rotate regularly
  - Never commit to version control

#### JWT_EXPIRY
- **Type:** String
- **Required:** No
- **Default:** `24h`
- **Description:** JWT token expiry time
- **Example:** `JWT_EXPIRY=24h`

#### JWT_REFRESH_EXPIRY
- **Type:** String
- **Required:** No
- **Default:** `7d`
- **Description:** Refresh token expiry time
- **Example:** `JWT_REFRESH_EXPIRY=7d`

#### CORS_ORIGIN
- **Type:** String
- **Required:** No
- **Default:** `http://localhost:3000`
- **Description:** CORS allowed origins (comma-separated)
- **Example:** `CORS_ORIGIN=https://app.zenai.com,https://staging.zenai.com`

#### CORS_CREDENTIALS
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Allow credentials in CORS requests
- **Example:** `CORS_CREDENTIALS=true`

---

### Socket.io Configuration

#### SOCKET_IO_CORS_ORIGIN
- **Type:** String
- **Required:** No
- **Default:** `http://localhost:3000`
- **Description:** Socket.io CORS origin
- **Example:** `SOCKET_IO_CORS_ORIGIN=https://app.zenai.com`

#### SOCKET_IO_TRANSPORTS
- **Type:** String
- **Required:** No
- **Default:** `websocket,polling`
- **Description:** Socket.io transports (comma-separated)
- **Example:** `SOCKET_IO_TRANSPORTS=websocket,polling`

#### SOCKET_IO_PING_INTERVAL
- **Type:** Number
- **Required:** No
- **Default:** `25000` (milliseconds)
- **Description:** Socket.io ping interval
- **Example:** `SOCKET_IO_PING_INTERVAL=25000`

#### SOCKET_IO_PING_TIMEOUT
- **Type:** Number
- **Required:** No
- **Default:** `60000` (milliseconds)
- **Description:** Socket.io ping timeout
- **Example:** `SOCKET_IO_PING_TIMEOUT=60000`

---

### BullMQ Configuration

#### BULLMQ_CONCURRENCY
- **Type:** Number
- **Required:** No
- **Default:** `5`
- **Description:** Number of concurrent jobs
- **Example:** `BULLMQ_CONCURRENCY=10`
- **Notes:**
  - Increase for more parallel processing
  - Consider API rate limits and system resources

#### BULLMQ_MAX_ATTEMPTS
- **Type:** Number
- **Required:** No
- **Default:** `3`
- **Description:** Maximum job retry attempts
- **Example:** `BULLMQ_MAX_ATTEMPTS=3`

#### BULLMQ_BACKOFF_DELAY
- **Type:** Number
- **Required:** No
- **Default:** `2000` (milliseconds)
- **Description:** Initial backoff delay for retries
- **Example:** `BULLMQ_BACKOFF_DELAY=2000`

#### BULLMQ_BACKOFF_TYPE
- **Type:** String
- **Required:** No
- **Default:** `exponential`
- **Values:** `exponential`, `fixed`
- **Description:** Backoff strategy
- **Example:** `BULLMQ_BACKOFF_TYPE=exponential`

#### BULLMQ_REMOVE_ON_COMPLETE
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Remove completed jobs from queue
- **Example:** `BULLMQ_REMOVE_ON_COMPLETE=true`

---

### Logging & Monitoring

#### SENTRY_DSN
- **Type:** String
- **Required:** No
- **Description:** Sentry error tracking DSN
- **Example:** `SENTRY_DSN=https://...@sentry.io/...`
- **Notes:**
  - Only required if using Sentry
  - Create project in Sentry dashboard

#### SENTRY_ENVIRONMENT
- **Type:** String
- **Required:** No
- **Default:** Same as NODE_ENV
- **Description:** Sentry environment
- **Example:** `SENTRY_ENVIRONMENT=production`

#### SENTRY_TRACES_SAMPLE_RATE
- **Type:** Number
- **Required:** No
- **Default:** `0.1`
- **Range:** `0.0` - `1.0`
- **Description:** Sentry traces sample rate
- **Example:** `SENTRY_TRACES_SAMPLE_RATE=0.1`

#### LOG_FORMAT
- **Type:** String
- **Required:** No
- **Default:** `json`
- **Values:** `json`, `text`
- **Description:** Log format
- **Example:** `LOG_FORMAT=json`

#### LOG_FILE_PATH
- **Type:** String
- **Required:** No
- **Default:** `/var/log/zenai/app.log`
- **Description:** Log file path
- **Example:** `LOG_FILE_PATH=/var/log/zenai/app.log`

---

### Feature Flags

#### ENABLE_NOTION_SYNC
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Enable Notion integration
- **Example:** `ENABLE_NOTION_SYNC=true`

#### ENABLE_CHAT_AGENT
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Enable chat agent
- **Example:** `ENABLE_CHAT_AGENT=true`

#### ENABLE_AIPM_AGENT
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Enable AIPM agent
- **Example:** `ENABLE_AIPM_AGENT=true`

#### ENABLE_SUGGESTIONS_AGENT
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Enable suggestions agent
- **Example:** `ENABLE_SUGGESTIONS_AGENT=true`

---

## Configuration Examples

### Development Environment

```bash
NODE_ENV=development
PORT=3000
API_BASE_URL=http://localhost:3000

DATABASE_URL=postgresql://zenai_user:password@localhost:5432/zenai_dev
REDIS_URL=redis://localhost:6379

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=zenai-dev-meetings

JWT_SECRET=dev-secret-key
LOG_LEVEL=debug

BULLMQ_CONCURRENCY=2
```

### Staging Environment

```bash
NODE_ENV=staging
PORT=3000
API_BASE_URL=https://staging-api.zenai.com

DATABASE_URL=postgresql://zenai_user:password@staging-db.zenai.com:5432/zenai_staging
REDIS_URL=redis://staging-redis.zenai.com:6379

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=zenai-staging-meetings

JWT_SECRET=staging-secret-key
LOG_LEVEL=info

SENTRY_DSN=https://...@sentry.io/...
BULLMQ_CONCURRENCY=5
```

### Production Environment

```bash
NODE_ENV=production
PORT=3000
API_BASE_URL=https://api.zenai.com

DATABASE_URL=postgresql://zenai_user:password@prod-db.zenai.com:5432/zenai_prod
DATABASE_POOL_SIZE=50
DATABASE_SSL=true
REDIS_URL=redis://:password@prod-redis.zenai.com:6379

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_RATE_LIMIT=3500

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=zenai-prod-meetings

JWT_SECRET=prod-secret-key
LOG_LEVEL=warn

SENTRY_DSN=https://...@sentry.io/...
SENTRY_TRACES_SAMPLE_RATE=0.1
BULLMQ_CONCURRENCY=20
```

---

## Configuration Validation

### Validate Configuration

```bash
# Check all required variables are set
npm run validate:config

# Or manually check
node -e "
const required = [
  'DATABASE_URL',
  'REDIS_URL',
  'OPENAI_API_KEY',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'JWT_SECRET'
];

required.forEach(key => {
  if (!process.env[key]) {
    console.error(\`Missing required: \${key}\`);
    process.exit(1);
  }
});

console.log('All required variables set');
"
```

---

## Security Best Practices

1. **Never commit secrets to version control**
   - Use `.env` file (add to `.gitignore`)
   - Use environment variables in CI/CD

2. **Rotate secrets regularly**
   - API keys: Every 90 days
   - JWT secret: Every 6 months
   - Database password: Every 90 days

3. **Use strong values**
   - Minimum 32 characters for secrets
   - Use random, non-dictionary values
   - Avoid personal information

4. **Restrict access**
   - Limit who can view/modify environment variables
   - Use IAM roles for AWS credentials
   - Use secrets management tools (Vault, AWS Secrets Manager)

5. **Monitor usage**
   - Track API key usage
   - Alert on unusual activity
   - Review access logs regularly

---

## Troubleshooting

### Configuration Not Loading

```bash
# Check if .env file exists
ls -la .env

# Check if variables are set
echo $DATABASE_URL

# Check if .env is in .gitignore
grep .env .gitignore
```

### Invalid Configuration Values

```bash
# Validate database URL
psql $DATABASE_URL -c "SELECT 1;"

# Validate Redis connection
redis-cli -u $REDIS_URL ping

# Validate OpenAI API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Environment Variable Precedence

Variables are loaded in this order (later overrides earlier):
1. `.env` file
2. System environment variables
3. Command-line arguments

```bash
# Override with environment variable
DATABASE_URL=postgresql://... npm start

# Or set in shell
export DATABASE_URL=postgresql://...
npm start
```

---

## Support

For configuration issues:
- Check logs: `tail -f /var/log/zenai/app.log`
- Review configuration: `npm run validate:config`
- Contact: devops@zenai.com
