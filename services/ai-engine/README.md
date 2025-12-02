# ZenAI - AI Engine Service

AI orchestration engine for the ZenAI platform. Handles meeting transcription, summarization, task extraction, and intelligent chat capabilities.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- OpenAI API key
- Groq API key

### Run with Docker (Recommended)

```bash
# 1. Set up environment variables
cp .env_example .env
# Edit .env and add your API keys

# 2. Build and run
docker-compose up --build

# Service available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Run Locally (Development)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env_example .env
# Edit .env with your keys

# 4. Ensure PostgreSQL and Redis are running
# Then start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔑 Required Environment Variables

**Critical - Must be set:**
```bash
OPENAI_API_KEY=sk-your-key-here
GROQ_API_KEY=your-groq-key-here
INTERNAL_API_KEY=your-secure-internal-key  # Generate with: openssl rand -hex 32
```

**Optional:**
```bash
NOTION_API_KEY=...          # For Notion integration
AWS_ACCESS_KEY_ID=...       # For S3 audio storage
SMTP_HOST=...               # For email notifications
```

See `.env_example` for complete list.

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
pytest tests/test_transcription_agent.py -v
pytest tests/test_end_to_end_workflows.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

## 🏗️ Build

### Build Docker Image
```bash
docker build -t zenai-ai-engine:latest .
```

### Build for Production
```bash
docker build -t zenai-ai-engine:prod .
```

## 🔍 Smoke Tests

### Health Check
```bash
curl http://localhost:8000/
```

Expected: `{"status":"ok"}`

### API Documentation
```bash
curl http://localhost:8000/docs
```

Expected: OpenAPI documentation page

### Process Meeting (Example)
```bash
curl -X POST http://localhost:8000/api/meetings/meeting_123/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INTERNAL_API_KEY}" \
  -d @mockdata/process_meeting_request.json
```

Expected: Job IDs for transcription, summarization, task extraction

### Chat Query (Example)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INTERNAL_API_KEY}" \
  -d @mockdata/chat_request.json
```

Expected: AI-generated response with context

### Get Project Insights
```bash
curl http://localhost:8000/api/projects/proj_456/aipm \
  -H "Authorization: Bearer ${INTERNAL_API_KEY}"
```

Expected: Project health analysis and recommendations

## 📁 Project Structure

```
ai-engine/
├── app/                    # FastAPI application
│   ├── agents/            # AI agents (transcription, summarization, etc.)
│   ├── api/               # REST API routes
│   ├── core/              # Core utilities (embeddings, context, etc.)
│   ├── middleware/        # Auth, logging, error handling
│   ├── queue/             # Job queue and orchestration
│   └── main.py            # Application entry point
├── tests/                 # Test suite
├── mockdata/              # Sample request/response files
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
├── .env_example          # Environment variable template
└── README.md             # This file
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/docs` | GET | API documentation |
| `/api/meetings/:id/process` | POST | Process meeting audio |
| `/api/chat` | POST | Chat with AI about projects |
| `/api/projects/:id/aipm` | GET | Get project insights |
| `/api/projects/:id/suggestions` | GET | Get dashboard suggestions |

## 🛠️ Development

### Code Style
```bash
# Format code
black app/ tests/

# Lint
flake8 app/ tests/
```

### Database Migrations
```bash
python -m app.core.migrations
```

### View Logs
```bash
# Docker
docker-compose logs -f

# Local
tail -f logs/app.log
```

## 🐛 Troubleshooting

**Issue: Port 8000 already in use**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

**Issue: Database connection failed**
- Ensure PostgreSQL is running
- Check DATABASE_URL in .env
- Verify pgvector extension is installed

**Issue: Redis connection failed**
- Ensure Redis is running
- Check REDIS_HOST and REDIS_PORT in .env

## 📊 Performance Metrics

- Transcription: < 2x audio duration
- Summarization: < 30 seconds
- Chat response: < 5 seconds
- Cache hit rate: > 70%

## 🔐 Security Notes

- **INTERNAL_API_KEY**: Used for service-to-service authentication. Keep secret!
- Never commit `.env` file
- Rotate API keys regularly
- Use HTTPS in production

## 📝 License

Proprietary - ZenAI Platform

---

**Questions?** Contact the ZenAI team or check the documentation.
