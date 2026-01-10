# ZenAI - AI Project Manager Agent

An intelligent, multi-modal AI project management system with autonomous follow-up capabilities. It analyzes meetings (audio/text), extracts tasks, orchestrates workflows across Notion, Jira, Slack, and Microsoft Teams, and proactively monitors task health.

**Powered primarily by Google Gemini (1.5 Pro)** for massive context window and native multi-modal capabilities.

## 🚀 Features

### **Core Capabilities**
- **Multi-Modal Meeting Analysis**: 
  - **Audio**: Transcribes and analyzes audio using **Gemini 1.5 Pro** (Native Multi-modal).
  - **Text**: Extracts insights from meeting transcripts.
- **Intelligent Task Extraction**: Automatically identifies Action Items, Decisions, Risks, and Blockers.
- **Cross-Platform Integration**: 
  - **Notion**: Full 2-way sync for Task Database.
  - **Jira**: Creates issues for blocked tasks and tracks status.
  - **Slack**: Sends real-time notifications and nudges.
  - **Microsoft Teams**: Delivers corporate alerts and updates.

### **🤖 Autonomous Follow-up Agent**
- **Inactivity Detection**: Automatically flags tasks not updated in > 3 days
- **Contextual Awareness**: Checks if stalled tasks were mentioned as blocked in recent meetings
- **Actionable Nudges**: Sends tailored emails:
  - "Update reminder" for general inactivity
  - "Need help?" for potentially blocked tasks
- **Daily Reporting**: Generates comprehensive reports summarizing:
  - Active tasks
  - Inactive tasks
  - Blocked tasks
  - Team workload distribution
- **Repetition Detector**: Identifies recurring issues and patterns across meetings

### **🔒 Security & Privacy**
- **Privacy Mode**: Projects ending in `-secure` have:
  - Logs automatically redacted
  - `do_not_train` flag set to true for AI providers
- **Vector Isolation**: Separate embedding storage for secure projects
- **Fail-safe Mechanisms**: Graceful degradation when services are unavailable

### **🧠 Enhanced AI Insights**
- **Explainable AI**: Every suggestion includes:
  - **Why**: Reasoning behind the recommendation
  - **How Sure**: Confidence level (0-100%)
- **Context-Aware Analysis**: Leverages meeting history and task patterns

### **⚙️ System Features**
- **Resilient Architecture**: Server starts gracefully even if optional integration keys (Notion, Email) are missing
- **Scheduler Service**: Background task automation with APScheduler
- **Real-time Updates**: WebSocket support for live notifications
- **Docker Support**: Containerized deployment ready

## 🛠️ Prerequisites

- Python 3.10+
- PostgreSQL (with `pgvector` extension) - Required for advanced features
- Redis - For job queue management
- Docker (optional)

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ZenAI-Org/ZenAI_AI_Backend
cd ZenAI_AI_Backend
```

### 2. Configure Environment
Copy the example env file and update it with your keys:
```bash
cp .env_example .env
```

**Key Configuration Options (`.env`):**
```ini
# --- AI Providers ---
# Primary Brain (Required)
GOOGLE_API_KEY=AI...

# --- Database (Required for advanced features) ---
DATABASE_URL=postgresql://user:password@localhost:5432/zenai
REDIS_URL=redis://localhost:6379

# --- Integrations ---
# Notion (Required for Task DB Sync)
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=...

# Email (Required for Follow-up Agent nudges)
EMAIL_SERVICE=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Communication (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/...

# Issue Tracking (Optional)
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=email@example.com
JIRA_API_TOKEN=...
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
# Run database migrations
python -m app.core.pgvector_setup
```

### 5. Run Locally
```bash
uvicorn app.main:app --reload
```
The API will be available at: `http://localhost:8000`
Interactive Docs: `http://localhost:8000/docs`

## 🐳 Docker Support

Run the entire stack (App, Postgres, Redis) with one command:
```bash
docker-compose up -d
```

## 🏗️ Architecture

ZenAI follows a modular Agentic architecture:

```
app/
├── agents/             # Intelligent Workers
│   ├── followup_agent.py      # Monitors task inactivity & sends nudges
│   ├── summarization_agent.py # Summarizes meetings
│   ├── task_extraction_agent.py # Extracts action items
│   ├── suggestions_agent.py   # Generates AI suggestions with explanations
│   └── aipm_agent.py          # AI Project Manager analysis
├── integrations/       # External Adapters
│   ├── notion_integration.py  # Task Database Sync
│   ├── jira_integration.py    # Issue Tracking
│   ├── slack_integration.py   # Messaging
│   └── teams_integration.py   # Enterprise Chat
├── core/               # Shared Logic
│   ├── audio_processor.py     # Gemini Audio Processing
│   ├── scheduler.py           # Background task scheduling
│   ├── security.py            # Privacy & security manager
│   └── pgvector_setup.py      # Vector database setup
├── services/           # Business Logic
│   ├── repetition_detector.py # Identifies recurring patterns
│   └── email_service.py       # Email notifications
├── api/                # REST API
│   └── routes.py              # All endpoints
└── main.py             # FastAPI Entrypoint
```

## 📡 Key Endpoints

### Meeting Analysis
- `POST /analyze-meeting` - Analyze text transcript
- `POST /analyze-meeting-audio` - Upload & analyze audio file
- `POST /api/meetings/{meeting_id}/process` - Full meeting workflow

### Task Management
- `GET /api/projects/{project_id}/suggestions` - Get AI suggestions
- `POST /api/chat` - Chat with AI about project

### Follow-up Agent
- `POST /agents/followup/run` - Manually trigger follow-up check
- `GET /reports/daily` - Generate daily task report

### Monitoring
- `GET /health` - Health check with queue stats
- `GET /errors/metrics` - Error metrics dashboard

## 🧪 Testing

Run the comprehensive test suite:
```bash
pytest tests/
```

Specific test suites:
```bash
pytest tests/test_followup_agent.py      # Follow-up agent tests
pytest tests/test_security_manager.py    # Security features
pytest tests/test_repetition_detector.py # Pattern detection
```

## 🔐 Security Features

### Privacy Mode
Projects with names ending in `-secure` automatically get:
- Redacted logs (sensitive data removed)
- Isolated vector storage
- `do_not_train` flag for AI providers

### Best Practices
- Store API keys in `.env` (never commit)
- Use environment-specific configurations
- Enable HTTPS in production
- Regularly rotate API tokens

## 📊 Monitoring & Observability

The system provides comprehensive monitoring:
- **Queue Stats**: Track job processing metrics
- **Error Metrics**: Monitor error rates and patterns
- **Scheduler Status**: View background task health
- **Service Health**: Check all integration statuses

Access the health dashboard at: `http://localhost:8000/health`

## 🚀 Deployment

### Production Checklist
- [ ] Set `debug=False` in FastAPI initialization
- [ ] Configure production database (PostgreSQL)
- [ ] Set up Redis for job queue
- [ ] Enable HTTPS/SSL
- [ ] Configure email service for notifications
- [ ] Set up monitoring/alerting
- [ ] Review security settings

### Environment Variables
Ensure all required variables are set in production:
- `GOOGLE_API_KEY`
- `DATABASE_URL`
- `REDIS_URL`
- `NOTION_API_KEY` (if using Notion)
- Email configuration (if using Follow-up Agent)

## 📚 Documentation
- [Agent Architecture](AGENT_ARCHITECTURE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Security Best Practices](SECURITY.md)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

## 📝 License

[Your License Here]

## 🆘 Support

For issues and questions:
- GitHub Issues: [Report a bug](https://github.com/ZenAI-Org/ZenAI_AI_Backend/issues)
- Documentation: [Read the docs](https://github.com/ZenAI-Org/ZenAI_AI_Backend/wiki)

---

**Built with ❤️ using Google Gemini 1.5 Pro**