# ZenAI - AI Project Manager Agent

An intelligent, multi-modal AI project management system. It analyzes meetings (audio/text), extracts tasks, and orchestrates workflows across Notion, Jira, Slack, and Microsoft Teams.

**Powered primarily by Google Gemini (1.5 Pro)** for massive context window and native multi-modal capabilities.

## 🚀 Features

- **Multi-Modal Meeting Analysis**: 
  - **Audio**: Transcribes and analyzes audio using **Gemini 1.5 Pro** (Native Multi-modal).
  - **Text**: Extracts insights from meeting transcripts.
- **Intelligent Task Extraction**: Automatically identifies Action Items, Decisions, Risks, and Blockers.
- **Cross-Platform Integration**: 
  - **Notion**: Full 2-way sync for Task Database.
  - **Jira**: Creates issues for blocked tasks and tracks status.
  - **Slack**: Sends real-time notifications and nudges.
  - **Microsoft Teams**: Delivers corporate alerts and updates.
- **Smart Follow-Up Agent**: 
  - Monitors task inactivity (stalled tasks).
  - Context-aware nudges (e.g., "Are you blocked?" vs "Update reminder").
  - Auto-escalates blockers to Jira.
- **Resilient Architecture**: Server starts gracefully even if optional integration keys (Notion, Email) are missing.
- **Real-time & Docker**: WebSocket updates and containerized deployment.

## 🛠️ Prerequisites

- Python 3.10+
- PostgreSQL (with `pgvector` extension)
- Redis
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

# --- Integrations ---
# Notion (Required for Task DB Sync)
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=...

# Email (Optional - for reports)
EMAIL_SERVICE=...

# Communication (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/...
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run Locally
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
│   ├── followup.py     # Checks inactivity, pings Slack/Teams
│   ├── summarizer.py   # Summarizes meetings
│   └── ...
├── integrations/       # External Adapters
│   ├── notion.py       # Task Database Sync
│   ├── jira.py         # Issue Tracking
│   ├── slack.py        # Messaging
│   └── teams.py        # Enterprise Chat
├── core/               # Shared Logic
│   └── audio.py        # Audio Processing
└── main.py             # FastAPI Entrypoint
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
pytest tests/
```

## 📚 Documentation
- [Agent Architecture](AGENT_ARCHITECTURE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)