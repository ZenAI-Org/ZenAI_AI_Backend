# ZenAI - AI Project Manager Agent

An intelligent, multi-modal AI project management system. It analyzes meetings (audio/text), extracts tasks, and orchestrates workflows across Notion, Jira, Slack, and Microsoft Teams.

Supported by **Google Gemini (1.5 Pro)** (Default), OpenAI, and Groq.

## 🚀 Features

- **Multi-Modal Meeting Analysis**: 
  - **Audio**: Transcribes and analyzes audio using **Gemini 1.5 Pro** (Native Multi-modal) or OpenAI Whisper.
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
- **Flexible AI Backend**: Defaulting to **Gemini 1.5 Pro** for massive context window and multi-modal capabilities. Compatible with OpenAI and Groq.
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
# Primary Brain
LLM_PROVIDER=gemini 
GOOGLE_API_KEY=AI...

# Fallbacks (Optional)
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# --- Integrations ---
# Notion (Required for Task DB)
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=...

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
│   ├── summarizer.py   # Summarizes meetings (LangChain)
│   └── ...
├── integrations/       # External Adapters
│   ├── notion.py       # Task Database Sync
│   ├── jira.py         # Issue Tracking
│   ├── slack.py        # Messaging
│   └── teams.py        # Enterprise Chat
├── core/               # Shared Logic
│   └── audio.py        # Whisper / Gemini Audio Processing
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