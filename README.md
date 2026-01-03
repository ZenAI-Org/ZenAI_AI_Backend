# ZenAI - AI Project Manager Agent

An intelligent AI-powered project management system that analyzes meetings, extracts tasks, and integrates with Notion for seamless workflow automation.

## Features

- Audio transcription and meeting analysis
- Automatic task extraction from meeting transcripts
- Notion integration for task management
- Project dashboard with metrics and insights
- Email notifications for overdue tasks
- Real-time updates via WebSocket
- Docker support for easy deployment

## Prerequisites

- Python 3.12 or higher
- PostgreSQL with pgvector extension
- Redis
- Docker and Docker Compose (optional, for containerized deployment)

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ZenAI-Org/ZenAI_AI_Backend
cd ZenAI
```

### 2. Set Up Environment Variables

Copy the example environment file and configure it with your credentials:

```bash
cp .env_example .env
```

Edit `.env` and fill in the required values (see [Required Environment Variables](#required-environment-variables) section below).

### 3. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL with pgvector

Ensure you have PostgreSQL running with the pgvector extension installed. You can use Docker:

```bash
docker run -d \
  --name zenai-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=zenai \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 5. Set Up Redis

```bash
docker run -d \
  --name zenai-redis \
  -p 6379:6379 \
  redis:7-alpine
```

### 6. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## Running Tests

The project includes comprehensive test coverage for all major components.

### Run All Tests

```bash
source venv/bin/activate
pytest
pytest -v
pytest --cov=app --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_summarization_agent.py
pytest tests/test_api_endpoints.py
pytest tests/test_orchestration_engine.py
```

### Run Tests by Category

```bash
pytest tests/test_end_to_end_workflows.py
pytest tests/test_error_handling.py
```

## Building the Docker Container

### Build the Docker Image

```bash
docker build -t zenai:latest .
docker build -t zenai:v1.0.0 .
```

### Run with Docker Compose

The easiest way to run the entire stack (app + PostgreSQL + Redis):

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
docker-compose down -v
```

### Run the Docker Container Manually

```bash
docker run -d \
  --name zenai-app \
  -p 8000:8000 \
  --env-file .env \
  zenai:latest
docker logs -f zenai-app
docker stop zenai-app
docker rm zenai-app
docker rm zenai-app
```

## Required Environment Variables

### AI API Keys

```bash
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=sk-your_openai_key_here
```

### OpenAI Configuration 

```bash
OPENAI_MODEL_NAME=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2048
OPENAI_TIMEOUT=300
```

### Notion Integration (for task sync)

```bash
NOTION_API_KEY=your_notion_api_key_here
NOTION_DATABASE_ID=your_notion_database_id_here
```

### Database Configuration 

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/zenai
```

### Redis Configuration
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### AWS S3 Configuration (for audio file storage)

```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket-name
```

### Email Configuration ( now these are optional) 

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_email_password
```

### Application Configuration

```bash
APP_ENV=development
LOG_LEVEL=INFO
```

## API Documentation

Once the application is running, you can access:

- **Interactive API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

## Project Structure

```
ZenAI/
├── app/
│   ├── agents/           # AI agents for different tasks
│   ├── api/              # API routes
│   ├── core/             # Core functionality
│   ├── integrations/     # External integrations (Notion, etc.)
│   ├── middleware/       # Request/response middleware
│   ├── queue/            # Job queue and orchestration
│   └── services/         # Business logic services
├── tests/                # Test suite
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Multi-container setup
├── requirements.txt      # Python dependencies
└── .env_example          # Environment variables template
```

## For further needs : 

- [Project Overview](PROJECT_OVERVIEW.md)
- [Agent Architecture](AGENT_ARCHITECTURE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)