# Migration Guide: ZenAI → zenai-ai-engine

## 📍 Repository Structure

**Working Repo (This one):** `/Users/shashankmk/Documents/Ubuntu-applications/ZenAI`
- This is where you develop and maintain the AI engine code
- All features are developed here
- Git commits and branches are managed here

**Production Repo:** `/Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine`
- This is the canonical repo for deployment
- You copy code from working repo to here for PRs
- This is what gets deployed

---

## 🚀 How to Move Code to Production Repo

### Step 1: Copy Files from Working Repo

```bash
# Navigate to production repo
cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine

# Copy all application files from working repo
cp -r /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/app .
cp -r /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/tests .
cp /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/Dockerfile .
cp /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/docker-compose.yml .
cp /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/requirements.txt .
cp /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/.env_example .
```

### Step 2: Create README.md in Production Repo

```bash
cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine

# Create README with run/test/build instructions
cat > README.md << 'EOF'
# ZenAI - AI Engine Service

## Quick Start

### Run with Docker
\`\`\`bash
cp .env_example .env
# Edit .env with your API keys
docker-compose up --build
\`\`\`

### Run Tests
\`\`\`bash
pytest tests/ -v
\`\`\`

### Build
\`\`\`bash
docker build -t zenai-ai-engine:latest .
\`\`\`

### Smoke Test
\`\`\`bash
# Health check
curl http://localhost:8000/

# API docs
curl http://localhost:8000/docs
\`\`\`

## Required Environment Variables
- OPENAI_API_KEY
- GROQ_API_KEY
- INTERNAL_API_KEY (generate with: openssl rand -hex 32)

See .env_example for complete list.
EOF
```

### Step 3: Create mockdata/ Directory

```bash
cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine
mkdir -p mockdata

# Create sample request
cat > mockdata/process_meeting_request.json << 'EOF'
{
  "meeting_id": "meeting_123",
  "s3_key": "meetings/audio/meeting_123.mp3",
  "project_id": "proj_456"
}
EOF

# Create sample response
cat > mockdata/process_meeting_response.json << 'EOF'
{
  "meeting_id": "meeting_123",
  "status": "processing",
  "jobs": {
    "transcription": "job_trans_001"
  }
}
EOF
```

### Step 4: Update .env_example with INTERNAL_API_KEY

```bash
cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine

# Add to .env_example
echo "" >> .env_example
echo "# IMPORTANT: Internal API Key for service-to-service authentication" >> .env_example
echo "# Generate with: openssl rand -hex 32" >> .env_example
echo "INTERNAL_API_KEY=your-secure-internal-api-key-here" >> .env_example
```

### Step 5: Commit and Push to Production Repo

```bash
cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine

# Check what's changed
git status

# Add all files
git add .

# Commit
git commit -m "feat: Complete AI engine implementation

- Add all AI agents (transcription, summarization, task extraction, AIPM, chat, suggestions)
- Add orchestration engine with batch processing
- Add performance optimization (Redis caching, pgvector indexing)
- Add comprehensive test suite
- Add Docker deployment configuration
- Add mockdata for testing"

# Push to create PR
git push origin main
# or push to a feature branch:
# git checkout -b feature/ai-engine-complete
# git push origin feature/ai-engine-complete
```

---

## 🔄 Workflow for Future Updates

1. **Develop in working repo** (`/Users/shashankmk/Documents/Ubuntu-applications/ZenAI`)
   - Make changes
   - Test locally
   - Commit to git

2. **Copy to production repo** when ready for PR
   ```bash
   # Copy updated files
   cp -r /Users/shashankmk/Documents/Ubuntu-applications/ZenAI/app /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine/
   # ... copy other changed files
   ```

3. **Create PR in production repo**
   ```bash
   cd /Users/shashankmk/Documents/ZenAi-Production-/zenai-ai-engine
   git checkout -b feature/your-feature-name
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

---

## ✅ Verification Checklist

After copying to production repo:

- [ ] All files copied successfully
- [ ] README.md created with run/test/build/smoke-test instructions
- [ ] .env_example has INTERNAL_API_KEY note
- [ ] mockdata/ directory created with sample JSON files
- [ ] Docker builds: `docker build -t zenai-ai-engine .`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Service starts: `docker-compose up`
- [ ] Health check works: `curl http://localhost:8000/`

---

**Remember:** This working repo (ZenAI) is your source of truth. Always develop here first, then copy to production repo for PRs.
