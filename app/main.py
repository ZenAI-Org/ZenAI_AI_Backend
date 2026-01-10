from dotenv import load_dotenv 
import os 

# Load environment variables
load_dotenv() 

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import json
import tempfile
from app.integrations.notion_integration import NotionIntegration
from datetime import datetime
from app.services.email_service import EmailService 
from typing import List, Optional 
from app.middleware.error_middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from app.middleware.request_metrics import RequestMetricsMiddleware
from app.middleware.auth_middleware import AuthenticationMiddleware, AuthorizationMiddleware
from app.core.logger import get_logger
from app.api.routes import router as ai_workflows_router

# Socket.io imports
try:
    import socketio
    from fastapi_socketio import SocketManager
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    logger_init = get_logger(__name__)
    logger_init.warning("Socket.io not available, real-time updates disabled")

# Scheduler import
try:
    from app.core.scheduler import SchedulerService
    scheduler_service = SchedulerService()
except Exception as e:
    print(f"[ERROR] Scheduler initialization failed: {e}")
    scheduler_service = None

app = FastAPI(title="AI Project Manager Agent", version="1.0.0")

# Initialize Socket.io if available
if SOCKETIO_AVAILABLE:
    from app.queue.socketio_manager import get_socketio_manager
    from app.queue.socketio_handlers import get_socketio_handlers
    from app.queue.job_listeners import get_job_event_listener
    
    # Create Socket.io server
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        ping_timeout=60,
        ping_interval=25,
    )
    
    # Wrap FastAPI app with Socket.io ASGI app
    app = socketio.ASGIApp(sio, app)
    
    # Initialize Socket.io manager and handlers
    socketio_manager = get_socketio_manager()
    socketio_manager.set_socketio_server(sio)
    
    socketio_handlers = get_socketio_handlers()
    socketio_handlers.set_socketio_server(sio)
    
    # Register Socket.io listeners with job event listener
    job_event_listener = get_job_event_listener()
    socketio_manager.register_job_listeners(job_event_listener)

# Add middleware (order matters - auth should be before error handling)
app.add_middleware(AuthorizationMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RequestMetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Include AI workflows router
app.include_router(ai_workflows_router)

logger = get_logger(__name__)

# Request model
class MeetingRequest(BaseModel):
    meeting_text: str

# Response models
class TaskItem(BaseModel):
    title: str
    description: str
    assignee: Optional[str] = None
    priority: str  # High, Medium, Low
    due_date: Optional[str] = None

class MeetingAnalysis(BaseModel):
    key_decisions: List[str]
    action_items: List[TaskItem]
    risks_and_blockers: List[str]
    meeting_summary: str

# Initialize Groq (with error handling)
try:
    from groq import Groq
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    
    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # List available models and filter for chat-compatible ones
    models = groq_client.models.list()
    print(f"[DEBUG] Found {len(models.data)} available models")
    
    # Prefer these models in order
    preferred_models = [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'meta-llama/llama-4-scout-17b-16e-instruct',
        'gemma2-9b-it',
        'qwen/qwen3-32b'
    ]
    
    # Find the first available preferred model
    model_name = None
    for model in preferred_models:
        if any(m.id == model for m in models.data):
            model_name = model
            print(f"[INFO] Selected preferred model: {model_name}")
            break
    
    # Fallback to any model that might work if no preferred model found
    if not model_name:
        chat_models = [m.id for m in models.data if 'instruct' in m.id.lower() or 'chat' in m.id.lower()]
        if not chat_models:
            raise Exception("No suitable chat models available")
        model_name = chat_models[0]
        print(f"[INFO] Falling back to model: {model_name}")
    
    # Initialize the language model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0.1
    )
    print("[INFO] Groq client successfully initialized")
        
except Exception as e:
    print(f"[ERROR] Groq initialization failed: {e}")
    llm = None

# Initialize Audio Processor
try:
    from app.audio_processor import AudioProcessor
    audio_processor = AudioProcessor()
    print("[INFO] Audio processor initialized")
except Exception as e:
    print(f"[ERROR] Audio processor initialization failed: {e}")
    audio_processor = None

# Initialize Notion Integration
try:
    notion_integration = NotionIntegration()
    print("[INFO] Notion integration initialized")
except Exception as e:
    print(f"[ERROR] Notion integration initialization failed: {e}")
    notion_integration = None

# Initialize Email Service 
try: 
    email_service = EmailService()  
    print("[INFO] Email service initialized") 
except Exception as e: 
    print(f"[ERROR] Email service initialization failed: {e}") 
    email_service = None 

# Initialize Repetition Detector
try:
    from app.services.repetition_detector import RepetitionDetector
    # Need db connection. We can access it via notion_integration implicitly or created explicitly
    # For now, let's assume we can obtain it or pass existing dependencies.
    # Actually, main.py doesn't strictly hold the db connection object globally exposed easily.
    # However, PgvectorSetup does. 
    from app.core.pgvector_setup import PgvectorSetup
    pg_setup = PgvectorSetup()
    db_conn = pg_setup.get_connection()
    
    repetition_detector = RepetitionDetector(db_conn, notion_integration)
    print("[INFO] Repetition detector initialized")
except Exception as e:
    print(f"[ERROR] Repetition detector initialization failed: {e}")
    repetition_detector = None
    db_conn = None

# Initialize Security Manager
try:
    from app.core.security import SecurityManager
    # Reuse db_conn if available, otherwise try to create a new one
    if not db_conn:
        try:
            from app.core.pgvector_setup import PgvectorSetup
            pg_setup = PgvectorSetup()
            db_conn = pg_setup.get_connection()
        except:
            db_conn = None
            
    security_manager = SecurityManager(db_conn)
    print("[INFO] Security Manager initialized")
except Exception as e:
    print(f"[ERROR] Security Manager initialization failed: {e}")
    security_manager = None

@app.get("/")
async def root():
    return {
        "message": "AI Project Manager Agent is running!", 
        "groq_status": "connected" if llm else "failed",
        "audio_status": "enabled" if audio_processor else "disabled",
        "notion_status": "connected" if notion_integration else "disabled",
        "email_status" : "enabled" if email_service else "disabled",
        "scheduler_status": "running" if scheduler_service and scheduler_service.is_running else "stopped"
    }

@app.on_event("startup")
async def startup_event():
    """Start background services"""
    if scheduler_service:
        scheduler_service.start()
        logger.info("Scheduler service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background services"""
    if scheduler_service:
        scheduler_service.shutdown()
        logger.info("Scheduler service stopped")

@app.get("/api/v1/ai/healthz")
async def health_check():
    """Health check endpoint for production monitoring"""
    return {
        "status": "ok",
        "service": "ai-engine",
        "version": "1.0.0"
    } 
@app.get("/test-notion") 
async def test_notion():
    # Test notion connection directly 
    if not notion_integration:
        return {"error" : "Notion not initialized"} 
    
    try : 
        # Trying to query the database 
        response = notion_integration.client.databases.retrieve(
            database_id = notion_integration.database_id 
        ) 
        return { 
            "success" : True, 
            "database_title " : response.get("title", [{}])[0].get("plain_text", "Unknown" ), 
            "database_id " : notion_integration.database_id 
            } 
    except Exception as e : 
        return { 
            "success" : False, 
            "error" : str(e) 
        }

@app.post("/analyze-meeting")
async def analyze_meeting_text(request: MeetingRequest):
    """
    Analyze meeting transcript and extract structured information
    """
    
    if not llm:
        raise HTTPException(status_code=500, detail="Groq API not initialized")
    
    meeting_text = request.meeting_text
    
    if not meeting_text.strip():
        raise HTTPException(status_code=400, detail="meeting_text cannot be empty")
    
    prompt = f"""
    You are an AI Project Manager. Analyze this meeting transcript and extract:
    
    1. Key decisions made
    2. Action items (with assignee if mentioned, priority, due date if mentioned)
    3. Risks and blockers identified
    4. Brief meeting summary
    5. List of 3-5 Key Topics discussed (short phrases like "API Latency", "Login Page Design")
    
    Meeting transcript:
    {meeting_text}
    
    Format your response as JSON with this structure:
    {{
        "key_decisions": ["decision 1", "decision 2"],
        "action_items": [
            {{
                "title": "task title",
                "description": "detailed description",
                "assignee": "person name or null",
                "priority": "High/Medium/Low",
                "due_date": "date if mentioned or null"
            }}
        ],
        "risks_and_blockers": ["risk 1", "risk 2"],
        "meeting_summary": "brief summary of the meeting",
        "key_topics": ["Topic 1", "Topic 2"]
    }}
    
    Only return valid JSON, no additional text.
    """
    
    try:
        # Security Check
        project_id = "default-project" # Placeholder - normally from auth/request
        privacy_settings = {"sanitize_logs": False, "do_not_train": False}
        
        if security_manager:
            privacy_settings = security_manager.check_privacy_settings(project_id)
            if privacy_settings["do_not_train"]:
                print(f"[INFO] Privacy Mode Enabled for Project {project_id}: 'Do Not Train' active.")
        
        # Log input (sanitized if needed)
        log_content = meeting_text[:100]
        if privacy_settings["sanitize_logs"]:
            log_content = "[REDACTED]"
            
        print(f"[INFO] Analyzing meeting text: {log_content}...")
        
        # Call Groq API
        message = HumanMessage(content=prompt)
        response = llm([message])
        
        print(f"[DEBUG] Groq response received")
        
        # Clean the response to handle markdown code blocks
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
            
        # Parse the JSON response
        analysis = json.loads(content)
        
        # Repetition Detection Logic
        if repetition_detector and "key_topics" in analysis:
            project_id = "default-project" # Placeholder
            topics = analysis["key_topics"]
            
            # Detect
            recurring = repetition_detector.detect_recurring_topics(project_id, topics)
            analysis["recurring_issues"] = recurring
            
            # Store (async ideally, but sync for now)
            # Use a dummy meeting ID or generate one
            import uuid
            meeting_id = str(uuid.uuid4())
            repetition_detector.store_topics(project_id, topics, meeting_id)
        
        return analysis
    
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {e}"
        print(f"[ERROR] {error_msg}")
        print(f"[DEBUG] Raw response: {response.content}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    
    except Exception as e:
        error_msg = f"Error during meeting analysis: {e}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-meeting-audio")
async def analyze_meeting_audio(audio_file: UploadFile = File(...)):
    """
    Upload audio file, transcribe it, and analyze the meeting
    Supported formats: MP3, MP4, M4A, WAV, WebM
    """
    
    if not llm:
        raise HTTPException(status_code=500, detail="Groq API not initialized")
    
    if not audio_processor:
        raise HTTPException(status_code=500, detail="Audio processor not initialized")
    
    # Validate file format
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in audio_processor.supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Supported: {', '.join(audio_processor.supported_formats)}"
        )
    
    # Create temporary file
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            await audio_processor.save_upload_file(audio_file, temp_file_path)
        
        print(f"[INFO] Audio file saved: {temp_file_path}")
        
        # Transcribe audio
        transcript = audio_processor.transcribe_audio(temp_file_path)
        
        print(f"[INFO] Transcript: {transcript[:200]}...")
        
        # Analyze the transcript using existing logic
        prompt = f"""
        You are an AI Project Manager. Analyze this meeting transcript and extract:
        
        1. Key decisions made
        2. Action items (with assignee if mentioned, priority, due date if mentioned)
        3. Risks and blockers identified
        4. Brief meeting summary
        
        Meeting transcript:
        {transcript}
        
        Format your response as JSON with this structure:
        {{
            "key_decisions": ["decision 1", "decision 2"],
            "action_items": [
                {{
                    "title": "task title",
                    "description": "detailed description",
                    "assignee": "person name or null",
                    "priority": "High/Medium/Low",
                    "due_date": "date if mentioned or null"
                }}
            ],
            "risks_and_blockers": ["risk 1", "risk 2"],
            "meeting_summary": "brief summary of the meeting"
        }}
        
        Only return valid JSON, no additional text.
        """
        
        # Call Groq API
        message = HumanMessage(content=prompt)
        response = llm([message])
        
        # Clean and parse response
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        
        analysis = json.loads(content)
        
        # Add transcript to response
        analysis['transcript'] = transcript
        
        return analysis
    
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    
    except Exception as e:
        print(f"[ERROR] Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if temp_file_path:
            audio_processor.cleanup_file(temp_file_path)

@app.post("/analyze-and-sync")
async def analyze_and_sync_to_notion(request: MeetingRequest):
    """
    Analyze meeting text AND automatically create tasks in Notion
    """
    
    if not llm:
        raise HTTPException(status_code=500, detail="Groq API not initialized")
    
    if not notion_integration:
        raise HTTPException(status_code=500, detail="Notion integration not available")
    
    meeting_text = request.meeting_text
    
    if not meeting_text.strip():
        raise HTTPException(status_code=400, detail="meeting_text cannot be empty")
    
    # First, analyze the meeting (reuse existing logic)
    prompt = f"""
    You are an AI Project Manager. Analyze this meeting transcript and extract:
    
    1. Key decisions made
    2. Action items (with assignee if mentioned, priority, due date if mentioned)
    3. Risks and blockers identified
    4. Brief meeting summary
    5. List of 3-5 Key Topics discussed (short phrases like "API Latency", "Login Page Design")
    
    Meeting transcript:
    {meeting_text}
    
    Format your response as JSON with this structure:
    {{
        "key_decisions": ["decision 1", "decision 2"],
        "action_items": [
            {{
                "title": "task title",
                "description": "detailed description",
                "assignee": "person name or null",
                "priority": "High/Medium/Low",
                "due_date": "date if mentioned or null"
            }}
        ],
        "risks_and_blockers": ["risk 1", "risk 2"],
        "meeting_summary": "brief summary of the meeting",
        "key_topics": ["Topic 1", "Topic 2"]
    }}
    
    Only return valid JSON, no additional text.
    """
    
    try:
        print(f"[INFO] Analyzing meeting and syncing to Notion...")
        
        # Call Groq API
        message = HumanMessage(content=prompt)
        response = llm([message])
        
        # Clean and parse response
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        
        analysis = json.loads(content)
        
        # Create tasks in Notion
        notion_results = notion_integration.create_tasks_from_meeting(
            action_items=analysis.get("action_items", []),
            meeting_summary=analysis.get("meeting_summary", "Meeting"),
            meeting_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Add Notion sync results to response
        analysis["notion_sync"] = {
            "total_tasks": len(notion_results),
            "successful": sum(1 for r in notion_results if r.get("success")),
            "failed": sum(1 for r in notion_results if not r.get("success")),
            "tasks": notion_results
        }
        
        return analysis
    
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    
    except Exception as e:
        print(f"[ERROR] Analysis/sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")

import requests

@app.get("/dashboard")
async def get_dashboard():
    """Get project overview dashboard with task metrics"""
    
    if not notion_integration:
        raise HTTPException(status_code=500, detail="Notion not available")
    
    try:
        # Query all tasks with emails
        tasks = notion_integration.query_all_tasks_with_emails()
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed = sum(1 for t in tasks if t['status'] == "Done")
        in_progress = sum(1 for t in tasks if t['status'] == "In Progress")
        todo = sum(1 for t in tasks if t['status'] == "To Do")
        
        # Get today's date for overdue calculation
        from datetime import datetime, date
        today = date.today()
        
        overdue_count = 0
        
        for task in tasks:
            if task['due_date'] and task['status'] != "Done":
                try:
                    due = datetime.strptime(task['due_date'], "%Y-%m-%d").date()
                    if due < today:
                        task['is_overdue'] = True
                        overdue_count += 1
                    else:
                        task['is_overdue'] = False
                except:
                    task['is_overdue'] = False
            else:
                task['is_overdue'] = False
        
        return {
            "summary": {
                "total_tasks": total_tasks,
                "completed": completed,
                "in_progress": in_progress,
                "todo": todo,
                "overdue": overdue_count,
                "completion_rate": f"{(completed/total_tasks*100):.1f}%" if total_tasks > 0 else "0%"
            },
            "tasks": tasks
        }
    
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/tasks/overdue")
async def get_overdue_tasks():
    """Get list of overdue tasks with assignee emails"""
    
    if not notion_integration:
        raise HTTPException(status_code=500, detail="Notion not available")
    
    try:
        tasks = notion_integration.query_all_tasks_with_emails()
        
        from datetime import datetime, date
        today = date.today()
        
        overdue_tasks = []
        
        for task in tasks:
            if task['status'] == "Done":
                continue
            
            if task['due_date']:
                try:
                    due = datetime.strptime(task['due_date'], "%Y-%m-%d").date()
                    if due < today:
                        days_overdue = (today - due).days
                        overdue_tasks.append({
                            "title": task['title'],
                            "assignee": task['assignee_name'],
                            "assignee_email": task['assignee_email'],
                            "priority": task['priority'],
                            "due_date": task['due_date'],
                            "days_overdue": days_overdue,
                            "status": task['status'],
                            "url": task['url']
                        })
                except:
                    pass
        
        overdue_tasks.sort(key=lambda x: x['days_overdue'], reverse=True)
        
        return {
            "total_overdue": len(overdue_tasks),
            "tasks": overdue_tasks
        }
    
    except Exception as e:
        print(f"[ERROR] Overdue tasks error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/tasks/at-risk")
async def get_at_risk_tasks():
    """Get tasks at risk with assignee emails"""
    
    if not notion_integration:
        raise HTTPException(status_code=500, detail="Notion not available")
    
    try:
        tasks = notion_integration.query_all_tasks_with_emails()
        
        from datetime import datetime, date, timedelta
        today = date.today()
        risk_threshold = today + timedelta(days=2)
        
        at_risk_tasks = []
        
        for task in tasks:
            if task['status'] == "Done":
                continue
            
            if task['due_date']:
                try:
                    due = datetime.strptime(task['due_date'], "%Y-%m-%d").date()
                    
                    if today <= due <= risk_threshold:
                        days_until_due = (due - today).days
                        at_risk_tasks.append({
                            "title": task['title'],
                            "assignee": task['assignee_name'],
                            "assignee_email": task['assignee_email'],
                            "priority": task['priority'],
                            "due_date": task['due_date'],
                            "days_until_due": days_until_due,
                            "status": task['status'],
                            "url": task['url']
                        })
                except:
                    pass
        
        at_risk_tasks.sort(key=lambda x: x['days_until_due'])
        
        return {
            "total_at_risk": len(at_risk_tasks),
            "tasks": at_risk_tasks
        }
    
    except Exception as e:
        print(f"[ERROR] At-risk tasks error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.get("/reports/daily") 
async def generate_daily_report(): 
    """
        Generate daily progress report 
    """ 
    if not notion_integration: 
        raise HTTPException(status_code=500, detail="Notion not available") 
    try : 
        # Get dashboard data 
        dashboard = await get_dashboard() 
        overdue = await get_overdue_tasks()  
        at_risk = await get_at_risk_tasks() 
        
        from datetime import datetime 
        today = datetime.now().strftime("%A, %B %d, %Y") 
        
        # Build report 
        report_lines = [
            f"# Daily Project Report - {today}",
            "",
            "## Summary",
            f"- **Total Tasks**: {dashboard['summary']['total_tasks']}",
            f"- **Completed**: {dashboard['summary']['completed']} ({dashboard['summary']['completion_rate']})",
            f"- **In Progress**: {dashboard['summary']['in_progress']}",
            f"- **To Do**: {dashboard['summary']['todo']}",
            f"- **Overdue**: {dashboard['summary']['overdue']}",
            f"- **At Risk**: {len(at_risk['tasks'])}",
            "",
            f"## Overdue Tasks ({overdue['total_overdue']})",
            ""
        ] 
        
        if overdue['tasks']:
            for task in overdue['tasks']:
                report_lines.append(f"- **{task['title']}** ({task['assignee']}) - {task['days_overdue']} days overdue")
        else:
            report_lines.append("- No overdue tasks!")
        
        report_lines.extend([
            "",
            f"## At-Risk Tasks ({at_risk['total_at_risk']})",
            ""
        ]) 
          
        if at_risk['tasks']:
            for task in at_risk['tasks']:
                report_lines.append(f"- **{task['title']}** ({task['assignee']}) - due in {task['days_until_due']} days")
        else:
            report_lines.append("- No at-risk tasks")
        
        report_lines.extend([
            "",
            "## Team Workload",
            ""
        ])
        # gropus by assignee 
        assignee_counts = {}
        for task in dashboard['tasks']:
            assignee = task['assignee']
            if task['status'] != "Done":
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        for assignee, count in sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- **{assignee}**: {count} active tasks")
        
        # Join all lines
        report = "\n".join(report_lines)
        
        return { 
                "reported_date" : today, 
                "markdown" : report, 
                "summary" : dashboard['summary'], 
                "overdue_count" : overdue['total_overdue'], 
                "at_risk_count" : at_risk['total_at_risk']
            } 
    except Exception as e : 
        raise HTTPException(status_code = 500, details = f"Report generation failed: {str(e)}")
@app.post("/notifications/send-daily-report")
async def send_daily_report_email(email: Optional[str] = None):
    """
    Send daily report via email
    - If email is provided: send to that specific email
    - If no email: send to all team members who have tasks
    """
    
    if not email_service:
        raise HTTPException(status_code=500, detail="Email service not configured")
    
    try:
        # Generate report
        report_data = await generate_daily_report()
        
        results = []
        
        if email:
            # Send to specific email
            success = email_service.send_daily_report(
                report_markdown=report_data['markdown'],
                to_email=email
            )
            results.append({
                "email": email,
                "sent": success
            })
        else:
            # Send to all team members from Notion
            dashboard = await get_dashboard()
            
            # Collect unique assignee emails
            unique_emails = set()
            for task in dashboard['tasks']:
                if task.get('assignee_email'):
                    unique_emails.add((task['assignee_name'], task['assignee_email']))
            
            # Also send to notification email (you)
            notification_email = os.getenv("NOTIFICATION_EMAIL")
            if notification_email:
                unique_emails.add(("Team Lead", notification_email))
            
            # Send to each unique person
            for assignee_name, assignee_email in unique_emails:
                success = email_service.send_daily_report(
                    report_markdown=report_data['markdown'],
                    to_email=assignee_email
                )
                results.append({
                    "assignee": assignee_name,
                    "email": assignee_email,
                    "sent": success
                })
        
        return {
            "total_sent": len(results),
            "successful": sum(1 for r in results if r.get('sent')),
            "failed": sum(1 for r in results if not r.get('sent')),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/notifications/send-overdue-alerts")
async def send_overdue_alerts_email():
    """Send overdue task alerts to individual assignees using their Notion emails"""
    
    if not email_service:
        raise HTTPException(status_code=500, detail="Email service not configured")
    
    try:
        overdue = await get_overdue_tasks()
        
        results = []
        fallback_email = os.getenv("NOTIFICATION_EMAIL")
        
        for task in overdue['tasks']:
            # Use assignee's email from Notion, fallback to notification email
            target_email = task.get('assignee_email') or fallback_email
            
            if target_email:
                success = email_service.send_overdue_alert(
                    task_title=task['title'],
                    assignee=task['assignee'],
                    days_overdue=task['days_overdue'],
                    task_url=task['url'],
                    to_email=target_email
                )
                results.append({
                    "task": task['title'],
                    "assignee": task['assignee'],
                    "email": target_email,
                    "email_source": "notion" if task.get('assignee_email') else "fallback",
                    "sent": success
                })
            else:
                results.append({
                    "task": task['title'],
                    "assignee": task['assignee'],
                    "email": None,
                    "sent": False,
                    "error": "No email available"
                })
        
        return {
            "total_alerts": len(results),
            "successful": sum(1 for r in results if r.get('sent')),
            "failed": sum(1 for r in results if not r.get('sent')),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/notifications/send-at-risk-reminders")
async def send_at_risk_reminders_email():
    """Send deadline reminders using Notion emails"""
    
    if not email_service:
        raise HTTPException(status_code=500, detail="Email service not configured")
    
    try:
        at_risk = await get_at_risk_tasks()
        
        results = []
        fallback_email = os.getenv("NOTIFICATION_EMAIL")
        
        for task in at_risk['tasks']:
            target_email = task.get('assignee_email') or fallback_email
            
            if target_email:
                success = email_service.send_deadline_reminder(
                    task_title=task['title'],
                    assignee=task['assignee'],
                    days_until_due=task['days_until_due'],
                    task_url=task['url'],
                    to_email=target_email
                )
                results.append({
                    "task": task['title'],
                    "assignee": task['assignee'],
                    "email": target_email,
                    "email_source": "notion" if task.get('assignee_email') else "fallback",
                    "sent": success
                })
            else:
                results.append({
                    "task": task['title'],
                    "assignee": task['assignee'],
                    "email": None,
                    "sent": False,
                    "error": "No email available"
                })
        
        return {
            "total_reminders": len(results),
            "successful": sum(1 for r in results if r.get('sent')),
            "failed": sum(1 for r in results if not r.get('sent')),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.get("/api/projects/{project_id}/suggestions")
async def get_project_suggestions(project_id: str, force_refresh: bool = False):
    """
    Get AI-powered suggestions for a project dashboard.
    
    Args:
        project_id: Project ID
        force_refresh: Force refresh suggestions even if cached
        
    Returns:
        Dashboard suggestions grouped by card type
    """
    try:
        from app.agents.suggestions_agent import SuggestionsAgent
        from app.agents.base_agent import AgentConfig
        
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = SuggestionsAgent(config, db_connection=None)
        
        result = await agent.execute(
            project_id=project_id,
            force_refresh=force_refresh
        )
        
        if result.status.value == "error":
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "success": True,
            "data": result.data,
            "metadata": result.metadata
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to generate suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
