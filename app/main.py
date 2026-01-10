from dotenv import load_dotenv
import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import tempfile
from app.api.routes import router as ai_workflows_router
from app.integrations.notion_integration import NotionIntegration
from app.services.email_service import EmailService
from app.audio_processor import AudioProcessor
from langchain.schema import HumanMessage
from app.core.logger import get_logger

# Request models
class MeetingRequest(BaseModel):
    meeting_text: str

class TaskItem(BaseModel):
    title: str
    description: str
    assignee: Optional[str] = None
    priority: str  # High, Medium, Low
    due_date: Optional[str] = None

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Project Manager Agent", version="1.0.0", debug=True)

logger = get_logger(__name__)

# Initialize Google Gemini
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Configure Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY not found in environment variables. AI features will be disabled.")
        llm = None
    else:
        genai.configure(api_key=api_key)
        
        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        print(f"[INFO] Google Gemini client successfully initialized with model: gemini-1.5-pro")
        
except Exception as e:
    print(f"[ERROR] Gemini initialization failed: {e}")
    llm = None

# Initialize Audio Processor
try:
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

# Initialize Database Connection for advanced features
db_conn = None
try:
    from app.core.pgvector_setup import PgvectorSetup
    pg_setup = PgvectorSetup()
    db_conn = pg_setup.get_connection()
    print("[INFO] Database connection established")
except Exception as e:
    print(f"[ERROR] Database connection failed: {e}")
    db_conn = None

# Initialize Repetition Detector
try:
    from app.services.repetition_detector import RepetitionDetector
    if db_conn and notion_integration:
        repetition_detector = RepetitionDetector(db_conn, notion_integration)
        print("[INFO] Repetition detector initialized")
    else:
        repetition_detector = None
        print("[WARNING] Repetition detector disabled (requires DB and Notion)")
except Exception as e:
    print(f"[ERROR] Repetition detector initialization failed: {e}")
    repetition_detector = None

# Initialize Security Manager
try:
    from app.core.security import SecurityManager
    if db_conn:
        security_manager = SecurityManager(db_conn)
        print("[INFO] Security Manager initialized")
    else:
        security_manager = None
        print("[WARNING] Security Manager disabled (requires DB)")
except Exception as e:
    print(f"[ERROR] Security Manager initialization failed: {e}")
    security_manager = None

# Initialize Scheduler Service
try:
    from app.core.scheduler import SchedulerService
    scheduler_service = SchedulerService()
    print("[INFO] Scheduler service initialized")
except Exception as e:
    print(f"[ERROR] Scheduler initialization failed: {e}")
    scheduler_service = None

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.io imports
try:
    import socketio
    from fastapi_socketio import SocketManager
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("[WARNING] Socket.io not available")

# Include Router
app.include_router(ai_workflows_router)

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
    
    # Initialize Socket.io manager and handlers
    socketio_manager = get_socketio_manager()
    socketio_manager.set_socketio_server(sio)
    
    socketio_handlers = get_socketio_handlers()
    socketio_handlers.set_socketio_server(sio)
    
    # Register Socket.io listeners with job event listener
    job_event_listener = get_job_event_listener()
    socketio_manager.register_job_listeners(job_event_listener)

    # Wrap FastAPI app with Socket.io ASGI app
    # IMPORTANT: This must be done AFTER adding all middleware to FastAPI app
    app = socketio.ASGIApp(sio, app)

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

@app.get("/")
async def root():
    return {
        "message": "AI Project Manager Agent is running!", 
        "gemini_status": "connected" if llm else "failed",
        "audio_status": "enabled" if audio_processor else "disabled",
        "notion_status": "connected" if notion_integration else "disabled",
        "email_status": "enabled" if email_service else "disabled",
        "scheduler_status": "running" if scheduler_service and hasattr(scheduler_service, 'is_running') and scheduler_service.is_running else "stopped",
        "security_status": "enabled" if security_manager else "disabled",
        "repetition_detector_status": "enabled" if repetition_detector else "disabled"
    }

@app.get("/api/v1/ai/healthz")
async def health_check():
    """Health check endpoint for production monitoring"""
    return {
        "status": "ok",
        "service": "ai-engine",
        "version": "1.0.0"
    }

@app.post("/analyze-meeting")
async def analyze_meeting_text(request: MeetingRequest):
    """
    Analyze meeting transcript and extract structured information using Gemini
    """
    
    if not llm:
        raise HTTPException(status_code=500, detail="Gemini API not initialized")
    
    meeting_text = request.meeting_text
    
    if not meeting_text.strip():
        raise HTTPException(status_code=400, detail="meeting_text cannot be empty")
    
    prompt = f"""
    You are an AI Project Manager. Analyze this meeting transcript and extract:
    
    1. Key decisions made
    2. Action items (with assignee if mentioned, priority, due date if mentioned)
    3. Risks and blockers identified
    4. Brief meeting summary
    
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
        "meeting_summary": "brief summary of the meeting"
    }}
    
    Only return valid JSON, no additional text.
    """
    
    try:
        print(f"[INFO] Analyzing meeting text with Gemini: {meeting_text[:100]}...")
        
        # Call Gemini API
        message = HumanMessage(content=prompt)
        response = llm([message])
        
        print(f"[DEBUG] Gemini response received")
        
        # Clean the response to handle markdown code blocks
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
            
        # Parse the JSON response
        analysis = json.loads(content)
        
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
    Upload audio file, transcribe it using Gemini, and analyze the meeting
    """
    
    if not llm:
        raise HTTPException(status_code=500, detail="Gemini API not initialized")
    
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
        # Note: audio_processor.transcribe_audio defaults to Gemini
        transcript = audio_processor.transcribe_audio(temp_file_path)
        
        print(f"[INFO] Transcript length: {len(transcript)} chars")
        
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
        
        # Call Gemini API
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
