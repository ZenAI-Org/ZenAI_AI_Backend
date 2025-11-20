"""
Transcription Agent - Converts meeting audio to text using Whisper API.
Handles audio download from S3, large file chunking, and transcript storage.
Includes comprehensive error handling and graceful degradation.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
import boto3
from botocore.exceptions import ClientError

from app.agents.base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus
from app.core.api_retry import APIRetryHandler, RetryConfig
from app.core.error_notifications import get_error_notification_manager
from app.core.graceful_degradation import TranscriptionDegradation
from app.core.error_dashboard import get_error_metrics

logger = logging.getLogger(__name__)


class TranscriptionAgent(BaseAgent):
    """
    Agent responsible for transcribing meeting audio using OpenAI's Whisper API.
    
    Handles:
    - Audio download from S3
    - Large file chunking (>25MB)
    - Whisper API transcription
    - Transcript storage
    - Status updates and error handling
    """
    
    # Whisper API supports files up to 25MB
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
    CHUNK_SIZE = 20 * 1024 * 1024  # 20MB chunks for safety
    
    def __init__(self, config: AgentConfig):
        """
        Initialize Transcription Agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        self.temp_dir = "/tmp/transcription"
        self._ensure_temp_dir()
        
        # Initialize error handling
        self.retry_handler = APIRetryHandler(
            RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0)
        )
        self.error_notifier = get_error_notification_manager()
        self.error_metrics = get_error_metrics()
    
    def _ensure_temp_dir(self) -> None:
        """Ensure temporary directory exists."""
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def execute(
        self,
        meeting_id: str,
        s3_key: str,
        s3_bucket: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute transcription workflow.
        
        Args:
            meeting_id: Meeting ID
            s3_key: S3 object key for audio file
            s3_bucket: S3 bucket name (from env if not provided)
            
        Returns:
            AgentResult with transcript data
        """
        try:
            s3_bucket = s3_bucket or os.getenv("AWS_S3_BUCKET")
            if not s3_bucket:
                raise ValueError("S3 bucket not configured")
            
            self._log_execution(
                "Starting transcription",
                {"meeting_id": meeting_id, "s3_key": s3_key},
            )
            
            # Download audio from S3
            local_file_path = await self._download_from_s3(s3_bucket, s3_key)
            
            # Check file size and chunk if necessary
            file_size = os.path.getsize(local_file_path)
            self._log_execution(
                f"Audio file downloaded: {file_size} bytes",
                {"meeting_id": meeting_id},
            )
            
            # Transcribe audio
            if file_size > self.MAX_FILE_SIZE:
                transcript = await self._transcribe_chunked(local_file_path)
            else:
                transcript = await self._transcribe_file(local_file_path)
            
            self._log_execution(
                "Transcription completed",
                {
                    "meeting_id": meeting_id,
                    "transcript_length": len(transcript),
                },
            )
            
            # Cleanup temporary file
            self._cleanup_file(local_file_path)
            
            return self._create_success_result(
                data={
                    "meeting_id": meeting_id,
                    "transcript": transcript,
                    "language": "en",
                    "file_size": file_size,
                },
                metadata={
                    "agent": "TranscriptionAgent",
                    "model": "whisper-1",
                },
            )
        
        except Exception as e:
            self._log_error(
                f"Transcription failed: {str(e)}",
                {"meeting_id": meeting_id},
            )
            return self._create_error_result(
                error=f"Transcription failed: {str(e)}",
                metadata={"meeting_id": meeting_id},
            )
    
    async def _download_from_s3(self, bucket: str, key: str) -> str:
        """
        Download audio file from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Local file path
            
        Raises:
            ClientError: If S3 download fails
        """
        try:
            local_file_path = os.path.join(self.temp_dir, os.path.basename(key))
            
            self._log_execution(
                f"Downloading from S3: s3://{bucket}/{key}",
                {"local_path": local_file_path},
            )
            
            self.s3_client.download_file(bucket, key, local_file_path)
            
            return local_file_path
        
        except ClientError as e:
            self._log_error(
                f"S3 download failed: {str(e)}",
                {"bucket": bucket, "key": key},
            )
            raise
    
    async def _transcribe_file(self, file_path: str) -> str:
        """
        Transcribe a single audio file using Whisper API with retry logic.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
            
        Raises:
            Exception: If transcription fails after retries
        """
        try:
            def transcribe_sync():
                with open(file_path, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                    )
                return transcript
            
            # Use retry handler for API call
            transcript = self.retry_handler.retry_sync(
                transcribe_sync,
                api_name="whisper",
                endpoint="transcriptions.create",
            )
            
            return transcript
        
        except Exception as e:
            # Record error in metrics
            self.error_metrics.record_error(
                error_type="api_error",
                api_name="whisper",
                severity="high",
                message=str(e),
                details={"file_path": file_path},
            )
            
            self._log_error(
                f"Whisper API call failed: {str(e)}",
                {"file_path": file_path},
            )
            raise
    
    async def _transcribe_chunked(self, file_path: str) -> str:
        """
        Transcribe large audio file by splitting into chunks.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Combined transcribed text from all chunks
            
        Raises:
            Exception: If transcription fails
        """
        try:
            file_size = os.path.getsize(file_path)
            num_chunks = (file_size + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
            
            self._log_execution(
                f"Splitting file into {num_chunks} chunks",
                {"file_path": file_path, "file_size": file_size},
            )
            
            transcripts = []
            
            with open(file_path, "rb") as f:
                for chunk_num in range(num_chunks):
                    chunk_data = f.read(self.CHUNK_SIZE)
                    
                    if not chunk_data:
                        break
                    
                    # Create temporary chunk file
                    chunk_file_path = f"{file_path}.chunk{chunk_num}"
                    with open(chunk_file_path, "wb") as chunk_file:
                        chunk_file.write(chunk_data)
                    
                    # Transcribe chunk
                    chunk_transcript = await self._transcribe_file(chunk_file_path)
                    transcripts.append(chunk_transcript)
                    
                    # Cleanup chunk file
                    self._cleanup_file(chunk_file_path)
                    
                    self._log_execution(
                        f"Transcribed chunk {chunk_num + 1}/{num_chunks}",
                        {"chunk_size": len(chunk_data)},
                    )
            
            # Combine transcripts with space
            combined_transcript = " ".join(transcripts)
            
            return combined_transcript
        
        except Exception as e:
            self._log_error(
                f"Chunked transcription failed: {str(e)}",
                {"file_path": file_path},
            )
            raise
    
    def _cleanup_file(self, file_path: str) -> None:
        """
        Remove temporary file.
        
        Args:
            file_path: Path to file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self._log_execution(
                    f"Cleaned up temporary file",
                    {"file_path": file_path},
                )
        except Exception as e:
            self._log_error(
                f"Failed to cleanup file: {str(e)}",
                {"file_path": file_path},
            )



# Job function for background processing
async def transcribe_audio_job(
    meeting_id: str,
    s3_key: str,
    s3_bucket: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job function for transcribing audio with error handling.
    
    Args:
        meeting_id: Meeting ID
        s3_key: S3 object key for audio file
        s3_bucket: S3 bucket name (from env if not provided)
        project_id: Optional project ID for error notifications
        
    Returns:
        Dictionary with transcription result
    """
    try:
        config = AgentConfig(
            name="TranscriptionAgent",
            model="whisper-1",
            temperature=0.0,
        )
        agent = TranscriptionAgent(config)
        result = await agent.execute(
            meeting_id=meeting_id,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
        )
        
        if result.status == AgentStatus.SUCCESS:
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "text": result.data.get("transcript", ""),
                "language": result.data.get("language", "en"),
                "file_size": result.data.get("file_size", 0),
            }
        else:
            # Handle transcription failure with graceful degradation
            error_msg = result.error or "Unknown transcription error"
            
            # Notify user if project_id provided
            if project_id:
                error_notifier = get_error_notification_manager()
                error_notifier.notify_error(
                    project_id=project_id,
                    error_type="api_error",
                    api_name="whisper",
                    message=error_msg,
                    severity="high",
                    user_message="Audio transcription failed. Please try uploading the audio again.",
                )
            
            return {
                "status": "error",
                "meeting_id": meeting_id,
                "error": error_msg,
            }
    
    except Exception as e:
        logger.error(f"Transcription job failed: {str(e)}")
        
        # Record error in metrics
        error_metrics = get_error_metrics()
        error_metrics.record_error(
            error_type="job_error",
            api_name="whisper",
            severity="high",
            message=str(e),
            details={"meeting_id": meeting_id, "s3_key": s3_key},
        )
        
        # Notify user if project_id provided
        if project_id:
            error_notifier = get_error_notification_manager()
            error_notifier.notify_error(
                project_id=project_id,
                error_type="job_error",
                api_name="whisper",
                message=str(e),
                severity="high",
                user_message="Transcription job encountered an error. Please try again.",
            )
        
        return {
            "status": "error",
            "meeting_id": meeting_id,
            "error": str(e),
        }
