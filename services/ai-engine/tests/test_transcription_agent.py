"""
Unit tests for Transcription Agent.
Tests Whisper API integration, file chunking, error handling, and transcript storage.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock, mock_open
from pathlib import Path

from app.agents.transcription_agent import TranscriptionAgent
from app.agents.base_agent import AgentConfig, AgentStatus


class TestTranscriptionAgentInitialization:
    """Tests for TranscriptionAgent initialization."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    def test_agent_initialization(self, mock_openai, mock_boto3):
        """Test TranscriptionAgent initialization."""
        config = AgentConfig(model_name="gpt-4", temperature=0.1)
        agent = TranscriptionAgent(config)
        
        assert agent.config == config
        assert agent.openai_client is not None
        assert agent.s3_client is not None
        assert agent.MAX_FILE_SIZE == 25 * 1024 * 1024
        assert agent.CHUNK_SIZE == 20 * 1024 * 1024
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    def test_temp_dir_creation(self, mock_openai, mock_boto3):
        """Test temporary directory is created."""
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        
        assert os.path.exists(agent.temp_dir)


class TestTranscriptionAgentS3Download:
    """Tests for S3 audio download functionality."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_download_from_s3_success(self, mock_openai, mock_boto3):
        """Test successful S3 download."""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.s3_client = mock_s3
        
        # Mock S3 download
        mock_s3.download_file = MagicMock()
        
        local_path = await agent._download_from_s3("test-bucket", "audio.mp3")
        
        assert "audio.mp3" in local_path
        mock_s3.download_file.assert_called_once_with(
            "test-bucket", "audio.mp3", local_path
        )
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_download_from_s3_failure(self, mock_openai, mock_boto3):
        """Test S3 download failure handling."""
        from botocore.exceptions import ClientError
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.s3_client = mock_s3
        
        # Mock S3 download failure
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        mock_s3.download_file.side_effect = ClientError(error_response, "GetObject")
        
        with pytest.raises(ClientError):
            await agent._download_from_s3("test-bucket", "nonexistent.mp3")


class TestTranscriptionAgentWhisperAPI:
    """Tests for Whisper API transcription."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_transcribe_file_success(self, mock_openai, mock_boto3):
        """Test successful file transcription."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock Whisper API response
        mock_transcription = MagicMock()
        mock_transcription.text = "This is a test transcript"
        mock_openai_instance.audio.transcriptions.create.return_value = "This is a test transcript"
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake audio data")
        
        try:
            transcript = await agent._transcribe_file(tmp_path)
            
            assert transcript == "This is a test transcript"
            mock_openai_instance.audio.transcriptions.create.assert_called_once()
        finally:
            os.unlink(tmp_path)
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_transcribe_file_api_error(self, mock_openai, mock_boto3):
        """Test Whisper API error handling."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock API error
        mock_openai_instance.audio.transcriptions.create.side_effect = Exception(
            "API rate limit exceeded"
        )
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake audio data")
        
        try:
            with pytest.raises(Exception):
                await agent._transcribe_file(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestTranscriptionAgentChunking:
    """Tests for large file chunking logic."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_transcribe_chunked_file(self, mock_openai, mock_boto3):
        """Test chunked transcription for large files."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock Whisper API to return different transcripts for each chunk
        mock_openai_instance.audio.transcriptions.create.side_effect = [
            "First chunk transcript",
            "Second chunk transcript",
        ]
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        
        # Create a file larger than MAX_FILE_SIZE
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            # Write data larger than MAX_FILE_SIZE
            tmp.write(b"x" * (agent.MAX_FILE_SIZE + 1000))
        
        try:
            transcript = await agent._transcribe_chunked(tmp_path)
            
            # Should combine both transcripts
            assert "First chunk transcript" in transcript
            assert "Second chunk transcript" in transcript
            assert mock_openai_instance.audio.transcriptions.create.call_count == 2
        finally:
            os.unlink(tmp_path)
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_chunk_size_calculation(self, mock_openai, mock_boto3):
        """Test correct chunk size calculation."""
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        
        # Test file size calculations
        file_size = agent.MAX_FILE_SIZE + 1
        num_chunks = (file_size + agent.CHUNK_SIZE - 1) // agent.CHUNK_SIZE
        
        assert num_chunks == 2
        
        # Test exact multiple
        file_size = agent.CHUNK_SIZE * 3
        num_chunks = (file_size + agent.CHUNK_SIZE - 1) // agent.CHUNK_SIZE
        
        assert num_chunks == 3


class TestTranscriptionAgentExecution:
    """Tests for complete transcription workflow."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_execute_transcription_success(self, mock_openai, mock_boto3):
        """Test successful end-to-end transcription."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        # Mock Whisper API
        mock_openai_instance.audio.transcriptions.create.return_value = (
            "Meeting transcript content"
        )
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        agent.s3_client = mock_s3
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake audio data")
        
        # Mock S3 download to copy test file
        def mock_download(bucket, key, local_path):
            with open(tmp_path, "rb") as src:
                with open(local_path, "wb") as dst:
                    dst.write(src.read())
        
        mock_s3.download_file.side_effect = mock_download
        
        try:
            result = await agent.execute(
                meeting_id="meeting-123",
                s3_key="audio.mp3",
                s3_bucket="test-bucket",
            )
            
            assert result.status == AgentStatus.SUCCESS
            assert result.data["meeting_id"] == "meeting-123"
            assert result.data["transcript"] == "Meeting transcript content"
            assert result.data["language"] == "en"
        finally:
            os.unlink(tmp_path)
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_execute_transcription_s3_error(self, mock_openai, mock_boto3):
        """Test transcription with S3 error."""
        from botocore.exceptions import ClientError
        
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        # Mock S3 error
        error_response = {"Error": {"Code": "NoSuchBucket"}}
        mock_s3.download_file.side_effect = ClientError(error_response, "GetObject")
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        agent.s3_client = mock_s3
        
        result = await agent.execute(
            meeting_id="meeting-123",
            s3_key="audio.mp3",
            s3_bucket="nonexistent-bucket",
        )
        
        assert result.status == AgentStatus.ERROR
        assert result.error is not None
        assert "Transcription failed" in result.error
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_execute_transcription_missing_bucket(self, mock_openai, mock_boto3):
        """Test transcription with missing S3 bucket configuration."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        agent.s3_client = mock_s3
        
        # Don't provide s3_bucket and ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if "AWS_S3_BUCKET" in os.environ:
                del os.environ["AWS_S3_BUCKET"]
            
            result = await agent.execute(
                meeting_id="meeting-123",
                s3_key="audio.mp3",
            )
            
            assert result.status == AgentStatus.ERROR
            assert "S3 bucket not configured" in result.error


class TestTranscriptionAgentErrorHandling:
    """Tests for error handling and recovery."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_cleanup_file_success(self, mock_openai, mock_boto3):
        """Test successful file cleanup."""
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        assert os.path.exists(tmp_path)
        
        agent._cleanup_file(tmp_path)
        
        assert not os.path.exists(tmp_path)
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    def test_cleanup_nonexistent_file(self, mock_openai, mock_boto3):
        """Test cleanup of nonexistent file doesn't raise error."""
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        
        # Should not raise exception
        agent._cleanup_file("/nonexistent/path/file.mp3")
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_execute_with_api_timeout(self, mock_openai, mock_boto3):
        """Test handling of API timeout."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        # Mock timeout error
        mock_openai_instance.audio.transcriptions.create.side_effect = TimeoutError(
            "Request timeout"
        )
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        agent.s3_client = mock_s3
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake audio data")
        
        def mock_download(bucket, key, local_path):
            with open(tmp_path, "rb") as src:
                with open(local_path, "wb") as dst:
                    dst.write(src.read())
        
        mock_s3.download_file.side_effect = mock_download
        
        try:
            result = await agent.execute(
                meeting_id="meeting-123",
                s3_key="audio.mp3",
                s3_bucket="test-bucket",
            )
            
            assert result.status == AgentStatus.ERROR
            assert "Transcription failed" in result.error
        finally:
            os.unlink(tmp_path)


class TestTranscriptionAgentMetadata:
    """Tests for result metadata and logging."""
    
    @patch("app.agents.transcription_agent.boto3.client")
    @patch("app.agents.transcription_agent.OpenAI")
    @pytest.mark.asyncio
    async def test_result_metadata(self, mock_openai, mock_boto3):
        """Test result includes proper metadata."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        mock_openai_instance.audio.transcriptions.create.return_value = (
            "Test transcript"
        )
        
        config = AgentConfig()
        agent = TranscriptionAgent(config)
        agent.openai_client = mock_openai_instance
        agent.s3_client = mock_s3
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake audio data")
        
        def mock_download(bucket, key, local_path):
            with open(tmp_path, "rb") as src:
                with open(local_path, "wb") as dst:
                    dst.write(src.read())
        
        mock_s3.download_file.side_effect = mock_download
        
        try:
            result = await agent.execute(
                meeting_id="meeting-123",
                s3_key="audio.mp3",
                s3_bucket="test-bucket",
            )
            
            assert result.metadata["agent"] == "TranscriptionAgent"
            assert result.metadata["model"] == "whisper-1"
            assert result.data["file_size"] > 0
        finally:
            os.unlink(tmp_path)
