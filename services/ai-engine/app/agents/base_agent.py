"""
Base agent interface and abstract class for all AI agents.
All specialized agents (Transcription, Summarization, Task Extraction, etc.)
extend this base class to ensure consistent behavior and interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status enum for agent execution."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class AgentConfig:
    """Configuration for agent initialization."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 300,
    ):
        """
        Initialize agent configuration.
        
        Args:
            model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature for model (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout


class AgentResult:
    """Result object returned by agent execution."""
    
    def __init__(
        self,
        status: AgentStatus,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize agent result.
        
        Args:
            status: Execution status
            data: Result data (if successful)
            error: Error message (if failed)
            metadata: Additional metadata about execution
        """
        self.status = status
        self.data = data or {}
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    Defines the interface and common functionality that all agents must implement.
    Agents are responsible for specific tasks in the AI orchestration pipeline.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        if not self.config.model_name:
            raise ValueError("model_name is required in agent configuration")
        
        if not 0.0 <= self.config.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.config.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the agent's primary task.
        
        This method must be implemented by all subclasses.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            AgentResult with execution status and data
        """
        pass
    
    def _log_execution(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log agent execution details.
        
        Args:
            action: Action being performed
            details: Additional details to log
        """
        log_msg = f"[{self.__class__.__name__}] {action}"
        if details:
            log_msg += f" - {details}"
        self.logger.info(log_msg)
    
    def _log_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log agent error.
        
        Args:
            error: Error message
            details: Additional error details
        """
        log_msg = f"[{self.__class__.__name__}] ERROR: {error}"
        if details:
            log_msg += f" - {details}"
        self.logger.error(log_msg)
    
    def _create_success_result(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Create a successful agent result.
        
        Args:
            data: Result data
            metadata: Optional metadata
            
        Returns:
            AgentResult with success status
        """
        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=data,
            metadata=metadata,
        )
    
    def _create_error_result(
        self,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Create an error agent result.
        
        Args:
            error: Error message
            metadata: Optional metadata
            
        Returns:
            AgentResult with error status
        """
        return AgentResult(
            status=AgentStatus.ERROR,
            error=error,
            metadata=metadata,
        )
