"""
Graceful degradation strategies for handling component failures.
Allows system to continue operating with reduced functionality when components fail.
"""

from typing import Dict, Any, Optional, Callable, TypeVar
from enum import Enum
from app.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DegradationStrategy(Enum):
    """Strategies for graceful degradation."""
    
    SKIP_COMPONENT = "skip_component"  # Skip the failed component
    USE_FALLBACK = "use_fallback"  # Use fallback data/function
    PARTIAL_RESULT = "partial_result"  # Return partial results
    RETRY_LATER = "retry_later"  # Queue for retry
    NOTIFY_USER = "notify_user"  # Notify user of degradation


class GracefulDegradationHandler:
    """Handle graceful degradation for failed components."""
    
    def __init__(self):
        """Initialize graceful degradation handler."""
        self.fallback_functions: Dict[str, Callable] = {}
        self.degradation_strategies: Dict[str, DegradationStrategy] = {}
    
    def register_fallback(
        self,
        component_name: str,
        fallback_func: Callable,
    ) -> None:
        """
        Register a fallback function for a component.
        
        Args:
            component_name: Name of the component
            fallback_func: Fallback function to use if component fails
        """
        self.fallback_functions[component_name] = fallback_func
        logger.info(f"Registered fallback for component: {component_name}")
    
    def register_strategy(
        self,
        component_name: str,
        strategy: DegradationStrategy,
    ) -> None:
        """
        Register a degradation strategy for a component.
        
        Args:
            component_name: Name of the component
            strategy: Degradation strategy to use
        """
        self.degradation_strategies[component_name] = strategy
        logger.info(f"Registered strategy for component {component_name}: {strategy.value}")
    
    def handle_component_failure(
        self,
        component_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle failure of a component with graceful degradation.
        
        Args:
            component_name: Name of the failed component
            error: Exception that was raised
            context: Additional context about the failure
            
        Returns:
            Dictionary with degradation result
        """
        context = context or {}
        strategy = self.degradation_strategies.get(
            component_name,
            DegradationStrategy.SKIP_COMPONENT,
        )
        
        logger.warning(
            f"Component failure: {component_name}",
            extra_fields={
                "component": component_name,
                "error": str(error),
                "strategy": strategy.value,
                "context": context,
            },
        )
        
        if strategy == DegradationStrategy.SKIP_COMPONENT:
            return self._handle_skip_component(component_name, error, context)
        
        elif strategy == DegradationStrategy.USE_FALLBACK:
            return self._handle_use_fallback(component_name, error, context)
        
        elif strategy == DegradationStrategy.PARTIAL_RESULT:
            return self._handle_partial_result(component_name, error, context)
        
        elif strategy == DegradationStrategy.RETRY_LATER:
            return self._handle_retry_later(component_name, error, context)
        
        elif strategy == DegradationStrategy.NOTIFY_USER:
            return self._handle_notify_user(component_name, error, context)
        
        else:
            return self._handle_skip_component(component_name, error, context)
    
    def _handle_skip_component(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle by skipping the failed component.
        
        Args:
            component_name: Name of the component
            error: Exception that was raised
            context: Additional context
            
        Returns:
            Degradation result
        """
        logger.info(f"Skipping component: {component_name}")
        
        return {
            "status": "degraded",
            "strategy": "skip_component",
            "component": component_name,
            "error": str(error),
            "message": f"Component {component_name} is unavailable. Continuing without it.",
            "context": context,
        }
    
    def _handle_use_fallback(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle by using fallback function.
        
        Args:
            component_name: Name of the component
            error: Exception that was raised
            context: Additional context
            
        Returns:
            Degradation result
        """
        fallback_func = self.fallback_functions.get(component_name)
        
        if not fallback_func:
            logger.warning(f"No fallback function registered for {component_name}")
            return self._handle_skip_component(component_name, error, context)
        
        try:
            logger.info(f"Using fallback for component: {component_name}")
            result = fallback_func(context)
            
            return {
                "status": "degraded",
                "strategy": "use_fallback",
                "component": component_name,
                "error": str(error),
                "message": f"Using fallback for {component_name}",
                "result": result,
                "context": context,
            }
        
        except Exception as fallback_error:
            logger.error(f"Fallback function failed for {component_name}: {fallback_error}")
            return self._handle_skip_component(component_name, error, context)
    
    def _handle_partial_result(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle by returning partial results.
        
        Args:
            component_name: Name of the component
            error: Exception that was raised
            context: Additional context
            
        Returns:
            Degradation result
        """
        logger.info(f"Returning partial result for component: {component_name}")
        
        # Extract partial data from context if available
        partial_data = context.get("partial_data", {})
        
        return {
            "status": "degraded",
            "strategy": "partial_result",
            "component": component_name,
            "error": str(error),
            "message": f"Partial results available for {component_name}",
            "partial_data": partial_data,
            "context": context,
        }
    
    def _handle_retry_later(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle by queuing for retry.
        
        Args:
            component_name: Name of the component
            error: Exception that was raised
            context: Additional context
            
        Returns:
            Degradation result
        """
        logger.info(f"Queuing component for retry: {component_name}")
        
        return {
            "status": "degraded",
            "strategy": "retry_later",
            "component": component_name,
            "error": str(error),
            "message": f"Component {component_name} will be retried later",
            "context": context,
        }
    
    def _handle_notify_user(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle by notifying user.
        
        Args:
            component_name: Name of the component
            error: Exception that was raised
            context: Additional context
            
        Returns:
            Degradation result
        """
        logger.info(f"Notifying user about component failure: {component_name}")
        
        return {
            "status": "degraded",
            "strategy": "notify_user",
            "component": component_name,
            "error": str(error),
            "message": f"Component {component_name} encountered an issue. Please try again.",
            "context": context,
        }


class TranscriptionDegradation:
    """Graceful degradation for transcription failures."""
    
    @staticmethod
    def handle_transcription_failure(
        meeting_id: str,
        error: Exception,
        partial_transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle transcription failure gracefully.
        
        Args:
            meeting_id: Meeting ID
            error: Exception that was raised
            partial_transcript: Partial transcript if available
            
        Returns:
            Degradation result
        """
        logger.warning(
            f"Transcription failed for meeting {meeting_id}",
            extra_fields={
                "meeting_id": meeting_id,
                "error": str(error),
                "has_partial": partial_transcript is not None,
            },
        )
        
        return {
            "status": "degraded",
            "component": "transcription",
            "meeting_id": meeting_id,
            "error": str(error),
            "message": "Transcription failed. Please try uploading the audio again.",
            "partial_transcript": partial_transcript,
            "available_data": {
                "transcript": partial_transcript or "",
            },
        }


class SummarizationDegradation:
    """Graceful degradation for summarization failures."""
    
    @staticmethod
    def handle_summarization_failure(
        meeting_id: str,
        transcript: str,
        error: Exception,
    ) -> Dict[str, Any]:
        """
        Handle summarization failure gracefully.
        
        Args:
            meeting_id: Meeting ID
            transcript: Available transcript
            error: Exception that was raised
            
        Returns:
            Degradation result with transcript still available
        """
        logger.warning(
            f"Summarization failed for meeting {meeting_id}",
            extra_fields={
                "meeting_id": meeting_id,
                "error": str(error),
                "transcript_length": len(transcript),
            },
        )
        
        return {
            "status": "degraded",
            "component": "summarization",
            "meeting_id": meeting_id,
            "error": str(error),
            "message": "Summary generation failed, but transcript is available.",
            "available_data": {
                "transcript": transcript,
                "summary": None,
            },
        }


class TaskExtractionDegradation:
    """Graceful degradation for task extraction failures."""
    
    @staticmethod
    def handle_extraction_failure(
        meeting_id: str,
        transcript: str,
        error: Exception,
        partial_tasks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Handle task extraction failure gracefully.
        
        Args:
            meeting_id: Meeting ID
            transcript: Available transcript
            error: Exception that was raised
            partial_tasks: Partially extracted tasks if available
            
        Returns:
            Degradation result with transcript still available
        """
        logger.warning(
            f"Task extraction failed for meeting {meeting_id}",
            extra_fields={
                "meeting_id": meeting_id,
                "error": str(error),
                "partial_tasks": len(partial_tasks or []),
            },
        )
        
        return {
            "status": "degraded",
            "component": "task_extraction",
            "meeting_id": meeting_id,
            "error": str(error),
            "message": "Task extraction failed, but transcript is available for manual review.",
            "available_data": {
                "transcript": transcript,
                "tasks": partial_tasks or [],
            },
        }


# Global graceful degradation handler
_degradation_handler = None


def get_degradation_handler() -> GracefulDegradationHandler:
    """Get or create global graceful degradation handler."""
    global _degradation_handler
    if _degradation_handler is None:
        _degradation_handler = GracefulDegradationHandler()
    return _degradation_handler
