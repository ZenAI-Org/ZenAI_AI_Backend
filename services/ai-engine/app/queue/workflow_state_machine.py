"""
Workflow State Machine for orchestrating multi-agent workflows.

Manages workflow state transitions, validates transitions, and coordinates
agent execution based on workflow state.
"""

import logging
from typing import Any, Dict, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class WorkflowPhase(str, Enum):
    """Phases of a workflow."""
    INITIALIZATION = "initialization"
    TRANSCRIPTION = "transcription"
    SUMMARIZATION = "summarization"
    TASK_EXTRACTION = "task_extraction"
    NOTION_SYNC = "notion_sync"
    COMPLETION = "completion"


@dataclass
class StateTransition:
    """Represents a valid state transition."""
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None


class WorkflowStateMachine:
    """
    State machine for workflow orchestration.
    
    Manages:
    - Valid state transitions
    - Transition conditions and actions
    - Workflow phase management
    - Error state handling
    """
    
    def __init__(self):
        """Initialize workflow state machine."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._transitions: Dict[str, List[StateTransition]] = {}
        self._current_states: Dict[str, str] = {}
        self._phase_handlers: Dict[WorkflowPhase, Callable] = {}
        self._error_handlers: Dict[str, Callable] = {}
        
        self._initialize_transitions()
    
    def _initialize_transitions(self) -> None:
        """Initialize valid state transitions for meeting processing workflow."""
        # Meeting processing workflow transitions
        transitions = [
            StateTransition("initiated", "transcription_queued"),
            StateTransition("transcription_queued", "transcription_running"),
            StateTransition("transcription_running", "transcription_completed"),
            StateTransition("transcription_completed", "summarization_queued"),
            StateTransition("summarization_queued", "summarization_running"),
            StateTransition("summarization_running", "summarization_completed"),
            StateTransition("summarization_completed", "task_extraction_queued"),
            StateTransition("task_extraction_queued", "task_extraction_running"),
            StateTransition("task_extraction_running", "task_extraction_completed"),
            StateTransition("task_extraction_completed", "notion_sync_queued"),
            StateTransition("notion_sync_queued", "notion_sync_running"),
            StateTransition("notion_sync_running", "notion_sync_completed"),
            StateTransition("notion_sync_completed", "completed"),
            
            # Error transitions
            StateTransition("transcription_queued", "error"),
            StateTransition("transcription_running", "error"),
            StateTransition("summarization_queued", "error"),
            StateTransition("summarization_running", "error"),
            StateTransition("task_extraction_queued", "error"),
            StateTransition("task_extraction_running", "error"),
            StateTransition("notion_sync_queued", "error"),
            StateTransition("notion_sync_running", "error"),
            
            # Recovery transitions
            StateTransition("error", "initiated"),
            StateTransition("error", "transcription_queued"),
            StateTransition("error", "summarization_queued"),
            StateTransition("error", "task_extraction_queued"),
        ]
        
        for transition in transitions:
            if transition.from_state not in self._transitions:
                self._transitions[transition.from_state] = []
            self._transitions[transition.from_state].append(transition)
        
        self.logger.info(f"Initialized {len(transitions)} state transitions")
    
    def register_phase_handler(
        self,
        phase: WorkflowPhase,
        handler: Callable,
    ) -> None:
        """
        Register a handler for a workflow phase.
        
        Args:
            phase: Workflow phase
            handler: Callable to handle the phase
        """
        self._phase_handlers[phase] = handler
        self.logger.info(f"Phase handler registered: {phase.value}")
    
    def register_error_handler(
        self,
        error_type: str,
        handler: Callable,
    ) -> None:
        """
        Register a handler for an error type.
        
        Args:
            error_type: Type of error
            handler: Callable to handle the error
        """
        self._error_handlers[error_type] = handler
        self.logger.info(f"Error handler registered: {error_type}")
    
    def can_transition(
        self,
        workflow_id: str,
        from_state: str,
        to_state: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            workflow_id: Workflow ID
            from_state: Current state
            to_state: Desired state
            context: Optional context for condition evaluation
            
        Returns:
            True if transition is valid, False otherwise
        """
        if from_state not in self._transitions:
            self.logger.warning(
                f"No transitions defined for state: {from_state}"
            )
            return False
        
        for transition in self._transitions[from_state]:
            if transition.to_state == to_state:
                # Check condition if defined
                if transition.condition:
                    try:
                        if not transition.condition(context or {}):
                            self.logger.warning(
                                f"Transition condition failed: "
                                f"{from_state} -> {to_state}"
                            )
                            return False
                    except Exception as e:
                        self.logger.error(
                            f"Error evaluating transition condition: {e}"
                        )
                        return False
                
                return True
        
        self.logger.warning(
            f"Invalid transition: {from_state} -> {to_state}"
        )
        return False
    
    def transition(
        self,
        workflow_id: str,
        from_state: str,
        to_state: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Perform a state transition.
        
        Args:
            workflow_id: Workflow ID
            from_state: Current state
            to_state: Desired state
            context: Optional context for action execution
            
        Returns:
            True if transition successful, False otherwise
        """
        if not self.can_transition(workflow_id, from_state, to_state, context):
            return False
        
        # Find transition and execute action
        for transition in self._transitions[from_state]:
            if transition.to_state == to_state:
                if transition.action:
                    try:
                        transition.action(workflow_id, context or {})
                    except Exception as e:
                        self.logger.error(
                            f"Error executing transition action: {e}"
                        )
                        return False
                
                # Update current state
                self._current_states[workflow_id] = to_state
                
                self.logger.info(
                    f"State transition: {workflow_id} "
                    f"{from_state} -> {to_state}"
                )
                
                return True
        
        return False
    
    def get_current_state(self, workflow_id: str) -> Optional[str]:
        """
        Get current state of a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Current state or None if not found
        """
        return self._current_states.get(workflow_id)
    
    def set_current_state(self, workflow_id: str, state: str) -> None:
        """
        Set current state of a workflow.
        
        Args:
            workflow_id: Workflow ID
            state: New state
        """
        self._current_states[workflow_id] = state
        self.logger.info(f"State set: {workflow_id} -> {state}")
    
    def get_valid_transitions(self, from_state: str) -> List[str]:
        """
        Get valid next states from current state.
        
        Args:
            from_state: Current state
            
        Returns:
            List of valid next states
        """
        if from_state not in self._transitions:
            return []
        
        return [t.to_state for t in self._transitions[from_state]]
    
    def get_workflow_phase(self, state: str) -> Optional[WorkflowPhase]:
        """
        Get workflow phase for a state.
        
        Args:
            state: Workflow state
            
        Returns:
            WorkflowPhase or None if not found
        """
        state_lower = state.lower()
        
        if "transcription" in state_lower:
            return WorkflowPhase.TRANSCRIPTION
        elif "summarization" in state_lower:
            return WorkflowPhase.SUMMARIZATION
        elif "task_extraction" in state_lower:
            return WorkflowPhase.TASK_EXTRACTION
        elif "notion_sync" in state_lower:
            return WorkflowPhase.NOTION_SYNC
        elif state_lower == "completed":
            return WorkflowPhase.COMPLETION
        elif state_lower == "initiated":
            return WorkflowPhase.INITIALIZATION
        
        return None
    
    def execute_phase_handler(
        self,
        phase: WorkflowPhase,
        workflow_id: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Execute handler for a workflow phase.
        
        Args:
            phase: Workflow phase
            workflow_id: Workflow ID
            context: Workflow context
            
        Returns:
            True if handler executed successfully
        """
        handler = self._phase_handlers.get(phase)
        
        if not handler:
            self.logger.warning(f"No handler registered for phase: {phase.value}")
            return True  # Continue if no handler
        
        try:
            handler(workflow_id, context)
            self.logger.info(
                f"Phase handler executed: {phase.value} for {workflow_id}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Error executing phase handler for {phase.value}: {e}"
            )
            return False
    
    def handle_error(
        self,
        workflow_id: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle an error in workflow execution.
        
        Args:
            workflow_id: Workflow ID
            error_type: Type of error
            error_message: Error message
            context: Optional context
            
        Returns:
            True if error handled successfully
        """
        handler = self._error_handlers.get(error_type)
        
        if not handler:
            self.logger.warning(
                f"No handler registered for error type: {error_type}"
            )
            return False
        
        try:
            handler(workflow_id, error_message, context or {})
            self.logger.info(
                f"Error handled: {error_type} for {workflow_id}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Error executing error handler: {e}"
            )
            return False
    
    def reset_workflow(self, workflow_id: str) -> None:
        """
        Reset workflow to initial state.
        
        Args:
            workflow_id: Workflow ID
        """
        if workflow_id in self._current_states:
            del self._current_states[workflow_id]
        
        self.logger.info(f"Workflow reset: {workflow_id}")
    
    def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get progress information for a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Dictionary with progress information
        """
        current_state = self.get_current_state(workflow_id)
        
        if not current_state:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "progress": 0,
            }
        
        # Calculate progress based on state
        phase_order = [
            "initiated",
            "transcription_queued",
            "transcription_running",
            "transcription_completed",
            "summarization_queued",
            "summarization_running",
            "summarization_completed",
            "task_extraction_queued",
            "task_extraction_running",
            "task_extraction_completed",
            "notion_sync_queued",
            "notion_sync_running",
            "notion_sync_completed",
            "completed",
        ]
        
        try:
            current_index = phase_order.index(current_state)
            progress = int((current_index / len(phase_order)) * 100)
        except ValueError:
            progress = 0
        
        phase = self.get_workflow_phase(current_state)
        
        return {
            "workflow_id": workflow_id,
            "current_state": current_state,
            "phase": phase.value if phase else None,
            "progress": progress,
            "valid_next_states": self.get_valid_transitions(current_state),
        }


# Global state machine instance
_state_machine = None


def get_workflow_state_machine() -> WorkflowStateMachine:
    """Get or create global workflow state machine."""
    global _state_machine
    if _state_machine is None:
        _state_machine = WorkflowStateMachine()
    return _state_machine
