"""
Agent Coordinator for multi-agent communication and state management.

Manages inter-agent messaging, shared context, and workflow orchestration
for complex multi-step AI workflows.
"""

import logging
import json
from typing import Any, Dict, Optional, List, Callable
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict

from app.queue.redis_config import get_redis_client

logger = logging.getLogger(__name__)


class AgentMessageType(str, Enum):
    """Types of messages agents can send."""
    REQUEST = "request"
    RESPONSE = "response"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    COMPLETION = "completion"


class WorkflowState(str, Enum):
    """States of a workflow."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    message_id: str
    sender_agent: str
    recipient_agent: Optional[str]
    message_type: AgentMessageType
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender_agent": self.sender_agent,
            "recipient_agent": self.recipient_agent,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender_agent=data["sender_agent"],
            recipient_agent=data.get("recipient_agent"),
            message_type=AgentMessageType(data["message_type"]),
            payload=data["payload"],
            timestamp=data["timestamp"],
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class WorkflowContext:
    """Shared context for a workflow execution."""
    workflow_id: str
    workflow_type: str
    state: WorkflowState
    input_data: Dict[str, Any]
    shared_state: Dict[str, Any]
    agent_results: Dict[str, Any]
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "state": self.state.value,
            "input_data": self.input_data,
            "shared_state": self.shared_state,
            "agent_results": self.agent_results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Create context from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            workflow_type=data["workflow_type"],
            state=WorkflowState(data["state"]),
            input_data=data["input_data"],
            shared_state=data["shared_state"],
            agent_results=data["agent_results"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


class AgentCoordinator:
    """
    Coordinates communication and state management between agents.
    
    Manages:
    - Inter-agent messaging
    - Shared context store
    - Workflow state machine
    - Agent failure handling and recovery
    - Logging of all interactions
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize agent coordinator.
        
        Args:
            redis_client: Redis client for state storage (uses default if None)
        """
        self.redis = redis_client or get_redis_client()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._workflow_callbacks: Dict[str, List[Callable]] = {}
    
    # ==================== Message Management ====================
    
    def send_message(
        self,
        sender_agent: str,
        recipient_agent: Optional[str],
        message_type: AgentMessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Send a message from one agent to another.
        
        Args:
            sender_agent: Name of sending agent
            recipient_agent: Name of receiving agent (None for broadcast)
            message_type: Type of message
            payload: Message payload
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            Message ID
        """
        import uuid
        
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        message = AgentMessage(
            message_id=message_id,
            sender_agent=sender_agent,
            recipient_agent=recipient_agent,
            message_type=message_type,
            payload=payload,
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        
        # Store message in Redis
        message_key = f"agent_message:{message_id}"
        self.redis.setex(
            message_key,
            86400,  # 24 hour TTL
            json.dumps(message.to_dict())
        )
        
        # Add to message queue for recipient
        if recipient_agent:
            queue_key = f"agent_queue:{recipient_agent}"
        else:
            queue_key = "agent_broadcast_queue"
        
        self.redis.lpush(queue_key, message_id)
        
        self.logger.info(
            f"Message sent: {message_id} from {sender_agent} "
            f"to {recipient_agent or 'broadcast'} "
            f"(type: {message_type.value})"
        )
        
        return message_id
    
    def receive_messages(
        self,
        agent_name: str,
        max_messages: int = 10,
    ) -> List[AgentMessage]:
        """
        Receive messages for an agent.
        
        Args:
            agent_name: Name of agent receiving messages
            max_messages: Maximum messages to retrieve
            
        Returns:
            List of AgentMessage objects
        """
        queue_key = f"agent_queue:{agent_name}"
        message_ids = self.redis.lrange(queue_key, 0, max_messages - 1)
        
        messages = []
        for message_id in message_ids:
            message_key = f"agent_message:{message_id}"
            message_data = self.redis.get(message_key)
            
            if message_data:
                try:
                    message_dict = json.loads(message_data)
                    message = AgentMessage.from_dict(message_dict)
                    messages.append(message)
                    
                    # Remove from queue
                    self.redis.lrem(queue_key, 1, message_id)
                except Exception as e:
                    self.logger.error(f"Failed to parse message {message_id}: {e}")
        
        return messages
    
    def register_message_handler(
        self,
        agent_name: str,
        message_type: AgentMessageType,
        handler: Callable,
    ) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            agent_name: Name of agent
            message_type: Type of message to handle
            handler: Callable to handle the message
        """
        key = f"{agent_name}:{message_type.value}"
        
        if key not in self._message_handlers:
            self._message_handlers[key] = []
        
        self._message_handlers[key].append(handler)
        self.logger.info(f"Message handler registered: {key}")
    
    def handle_messages(
        self,
        agent_name: str,
        messages: List[AgentMessage],
    ) -> List[Any]:
        """
        Handle received messages using registered handlers.
        
        Args:
            agent_name: Name of agent
            messages: List of messages to handle
            
        Returns:
            List of handler results
        """
        results = []
        
        for message in messages:
            key = f"{agent_name}:{message.message_type.value}"
            handlers = self._message_handlers.get(key, [])
            
            if not handlers:
                self.logger.warning(
                    f"No handlers registered for {key}"
                )
                continue
            
            for handler in handlers:
                try:
                    result = handler(message)
                    results.append(result)
                    self.logger.info(
                        f"Message handled: {message.message_id} "
                        f"by {agent_name}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error handling message {message.message_id}: {e}"
                    )
        
        return results
    
    # ==================== Workflow Context Management ====================
    
    def create_workflow_context(
        self,
        workflow_id: str,
        workflow_type: str,
        input_data: Dict[str, Any],
    ) -> WorkflowContext:
        """
        Create a new workflow context.
        
        Args:
            workflow_id: Unique workflow ID
            workflow_type: Type of workflow
            input_data: Input data for workflow
            
        Returns:
            WorkflowContext object
        """
        now = datetime.now().isoformat()
        
        context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            state=WorkflowState.INITIATED,
            input_data=input_data,
            shared_state={},
            agent_results={},
            created_at=now,
            updated_at=now,
        )
        
        # Store in Redis
        context_key = f"workflow_context:{workflow_id}"
        self.redis.setex(
            context_key,
            86400,  # 24 hour TTL
            json.dumps(context.to_dict())
        )
        
        self.logger.info(
            f"Workflow context created: {workflow_id} "
            f"(type: {workflow_type})"
        )
        
        return context
    
    def get_workflow_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """
        Retrieve workflow context.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            WorkflowContext or None if not found
        """
        context_key = f"workflow_context:{workflow_id}"
        context_data = self.redis.get(context_key)
        
        if not context_data:
            return None
        
        try:
            context_dict = json.loads(context_data)
            return WorkflowContext.from_dict(context_dict)
        except Exception as e:
            self.logger.error(f"Failed to parse workflow context: {e}")
            return None
    
    def update_workflow_context(
        self,
        workflow_id: str,
        updates: Dict[str, Any],
    ) -> Optional[WorkflowContext]:
        """
        Update workflow context with new data.
        
        Args:
            workflow_id: Workflow ID
            updates: Dictionary of updates
            
        Returns:
            Updated WorkflowContext or None if not found
        """
        context = self.get_workflow_context(workflow_id)
        
        if not context:
            self.logger.error(f"Workflow context not found: {workflow_id}")
            return None
        
        # Update fields
        if "state" in updates:
            context.state = WorkflowState(updates["state"])
        if "shared_state" in updates:
            context.shared_state.update(updates["shared_state"])
        if "agent_results" in updates:
            context.agent_results.update(updates["agent_results"])
        
        context.updated_at = datetime.now().isoformat()
        
        # Store updated context
        context_key = f"workflow_context:{workflow_id}"
        self.redis.setex(
            context_key,
            86400,
            json.dumps(context.to_dict())
        )
        
        self.logger.info(f"Workflow context updated: {workflow_id}")
        
        return context
    
    def update_shared_state(
        self,
        workflow_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """
        Update a specific key in shared state.
        
        Args:
            workflow_id: Workflow ID
            key: State key
            value: State value
            
        Returns:
            True if successful, False otherwise
        """
        context = self.get_workflow_context(workflow_id)
        
        if not context:
            return False
        
        context.shared_state[key] = value
        context.updated_at = datetime.now().isoformat()
        
        context_key = f"workflow_context:{workflow_id}"
        self.redis.setex(
            context_key,
            86400,
            json.dumps(context.to_dict())
        )
        
        self.logger.info(
            f"Shared state updated: {workflow_id}[{key}]"
        )
        
        return True
    
    def store_agent_result(
        self,
        workflow_id: str,
        agent_name: str,
        result: Dict[str, Any],
    ) -> bool:
        """
        Store result from agent execution.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of agent
            result: Agent result
            
        Returns:
            True if successful, False otherwise
        """
        context = self.get_workflow_context(workflow_id)
        
        if not context:
            return False
        
        context.agent_results[agent_name] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        context.updated_at = datetime.now().isoformat()
        
        context_key = f"workflow_context:{workflow_id}"
        self.redis.setex(
            context_key,
            86400,
            json.dumps(context.to_dict())
        )
        
        self.logger.info(
            f"Agent result stored: {workflow_id} <- {agent_name}"
        )
        
        return True
    
    # ==================== State Machine ====================
    
    def transition_workflow_state(
        self,
        workflow_id: str,
        new_state: WorkflowState,
    ) -> bool:
        """
        Transition workflow to a new state.
        
        Args:
            workflow_id: Workflow ID
            new_state: New workflow state
            
        Returns:
            True if successful, False otherwise
        """
        context = self.get_workflow_context(workflow_id)
        
        if not context:
            return False
        
        old_state = context.state
        context.state = new_state
        context.updated_at = datetime.now().isoformat()
        
        context_key = f"workflow_context:{workflow_id}"
        self.redis.setex(
            context_key,
            86400,
            json.dumps(context.to_dict())
        )
        
        self.logger.info(
            f"Workflow state transitioned: {workflow_id} "
            f"{old_state.value} -> {new_state.value}"
        )
        
        # Trigger callbacks
        self._trigger_workflow_callbacks(workflow_id, old_state, new_state)
        
        return True
    
    def register_workflow_callback(
        self,
        workflow_type: str,
        state_transition: tuple,
        callback: Callable,
    ) -> None:
        """
        Register a callback for workflow state transitions.
        
        Args:
            workflow_type: Type of workflow
            state_transition: Tuple of (from_state, to_state)
            callback: Callable to execute
        """
        key = f"{workflow_type}:{state_transition[0].value}->{state_transition[1].value}"
        
        if key not in self._workflow_callbacks:
            self._workflow_callbacks[key] = []
        
        self._workflow_callbacks[key].append(callback)
        self.logger.info(f"Workflow callback registered: {key}")
    
    def _trigger_workflow_callbacks(
        self,
        workflow_id: str,
        old_state: WorkflowState,
        new_state: WorkflowState,
    ) -> None:
        """
        Trigger callbacks for state transition.
        
        Args:
            workflow_id: Workflow ID
            old_state: Previous state
            new_state: New state
        """
        context = self.get_workflow_context(workflow_id)
        if not context:
            return
        
        key = f"{context.workflow_type}:{old_state.value}->{new_state.value}"
        callbacks = self._workflow_callbacks.get(key, [])
        
        for callback in callbacks:
            try:
                callback(workflow_id, context)
                self.logger.info(
                    f"Workflow callback executed: {key} for {workflow_id}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error executing workflow callback: {e}"
                )
    
    # ==================== Failure Handling & Recovery ====================
    
    def record_agent_failure(
        self,
        workflow_id: str,
        agent_name: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record an agent failure for recovery.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of failed agent
            error: Error message
            error_details: Additional error details
            
        Returns:
            True if successful, False otherwise
        """
        failure_key = f"agent_failure:{workflow_id}:{agent_name}"
        failure_data = {
            "agent_name": agent_name,
            "error": error,
            "error_details": error_details or {},
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0,
        }
        
        self.redis.setex(
            failure_key,
            86400,
            json.dumps(failure_data)
        )
        
        self.logger.error(
            f"Agent failure recorded: {workflow_id} <- {agent_name}: {error}"
        )
        
        return True
    
    def get_agent_failure(
        self,
        workflow_id: str,
        agent_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve recorded agent failure.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of agent
            
        Returns:
            Failure data or None if not found
        """
        failure_key = f"agent_failure:{workflow_id}:{agent_name}"
        failure_data = self.redis.get(failure_key)
        
        if not failure_data:
            return None
        
        try:
            return json.loads(failure_data)
        except Exception as e:
            self.logger.error(f"Failed to parse failure data: {e}")
            return None
    
    def increment_retry_count(
        self,
        workflow_id: str,
        agent_name: str,
    ) -> int:
        """
        Increment retry count for a failed agent.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of agent
            
        Returns:
            New retry count
        """
        failure = self.get_agent_failure(workflow_id, agent_name)
        
        if not failure:
            return 0
        
        failure["retry_count"] += 1
        failure_key = f"agent_failure:{workflow_id}:{agent_name}"
        
        self.redis.setex(
            failure_key,
            86400,
            json.dumps(failure)
        )
        
        self.logger.info(
            f"Retry count incremented: {workflow_id} <- {agent_name} "
            f"(count: {failure['retry_count']})"
        )
        
        return failure["retry_count"]
    
    def clear_agent_failure(
        self,
        workflow_id: str,
        agent_name: str,
    ) -> bool:
        """
        Clear failure record for an agent.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of agent
            
        Returns:
            True if successful
        """
        failure_key = f"agent_failure:{workflow_id}:{agent_name}"
        self.redis.delete(failure_key)
        
        self.logger.info(
            f"Agent failure cleared: {workflow_id} <- {agent_name}"
        )
        
        return True
    
    # ==================== Logging & Monitoring ====================
    
    def log_agent_interaction(
        self,
        workflow_id: str,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log agent interaction for debugging and monitoring.
        
        Args:
            workflow_id: Workflow ID
            agent_name: Name of agent
            action: Action performed
            details: Additional details
        """
        log_entry = {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        log_key = f"agent_interaction_log:{workflow_id}"
        self.redis.lpush(log_key, json.dumps(log_entry))
        self.redis.expire(log_key, 86400)  # 24 hour TTL
        
        self.logger.info(
            f"Agent interaction logged: {workflow_id} <- {agent_name}: {action}"
        )
    
    def get_agent_interaction_log(
        self,
        workflow_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent interaction log for a workflow.
        
        Args:
            workflow_id: Workflow ID
            limit: Maximum log entries to retrieve
            
        Returns:
            List of log entries
        """
        log_key = f"agent_interaction_log:{workflow_id}"
        log_entries = self.redis.lrange(log_key, 0, limit - 1)
        
        entries = []
        for entry_data in log_entries:
            try:
                entry = json.loads(entry_data)
                entries.append(entry)
            except Exception as e:
                self.logger.error(f"Failed to parse log entry: {e}")
        
        return entries


# Global coordinator instance
_agent_coordinator = None


def get_agent_coordinator() -> AgentCoordinator:
    """Get or create global agent coordinator."""
    global _agent_coordinator
    if _agent_coordinator is None:
        _agent_coordinator = AgentCoordinator()
    return _agent_coordinator
