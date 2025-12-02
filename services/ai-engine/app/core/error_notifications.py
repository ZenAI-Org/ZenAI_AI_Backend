"""
Error notification system for emitting error events via Socket.io.
Provides user-friendly error messages and real-time error notifications.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from app.core.logger import get_logger
from app.core.error_dashboard import get_error_metrics

logger = get_logger(__name__)


class ErrorNotificationManager:
    """Manage error notifications and user-friendly messages."""
    
    # User-friendly error messages
    ERROR_MESSAGES = {
        "whisper_timeout": "Audio transcription is taking longer than expected. Please try again.",
        "whisper_api_error": "Unable to transcribe audio. Please check the audio file and try again.",
        "whisper_rate_limit": "Transcription service is busy. Please try again in a few moments.",
        "openai_timeout": "AI service is taking longer than expected. Please try again.",
        "openai_api_error": "Unable to process with AI service. Please try again.",
        "openai_rate_limit": "AI service is busy. Please try again in a few moments.",
        "notion_timeout": "Notion sync is taking longer than expected. Please try again.",
        "notion_api_error": "Unable to sync with Notion. Please check your Notion integration.",
        "notion_rate_limit": "Notion service is busy. Please try again in a few moments.",
        "database_error": "Database operation failed. Please try again.",
        "validation_error": "Invalid data provided. Please check your input.",
        "unknown_error": "An unexpected error occurred. Please try again.",
    }
    
    def __init__(self, socketio_manager: Optional[Any] = None):
        """
        Initialize error notification manager.
        
        Args:
            socketio_manager: Optional Socket.io manager for emitting notifications
        """
        self.socketio_manager = socketio_manager
        self.error_metrics = get_error_metrics()
    
    def set_socketio_manager(self, socketio_manager: Any) -> None:
        """
        Set Socket.io manager for emitting notifications.
        
        Args:
            socketio_manager: Socket.io manager instance
        """
        self.socketio_manager = socketio_manager
    
    def get_user_friendly_message(self, error_key: str) -> str:
        """
        Get user-friendly error message.
        
        Args:
            error_key: Error key (e.g., "whisper_timeout")
            
        Returns:
            User-friendly error message
        """
        return self.ERROR_MESSAGES.get(error_key, self.ERROR_MESSAGES["unknown_error"])
    
    def notify_error(
        self,
        project_id: str,
        error_type: str,
        api_name: str,
        message: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        """
        Notify users of an error via Socket.io.
        
        Args:
            project_id: Project ID for targeting notification
            error_type: Type of error
            api_name: Name of the API that failed
            message: Technical error message
            severity: Error severity level
            details: Additional error details
            user_message: User-friendly message (auto-generated if not provided)
        """
        try:
            # Record error in metrics
            self.error_metrics.record_error(
                error_type=error_type,
                api_name=api_name,
                severity=severity,
                message=message,
                details=details,
            )
            
            # Generate user-friendly message if not provided
            if not user_message:
                error_key = f"{api_name}_{error_type}".lower()
                user_message = self.get_user_friendly_message(error_key)
            
            # Create notification payload
            notification = {
                "type": "error",
                "error_type": error_type,
                "api_name": api_name,
                "severity": severity,
                "message": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {},
            }
            
            # Emit via Socket.io if available
            if self.socketio_manager:
                try:
                    self.socketio_manager.emit_to_project_async(
                        project_id,
                        "error_notification",
                        notification,
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit error notification: {e}")
            
            # Log the notification
            logger.warning(
                f"Error notification: {error_type} from {api_name}",
                extra_fields={
                    "project_id": project_id,
                    "error_type": error_type,
                    "api_name": api_name,
                    "severity": severity,
                    "user_message": user_message,
                },
            )
        
        except Exception as e:
            logger.error(f"Failed to notify error: {e}")
    
    def notify_recovery(
        self,
        project_id: str,
        api_name: str,
        message: str = "Service recovered",
    ) -> None:
        """
        Notify users that a service has recovered.
        
        Args:
            project_id: Project ID for targeting notification
            api_name: Name of the API that recovered
            message: Recovery message
        """
        try:
            notification = {
                "type": "recovery",
                "api_name": api_name,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Emit via Socket.io if available
            if self.socketio_manager:
                try:
                    self.socketio_manager.emit_to_project_async(
                        project_id,
                        "service_recovery",
                        notification,
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit recovery notification: {e}")
            
            logger.info(
                f"Service recovery notification: {api_name}",
                extra_fields={
                    "project_id": project_id,
                    "api_name": api_name,
                },
            )
        
        except Exception as e:
            logger.error(f"Failed to notify recovery: {e}")
    
    def notify_alert(
        self,
        project_id: str,
        alert_type: str,
        message: str,
        severity: str = "high",
    ) -> None:
        """
        Notify users of an alert (e.g., high error rate).
        
        Args:
            project_id: Project ID for targeting notification
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
        """
        try:
            alert = {
                "type": "alert",
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Emit via Socket.io if available
            if self.socketio_manager:
                try:
                    self.socketio_manager.emit_to_project_async(
                        project_id,
                        "system_alert",
                        alert,
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit alert notification: {e}")
            
            logger.warning(
                f"System alert: {alert_type}",
                extra_fields={
                    "project_id": project_id,
                    "alert_type": alert_type,
                    "severity": severity,
                },
            )
        
        except Exception as e:
            logger.error(f"Failed to notify alert: {e}")
    
    def check_and_alert_error_rate(self, project_id: str, threshold: float = 5.0) -> bool:
        """
        Check error rate and send alert if threshold exceeded.
        
        Args:
            project_id: Project ID for targeting notification
            threshold: Alert threshold as percentage
            
        Returns:
            True if alert was sent
        """
        if self.error_metrics.should_alert(threshold):
            metrics = self.error_metrics.get_metrics_summary()
            
            self.notify_alert(
                project_id,
                "high_error_rate",
                f"Error rate is {metrics['error_rate']:.1f}% (threshold: {threshold}%)",
                severity="high",
            )
            
            return True
        
        return False


# Global error notification manager instance
_error_notification_manager = None


def get_error_notification_manager() -> ErrorNotificationManager:
    """Get or create global error notification manager."""
    global _error_notification_manager
    if _error_notification_manager is None:
        _error_notification_manager = ErrorNotificationManager()
    return _error_notification_manager
