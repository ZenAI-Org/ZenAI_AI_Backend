"""
Structured logging module using Python's logging with JSON formatting.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper around Python logger for structured logging."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up logging handlers with JSON formatting."""
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            formatter = JSONFormatter()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log with extra context fields."""
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            exc_info=sys.exc_info() if exc_info else None,
        )
        
        if extra_fields:
            record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def debug(
        self,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, extra_fields)
    
    def info(
        self,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, extra_fields)
    
    def warning(
        self,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, extra_fields)
    
    def error(
        self,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, extra_fields, exc_info)
    
    def critical(
        self,
        message: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, extra_fields, exc_info)


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger."""
    return StructuredLogger(name)
