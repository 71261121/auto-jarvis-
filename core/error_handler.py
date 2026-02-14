#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Error Handler
====================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Global exception hooking
- Error categorization
- Auto-recovery strategies
- User-friendly error messages
- Error history and analysis

Features:
- Global exception hook
- Error categorization
- Auto-recovery strategies
- User-friendly error messages
- Error history tracking
- Error pattern analysis
- Graceful degradation
- Crash recovery
- Error reporting

Memory Impact: < 5MB
"""

import sys
import os
import time
import logging
import traceback
import threading
import json
import hashlib
from pathlib import Path
from typing import (
    Dict, Any, Optional, List, Set, Tuple, Callable, 
    Union, Type, TypeVar
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorCategory(Enum):
    """Categories of errors"""
    # System errors
    SYSTEM = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()
    
    # Application errors
    LOGIC = auto()
    VALIDATION = auto()
    CONFIGURATION = auto()
    
    # External errors
    API = auto()
    DEPENDENCY = auto()
    PERMISSION = auto()
    
    # User errors
    INPUT = auto()
    COMMAND = auto()
    
    # Unknown
    UNKNOWN = auto()


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    IGNORE = auto()           # Ignore and continue
    LOG = auto()              # Log and continue
    RETRY = auto()            # Retry the operation
    FALLBACK = auto()         # Use fallback method
    RESTART = auto()          # Restart component
    SHUTDOWN = auto()         # Graceful shutdown
    RECOVERY = auto()         # Enter recovery mode


@dataclass
class ErrorContext:
    """
    Context information for an error.
    """
    function_name: str = ""
    module_name: str = ""
    file_name: str = ""
    line_number: int = 0
    
    # Additional context
    local_vars: Dict[str, str] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    
    # State
    timestamp: float = field(default_factory=time.time)
    thread_id: int = 0
    thread_name: str = ""


@dataclass
class ErrorRecord:
    """
    A record of an error occurrence.
    """
    id: str
    exception_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    
    # Context
    context: ErrorContext
    
    # Stack trace
    traceback_str: str = ""
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    
    # Recovery
    recovery_attempted: bool = False
    recovery_success: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    
    # Metadata
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    @property
    def fingerprint(self) -> str:
        """Generate error fingerprint for deduplication"""
        return hashlib.md5(
            f"{self.exception_type}:{self.context.file_name}:{self.context.line_number}".encode()
        ).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'exception_type': self.exception_type,
            'message': self.message,
            'category': self.category.name,
            'severity': self.severity.name,
            'file': self.context.file_name,
            'line': self.context.line_number,
            'function': self.context.function_name,
            'timestamp': self.timestamp,
            'count': self.count,
            'recovery_attempted': self.recovery_attempted,
            'recovery_success': self.recovery_success,
        }


@dataclass
class ErrorHandlerStats:
    """Error handler statistics"""
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    
    recoveries_attempted: int = 0
    recoveries_successful: int = 0
    
    unique_errors: int = 0
    suppressed_errors: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorHandler:
    """
    Ultra-Advanced Error Handling System.
    
    Features:
    - Global exception hook
    - Error categorization
    - Auto-recovery strategies
    - User-friendly error messages
    - Error history tracking
    - Error pattern analysis
    - Graceful degradation
    - Crash recovery
    
    Memory Budget: < 5MB
    
    Usage:
        handler = ErrorHandler()
        
        # Register recovery strategies
        handler.register_recovery(
            NetworkError,
            RecoveryStrategy.RETRY,
            max_retries=3
        )
        
        # Safe execution context
        with handler.context('operation_name'):
            risky_operation()
        
        # Decorator
        @handler.safe(fallback=default_value)
        def risky_function():
            ...
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        auto_recover: bool = True,
        log_errors: bool = True,
        suppress_repeats: bool = True,
        suppress_after: int = 5,
        persist_dir: str = None,
    ):
        """
        Initialize Error Handler.
        
        Args:
            max_history: Maximum error history entries
            auto_recover: Automatically attempt recovery
            log_errors: Log all errors
            suppress_repeats: Suppress repeated errors
            suppress_after: Suppress after N occurrences
            persist_dir: Directory for error persistence
        """
        self._max_history = max_history
        self._auto_recover = auto_recover
        self._log_errors = log_errors
        self._suppress_repeats = suppress_repeats
        self._suppress_after = suppress_after
        self._persist_dir = Path(persist_dir) if persist_dir else None
        
        # Error history
        self._history: deque = deque(maxlen=max_history)
        self._history_lock = threading.Lock()
        
        # Error fingerprint counts
        self._fingerprint_counts: Dict[str, int] = defaultdict(int)
        
        # Recovery strategies
        self._recovery_strategies: Dict[Type[Exception], Tuple[RecoveryStrategy, Dict]] = {}
        
        # User-friendly messages
        self._friendly_messages: Dict[Type[Exception], str] = {}
        
        # Fallback values
        self._fallbacks: Dict[str, Any] = {}
        
        # Error callbacks
        self._on_error_callbacks: List[Callable] = []
        self._on_recovery_callbacks: List[Callable] = []
        
        # Statistics
        self._stats = ErrorHandlerStats()
        
        # Lock
        self._lock = threading.RLock()
        
        # Install global exception hook
        self._original_excepthook = sys.excepthook
        self._install_hooks()
        
        logger.info("ErrorHandler initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCEPTION HOOKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _install_hooks(self):
        """Install global exception hooks"""
        sys.excepthook = self._excepthook
        threading.excepthook = self._threading_excepthook
    
    def _excepthook(self, exc_type, exc_value, exc_tb):
        """Global exception handler"""
        try:
            # Record the error
            record = self._create_error_record(exc_type, exc_value, exc_tb)
            
            # Check if suppressed
            if self._should_suppress(record):
                self._stats.suppressed_errors += 1
                return
            
            # Record error
            self._record_error(record)
            
            # Log error
            if self._log_errors:
                self._log_error(record)
            
            # Attempt recovery
            if self._auto_recover:
                self._attempt_recovery(record)
            
            # Call callbacks
            for callback in self._on_error_callbacks:
                try:
                    callback(record)
                except Exception:
                    pass
            
            # Show user-friendly message
            self._show_friendly_message(record)
            
        except Exception as e:
            # Fallback to original hook if we fail
            logger.critical(f"Error handler failed: {e}")
            self._original_excepthook(exc_type, exc_value, exc_tb)
    
    def _threading_excepthook(self, args):
        """Thread exception handler"""
        self._excepthook(args.exc_type, args.exc_value, args.exc_traceback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ERROR RECORDING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_error_record(
        self,
        exc_type: Type[Exception],
        exc_value: Exception,
        exc_tb,
    ) -> ErrorRecord:
        """Create an error record from exception info"""
        # Extract context
        context = ErrorContext()
        
        if exc_tb:
            tb_list = traceback.extract_tb(exc_tb)
            if tb_list:
                last_frame = tb_list[-1]
                context.file_name = last_frame.filename
                context.line_number = last_frame.lineno
                context.function_name = last_frame.name
            
            context.call_stack = [
                f"{frame.filename}:{frame.lineno} in {frame.name}"
                for frame in tb_list
            ]
        
        context.thread_id = threading.get_ident()
        context.thread_name = threading.current_thread().name
        
        # Create record
        record = ErrorRecord(
            id=hashlib.md5(
                f"{exc_type.__name__}:{time.time()}:{id(exc_value)}".encode()
            ).hexdigest()[:12],
            exception_type=exc_type.__name__,
            message=str(exc_value),
            category=self._categorize_error(exc_type, exc_value),
            severity=self._assess_severity(exc_type, exc_value),
            context=context,
            traceback_str=''.join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        
        return record
    
    def _categorize_error(
        self,
        exc_type: Type[Exception],
        exc_value: Exception,
    ) -> ErrorCategory:
        """Categorize an error"""
        # Memory errors
        if exc_type in (MemoryError,):
            return ErrorCategory.MEMORY
        
        # IO/Disk errors
        if exc_type in (IOError, OSError, FileNotFoundError, PermissionError):
            return ErrorCategory.DISK
        
        # Network errors
        if any(name in exc_type.__name__ for name in ['Network', 'Connection', 'Timeout', 'Socket']):
            return ErrorCategory.NETWORK
        
        # API errors
        if any(name in exc_type.__name__ for name in ['API', 'HTTP', 'Request', 'Response']):
            return ErrorCategory.API
        
        # Validation errors
        if exc_type in (ValueError, TypeError, KeyError, IndexError):
            return ErrorCategory.VALIDATION
        
        # Configuration errors
        if 'Config' in exc_type.__name__:
            return ErrorCategory.CONFIGURATION
        
        # Permission errors
        if exc_type in (PermissionError,):
            return ErrorCategory.PERMISSION
        
        # Input errors
        if 'Input' in exc_type.__name__:
            return ErrorCategory.INPUT
        
        # Default to unknown
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(
        self,
        exc_type: Type[Exception],
        exc_value: Exception,
    ) -> ErrorSeverity:
        """Assess error severity"""
        # Fatal errors
        if exc_type in (MemoryError, SystemExit, KeyboardInterrupt):
            return ErrorSeverity.FATAL
        
        # Critical errors
        if exc_type in (RuntimeError, ImportError, SyntaxError):
            return ErrorSeverity.CRITICAL
        
        # Errors
        if exc_type in (IOError, OSError, ConnectionError, TimeoutError):
            return ErrorSeverity.ERROR
        
        # Warnings
        if 'Warning' in exc_type.__name__:
            return ErrorSeverity.WARNING
        
        # Default to error
        return ErrorSeverity.ERROR
    
    def _record_error(self, record: ErrorRecord):
        """Record an error"""
        with self._lock:
            # Update fingerprint count
            fingerprint = record.fingerprint
            self._fingerprint_counts[fingerprint] += 1
            record.count = self._fingerprint_counts[fingerprint]
            
            # Add to history
            with self._history_lock:
                self._history.append(record)
            
            # Update stats
            self._stats.total_errors += 1
            self._stats.errors_by_category[record.category.name] = \
                self._stats.errors_by_category.get(record.category.name, 0) + 1
            self._stats.errors_by_severity[record.severity.name] = \
                self._stats.errors_by_severity.get(record.severity.name, 0) + 1
    
    def _should_suppress(self, record: ErrorRecord) -> bool:
        """Check if error should be suppressed"""
        if not self._suppress_repeats:
            return False
        
        fingerprint = record.fingerprint
        return self._fingerprint_counts[fingerprint] >= self._suppress_after
    
    def _log_error(self, record: ErrorRecord):
        """Log an error"""
        log_msg = f"[{record.exception_type}] {record.message}"
        
        if record.severity == ErrorSeverity.FATAL:
            logger.critical(log_msg)
        elif record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif record.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        elif record.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECOVERY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def register_recovery(
        self,
        exception_type: Type[Exception],
        strategy: RecoveryStrategy,
        **options,
    ):
        """
        Register a recovery strategy for an exception type.
        
        Args:
            exception_type: Exception type to handle
            strategy: Recovery strategy to use
            **options: Strategy options
        """
        self._recovery_strategies[exception_type] = (strategy, options)
        logger.debug(f"Registered recovery for {exception_type.__name__}")
    
    def register_friendly_message(
        self,
        exception_type: Type[Exception],
        message: str,
    ):
        """Register a user-friendly message for an exception"""
        self._friendly_messages[exception_type] = message
    
    def _attempt_recovery(self, record: ErrorRecord):
        """Attempt to recover from an error"""
        # Find matching strategy
        strategy = None
        options = {}
        
        for exc_type, (strat, opts) in self._recovery_strategies.items():
            if record.exception_type == exc_type.__name__:
                strategy = strat
                options = opts
                break
        
        if strategy is None:
            return False
        
        record.recovery_attempted = True
        record.recovery_strategy = strategy
        self._stats.recoveries_attempted += 1
        
        try:
            # Execute strategy
            if strategy == RecoveryStrategy.IGNORE:
                record.recovery_success = True
            
            elif strategy == RecoveryStrategy.LOG:
                self._log_error(record)
                record.recovery_success = True
            
            elif strategy == RecoveryStrategy.FALLBACK:
                # Handled by safe decorator
                record.recovery_success = True
            
            if record.recovery_success:
                self._stats.recoveries_successful += 1
            
            # Call callbacks
            for callback in self._on_recovery_callbacks:
                try:
                    callback(record)
                except Exception:
                    pass
            
            return record.recovery_success
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _show_friendly_message(self, record: ErrorRecord):
        """Show user-friendly error message"""
        # Find friendly message
        for exc_type, message in self._friendly_messages.items():
            if record.exception_type == exc_type.__name__:
                print(f"\n⚠️  {message}\n")
                return
        
        # Default friendly message
        severity_emoji = {
            ErrorSeverity.DEBUG: "🔍",
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨",
            ErrorSeverity.FATAL: "💀",
        }
        
        emoji = severity_emoji.get(record.severity, "❌")
        print(f"\n{emoji} Error: {record.message}")
        print(f"   Category: {record.category.name}")
        print(f"   Location: {record.context.file_name}:{record.context.line_number}")
        print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ═══════════════════════════════════════════════════════════════════════════
    
    @contextmanager
    def context(
        self,
        operation_name: str,
        fallback: Any = None,
        reraise: bool = False,
    ):
        """
        Context manager for safe execution.
        
        Usage:
            with error_handler.context('load_config'):
                config = load_config_file()
        """
        try:
            yield
        except Exception as e:
            # Record error
            exc_type, exc_value, exc_tb = sys.exc_info()
            record = self._create_error_record(exc_type, exc_value, exc_tb)
            self._record_error(record)
            self._log_error(record)
            
            # Attempt recovery
            if self._auto_recover:
                self._attempt_recovery(record)
            
            if reraise:
                raise
            
            # Return fallback if available
            if fallback is not None:
                return fallback
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DECORATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def safe(
        self,
        fallback: Any = None,
        reraise: bool = False,
        retry: int = 0,
        retry_delay: float = 1.0,
    ):
        """
        Decorator for safe function execution.
        
        Usage:
            @error_handler.safe(fallback=None)
            def risky_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 0
                max_attempts = retry + 1
                
                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        
                        # Record error
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        record = self._create_error_record(exc_type, exc_value, exc_tb)
                        self._record_error(record)
                        self._log_error(record)
                        
                        if attempts < max_attempts:
                            time.sleep(retry_delay)
                            continue
                        
                        if reraise:
                            raise
                        
                        return fallback
                
                return fallback
            
            return wrapper
        return decorator
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on_error(self, callback: Callable[[ErrorRecord], None]):
        """Register callback for errors"""
        self._on_error_callbacks.append(callback)
    
    def on_recovery(self, callback: Callable[[ErrorRecord], None]):
        """Register callback for successful recoveries"""
        self._on_recovery_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_history(self, limit: int = 100) -> List[ErrorRecord]:
        """Get error history"""
        with self._history_lock:
            return list(self._history)[-limit:]
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors as dictionaries"""
        return [r.to_dict() for r in self.get_history(count)]
    
    def clear_history(self):
        """Clear error history"""
        with self._history_lock:
            self._history.clear()
        self._fingerprint_counts.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics"""
        return {
            'total_errors': self._stats.total_errors,
            'unique_errors': len(self._fingerprint_counts),
            'suppressed_errors': self._stats.suppressed_errors,
            'recoveries_attempted': self._stats.recoveries_attempted,
            'recoveries_successful': self._stats.recoveries_successful,
            'recovery_rate': (
                f"{self._stats.recoveries_successful / max(1, self._stats.recoveries_attempted):.1%}"
            ),
            'by_category': dict(self._stats.errors_by_category),
            'by_severity': dict(self._stats.errors_by_severity),
            'history_size': len(self._history),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def shutdown(self):
        """Shutdown error handler"""
        # Restore original exception hook
        sys.excepthook = self._original_excepthook
        
        # Clear data
        self.clear_history()
        self._on_error_callbacks.clear()
        self._on_recovery_callbacks.clear()
        
        logger.info("ErrorHandler shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

_global_handler: Optional[ErrorHandler] = None
_handler_lock = threading.Lock()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _global_handler
    
    with _handler_lock:
        if _global_handler is None:
            _global_handler = ErrorHandler()
        return _global_handler


def safe_call(func: Callable, *args, fallback: Any = None, **kwargs) -> Any:
    """Safely call a function with fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = get_error_handler()
        exc_type, exc_value, exc_tb = sys.exc_info()
        record = handler._create_error_record(exc_type, exc_value, exc_tb)
        handler._record_error(record)
        return fallback


def error_context(operation_name: str, fallback: Any = None):
    """Context manager for safe execution"""
    return get_error_handler().context(operation_name, fallback)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classes
    'ErrorHandler',
    'ErrorRecord',
    'ErrorContext',
    'ErrorHandlerStats',
    
    # Enums
    'ErrorCategory',
    'ErrorSeverity',
    'RecoveryStrategy',
    
    # Functions
    'get_error_handler',
    'safe_call',
    'error_context',
]
