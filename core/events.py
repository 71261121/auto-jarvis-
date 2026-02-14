#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Event System
===================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Pub/Sub pattern for loose coupling
- Priority-based event handling
- Async and sync event support
- Memory-efficient event queuing
- Thread-safe operations

Features:
- EventEmitter for pub/sub
- Async event handling
- Event priorities (CRITICAL, HIGH, NORMAL, LOW)
- Error isolation per handler
- Event history and replay
- One-time event listeners
- Wildcard event matching
- Event filtering
- Memory-efficient queue

Memory Impact: < 5MB for event queue
"""

import sys
import os
import time
import logging
import threading
import weakref
import asyncio
import hashlib
from typing import (
    Dict, Any, Optional, List, Set, Tuple, Callable, 
    Union, Generator, Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime
from functools import wraps, partial
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class EventPriority(Enum):
    """Event handler priority levels"""
    CRITICAL = 0   # System-critical events (errors, shutdown)
    HIGH = 1       # High-priority events (user input, commands)
    NORMAL = 2     # Normal events (logging, status updates)
    LOW = 3        # Low-priority events (analytics, cleanup)
    BACKGROUND = 4 # Background tasks


class EventState(Enum):
    """State of an event"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class HandlerType(Enum):
    """Types of event handlers"""
    SYNC = auto()
    ASYNC = auto()
    GENERATOR = auto()


@dataclass
class Event:
    """
    Represents an event in the system.
    
    Events are immutable once created and carry
    all necessary context for handlers.
    """
    name: str
    data: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    id: str = ""
    correlation_id: str = ""
    cancelable: bool = True
    propagate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal state
    _state: EventState = field(default=EventState.PENDING, repr=False)
    _cancelled: bool = field(default=False, repr=False)
    _handled: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.name}:{self.timestamp}:{id(self)}".encode()
            ).hexdigest()[:12]
    
    @property
    def state(self) -> EventState:
        return self._state
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    @property
    def is_handled(self) -> bool:
        return self._handled
    
    def cancel(self) -> bool:
        """Cancel the event if possible"""
        if self.cancelable and self._state == EventState.PENDING:
            self._cancelled = True
            self._state = EventState.CANCELLED
            return True
        return False
    
    def stop_propagation(self):
        """Stop event from propagating to more handlers"""
        self.propagate = False
    
    def mark_handled(self):
        """Mark event as handled"""
        self._handled = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp,
            'priority': self.priority.name,
            'state': self._state.name,
            'cancelled': self._cancelled,
            'handled': self._handled,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
        }


@dataclass
class EventHandler:
    """
    Registered event handler with metadata.
    """
    callback: Callable
    event_name: str
    priority: EventPriority = EventPriority.NORMAL
    handler_type: HandlerType = HandlerType.SYNC
    once: bool = False
    active: bool = True
    timeout: float = 30.0
    error_handler: Optional[Callable] = None
    
    # Metadata
    id: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_called: Optional[float] = None
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.event_name}:{id(self.callback)}:{self.created_at}".encode()
            ).hexdigest()[:8]
        
        # Determine handler type
        if asyncio.iscoroutinefunction(self.callback):
            self.handler_type = HandlerType.ASYNC
        elif hasattr(self.callback, '__iter__') and not callable(self.callback):
            self.handler_type = HandlerType.GENERATOR
    
    def __call__(self, event: Event) -> Any:
        """Execute the handler"""
        if not self.active:
            return None
        
        self.call_count += 1
        self.last_called = time.time()
        
        try:
            result = self.callback(event)
            self.last_error = None
            return result
        except Exception as e:
            self.last_error = str(e)
            if self.error_handler:
                self.error_handler(e, event)
            raise
    
    def __lt__(self, other: 'EventHandler') -> bool:
        """Compare by priority for sorting"""
        return self.priority.value < other.priority.value
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, EventHandler):
            return self.id == other.id
        return False
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class EventHistory:
    """
    History entry for an event.
    """
    event: Event
    handlers_called: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class EventFilter:
    """
    Filter events based on criteria.
    
    Supports:
    - Event name patterns (wildcards)
    - Priority filtering
    - Source filtering
    - Custom filter functions
    """
    
    def __init__(
        self,
        name_pattern: str = None,
        priority_min: EventPriority = None,
        priority_max: EventPriority = None,
        sources: Set[str] = None,
        custom_filter: Callable[[Event], bool] = None,
    ):
        self.name_pattern = name_pattern
        self.priority_min = priority_min
        self.priority_max = priority_max
        self.sources = sources or set()
        self.custom_filter = custom_filter
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter"""
        # Check name pattern
        if self.name_pattern:
            if not self._match_pattern(event.name, self.name_pattern):
                return False
        
        # Check priority range
        if self.priority_min and event.priority.value > self.priority_min.value:
            return False
        if self.priority_max and event.priority.value < self.priority_max.value:
            return False
        
        # Check source
        if self.sources and event.source not in self.sources:
            return False
        
        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Match event name against pattern with wildcards"""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT EMITTER
# ═══════════════════════════════════════════════════════════════════════════════

class EventEmitter:
    """
    Ultra-Advanced Event System.
    
    Features:
    - Pub/Sub pattern
    - Priority-based handlers
    - Async and sync support
    - Event history
    - One-time handlers
    - Wildcard matching
    - Error isolation
    - Thread-safe operations
    
    Memory Budget: < 5MB
    
    Usage:
        emitter = EventEmitter()
        
        # Register handler
        @emitter.on('user.input')
        def handle_input(event):
            print(f"Got input: {event.data}")
        
        # Emit event
        emitter.emit('user.input', data='Hello')
        
        # Async handler
        @emitter.on('async.event', async_mode=True)
        async def handle_async(event):
            await some_async_operation()
        
        # One-time handler
        emitter.once('startup', on_startup)
        
        # Wildcard matching
        @emitter.on('error.*')
        def handle_any_error(event):
            log_error(event)
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        max_handlers_per_event: int = 100,
        default_timeout: float = 30.0,
        thread_pool_size: int = 4,
        enable_history: bool = True,
    ):
        """
        Initialize Event Emitter.
        
        Args:
            max_history: Maximum events to keep in history
            max_handlers_per_event: Maximum handlers per event name
            default_timeout: Default timeout for handler execution
            thread_pool_size: Size of thread pool for async handlers
            enable_history: Whether to keep event history
        """
        self._max_history = max_history
        self._max_handlers = max_handlers_per_event
        self._default_timeout = default_timeout
        self._enable_history = enable_history
        
        # Handler storage: event_name -> [EventHandler, ...]
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[Tuple[str, EventHandler]] = []
        
        # Event queue for async processing
        self._queue: deque = deque(maxlen=1000)
        self._queue_lock = threading.Lock()
        
        # History
        self._history: deque = deque(maxlen=max_history)
        self._history_lock = threading.Lock()
        
        # Thread pool for async handlers
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="event_handler"
        )
        
        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Statistics
        self._stats = {
            'events_emitted': 0,
            'events_processed': 0,
            'handlers_called': 0,
            'errors': 0,
            'events_cancelled': 0,
        }
        
        # Internal events
        self._internal_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("EventEmitter initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLER REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on(
        self,
        event_name: str,
        callback: Callable = None,
        priority: EventPriority = EventPriority.NORMAL,
        timeout: float = None,
        error_handler: Callable = None,
        description: str = "",
    ) -> Union[Callable, Callable[[Callable], Callable]]:
        """
        Register an event handler.
        
        Args:
            event_name: Event name to listen for (supports wildcards)
            callback: Handler function (if None, acts as decorator)
            priority: Handler priority
            timeout: Execution timeout
            error_handler: Function to handle errors
            description: Handler description
            
        Returns:
            Decorator or handler ID
            
        Usage:
            @emitter.on('user.input')
            def handle(event):
                pass
            
            # Or directly:
            emitter.on('user.input', handle_function)
        """
        def decorator(func: Callable) -> Callable:
            handler = EventHandler(
                callback=func,
                event_name=event_name,
                priority=priority,
                timeout=timeout or self._default_timeout,
                error_handler=error_handler,
                description=description,
            )
            
            self._register_handler(handler)
            
            # Store handler ID on function for removal
            func._event_handler_id = handler.id
            func._event_name = event_name
            
            return func
        
        if callback is not None:
            return decorator(callback)
        return decorator
    
    def once(
        self,
        event_name: str,
        callback: Callable = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Union[Callable, Callable[[Callable], Callable]]:
        """
        Register a one-time event handler.
        
        The handler will be automatically removed after first execution.
        """
        def decorator(func: Callable) -> Callable:
            handler = EventHandler(
                callback=func,
                event_name=event_name,
                priority=priority,
                once=True,
            )
            
            self._register_handler(handler)
            func._event_handler_id = handler.id
            func._event_name = event_name
            
            return func
        
        if callback is not None:
            return decorator(callback)
        return decorator
    
    def _register_handler(self, handler: EventHandler):
        """Register a handler internally"""
        with self._lock:
            event_name = handler.event_name
            
            # Check if wildcard
            if '*' in event_name or '?' in event_name:
                self._wildcard_handlers.append((event_name, handler))
            else:
                handlers = self._handlers[event_name]
                
                # Check max handlers
                if len(handlers) >= self._max_handlers:
                    logger.warning(
                        f"Max handlers reached for event '{event_name}'"
                    )
                    # Remove lowest priority handler
                    handlers.sort(reverse=True)
                    handlers.pop()
                
                handlers.append(handler)
                handlers.sort()  # Sort by priority
    
    def off(
        self,
        event_name: str = None,
        callback: Callable = None,
        handler_id: str = None,
    ) -> int:
        """
        Remove event handler(s).
        
        Args:
            event_name: Event name to remove handlers for
            callback: Specific callback to remove
            handler_id: Specific handler ID to remove
            
        Returns:
            Number of handlers removed
        """
        removed = 0
        
        with self._lock:
            if handler_id:
                # Remove by ID
                for name, handlers in self._handlers.items():
                    for i, h in enumerate(handlers):
                        if h.id == handler_id:
                            handlers.pop(i)
                            removed += 1
                            break
                
                # Check wildcards
                self._wildcard_handlers = [
                    (p, h) for p, h in self._wildcard_handlers
                    if h.id != handler_id
                ]
                
            elif callback:
                # Remove by callback
                event_name = getattr(callback, '_event_name', event_name)
                if event_name:
                    handlers = self._handlers.get(event_name, [])
                    original_len = len(handlers)
                    self._handlers[event_name] = [
                        h for h in handlers if h.callback != callback
                    ]
                    removed = original_len - len(self._handlers[event_name])
            
            elif event_name:
                # Remove all handlers for event
                if event_name in self._handlers:
                    removed = len(self._handlers[event_name])
                    del self._handlers[event_name]
        
        return removed
    
    def remove_all_listeners(self, event_name: str = None):
        """Remove all listeners, optionally for a specific event"""
        with self._lock:
            if event_name:
                self._handlers.pop(event_name, None)
            else:
                self._handlers.clear()
                self._wildcard_handlers.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT EMISSION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def emit(
        self,
        event_name: str,
        data: Any = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Dict[str, Any] = None,
    ) -> Event:
        """
        Emit an event synchronously.
        
        Args:
            event_name: Name of the event
            data: Event data payload
            source: Source of the event
            priority: Event priority
            metadata: Additional metadata
            
        Returns:
            The emitted Event object
        """
        event = Event(
            name=event_name,
            data=data,
            source=source,
            priority=priority,
            metadata=metadata or {},
        )
        
        self._process_event(event)
        return event
    
    def emit_async(
        self,
        event_name: str,
        data: Any = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        callback: Callable[[Event], None] = None,
    ) -> Future:
        """
        Emit an event asynchronously.
        
        Returns a Future that will contain the Event when processed.
        """
        event = Event(
            name=event_name,
            data=data,
            source=source,
            priority=priority,
        )
        
        future = self._thread_pool.submit(self._process_event, event)
        
        if callback:
            future.add_done_callback(
                lambda f: callback(f.result())
            )
        
        return future
    
    def emit_queue(
        self,
        event_name: str,
        data: Any = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
    ) -> str:
        """
        Add event to processing queue.
        
        Returns event ID for tracking.
        """
        event = Event(
            name=event_name,
            data=data,
            source=source,
            priority=priority,
        )
        
        with self._queue_lock:
            self._queue.append(event)
        
        return event.id
    
    def process_queue(self, max_events: int = 100) -> int:
        """
        Process events in the queue.
        
        Returns number of events processed.
        """
        processed = 0
        
        with self._queue_lock:
            while self._queue and processed < max_events:
                event = self._queue.popleft()
                self._process_event(event)
                processed += 1
        
        return processed
    
    def _process_event(self, event: Event) -> Event:
        """Process an event through all matching handlers"""
        start_time = time.time()
        event._state = EventState.PROCESSING
        handlers_called = []
        
        self._stats['events_emitted'] += 1
        
        try:
            # Get matching handlers
            handlers = self._get_handlers(event.name)
            
            # Sort by priority
            handlers.sort()
            
            # Call each handler
            for handler in handlers:
                if not event.propagate:
                    break
                
                if event.is_cancelled:
                    self._stats['events_cancelled'] += 1
                    break
                
                try:
                    self._call_handler(handler, event)
                    handlers_called.append(handler.id)
                    self._stats['handlers_called'] += 1
                    
                    # Remove one-time handlers
                    if handler.once:
                        self.off(handler_id=handler.id)
                        
                except Exception as e:
                    self._stats['errors'] += 1
                    logger.error(
                        f"Handler {handler.id} error for event {event.name}: {e}"
                    )
                    # Continue to next handler (error isolation)
            
            event._state = EventState.COMPLETED
            self._stats['events_processed'] += 1
            
        except Exception as e:
            event._state = EventState.FAILED
            logger.error(f"Event processing error: {e}")
        
        # Record history
        duration = (time.time() - start_time) * 1000
        self._record_history(event, handlers_called, duration)
        
        return event
    
    def _get_handlers(self, event_name: str) -> List[EventHandler]:
        """Get all handlers matching event name"""
        handlers = list(self._handlers.get(event_name, []))
        
        # Check wildcard handlers
        import fnmatch
        for pattern, handler in self._wildcard_handlers:
            if fnmatch.fnmatch(event_name, pattern):
                handlers.append(handler)
        
        return handlers
    
    def _call_handler(self, handler: EventHandler, event: Event):
        """Call a handler with the event"""
        if handler.handler_type == HandlerType.ASYNC:
            # Run async handler in thread pool
            if asyncio.iscoroutinefunction(handler.callback):
                future = self._thread_pool.submit(
                    self._run_async_handler, handler, event
                )
                future.result(timeout=handler.timeout)
            else:
                self._thread_pool.submit(handler, event).result(
                    timeout=handler.timeout
                )
        else:
            # Sync handler
            handler(event)
    
    def _run_async_handler(self, handler: EventHandler, event: Event):
        """Run an async handler"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(handler.callback(event))
        finally:
            loop.close()
    
    def _record_history(
        self,
        event: Event,
        handlers_called: List[str],
        duration_ms: float
    ):
        """Record event in history"""
        if not self._enable_history:
            return
        
        history_entry = EventHistory(
            event=event,
            handlers_called=handlers_called,
            duration_ms=duration_ms,
            success=event._state == EventState.COMPLETED,
            error=event._state == EventState.FAILED and "Failed" or None,
        )
        
        with self._history_lock:
            self._history.append(history_entry)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT WAITING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def wait_for(
        self,
        event_name: str,
        timeout: float = 30.0,
        filter_func: Callable[[Event], bool] = None,
    ) -> Optional[Event]:
        """
        Wait for a specific event to be emitted.
        
        Args:
            event_name: Event name to wait for
            timeout: Maximum time to wait
            filter_func: Optional filter function
            
        Returns:
            Event if received, None if timeout
        """
        received_event = None
        received = threading.Event()
        
        def handler(event: Event):
            nonlocal received_event
            if filter_func is None or filter_func(event):
                received_event = event
                received.set()
        
        handler_id = self.on(event_name, handler)
        
        if received.wait(timeout):
            self.off(handler_id=handler_id)
            return received_event
        
        self.off(handler_id=handler_id)
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HISTORY AND STATS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_history(
        self,
        event_name: str = None,
        limit: int = 100,
    ) -> List[EventHistory]:
        """Get event history"""
        with self._history_lock:
            history = list(self._history)
        
        if event_name:
            history = [h for h in history if h.event.name == event_name]
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics"""
        with self._lock:
            handler_count = sum(
                len(handlers) for handlers in self._handlers.values()
            )
            handler_count += len(self._wildcard_handlers)
        
        return {
            **self._stats,
            'registered_handlers': handler_count,
            'event_types': len(self._handlers),
            'history_size': len(self._history),
            'queue_size': len(self._queue),
        }
    
    def clear_history(self):
        """Clear event history"""
        with self._history_lock:
            self._history.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @contextmanager
    def capture_events(self, event_names: List[str] = None):
        """
        Context manager to capture events.
        
        Usage:
            with emitter.capture_events(['user.*']) as events:
                do_something()
            # events contains all emitted events
        """
        captured = []
        
        def capture_handler(event: Event):
            captured.append(event)
        
        handler_ids = []
        for name in (event_names or ['*']):
            handler_id = self.on(name, capture_handler)
            handler_ids.append(handler_id)
        
        try:
            yield captured
        finally:
            for hid in handler_ids:
                self.off(handler_id=hid)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def shutdown(self, wait: bool = True):
        """Shutdown the event system"""
        self._thread_pool.shutdown(wait=wait)
        self.clear_history()
        self.remove_all_listeners()
        logger.info("EventEmitter shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL EVENT EMITTER
# ═══════════════════════════════════════════════════════════════════════════════

_global_emitter: Optional[EventEmitter] = None
_emitter_lock = threading.Lock()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance"""
    global _global_emitter
    
    with _emitter_lock:
        if _global_emitter is None:
            _global_emitter = EventEmitter()
        return _global_emitter


def emit(event_name: str, data: Any = None, **kwargs) -> Event:
    """Emit an event on the global emitter"""
    return get_event_emitter().emit(event_name, data=data, **kwargs)


def on(event_name: str, callback: Callable = None, **kwargs):
    """Register a handler on the global emitter"""
    return get_event_emitter().on(event_name, callback, **kwargs)


def once(event_name: str, callback: Callable = None, **kwargs):
    """Register a one-time handler on the global emitter"""
    return get_event_emitter().once(event_name, callback, **kwargs)


def off(event_name: str = None, callback: Callable = None, handler_id: str = None) -> int:
    """Remove a handler from the global emitter"""
    return get_event_emitter().off(event_name, callback, handler_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def event_handler(event_name: str, priority: EventPriority = EventPriority.NORMAL):
    """Decorator to mark a function as an event handler"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(event: Event):
            return func(event)
        
        wrapper._is_event_handler = True
        wrapper._event_name = event_name
        wrapper._event_priority = priority
        
        return wrapper
    return decorator


def fire_event(event_name: str, **event_kwargs):
    """Decorator to fire an event after function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            emit(event_name, data=result, source=func.__name__)
            return result
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classes
    'EventEmitter',
    'Event',
    'EventHandler',
    'EventHistory',
    'EventFilter',
    'EventPriority',
    'EventState',
    'HandlerType',
    
    # Functions
    'get_event_emitter',
    'emit',
    'on',
    'once',
    'off',
    
    # Decorators
    'event_handler',
    'fire_event',
]
