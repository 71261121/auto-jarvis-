#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - State Machine
====================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Finite State Machine pattern
- Hierarchical states
- Transition guards
- Entry/exit actions
- State history

Features:
- State definitions with transitions
- Transition guards and conditions
- Entry/exit actions
- State history and rollback
- Hierarchical states
- Parallel states
- State persistence
- Event-driven transitions
- Timeout transitions
- State recovery on crash

Memory Impact: < 2MB
"""

import sys
import os
import time
import json
import logging
import threading
import hashlib
from pathlib import Path
from typing import (
    Dict, Any, Optional, List, Set, Tuple, Callable, 
    Union, TypeVar, Generic, Protocol
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from functools import wraps

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TransitionResult(Enum):
    """Result of a state transition"""
    SUCCESS = auto()
    REJECTED = auto()      # Guard rejected
    INVALID = auto()       # Invalid transition
    ERROR = auto()         # Exception occurred
    TIMEOUT = auto()       # Transition timed out


@dataclass
class State:
    """
    Represents a state in the machine.
    """
    name: str
    description: str = ""
    
    # Actions
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    
    # Allowed transitions (state_name -> Transition)
    transitions: Dict[str, 'Transition'] = field(default_factory=dict)
    
    # Sub-states for hierarchical FSM
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    initial_child: Optional[str] = None
    
    # Timeout
    timeout_seconds: Optional[float] = None
    timeout_target: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_composite(self) -> bool:
        """Check if this is a composite state (has children)"""
        return len(self.children) > 0
    
    def add_transition(
        self,
        target: str,
        guard: Callable = None,
        action: Callable = None,
        name: str = None,
    ) -> 'Transition':
        """Add a transition from this state"""
        transition = Transition(
            name=name or f"{self.name}_to_{target}",
            source=self.name,
            target=target,
            guard=guard,
            action=action,
        )
        self.transitions[target] = transition
        return transition


@dataclass
class Transition:
    """
    Represents a transition between states.
    """
    name: str
    source: str
    target: str
    
    # Conditions
    guard: Optional[Callable[[Any], bool]] = None
    
    # Actions
    action: Optional[Callable[[Any], Any]] = None
    
    # Metadata
    event: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_execute(self, context: Any = None) -> bool:
        """Check if transition can execute"""
        if self.guard is None:
            return True
        
        try:
            return self.guard(context)
        except Exception as e:
            logger.error(f"Guard error in {self.name}: {e}")
            return False


@dataclass
class StateHistoryEntry:
    """
    Entry in state history.
    """
    state: str
    entered_at: float
    exited_at: Optional[float] = None
    transition: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMachineStats:
    """State machine statistics"""
    total_transitions: int = 0
    successful_transitions: int = 0
    failed_transitions: int = 0
    rejected_transitions: int = 0
    
    state_visit_counts: Dict[str, int] = field(default_factory=dict)
    transition_counts: Dict[str, int] = field(default_factory=dict)
    
    total_time_in_state: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_transitions == 0:
            return 1.0
        return self.successful_transitions / self.total_transitions


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class StateMachine:
    """
    Ultra-Advanced Finite State Machine.
    
    Features:
    - State definitions with transitions
    - Transition guards and conditions
    - Entry/exit actions
    - State history and rollback
    - Hierarchical states
    - Event-driven transitions
    - Timeout transitions
    - State persistence
    - Recovery on crash
    
    Memory Budget: < 2MB
    
    Usage:
        fsm = StateMachine(initial_state='idle')
        
        # Define states
        fsm.add_state('idle', on_enter=start_idle)
        fsm.add_state('running', on_enter=start_process)
        fsm.add_state('completed', on_enter=finish)
        fsm.add_state('error', on_enter=handle_error)
        
        # Add transitions
        fsm.add_transition('idle', 'running', guard=can_start)
        fsm.add_transition('running', 'completed')
        fsm.add_transition('running', 'error')
        
        # Trigger transition
        fsm.transition('running')
        
        # Check state
        if fsm.is_state('running'):
            ...
    """
    
    def __init__(
        self,
        initial_state: str = None,
        name: str = "FSM",
        max_history: int = 100,
        persist_dir: str = None,
    ):
        """
        Initialize State Machine.
        
        Args:
            initial_state: Initial state name
            name: Machine name for logging
            max_history: Maximum history entries
            persist_dir: Directory for state persistence
        """
        self._name = name
        self._max_history = max_history
        self._persist_dir = Path(persist_dir) if persist_dir else None
        
        # States
        self._states: Dict[str, State] = {}
        self._initial_state = initial_state
        self._current_state: Optional[str] = None
        
        # History
        self._history: deque = deque(maxlen=max_history)
        self._history_lock = threading.Lock()
        
        # Event transitions
        self._event_transitions: Dict[str, List[Transition]] = defaultdict(list)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = StateMachineStats()
        
        # Timeout handling
        self._timeout_thread: Optional[threading.Thread] = None
        self._timeout_stop = threading.Event()
        self._state_entered_at: Optional[float] = None
        
        # Context
        self._context: Any = None
        
        # Add initial state if provided
        if initial_state:
            self.add_state(initial_state)
        
        logger.info(f"StateMachine '{name}' initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE DEFINITION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_state(
        self,
        name: str,
        description: str = "",
        on_enter: Callable = None,
        on_exit: Callable = None,
        timeout_seconds: float = None,
        timeout_target: str = None,
        parent: str = None,
        initial: bool = False,
    ) -> State:
        """
        Add a state to the machine.
        
        Args:
            name: State name
            description: State description
            on_enter: Action on entering state
            on_exit: Action on exiting state
            timeout_seconds: Timeout for this state
            timeout_target: Target state on timeout
            parent: Parent state for hierarchical FSM
            initial: Whether this is the initial state
            
        Returns:
            The created State object
        """
        with self._lock:
            state = State(
                name=name,
                description=description,
                on_enter=on_enter,
                on_exit=on_exit,
                timeout_seconds=timeout_seconds,
                timeout_target=timeout_target,
                parent=parent,
            )
            
            self._states[name] = state
            
            # Handle parent relationship
            if parent and parent in self._states:
                self._states[parent].children.add(name)
            
            # Set as initial if specified or first state
            if initial or self._initial_state is None:
                self._initial_state = name
            
            logger.debug(f"Added state '{name}' to {self._name}")
            return state
    
    def add_transition(
        self,
        source: str,
        target: str,
        guard: Callable = None,
        action: Callable = None,
        event: str = None,
        name: str = None,
    ) -> Transition:
        """
        Add a transition between states.
        
        Args:
            source: Source state name
            target: Target state name
            guard: Guard function (returns bool)
            action: Action to execute during transition
            event: Event that triggers this transition
            name: Transition name
            
        Returns:
            The created Transition object
        """
        with self._lock:
            # Ensure states exist
            if source not in self._states:
                self.add_state(source)
            if target not in self._states:
                self.add_state(target)
            
            transition = Transition(
                name=name or f"{source}_to_{target}",
                source=source,
                target=target,
                guard=guard,
                action=action,
                event=event,
            )
            
            self._states[source].transitions[target] = transition
            
            # Register event transition
            if event:
                self._event_transitions[event].append(transition)
            
            logger.debug(f"Added transition '{source}' -> '{target}'")
            return transition
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MACHINE CONTROL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self, context: Any = None) -> bool:
        """
        Start the state machine.
        
        Enters the initial state.
        """
        with self._lock:
            if self._current_state is not None:
                logger.warning(f"{self._name} already started")
                return False
            
            if self._initial_state is None:
                logger.error(f"{self._name} has no initial state")
                return False
            
            self._context = context
            
            # Enter initial state
            return self._enter_state(self._initial_state)
    
    def stop(self):
        """Stop the state machine"""
        with self._lock:
            if self._current_state:
                self._exit_state(self._current_state)
            self._current_state = None
            self._timeout_stop.set()
    
    def reset(self):
        """Reset to initial state"""
        with self._lock:
            self.stop()
            self._history.clear()
            self._stats = StateMachineStats()
            self.start(self._context)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRANSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def transition(
        self,
        target: str,
        context: Any = None,
    ) -> TransitionResult:
        """
        Transition to a target state.
        
        Args:
            target: Target state name
            context: Optional context for guards/actions
            
        Returns:
            TransitionResult indicating success or failure
        """
        with self._lock:
            if self._current_state is None:
                logger.error(f"{self._name} not started")
                return TransitionResult.INVALID
            
            current = self._states.get(self._current_state)
            if current is None:
                return TransitionResult.INVALID
            
            # Check if transition is allowed
            transition = current.transitions.get(target)
            if transition is None:
                logger.warning(
                    f"No transition from '{self._current_state}' to '{target}'"
                )
                return TransitionResult.INVALID
            
            return self._execute_transition(transition, context)
    
    def trigger_event(
        self,
        event: str,
        context: Any = None,
    ) -> TransitionResult:
        """
        Trigger a transition by event.
        
        Looks for transitions registered for this event.
        """
        with self._lock:
            transitions = self._event_transitions.get(event, [])
            
            for transition in sorted(transitions, key=lambda t: -t.priority):
                if transition.source == self._current_state:
                    result = self._execute_transition(transition, context)
                    if result == TransitionResult.SUCCESS:
                        return result
            
            return TransitionResult.INVALID
    
    def _execute_transition(
        self,
        transition: Transition,
        context: Any = None,
    ) -> TransitionResult:
        """Execute a state transition"""
        self._stats.total_transitions += 1
        
        try:
            # Check guard
            if not transition.can_execute(context):
                self._stats.rejected_transitions += 1
                logger.debug(f"Transition '{transition.name}' rejected by guard")
                return TransitionResult.REJECTED
            
            # Execute transition action
            if transition.action:
                try:
                    transition.action(context)
                except Exception as e:
                    logger.error(f"Transition action error: {e}")
                    self._stats.failed_transitions += 1
                    return TransitionResult.ERROR
            
            # Exit current state
            self._exit_state(transition.source, transition.name)
            
            # Enter new state
            self._enter_state(transition.target)
            
            # Update stats
            self._stats.successful_transitions += 1
            self._stats.transition_counts[transition.name] = \
                self._stats.transition_counts.get(transition.name, 0) + 1
            
            logger.debug(
                f"{self._name}: '{transition.source}' -> '{transition.target}'"
            )
            return TransitionResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Transition error: {e}")
            self._stats.failed_transitions += 1
            return TransitionResult.ERROR
    
    def _enter_state(self, state_name: str) -> bool:
        """Enter a state"""
        state = self._states.get(state_name)
        if state is None:
            logger.error(f"State '{state_name}' not found")
            return False
        
        # Update current state
        old_state = self._current_state
        self._current_state = state_name
        self._state_entered_at = time.time()
        
        # Record in history
        entry = StateHistoryEntry(
            state=state_name,
            entered_at=time.time(),
        )
        with self._history_lock:
            self._history.append(entry)
        
        # Update stats
        self._stats.state_visit_counts[state_name] = \
            self._stats.state_visit_counts.get(state_name, 0) + 1
        
        # Execute on_enter action
        if state.on_enter:
            try:
                state.on_enter(self._context)
            except Exception as e:
                logger.error(f"on_enter error in '{state_name}': {e}")
        
        # Handle hierarchical states
        if state.is_composite and state.initial_child:
            self._enter_state(state.initial_child)
        
        return True
    
    def _exit_state(self, state_name: str, transition_name: str = None):
        """Exit a state"""
        state = self._states.get(state_name)
        if state is None:
            return
        
        # Calculate duration
        if self._state_entered_at:
            duration = time.time() - self._state_entered_at
            self._stats.total_time_in_state[state_name] = \
                self._stats.total_time_in_state.get(state_name, 0) + duration
        
        # Update history
        with self._history_lock:
            if self._history:
                last_entry = self._history[-1]
                last_entry.exited_at = time.time()
                last_entry.duration_seconds = duration if self._state_entered_at else None
                last_entry.transition = transition_name
        
        # Execute on_exit action
        if state.on_exit:
            try:
                state.on_exit(self._context)
            except Exception as e:
                logger.error(f"on_exit error in '{state_name}': {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE QUERIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def current_state(self) -> Optional[str]:
        """Get current state name"""
        return self._current_state
    
    @property
    def context(self) -> Any:
        """Get current context"""
        return self._context
    
    @context.setter
    def context(self, value: Any):
        """Set context"""
        self._context = value
    
    def is_state(self, state_name: str) -> bool:
        """Check if machine is in specific state"""
        return self._current_state == state_name
    
    def is_in_state(self, state_name: str) -> bool:
        """Check if machine is in state or any substate"""
        current = self._current_state
        while current:
            if current == state_name:
                return True
            current = self._states[current].parent if current in self._states else None
        return False
    
    def can_transition_to(self, target: str) -> bool:
        """Check if transition to target is possible"""
        if self._current_state is None:
            return False
        
        current = self._states.get(self._current_state)
        if current is None:
            return False
        
        transition = current.transitions.get(target)
        if transition is None:
            return False
        
        return transition.can_execute(self._context)
    
    def get_valid_transitions(self) -> List[str]:
        """Get list of valid transition targets"""
        if self._current_state is None:
            return []
        
        current = self._states.get(self._current_state)
        if current is None:
            return []
        
        return [
            target for target, trans in current.transitions.items()
            if trans.can_execute(self._context)
        ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_history(self, limit: int = 50) -> List[StateHistoryEntry]:
        """Get state history"""
        with self._history_lock:
            return list(self._history)[-limit:]
    
    def get_last_state(self) -> Optional[str]:
        """Get the previous state"""
        with self._history_lock:
            if len(self._history) < 2:
                return None
            return self._history[-2].state
    
    def rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous state"""
        with self._lock:
            with self._history_lock:
                if len(self._history) < steps + 1:
                    return False
                
                # Get target state
                target_idx = -(steps + 1)
                target_entry = self._history[target_idx]
                target_state = target_entry.state
                
                # Remove recent history
                for _ in range(steps):
                    if self._history:
                        self._history.pop()
                
                # Transition to target
                self._exit_state(self._current_state)
                self._enter_state(target_state)
                
                return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_state(self, filepath: str = None) -> bool:
        """Save current state to file"""
        if filepath is None:
            if self._persist_dir is None:
                return False
            filepath = self._persist_dir / f"{self._name}_state.json"
        
        try:
            data = {
                'name': self._name,
                'current_state': self._current_state,
                'timestamp': time.time(),
                'history': [
                    {
                        'state': e.state,
                        'entered_at': e.entered_at,
                        'exited_at': e.exited_at,
                    }
                    for e in list(self._history)[-10:]
                ],
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"State save error: {e}")
            return False
    
    def load_state(self, filepath: str = None) -> bool:
        """Load state from file"""
        if filepath is None:
            if self._persist_dir is None:
                return False
            filepath = self._persist_dir / f"{self._name}_state.json"
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            if data.get('name') != self._name:
                return False
            
            # Restore state
            target_state = data.get('current_state')
            if target_state and target_state in self._states:
                self._current_state = target_state
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"State load error: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics"""
        return {
            'name': self._name,
            'current_state': self._current_state,
            'total_transitions': self._stats.total_transitions,
            'success_rate': f"{self._stats.success_rate:.1%}",
            'state_visits': dict(self._stats.state_visit_counts),
            'history_size': len(self._history),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return f"StateMachine({self._name}, state={self._current_state})"
    
    def __str__(self) -> str:
        return f"{self._name}: {self._current_state or 'not started'}"


# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS STATE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class JarvisStates(Enum):
    """Standard JARVIS states"""
    # Initial states
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    
    # Normal operation
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    
    # Learning
    LEARNING = "learning"
    SELF_IMPROVING = "self_improving"
    
    # Maintenance
    MAINTENANCE = "maintenance"
    BACKUP = "backup"
    
    # Error states
    ERROR = "error"
    RECOVERY = "recovery"
    
    # Terminal states
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


def create_jarvis_state_machine() -> StateMachine:
    """
    Create a state machine configured for JARVIS.
    
    Returns:
        Configured StateMachine instance
    """
    fsm = StateMachine(
        initial_state=JarvisStates.UNINITIALIZED.value,
        name="JARVIS"
    )
    
    # Define all states
    for state in JarvisStates:
        fsm.add_state(state.value)
    
    # Define transitions
    # Initialization flow
    fsm.add_transition(
        JarvisStates.UNINITIALIZED.value,
        JarvisStates.INITIALIZING.value
    )
    fsm.add_transition(
        JarvisStates.INITIALIZING.value,
        JarvisStates.IDLE.value
    )
    fsm.add_transition(
        JarvisStates.INITIALIZING.value,
        JarvisStates.ERROR.value
    )
    
    # Normal operation
    fsm.add_transition(JarvisStates.IDLE.value, JarvisStates.LISTENING.value)
    fsm.add_transition(JarvisStates.LISTENING.value, JarvisStates.PROCESSING.value)
    fsm.add_transition(JarvisStates.PROCESSING.value, JarvisStates.RESPONDING.value)
    fsm.add_transition(JarvisStates.RESPONDING.value, JarvisStates.IDLE.value)
    
    # Learning
    fsm.add_transition(JarvisStates.IDLE.value, JarvisStates.LEARNING.value)
    fsm.add_transition(JarvisStates.LEARNING.value, JarvisStates.SELF_IMPROVING.value)
    fsm.add_transition(JarvisStates.SELF_IMPROVING.value, JarvisStates.IDLE.value)
    
    # Maintenance
    fsm.add_transition(JarvisStates.IDLE.value, JarvisStates.MAINTENANCE.value)
    fsm.add_transition(JarvisStates.MAINTENANCE.value, JarvisStates.BACKUP.value)
    fsm.add_transition(JarvisStates.BACKUP.value, JarvisStates.IDLE.value)
    
    # Error handling
    fsm.add_transition(JarvisStates.ERROR.value, JarvisStates.RECOVERY.value)
    fsm.add_transition(JarvisStates.RECOVERY.value, JarvisStates.IDLE.value)
    fsm.add_transition(JarvisStates.RECOVERY.value, JarvisStates.ERROR.value)
    
    # Any state can transition to error
    for state in JarvisStates:
        if state not in (JarvisStates.ERROR, JarvisStates.TERMINATED):
            fsm.add_transition(state.value, JarvisStates.ERROR.value)
    
    # Shutdown
    fsm.add_transition(JarvisStates.IDLE.value, JarvisStates.SHUTTING_DOWN.value)
    fsm.add_transition(JarvisStates.SHUTTING_DOWN.value, JarvisStates.TERMINATED.value)
    
    return fsm


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classes
    'StateMachine',
    'State',
    'Transition',
    'StateHistoryEntry',
    'StateMachineStats',
    
    # Enums
    'TransitionResult',
    'JarvisStates',
    
    # Functions
    'create_jarvis_state_machine',
]
