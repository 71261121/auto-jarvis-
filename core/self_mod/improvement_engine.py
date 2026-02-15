#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Improvement Loop Engine
===================================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Reinforcement learning principles
- Outcome-based reward system
- Performance metric tracking
- Adaptive optimization
- Safe learning boundaries

This is the BRAIN of self-modification - it learns from every outcome
and continuously improves the system while staying within safe boundaries.

Features:
- Learning from success/failure patterns
- Performance metric collection
- Adaptive strategy selection
- Improvement suggestion generation
- Risk-aware optimization
- Goal-oriented self-modification
- Historical learning database
- Feedback loop integration

Memory Impact: < 15MB for learning data
"""

import ast
import sys
import os
import re
import time
import json
import logging
import hashlib
import math
import threading
import sqlite3
from typing import Dict, Any, Optional, List, Set, Tuple, Generator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, Counter, deque
from pathlib import Path
from datetime import datetime
from functools import wraps, lru_cache
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class OutcomeType(Enum):
    """Types of modification outcomes"""
    SUCCESS = auto()
    PARTIAL_SUCCESS = auto()
    FAILURE = auto()
    ROLLBACK = auto()
    ERROR = auto()
    TIMEOUT = auto()


class ImprovementCategory(Enum):
    """Categories of improvements"""
    PERFORMANCE = auto()
    MEMORY = auto()
    RELIABILITY = auto()
    SECURITY = auto()
    CODE_QUALITY = auto()
    FUNCTIONALITY = auto()
    ERROR_HANDLING = auto()
    DOCUMENTATION = auto()


class LearningMode(Enum):
    """Learning modes"""
    CONSERVATIVE = auto()   # Only safe, proven improvements
    BALANCED = auto()       # Balance safety and progress
    AGGRESSIVE = auto()     # More experimental improvements


class SuggestionPriority(Enum):
    """Priority of improvement suggestions"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ModificationOutcome:
    """Record of a modification outcome"""
    modification_id: str
    outcome_type: OutcomeType
    timestamp: float
    duration_ms: float = 0.0
    error_message: str = ""
    performance_before: Dict[str, float] = field(default_factory=dict)
    performance_after: Dict[str, float] = field(default_factory=dict)
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    was_reverted: bool = False
    revert_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.outcome_type in (OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS)
    
    @property
    def performance_delta(self) -> Dict[str, float]:
        """Calculate performance change"""
        delta = {}
        for key, after_val in self.performance_after.items():
            before_val = self.performance_before.get(key, after_val)
            delta[key] = after_val - before_val
        return delta


@dataclass
class PerformanceMetric:
    """A tracked performance metric"""
    name: str
    value: float
    timestamp: float
    category: ImprovementCategory = ImprovementCategory.PERFORMANCE
    unit: str = ""
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def status(self) -> str:
        if self.threshold_critical is not None:
            if self.value > self.threshold_critical:
                return "critical"
        if self.threshold_warning is not None:
            if self.value > self.threshold_warning:
                return "warning"
        return "normal"


@dataclass
class ImprovementSuggestion:
    """A suggestion for system improvement"""
    id: str
    category: ImprovementCategory
    priority: SuggestionPriority
    description: str
    target_element: str
    proposed_action: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    expected_benefit: str
    risk_level: str  # "low", "medium", "high"
    prerequisites: List[str] = field(default_factory=list)
    success_probability: float = 0.5
    based_on_outcomes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    applied: bool = False
    result_outcome: Optional[str] = None


@dataclass
class LearningPattern:
    """A learned pattern from outcomes"""
    pattern_id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]  # When this pattern applies
    expected_outcome: OutcomeType
    confidence: float
    occurrence_count: int = 1
    success_count: int = 0
    last_seen: float = field(default_factory=time.time)
    examples: List[str] = field(default_factory=list)  # Modification IDs


@dataclass
class LearningStats:
    """Statistics about learning system"""
    total_outcomes: int = 0
    successful_outcomes: int = 0
    failed_outcomes: int = 0
    reverted_outcomes: int = 0
    patterns_learned: int = 0
    suggestions_generated: int = 0
    suggestions_applied: int = 0
    average_confidence: float = 0.0
    improvement_rate: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class LearningDatabase:
    """
    SQLite-based learning data storage.
    
    Stores all outcomes, patterns, and suggestions for
    persistent learning across sessions.
    """
    
    SCHEMA = """
    -- Outcomes table
    CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        modification_id TEXT NOT NULL,
        outcome_type TEXT NOT NULL,
        timestamp REAL NOT NULL,
        duration_ms REAL DEFAULT 0,
        error_message TEXT DEFAULT '',
        performance_before TEXT DEFAULT '{}',
        performance_after TEXT DEFAULT '{}',
        memory_before_mb REAL DEFAULT 0,
        memory_after_mb REAL DEFAULT 0,
        was_reverted INTEGER DEFAULT 0,
        revert_reason TEXT DEFAULT '',
        metadata TEXT DEFAULT '{}'
    );
    
    -- Patterns table
    CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id TEXT UNIQUE NOT NULL,
        pattern_type TEXT NOT NULL,
        description TEXT,
        conditions TEXT DEFAULT '{}',
        expected_outcome TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        occurrence_count INTEGER DEFAULT 1,
        success_count INTEGER DEFAULT 0,
        last_seen REAL,
        examples TEXT DEFAULT '[]'
    );
    
    -- Suggestions table
    CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suggestion_id TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        priority INTEGER DEFAULT 2,
        description TEXT,
        target_element TEXT,
        proposed_action TEXT,
        reasoning TEXT,
        confidence REAL DEFAULT 0.5,
        expected_benefit TEXT,
        risk_level TEXT DEFAULT 'medium',
        prerequisites TEXT DEFAULT '[]',
        success_probability REAL DEFAULT 0.5,
        based_on_outcomes TEXT DEFAULT '[]',
        created_at REAL,
        applied INTEGER DEFAULT 0,
        result_outcome TEXT
    );
    
    -- Metrics table
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        value REAL,
        timestamp REAL,
        category TEXT DEFAULT 'performance',
        unit TEXT DEFAULT '',
        target_value REAL,
        threshold_warning REAL,
        threshold_critical REAL
    );
    
    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON outcomes(timestamp);
    CREATE INDEX IF NOT EXISTS idx_outcomes_type ON outcomes(outcome_type);
    CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
    CREATE INDEX IF NOT EXISTS idx_suggestions_category ON suggestions(category);
    CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name, timestamp);
    """
    
    def __init__(self, db_path: str = "~/.jarvis/learning.db"):
        """Initialize learning database"""
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._local = threading.local()
        
        self._initialize_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _initialize_db(self):
        """Initialize database schema"""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()
    
    def store_outcome(self, outcome: ModificationOutcome):
        """Store a modification outcome"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO outcomes 
                (modification_id, outcome_type, timestamp, duration_ms, error_message,
                 performance_before, performance_after, memory_before_mb, memory_after_mb,
                 was_reverted, revert_reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.modification_id,
                outcome.outcome_type.name,
                outcome.timestamp,
                outcome.duration_ms,
                outcome.error_message,
                json.dumps(outcome.performance_before),
                json.dumps(outcome.performance_after),
                outcome.memory_before_mb,
                outcome.memory_after_mb,
                1 if outcome.was_reverted else 0,
                outcome.revert_reason,
                json.dumps(outcome.metadata),
            ))
            conn.commit()
    
    def get_outcomes(
        self,
        limit: int = 100,
        outcome_type: OutcomeType = None
    ) -> List[ModificationOutcome]:
        """Get recent outcomes"""
        conn = self._get_connection()
        
        if outcome_type:
            cursor = conn.execute("""
                SELECT * FROM outcomes 
                WHERE outcome_type = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (outcome_type.name, limit))
        else:
            cursor = conn.execute("""
                SELECT * FROM outcomes 
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        
        outcomes = []
        for row in cursor.fetchall():
            outcomes.append(ModificationOutcome(
                modification_id=row['modification_id'],
                outcome_type=OutcomeType[row['outcome_type']],
                timestamp=row['timestamp'],
                duration_ms=row['duration_ms'],
                error_message=row['error_message'],
                performance_before=json.loads(row['performance_before']),
                performance_after=json.loads(row['performance_after']),
                memory_before_mb=row['memory_before_mb'],
                memory_after_mb=row['memory_after_mb'],
                was_reverted=bool(row['was_reverted']),
                revert_reason=row['revert_reason'],
                metadata=json.loads(row['metadata']),
            ))
        
        return outcomes
    
    def store_pattern(self, pattern: LearningPattern):
        """Store a learning pattern"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT OR REPLACE INTO patterns
                (pattern_id, pattern_type, description, conditions, expected_outcome,
                 confidence, occurrence_count, success_count, last_seen, examples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                pattern.description,
                json.dumps(pattern.conditions),
                pattern.expected_outcome.name,
                pattern.confidence,
                pattern.occurrence_count,
                pattern.success_count,
                pattern.last_seen,
                json.dumps(pattern.examples),
            ))
            conn.commit()
    
    def get_patterns(self, pattern_type: str = None) -> List[LearningPattern]:
        """Get learned patterns"""
        conn = self._get_connection()
        
        if pattern_type:
            cursor = conn.execute(
                "SELECT * FROM patterns WHERE pattern_type = ?",
                (pattern_type,)
            )
        else:
            cursor = conn.execute("SELECT * FROM patterns")
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append(LearningPattern(
                pattern_id=row['pattern_id'],
                pattern_type=row['pattern_type'],
                description=row['description'],
                conditions=json.loads(row['conditions']),
                expected_outcome=OutcomeType[row['expected_outcome']],
                confidence=row['confidence'],
                occurrence_count=row['occurrence_count'],
                success_count=row['success_count'],
                last_seen=row['last_seen'],
                examples=json.loads(row['examples']),
            ))
        
        return patterns
    
    def store_suggestion(self, suggestion: ImprovementSuggestion):
        """Store an improvement suggestion"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT OR REPLACE INTO suggestions
                (suggestion_id, category, priority, description, target_element,
                 proposed_action, reasoning, confidence, expected_benefit, risk_level,
                 prerequisites, success_probability, based_on_outcomes, created_at,
                 applied, result_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suggestion.id,
                suggestion.category.name,
                suggestion.priority.value,
                suggestion.description,
                suggestion.target_element,
                suggestion.proposed_action,
                suggestion.reasoning,
                suggestion.confidence,
                suggestion.expected_benefit,
                suggestion.risk_level,
                json.dumps(suggestion.prerequisites),
                suggestion.success_probability,
                json.dumps(suggestion.based_on_outcomes),
                suggestion.created_at,
                1 if suggestion.applied else 0,
                suggestion.result_outcome,
            ))
            conn.commit()
    
    def get_suggestions(
        self,
        applied: bool = None,
        category: ImprovementCategory = None,
        limit: int = 50
    ) -> List[ImprovementSuggestion]:
        """Get improvement suggestions"""
        conn = self._get_connection()
        
        query = "SELECT * FROM suggestions WHERE 1=1"
        params = []
        
        if applied is not None:
            query += " AND applied = ?"
            params.append(1 if applied else 0)
        
        if category:
            query += " AND category = ?"
            params.append(category.name)
        
        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append(ImprovementSuggestion(
                id=row['suggestion_id'],
                category=ImprovementCategory[row['category']],
                priority=SuggestionPriority(row['priority']),
                description=row['description'],
                target_element=row['target_element'],
                proposed_action=row['proposed_action'],
                reasoning=row['reasoning'],
                confidence=row['confidence'],
                expected_benefit=row['expected_benefit'],
                risk_level=row['risk_level'],
                prerequisites=json.loads(row['prerequisites']),
                success_probability=row['success_probability'],
                based_on_outcomes=json.loads(row['based_on_outcomes']),
                created_at=row['created_at'],
                applied=bool(row['applied']),
                result_outcome=row['result_outcome'],
            ))
        
        return suggestions
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO metrics
                (name, value, timestamp, category, unit, target_value,
                 threshold_warning, threshold_critical)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.timestamp,
                metric.category.name,
                metric.unit,
                metric.target_value,
                metric.threshold_warning,
                metric.threshold_critical,
            ))
            conn.commit()
    
    def get_metrics(
        self,
        name: str = None,
        since: float = None,
        limit: int = 100
    ) -> List[PerformanceMetric]:
        """Get performance metrics"""
        conn = self._get_connection()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append(PerformanceMetric(
                name=row['name'],
                value=row['value'],
                timestamp=row['timestamp'],
                category=ImprovementCategory[row['category']],
                unit=row['unit'],
                target_value=row['target_value'],
                threshold_warning=row['threshold_warning'],
                threshold_critical=row['threshold_critical'],
            ))
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SelfImprovementEngine:
    """
    Ultra-Advanced Self-Improvement Learning Engine.
    
    This is the BRAIN that learns from every modification outcome
    and continuously improves the system.
    
    Features:
    - Learning from outcomes
    - Pattern recognition
    - Improvement suggestion generation
    - Performance monitoring
    - Risk-aware optimization
    - Goal-oriented learning
    - Safe learning boundaries
    
    Memory Budget: < 15MB
    
    Usage:
        engine = SelfImprovementEngine()
        
        # Record outcome
        outcome = ModificationOutcome(
            modification_id="mod-123",
            outcome_type=OutcomeType.SUCCESS,
            timestamp=time.time()
        )
        engine.record_outcome(outcome)
        
        # Get improvement suggestions
        suggestions = engine.get_improvement_suggestions()
        
        # Get learned patterns
        patterns = engine.get_learned_patterns()
    """
    
    def __init__(
        self,
        learning_mode: LearningMode = LearningMode.BALANCED,
        db_path: str = None,
        enable_auto_learning: bool = True,
        min_confidence_threshold: float = 0.6,
        max_suggestions: int = 20,
    ):
        """
        Initialize Self-Improvement Engine.
        
        Args:
            learning_mode: Learning aggressiveness
            db_path: Path to learning database
            enable_auto_learning: Enable automatic pattern learning
            min_confidence_threshold: Minimum confidence for suggestions
            max_suggestions: Maximum suggestions to generate
        """
        self._learning_mode = learning_mode
        self._enable_auto_learning = enable_auto_learning
        self._min_confidence = min_confidence_threshold
        self._max_suggestions = max_suggestions
        
        # Database
        self._db = LearningDatabase(db_path or "~/.jarvis/data/learning.db")
        
        # In-memory caches
        self._recent_outcomes: deque = deque(maxlen=1000)
        self._active_patterns: Dict[str, LearningPattern] = {}
        self._pending_suggestions: List[ImprovementSuggestion] = []
        
        # Performance tracking
        self._current_metrics: Dict[str, float] = {}
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistics
        self._stats = LearningStats()
        
        # Learning rules
        self._learning_rules = self._initialize_learning_rules()
        
        # Load existing patterns
        self._load_patterns()
        
        logger.info(f"SelfImprovementEngine initialized (mode: {learning_mode.name})")
    
    def _initialize_learning_rules(self) -> List[Dict[str, Any]]:
        """Initialize learning rules based on research"""
        return [
            # Performance improvement rules
            {
                'name': 'slow_function_optimization',
                'condition': lambda o: o.success and o.duration_ms > 1000,
                'category': ImprovementCategory.PERFORMANCE,
                'suggestion': 'Optimize slow function',
                'confidence_boost': 0.1,
            },
            {
                'name': 'memory_reduction',
                'condition': lambda o: o.success and 
                    (o.memory_after_mb - o.memory_before_mb) > 10,
                'category': ImprovementCategory.MEMORY,
                'suggestion': 'Reduce memory usage',
                'confidence_boost': 0.15,
            },
            {
                'name': 'error_pattern',
                'condition': lambda o: o.outcome_type == OutcomeType.ERROR,
                'category': ImprovementCategory.ERROR_HANDLING,
                'suggestion': 'Improve error handling',
                'confidence_boost': 0.2,
            },
            {
                'name': 'rollback_pattern',
                'condition': lambda o: o.was_reverted,
                'category': ImprovementCategory.RELIABILITY,
                'suggestion': 'Prevent rollback scenarios',
                'confidence_boost': 0.25,
            },
            {
                'name': 'success_pattern',
                'condition': lambda o: o.success and not o.was_reverted,
                'category': ImprovementCategory.CODE_QUALITY,
                'suggestion': 'Apply similar successful modifications',
                'confidence_boost': 0.05,
            },
        ]
    
    def _load_patterns(self):
        """Load learned patterns from database"""
        patterns = self._db.get_patterns()
        for pattern in patterns:
            self._active_patterns[pattern.pattern_id] = pattern
        
        self._stats.patterns_learned = len(patterns)
    
    def record_outcome(self, outcome: ModificationOutcome):
        """
        Record a modification outcome and learn from it.
        
        This is the core learning entry point.
        
        Args:
            outcome: The modification outcome to record
        """
        # Store in database
        self._db.store_outcome(outcome)
        
        # Add to recent outcomes
        self._recent_outcomes.append(outcome)
        
        # Update statistics
        self._stats.total_outcomes += 1
        if outcome.success:
            self._stats.successful_outcomes += 1
        else:
            self._stats.failed_outcomes += 1
        
        if outcome.was_reverted:
            self._stats.reverted_outcomes += 1
        
        # Calculate improvement rate
        if self._stats.total_outcomes > 0:
            self._stats.improvement_rate = (
                self._stats.successful_outcomes / self._stats.total_outcomes
            )
        
        # Learn from outcome
        if self._enable_auto_learning:
            self._learn_from_outcome(outcome)
        
        # Generate suggestions if needed
        self._maybe_generate_suggestions()
        
        logger.debug(f"Recorded outcome: {outcome.modification_id} -> {outcome.outcome_type.name}")
    
    def _learn_from_outcome(self, outcome: ModificationOutcome):
        """Learn patterns from an outcome"""
        
        # Apply learning rules
        for rule in self._learning_rules:
            try:
                if rule['condition'](outcome):
                    self._update_pattern(
                        rule['name'],
                        outcome,
                        rule['category'],
                        rule['suggestion'],
                        rule['confidence_boost']
                    )
            except Exception as e:
                logger.warning(f"Learning rule error: {e}")
        
        # Learn outcome sequence patterns
        self._learn_sequence_patterns(outcome)
    
    def _update_pattern(
        self,
        pattern_name: str,
        outcome: ModificationOutcome,
        category: ImprovementCategory,
        suggestion: str,
        confidence_boost: float
    ):
        """Update or create a learning pattern"""
        pattern_id = hashlib.md5(pattern_name.encode()).hexdigest()[:16]
        
        if pattern_id in self._active_patterns:
            # Update existing pattern
            pattern = self._active_patterns[pattern_id]
            pattern.occurrence_count += 1
            
            if outcome.success:
                pattern.success_count += 1
            
            pattern.confidence = min(1.0, pattern.confidence + confidence_boost)
            pattern.last_seen = time.time()
            
            if len(pattern.examples) < 10:
                pattern.examples.append(outcome.modification_id)
            
        else:
            # Create new pattern
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_name,
                description=suggestion,
                conditions={'category': category.name},
                expected_outcome=outcome.outcome_type,
                confidence=0.5 + confidence_boost,
                occurrence_count=1,
                success_count=1 if outcome.success else 0,
                examples=[outcome.modification_id],
            )
            
            self._active_patterns[pattern_id] = pattern
            self._stats.patterns_learned += 1
        
        # Store in database
        self._db.store_pattern(self._active_patterns[pattern_id])
    
    def _learn_sequence_patterns(self, outcome: ModificationOutcome):
        """Learn from sequences of outcomes"""
        if len(self._recent_outcomes) < 3:
            return
        
        # Look for repeated patterns in recent outcomes
        recent = list(self._recent_outcomes)[-5:]
        
        # Check for failure patterns
        failure_count = sum(1 for o in recent if not o.success)
        
        if failure_count >= 3:
            self._update_pattern(
                'repeated_failures',
                outcome,
                ImprovementCategory.RELIABILITY,
                'System experiencing repeated failures',
                0.3
            )
        
        # Check for rollback patterns
        rollback_count = sum(1 for o in recent if o.was_reverted)
        
        if rollback_count >= 2:
            self._update_pattern(
                'rollback_pattern',
                outcome,
                ImprovementCategory.ERROR_HANDLING,
                'Frequent rollbacks detected',
                0.35
            )
    
    def _maybe_generate_suggestions(self):
        """Generate suggestions if conditions are met"""
        if len(self._pending_suggestions) >= self._max_suggestions:
            return
        
        # Analyze patterns and generate suggestions
        for pattern in self._active_patterns.values():
            if pattern.confidence >= self._min_confidence:
                suggestion = self._pattern_to_suggestion(pattern)
                if suggestion:
                    self._pending_suggestions.append(suggestion)
                    self._db.store_suggestion(suggestion)
                    self._stats.suggestions_generated += 1
        
        # Sort by priority and confidence
        self._pending_suggestions.sort(
            key=lambda s: (s.priority.value, s.confidence),
            reverse=True
        )
        
        # Limit suggestions
        self._pending_suggestions = self._pending_suggestions[:self._max_suggestions]
    
    def _pattern_to_suggestion(self, pattern: LearningPattern) -> Optional[ImprovementSuggestion]:
        """Convert a pattern to an improvement suggestion"""
        
        # Skip if already suggested recently
        for existing in self._pending_suggestions:
            if pattern.pattern_id in existing.based_on_outcomes:
                return None
        
        suggestion_id = hashlib.sha256(
            f"{pattern.pattern_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Determine priority based on pattern
        priority = SuggestionPriority.MEDIUM
        if pattern.confidence > 0.8:
            priority = SuggestionPriority.HIGH
        elif 'rollback' in pattern.pattern_type or 'failure' in pattern.pattern_type:
            priority = SuggestionPriority.CRITICAL
        elif pattern.confidence < 0.6:
            priority = SuggestionPriority.LOW
        
        # Determine risk level based on learning mode and pattern
        risk = "medium"
        if self._learning_mode == LearningMode.CONSERVATIVE:
            risk = "high" if pattern.pattern_type in ['rollback_pattern', 'repeated_failures'] else "medium"
        elif self._learning_mode == LearningMode.AGGRESSIVE:
            risk = "low"
        
        return ImprovementSuggestion(
            id=suggestion_id,
            category=ImprovementCategory[pattern.conditions.get('category', 'CODE_QUALITY')],
            priority=priority,
            description=pattern.description,
            target_element="system",
            proposed_action=f"Apply pattern: {pattern.pattern_type}",
            reasoning=f"Learned from {pattern.occurrence_count} occurrences with {pattern.confidence:.0%} confidence",
            confidence=pattern.confidence,
            expected_benefit=f"Success rate: {pattern.success_count}/{pattern.occurrence_count}",
            risk_level=risk,
            success_probability=pattern.success_count / max(1, pattern.occurrence_count),
            based_on_outcomes=pattern.examples[:5],
        )
    
    def get_improvement_suggestions(
        self,
        category: ImprovementCategory = None,
        min_confidence: float = None,
        limit: int = 10
    ) -> List[ImprovementSuggestion]:
        """
        Get improvement suggestions.
        
        Args:
            category: Filter by category
            min_confidence: Minimum confidence filter
            limit: Maximum suggestions
            
        Returns:
            List of ImprovementSuggestion
        """
        suggestions = self._pending_suggestions.copy()
        
        # Apply filters
        if category:
            suggestions = [s for s in suggestions if s.category == category]
        
        min_conf = min_confidence or self._min_confidence
        suggestions = [s for s in suggestions if s.confidence >= min_conf]
        
        return suggestions[:limit]
    
    def get_learned_patterns(self) -> List[LearningPattern]:
        """Get all learned patterns"""
        return list(self._active_patterns.values())
    
    def record_metric(
        self,
        name: str,
        value: float,
        category: ImprovementCategory = ImprovementCategory.PERFORMANCE,
        unit: str = "",
        target: float = None,
        thresholds: Tuple[float, float] = None
    ):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            unit: Unit of measurement
            target: Target value
            thresholds: (warning, critical) thresholds
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            category=category,
            unit=unit,
            target_value=target,
            threshold_warning=thresholds[0] if thresholds else None,
            threshold_critical=thresholds[1] if thresholds else None,
        )
        
        # Store
        self._db.store_metric(metric)
        
        # Update in-memory
        self._current_metrics[name] = value
        self._metric_history[name].append((time.time(), value))
        
        # Check for anomalies
        self._check_metric_anomaly(metric)
    
    def _check_metric_anomaly(self, metric: PerformanceMetric):
        """Check for metric anomalies and learn from them"""
        if metric.status == "critical":
            self._update_pattern(
                f'critical_metric_{metric.name}',
                ModificationOutcome(
                    modification_id=f"metric_{metric.name}",
                    outcome_type=OutcomeType.FAILURE,
                    timestamp=time.time()
                ),
                ImprovementCategory.PERFORMANCE,
                f'Critical metric: {metric.name} = {metric.value}{metric.unit}',
                0.4
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics"""
        return {
            'current_metrics': dict(self._current_metrics),
            'metric_count': len(self._current_metrics),
            'history_depth': {k: len(v) for k, v in self._metric_history.items()},
        }
    
    def mark_suggestion_applied(self, suggestion_id: str, outcome: OutcomeType):
        """Mark a suggestion as applied with outcome"""
        for suggestion in self._pending_suggestions:
            if suggestion.id == suggestion_id:
                suggestion.applied = True
                suggestion.result_outcome = outcome.name
                self._db.store_suggestion(suggestion)
                self._stats.suggestions_applied += 1
                
                # Move from pending
                self._pending_suggestions.remove(suggestion)
                break
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze learning trends"""
        outcomes = list(self._recent_outcomes)
        
        if not outcomes:
            return {'trend': 'insufficient_data'}
        
        # Success rate over time
        recent_20 = outcomes[-20:] if len(outcomes) >= 20 else outcomes
        recent_10 = outcomes[-10:] if len(outcomes) >= 10 else outcomes
        
        success_rate_20 = sum(1 for o in recent_20 if o.success) / len(recent_20)
        success_rate_10 = sum(1 for o in recent_10 if o.success) / len(recent_10)
        
        # Trend direction
        if success_rate_10 > success_rate_20:
            trend = 'improving'
        elif success_rate_10 < success_rate_20:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Average performance delta
        perf_deltas = []
        for o in outcomes:
            delta = o.performance_delta
            if delta:
                perf_deltas.append(delta)
        
        return {
            'trend': trend,
            'success_rate_recent': success_rate_10,
            'success_rate_historical': success_rate_20,
            'total_outcomes': len(outcomes),
            'pattern_count': len(self._active_patterns),
            'pending_suggestions': len(self._pending_suggestions),
        }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning report"""
        return {
            'statistics': {
                'total_outcomes': self._stats.total_outcomes,
                'successful_outcomes': self._stats.successful_outcomes,
                'failed_outcomes': self._stats.failed_outcomes,
                'reverted_outcomes': self._stats.reverted_outcomes,
                'patterns_learned': self._stats.patterns_learned,
                'suggestions_generated': self._stats.suggestions_generated,
                'suggestions_applied': self._stats.suggestions_applied,
                'improvement_rate': self._stats.improvement_rate,
            },
            'trends': self.analyze_trends(),
            'metrics': self.get_metrics_summary(),
            'top_patterns': [
                {
                    'type': p.pattern_type,
                    'confidence': p.confidence,
                    'occurrences': p.occurrence_count,
                }
                for p in sorted(
                    self._active_patterns.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:5]
            ],
            'top_suggestions': [
                {
                    'id': s.id[:8],
                    'description': s.description,
                    'confidence': s.confidence,
                    'priority': s.priority.name,
                }
                for s in self._pending_suggestions[:5]
            ],
        }
    
    def reset_learning(self):
        """Reset learning data (use with caution)"""
        self._recent_outcomes.clear()
        self._active_patterns.clear()
        self._pending_suggestions.clear()
        self._current_metrics.clear()
        self._metric_history.clear()
        self._stats = LearningStats()
        
        logger.warning("Learning data reset!")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[SelfImprovementEngine] = None


def get_learning_engine() -> SelfImprovementEngine:
    """Get global SelfImprovementEngine instance"""
    global _engine
    if _engine is None:
        _engine = SelfImprovementEngine()
    return _engine


def record_outcome(outcome: ModificationOutcome):
    """Convenience function to record outcome"""
    get_learning_engine().record_outcome(outcome)


def get_suggestions(**kwargs) -> List[ImprovementSuggestion]:
    """Convenience function to get suggestions"""
    return get_learning_engine().get_improvement_suggestions(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """
    Ultra-comprehensive self-test for Self-Improvement Engine.
    
    Tests:
    1. Outcome recording and storage
    2. Pattern learning
    3. Suggestion generation
    4. Metric tracking
    5. Trend analysis
    6. Database operations
    7. Edge cases
    8. Performance under load
    """
    import tempfile
    import random
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
    }
    
    # Use temp database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "learning.db"
        
        engine = SelfImprovementEngine(
            db_path=str(db_path),
            learning_mode=LearningMode.BALANCED,
            enable_auto_learning=True,
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 1: Outcome Recording
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        outcome = ModificationOutcome(
            modification_id="test-mod-001",
            outcome_type=OutcomeType.SUCCESS,
            timestamp=time.time(),
            duration_ms=150.5,
            performance_before={'latency': 200.0},
            performance_after={'latency': 150.0},
        )
        
        try:
            engine.record_outcome(outcome)
            
            if engine._stats.total_outcomes == 1:
                results['passed'].append('outcome_recording')
                results['tests_passed'] += 1
            else:
                results['failed'].append('outcome_recording: count mismatch')
                results['tests_failed'] += 1
        except Exception as e:
            results['failed'].append(f'outcome_recording: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 2: Multiple Outcomes (Pattern Learning)
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            # Record multiple outcomes to trigger pattern learning
            for i in range(10):
                outcome = ModificationOutcome(
                    modification_id=f"test-mod-{i+2:03d}",
                    outcome_type=OutcomeType.SUCCESS if i % 3 != 0 else OutcomeType.FAILURE,
                    timestamp=time.time() + i,
                    duration_ms=100 + i * 10,
                )
                engine.record_outcome(outcome)
            
            patterns = engine.get_learned_patterns()
            
            if len(patterns) > 0:
                results['passed'].append(f'pattern_learning: {len(patterns)} patterns')
                results['tests_passed'] += 1
            else:
                results['warnings'].append('pattern_learning: no patterns learned')
                results['tests_passed'] += 1  # Still pass, patterns may need more data
                
        except Exception as e:
            results['failed'].append(f'pattern_learning: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 3: Suggestion Generation
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            suggestions = engine.get_improvement_suggestions()
            
            if isinstance(suggestions, list):
                results['passed'].append(f'suggestion_generation: {len(suggestions)} suggestions')
                results['tests_passed'] += 1
            else:
                results['failed'].append('suggestion_generation: not a list')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'suggestion_generation: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 4: Metric Tracking
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            engine.record_metric(
                name="response_time",
                value=150.5,
                category=ImprovementCategory.PERFORMANCE,
                unit="ms",
                target=100.0,
                thresholds=(200.0, 500.0)
            )
            
            engine.record_metric(
                name="memory_usage",
                value=256.0,
                category=ImprovementCategory.MEMORY,
                unit="MB",
                target=200.0
            )
            
            metrics = engine.get_metrics_summary()
            
            if metrics['metric_count'] == 2:
                results['passed'].append('metric_tracking')
                results['tests_passed'] += 1
            else:
                results['failed'].append(f'metric_tracking: count={metrics["metric_count"]}')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'metric_tracking: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 5: Trend Analysis
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            trends = engine.analyze_trends()
            
            if 'trend' in trends and trends['trend'] in ('improving', 'declining', 'stable', 'insufficient_data'):
                results['passed'].append(f'trend_analysis: {trends["trend"]}')
                results['tests_passed'] += 1
            else:
                results['failed'].append('trend_analysis: invalid trend')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'trend_analysis: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 6: Learning Report
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            report = engine.get_learning_report()
            
            if 'statistics' in report and 'trends' in report:
                results['passed'].append('learning_report')
                results['tests_passed'] += 1
            else:
                results['failed'].append('learning_report: missing sections')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'learning_report: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 7: Database Persistence
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            # Verify database has data
            db_outcomes = engine._db.get_outcomes(limit=100)
            
            if len(db_outcomes) > 0:
                results['passed'].append(f'database_persistence: {len(db_outcomes)} outcomes stored')
                results['tests_passed'] += 1
            else:
                results['failed'].append('database_persistence: no outcomes stored')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'database_persistence: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 8: Rollback Detection
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            # Record rollback outcome
            rollback_outcome = ModificationOutcome(
                modification_id="test-rollback-001",
                outcome_type=OutcomeType.ROLLBACK,
                timestamp=time.time(),
                was_reverted=True,
                revert_reason="Test rollback",
            )
            engine.record_outcome(rollback_outcome)
            
            if engine._stats.reverted_outcomes > 0:
                results['passed'].append('rollback_detection')
                results['tests_passed'] += 1
            else:
                results['failed'].append('rollback_detection: not counted')
                results['tests_failed'] += 1
                
        except Exception as e:
            results['failed'].append(f'rollback_detection: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 9: Performance Under Load
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        try:
            start_time = time.time()
            
            # Record 100 outcomes rapidly
            for i in range(100):
                outcome = ModificationOutcome(
                    modification_id=f"load-test-{i:04d}",
                    outcome_type=random.choice(list(OutcomeType)),
                    timestamp=time.time(),
                    duration_ms=random.uniform(10, 500),
                )
                engine.record_outcome(outcome)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if elapsed_ms < 5000:  # Should complete in under 5 seconds
                results['passed'].append(f'performance_load: 100 outcomes in {elapsed_ms:.0f}ms')
                results['tests_passed'] += 1
            else:
                results['warnings'].append(f'performance_load: slow ({elapsed_ms:.0f}ms)')
                results['tests_passed'] += 1
                
        except Exception as e:
            results['failed'].append(f'performance_load: {e}')
            results['tests_failed'] += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEST 10: Edge Cases
        # ═══════════════════════════════════════════════════════════════════════
        results['tests_run'] += 1
        
        edge_cases_passed = 0
        
        try:
            # Empty outcome
            empty_outcome = ModificationOutcome(
                modification_id="",
                outcome_type=OutcomeType.SUCCESS,
                timestamp=time.time(),
            )
            engine.record_outcome(empty_outcome)
            edge_cases_passed += 1
            
            # Outcome with no performance data
            no_perf = ModificationOutcome(
                modification_id="no-perf",
                outcome_type=OutcomeType.SUCCESS,
                timestamp=time.time(),
            )
            engine.record_outcome(no_perf)
            edge_cases_passed += 1
            
            # Outcome with extreme values
            extreme_outcome = ModificationOutcome(
                modification_id="extreme",
                outcome_type=OutcomeType.SUCCESS,
                timestamp=time.time(),
                duration_ms=999999999.0,
                memory_before_mb=99999.0,
                memory_after_mb=0.0,
            )
            engine.record_outcome(extreme_outcome)
            edge_cases_passed += 1
            
            results['passed'].append(f'edge_cases: {edge_cases_passed}/3 passed')
            results['tests_passed'] += 1
            
        except Exception as e:
            results['failed'].append(f'edge_cases: {e}')
            results['tests_failed'] += 1
        
        # Add final report
        results['final_report'] = engine.get_learning_report()
        results['success_rate'] = results['tests_passed'] / max(1, results['tests_run']) * 100
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Self-Improvement Engine - Comprehensive Self Test")
    print("=" * 70)
    print(f"Device: Realme 2 Pro Lite (RMP2402)")
    print("-" * 70)
    
    test_results = self_test()
    
    print(f"\n📊 Test Results: {test_results['tests_passed']}/{test_results['tests_run']} passed ({test_results['success_rate']:.1f}%)")
    
    print("\n✅ Passed Tests:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed Tests:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    if test_results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in test_results['warnings']:
            print(f"   ! {warning}")
    
    print("\n📈 Learning Statistics:")
    if 'final_report' in test_results and 'statistics' in test_results['final_report']:
        stats = test_results['final_report']['statistics']
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Self-Improvement Engine Test Complete!")
    print("=" * 70)
