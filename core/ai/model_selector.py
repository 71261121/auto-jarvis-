#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Intelligent Model Selection Engine
=========================================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Capability-based routing (reasoning, coding, math, etc.)
- Cost optimization (always FREE for JARVIS)
- Context window awareness
- Performance-based scoring
- Automatic fallback chains

Features:
- Task-type detection and routing
- Model capability matrix
- Performance tracking per model
- Intelligent fallback selection
- Context length optimization
- Memory-efficient caching

Memory Impact: < 200KB
"""

import time
import threading
import logging
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TaskType(Enum):
    """Types of tasks for model routing"""
    GENERAL_CHAT = auto()
    REASONING = auto()
    CODING = auto()
    MATH = auto()
    ANALYSIS = auto()
    CREATIVE_WRITING = auto()
    SUMMARIZATION = auto()
    TRANSLATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    LONG_CONTEXT = auto()
    QUICK_RESPONSE = auto()
    SELF_MODIFICATION = auto()


class ModelCapability(Enum):
    """Capabilities that models can have"""
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    LONG_CONTEXT = "long_context"
    FAST_RESPONSE = "fast_response"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    MULTIMODAL = "multimodal"
    SELF_MODIFY = "self_modify"


class ModelStatus(Enum):
    """Status of a model"""
    AVAILABLE = auto()
    DEGRADED = auto()
    UNAVAILABLE = auto()
    RATE_LIMITED = auto()
    UNKNOWN = auto()


@dataclass
class ModelInfo:
    """Information about a model"""
    id: str
    name: str
    provider: str
    context_length: int = 32000
    capabilities: Set[ModelCapability] = field(default_factory=set)
    is_free: bool = True
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    last_used: float = 0.0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    status: ModelStatus = ModelStatus.AVAILABLE
    rate_limit_reset: float = 0.0
    
    def __hash__(self):
        return hash(self.id)
    
    def update_stats(self, success: bool, latency_ms: float):
        """Update model statistics"""
        self.total_requests += 1
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
        
        # Update running averages
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        
        self.success_rate = self.total_successes / max(1, self.total_requests)
        self.last_used = time.time()


@dataclass
class SelectionResult:
    """Result of model selection"""
    model_id: str
    model_info: Optional[ModelInfo]
    score: float
    reason: str
    fallback_chain: List[str] = field(default_factory=list)
    estimated_latency_ms: float = 0.0
    estimated_tokens_available: int = 0


@dataclass
class TaskProfile:
    """Profile of a task for routing"""
    task_type: TaskType
    required_capabilities: Set[ModelCapability]
    estimated_tokens: int = 1000
    priority: int = 1  # 1=normal, 2=high, 3=critical
    prefer_speed: bool = False
    prefer_quality: bool = True
    allow_fallback: bool = True
    max_latency_ms: float = 60000.0


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY (FREE Models on OpenRouter)
# ═══════════════════════════════════════════════════════════════════════════════

# Capability mapping for task types
TASK_CAPABILITY_MAP = {
    TaskType.REASONING: {ModelCapability.REASONING},
    TaskType.CODING: {ModelCapability.CODING},
    TaskType.MATH: {ModelCapability.MATH, ModelCapability.REASONING},
    TaskType.ANALYSIS: {ModelCapability.ANALYSIS, ModelCapability.REASONING},
    TaskType.CODE_REVIEW: {ModelCapability.CODING, ModelCapability.ANALYSIS},
    TaskType.DEBUGGING: {ModelCapability.CODING, ModelCapability.REASONING},
    TaskType.LONG_CONTEXT: {ModelCapability.LONG_CONTEXT},
    TaskType.QUICK_RESPONSE: {ModelCapability.FAST_RESPONSE},
    TaskType.SELF_MODIFICATION: {ModelCapability.SELF_MODIFY, ModelCapability.CODING},
    TaskType.CREATIVE_WRITING: {ModelCapability.CREATIVE},
    TaskType.SUMMARIZATION: {ModelCapability.LONG_CONTEXT, ModelCapability.ANALYSIS},
    TaskType.TRANSLATION: set(),  # Any model can translate
    TaskType.GENERAL_CHAT: set(),  # Any model can chat
}

# FREE models with their capabilities (research-based)
FREE_MODELS = {
    # OpenRouter Auto (best free selection)
    "openrouter/free": ModelInfo(
        id="openrouter/free",
        name="OpenRouter Auto Free",
        provider="openrouter",
        context_length=128000,
        capabilities={
            ModelCapability.CODING,
            ModelCapability.FAST_RESPONSE,
        },
        is_free=True,
    ),
    
    # Aurora Alpha (advanced reasoning)
    "openrouter/aurora-alpha": ModelInfo(
        id="openrouter/aurora-alpha",
        name="Aurora Alpha",
        provider="openrouter",
        context_length=128000,
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.CODING,
            ModelCapability.MATH,
            ModelCapability.ANALYSIS,
            ModelCapability.SELF_MODIFY,
        },
        is_free=True,
    ),
    
    # Step 3.5 Flash (fast responses)
    "stepfun/step-3.5-flash:free": ModelInfo(
        id="stepfun/step-3.5-flash:free",
        name="Step 3.5 Flash",
        provider="stepfun",
        context_length=128000,
        capabilities={
            ModelCapability.FAST_RESPONSE,
            ModelCapability.CODING,
        },
        is_free=True,
    ),
    
    # Trinity Large (large model)
    "arcee-ai/trinity-large-preview:free": ModelInfo(
        id="arcee-ai/trinity-large-preview:free",
        name="Trinity Large",
        provider="arcee-ai",
        context_length=128000,
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.ANALYSIS,
        },
        is_free=True,
    ),
    
    # Solar Pro (general purpose)
    "upstage/solar-pro-3:free": ModelInfo(
        id="upstage/solar-pro-3:free",
        name="Solar Pro 3",
        provider="upstage",
        context_length=128000,
        capabilities={
            ModelCapability.CODING,
            ModelCapability.ANALYSIS,
        },
        is_free=True,
    ),
    
    # LFM Thinking (reasoning model)
    "liquid/lfm-2.5-1.2b-thinking:free": ModelInfo(
        id="liquid/lfm-2.5-1.2b-thinking:free",
        name="LFM Thinking",
        provider="liquid",
        context_length=32000,
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.FAST_RESPONSE,
        },
        is_free=True,
    ),
    
    # LFM Instruct
    "liquid/lfm-2.5-1.2b-instruct:free": ModelInfo(
        id="liquid/lfm-2.5-1.2b-instruct:free",
        name="LFM Instruct",
        provider="liquid",
        context_length=32000,
        capabilities={
            ModelCapability.FAST_RESPONSE,
        },
        is_free=True,
    ),
    
    # DeepSeek R1 (legacy - may not always work)
    "deepseek/deepseek-r1-0528:free": ModelInfo(
        id="deepseek/deepseek-r1-0528:free",
        name="DeepSeek R1",
        provider="deepseek",
        context_length=164000,
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.CODING,
            ModelCapability.MATH,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.SELF_MODIFY,
        },
        is_free=True,
    ),
    
    # Gemini Flash (legacy - may not always work)
    "google/gemini-2.0-flash-exp:free": ModelInfo(
        id="google/gemini-2.0-flash-exp:free",
        name="Gemini 2.0 Flash",
        provider="google",
        context_length=1000000,  # 1M context!
        capabilities={
            ModelCapability.LONG_CONTEXT,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.MULTIMODAL,
        },
        is_free=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TASK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDetector:
    """
    Detects task type from user input.
    
    Uses pattern matching and heuristics to classify tasks.
    Memory-efficient: no ML models, just pattern matching.
    """
    
    # Patterns for task detection
    CODING_PATTERNS = [
        'write code', 'write a function', 'write a class', 'implement',
        'debug', 'fix this code', 'error in', 'bug in', 'not working',
        'python', 'javascript', 'code', 'function', 'class', 'method',
        'variable', 'loop', 'condition', 'import', 'module', 'package',
        'api', 'json', 'http', 'request', 'response', 'async', 'await',
        'def ', 'class ', 'import ', 'from ', 'return ', 'print(',
        '```python', '```javascript', '```java', '```cpp',
    ]
    
    REASONING_PATTERNS = [
        'why', 'explain why', 'reason', 'logic', 'because', 'therefore',
        'conclude', 'deduce', 'infer', 'analyze', 'argument', 'proof',
        'think through', 'step by step', 'reasoning', 'rationale',
    ]
    
    MATH_PATTERNS = [
        'calculate', 'compute', 'solve', 'equation', 'formula', 'math',
        'algebra', 'calculus', 'geometry', 'statistics', 'probability',
        'number', 'sum', 'average', 'mean', 'median', 'percentage',
        'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'integral', 'derivative',
        '^2', '^3', 'x^2', 'n^2', '+', '-', '*', '/', '=',
    ]
    
    ANALYSIS_PATTERNS = [
        'analyze', 'analysis', 'compare', 'contrast', 'evaluate', 'assess',
        'review', 'critique', 'examine', 'investigate', 'study', 'breakdown',
        'pros and cons', 'advantages', 'disadvantages', 'strengths', 'weaknesses',
    ]
    
    CREATIVE_PATTERNS = [
        'write a story', 'write a poem', 'creative', 'imagine', 'fiction',
        'narrative', 'tale', 'novel', 'character', 'plot', 'dialogue',
        'song', 'lyrics', 'script', 'screenplay', 'creative writing',
    ]
    
    SUMMARY_PATTERNS = [
        'summarize', 'summary', 'brief', 'short version', 'tldr', 'tl;dr',
        'key points', 'main points', 'overview', 'recap', 'condense',
        'in a nutshell', 'in short', 'to summarize', 'essence',
    ]
    
    DEBUG_PATTERNS = [
        'debug', 'fix', 'error', 'exception', 'traceback', 'bug', 'issue',
        'not working', 'crash', 'fail', 'problem', 'wrong', 'incorrect',
        'help me fix', 'what\'s wrong', 'what is wrong', 'why does this fail',
    ]
    
    LONG_CONTEXT_PATTERNS = [
        'entire document', 'whole file', 'large text', 'long document',
        'multiple files', 'codebase', 'project', 'repository', 'all the code',
    ]
    
    QUICK_PATTERNS = [
        'quick', 'fast', 'briefly', 'shortly', 'in few words', 'simply',
        'just tell me', 'just say', 'one word', 'yes or no',
    ]
    
    SELF_MOD_PATTERNS = [
        'modify yourself', 'change your code', 'self modify', 'improve yourself',
        'update your', 'rewrite your', 'optimize yourself', 'upgrade yourself',
        'evolve', 'self improvement', 'modify jarvis', 'jarvis modify',
    ]
    
    def __init__(self):
        self._cache: Dict[str, TaskType] = {}
        self._lock = threading.Lock()
    
    def detect(self, text: str) -> TaskProfile:
        """
        Detect task type from input text.
        
        Args:
            text: User input text
            
        Returns:
            TaskProfile with detected type and requirements
        """
        text_lower = text.lower()
        
        # Check cache
        cache_key = hashlib.md5(text_lower.encode()).hexdigest()[:16]
        with self._lock:
            if cache_key in self._cache:
                task_type = self._cache[cache_key]
            else:
                task_type = self._detect_type(text_lower)
                self._cache[cache_key] = task_type
        
        # Build profile
        profile = TaskProfile(
            task_type=task_type,
            required_capabilities=TASK_CAPABILITY_MAP.get(task_type, set()),
            estimated_tokens=self._estimate_tokens(text),
            prefer_speed=self._check_patterns(text_lower, self.QUICK_PATTERNS),
        )
        
        # Adjust for specific cases
        if self._check_patterns(text_lower, self.LONG_CONTEXT_PATTERNS):
            profile.required_capabilities.add(ModelCapability.LONG_CONTEXT)
            profile.estimated_tokens = max(profile.estimated_tokens, 50000)
        
        if self._check_patterns(text_lower, self.DEBUG_PATTERNS):
            profile.task_type = TaskType.DEBUGGING
            profile.required_capabilities = {ModelCapability.CODING, ModelCapability.REASONING}
        
        return profile
    
    def _detect_type(self, text_lower: str) -> TaskType:
        """Detect the primary task type"""
        scores = {
            TaskType.CODING: self._score_patterns(text_lower, self.CODING_PATTERNS),
            TaskType.REASONING: self._score_patterns(text_lower, self.REASONING_PATTERNS),
            TaskType.MATH: self._score_patterns(text_lower, self.MATH_PATTERNS),
            TaskType.ANALYSIS: self._score_patterns(text_lower, self.ANALYSIS_PATTERNS),
            TaskType.CREATIVE_WRITING: self._score_patterns(text_lower, self.CREATIVE_PATTERNS),
            TaskType.SUMMARIZATION: self._score_patterns(text_lower, self.SUMMARY_PATTERNS),
            TaskType.DEBUGGING: self._score_patterns(text_lower, self.DEBUG_PATTERNS),
            TaskType.LONG_CONTEXT: self._score_patterns(text_lower, self.LONG_CONTEXT_PATTERNS),
            TaskType.QUICK_RESPONSE: self._score_patterns(text_lower, self.QUICK_PATTERNS),
            TaskType.SELF_MODIFICATION: self._score_patterns(text_lower, self.SELF_MOD_PATTERNS),
        }
        
        # Find highest scoring type
        max_type = max(scores.items(), key=lambda x: x[1])
        
        if max_type[1] > 0:
            return max_type[0]
        
        return TaskType.GENERAL_CHAT
    
    def _score_patterns(self, text: str, patterns: List[str]) -> int:
        """Count pattern matches"""
        score = 0
        for pattern in patterns:
            if pattern in text:
                score += 1
        return score
    
    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if any pattern matches"""
        return any(p in text for p in patterns)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 chars per token
        return max(100, len(text) // 4)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ModelSelector:
    """
    Intelligent Model Selection Engine.
    
    Selects the best model based on:
    1. Task requirements
    2. Model capabilities
    3. Historical performance
    4. Context length needs
    5. Current availability
    
    Memory Budget: < 500KB
    
    Usage:
        selector = ModelSelector()
        
        # Automatic selection
        result = selector.select("Write a Python function to sort a list")
        print(f"Selected: {result.model_id}")
        
        # With specific task type
        result = selector.select_for_task(TaskType.CODING, estimated_tokens=500)
        
        # Record outcome
        selector.record_result(result.model_id, success=True, latency_ms=1500)
    """
    
    # Default fallback chain for general tasks
    DEFAULT_FALLBACK = [
        "openrouter/free",
        "openrouter/aurora-alpha",
        "stepfun/step-3.5-flash:free",
        "upstage/solar-pro-3:free",
    ]
    
    # Fallback chain for reasoning tasks
    REASONING_FALLBACK = [
        "openrouter/aurora-alpha",
        "arcee-ai/trinity-large-preview:free",
        "deepseek/deepseek-r1-0528:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
    ]
    
    # Fallback chain for coding tasks
    CODING_FALLBACK = [
        "openrouter/aurora-alpha",
        "openrouter/free",
        "stepfun/step-3.5-flash:free",
        "upstage/solar-pro-3:free",
    ]
    
    # Fallback chain for long context
    LONG_CONTEXT_FALLBACK = [
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1-0528:free",
        "arcee-ai/trinity-large-preview:free",
    ]
    
    # Fallback chain for quick responses
    QUICK_FALLBACK = [
        "stepfun/step-3.5-flash:free",
        "liquid/lfm-2.5-1.2b-instruct:free",
        "openrouter/free",
    ]
    
    # Fallback for self-modification (needs reasoning + coding)
    SELF_MOD_FALLBACK = [
        "openrouter/aurora-alpha",
        "deepseek/deepseek-r1-0528:free",
        "arcee-ai/trinity-large-preview:free",
    ]
    
    def __init__(self, models: Dict[str, ModelInfo] = None):
        """
        Initialize model selector.
        
        Args:
            models: Dictionary of model info (defaults to FREE_MODELS)
        """
        self._models = models or dict(FREE_MODELS)
        self._detector = TaskDetector()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'avg_latency': 0.0,
            'success_rate': 1.0,
            'total_requests': 0,
        })
        
        # Selection history
        self._selection_history: List[Tuple[str, TaskType, float]] = []
        
        # Statistics
        self._stats = {
            'total_selections': 0,
            'selections_by_type': defaultdict(int),
            'fallbacks_used': 0,
        }
    
    def select(
        self,
        text: str,
        system_prompt: str = None,
        context_tokens: int = 0
    ) -> SelectionResult:
        """
        Select the best model for a request.
        
        Args:
            text: User input text
            system_prompt: Optional system prompt
            context_tokens: Additional context tokens
            
        Returns:
            SelectionResult with selected model
        """
        # Detect task type
        profile = self._detector.detect(text)
        
        # Add context tokens
        profile.estimated_tokens += context_tokens
        
        # Select based on task
        return self.select_for_task(
            task_type=profile.task_type,
            required_capabilities=profile.required_capabilities,
            estimated_tokens=profile.estimated_tokens,
            prefer_speed=profile.prefer_speed
        )
    
    def select_for_task(
        self,
        task_type: TaskType,
        required_capabilities: Set[ModelCapability] = None,
        estimated_tokens: int = 1000,
        prefer_speed: bool = False,
        prefer_quality: bool = True
    ) -> SelectionResult:
        """
        Select model for a specific task type.
        
        Args:
            task_type: Type of task
            required_capabilities: Required capabilities
            estimated_tokens: Estimated tokens needed
            prefer_speed: Prefer faster models
            prefer_quality: Prefer higher quality models
            
        Returns:
            SelectionResult with selected model
        """
        with self._lock:
            self._stats['total_selections'] += 1
            self._stats['selections_by_type'][task_type.name] += 1
            
            # Get required capabilities
            capabilities = required_capabilities or TASK_CAPABILITY_MAP.get(task_type, set())
            
            # Get fallback chain for task type
            fallback_chain = self._get_fallback_chain(task_type)
            
            # Score all models
            scored_models = []
            
            for model_id, model_info in self._models.items():
                score, reason = self._score_model(
                    model_info=model_info,
                    required_capabilities=capabilities,
                    estimated_tokens=estimated_tokens,
                    prefer_speed=prefer_speed,
                    prefer_quality=prefer_quality,
                    task_type=task_type
                )
                
                if score > 0:
                    scored_models.append((model_id, model_info, score, reason))
            
            # Sort by score
            scored_models.sort(key=lambda x: x[2], reverse=True)
            
            # Select best available
            for model_id, model_info, score, reason in scored_models:
                if model_info.status in (ModelStatus.AVAILABLE, ModelStatus.DEGRADED):
                    # Build fallback chain excluding selected model
                    model_fallback = [m for m in fallback_chain if m != model_id]
                    
                    result = SelectionResult(
                        model_id=model_id,
                        model_info=model_info,
                        score=score,
                        reason=reason,
                        fallback_chain=model_fallback,
                        estimated_latency_ms=model_info.avg_latency_ms,
                        estimated_tokens_available=model_info.context_length - estimated_tokens
                    )
                    
                    # Record selection
                    self._selection_history.append((model_id, task_type, time.time()))
                    
                    return result
            
            # No suitable model found, use first fallback
            default_model = fallback_chain[0] if fallback_chain else "openrouter/free"
            model_info = self._models.get(default_model)
            
            return SelectionResult(
                model_id=default_model,
                model_info=model_info,
                score=0,
                reason="No optimal model found, using default",
                fallback_chain=fallback_chain[1:],
            )
    
    def _score_model(
        self,
        model_info: ModelInfo,
        required_capabilities: Set[ModelCapability],
        estimated_tokens: int,
        prefer_speed: bool,
        prefer_quality: bool,
        task_type: TaskType
    ) -> Tuple[float, str]:
        """
        Score a model for a task.
        
        Returns:
            Tuple of (score, reason)
        """
        score = 0.0
        reasons = []
        
        # Check if model is available
        if model_info.status == ModelStatus.UNAVAILABLE:
            return 0, "Model unavailable"
        
        if model_info.status == ModelStatus.RATE_LIMITED:
            if time.time() < model_info.rate_limit_reset:
                return 0, "Rate limited"
        
        # Check context length
        if estimated_tokens > model_info.context_length:
            return 0, f"Context too long ({estimated_tokens} > {model_info.context_length})"
        
        # Capability matching (most important)
        if required_capabilities:
            matched = len(required_capabilities & model_info.capabilities)
            required = len(required_capabilities)
            capability_score = (matched / required) * 50 if required > 0 else 25
            score += capability_score
            
            if matched == required:
                reasons.append("All capabilities matched")
            elif matched > 0:
                reasons.append(f"{matched}/{required} capabilities matched")
        else:
            score += 25  # No specific requirements
        
        # Performance score (based on history)
        perf_score = model_info.success_rate * 20
        score += perf_score
        
        if model_info.success_rate > 0.9:
            reasons.append("High success rate")
        
        # Speed preference
        if prefer_speed:
            if ModelCapability.FAST_RESPONSE in model_info.capabilities:
                score += 15
                reasons.append("Fast response model")
            
            # Penalize slow models
            if model_info.avg_latency_ms > 5000:
                score -= 10
        
        # Quality preference
        if prefer_quality:
            if ModelCapability.REASONING in model_info.capabilities:
                score += 10
            if task_type in (TaskType.CODING, TaskType.DEBUGGING, TaskType.SELF_MODIFICATION):
                if ModelCapability.CODING in model_info.capabilities:
                    score += 15
                    reasons.append("Code-capable model")
        
        # Context efficiency bonus
        if estimated_tokens > 50000 and ModelCapability.LONG_CONTEXT in model_info.capabilities:
            score += 20
            reasons.append("Long context model")
        
        # Free model bonus (all our models are free)
        if model_info.is_free:
            score += 5
        
        # Availability status penalty
        if model_info.status == ModelStatus.DEGRADED:
            score -= 10
            reasons.append("Degraded performance")
        
        reason = "; ".join(reasons) if reasons else "General purpose selection"
        
        return score, reason
    
    def _get_fallback_chain(self, task_type: TaskType) -> List[str]:
        """Get fallback chain for task type"""
        chains = {
            TaskType.REASONING: self.REASONING_FALLBACK,
            TaskType.MATH: self.REASONING_FALLBACK,
            TaskType.ANALYSIS: self.REASONING_FALLBACK,
            TaskType.CODING: self.CODING_FALLBACK,
            TaskType.CODE_REVIEW: self.CODING_FALLBACK,
            TaskType.DEBUGGING: self.CODING_FALLBACK,
            TaskType.LONG_CONTEXT: self.LONG_CONTEXT_FALLBACK,
            TaskType.SUMMARIZATION: self.LONG_CONTEXT_FALLBACK,
            TaskType.QUICK_RESPONSE: self.QUICK_FALLBACK,
            TaskType.SELF_MODIFICATION: self.SELF_MOD_FALLBACK,
        }
        
        chain = chains.get(task_type, self.DEFAULT_FALLBACK)
        
        # Filter to available models
        return [m for m in chain if m in self._models]
    
    def record_result(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        error: str = None
    ):
        """
        Record the result of using a model.
        
        Args:
            model_id: Model that was used
            success: Whether request succeeded
            latency_ms: Request latency
            error: Error message if failed
        """
        with self._lock:
            if model_id in self._models:
                model_info = self._models[model_id]
                model_info.update_stats(success, latency_ms)
                
                # Update status based on failures
                if not success:
                    if "rate limit" in (error or "").lower():
                        model_info.status = ModelStatus.RATE_LIMITED
                        model_info.rate_limit_reset = time.time() + 60  # 1 min cooldown
                    elif model_info.success_rate < 0.5:
                        model_info.status = ModelStatus.DEGRADED
                else:
                    if model_info.status == ModelStatus.DEGRADED:
                        if model_info.success_rate > 0.8:
                            model_info.status = ModelStatus.AVAILABLE
    
    def mark_unavailable(self, model_id: str, duration_seconds: float = 300):
        """Mark a model as temporarily unavailable"""
        with self._lock:
            if model_id in self._models:
                self._models[model_id].status = ModelStatus.UNAVAILABLE
                self._models[model_id].rate_limit_reset = time.time() + duration_seconds
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        return self._models.get(model_id)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return [
            mid for mid, m in self._models.items()
            if m.status == ModelStatus.AVAILABLE
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics"""
        with self._lock:
            model_stats = {
                mid: {
                    'success_rate': m.success_rate,
                    'avg_latency_ms': m.avg_latency_ms,
                    'total_requests': m.total_requests,
                    'status': m.status.name,
                }
                for mid, m in self._models.items()
            }
            
            return {
                'total_selections': self._stats['total_selections'],
                'selections_by_type': dict(self._stats['selections_by_type']),
                'fallbacks_used': self._stats['fallbacks_used'],
                'models': model_stats,
            }
    
    def reset_stats(self):
        """Reset all statistics"""
        with self._lock:
            for model in self._models.values():
                model.total_requests = 0
                model.total_successes = 0
                model.total_failures = 0
                model.success_rate = 1.0
                model.avg_latency_ms = 0.0
                model.status = ModelStatus.AVAILABLE
            
            self._selection_history.clear()
            self._stats = {
                'total_selections': 0,
                'selections_by_type': defaultdict(int),
                'fallbacks_used': 0,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get global model selector instance"""
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector


def select_model(text: str, **kwargs) -> SelectionResult:
    """Convenience function to select model"""
    return get_model_selector().select(text, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for model selector"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    selector = ModelSelector()
    
    # Test 1: Detect coding task
    profile = selector._detector.detect("Write a Python function to sort a list")
    if profile.task_type == TaskType.CODING:
        results['passed'].append('detect_coding')
    else:
        results['failed'].append(f'detect_coding (got {profile.task_type.name})')
    
    # Test 2: Detect reasoning task
    profile = selector._detector.detect("Explain why the sky is blue")
    if profile.task_type == TaskType.REASONING:
        results['passed'].append('detect_reasoning')
    else:
        results['failed'].append(f'detect_reasoning (got {profile.task_type.name})')
    
    # Test 3: Detect math task
    profile = selector._detector.detect("Calculate the square root of 144")
    if profile.task_type == TaskType.MATH:
        results['passed'].append('detect_math')
    else:
        results['failed'].append(f'detect_math (got {profile.task_type.name})')
    
    # Test 4: Select model for coding
    result = selector.select_for_task(TaskType.CODING)
    if result.model_id and result.model_info:
        results['passed'].append(f'select_coding: {result.model_id}')
    else:
        results['failed'].append('select_coding')
    
    # Test 5: Select model for long context
    result = selector.select_for_task(TaskType.LONG_CONTEXT, estimated_tokens=100000)
    if result.model_id:
        model = selector.get_model_info(result.model_id)
        if model and model.context_length >= 100000:
            results['passed'].append('select_long_context')
        else:
            results['warnings'].append(f'long_context: {result.model_id} ({model.context_length if model else 0} tokens)')
    else:
        results['failed'].append('select_long_context')
    
    # Test 6: Record result and check stats
    selector.record_result("openrouter/free", success=True, latency_ms=1500)
    stats = selector.get_stats()
    if stats['total_selections'] > 0:
        results['passed'].append('record_stats')
    else:
        results['failed'].append('record_stats')
    
    # Test 7: Fallback chain
    result = selector.select_for_task(TaskType.REASONING)
    if result.fallback_chain:
        results['passed'].append(f'fallback_chain: {len(result.fallback_chain)} models')
    else:
        results['warnings'].append('fallback_chain: empty')
    
    results['stats'] = selector.get_stats()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Model Selection Engine - Self Test")
    print("=" * 70)
    print(f"Device: Realme 2 Pro Lite (RMP2402)")
    print("-" * 70)
    
    test_results = self_test()
    
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
    
    # Demo
    print("\n" + "=" * 70)
    print("Model Selection Demo:")
    print("-" * 70)
    
    selector = get_model_selector()
    
    test_prompts = [
        "Write a Python function to merge two sorted lists",
        "Explain the theory of relativity step by step",
        "Calculate the derivative of x^2 + 2x + 1",
        "Quick question: What is 2+2?",
    ]
    
    for prompt in test_prompts:
        result = selector.select(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  → Model: {result.model_id}")
        print(f"  → Score: {result.score:.1f}")
        print(f"  → Reason: {result.reason}")
    
    print("\n" + "=" * 70)
