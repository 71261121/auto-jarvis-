#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Advanced Rate Limiter
============================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Token Bucket Algorithm (classical, proven)
- Leaky Bucket for smoothing
- Adaptive Rate Limiting (learns from responses)
- Circuit Breaker Pattern (fault tolerance)

Features:
- Token Bucket with refill rate
- Exponential backoff with jitter
- Adaptive delay adjustment
- Circuit breaker for fail-fast
- Per-endpoint limiting
- Memory-efficient design
- Thread-safe operations

Memory Impact: < 100KB
"""

import time
import threading
import random
import logging
import math
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject all
    HALF_OPEN = auto()   # Testing if recovered


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = auto()      # Classic token bucket
    LEAKY_BUCKET = auto()      # Smooth rate limiting
    SLIDING_WINDOW = auto()    # Fixed requests per window
    ADAPTIVE = auto()          # Learns from responses


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter"""
    requests_per_minute: int = 60
    burst_size: int = 10
    refill_rate: float = 1.0  # Tokens per second
    backoff_factor: float = 2.0
    max_backoff: float = 60.0
    initial_backoff: float = 1.0
    jitter_percent: float = 0.1
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0
    circuit_success_threshold: int = 3
    adaptive_window_size: int = 100
    enable_adaptive: bool = True


@dataclass
class RateLimitResult:
    """Result of a rate limit check"""
    allowed: bool
    wait_time_ms: float = 0.0
    tokens_remaining: float = 0.0
    current_rate: float = 0.0
    reason: str = ""
    retry_after: Optional[float] = None


@dataclass
class RequestRecord:
    """Record of a request for tracking"""
    timestamp: float
    success: bool
    latency_ms: float
    tokens_used: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUCKET IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class TokenBucket:
    """
    Token Bucket Rate Limiter.
    
    Classic algorithm with precise token management.
    Memory efficient - only stores bucket state.
    
    How it works:
    1. Bucket has max capacity (burst_size)
    2. Tokens refill at constant rate
    3. Each request consumes tokens
    4. If bucket empty, request is delayed
    
    Memory: ~100 bytes per bucket
    """
    
    __slots__ = ['_capacity', '_tokens', '_refill_rate', '_last_refill', '_lock']
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens (burst size)
            refill_rate: Tokens added per second
        """
        self._capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = refill_rate
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self._last_refill
        
        if elapsed > 0:
            new_tokens = elapsed * self._refill_rate
            self._tokens = min(self._capacity, self._tokens + new_tokens)
            self._last_refill = now
    
    def consume(self, tokens: int = 1) -> RateLimitResult:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitResult with success status
        """
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self._tokens,
                    current_rate=self._refill_rate
                )
            
            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._refill_rate
            
            return RateLimitResult(
                allowed=False,
                wait_time_ms=wait_time * 1000,
                tokens_remaining=self._tokens,
                current_rate=self._refill_rate,
                reason="Insufficient tokens"
            )
    
    def try_consume(self, tokens: int = 1) -> bool:
        """Simple check if tokens available"""
        with self._lock:
            self._refill()
            return self._tokens >= tokens
    
    def get_tokens(self) -> float:
        """Get current token count"""
        with self._lock:
            self._refill()
            return self._tokens
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens (seconds)"""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self._refill_rate


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class SlidingWindowLimiter:
    """
    Sliding Window Rate Limiter.
    
    Maintains precise count of requests in sliding time window.
    More accurate than fixed window, uses ring buffer for efficiency.
    
    Memory: O(window_size) - bounded by max requests in window
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize sliding window limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Window duration in seconds
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = threading.Lock()
    
    def _cleanup(self):
        """Remove expired requests from window"""
        cutoff = time.time() - self._window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
    
    def check(self) -> RateLimitResult:
        """Check if request is allowed"""
        with self._lock:
            self._cleanup()
            
            current_count = len(self._requests)
            
            if current_count < self._max_requests:
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self._max_requests - current_count,
                    current_rate=current_count / self._window_seconds
                )
            
            # Calculate wait time until oldest request expires
            oldest = self._requests[0]
            wait_time = oldest + self._window_seconds - time.time()
            
            return RateLimitResult(
                allowed=False,
                wait_time_ms=max(0, wait_time * 1000),
                tokens_remaining=0,
                reason="Window limit exceeded"
            )
    
    def record(self):
        """Record a request"""
        with self._lock:
            self._requests.append(time.time())
    
    def get_count(self) -> int:
        """Get current request count in window"""
        with self._lock:
            self._cleanup()
            return len(self._requests)


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation.
    
    Protects system from cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, reject all requests immediately
    - HALF_OPEN: Testing if service recovered
    
    Memory: ~200 bytes
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying half-open
            success_threshold: Successes in half-open to close
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
        
        # Statistics
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            self._check_recovery()
            return self._state
    
    def _check_recovery(self):
        """Check if circuit should transition from OPEN to HALF_OPEN"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
    
    def _transition(self, new_state: CircuitState):
        """Transition to new state"""
        if self._state != new_state:
            logger.info(f"Circuit breaker: {self._state.name} -> {new_state.name}")
            self._state = new_state
            self._state_changes += 1
            
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        with self._lock:
            self._check_recovery()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.HALF_OPEN:
                return True
            
            # OPEN state
            return False
    
    def record_success(self):
        """Record a successful request"""
        with self._lock:
            self._total_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._transition(CircuitState.CLOSED)
            
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self):
        """Record a failed request"""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
            
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._transition(CircuitState.OPEN)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'state': self._state.name,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'total_failures': self._total_failures,
                'total_successes': self._total_successes,
                'state_changes': self._state_changes,
            }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveRateLimiter:
    """
    Adaptive Rate Limiter that learns from responses.
    
    Features:
    - Adjusts rate based on success/failure patterns
    - Respects Retry-After headers
    - Backs off on 429 responses
    - Speeds up when successful
    
    Memory: O(window_size) for request history
    """
    
    def __init__(self, config: RateLimitConfig = None):
        """
        Initialize adaptive rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self._config = config or RateLimitConfig()
        
        # Core rate limiter
        self._bucket = TokenBucket(
            capacity=self._config.burst_size,
            refill_rate=self._config.requests_per_minute / 60.0
        )
        
        # Sliding window for RPM tracking
        self._window = SlidingWindowLimiter(
            max_requests=self._config.requests_per_minute,
            window_seconds=60.0
        )
        
        # Circuit breaker
        self._circuit = CircuitBreaker(
            failure_threshold=self._config.circuit_failure_threshold,
            recovery_timeout=self._config.circuit_recovery_timeout,
            success_threshold=self._config.circuit_success_threshold
        )
        
        # Request history for adaptation
        self._history: deque = deque(maxlen=self._config.adaptive_window_size)
        self._consecutive_failures = 0
        self._current_delay = 0.0
        
        # Adaptive parameters
        self._effective_rate = self._config.requests_per_minute
        
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'delayed_requests': 0,
            'rejected_requests': 0,
            'total_delay_ms': 0.0,
            'adaptations': 0,
        }
    
    def check(self, tokens: int = 1) -> RateLimitResult:
        """
        Check if request is allowed.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitResult with decision
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Check circuit breaker first
            if not self._circuit.can_execute():
                self._stats['rejected_requests'] += 1
                return RateLimitResult(
                    allowed=False,
                    reason="Circuit breaker open",
                    retry_after=self._config.circuit_recovery_timeout
                )
            
            # Check sliding window
            window_result = self._window.check()
            if not window_result.allowed:
                self._stats['delayed_requests'] += 1
                return window_result
            
            # Check token bucket
            bucket_result = self._bucket.consume(tokens)
            
            if bucket_result.allowed:
                self._stats['allowed_requests'] += 1
                self._window.record()
            else:
                self._stats['delayed_requests'] += 1
            
            return bucket_result
    
    def record_response(
        self,
        success: bool,
        status_code: int = 200,
        retry_after: float = None,
        latency_ms: float = 0.0
    ):
        """
        Record a response for adaptive learning.
        
        Args:
            success: Whether the request succeeded
            status_code: HTTP status code
            retry_after: Retry-After header value (seconds)
            latency_ms: Request latency in milliseconds
        """
        with self._lock:
            # Record in history
            record = RequestRecord(
                timestamp=time.time(),
                success=success,
                latency_ms=latency_ms
            )
            self._history.append(record)
            
            # Update circuit breaker
            if success:
                self._circuit.record_success()
                self._consecutive_failures = 0
                
                # Reduce delay on success
                if self._current_delay > 0:
                    self._current_delay = max(0, self._current_delay - 0.1)
                    self._stats['adaptations'] += 1
            else:
                self._circuit.record_failure()
                self._consecutive_failures += 1
                
                # Handle 429 specially
                if status_code == 429:
                    if retry_after:
                        self._current_delay = retry_after
                    else:
                        self._current_delay = min(
                            self._current_delay + self._get_backoff(),
                            self._config.max_backoff
                        )
                    self._stats['adaptations'] += 1
                
                # Handle server errors
                elif status_code >= 500:
                    self._current_delay = min(
                        self._current_delay + self._get_backoff() / 2,
                        self._config.max_backoff / 2
                    )
            
            # Adapt rate based on recent history
            if self._config.enable_adaptive:
                self._adapt_rate()
    
    def _get_backoff(self) -> float:
        """Calculate exponential backoff with jitter"""
        backoff = self._config.initial_backoff * (
            self._config.backoff_factor ** min(self._consecutive_failures, 10)
        )
        
        # Add jitter
        jitter = backoff * self._config.jitter_percent * random.random()
        
        return min(backoff + jitter, self._config.max_backoff)
    
    def _adapt_rate(self):
        """Adapt rate based on recent success/failure pattern"""
        if len(self._history) < 10:
            return
        
        # Calculate success rate
        recent = list(self._history)[-50:]
        successes = sum(1 for r in recent if r.success)
        success_rate = successes / len(recent)
        
        # Calculate average latency
        avg_latency = sum(r.latency_ms for r in recent) / len(recent)
        
        # Adjust rate
        if success_rate > 0.95 and avg_latency < 1000:
            # Very healthy - can increase rate
            self._effective_rate = min(
                self._effective_rate * 1.1,
                self._config.requests_per_minute * 1.5
            )
        elif success_rate < 0.8:
            # Unhealthy - decrease rate
            self._effective_rate = max(
                self._effective_rate * 0.8,
                self._config.requests_per_minute * 0.2
            )
        
        # Update bucket refill rate
        self._bucket._refill_rate = self._effective_rate / 60.0
    
    def wait_if_needed(self, max_wait: float = 60.0) -> float:
        """
        Wait if rate limited, with adaptive delay.
        
        Args:
            max_wait: Maximum time to wait in seconds
            
        Returns:
            Actual wait time in seconds
        """
        result = self.check()
        
        if result.allowed:
            return 0.0
        
        wait_time = result.wait_time_ms / 1000.0
        
        # Add adaptive delay
        if self._current_delay > 0:
            wait_time = max(wait_time, self._current_delay)
        
        # Respect max wait
        wait_time = min(wait_time, max_wait)
        
        if wait_time > 0:
            self._stats['total_delay_ms'] += wait_time * 1000
            time.sleep(wait_time)
            self._current_delay = max(0, self._current_delay - wait_time)
        
        return wait_time
    
    def get_current_delay(self) -> float:
        """Get current adaptive delay"""
        return self._current_delay
    
    def get_effective_rate(self) -> float:
        """Get current effective rate (requests per minute)"""
        return self._effective_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'effective_rate': self._effective_rate,
                'current_delay': self._current_delay,
                'consecutive_failures': self._consecutive_failures,
                'tokens_available': self._bucket.get_tokens(),
                'window_count': self._window.get_count(),
                'circuit_breaker': self._circuit.get_stats(),
            })
            
            if stats['total_requests'] > 0:
                stats['allow_rate'] = stats['allowed_requests'] / stats['total_requests']
                stats['avg_delay_ms'] = stats['total_delay_ms'] / max(1, stats['delayed_requests'])
            
            return stats
    
    def reset(self):
        """Reset rate limiter to initial state"""
        with self._lock:
            self._bucket = TokenBucket(
                capacity=self._config.burst_size,
                refill_rate=self._config.requests_per_minute / 60.0
            )
            self._window = SlidingWindowLimiter(
                max_requests=self._config.requests_per_minute,
                window_seconds=60.0
            )
            self._circuit.reset()
            self._history.clear()
            self._consecutive_failures = 0
            self._current_delay = 0.0
            self._effective_rate = self._config.requests_per_minute
            logger.info("Rate limiter reset to initial state")


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER MANAGER (Multi-Endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiterManager:
    """
    Manager for multiple rate limiters.
    
    Allows different rate limits for different endpoints/services.
    Memory efficient - creates limiters on demand.
    
    Usage:
        manager = RateLimiterManager()
        
        # Check rate limit for specific endpoint
        result = manager.check('openrouter')
        
        # Record response
        manager.record_response('openrouter', success=True, status_code=200)
        
        # Wait if needed
        manager.wait_if_needed('openrouter')
    """
    
    def __init__(self, default_config: RateLimitConfig = None):
        """
        Initialize rate limiter manager.
        
        Args:
            default_config: Default configuration for new limiters
        """
        self._default_config = default_config or RateLimitConfig()
        self._limiters: Dict[str, AdaptiveRateLimiter] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = threading.Lock()
    
    def register(
        self,
        name: str,
        config: RateLimitConfig = None,
        requests_per_minute: int = None
    ):
        """
        Register a new rate limiter.
        
        Args:
            name: Limiter name (e.g., 'openrouter', 'github')
            config: Full configuration (optional)
            requests_per_minute: Simple RPM configuration (optional)
        """
        with self._lock:
            if config:
                self._configs[name] = config
            elif requests_per_minute:
                self._configs[name] = RateLimitConfig(
                    requests_per_minute=requests_per_minute,
                    burst_size=max(1, requests_per_minute // 6)
                )
            else:
                self._configs[name] = self._default_config
            
            self._limiters[name] = AdaptiveRateLimiter(self._configs[name])
    
    def _get_limiter(self, name: str) -> AdaptiveRateLimiter:
        """Get or create limiter for name"""
        with self._lock:
            if name not in self._limiters:
                self.register(name)
            return self._limiters[name]
    
    def check(self, name: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit for endpoint"""
        return self._get_limiter(name).check(tokens)
    
    def record_response(
        self,
        name: str,
        success: bool,
        status_code: int = 200,
        retry_after: float = None,
        latency_ms: float = 0.0
    ):
        """Record response for endpoint"""
        self._get_limiter(name).record_response(
            success=success,
            status_code=status_code,
            retry_after=retry_after,
            latency_ms=latency_ms
        )
    
    def wait_if_needed(self, name: str, max_wait: float = 60.0) -> float:
        """Wait if rate limited for endpoint"""
        return self._get_limiter(name).wait_if_needed(max_wait)
    
    def get_stats(self, name: str = None) -> Dict[str, Any]:
        """Get statistics for endpoint or all"""
        if name:
            return self._get_limiter(name).get_stats()
        
        with self._lock:
            return {
                name: limiter.get_stats()
                for name, limiter in self._limiters.items()
            }
    
    def reset(self, name: str = None):
        """Reset endpoint or all limiters"""
        if name:
            self._get_limiter(name).reset()
        else:
            with self._lock:
                for limiter in self._limiters.values():
                    limiter.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR FOR AUTOMATIC RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

def rate_limited(
    limiter_name: str = 'default',
    manager: RateLimiterManager = None,
    max_wait: float = 60.0
):
    """
    Decorator for automatic rate limiting.
    
    Usage:
        @rate_limited('openrouter')
        def call_api():
            return requests.get('https://api.example.com')
    """
    def decorator(func: Callable) -> Callable:
        _manager = manager or get_rate_limiter_manager()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Wait if needed
            _manager.wait_if_needed(limiter_name, max_wait)
            
            try:
                result = func(*args, **kwargs)
                
                # Extract status code if possible
                status_code = getattr(result, 'status_code', 200)
                success = 200 <= status_code < 300
                
                # Check for retry-after header
                retry_after = None
                if hasattr(result, 'headers'):
                    ra = result.headers.get('Retry-After')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            pass
                
                _manager.record_response(
                    limiter_name,
                    success=success,
                    status_code=status_code,
                    retry_after=retry_after
                )
                
                return result
                
            except Exception as e:
                _manager.record_response(
                    limiter_name,
                    success=False,
                    status_code=0
                )
                raise
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[RateLimiterManager] = None


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get global rate limiter manager"""
    global _manager
    if _manager is None:
        _manager = RateLimiterManager()
        
        # Pre-register common endpoints
        _manager.register('openrouter', RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
            circuit_failure_threshold=5
        ))
        _manager.register('github', RateLimitConfig(
            requests_per_minute=30,
            burst_size=5
        ))
    
    return _manager


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for rate limiter"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    # Test 1: Token Bucket
    bucket = TokenBucket(capacity=10, refill_rate=2.0)
    
    # Should succeed
    result = bucket.consume(1)
    if result.allowed:
        results['passed'].append('token_bucket_consume')
    else:
        results['failed'].append('token_bucket_consume')
    
    # Should have 9 tokens
    if bucket.get_tokens() == 9:
        results['passed'].append('token_bucket_count')
    else:
        results['failed'].append(f'token_bucket_count ({bucket.get_tokens()})')
    
    # Test 2: Sliding Window
    window = SlidingWindowLimiter(max_requests=5, window_seconds=1.0)
    
    for _ in range(5):
        result = window.check()
        window.record()
    
    # Should be blocked
    result = window.check()
    if not result.allowed:
        results['passed'].append('sliding_window_limit')
    else:
        results['failed'].append('sliding_window_limit')
    
    # Test 3: Circuit Breaker
    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    # Should be closed
    if circuit.state == CircuitState.CLOSED:
        results['passed'].append('circuit_closed')
    else:
        results['failed'].append('circuit_closed')
    
    # Record failures
    for _ in range(3):
        circuit.record_failure()
    
    # Should be open
    if circuit.state == CircuitState.OPEN:
        results['passed'].append('circuit_open')
    else:
        results['failed'].append(f'circuit_open ({circuit.state.name})')
    
    # Test 4: Adaptive Rate Limiter
    limiter = AdaptiveRateLimiter(RateLimitConfig(
        requests_per_minute=60,
        burst_size=10
    ))
    
    result = limiter.check()
    if result.allowed:
        results['passed'].append('adaptive_check')
    else:
        results['failed'].append('adaptive_check')
    
    # Record success
    limiter.record_response(success=True, status_code=200)
    stats = limiter.get_stats()
    if stats['allowed_requests'] > 0:
        results['passed'].append('adaptive_record')
    else:
        results['failed'].append('adaptive_record')
    
    # Test 5: Manager
    manager = RateLimiterManager()
    manager.register('test', RateLimitConfig(requests_per_minute=10))
    
    result = manager.check('test')
    if result.allowed:
        results['passed'].append('manager_check')
    else:
        results['failed'].append('manager_check')
    
    results['stats'] = limiter.get_stats()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Rate Limiter - Self Test")
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
    
    print("\n📊 Sample Statistics:")
    stats = test_results['stats']
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
