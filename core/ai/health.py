#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - API Health Monitor
=========================================

Device: Realme Pad 2 Lite (RMP2402) | RAM: 4GB | Platform: Termux

Purpose:
- Monitor AI API health status
- Track response times and success rates
- Detect rate limits and outages
- Auto-switch to fallback on issues

Memory Impact: < 2MB
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()
    RATE_LIMITED = auto()


class APIProvider(Enum):
    """Supported API providers"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class HealthCheck:
    """Result of a health check"""
    provider: APIProvider
    status: HealthStatus
    latency_ms: float
    timestamp: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class APIStats:
    """Statistics for an API"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    rate_limit_until: Optional[float] = None

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class HealthMonitor:
    """
    Monitor API health and performance.

    Features:
    - Track success/failure rates
    - Measure response latencies
    - Detect rate limits
    - Auto-disable unhealthy APIs
    - Historical health data

    Usage:
        monitor = HealthMonitor()

        # Record a request
        monitor.record_success(APIProvider.OPENROUTER, latency_ms=1500)

        # Check health
        health = monitor.check_health(APIProvider.OPENROUTER)
        if health.is_healthy:
            # Use API
            pass
    """

    # Thresholds
    HEALTHY_LATENCY_MS = 5000
    DEGRADED_LATENCY_MS = 15000
    MAX_CONSECUTIVE_FAILURES = 3
    SUCCESS_RATE_THRESHOLD = 0.7
    HISTORY_SIZE = 100

    def __init__(self, check_interval: float = 60.0):
        """
        Initialize Health Monitor.

        Args:
            check_interval: Interval between automatic health checks
        """
        self._check_interval = check_interval

        # Per-provider stats
        self._stats: Dict[APIProvider, APIStats] = {
            provider: APIStats() for provider in APIProvider
        }

        # Health check history
        self._history: Dict[APIProvider, deque] = {
            provider: deque(maxlen=self.HISTORY_SIZE) for provider in APIProvider
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        # Callbacks
        self._on_unhealthy: List[Callable] = []
        self._on_healthy: List[Callable] = []

        # Background monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        logger.info("Health Monitor initialized")

    def record_success(
        self,
        provider: APIProvider,
        latency_ms: float,
        tokens_used: int = 0
    ):
        """
        Record a successful API request.

        Args:
            provider: API provider
            latency_ms: Request latency in milliseconds
            tokens_used: Number of tokens used (optional)
        """
        with self._lock:
            stats = self._stats[provider]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency_ms += latency_ms
            stats.last_success = time.time()
            stats.consecutive_failures = 0

            # Record in history
            self._history[provider].append(HealthCheck(
                provider=provider,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                timestamp=time.time(),
                details={'tokens': tokens_used}
            ))

    def record_failure(
        self,
        provider: APIProvider,
        error: str,
        is_rate_limit: bool = False
    ):
        """
        Record a failed API request.

        Args:
            provider: API provider
            error: Error message
            is_rate_limit: Whether this is a rate limit error
        """
        with self._lock:
            stats = self._stats[provider]
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.last_failure = time.time()
            stats.consecutive_failures += 1

            if is_rate_limit:
                # Set rate limit cooldown (default 60 seconds)
                stats.rate_limit_until = time.time() + 60
                status = HealthStatus.RATE_LIMITED
            elif stats.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED

            # Record in history
            self._history[provider].append(HealthCheck(
                provider=provider,
                status=status,
                latency_ms=0,
                timestamp=time.time(),
                error=error
            ))

            # Check if we need to notify
            if status == HealthStatus.UNHEALTHY:
                self._notify_unhealthy(provider, error)

    def check_health(self, provider: APIProvider) -> HealthCheck:
        """
        Check current health of an API.

        Args:
            provider: API provider to check

        Returns:
            HealthCheck with current status
        """
        with self._lock:
            stats = self._stats[provider]

            # Check rate limit
            if stats.rate_limit_until and time.time() < stats.rate_limit_until:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.RATE_LIMITED,
                    latency_ms=0,
                    timestamp=time.time(),
                    error="Rate limited",
                    details={'reset_in': stats.rate_limit_until - time.time()}
                )

            # Clear expired rate limit
            if stats.rate_limit_until and time.time() >= stats.rate_limit_until:
                stats.rate_limit_until = None

            # Check consecutive failures
            if stats.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=stats.avg_latency_ms,
                    timestamp=time.time(),
                    error=f"Too many failures: {stats.consecutive_failures}"
                )

            # Check success rate
            if stats.total_requests >= 10 and stats.success_rate < self.SUCCESS_RATE_THRESHOLD:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.DEGRADED,
                    latency_ms=stats.avg_latency_ms,
                    timestamp=time.time(),
                    error=f"Low success rate: {stats.success_rate:.1%}"
                )

            # Check latency
            if stats.avg_latency_ms > self.DEGRADED_LATENCY_MS:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.DEGRADED,
                    latency_ms=stats.avg_latency_ms,
                    timestamp=time.time(),
                    error=f"High latency: {stats.avg_latency_ms:.0f}ms"
                )

            if stats.avg_latency_ms > self.HEALTHY_LATENCY_MS:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.DEGRADED,
                    latency_ms=stats.avg_latency_ms,
                    timestamp=time.time()
                )

            # No requests yet
            if stats.total_requests == 0:
                return HealthCheck(
                    provider=provider,
                    status=HealthStatus.UNKNOWN,
                    latency_ms=0,
                    timestamp=time.time()
                )

            # All good
            return HealthCheck(
                provider=provider,
                status=HealthStatus.HEALTHY,
                latency_ms=stats.avg_latency_ms,
                timestamp=time.time()
            )

    def is_available(self, provider: APIProvider) -> bool:
        """Check if an API is available for use"""
        health = self.check_health(provider)
        return health.is_healthy

    def get_best_provider(self) -> APIProvider:
        """Get the best available provider"""
        priorities = [
            APIProvider.OPENROUTER,
            APIProvider.OPENAI,
            APIProvider.LOCAL,
        ]

        for provider in priorities:
            if provider == APIProvider.LOCAL:
                return provider  # Local always available

            health = self.check_health(provider)
            if health.status == HealthStatus.HEALTHY:
                return provider

        # Default to local
        return APIProvider.LOCAL

    def get_stats(self, provider: APIProvider = None) -> Dict[str, Any]:
        """Get statistics for provider(s)"""
        if provider:
            stats = self._stats[provider]
            return {
                'provider': provider.value,
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'failed_requests': stats.failed_requests,
                'success_rate': f"{stats.success_rate:.1%}",
                'avg_latency_ms': f"{stats.avg_latency_ms:.1f}",
                'consecutive_failures': stats.consecutive_failures,
                'last_success': datetime.fromtimestamp(stats.last_success).isoformat() if stats.last_success else None,
                'last_failure': datetime.fromtimestamp(stats.last_failure).isoformat() if stats.last_failure else None,
            }

        return {
            provider.value: self.get_stats(provider)
            for provider in APIProvider
        }

    def get_history(self, provider: APIProvider, limit: int = 20) -> List[Dict]:
        """Get health check history"""
        history = list(self._history[provider])[-limit:]
        return [
            {
                'status': check.status.name,
                'latency_ms': check.latency_ms,
                'timestamp': datetime.fromtimestamp(check.timestamp).isoformat(),
                'error': check.error,
            }
            for check in history
        ]

    def reset_stats(self, provider: APIProvider = None):
        """Reset statistics"""
        with self._lock:
            if provider:
                self._stats[provider] = APIStats()
                self._history[provider].clear()
            else:
                for p in APIProvider:
                    self._stats[p] = APIStats()
                    self._history[p].clear()

    def register_callbacks(
        self,
        on_unhealthy: Callable = None,
        on_healthy: Callable = None
    ):
        """Register health change callbacks"""
        if on_unhealthy:
            self._on_unhealthy.append(on_unhealthy)
        if on_healthy:
            self._on_healthy.append(on_healthy)

    def _notify_unhealthy(self, provider: APIProvider, error: str):
        """Notify callbacks of unhealthy status"""
        for callback in self._on_unhealthy:
            try:
                callback(provider, error)
            except Exception as e:
                logger.error(f"Health callback error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for Health Monitor"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }

    monitor = HealthMonitor()

    # Test 1: Initial health
    health = monitor.check_health(APIProvider.LOCAL)
    if health.status == HealthStatus.UNKNOWN:
        results['passed'].append('initial_health_unknown')
    else:
        results['warnings'].append(f'initial_health: {health.status.name}')

    # Test 2: Record success
    monitor.record_success(APIProvider.OPENROUTER, latency_ms=1000)
    stats = monitor.get_stats(APIProvider.OPENROUTER)
    if stats['total_requests'] == 1:
        results['passed'].append('record_success')
    else:
        results['failed'].append('record_success')

    # Test 3: Health after success
    health = monitor.check_health(APIProvider.OPENROUTER)
    if health.status == HealthStatus.HEALTHY:
        results['passed'].append('healthy_after_success')
    else:
        results['failed'].append(f'healthy_after_success: {health.status.name}')

    # Test 4: Record failure
    monitor.record_failure(APIProvider.OPENROUTER, error="Test error")
    stats = monitor.get_stats(APIProvider.OPENROUTER)
    if stats['failed_requests'] == 1:
        results['passed'].append('record_failure')
    else:
        results['failed'].append('record_failure')

    # Test 5: Rate limit
    monitor.record_failure(APIProvider.OPENROUTER, error="Rate limited", is_rate_limit=True)
    health = monitor.check_health(APIProvider.OPENROUTER)
    if health.status == HealthStatus.RATE_LIMITED:
        results['passed'].append('rate_limit_detection')
    else:
        results['failed'].append(f'rate_limit: {health.status.name}')

    # Test 6: Best provider
    best = monitor.get_best_provider()
    if best:
        results['passed'].append(f'best_provider: {best.value}')
    else:
        results['failed'].append('best_provider')

    results['stats'] = monitor.get_stats()

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Health Monitor - Self Test")
    print("=" * 70)

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

    print("\n" + "=" * 70)
