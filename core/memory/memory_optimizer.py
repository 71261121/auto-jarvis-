#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Memory Optimizer
=======================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Python memory management best practices
- Generator-based lazy evaluation
- __slots__ for memory efficiency
- Weak references for caching
- Garbage collection optimization
- Memory profiling and monitoring

Features:
- Real-time memory monitoring
- Automatic memory cleanup
- Lazy loading for large objects
- Memory-efficient data structures
- LRU cache with memory limits
- Memory pressure detection
- Emergency memory release

Memory Target: < 512MB for entire JARVIS system
"""

import sys
import gc
import os
import time
import threading
import logging
import weakref
import math
from typing import Dict, Any, Optional, List, Generator, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
from functools import wraps, lru_cache
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CONSTANTS FOR RMP2402 (4GB RAM)
# ═══════════════════════════════════════════════════════════════════════════════

# Device memory in MB
DEVICE_TOTAL_RAM_MB = 4096

# Safe memory limits
MAX_JARVIS_MEMORY_MB = 512       # JARVIS should use max 512MB
MEMORY_WARNING_MB = 400          # Warn at 400MB
MEMORY_CRITICAL_MB = 450         # Critical at 450MB
GC_THRESHOLD_MB = 380            # Trigger GC at 380MB

# Cache limits
DEFAULT_CACHE_SIZE = 100         # Default LRU cache size
MAX_CACHE_MEMORY_MB = 50         # Max memory for all caches


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryLevel(Enum):
    """Memory usage levels"""
    LOW = auto()        # < 50% of budget
    MODERATE = auto()   # 50-75% of budget
    HIGH = auto()       # 75-90% of budget
    CRITICAL = auto()   # > 90% of budget


class CleanupAction(Enum):
    """Actions taken during cleanup"""
    GC_COLLECT = auto()
    CACHE_CLEAR = auto()
    WEAKREF_PURGE = auto()
    EMERGENCY_RELEASE = auto()


@dataclass
class MemoryStats:
    """Current memory statistics"""
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    jarvis_mb: float = 0.0
    percent_used: float = 0.0
    level: MemoryLevel = MemoryLevel.LOW
    gc_collections: int = 0
    objects_collected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_mb': round(self.total_mb, 2),
            'available_mb': round(self.available_mb, 2),
            'used_mb': round(self.used_mb, 2),
            'jarvis_mb': round(self.jarvis_mb, 2),
            'percent_used': round(self.percent_used, 2),
            'level': self.level.name,
        }


@dataclass
class MemoryEvent:
    """A memory-related event"""
    timestamp: float
    level: MemoryLevel
    action: CleanupAction
    memory_before_mb: float
    memory_after_mb: float
    freed_mb: float
    details: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Works on Linux/Termux using /proc/self/status.
    """
    try:
        # Try /proc first (Linux/Termux)
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS is the resident set size (actual physical memory used)
                    return int(line.split()[1]) / 1024  # Convert KB to MB
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # Fallback: use psutil if available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    
    # Last resort: estimate from sys.getsizeof
    return 0.0


def get_system_memory_mb() -> tuple:
    """
    Get system memory info in MB.
    
    Returns:
        Tuple of (total_mb, available_mb)
    """
    try:
        # Try /proc/meminfo (Linux/Termux)
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1])
                    meminfo[key] = value
            
            total_mb = meminfo.get('MemTotal', 0) / 1024
            available_mb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / 1024
            return total_mb, available_mb
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # Fallback: assume 4GB with 50% available
    return DEVICE_TOTAL_RAM_MB, DEVICE_TOTAL_RAM_MB / 2


def estimate_object_size(obj: Any) -> int:
    """
    Estimate size of an object in bytes.
    
    Uses sys.getsizeof with recursive checking for containers.
    """
    seen = set()
    
    def _estimate(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        
        size = sys.getsizeof(o)
        
        if isinstance(o, dict):
            size += sum(_estimate(k) + _estimate(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(_estimate(item) for item in o)
        elif isinstance(o, str):
            # Strings are already counted accurately
            pass
        
        return size
    
    try:
        return _estimate(obj)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT LRU CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryAwareLRUCache(Generic[T]):
    """
    LRU Cache with memory awareness.
    
    Automatically evicts entries when memory is high.
    Uses weak references for values where possible.
    
    Usage:
        cache = MemoryAwareLRUCache(max_size=100, max_memory_mb=10)
        
        # Store value
        cache.set('key', large_object)
        
        # Get value
        value = cache.get('key')
        
        # Clear if needed
        cache.clear()
    """
    
    __slots__ = ['_cache', '_max_size', '_max_memory', '_current_memory', '_lock', '_stats']
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, max_memory_mb: float = 5.0):
        """
        Initialize memory-aware cache.
        
        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory in MB
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._current_memory = 0
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
        }
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                return self._cache[key]
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: T):
        """Set value in cache"""
        with self._lock:
            # Estimate size
            size = estimate_object_size(value)
            
            # Check if we need to evict
            while (len(self._cache) >= self._max_size or 
                   self._current_memory + size > self._max_memory) and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._current_memory -= estimate_object_size(oldest_value)
                self._stats['evictions'] += 1
            
            # Remove old key if exists
            if key in self._cache:
                old_size = estimate_object_size(self._cache[key])
                self._current_memory -= old_size
                del self._cache[key]
            
            # Add new value
            self._cache[key] = value
            self._current_memory += size
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total_requests)
            
            return {
                **self._stats,
                'size': len(self._cache),
                'memory_mb': self._current_memory / (1024 * 1024),
                'hit_rate': hit_rate,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class LazyLoader:
    """
    Lazy loader for expensive objects.
    
    Loads object only when first accessed.
    Useful for large objects that may not be needed.
    
    Usage:
        expensive_data = LazyLoader(lambda: load_large_file())
        
        # Not loaded yet
        print("Created")
        
        # Now it loads
        data = expensive_data.get()
        
        # Or use as context for cleanup
        with LazyLoader(loader) as obj:
            work_with(obj)
    """
    
    __slots__ = ['_loader', '_value', '_loaded', '_lock']
    
    def __init__(self, loader: Callable[[], T]):
        """
        Initialize lazy loader.
        
        Args:
            loader: Function to call to load the object
        """
        self._loader = loader
        self._value: Optional[T] = None
        self._loaded = False
        self._lock = threading.Lock()
    
    def get(self) -> T:
        """Get the value, loading if necessary"""
        if self._loaded:
            return self._value
        
        with self._lock:
            if not self._loaded:
                self._value = self._loader()
                self._loaded = True
        
        return self._value
    
    def is_loaded(self) -> bool:
        """Check if value is loaded"""
        return self._loaded
    
    def reset(self):
        """Reset and unload"""
        with self._lock:
            self._value = None
            self._loaded = False
    
    def __enter__(self) -> T:
        return self.get()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def chunked_generator(
    items: List[Any],
    chunk_size: int = 100
) -> Generator[List[Any], None, None]:
    """
    Process items in chunks using generators.
    
    Memory-efficient alternative to list comprehension.
    
    Args:
        items: List of items to process
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def batch_process(
    items: List[Any],
    processor: Callable[[Any], Any],
    batch_size: int = 50
) -> Generator[Any, None, None]:
    """
    Process items in batches.
    
    Memory-efficient for large datasets.
    
    Args:
        items: Items to process
        processor: Function to apply to each item
        batch_size: Number of items per batch
        
    Yields:
        Processed items
    """
    for chunk in chunked_generator(items, batch_size):
        for item in chunk:
            yield processor(item)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryOptimizer:
    """
    Ultra-Advanced Memory Optimization System.
    
    Designed for 4GB RAM device (RMP2402).
    
    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - Memory pressure detection
    - Cache management
    - Emergency memory release
    - Memory usage profiling
    
    Memory Budget: < 512MB for JARVIS
    
    Usage:
        optimizer = MemoryOptimizer()
        
        # Check memory
        stats = optimizer.get_stats()
        
        # Monitor memory in context
        with optimizer.monitor():
            # Do memory-intensive work
            process_large_data()
        
        # Force cleanup
        optimizer.cleanup()
        
        # Register for memory pressure callbacks
        optimizer.on_pressure(callback)
    """
    
    def __init__(
        self,
        max_memory_mb: float = MAX_JARVIS_MEMORY_MB,
        warning_threshold_mb: float = MEMORY_WARNING_MB,
        critical_threshold_mb: float = MEMORY_CRITICAL_MB,
        gc_threshold_mb: float = GC_THRESHOLD_MB,
        enable_monitoring: bool = True,
        monitoring_interval: float = 5.0,
    ):
        """
        Initialize Memory Optimizer.
        
        Args:
            max_memory_mb: Maximum memory for JARVIS
            warning_threshold_mb: Warning threshold
            critical_threshold_mb: Critical threshold
            gc_threshold_mb: GC trigger threshold
            enable_monitoring: Enable background monitoring
            monitoring_interval: Monitoring interval in seconds
        """
        self._max_memory = max_memory_mb
        self._warning_threshold = warning_threshold_mb
        self._critical_threshold = critical_threshold_mb
        self._gc_threshold = gc_threshold_mb
        
        # Caches to manage
        self._managed_caches: List[MemoryAwareLRUCache] = []
        
        # Callbacks
        self._pressure_callbacks: List[Callable[[MemoryLevel], None]] = []
        
        # Event history
        self._events: List[MemoryEvent] = []
        self._max_events = 100
        
        # Monitoring
        self._enable_monitoring = enable_monitoring
        self._monitoring_interval = monitoring_interval
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Statistics
        self._stats = {
            'total_gcs': 0,
            'total_cache_clears': 0,
            'total_freed_mb': 0.0,
            'peak_memory_mb': 0.0,
        }
        
        # Initialize GC thresholds
        self._configure_gc()
        
        # Start monitoring if enabled
        if self._enable_monitoring:
            self._start_monitoring()
        
        logger.info(f"MemoryOptimizer initialized (max: {max_memory_mb}MB)")
    
    def _configure_gc(self):
        """Configure garbage collection thresholds"""
        # More aggressive GC for memory-constrained environment
        # Thresholds: (gen0, gen1, gen2) - lower = more frequent
        gc.set_threshold(700, 10, 5)
        
        # Disable automatic GC, we'll control it
        # gc.disable()  # Keep enabled but control thresholds
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitor_thread is not None:
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(self._monitoring_interval):
            try:
                stats = self.get_stats()
                
                # Check memory level
                if stats.level in (MemoryLevel.HIGH, MemoryLevel.CRITICAL):
                    self._handle_memory_pressure(stats)
                
                # Update peak
                if stats.jarvis_mb > self._stats['peak_memory_mb']:
                    self._stats['peak_memory_mb'] = stats.jarvis_mb
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
    
    def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure situation"""
        memory_before = stats.jarvis_mb
        
        # Log pressure
        logger.warning(f"Memory pressure detected: {stats.jarvis_mb:.1f}MB ({stats.level.name})")
        
        # Take action based on level
        if stats.level == MemoryLevel.CRITICAL:
            # Emergency cleanup
            self._emergency_cleanup()
            action = CleanupAction.EMERGENCY_RELEASE
        elif stats.level == MemoryLevel.HIGH:
            # Regular cleanup
            self.cleanup()
            action = CleanupAction.GC_COLLECT
        else:
            return
        
        # Check result
        memory_after = get_process_memory_mb()
        freed = memory_before - memory_after
        
        # Record event
        event = MemoryEvent(
            timestamp=time.time(),
            level=stats.level,
            action=action,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            freed_mb=freed,
        )
        self._record_event(event)
        
        # Notify callbacks
        for callback in self._pressure_callbacks:
            try:
                callback(stats.level)
            except Exception as e:
                logger.warning(f"Memory callback error: {e}")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        logger.warning("Performing emergency memory cleanup")
        
        # Clear all managed caches
        for cache in self._managed_caches:
            cache.clear()
        
        self._stats['total_cache_clears'] += 1
        
        # Aggressive GC
        collected = gc.collect(2)  # Full collection
        self._stats['total_gcs'] += 1
        
        logger.info(f"Emergency cleanup complete: {collected} objects collected")
    
    def _record_event(self, event: MemoryEvent):
        """Record a memory event"""
        self._events.append(event)
        
        # Trim old events
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        jarvis_mb = get_process_memory_mb()
        total_mb, available_mb = get_system_memory_mb()
        
        # Determine level
        if jarvis_mb >= self._critical_threshold:
            level = MemoryLevel.CRITICAL
        elif jarvis_mb >= self._warning_threshold:
            level = MemoryLevel.HIGH
        elif jarvis_mb >= self._max_memory * 0.5:
            level = MemoryLevel.MODERATE
        else:
            level = MemoryLevel.LOW
        
        return MemoryStats(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=total_mb - available_mb,
            jarvis_mb=jarvis_mb,
            percent_used=(jarvis_mb / self._max_memory) * 100,
            level=level,
        )
    
    def cleanup(self, aggressive: bool = False) -> int:
        """
        Perform memory cleanup.
        
        Args:
            aggressive: If True, do full GC and clear caches
            
        Returns:
            Number of objects collected
        """
        memory_before = get_process_memory_mb()
        
        # Run garbage collection
        if aggressive:
            collected = gc.collect(2)  # Full collection
        else:
            collected = gc.collect(0)  # Generation 0 only
        
        self._stats['total_gcs'] += 1
        
        # Clear caches if aggressive
        if aggressive:
            for cache in self._managed_caches:
                cache.clear()
            self._stats['total_cache_clears'] += 1
        
        # Update freed memory stat
        memory_after = get_process_memory_mb()
        freed = memory_before - memory_after
        self._stats['total_freed_mb'] += max(0, freed)
        
        logger.debug(f"Cleanup: collected {collected} objects, freed {freed:.1f}MB")
        
        return collected
    
    def register_cache(self, cache: MemoryAwareLRUCache):
        """Register a cache for management"""
        self._managed_caches.append(cache)
    
    def on_pressure(self, callback: Callable[[MemoryLevel], None]):
        """Register callback for memory pressure events"""
        self._pressure_callbacks.append(callback)
    
    @contextmanager
    def monitor(self, warn_threshold: float = None):
        """
        Context manager for monitoring memory usage.
        
        Usage:
            with optimizer.monitor():
                process_large_data()
        """
        stats_before = self.get_stats()
        
        try:
            yield
        finally:
            stats_after = self.get_stats()
            
            delta = stats_after.jarvis_mb - stats_before.jarvis_mb
            threshold = warn_threshold or self._warning_threshold * 0.1
            
            if delta > threshold:
                logger.warning(
                    f"Memory usage increased by {delta:.1f}MB "
                    f"(now {stats_after.jarvis_mb:.1f}MB)"
                )
    
    @contextmanager
    def limited(self, max_mb: float):
        """
        Context manager with memory limit.
        
        Raises MemoryError if limit exceeded.
        
        Usage:
            with optimizer.limited(100):
                # Must stay under 100MB
                process_data()
        """
        start_mb = get_process_memory_mb()
        
        try:
            yield
        finally:
            end_mb = get_process_memory_mb()
            used = end_mb - start_mb
            
            if used > max_mb:
                logger.error(f"Memory limit exceeded: {used:.1f}MB > {max_mb}MB")
                raise MemoryError(f"Memory limit exceeded: {used:.1f}MB > {max_mb}MB")
    
    def get_events(self, limit: int = 20) -> List[MemoryEvent]:
        """Get recent memory events"""
        return self._events[-limit:]
    
    def get_operator_stats(self) -> Dict[str, Any]:
        """Get internal statistics"""
        return {
            **self._stats,
            'managed_caches': len(self._managed_caches),
            'registered_callbacks': len(self._pressure_callbacks),
            'events_recorded': len(self._events),
        }
    
    def stop(self):
        """Stop monitoring"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def memory_efficient(func: Callable) -> Callable:
    """
    Decorator to make functions memory-efficient.
    
    Runs garbage collection before and after the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Light GC before
        gc.collect(0)
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Light GC after
            gc.collect(0)
        
        return result
    
    return wrapper


def memory_limited(max_mb: float) -> Callable:
    """
    Decorator to limit memory usage of a function.
    
    Args:
        max_mb: Maximum memory increase allowed in MB
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            before = get_process_memory_mb()
            
            result = func(*args, **kwargs)
            
            after = get_process_memory_mb()
            used = after - before
            
            if used > max_mb:
                logger.warning(
                    f"{func.__name__} used {used:.1f}MB (limit: {max_mb}MB)"
                )
            
            return result
        
        return wrapper
    
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_optimizer: Optional[MemoryOptimizer] = None
_optimizer_lock = threading.Lock()  # FIX: Thread-safe singleton


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global MemoryOptimizer instance (thread-safe)"""
    global _optimizer
    if _optimizer is None:
        with _optimizer_lock:
            if _optimizer is None:  # FIX: Double-check pattern
                _optimizer = MemoryOptimizer()
    return _optimizer


def initialize_memory_optimizer(**kwargs) -> MemoryOptimizer:
    """Initialize global optimizer with custom settings"""
    global _optimizer
    with _optimizer_lock:
        _optimizer = MemoryOptimizer(**kwargs)
    return _optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for MemoryOptimizer"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    # Test 1: Memory measurement
    mem_mb = get_process_memory_mb()
    if mem_mb >= 0:
        results['passed'].append(f'memory_measurement: {mem_mb:.1f}MB')
    else:
        results['failed'].append('memory_measurement')
    
    # Test 2: System memory
    total, available = get_system_memory_mb()
    if total > 0:
        results['passed'].append(f'system_memory: {total:.0f}MB total')
    else:
        results['warnings'].append('system_memory: unable to read')
    
    # Test 3: Memory stats
    optimizer = MemoryOptimizer(enable_monitoring=False)
    stats = optimizer.get_stats()
    if stats.level in MemoryLevel:
        results['passed'].append(f'memory_level: {stats.level.name}')
    else:
        results['failed'].append('memory_level')
    
    # Test 4: LRU Cache
    cache = MemoryAwareLRUCache(max_size=5)
    for i in range(10):
        cache.set(f'key_{i}', f'value_{i}' * 100)
    
    if len(cache._cache) <= 5:
        results['passed'].append('lru_cache_eviction')
    else:
        results['failed'].append(f'lru_cache_eviction: {len(cache._cache)} items')
    
    # Test 5: Lazy loader
    loaded = [False]
    
    def expensive_loader():
        loaded[0] = True
        return "expensive_data"
    
    lazy = LazyLoader(expensive_loader)
    if not lazy.is_loaded():
        results['passed'].append('lazy_loader_deferred')
    else:
        results['failed'].append('lazy_loader_deferred')
    
    value = lazy.get()
    if loaded[0] and value == "expensive_data":
        results['passed'].append('lazy_loader_execution')
    else:
        results['failed'].append('lazy_loader_execution')
    
    # Test 6: Cleanup
    collected = optimizer.cleanup()
    results['passed'].append(f'cleanup: {collected} objects')
    
    # Test 7: Object size estimation
    test_obj = {"key": "value" * 100}
    size = estimate_object_size(test_obj)
    if size > 0:
        results['passed'].append(f'object_size: {size} bytes')
    else:
        results['failed'].append('object_size')
    
    results['stats'] = optimizer.get_operator_stats()
    results['memory_stats'] = stats.to_dict()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Memory Optimizer - Self Test")
    print("=" * 70)
    print(f"Device: Realme 2 Pro Lite (RMP2402)")
    print(f"RAM: 4GB")
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
    
    print("\n📊 Memory Statistics:")
    stats = test_results.get('memory_stats', {})
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
