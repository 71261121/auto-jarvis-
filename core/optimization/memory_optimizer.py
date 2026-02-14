#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 73: Memory Optimization
==================================================

Comprehensive memory optimization for 4GB RAM devices:
- Lazy loading for all modules
- Generator-based processing
- Memory-efficient data structures
- Garbage collection hints
- Memory profiling and monitoring

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import gc
import os
import sys
import weakref
import threading
import functools
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Iterator, 
    Generator, TypeVar, Generic, Union, Tuple
)
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
from contextlib import contextmanager
import tracemalloc

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    current_mb: float
    peak_mb: float
    available_mb: float
    objects_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'current_mb': self.current_mb,
            'peak_mb': self.peak_mb,
            'available_mb': self.available_mb,
            'objects_count': self.objects_count
        }


class MemoryProfiler:
    """
    Memory profiler for monitoring and analysis
    
    Features:
    - Track memory usage over time
    - Identify memory leaks
    - Generate reports
    - Set memory limits
    """
    
    def __init__(self, enable_tracing: bool = True):
        self.snapshots: List[MemorySnapshot] = []
        self.max_snapshots = 100
        self._lock = threading.Lock()
        self._tracing = False
        
        if enable_tracing:
            self.start_tracing()
    
    def start_tracing(self) -> None:
        """Start memory tracing"""
        if not self._tracing:
            try:
                tracemalloc.start()
                self._tracing = True
            except Exception:
                pass  # May not be available on all platforms
    
    def stop_tracing(self) -> None:
        """Stop memory tracing"""
        if self._tracing:
            try:
                tracemalloc.stop()
                self._tracing = False
            except Exception:
                pass
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: estimate from tracemalloc
            if self._tracing:
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024
            return 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        if self._tracing:
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024
        return self.get_current_memory()
    
    def get_available_memory(self) -> float:
        """Get available system memory in MB"""
        try:
            import psutil
            return psutil.virtual_memory().available / 1024 / 1024
        except ImportError:
            return 2048.0  # Assume 2GB available
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=self.get_current_memory(),
            peak_mb=self.get_peak_memory(),
            available_mb=self.get_available_memory(),
            objects_count=len(gc.get_objects())
        )
        
        with self._lock:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)
        
        return snapshot
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend"""
        if len(self.snapshots) < 2:
            return {'trend': 'insufficient_data'}
        
        first = self.snapshots[0].current_mb
        last = self.snapshots[-1].current_mb
        change = last - first
        change_percent = (change / first * 100) if first > 0 else 0
        
        if change_percent > 10:
            trend = 'increasing'
        elif change_percent < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_mb': change,
            'change_percent': change_percent,
            'first_mb': first,
            'last_mb': last,
            'avg_mb': sum(s.current_mb for s in self.snapshots) / len(self.snapshots),
            'peak_mb': max(s.peak_mb for s in self.snapshots)
        }
    
    def detect_leak(self, threshold_mb: float = 50.0) -> Optional[Dict[str, Any]]:
        """Detect potential memory leak"""
        if len(self.snapshots) < 5:
            return None
        
        # Check if memory consistently increases
        recent = self.snapshots[-5:]
        increasing = all(
            recent[i].current_mb < recent[i+1].current_mb 
            for i in range(len(recent)-1)
        )
        
        if increasing:
            increase = recent[-1].current_mb - recent[0].current_mb
            if increase > threshold_mb:
                return {
                    'detected': True,
                    'increase_mb': increase,
                    'snapshots': [s.to_dict() for s in recent]
                }
        
        return None
    
    def generate_report(self) -> str:
        """Generate memory report"""
        lines = [
            "═══════════════════════════════════════════",
            "       MEMORY USAGE REPORT",
            "═══════════════════════════════════════════",
            f"Current Memory: {self.get_current_memory():.2f} MB",
            f"Peak Memory:    {self.get_peak_memory():.2f} MB",
            f"Available:      {self.get_available_memory():.2f} MB",
            f"Objects:        {len(gc.get_objects()):,}",
            "",
        ]
        
        trend = self.get_memory_trend()
        lines.extend([
            "Trend Analysis:",
            f"  Direction: {trend['trend'].upper()}",
            f"  Average:   {trend['avg_mb']:.2f} MB",
            "",
        ])
        
        leak = self.detect_leak()
        if leak:
            lines.extend([
                "⚠️  MEMORY LEAK DETECTED!",
                f"  Increase: {leak['increase_mb']:.2f} MB",
                ""
            ])
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class LazyLoader:
    """
    Lazy loader for modules and objects
    
    Delays import/creation until first access, reducing memory footprint.
    
    Example:
        heavy_module = LazyLoader('some.heavy.module')
        # Module not loaded yet
        result = heavy_module.some_function()  # Loads now
    """
    
    def __init__(self, module_name: str, import_func: Optional[Callable] = None):
        self._module_name = module_name
        self._import_func = import_func
        self._module = None
        self._loaded = False
        self._load_time: Optional[float] = None
    
    def _load(self) -> Any:
        """Load the module"""
        if not self._loaded:
            start = datetime.now()
            
            if self._import_func:
                self._module = self._import_func()
            else:
                import importlib
                self._module = importlib.import_module(self._module_name)
            
            self._load_time = (datetime.now() - start).total_seconds()
            self._loaded = True
        
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to loaded module"""
        module = self._load()
        return getattr(module, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make lazy loader callable"""
        module = self._load()
        if callable(module):
            return module(*args, **kwargs)
        raise TypeError(f"{self._module_name} is not callable")
    
    @property
    def is_loaded(self) -> bool:
        """Check if module is loaded"""
        return self._loaded
    
    @property
    def load_time(self) -> Optional[float]:
        """Get load time in seconds"""
        return self._load_time
    
    def preload(self) -> None:
        """Force preload the module"""
        self._load()
    
    def unload(self) -> None:
        """Unload the module (if possible)"""
        if self._loaded:
            self._module = None
            self._loaded = False
            gc.collect()


class LazyProperty:
    """
    Lazy property decorator for classes
    
    Example:
        class MyClass:
            @LazyProperty
            def expensive_data(self):
                return self._compute_expensive_data()
    """
    
    def __init__(self, func: Callable):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"
    
    def __get__(self, obj, objtype=None) -> Any:
        if obj is None:
            return self
        
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        
        return getattr(obj, self.attr_name)
    
    def __set__(self, obj, value: Any) -> None:
        setattr(obj, self.attr_name, value)
    
    def __delete__(self, obj) -> None:
        if hasattr(obj, self.attr_name):
            delattr(obj, self.attr_name)


def lazy_import(module_name: str) -> LazyLoader:
    """Convenience function for lazy imports"""
    return LazyLoader(module_name)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryEfficientDict(Generic[K, V]):
    """
    Memory-efficient dictionary with size limits and eviction
    
    Features:
    - Maximum size limit
    - LRU eviction
    - Memory usage tracking
    - Weak key references option
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_memory_mb: float = 10.0,
        use_weak_keys: bool = False
    ):
        self._data: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._max_memory_mb = max_memory_mb
        self._use_weak_keys = use_weak_keys
        self._weak_refs: Dict[int, weakref.ref] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def __getitem__(self, key: K) -> V:
        with self._lock:
            if key in self._data:
                self._hits += 1
                # Move to end (most recently used)
                self._data.move_to_end(key)
                return self._data[key]
            self._misses += 1
            raise KeyError(key)
    
    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                # Update existing
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                # Check limits
                self._check_limits()
                self._data[key] = value
    
    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]
    
    def __contains__(self, key: K) -> bool:
        return key in self._data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def _check_limits(self) -> None:
        """Check and enforce limits"""
        # Size limit
        while len(self._data) >= self._max_size:
            self._data.popitem(last=False)  # Remove oldest
        
        # Memory limit (approximate)
        # Rough estimate: each entry is ~100 bytes overhead
        estimated_mb = len(self._data) * 0.0001  # 100 bytes per entry
        while estimated_mb > self._max_memory_mb:
            self._data.popitem(last=False)
            estimated_mb = len(self._data) * 0.0001
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        try:
            return self[key]
        except KeyError:
            return default
    
    def setdefault(self, key: K, default: V) -> V:
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default
    
    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'size': len(self._data),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }


class MemoryEfficientList(Generic[T]):
    """
    Memory-efficient list with chunking for large datasets
    
    Features:
    - Automatic chunking for large lists
    - Memory-efficient iteration
    - Lazy loading of chunks
    """
    
    def __init__(self, chunk_size: int = 1000):
        self._chunks: List[List[T]] = [[]]
        self._chunk_size = chunk_size
        self._total_size = 0
        self._lock = threading.Lock()
    
    def append(self, item: T) -> None:
        with self._lock:
            if len(self._chunks[-1]) >= self._chunk_size:
                self._chunks.append([])
            self._chunks[-1].append(item)
            self._total_size += 1
    
    def extend(self, items: List[T]) -> None:
        for item in items:
            self.append(item)
    
    def __getitem__(self, index: int) -> T:
        if index < 0:
            index += self._total_size
        if index < 0 or index >= self._total_size:
            raise IndexError(index)
        
        chunk_index = index // self._chunk_size
        item_index = index % self._chunk_size
        return self._chunks[chunk_index][item_index]
    
    def __len__(self) -> int:
        return self._total_size
    
    def __iter__(self) -> Iterator[T]:
        for chunk in self._chunks:
            yield from chunk
    
    def __contains__(self, item: T) -> bool:
        return any(item in chunk for chunk in self._chunks)
    
    def clear(self) -> None:
        with self._lock:
            self._chunks = [[]]
            self._total_size = 0
            gc.collect()
    
    def get_chunk_count(self) -> int:
        return len(self._chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratorProcessor(Generic[T]):
    """
    Process large datasets using generators
    
    Memory-efficient processing that doesn't load everything into memory.
    """
    
    def __init__(self, source: Optional[Generator[T, None, None]] = None):
        self._source = source
        self._processors: List[Callable] = []
    
    def from_iterable(self, iterable: Any) -> 'GeneratorProcessor[T]':
        """Create processor from any iterable"""
        self._source = (item for item in iterable)
        return self
    
    def from_file(self, filepath: str, mode: str = 'r') -> 'GeneratorProcessor[str]':
        """Create processor from file"""
        def file_generator():
            with open(filepath, mode) as f:
                for line in f:
                    yield line
        self._source = file_generator()
        return self
    
    def map(self, func: Callable[[T], Any]) -> 'GeneratorProcessor':
        """Apply transformation"""
        self._processors.append(('map', func))
        return self
    
    def filter(self, predicate: Callable[[T], bool]) -> 'GeneratorProcessor':
        """Filter items"""
        self._processors.append(('filter', predicate))
        return self
    
    def batch(self, size: int) -> 'GeneratorProcessor[List[T]]':
        """Batch items"""
        self._processors.append(('batch', size))
        return self
    
    def execute(self) -> Generator[Any, None, None]:
        """Execute the processing pipeline"""
        if self._source is None:
            raise ValueError("No data source set")
        
        current = self._source
        
        for op, func in self._processors:
            if op == 'map':
                current = (func(item) for item in current)
            elif op == 'filter':
                current = (item for item in current if func(item))
            elif op == 'batch':
                current = self._batch_generator(current, func)
        
        yield from current
    
    def _batch_generator(self, source: Generator, size: int) -> Generator[List, None, None]:
        """Create batches from source"""
        batch = []
        for item in source:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    def to_list(self) -> List[Any]:
        """Convert to list (use with caution for large datasets)"""
        return list(self.execute())
    
    def first(self, n: int = 1) -> List[Any]:
        """Get first n items"""
        results = []
        for i, item in enumerate(self.execute()):
            if i >= n:
                break
            results.append(item)
        return results
    
    def count(self) -> int:
        """Count items (consumes generator)"""
        return sum(1 for _ in self.execute())
    
    def reduce(self, func: Callable[[Any, T], Any], initial: Any = None) -> Any:
        """Reduce items"""
        result = initial
        for item in self.execute():
            if result is None:
                result = item
            else:
                result = func(result, item)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# GARBAGE COLLECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class GCManager:
    """
    Garbage collection manager for optimal memory management
    
    Features:
    - Automatic GC tuning
    - Manual GC hints
    - Generation monitoring
    """
    
    def __init__(self):
        self._original_thresholds = gc.get_threshold()
        self._collections_count = gc.get_count()
    
    def optimize_for_low_memory(self) -> None:
        """Optimize GC for low memory situations"""
        # More aggressive collection
        gc.set_threshold(500, 8, 8)
    
    def optimize_for_speed(self) -> None:
        """Optimize GC for speed (less frequent collection)"""
        gc.set_threshold(10000, 20, 20)
    
    def restore_defaults(self) -> None:
        """Restore original GC thresholds"""
        gc.set_threshold(*self._original_thresholds)
    
    def hint_collection(self, generation: int = 0) -> int:
        """Hint that collection might be beneficial"""
        return gc.collect(geration=generation)
    
    def force_full_collection(self) -> Tuple[int, int, int]:
        """Force full collection of all generations"""
        return (
            gc.collect(0),
            gc.collect(1),
            gc.collect(2)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GC statistics"""
        return {
            'thresholds': gc.get_threshold(),
            'counts': gc.get_count(),
            'enabled': gc.isenabled(),
            'objects': len(gc.get_objects()),
            'garbage': len(gc.garbage)
        }
    
    @contextmanager
    def disabled(self):
        """Context manager for temporarily disabling GC"""
        gc.disable()
        try:
            yield
        finally:
            gc.enable()


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY OPTIMIZER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryOptimizer:
    """
    Main memory optimization controller
    
    Coordinates all memory optimization features:
    - Memory profiling
    - Lazy loading
    - Efficient data structures
    - GC management
    """
    
    def __init__(self, target_memory_mb: float = 500.0):
        self.target_memory_mb = target_memory_mb
        self.profiler = MemoryProfiler()
        self.gc_manager = GCManager()
        self._lazy_modules: Dict[str, LazyLoader] = {}
        self._optimizations_applied = False
    
    def register_lazy_module(self, name: str, module_path: str) -> LazyLoader:
        """Register a module for lazy loading"""
        loader = LazyLoader(module_path)
        self._lazy_modules[name] = loader
        return loader
    
    def get_lazy_module(self, name: str) -> Optional[LazyLoader]:
        """Get registered lazy module"""
        return self._lazy_modules.get(name)
    
    def optimize_for_device(self) -> Dict[str, Any]:
        """Apply optimal settings for current device"""
        available_mb = self.profiler.get_available_memory()
        
        results = {
            'available_memory_mb': available_mb,
            'optimizations': []
        }
        
        # Low memory device (< 1GB available)
        if available_mb < 1024:
            self.gc_manager.optimize_for_low_memory()
            results['optimizations'].append('aggressive_gc')
        else:
            self.gc_manager.restore_defaults()
            results['optimizations'].append('standard_gc')
        
        # Take baseline snapshot
        self.profiler.take_snapshot()
        
        self._optimizations_applied = True
        return results
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if under memory pressure"""
        current_mb = self.profiler.get_current_memory()
        available_mb = self.profiler.get_available_memory()
        
        pressure_ratio = current_mb / self.target_memory_mb
        
        if pressure_ratio > 0.9:
            level = 'critical'
        elif pressure_ratio > 0.7:
            level = 'high'
        elif pressure_ratio > 0.5:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'level': level,
            'current_mb': current_mb,
            'target_mb': self.target_memory_mb,
            'available_mb': available_mb,
            'pressure_ratio': pressure_ratio
        }
    
    def relieve_pressure(self) -> Dict[str, Any]:
        """Attempt to relieve memory pressure"""
        results = {
            'actions_taken': []
        }
        
        # Force garbage collection
        collected = self.gc_manager.force_full_collection()
        results['actions_taken'].append(f'gc_collected_{sum(collected)}_objects')
        
        # Clear unloaded lazy modules
        for name, loader in self._lazy_modules.items():
            if not loader.is_loaded:
                results['actions_taken'].append(f'lazy_{name}_not_loaded')
        
        # Check result
        after_mb = self.profiler.get_current_memory()
        results['memory_after_mb'] = after_mb
        
        return results
    
    def create_efficient_dict(self, max_size: int = 1000) -> MemoryEfficientDict:
        """Create a memory-efficient dictionary"""
        return MemoryEfficientDict(max_size=max_size)
    
    def create_efficient_list(self, chunk_size: int = 1000) -> MemoryEfficientList:
        """Create a memory-efficient list"""
        return MemoryEfficientList(chunk_size=chunk_size)
    
    def create_generator_processor(self) -> GeneratorProcessor:
        """Create a generator processor"""
        return GeneratorProcessor()
    
    def get_report(self) -> str:
        """Get comprehensive memory report"""
        lines = [
            self.profiler.generate_report(),
            "",
            "GC Settings:",
            f"  Thresholds: {self.gc_manager.get_stats()['thresholds']}",
            "",
            f"Lazy Modules: {len(self._lazy_modules)} registered",
            f"  Loaded: {sum(1 for l in self._lazy_modules.values() if l.is_loaded)}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def memory_efficient(func: Callable) -> Callable:
    """Decorator to make function memory-efficient"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Force GC before
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Hint collection after
            gc.collect(0)
        
        return result
    return wrapper


def limit_memory(max_mb: float):
    """Decorator to limit memory usage of function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = MemoryProfiler(enable_tracing=False)
            start_mb = profiler.get_current_memory()
            
            result = func(*args, **kwargs)
            
            end_mb = profiler.get_current_memory()
            used_mb = end_mb - start_mb
            
            if used_mb > max_mb:
                import warnings
                warnings.warn(
                    f"{func.__name__} used {used_mb:.2f}MB (limit: {max_mb}MB)"
                )
            
            return result
        return wrapper
    return decorator


# Global instance for convenience
_memory_optimizer: Optional[MemoryOptimizer] = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer
