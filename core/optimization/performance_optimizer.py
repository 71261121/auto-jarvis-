#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 74: Performance Optimization
======================================================

Performance optimization for optimal responsiveness:
- Async I/O where beneficial
- Caching strategies
- Connection pooling
- Request batching

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import asyncio
import functools
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import concurrent.futures

T = TypeVar('T')


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    ops_per_second: float


class AsyncIOManager:
    """Async I/O manager for efficient operations"""
    
    def __init__(self, max_workers: int = 4):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
    
    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def run_async(self, coro):
        """Run async coroutine synchronously"""
        loop = self.get_loop()
        return loop.run_until_complete(coro)
    
    def run_in_executor(self, func: Callable, *args, **kwargs):
        """Run blocking function in executor"""
        loop = self.get_loop()
        return loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
    
    async def gather_with_concurrency(self, n: int, *tasks):
        """Run tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(n)
        
        async def sem_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[sem_task(t) for t in tasks])


class ConnectionPool(Generic[T]):
    """Generic connection pool"""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 10):
        self._factory = factory
        self._max_size = max_size
        self._pool: deque = deque(maxlen=max_size)
        self._in_use: int = 0
        self._lock = threading.Lock()
        self._created: int = 0
    
    def acquire(self) -> T:
        """Acquire a connection"""
        with self._lock:
            if self._pool:
                self._in_use += 1
                return self._pool.pop()
            
            if self._created < self._max_size:
                self._created += 1
                self._in_use += 1
                return self._factory()
        
        # Wait for available connection
        while True:
            time.sleep(0.01)
            with self._lock:
                if self._pool:
                    self._in_use += 1
                    return self._pool.pop()
    
    def release(self, conn: T) -> None:
        """Release a connection"""
        with self._lock:
            self._in_use -= 1
            self._pool.append(conn)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'available': len(self._pool),
            'in_use': self._in_use,
            'created': self._created,
            'max_size': self._max_size
        }


class RequestBatcher(Generic[T]):
    """Batch requests for efficient processing"""
    
    def __init__(
        self,
        processor: Callable[[List[T]], List[Any]],
        batch_size: int = 100,
        timeout_ms: int = 100
    ):
        self._processor = processor
        self._batch_size = batch_size
        self._timeout_ms = timeout_ms
        self._pending: List[T] = []
        self._results: Dict[int, Any] = {}
        self._lock = threading.Lock()
        self._request_id = 0
    
    def add(self, item: T) -> int:
        """Add item to batch"""
        with self._lock:
            request_id = self._request_id
            self._request_id += 1
            self._pending.append((request_id, item))
            
            if len(self._pending) >= self._batch_size:
                self._process_batch()
        
        return request_id
    
    def _process_batch(self) -> None:
        """Process current batch"""
        if not self._pending:
            return
        
        batch = self._pending[:]
        self._pending.clear()
        
        items = [item for _, item in batch]
        ids = [rid for rid, _ in batch]
        
        try:
            results = self._processor(items)
            for rid, result in zip(ids, results):
                self._results[rid] = result
        except Exception as e:
            for rid in ids:
                self._results[rid] = e
    
    def get_result(self, request_id: int, timeout: float = 5.0) -> Any:
        """Get result for request"""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if request_id in self._results:
                    return self._results.pop(request_id)
            time.sleep(0.01)
        raise TimeoutError(f"Request {request_id} timed out")


class BenchmarkRunner:
    """Run performance benchmarks"""
    
    def __init__(self, warmup: int = 3):
        self._warmup = warmup
        self._results: List[BenchmarkResult] = []
    
    def benchmark(
        self,
        func: Callable,
        name: Optional[str] = None,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Run benchmark"""
        name = name or func.__name__
        
        # Warmup
        for _ in range(self._warmup):
            func()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=sum(times),
            avg_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            ops_per_second=iterations / sum(times)
        )
        
        self._results.append(result)
        return result
    
    def compare(self) -> str:
        """Compare all benchmarks"""
        if not self._results:
            return "No benchmarks run"
        
        lines = [
            f"{'Name':<30} {'Avg':>10} {'Min':>10} {'Max':>10} {'Ops/s':>12}",
            "-" * 75
        ]
        
        for r in sorted(self._results, key=lambda x: x.avg_time):
            lines.append(
                f"{r.name:<30} {r.avg_time*1000:>9.3f}ms {r.min_time*1000:>9.3f}ms "
                f"{r.max_time*1000:>9.3f}ms {r.ops_per_second:>11.0f}"
            )
        
        return "\n".join(lines)


class PerformanceOptimizer:
    """Main performance optimization controller"""
    
    def __init__(self):
        self.async_manager = AsyncIOManager()
        self._pools: Dict[str, ConnectionPool] = {}
        self._batchers: Dict[str, RequestBatcher] = {}
        self.benchmark_runner = BenchmarkRunner()
    
    def create_pool(self, name: str, factory: Callable, max_size: int = 10) -> ConnectionPool:
        """Create a connection pool"""
        pool = ConnectionPool(factory, max_size)
        self._pools[name] = pool
        return pool
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool"""
        return self._pools.get(name)
    
    def create_batcher(
        self,
        name: str,
        processor: Callable,
        batch_size: int = 100
    ) -> RequestBatcher:
        """Create a request batcher"""
        batcher = RequestBatcher(processor, batch_size)
        self._batchers[name] = batcher
        return batcher
    
    def benchmark(self, func: Callable, iterations: int = 100) -> BenchmarkResult:
        """Run a benchmark"""
        return self.benchmark_runner.benchmark(func, iterations=iterations)


# Decorators
def cached(ttl_seconds: int = 60):
    """Caching decorator"""
    cache: Dict = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        
        return wrapper
    return decorator


def timed(func: Callable) -> Callable:
    """Timing decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper
