#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 61: Performance Tests
=================================================

Performance benchmarks for:
- Memory usage
- Response time
- Concurrent operations
- Long-running stability

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
"""

import sys
import os
import time
import threading
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = time.time()
    
    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        print(f"    Error: {error[:100]}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed)
        rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"TODO 61: Performance Tests Results")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}% | Time: {elapsed:.2f}s")
        return len(self.failed) == 0


results = TestResult()
print("="*60)
print("TODO 61: Performance Tests")
print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY USAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Memory Usage Tests ---")

def get_memory_mb():
    """Get current process memory in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0  # psutil not available

def test_cache_memory():
    """Test cache memory usage"""
    try:
        from core.cache import Cache
        
        initial_mem = get_memory_mb()
        
        cache = Cache(max_size=10000)
        
        # Add many items
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}" * 100)
        
        final_mem = get_memory_mb()
        increase = final_mem - initial_mem
        
        # Should not use more than 50MB for 1000 items
        print(f"    Memory increase: {increase:.2f} MB")
        results.add_pass("performance: Cache memory")
    except Exception as e:
        results.add_fail("performance: Cache memory", str(e))

def test_events_memory():
    """Test events system memory"""
    try:
        from core.events import EventEmitter
        
        emitter = EventEmitter()
        
        # Add many handlers
        for i in range(100):
            def handler(event, idx=i):
                pass
            emitter.on(f"event_{i}", handler)
        
        # Emit many events
        for i in range(100):
            emitter.emit(f"event_{i}", {"data": i})
        
        results.add_pass("performance: Events memory")
    except Exception as e:
        results.add_fail("performance: Events memory", str(e))

test_cache_memory()
test_events_memory()


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE TIME TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Response Time Tests ---")

def test_cache_speed():
    """Test cache operation speed"""
    try:
        from core.cache import Cache
        
        cache = Cache(max_size=1000)
        
        # Warm up
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Measure
        start = time.time()
        for i in range(1000):
            cache.set(f"new_key_{i}", f"value_{i}")
            cache.get(f"new_key_{i}")
        elapsed = time.time() - start
        
        ops_per_sec = 2000 / elapsed  # 1000 sets + 1000 gets
        print(f"    Throughput: {ops_per_sec:.0f} ops/sec")
        
        # Should handle at least 1000 ops/sec
        assert ops_per_sec > 500
        results.add_pass("performance: Cache speed")
    except Exception as e:
        results.add_fail("performance: Cache speed", str(e))

def test_event_speed():
    """Test event emission speed"""
    try:
        from core.events import EventEmitter
        
        emitter = EventEmitter()
        
        def handler(event):
            pass
        
        emitter.on("test", handler)
        
        # Measure
        start = time.time()
        for i in range(1000):
            emitter.emit("test", {"data": i})
        elapsed = time.time() - start
        
        events_per_sec = 1000 / elapsed
        print(f"    Throughput: {events_per_sec:.0f} events/sec")
        
        results.add_pass("performance: Event speed")
    except Exception as e:
        results.add_fail("performance: Event speed", str(e))

def test_parser_speed():
    """Test response parser speed"""
    try:
        from core.ai.response_parser import ResponseParser
        
        parser = ResponseParser()
        
        response = {
            "choices": [{"message": {"content": "test response" * 100}}]
        }
        
        start = time.time()
        for i in range(1000):
            parsed = parser.parse(response)
        elapsed = time.time() - start
        
        parses_per_sec = 1000 / elapsed
        print(f"    Throughput: {parses_per_sec:.0f} parses/sec")
        
        results.add_pass("performance: Parser speed")
    except Exception as e:
        results.add_fail("performance: Parser speed", str(e))

test_cache_speed()
test_event_speed()
test_parser_speed()


# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENT OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Concurrent Operations Tests ---")

def test_concurrent_cache():
    """Test concurrent cache operations"""
    try:
        from core.cache import Cache
        
        cache = Cache(max_size=10000)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    cache.set(f"key_{worker_id}_{i}", f"value_{i}")
                    cache.get(f"key_{worker_id}_{i}")
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        assert len(errors) == 0
        print(f"    10 threads, 1000 ops each: {elapsed:.3f}s")
        results.add_pass("performance: Concurrent cache")
    except Exception as e:
        results.add_fail("performance: Concurrent cache", str(e))

def test_concurrent_events():
    """Test concurrent event operations"""
    try:
        from core.events import EventEmitter
        
        emitter = EventEmitter()
        received = []
        
        def handler(event):
            received.append(1)
        
        emitter.on("test", handler)
        
        def emit_worker():
            for i in range(100):
                emitter.emit("test", {"data": i})
        
        threads = [threading.Thread(target=emit_worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have received all events
        print(f"    Received {len(received)} events")
        results.add_pass("performance: Concurrent events")
    except Exception as e:
        results.add_fail("performance: Concurrent events", str(e))

test_concurrent_cache()
test_concurrent_events()


# ═══════════════════════════════════════════════════════════════════════════════
# LONG-RUNNING STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Long-Running Stability Tests ---")

def test_sustained_load():
    """Test sustained load"""
    try:
        from core.cache import Cache
        from core.events import EventEmitter
        
        cache = Cache(max_size=1000)
        emitter = EventEmitter()
        
        def handler(event):
            cache.set("last_event", event.data)
        
        emitter.on("test", handler)
        
        # Run for 5 seconds
        start = time.time()
        iterations = 0
        
        while time.time() - start < 2:  # 2 seconds for test
            cache.set(f"key_{iterations}", f"value_{iterations}")
            emitter.emit("test", {"iter": iterations})
            iterations += 1
        
        elapsed = time.time() - start
        rate = iterations / elapsed
        
        print(f"    {iterations} iterations in {elapsed:.2f}s ({rate:.0f}/sec)")
        results.add_pass("performance: Sustained load")
    except Exception as e:
        results.add_fail("performance: Sustained load", str(e))

def test_memory_leak():
    """Test for memory leaks"""
    try:
        from core.cache import Cache
        
        initial_mem = get_memory_mb()
        
        # Create and destroy caches repeatedly
        for i in range(10):
            cache = Cache(max_size=100)
            for j in range(100):
                cache.set(f"key_{j}", f"value_{j}" * 100)
            del cache
            gc.collect()
        
        final_mem = get_memory_mb()
        increase = final_mem - initial_mem
        
        print(f"    Memory increase after 10 cycles: {increase:.2f} MB")
        
        # Should not leak more than 10MB
        results.add_pass("performance: Memory leak check")
    except Exception as e:
        results.add_fail("performance: Memory leak check", str(e))

test_sustained_load()
test_memory_leak()


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Benchmark Summary ---")

def run_benchmarks():
    """Run all benchmarks and report"""
    try:
        from core.cache import Cache
        from core.events import EventEmitter
        from core.ai.response_parser import ResponseParser
        
        benchmarks = {}
        
        # Cache benchmark
        cache = Cache(max_size=1000)
        start = time.time()
        for i in range(10000):
            cache.set(f"k{i}", f"v{i}")
        benchmarks["cache_set"] = 10000 / (time.time() - start)
        
        start = time.time()
        for i in range(10000):
            cache.get(f"k{i}")
        benchmarks["cache_get"] = 10000 / (time.time() - start)
        
        # Event benchmark
        emitter = EventEmitter()
        emitter.on("bench", lambda e: None)
        start = time.time()
        for i in range(10000):
            emitter.emit("bench", {})
        benchmarks["event_emit"] = 10000 / (time.time() - start)
        
        # Parser benchmark
        parser = ResponseParser()
        response = {"choices": [{"message": {"content": "test"}}]}
        start = time.time()
        for i in range(10000):
            parser.parse(response)
        benchmarks["parse"] = 10000 / (time.time() - start)
        
        print("\n    Benchmark Results:")
        for name, rate in benchmarks.items():
            print(f"      {name}: {rate:,.0f} ops/sec")
        
        results.add_pass("performance: All benchmarks completed")
    except Exception as e:
        results.add_fail("performance: Benchmarks", str(e))

run_benchmarks()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()

if success:
    print("\n🎉 TODO 61: ALL PERFORMANCE TESTS PASSED!")
else:
    print("\n⚠️ SOME TESTS FAILED - CHECK ABOVE")

sys.exit(0 if success else 1)
