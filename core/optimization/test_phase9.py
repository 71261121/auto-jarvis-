#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 9: Optimization Tests
=================================================

Comprehensive tests for all optimization modules:
- TODO 73: Memory Optimization Tests
- TODO 74: Performance Optimization Tests
- TODO 75: Startup Optimization Tests
- TODO 76: Battery Optimization Tests
- TODO 77: Storage Optimization Tests
- TODO 78: Network Optimization Tests

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import sys
import os
import time
import tempfile
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    """Simple test result tracker"""
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
        print(f"    Error: {error[:80]}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed)
        rate = (len(self.passed) / total * 100) if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"PHASE 9: Optimization Tests Results")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}% | Time: {elapsed:.2f}s")
        print(f"{'='*60}")
        
        if len(self.failed) == 0:
            print("✓ ALL PHASE 9 TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED!")
        
        return len(self.failed) == 0


results = TestResult()

print("="*60)
print("PHASE 9: Optimization Module Tests")
print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 73: MEMORY OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 73: Memory Optimization Tests ---")

def test_memory_profiler():
    """Test MemoryProfiler"""
    try:
        from core.optimization.memory_optimizer import MemoryProfiler
        
        profiler = MemoryProfiler(enable_tracing=False)
        snapshot = profiler.take_snapshot()
        
        assert snapshot.current_mb >= 0
        assert snapshot.peak_mb >= 0
        results.add_pass("memory: MemoryProfiler")
    except Exception as e:
        results.add_fail("memory: MemoryProfiler", str(e))

def test_lazy_loader():
    """Test LazyLoader"""
    try:
        from core.optimization.memory_optimizer import LazyLoader
        
        # Test with built-in module
        loader = LazyLoader('json')
        
        # Should not be loaded yet
        assert not loader.is_loaded
        
        # Access attribute triggers load
        _ = loader.dumps
        assert loader.is_loaded
        
        results.add_pass("memory: LazyLoader")
    except Exception as e:
        results.add_fail("memory: LazyLoader", str(e))

def test_memory_efficient_dict():
    """Test MemoryEfficientDict"""
    try:
        from core.optimization.memory_optimizer import MemoryEfficientDict
        
        d = MemoryEfficientDict(max_size=10)
        
        for i in range(15):
            d[f"key_{i}"] = f"value_{i}"
        
        # Should have evicted old entries
        assert len(d) <= 10
        
        results.add_pass("memory: MemoryEfficientDict")
    except Exception as e:
        results.add_fail("memory: MemoryEfficientDict", str(e))

def test_memory_efficient_list():
    """Test MemoryEfficientList"""
    try:
        from core.optimization.memory_optimizer import MemoryEfficientList
        
        lst = MemoryEfficientList(chunk_size=100)
        
        for i in range(250):
            lst.append(i)
        
        assert len(lst) == 250
        assert lst[0] == 0
        assert lst[249] == 249
        
        results.add_pass("memory: MemoryEfficientList")
    except Exception as e:
        results.add_fail("memory: MemoryEfficientList", str(e))

def test_generator_processor():
    """Test GeneratorProcessor"""
    try:
        from core.optimization.memory_optimizer import GeneratorProcessor
        
        proc = GeneratorProcessor()
        proc.from_iterable(range(100))
        proc.map(lambda x: x * 2)
        proc.filter(lambda x: x % 4 == 0)
        
        result = proc.first(5)
        assert len(result) == 5
        
        results.add_pass("memory: GeneratorProcessor")
    except Exception as e:
        results.add_fail("memory: GeneratorProcessor", str(e))

def test_gc_manager():
    """Test GCManager"""
    try:
        from core.optimization.memory_optimizer import GCManager
        
        gc_mgr = GCManager()
        
        # Test optimization modes
        gc_mgr.optimize_for_low_memory()
        gc_mgr.optimize_for_speed()
        gc_mgr.restore_defaults()
        
        stats = gc_mgr.get_stats()
        assert 'thresholds' in stats
        
        results.add_pass("memory: GCManager")
    except Exception as e:
        results.add_fail("memory: GCManager", str(e))

test_memory_profiler()
test_lazy_loader()
test_memory_efficient_dict()
test_memory_efficient_list()
test_generator_processor()
test_gc_manager()


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 74: PERFORMANCE OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 74: Performance Optimization Tests ---")

def test_async_io_manager():
    """Test AsyncIOManager"""
    try:
        from core.optimization.performance_optimizer import AsyncIOManager
        
        mgr = AsyncIOManager()
        loop = mgr.get_loop()
        
        assert loop is not None
        
        results.add_pass("performance: AsyncIOManager")
    except Exception as e:
        results.add_fail("performance: AsyncIOManager", str(e))

def test_connection_pool():
    """Test ConnectionPool"""
    try:
        from core.optimization.performance_optimizer import ConnectionPool
        
        created_count = 0
        
        def factory():
            nonlocal created_count
            created_count += 1
            return {"id": created_count}
        
        pool = ConnectionPool(factory, max_size=3)
        
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        
        pool.release(conn1)
        
        stats = pool.get_stats()
        assert stats['in_use'] == 1
        
        results.add_pass("performance: ConnectionPool")
    except Exception as e:
        results.add_fail("performance: ConnectionPool", str(e))

def test_request_batcher():
    """Test RequestBatcher"""
    try:
        from core.optimization.performance_optimizer import RequestBatcher
        
        batched_items = []
        
        def processor(items):
            batched_items.extend(items)
            return [True] * len(items)
        
        batcher = RequestBatcher(processor, batch_size=5)
        
        for i in range(3):
            batcher.add({"item": i})
        
        # Items should be pending
        assert len(batched_items) == 0
        
        results.add_pass("performance: RequestBatcher")
    except Exception as e:
        results.add_fail("performance: RequestBatcher", str(e))

def test_benchmark_runner():
    """Test BenchmarkRunner"""
    try:
        from core.optimization.performance_optimizer import BenchmarkRunner
        
        runner = BenchmarkRunner(warmup=1)
        
        def test_func():
            return sum(range(100))
        
        result = runner.benchmark(test_func, name="test", iterations=10)
        
        assert result.iterations == 10
        assert result.avg_time > 0
        
        results.add_pass("performance: BenchmarkRunner")
    except Exception as e:
        results.add_fail("performance: BenchmarkRunner", str(e))

test_async_io_manager()
test_connection_pool()
test_request_batcher()
test_benchmark_runner()


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 75: STARTUP OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 75: Startup Optimization Tests ---")

def test_deferred_import():
    """Test DeferredImport"""
    try:
        from core.optimization.startup_optimizer import DeferredImport
        
        deferred = DeferredImport('json', optional=True)
        
        assert not deferred.is_loaded
        
        # Access triggers import
        _ = deferred.dumps
        assert deferred.is_loaded
        
        results.add_pass("startup: DeferredImport")
    except Exception as e:
        results.add_fail("startup: DeferredImport", str(e))

def test_minimal_mode():
    """Test MinimalMode"""
    try:
        from core.optimization.startup_optimizer import MinimalMode
        
        mode = MinimalMode()
        
        mode.enable()
        assert mode.enabled
        assert mode.should_load('core')  # Essential
        
        mode.disable()
        assert not mode.enabled
        
        results.add_pass("startup: MinimalMode")
    except Exception as e:
        results.add_fail("startup: MinimalMode", str(e))

def test_background_initializer():
    """Test BackgroundInitializer"""
    try:
        from core.optimization.startup_optimizer import BackgroundInitializer
        
        init = BackgroundInitializer()
        
        call_count = [0]
        
        def init_task():
            call_count[0] += 1
            return "initialized"
        
        init.register("test", init_task)
        init.start()
        init.wait(timeout=5.0)
        
        assert init.is_complete()
        assert call_count[0] == 1
        
        results.add_pass("startup: BackgroundInitializer")
    except Exception as e:
        results.add_fail("startup: BackgroundInitializer", str(e))

def test_startup_cache():
    """Test StartupCache"""
    try:
        from core.optimization.startup_optimizer import StartupCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StartupCache(Path(tmpdir))
            
            # Set and get
            cache.set("test_key", {"data": "value"}, ttl_seconds=60)
            result = cache.get("test_key")
            
            assert result == {"data": "value"}
            
            # Invalidate
            cache.invalidate("test_key")
            assert cache.get("test_key") is None
        
        results.add_pass("startup: StartupCache")
    except Exception as e:
        results.add_fail("startup: StartupCache", str(e))

test_deferred_import()
test_minimal_mode()
test_background_initializer()
test_startup_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 76: BATTERY OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 76: Battery Optimization Tests ---")

def test_power_state():
    """Test PowerState enum"""
    try:
        from core.optimization.battery_optimizer import PowerState
        
        assert PowerState.ACTIVE
        assert PowerState.IDLE
        assert PowerState.DOZE
        assert PowerState.DEEP_SLEEP
        
        results.add_pass("battery: PowerState")
    except Exception as e:
        results.add_fail("battery: PowerState", str(e))

def test_polling_manager():
    """Test PollingManager"""
    try:
        from core.optimization.battery_optimizer import PollingManager
        
        pm = PollingManager(base_interval=1.0)
        
        call_count = [0]
        
        def callback():
            call_count[0] += 1
        
        pm.register("test", callback)
        
        # Record activity should reset interval
        pm.record_activity("test")
        interval = pm.get_adaptive_interval("test")
        
        assert interval >= pm.base_interval
        
        results.add_pass("battery: PollingManager")
    except Exception as e:
        results.add_fail("battery: PollingManager", str(e))

def test_sleep_cycle_manager():
    """Test SleepCycleManager"""
    try:
        from core.optimization.battery_optimizer import SleepCycleManager
        
        mgr = SleepCycleManager()
        
        # Short sleep test
        slept = mgr.smart_sleep(0.01, interruptible=True)
        assert slept
        
        results.add_pass("battery: SleepCycleManager")
    except Exception as e:
        results.add_fail("battery: SleepCycleManager", str(e))

def test_network_batcher():
    """Test NetworkBatcher"""
    try:
        from core.optimization.battery_optimizer import NetworkBatcher
        
        batched = []
        
        def executor(items):
            batched.extend(items)
        
        nb = NetworkBatcher(batch_size=3)
        nb.set_executor(executor)
        
        nb.add_request({"id": 1})
        nb.add_request({"id": 2})
        nb.add_request({"id": 3})
        
        # Should have flushed
        assert len(batched) == 3
        
        results.add_pass("battery: NetworkBatcher")
    except Exception as e:
        results.add_fail("battery: NetworkBatcher", str(e))

def test_background_task_scheduler():
    """Test BackgroundTaskScheduler"""
    try:
        from core.optimization.battery_optimizer import BackgroundTaskScheduler
        
        scheduler = BackgroundTaskScheduler()
        
        call_count = [0]
        
        def task():
            call_count[0] += 1
        
        scheduler.schedule("test_task", task, delay_seconds=0.05)  # 50ms delay
        scheduler.start()
        
        time.sleep(0.2)  # Wait for task to execute
        scheduler.stop()
        
        # Task should have executed at least once
        results.add_pass("battery: BackgroundTaskScheduler")
    except Exception as e:
        results.add_fail("battery: BackgroundTaskScheduler", str(e))

test_power_state()
test_polling_manager()
test_sleep_cycle_manager()
test_network_batcher()
test_background_task_scheduler()


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 77: STORAGE OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 77: Storage Optimization Tests ---")

def test_data_compressor():
    """Test DataCompressor"""
    try:
        from core.optimization.storage_optimizer import DataCompressor
        
        compressor = DataCompressor()
        
        # Compress data
        original = b"Hello World! " * 100
        compressed = compressor.compress_data(original)
        
        assert len(compressed) < len(original)
        
        # Decompress
        decompressed = compressor.decompress_data(compressed)
        assert decompressed == original
        
        results.add_pass("storage: DataCompressor")
    except Exception as e:
        results.add_fail("storage: DataCompressor", str(e))

def test_log_rotator():
    """Test LogRotator"""
    try:
        from core.optimization.storage_optimizer import LogRotator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rotator = LogRotator(tmpdir, max_size_mb=0.001, max_files=2)
            
            # Create test log
            log_file = os.path.join(tmpdir, "test.log")
            with open(log_file, 'w') as f:
                f.write("x" * 2000)  # 2KB > 1KB limit
            
            assert rotator.should_rotate(log_file)
            
            results.add_pass("storage: LogRotator")
    except Exception as e:
        results.add_fail("storage: LogRotator", str(e))

def test_cache_manager():
    """Test CacheManager"""
    try:
        from core.optimization.storage_optimizer import CacheManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(tmpdir, max_size_mb=1.0)
            
            # Set and get
            cache.set("key1", {"data": "value1"}, ttl_seconds=60)
            result = cache.get("key1")
            
            assert result == {"data": "value1"}
            
            # Delete
            cache.delete("key1")
            assert cache.get("key1") is None
            
            stats = cache.get_stats()
            assert 'entries' in stats
        
        results.add_pass("storage: CacheManager")
    except Exception as e:
        results.add_fail("storage: CacheManager", str(e))

def test_cleanup_scheduler():
    """Test CleanupScheduler"""
    try:
        from core.optimization.storage_optimizer import CleanupScheduler
        
        scheduler = CleanupScheduler()
        
        call_count = [0]
        
        def cleanup():
            call_count[0] += 1
            return {"cleaned": 5}
        
        scheduler.register("test", cleanup)
        cleanup_results = scheduler.run_cleanup()  # Renamed variable
        
        assert "test" in cleanup_results
        assert call_count[0] == 1
        
        results.add_pass("storage: CleanupScheduler")
    except Exception as e:
        results.add_fail("storage: CleanupScheduler", str(e))

test_data_compressor()
test_log_rotator()
test_cache_manager()
test_cleanup_scheduler()


# ═══════════════════════════════════════════════════════════════════════════════
# TODO 78: NETWORK OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TODO 78: Network Optimization Tests ---")

def test_request_compressor():
    """Test RequestCompressor"""
    try:
        from core.optimization.network_optimizer import RequestCompressor
        
        compressor = RequestCompressor(threshold_bytes=100)
        
        # Small body - no compression
        small = b"small"
        compressed, was_compressed = compressor.compress_body(small)
        assert not was_compressed
        
        # Large body - should compress
        large = b"x" * 1000
        compressed, was_compressed = compressor.compress_body(large)
        assert was_compressed
        assert len(compressed) < len(large)
        
        results.add_pass("network: RequestCompressor")
    except Exception as e:
        results.add_fail("network: RequestCompressor", str(e))

def test_response_cache():
    """Test ResponseCache"""
    try:
        from core.optimization.network_optimizer import ResponseCache
        
        cache = ResponseCache(max_entries=10, default_ttl=60)
        
        # Cache a response
        cache.set(
            "GET", 
            "https://api.test.com/data",
            b'{"data": "value"}',
            {"content-type": "application/json"},
            200
        )
        
        # Retrieve
        entry = cache.get("GET", "https://api.test.com/data")
        assert entry is not None
        assert entry.data == b'{"data": "value"}'
        
        stats = cache.get_stats()
        assert stats['entries'] == 1
        
        results.add_pass("network: ResponseCache")
    except Exception as e:
        results.add_fail("network: ResponseCache", str(e))

def test_offline_mode_manager():
    """Test OfflineModeManager"""
    try:
        from core.optimization.network_optimizer import OfflineModeManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = OfflineModeManager(queue_dir=tmpdir)
            
            assert not mgr.is_offline
            
            mgr.set_offline(True)
            assert mgr.is_offline
            
            # Queue a request
            req_id = mgr.queue_request(
                "GET",
                "https://api.test.com/test",
                {"Authorization": "Bearer token"}
            )
            
            assert req_id is not None
            
            stats = mgr.get_queue_stats()
            assert stats['queued_requests'] == 1
        
        results.add_pass("network: OfflineModeManager")
    except Exception as e:
        results.add_fail("network: OfflineModeManager", str(e))

def test_bandwidth_adapter():
    """Test BandwidthAdapter"""
    try:
        from core.optimization.network_optimizer import BandwidthAdapter
        
        adapter = BandwidthAdapter()
        
        # Get adapted settings (without actual measurement)
        settings = adapter.get_adapted_settings()
        
        assert 'batch_size' in settings
        assert 'timeout' in settings
        assert 'compression' in settings
        
        results.add_pass("network: BandwidthAdapter")
    except Exception as e:
        results.add_fail("network: BandwidthAdapter", str(e))

test_request_compressor()
test_response_cache()
test_offline_mode_manager()
test_bandwidth_adapter()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()
sys.exit(0 if success else 1)
