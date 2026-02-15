#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Optimization Package
==========================================

Phase 9: Optimization (TODO 73-78)

This package provides comprehensive optimization for:
- Memory (TODO 73): Lazy loading, GC optimization, generators
- Performance (TODO 74): Async I/O, caching, connection pooling
- Startup (TODO 75): Deferred imports, minimal mode
- Battery (TODO 76): Efficient polling, sleep cycles
- Storage (TODO 77): Compression, rotation, cleanup
- Network (TODO 78): Request batching, offline mode

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

from .memory_optimizer import (
    MemoryOptimizer,
    LazyLoader,
    MemoryEfficientDict,
    MemoryEfficientList,
    GeneratorProcessor,
    MemoryProfiler,
    GCManager,
)

from .performance_optimizer import (
    PerformanceOptimizer,
    AsyncIOManager,
    ConnectionPool,
    RequestBatcher,
    BenchmarkRunner,
)

from .startup_optimizer import (
    StartupOptimizer,
    DeferredImport,
    MinimalMode,
    BackgroundInitializer,
    StartupCache,
)

from .battery_optimizer import (
    BatteryOptimizer,
    PollingManager,
    SleepCycleManager,
    NetworkBatcher,
    BackgroundTaskScheduler,
)

from .storage_optimizer import (
    StorageOptimizer,
    DataCompressor,
    LogRotator,
    CacheManager,
    CleanupScheduler,
)

from .network_optimizer import (
    NetworkOptimizer,
    RequestCompressor,
    ResponseCache,
    OfflineModeManager,
    BandwidthAdapter,
)

__all__ = [
    # Memory
    'MemoryOptimizer',
    'LazyLoader',
    'MemoryEfficientDict',
    'MemoryEfficientList',
    'GeneratorProcessor',
    'MemoryProfiler',
    'GCManager',
    # Performance
    'PerformanceOptimizer',
    'AsyncIOManager',
    'ConnectionPool',
    'RequestBatcher',
    'BenchmarkRunner',
    # Startup
    'StartupOptimizer',
    'DeferredImport',
    'MinimalMode',
    'BackgroundInitializer',
    'StartupCache',
    # Battery
    'BatteryOptimizer',
    'PollingManager',
    'SleepCycleManager',
    'NetworkBatcher',
    'BackgroundTaskScheduler',
    # Storage
    'StorageOptimizer',
    'DataCompressor',
    'LogRotator',
    'CacheManager',
    'CleanupScheduler',
    # Network
    'NetworkOptimizer',
    'RequestCompressor',
    'ResponseCache',
    'OfflineModeManager',
    'BandwidthAdapter',
]

__version__ = '1.0.0'
