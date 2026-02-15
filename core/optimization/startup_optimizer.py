#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 75: Startup Optimization
==================================================

Startup optimization for fast initialization:
- Deferred imports
- Minimal startup mode
- Background initialization
- Startup caching

Target: <3 second startup on 4GB device

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import time
import threading
import json
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class StartupMetric:
    """Startup timing metric"""
    name: str
    duration_ms: float
    timestamp: datetime


class DeferredImport:
    """Defer module import until needed"""
    
    _registry: Dict[str, 'DeferredImport'] = {}
    
    def __init__(self, module_name: str, optional: bool = True):
        self.module_name = module_name
        self.optional = optional
        self._module = None
        self._imported = False
        self._import_time: Optional[float] = None
        self._error: Optional[Exception] = None
        DeferredImport._registry[module_name] = self
    
    def _import(self):
        """Perform the import"""
        if not self._imported:
            start = time.perf_counter()
            try:
                import importlib
                self._module = importlib.import_module(self.module_name)
            except Exception as e:
                self._error = e
                if not self.optional:
                    raise
            self._import_time = time.perf_counter() - start
            self._imported = True
        return self._module
    
    def __getattr__(self, name: str):
        module = self._import()
        if module is None:
            raise ImportError(f"Could not import {self.module_name}")
        return getattr(module, name)
    
    @property
    def is_loaded(self) -> bool:
        return self._imported
    
    @property
    def import_time_ms(self) -> Optional[float]:
        return self._import_time * 1000 if self._import_time else None
    
    @classmethod
    def get_all_times(cls) -> Dict[str, float]:
        """Get import times for all deferred imports"""
        return {
            name: imp.import_time_ms 
            for name, imp in cls._registry.items() 
            if imp.import_time_ms is not None
        }


class MinimalMode:
    """Minimal startup mode for fastest initialization"""
    
    def __init__(self):
        self.enabled = False
        self._skipped_modules: Set[str] = set()
        self._loaded_modules: Set[str] = set()
    
    def enable(self):
        """Enable minimal mode"""
        self.enabled = True
        os.environ['JARVIS_MINIMAL'] = '1'
    
    def disable(self):
        """Disable minimal mode"""
        self.enabled = False
        os.environ.pop('JARVIS_MINIMAL', None)
    
    def should_load(self, module: str) -> bool:
        """Check if module should be loaded"""
        if not self.enabled:
            return True
        
        # Essential modules
        essential = {'core', 'config', 'security.auth'}
        if module in essential:
            self._loaded_modules.add(module)
            return True
        
        self._skipped_modules.add(module)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'loaded': list(self._loaded_modules),
            'skipped': list(self._skipped_modules)
        }


class BackgroundInitializer:
    """Initialize components in background"""
    
    def __init__(self):
        self._tasks: List[Callable] = []
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._completed = False
    
    def register(self, name: str, init_func: Callable) -> None:
        """Register initialization task"""
        def task():
            try:
                self._results[name] = init_func()
            except Exception as e:
                self._errors[name] = e
        
        self._tasks.append(task)
    
    def start(self) -> None:
        """Start background initialization"""
        if self._thread is not None:
            return
        
        def run():
            self._running = True
            for task in self._tasks:
                if not self._running:
                    break
                task()
            self._completed = True
        
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for initialization to complete"""
        if self._thread is None:
            return True
        
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
    
    def stop(self) -> None:
        """Stop background initialization"""
        self._running = False
    
    def is_complete(self) -> bool:
        """Check if initialization is complete"""
        return self._completed
    
    def get_result(self, name: str) -> Optional[Any]:
        """Get result for a task"""
        return self._results.get(name)
    
    def get_errors(self) -> Dict[str, Exception]:
        """Get all errors"""
        return self._errors.copy()


class StartupCache:
    """Cache for faster startup"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.jarvis' / 'startup_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
    
    def _get_cache_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Check expiry
                    if data.get('expires', float('inf')) > time.time():
                        self._memory_cache[key] = data['value']
                        return data['value']
            except Exception:
                pass
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        """Set cached value"""
        self._memory_cache[key] = value
        
        cache_path = self._get_cache_path(key)
        try:
            data = {
                'value': value,
                'expires': time.time() + ttl_seconds,
                'created': datetime.now().isoformat()
            }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""
        self._memory_cache.pop(key, None)
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self) -> None:
        """Clear all cache"""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()


class StartupOptimizer:
    """Main startup optimization controller"""
    
    def __init__(self):
        self.minimal_mode = MinimalMode()
        self.background_init = BackgroundInitializer()
        self.cache = StartupCache()
        self._metrics: List[StartupMetric] = []
        self._start_time: Optional[float] = None
    
    def start_timing(self) -> None:
        """Start timing startup"""
        self._start_time = time.perf_counter()
    
    def record_metric(self, name: str) -> StartupMetric:
        """Record a startup metric"""
        now = time.perf_counter()
        start = self._start_time or now
        metric = StartupMetric(
            name=name,
            duration_ms=(now - start) * 1000,
            timestamp=datetime.now()
        )
        self._metrics.append(metric)
        return metric
    
    def enable_minimal_mode(self) -> None:
        """Enable minimal startup mode"""
        self.minimal_mode.enable()
    
    def register_background_task(self, name: str, func: Callable) -> None:
        """Register background initialization task"""
        self.background_init.register(name, func)
    
    def start_background_init(self) -> None:
        """Start background initialization"""
        self.background_init.start()
    
    def wait_for_init(self, timeout: float = 10.0) -> bool:
        """Wait for initialization"""
        return self.background_init.wait(timeout)
    
    def get_startup_time(self) -> float:
        """Get total startup time in seconds"""
        if not self._metrics:
            return 0.0
        return self._metrics[-1].duration_ms / 1000
    
    def get_report(self) -> str:
        """Generate startup report"""
        lines = [
            "═══════════════════════════════════════════",
            "       STARTUP PERFORMANCE REPORT",
            "═══════════════════════════════════════════",
            f"Total Time: {self.get_startup_time()*1000:.1f}ms",
            f"Target:    3000ms",
            f"Status:    {'✓ PASS' if self.get_startup_time() < 3 else '✗ SLOW'}",
            "",
            "Breakdown:",
        ]
        
        for metric in self._metrics:
            lines.append(f"  {metric.name:<30} {metric.duration_ms:>8.1f}ms")
        
        lines.extend([
            "",
            f"Minimal Mode: {'Enabled' if self.minimal_mode.enabled else 'Disabled'}",
            f"Background Init: {'Complete' if self.background_init.is_complete() else 'Running'}",
        ])
        
        return "\n".join(lines)
    
    def optimize_for_speed(self) -> Dict[str, Any]:
        """Apply speed optimizations"""
        results = {'optimizations': []}
        
        # Enable minimal mode for low-memory devices
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 2:
                self.enable_minimal_mode()
                results['optimizations'].append('minimal_mode_enabled')
        except ImportError:
            pass
        
        # Clear any stale cache
        self.cache.clear()
        results['optimizations'].append('cache_cleared')
        
        return results


# Global instance
_startup_optimizer: Optional[StartupOptimizer] = None

def get_startup_optimizer() -> StartupOptimizer:
    """Get global startup optimizer"""
    global _startup_optimizer
    if _startup_optimizer is None:
        _startup_optimizer = StartupOptimizer()
    return _startup_optimizer
