#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Cache System
===================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- LRU (Least Recently Used) eviction
- TTL (Time To Live) support
- Memory-efficient storage
- Thread-safe operations
- Disk persistence option

Features:
- In-memory cache with LRU eviction
- TTL-based expiration
- Disk cache for persistence
- Cache invalidation strategies
- Cache warming
- Statistics and monitoring
- Decorator for function caching
- Tag-based cache groups
- Memory pressure handling

Memory Impact: Configurable, default < 20MB
"""

import sys
import os
import time
import json
import pickle
import hashlib
import logging
import threading
import tempfile
import shutil
from pathlib import Path
from typing import (
    Dict, Any, Optional, List, Set, Tuple, Callable, 
    Union, TypeVar, Generic, Hashable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = auto()      # Least Recently Used
    LFU = auto()      # Least Frequently Used
    FIFO = auto()     # First In First Out
    TTL = auto()      # Time To Live only


class CachePriority(Enum):
    """Cache entry priority"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EvictionReason(Enum):
    """Reason for cache eviction"""
    CAPACITY = auto()
    EXPIRED = auto()
    MANUAL = auto()
    MEMORY_PRESSURE = auto()
    TAG_INVALIDATION = auto()


@dataclass
class CacheEntry:
    """
    A cache entry with metadata.
    """
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    priority: CachePriority = CachePriority.NORMAL
    tags: Set[str] = field(default_factory=set)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.created_at
    
    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL"""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)
    
    def touch(self):
        """Update last accessed time and count"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def __sizeof__(self) -> int:
        """Estimate memory size"""
        try:
            return sys.getsizeof(self.value) + 200  # Base overhead
        except Exception:
            return self.size_bytes or 100


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    sets: int = 0
    deletes: int = 0
    
    # Size tracking
    current_entries: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    max_entries: int = 0
    
    # Timing
    total_get_time_ms: float = 0.0
    total_set_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate"""
        return 1.0 - self.hit_rate
    
    @property
    def avg_get_time_ms(self) -> float:
        """Average get operation time"""
        if self.hits + self.misses == 0:
            return 0.0
        return self.total_get_time_ms / (self.hits + self.misses)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hit_rate:.2%}",
            'evictions': self.evictions,
            'expired': self.expired,
            'sets': self.sets,
            'deletes': self.deletes,
            'entries': self.current_entries,
            'size_mb': self.current_size_bytes / (1024 * 1024),
            'avg_get_time_ms': f"{self.avg_get_time_ms:.3f}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryCache:
    """
    In-Memory Cache with LRU Eviction.
    
    Features:
    - LRU eviction
    - TTL support
    - Memory limits
    - Thread-safe
    - Tag-based grouping
    - Statistics tracking
    
    Memory Budget: Configurable
    
    Usage:
        cache = MemoryCache(max_size_mb=10)
        
        # Set with TTL
        cache.set('key', 'value', ttl=60)
        
        # Get
        value = cache.get('key')
        
        # With tags
        cache.set('user:1', data, tags=['users', 'active'])
        
        # Invalidate by tag
        cache.invalidate_tag('users')
    """
    
    def __init__(
        self,
        max_size_mb: float = 10.0,
        max_entries: int = 10000,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.LRU,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize Memory Cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            default_ttl: Default TTL in seconds
            strategy: Eviction strategy
            cleanup_interval: Interval for automatic cleanup
        """
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._strategy = strategy
        self._cleanup_interval = cleanup_interval
        
        # Storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self._stats = CacheStats(
            max_size_bytes=self._max_size,
            max_entries=max_entries,
        )
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop = threading.Event()
        self._start_cleanup_thread()
        
        logger.info(f"MemoryCache initialized: {max_size_mb}MB, {max_entries} entries")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while not self._cleanup_stop.wait(self._cleanup_interval):
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="cache-cleanup"
        )
        self._cleanup_thread.start()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get(
        self,
        key: str,
        default: Any = None,
        touch: bool = True,
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            touch: Whether to update access time
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                return default
            
            if entry.is_expired:
                self._delete_entry(key, EvictionReason.EXPIRED)
                self._stats.misses += 1
                self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                return default
            
            if touch:
                entry.touch()
                # Move to end for LRU
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.total_get_time_ms += (time.time() - start_time) * 1000
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Set[str] = None,
        priority: CachePriority = CachePriority.NORMAL,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for grouping
            priority: Entry priority
            metadata: Additional metadata
            
        Returns:
            True if set successfully
        """
        start_time = time.time()
        
        with self._lock:
            # Calculate expiry
            if ttl is None:
                ttl = self._default_ttl
            
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            # Estimate size
            try:
                size = sys.getsizeof(value) + 200
            except Exception:
                size = 100
            
            # Check if we need to evict
            self._ensure_space(size)
            
            # Remove old entry if exists
            if key in self._cache:
                self._delete_entry(key, EvictionReason.MANUAL)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size,
                priority=priority,
                tags=tags or set(),
                metadata=metadata or {},
            )
            
            # Store entry
            self._cache[key] = entry
            
            # Update tag index
            for tag in entry.tags:
                self._tags[tag].add(key)
            
            # Update stats
            self._stats.sets += 1
            self._stats.current_entries = len(self._cache)
            self._stats.current_size_bytes += size
            self._stats.total_set_time_ms += (time.time() - start_time) * 1000
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                self._delete_entry(key, EvictionReason.MANUAL)
                self._stats.deletes += 1
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._delete_entry(key, EvictionReason.EXPIRED)
                return False
            return True
    
    def _delete_entry(self, key: str, reason: EvictionReason):
        """Internal delete with cleanup"""
        if key not in self._cache:
            return
        
        entry = self._cache.pop(key)
        
        # Update stats
        self._stats.current_entries -= 1
        self._stats.current_size_bytes -= entry.size_bytes
        
        if reason == EvictionReason.EXPIRED:
            self._stats.expired += 1
        elif reason == EvictionReason.CAPACITY:
            self._stats.evictions += 1
        
        # Remove from tag index
        for tag in entry.tags:
            self._tags[tag].discard(key)
            if not self._tags[tag]:
                del self._tags[tag]
    
    def _ensure_space(self, required_size: int):
        """Ensure there's enough space for new entry"""
        while (
            (self._stats.current_size_bytes + required_size > self._max_size or
             len(self._cache) >= self._max_entries) and
            self._cache
        ):
            self._evict_one()
    
    def _evict_one(self):
        """Evict one entry based on strategy"""
        if not self._cache:
            return
        
        if self._strategy == CacheStrategy.LRU:
            # Evict oldest (first in OrderedDict)
            key = next(iter(self._cache))
            self._delete_entry(key, EvictionReason.CAPACITY)
        
        elif self._strategy == CacheStrategy.LFU:
            # Find least frequently used
            min_count = float('inf')
            min_key = None
            for key, entry in self._cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    min_key = key
            if min_key:
                self._delete_entry(min_key, EvictionReason.CAPACITY)
        
        elif self._strategy == CacheStrategy.FIFO:
            # Evict first inserted
            key = next(iter(self._cache))
            self._delete_entry(key, EvictionReason.CAPACITY)
        
        elif self._strategy == CacheStrategy.TTL:
            # Find entry with shortest remaining TTL
            min_ttl = float('inf')
            min_key = None
            for key, entry in self._cache.items():
                if entry.expires_at and entry.ttl_remaining < min_ttl:
                    min_ttl = entry.ttl_remaining
                    min_key = key
            if min_key:
                self._delete_entry(min_key, EvictionReason.CAPACITY)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[float] = None,
        tags: Set[str] = None,
    ) -> int:
        """Set multiple values"""
        count = 0
        for key, value in items.items():
            if self.set(key, value, ttl=ttl, tags=tags):
                count += 1
        return count
    
    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple entries"""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAG OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def invalidate_tag(self, tag: str) -> int:
        """Invalidate all entries with a tag"""
        with self._lock:
            keys = list(self._tags.get(tag, set()))
            count = 0
            for key in keys:
                self._delete_entry(key, EvictionReason.TAG_INVALIDATION)
                count += 1
            return count
    
    def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get all entries with a tag"""
        result = {}
        with self._lock:
            for key in self._tags.get(tag, set()):
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    result[key] = entry.value
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries"""
        count = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                self._delete_entry(key, EvictionReason.EXPIRED)
                count += 1
        return count
    
    def clear(self):
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
            self._tags.clear()
            self._stats.current_entries = 0
            self._stats.current_size_bytes = 0
    
    def compact(self):
        """Compact cache by removing low-priority entries under memory pressure"""
        with self._lock:
            # Remove all low priority entries
            low_priority_keys = [
                key for key, entry in self._cache.items()
                if entry.priority == CachePriority.LOW
            ]
            for key in low_priority_keys:
                self._delete_entry(key, EvictionReason.MEMORY_PRESSURE)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self._stats.current_entries = len(self._cache)
            return self._stats
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an entry"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            return {
                'key': key,
                'created_at': entry.created_at,
                'expires_at': entry.expires_at,
                'ttl_remaining': entry.ttl_remaining,
                'age_seconds': entry.age_seconds,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'priority': entry.priority.name,
                'tags': list(entry.tags),
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_stop.set()
        return False
    
    def shutdown(self):
        """Shutdown the cache"""
        self._cleanup_stop.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
        self.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# DISK CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class DiskCache:
    """
    Persistent Disk Cache.
    
    Features:
    - File-based storage
    - JSON and pickle support
    - Compression option
    - Automatic cleanup
    
    Memory Budget: Minimal (only index in memory)
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        max_size_mb: float = 100.0,
        default_ttl: Optional[float] = None,
        use_compression: bool = True,
        serialize_format: str = 'pickle',  # 'pickle' or 'json'
    ):
        """
        Initialize Disk Cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum total size
            default_ttl: Default TTL
            use_compression: Use gzip compression
            serialize_format: Serialization format
        """
        self._cache_dir = Path(cache_dir or tempfile.gettempdir()) / 'jarvis_cache'
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._default_ttl = default_ttl
        self._use_compression = use_compression
        self._format = serialize_format
        
        # Index file
        self._index_file = self._cache_dir / 'index.json'
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Load existing index
        self._load_index()
        
        logger.info(f"DiskCache initialized: {self._cache_dir}")
    
    def _load_index(self):
        """Load cache index"""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache index: {e}")
            self._index = {}
    
    def _save_index(self):
        """Save cache index"""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from disk cache"""
        with self._lock:
            if key not in self._index:
                return default
            
            info = self._index[key]
            
            # Check expiry
            if info.get('expires_at') and time.time() > info['expires_at']:
                self.delete(key)
                return default
            
            file_path = self._get_file_path(key)
            if not file_path.exists():
                del self._index[key]
                return default
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                if self._use_compression:
                    import gzip
                    data = gzip.decompress(data)
                
                if self._format == 'json':
                    return json.loads(data.decode('utf-8'))
                else:
                    return pickle.loads(data)
                    
            except Exception as e:
                logger.error(f"Cache read error: {e}")
                return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set value in disk cache"""
        with self._lock:
            try:
                # Serialize
                if self._format == 'json':
                    data = json.dumps(value).encode('utf-8')
                else:
                    data = pickle.dumps(value)
                
                # Compress
                if self._use_compression:
                    import gzip
                    data = gzip.compress(data)
                
                # Write file
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                expires_at = None
                if ttl is not None or self._default_ttl is not None:
                    expires_at = time.time() + (ttl or self._default_ttl)
                
                self._index[key] = {
                    'file': file_path.name,
                    'size': len(data),
                    'created_at': time.time(),
                    'expires_at': expires_at,
                }
                
                self._save_index()
                return True
                
            except Exception as e:
                logger.error(f"Cache write error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete from disk cache"""
        with self._lock:
            if key not in self._index:
                return False
            
            file_path = self._get_file_path(key)
            try:
                if file_path.exists():
                    file_path.unlink()
                del self._index[key]
                self._save_index()
                return True
            except Exception as e:
                logger.error(f"Cache delete error: {e}")
                return False
    
    def clear(self):
        """Clear all cache files"""
        with self._lock:
            for key in list(self._index.keys()):
                self.delete(key)
            self._index.clear()
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(
                info.get('size', 0)
                for info in self._index.values()
            )
            
            return {
                'entries': len(self._index),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self._cache_dir),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════

def cached(
    cache: Union[MemoryCache, DiskCache] = None,
    key_func: Callable = None,
    ttl: Optional[float] = None,
    tags: Set[str] = None,
):
    """
    Decorator to cache function results.
    
    Usage:
        @cached(ttl=60)
        def expensive_function(arg):
            return compute(arg)
        
        # With custom key
        @cached(key_func=lambda a, b: f"{a}:{b}")
        def function(a, b):
            return a + b
    """
    if cache is None:
        cache = get_cache()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = f"{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl, tags=tags)
            return result
        
        wrapper.cache_clear = lambda: cache.delete_many([
            k for k in getattr(cache, '_cache', {}).keys()
            if k.startswith(func.__name__)
        ]) if isinstance(cache, MemoryCache) else None
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CACHE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_global_cache: Optional[MemoryCache] = None
_cache_lock = threading.Lock()


def get_cache() -> MemoryCache:
    """Get the global cache instance"""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = MemoryCache()
        return _global_cache


def cache_get(key: str, default: Any = None) -> Any:
    """Get from global cache"""
    return get_cache().get(key, default)


def cache_set(key: str, value: Any, ttl: Optional[float] = None) -> bool:
    """Set in global cache"""
    return get_cache().set(key, value, ttl=ttl)


def cache_delete(key: str) -> bool:
    """Delete from global cache"""
    return get_cache().delete(key)


def cache_clear():
    """Clear global cache"""
    get_cache().clear()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classes
    'MemoryCache',
    'DiskCache',
    'CacheEntry',
    'CacheStats',
    'CacheStrategy',
    'CachePriority',
    'EvictionReason',
    
    # Functions
    'get_cache',
    'cache_get',
    'cache_set',
    'cache_delete',
    'cache_clear',
    
    # Decorators
    'cached',
]
