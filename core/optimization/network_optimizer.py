#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 78: Network Optimization
==================================================

Network optimization for efficient API usage:
- Request compression
- Response caching
- Offline mode support
- Bandwidth adaptation

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import gzip
import json
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import os


@dataclass
class RequestStats:
    """Network request statistics"""
    total_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_requests: int = 0


class RequestCompressor:
    """Compress network requests"""
    
    def __init__(self, threshold_bytes: int = 1024):
        self.threshold_bytes = threshold_bytes
        self._stats = {'compressed': 0, 'saved_bytes': 0}
    
    def compress_body(self, body: bytes) -> Tuple[bytes, bool]:
        """Compress request body if beneficial"""
        if len(body) < self.threshold_bytes:
            return body, False
        
        compressed = gzip.compress(body)
        
        if len(compressed) < len(body):
            self._stats['compressed'] += 1
            self._stats['saved_bytes'] += len(body) - len(compressed)
            return compressed, True
        
        return body, False
    
    def decompress_body(self, body: bytes, compressed: bool) -> bytes:
        """Decompress response body"""
        if compressed:
            return gzip.decompress(body)
        return body
    
    def compress_json(self, data: Dict) -> Tuple[bytes, bool]:
        """Compress JSON data"""
        body = json.dumps(data).encode('utf-8')
        return self.compress_body(body)
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()


@dataclass
class CacheEntry:
    """Response cache entry"""
    data: bytes
    headers: Dict[str, str]
    status_code: int
    created: float
    expires: float
    etag: Optional[str] = None
    hit_count: int = 0


class ResponseCache:
    """HTTP response cache"""
    
    def __init__(
        self,
        max_entries: int = 1000,
        max_size_mb: float = 50.0,
        default_ttl: int = 300
    ):
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def _generate_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key"""
        key_data = f"{method}:{url}"
        if params:
            key_data += ":" + json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None
    ) -> Optional[CacheEntry]:
        """Get cached response"""
        key = self._generate_key(method, url, params)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            # Check expiry
            if entry.expires < time.time():
                del self._cache[key]
                self._stats['misses'] += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats['hits'] += 1
            
            return entry
    
    def set(
        self,
        method: str,
        url: str,
        data: bytes,
        headers: Dict[str, str],
        status_code: int,
        ttl: Optional[int] = None,
        params: Optional[Dict] = None
    ) -> None:
        """Cache response"""
        key = self._generate_key(method, url, params)
        
        # Determine TTL
        cache_ttl = ttl or self.default_ttl
        
        # Check Cache-Control header
        if 'cache-control' in headers:
            cc = headers['cache-control'].lower()
            if 'no-store' in cc or 'no-cache' in cc:
                return
            if 'max-age' in cc:
                try:
                    max_age = int(cc.split('max-age=')[1].split(',')[0])
                    cache_ttl = min(cache_ttl, max_age)
                except Exception:
                    pass
        
        entry = CacheEntry(
            data=data,
            headers=headers,
            status_code=status_code,
            created=time.time(),
            expires=time.time() + cache_ttl,
            etag=headers.get('etag')
        )
        
        with self._lock:
            self._cache[key] = entry
            self._enforce_limits()
    
    def invalidate(self, method: str, url: str, params: Optional[Dict] = None) -> bool:
        """Invalidate cached entry"""
        key = self._generate_key(method, url, params)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
    
    def _enforce_limits(self) -> None:
        """Enforce cache limits"""
        # Check entry count
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self._stats['evictions'] += 1
        
        # Check size
        total_size = sum(len(e.data) for e in self._cache.values())
        while total_size > self.max_size_bytes and self._cache:
            key, entry = self._cache.popitem(last=False)
            total_size -= len(entry.data)
            self._stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(len(e.data) for e in self._cache.values())
            hit_rate = (
                self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) * 100
                if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            )
            
            return {
                'entries': len(self._cache),
                'total_size_mb': total_size / (1024 * 1024),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions']
            }


class OfflineModeManager:
    """Manage offline mode and request queuing"""
    
    def __init__(self, queue_dir: Optional[str] = None):
        self.queue_dir = Path(queue_dir or Path.home() / '.jarvis' / 'offline_queue')
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self._is_offline = False
        self._queue: List[Dict] = []
        self._lock = threading.Lock()
        self._max_queue_size = 1000
    
    @property
    def is_offline(self) -> bool:
        """Check if in offline mode"""
        return self._is_offline
    
    def set_offline(self, offline: bool) -> None:
        """Set offline mode"""
        self._is_offline = offline
    
    def check_connectivity(self, test_url: str = "https://api.openai.com") -> bool:
        """Check network connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen(test_url, timeout=5)
            return True
        except Exception:
            self._is_offline = True
            return False
    
    def queue_request(
        self,
        method: str,
        url: str,
        headers: Dict,
        body: Optional[bytes] = None,
        priority: int = 0
    ) -> str:
        """Queue request for later execution"""
        request_id = hashlib.md5(f"{time.time()}:{url}".encode()).hexdigest()[:16]
        
        request = {
            'id': request_id,
            'method': method,
            'url': url,
            'headers': headers,
            'body': body.hex() if body else None,
            'priority': priority,
            'queued_at': time.time()
        }
        
        with self._lock:
            if len(self._queue) < self._max_queue_size:
                self._queue.append(request)
        
        # Persist to disk
        self._save_queue()
        
        return request_id
    
    def get_queued_requests(self) -> List[Dict]:
        """Get all queued requests"""
        with self._lock:
            return sorted(self._queue, key=lambda r: -r['priority'])
    
    def remove_queued_request(self, request_id: str) -> bool:
        """Remove request from queue"""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req['id'] == request_id:
                    self._queue.pop(i)
                    self._save_queue()
                    return True
            return False
    
    def _save_queue(self) -> None:
        """Save queue to disk"""
        try:
            queue_file = self.queue_dir / 'queue.json'
            with open(queue_file, 'w') as f:
                json.dump(self._queue, f)
        except Exception:
            pass
    
    def _load_queue(self) -> None:
        """Load queue from disk"""
        try:
            queue_file = self.queue_dir / 'queue.json'
            if queue_file.exists():
                with open(queue_file, 'r') as f:
                    self._queue = json.load(f)
        except Exception:
            pass
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                'queued_requests': len(self._queue),
                'max_queue_size': self._max_queue_size,
                'is_offline': self._is_offline
            }


class BandwidthAdapter:
    """Adapt to network bandwidth conditions"""
    
    def __init__(self):
        self._measurements: List[float] = []
        self._max_measurements = 10
        self._current_bandwidth: Optional[float] = None
        self._latency: Optional[float] = None
    
    def measure_bandwidth(
        self,
        test_url: str = "https://api.openai.com",
        test_size: int = 1024
    ) -> Optional[float]:
        """Measure current bandwidth"""
        try:
            import urllib.request
            
            start = time.time()
            response = urllib.request.urlopen(test_url, timeout=10)
            data = response.read(test_size)
            elapsed = time.time() - start
            
            if elapsed > 0:
                bandwidth = len(data) / elapsed  # bytes per second
                
                self._measurements.append(bandwidth)
                if len(self._measurements) > self._max_measurements:
                    self._measurements.pop(0)
                
                self._current_bandwidth = sum(self._measurements) / len(self._measurements)
                return self._current_bandwidth
        except Exception:
            pass
        
        return None
    
    def measure_latency(self, test_url: str = "https://api.openai.com") -> Optional[float]:
        """Measure network latency"""
        try:
            import urllib.request
            
            start = time.time()
            request = urllib.request.Request(test_url, method='HEAD')
            urllib.request.urlopen(request, timeout=5)
            self._latency = (time.time() - start) * 1000  # ms
            return self._latency
        except Exception:
            return None
    
    def get_adapted_settings(self) -> Dict[str, Any]:
        """Get settings adapted to current bandwidth"""
        bandwidth = self._current_bandwidth or 1000000  # Default 1 MB/s
        
        # Low bandwidth (< 100 KB/s)
        if bandwidth < 100000:
            return {
                'batch_size': 5,
                'timeout': 30,
                'retry_delay': 5,
                'compression': True,
                'quality': 'low'
            }
        
        # Medium bandwidth (100 KB/s - 1 MB/s)
        if bandwidth < 1000000:
            return {
                'batch_size': 10,
                'timeout': 20,
                'retry_delay': 2,
                'compression': True,
                'quality': 'medium'
            }
        
        # High bandwidth (> 1 MB/s)
        return {
            'batch_size': 20,
            'timeout': 10,
            'retry_delay': 1,
            'compression': False,
            'quality': 'high'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandwidth statistics"""
        return {
            'current_bandwidth_kbps': (self._current_bandwidth or 0) / 1024,
            'latency_ms': self._latency,
            'measurements': len(self._measurements),
            'adapted_settings': self.get_adapted_settings()
        }


class NetworkOptimizer:
    """Main network optimization controller"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.request_compressor = RequestCompressor()
        self.response_cache = ResponseCache()
        self.offline_manager = OfflineModeManager(cache_dir)
        self.bandwidth_adapter = BandwidthAdapter()
        self._stats = RequestStats()
    
    def before_request(
        self,
        method: str,
        url: str,
        headers: Dict,
        body: Optional[bytes]
    ) -> Tuple[Dict, Optional[bytes], Optional[CacheEntry]]:
        """Prepare request with optimizations"""
        # Check cache first for GET requests
        if method.upper() == 'GET':
            cached = self.response_cache.get(method, url)
            if cached:
                return headers, body, cached
        
        # Check offline mode
        if self.offline_manager.is_offline:
            self.offline_manager.queue_request(method, url, headers, body)
            raise ConnectionError("Offline mode - request queued")
        
        # Compress body
        compressed_body = None
        if body:
            compressed_body, was_compressed = self.request_compressor.compress_body(body)
            if was_compressed:
                headers = {**headers, 'Content-Encoding': 'gzip'}
        
        self._stats.total_requests += 1
        if body:
            self._stats.total_bytes_sent += len(body)
        
        return headers, compressed_body or body, None
    
    def after_response(
        self,
        method: str,
        url: str,
        response_data: bytes,
        headers: Dict,
        status_code: int
    ) -> None:
        """Process response with optimizations"""
        self._stats.total_bytes_received += len(response_data)
        
        if status_code >= 400:
            self._stats.failed_requests += 1
        
        # Cache successful GET responses
        if method.upper() == 'GET' and status_code == 200:
            self.response_cache.set(method, url, response_data, headers, status_code)
    
    def optimize(self) -> Dict[str, Any]:
        """Run network optimization analysis"""
        # Measure bandwidth
        self.bandwidth_adapter.measure_bandwidth()
        self.bandwidth_adapter.measure_latency()
        
        return {
            'bandwidth': self.bandwidth_adapter.get_stats(),
            'cache': self.response_cache.get_stats(),
            'offline': self.offline_manager.get_queue_stats(),
            'compression': self.request_compressor.get_stats(),
            'requests': {
                'total': self._stats.total_requests,
                'failed': self._stats.failed_requests,
                'bytes_sent': self._stats.total_bytes_sent,
                'bytes_received': self._stats.total_bytes_received
            }
        }
    
    def set_offline_mode(self, offline: bool) -> None:
        """Set offline mode"""
        self.offline_manager.set_offline(offline)
    
    def is_online(self) -> bool:
        """Check if online"""
        return not self.offline_manager.is_offline and \
               self.offline_manager.check_connectivity()
    
    def get_report(self) -> str:
        """Generate network optimization report"""
        stats = self.optimize()
        
        lines = [
            "═══════════════════════════════════════════",
            "       NETWORK OPTIMIZATION REPORT",
            "═══════════════════════════════════════════",
            f"Online:           {'Yes' if self.is_online() else 'No'}",
            f"Bandwidth:        {stats['bandwidth']['current_bandwidth_kbps']:.0f} KB/s",
            f"Latency:          {stats['bandwidth']['latency_ms'] or 0:.0f} ms",
            "",
            "Cache:",
            f"  Entries:        {stats['cache']['entries']}",
            f"  Hit Rate:       {stats['cache']['hit_rate']:.1f}%",
            "",
            "Requests:",
            f"  Total:          {stats['requests']['total']}",
            f"  Failed:         {stats['requests']['failed']}",
            f"  Bytes Sent:     {stats['requests']['bytes_sent'] / 1024:.1f} KB",
            f"  Bytes Received: {stats['requests']['bytes_received'] / 1024:.1f} KB",
        ]
        
        if self.offline_manager.is_offline:
            lines.extend([
                "",
                "Offline Queue:",
                f"  Pending:        {stats['offline']['queued_requests']}",
            ])
        
        return "\n".join(lines)
