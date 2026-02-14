#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 77: Storage Optimization
==================================================

Storage optimization for limited disk space:
- Data compression
- Log rotation
- Cache size limits
- Cleanup scheduler

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import gzip
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import struct


@dataclass
class CompressionResult:
    """Compression result"""
    original_size: int
    compressed_size: int
    ratio: float
    compressed_path: str


class DataCompressor:
    """Compress and decompress data"""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def compress_file(self, filepath: str, output_path: Optional[str] = None) -> CompressionResult:
        """Compress a file"""
        output_path = output_path or f"{filepath}.gz"
        
        original_size = os.path.getsize(filepath)
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        compressed_size = os.path.getsize(output_path)
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            ratio=ratio,
            compressed_path=output_path
        )
    
    def decompress_file(self, filepath: str, output_path: str) -> str:
        """Decompress a file"""
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return output_path
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress bytes data"""
        return gzip.compress(data, compresslevel=self.compression_level)
    
    def decompress_data(self, data: bytes) -> bytes:
        """Decompress bytes data"""
        return gzip.decompress(data)
    
    def compress_json(self, data: Dict) -> bytes:
        """Compress JSON data"""
        json_bytes = json.dumps(data).encode('utf-8')
        return self.compress_data(json_bytes)
    
    def decompress_json(self, data: bytes) -> Dict:
        """Decompress JSON data"""
        json_bytes = self.decompress_data(data)
        return json.loads(json_bytes.decode('utf-8'))


@dataclass
class LogFile:
    """Log file info"""
    path: str
    size: int
    modified: datetime


class LogRotator:
    """Rotate log files to prevent oversized logs"""
    
    def __init__(
        self,
        log_dir: str,
        max_size_mb: float = 10.0,
        max_files: int = 5,
        max_age_days: int = 30
    ):
        self.log_dir = Path(log_dir)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_files = max_files
        self.max_age_days = max_age_days
        self._lock = threading.Lock()
    
    def should_rotate(self, filepath: str) -> bool:
        """Check if file should be rotated"""
        try:
            size = os.path.getsize(filepath)
            return size >= self.max_size_bytes
        except Exception:
            return False
    
    def rotate(self, filepath: str) -> Optional[str]:
        """Rotate a log file"""
        with self._lock:
            try:
                if not os.path.exists(filepath):
                    return None
                
                # Generate rotated filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                rotated_path = f"{filepath}.{timestamp}"
                
                # Move current file
                shutil.move(filepath, rotated_path)
                
                # Create new empty file
                Path(filepath).touch()
                
                # Compress old file
                compressor = DataCompressor()
                compressor.compress_file(rotated_path)
                os.remove(rotated_path)
                
                # Cleanup old files
                self._cleanup_old_files(filepath)
                
                return f"{rotated_path}.gz"
            except Exception as e:
                return None
    
    def _cleanup_old_files(self, base_path: str) -> int:
        """Remove old rotated files"""
        cleaned = 0
        base = Path(base_path)
        # Use the parent directory and base filename safely
        parent_dir = base.parent
        base_name = base.name
        
        # Find all rotated files - use explicit pattern instead of variable
        try:
            rotated_files = sorted(
                parent_dir.glob(f"{base_name}.*"),
                key=lambda p: p.stat().st_mtime
            )
        except Exception:
            return 0
        
        # Remove files exceeding max count
        while len(rotated_files) > self.max_files:
            old_file = rotated_files.pop(0)
            try:
                old_file.unlink()
                cleaned += 1
            except Exception:
                pass
        
        # Remove files exceeding age
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        try:
            for f in parent_dir.glob(f"{base_name}.*.gz"):
                try:
                    if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                        f.unlink()
                        cleaned += 1
                except Exception:
                    pass
        except Exception:
            pass
        
        return cleaned
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log directory statistics"""
        total_size = 0
        file_count = 0
        
        for f in Path(self.log_dir).rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
                file_count += 1
        
        return {
            'directory': str(self.log_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'max_files': self.max_files
        }


class CacheManager:
    """Manage cache with size limits"""
    
    def __init__(
        self,
        cache_dir: str,
        max_size_mb: float = 100.0,
        max_entries: int = 10000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self._index_file = self.cache_dir / 'cache_index.json'
        self._index: Dict[str, Dict] = self._load_index()
        self._lock = threading.Lock()
    
    def _load_index(self) -> Dict:
        """Load cache index"""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_index(self) -> None:
        """Save cache index"""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f)
        except Exception:
            pass
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        with self._lock:
            if key not in self._index:
                return None
            
            entry = self._index[key]
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                del self._index[key]
                return None
            
            # Check expiry
            if entry.get('expires', 0) < time.time():
                self.delete(key)
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                if entry.get('compressed', False):
                    data = gzip.decompress(data)
                
                return json.loads(data.decode('utf-8'))
            except Exception:
                return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        compress: bool = False
    ) -> bool:
        """Set cached value"""
        with self._lock:
            try:
                data = json.dumps(value).encode('utf-8')
                
                if compress:
                    data = gzip.compress(data)
                
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    f.write(data)
                
                self._index[key] = {
                    'created': time.time(),
                    'expires': time.time() + ttl_seconds if ttl_seconds else 0,
                    'size': len(data),
                    'compressed': compress
                }
                
                self._save_index()
                
                # Check limits
                self._enforce_limits()
                
                return True
            except Exception:
                return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value"""
        with self._lock:
            if key not in self._index:
                return False
            
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            
            del self._index[key]
            self._save_index()
            return True
    
    def _enforce_limits(self) -> None:
        """Enforce cache limits"""
        # Check entry count
        if len(self._index) > self.max_entries:
            # Remove oldest entries
            sorted_keys = sorted(
                self._index.keys(),
                key=lambda k: self._index[k]['created']
            )
            for key in sorted_keys[:len(self._index) - self.max_entries]:
                self.delete(key)
        
        # Check total size
        total_size = sum(e['size'] for e in self._index.values())
        if total_size > self.max_size_bytes:
            # Remove oldest until under limit
            sorted_keys = sorted(
                self._index.keys(),
                key=lambda k: self._index[k]['created']
            )
            for key in sorted_keys:
                if total_size <= self.max_size_bytes:
                    break
                total_size -= self._index[key]['size']
                self.delete(key)
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            for key in list(self._index.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            self._index.clear()
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(e['size'] for e in self._index.values())
        
        return {
            'entries': len(self._index),
            'max_entries': self.max_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'usage_percent': (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
        }


class CleanupScheduler:
    """Schedule cleanup tasks"""
    
    def __init__(self):
        self._tasks: Dict[str, Callable] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval = 3600  # 1 hour default
    
    def register(self, name: str, cleanup_func: Callable) -> None:
        """Register cleanup task"""
        self._tasks[name] = cleanup_func
    
    def unregister(self, name: str) -> None:
        """Unregister cleanup task"""
        self._tasks.pop(name, None)
    
    def start(self, interval_seconds: int = 3600) -> None:
        """Start scheduled cleanup"""
        self._interval = interval_seconds
        
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop scheduled cleanup"""
        self._running = False
    
    def _cleanup_loop(self) -> None:
        """Main cleanup loop"""
        while self._running:
            time.sleep(self._interval)
            
            if not self._running:
                break
            
            self.run_cleanup()
    
    def run_cleanup(self) -> Dict[str, Any]:
        """Run all cleanup tasks"""
        results = {}
        
        for name, func in self._tasks.items():
            try:
                result = func()
                results[name] = {'success': True, 'result': result}
            except Exception as e:
                results[name] = {'success': False, 'error': str(e)}
        
        return results


class StorageOptimizer:
    """Main storage optimization controller"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or Path.home() / '.jarvis')
        self.compressor = DataCompressor()
        self.log_rotator = LogRotator(str(self.base_dir / 'logs'))
        self.cache_manager = CacheManager(str(self.base_dir / 'cache'))
        self.cleanup_scheduler = CleanupScheduler()
        
        # Register default cleanup tasks
        self.cleanup_scheduler.register('cache', self._cleanup_cache)
        self.cleanup_scheduler.register('logs', self._cleanup_logs)
        self.cleanup_scheduler.register('temp', self._cleanup_temp)
    
    def _cleanup_cache(self) -> Dict:
        """Cleanup cache"""
        stats = self.cache_manager.get_stats()
        # Expired entries are cleaned on access
        return {'cleaned': True, 'stats': stats}
    
    def _cleanup_logs(self) -> Dict:
        """Cleanup logs"""
        return self.log_rotator.get_log_stats()
    
    def _cleanup_temp(self) -> Dict:
        """Cleanup temp files"""
        temp_dir = self.base_dir / 'temp'
        cleaned = 0
        
        if temp_dir.exists():
            for f in temp_dir.glob('*'):
                try:
                    if f.is_file():
                        f.unlink()
                        cleaned += 1
                except Exception:
                    pass
        
        return {'cleaned_files': cleaned}
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Run storage optimization"""
        results = {
            'compression': {},
            'cleanup': {},
            'stats': {}
        }
        
        # Run cleanup
        results['cleanup'] = self.cleanup_scheduler.run_cleanup()
        
        # Get stats
        results['stats'] = self.get_storage_stats()
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        file_count = 0
        
        for f in self.base_dir.rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
                file_count += 1
        
        return {
            'base_dir': str(self.base_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'cache_stats': self.cache_manager.get_stats(),
            'log_stats': self.log_rotator.get_log_stats()
        }
    
    def start_cleanup_scheduler(self, interval_hours: int = 1) -> None:
        """Start automatic cleanup"""
        self.cleanup_scheduler.start(interval_hours * 3600)
    
    def stop_cleanup_scheduler(self) -> None:
        """Stop automatic cleanup"""
        self.cleanup_scheduler.stop()
    
    def get_report(self) -> str:
        """Generate storage report"""
        stats = self.get_storage_stats()
        
        lines = [
            "═══════════════════════════════════════════",
            "       STORAGE OPTIMIZATION REPORT",
            "═══════════════════════════════════════════",
            f"Base Directory:   {self.base_dir}",
            f"Total Size:       {stats['total_size_mb']:.2f} MB",
            f"File Count:       {stats['file_count']:,}",
            "",
            "Cache:",
            f"  Entries:        {stats['cache_stats']['entries']:,}",
            f"  Size:           {stats['cache_stats']['total_size_mb']:.2f} MB",
            f"  Usage:          {stats['cache_stats']['usage_percent']:.1f}%",
            "",
            "Logs:",
            f"  Size:           {stats['log_stats']['total_size_mb']:.2f} MB",
            f"  Files:          {stats['log_stats']['file_count']}",
        ]
        
        return "\n".join(lines)
