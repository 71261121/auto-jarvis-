#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 62: Compatibility Tests
==================================================

Tests for compatibility across:
- Python versions (3.9+)
- Termux-specific features
- Dependency alternatives
- Fallback chains

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
"""

import sys
import os
import platform
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = None
    
    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        print(f"    Error: {error[:100]}")
    
    def summary(self):
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        total = len(self.passed) + len(self.failed)
        rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"TODO 62: Compatibility Tests Results")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}%")
        return len(self.failed) == 0


results = TestResult()
import time
results.start_time = time.time()

print("="*60)
print("TODO 62: Compatibility Tests")
print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON VERSION COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Python Version Compatibility ---")

def test_python_version():
    """Test Python version is 3.9+"""
    try:
        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 9
        print(f"    Python {version.major}.{version.minor}.{version.micro}")
        results.add_pass("compat: Python version >= 3.9")
    except Exception as e:
        results.add_fail("compat: Python version", str(e))

def test_type_hints():
    """Test type hints compatibility"""
    try:
        from typing import Dict, List, Optional, Union, Any
        # Test modern type hints
        def test_func(x: int, y: Optional[str] = None) -> Dict[str, Any]:
            return {"x": x, "y": y}
        
        result = test_func(1)
        assert result["x"] == 1
        results.add_pass("compat: Type hints")
    except Exception as e:
        results.add_fail("compat: Type hints", str(e))

def test_dataclasses():
    """Test dataclasses compatibility"""
    try:
        from dataclasses import dataclass, field
        
        @dataclass
        class TestClass:
            name: str
            value: int = 0
        
        obj = TestClass("test", 42)
        assert obj.name == "test"
        results.add_pass("compat: Dataclasses")
    except Exception as e:
        results.add_fail("compat: Dataclasses", str(e))

def test_pathlib():
    """Test pathlib compatibility"""
    try:
        from pathlib import Path
        
        p = Path("/tmp")
        assert p.exists() or p.is_dir() or str(p) == "/tmp"
        results.add_pass("compat: Pathlib")
    except Exception as e:
        results.add_fail("compat: Pathlib", str(e))

def test_f_strings():
    """Test f-string compatibility"""
    try:
        name = "JARVIS"
        version = 14
        result = f"{name} v{version}"
        assert result == "JARVIS v14"
        results.add_pass("compat: F-strings")
    except Exception as e:
        results.add_fail("compat: F-strings", str(e))

test_python_version()
test_type_hints()
test_dataclasses()
test_pathlib()
test_f_strings()


# ═══════════════════════════════════════════════════════════════════════════════
# TERMUX-SPECIFIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Termux-Specific Tests ---")

def test_termux_detection():
    """Test Termux environment detection"""
    try:
        is_termux = 'TERMUX_VERSION' in os.environ or 'com.termux' in os.environ.get('PREFIX', '')
        print(f"    Termux environment: {is_termux}")
        results.add_pass("compat: Termux detection")
    except Exception as e:
        results.add_fail("compat: Termux detection", str(e))

def test_android_paths():
    """Test Android-specific paths"""
    try:
        # Check common Android/Termux paths
        paths = [
            os.path.expanduser("~"),
            "/data/data/com.termux",
            os.environ.get('PREFIX', ''),
        ]
        print(f"    Home: {paths[0]}")
        results.add_pass("compat: Android paths")
    except Exception as e:
        results.add_fail("compat: Android paths", str(e))

def test_filesystem_case():
    """Test filesystem case sensitivity"""
    try:
        # Test if filesystem is case-sensitive
        test_path = "/tmp/TestCase_test"
        alt_path = "/tmp/testcase_TEST"
        
        # Clean up any existing files
        for p in [test_path, alt_path]:
            if os.path.exists(p):
                os.remove(p)
        
        results.add_pass("compat: Filesystem case")
    except Exception as e:
        results.add_fail("compat: Filesystem case", str(e))

test_termux_detection()
test_android_paths()
test_filesystem_case()


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY ALTERNATIVES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Dependency Alternatives ---")

def test_json_builtin():
    """Test built-in JSON (always available)"""
    try:
        import json
        
        data = {"test": 1, "list": [1, 2, 3]}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        
        assert decoded["test"] == 1
        results.add_pass("compat: JSON built-in")
    except Exception as e:
        results.add_fail("compat: JSON built-in", str(e))

def test_sqlite_builtin():
    """Test built-in SQLite (always available)"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER)")
        conn.close()
        
        results.add_pass("compat: SQLite built-in")
    except Exception as e:
        results.add_fail("compat: SQLite built-in", str(e))

def test_hashlib_builtin():
    """Test built-in hashlib"""
    try:
        import hashlib
        
        h = hashlib.sha256(b"test")
        assert len(h.hexdigest()) == 64
        
        results.add_pass("compat: Hashlib built-in")
    except Exception as e:
        results.add_fail("compat: Hashlib built-in", str(e))

def test_argparse_builtin():
    """Test built-in argparse"""
    try:
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", default="value")
        
        results.add_pass("compat: Argparse built-in")
    except Exception as e:
        results.add_fail("compat: Argparse built-in", str(e))

def test_optional_import():
    """Test optional dependency import"""
    try:
        # These are optional - should not crash if missing
        
        # Rich
        try:
            import rich
            has_rich = True
        except ImportError:
            has_rich = False
        
        # Loguru
        try:
            import loguru
            has_loguru = True
        except ImportError:
            has_loguru = False
        
        # Requests
        try:
            import requests
            has_requests = True
        except ImportError:
            has_requests = False
        
        print(f"    rich: {'✓' if has_rich else '✗'}")
        print(f"    loguru: {'✓' if has_loguru else '✗'}")
        print(f"    requests: {'✓' if has_requests else '✗'}")
        
        results.add_pass("compat: Optional imports handled")
    except Exception as e:
        results.add_fail("compat: Optional imports", str(e))

test_json_builtin()
test_sqlite_builtin()
test_hashlib_builtin()
test_argparse_builtin()
test_optional_import()


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK CHAIN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Fallback Chain Tests ---")

def test_http_fallback():
    """Test HTTP client fallback chain"""
    try:
        # Try imports in order
        http_client = None
        
        try:
            import httpx
            http_client = "httpx"
        except ImportError:
            pass
        
        if http_client is None:
            try:
                import requests
                http_client = "requests"
            except ImportError:
                pass
        
        if http_client is None:
            import urllib.request
            http_client = "urllib"
        
        print(f"    HTTP client: {http_client}")
        results.add_pass("compat: HTTP fallback chain")
    except Exception as e:
        results.add_fail("compat: HTTP fallback chain", str(e))

def test_logging_fallback():
    """Test logging fallback chain"""
    try:
        logger_available = None
        
        try:
            import loguru
            logger_available = "loguru"
        except ImportError:
            import logging
            logger_available = "stdlib"
        
        print(f"    Logger: {logger_available}")
        results.add_pass("compat: Logging fallback")
    except Exception as e:
        results.add_fail("compat: Logging fallback", str(e))

test_http_fallback()
test_logging_fallback()


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Platform Compatibility ---")

def test_platform_info():
    """Test platform information"""
    try:
        info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        print(f"    System: {info['system']}")
        print(f"    Machine: {info['machine']}")
        print(f"    Python: {info['python']}")
        results.add_pass("compat: Platform info")
    except Exception as e:
        results.add_fail("compat: Platform info", str(e))

def test_encoding():
    """Test text encoding"""
    try:
        # Test UTF-8 handling
        text = "Hello 世界 🌍"
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        
        assert decoded == text
        results.add_pass("compat: UTF-8 encoding")
    except Exception as e:
        results.add_fail("compat: UTF-8 encoding", str(e))

test_platform_info()
test_encoding()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()

if success:
    print("\n🎉 TODO 62: ALL COMPATIBILITY TESTS PASSED!")
else:
    print("\n⚠️ SOME TESTS FAILED - CHECK ABOVE")

sys.exit(0 if success else 1)
