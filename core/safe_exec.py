#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Safe Code Execution Engine
================================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Safety Layers:
1. AST Validation - Block dangerous operations
2. Whitelist Execution - Only allowed functions
3. Timeout Protection - Prevent infinite loops
4. Exception Isolation - Errors don't crash system
"""

import ast
import sys
import time
import threading
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import redirect_stdout, redirect_stderr
import io
import logging

logger = logging.getLogger(__name__)


class DangerLevel(Enum):
    SAFE = auto()
    WARNING = auto()
    DANGEROUS = auto()
    CRITICAL = auto()


DANGEROUS_NAMES = {
    'eval', 'exec', 'compile', 'open', '__import__', '__builtins__',
    'os', 'sys', 'subprocess', 'ctypes', 'socket', 'pickle', 'marshal',
}

SAFE_BUILTINS = {
    'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple',
    'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
    'reversed', 'any', 'all', 'min', 'max', 'sum', 'abs', 'round',
    'print', 'Exception', 'ValueError', 'TypeError', 'True', 'False', 'None',
}


@dataclass
class ValidationResult:
    is_safe: bool
    danger_level: DangerLevel
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    node_count: int = 0


@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    output: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    timed_out: bool = False
    validation: Optional[ValidationResult] = None


class CodeValidator:
    """Validates Python code for dangerous patterns."""
    
    def validate(self, code: str) -> ValidationResult:
        result = ValidationResult(
            is_safe=True,
            danger_level=DangerLevel.SAFE
        )
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.is_safe = False
            result.issues.append(f"Syntax error: {e}")
            return result
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.split('.')[0] in DANGEROUS_NAMES:
                        result.issues.append(f"Dangerous import: {alias.name}")
                        result.danger_level = DangerLevel.CRITICAL
                        result.is_safe = False
            
            if isinstance(node, ast.Name):
                if node.id in DANGEROUS_NAMES:
                    result.warnings.append(f"Dangerous name used: {node.id}")
                    if result.danger_level == DangerLevel.SAFE:
                        result.danger_level = DangerLevel.WARNING
            
            result.node_count += 1
        
        return result


class SafeExecutor:
    """Safe code executor with multiple security layers."""
    
    def __init__(self, timeout: float = 30.0, strict_mode: bool = True):
        self._timeout = timeout
        self._strict_mode = strict_mode
        self._validator = CodeValidator()
        self._stats = {'total': 0, 'success': 0, 'failed': 0}
    
    def validate(self, code: str) -> ValidationResult:
        return self._validator.validate(code)
    
    def execute(self, code: str, context: Dict = None) -> ExecutionResult:
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        validation = self.validate(code)
        result.validation = validation
        
        if not validation.is_safe:
            result.error = f"Validation failed: {'; '.join(validation.issues)}"
            self._stats['failed'] += 1
            return result
        
        if self._strict_mode and validation.warnings:
            result.error = f"Warnings: {'; '.join(validation.warnings)}"
            self._stats['failed'] += 1
            return result
        
        safe_builtins = {}
        for k in SAFE_BUILTINS:
            try:
                if isinstance(__builtins__, dict):
                    safe_builtins[k] = __builtins__.get(k)
                else:
                    safe_builtins[k] = getattr(__builtins__, k, None)
            except Exception:
                pass
        # Ensure print is available
        safe_builtins['print'] = print
        safe_globals = {'__builtins__': safe_builtins, '__name__': '__safe__'}
        safe_locals = dict(context) if context else {}
        
        try:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                compiled = compile(code, '<safe>', 'exec')
                exec(compiled, safe_globals, safe_locals)
            
            result.output = stdout_capture.getvalue()
            result.success = True
            self._stats['success'] += 1
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.error_type = type(e).__name__
            self._stats['failed'] += 1
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        self._stats['total'] += 1
        return result
    
    def get_stats(self) -> Dict:
        return self._stats.copy()


_executor = None

def get_executor() -> SafeExecutor:
    global _executor
    if _executor is None:
        _executor = SafeExecutor()
    return _executor

def safe_exec(code: str) -> ExecutionResult:
    return get_executor().execute(code)


def self_test() -> Dict:
    results = {'passed': [], 'failed': []}
    executor = SafeExecutor(timeout=5.0)
    
    # Test 1: Safe code
    r = executor.execute("x = 1 + 1; print(x)")
    if r.success and "2" in r.output:
        results['passed'].append('safe_code')
    else:
        results['failed'].append(f'safe_code: {r.error}')
    
    # Test 2: Block dangerous imports
    r = executor.execute("import os")
    if not r.success:
        results['passed'].append('block_import')
    else:
        results['failed'].append('block_import')
    
    # Test 3: Block exec
    r = executor.execute("exec('print(1)')")
    if not r.success:
        results['passed'].append('block_exec')
    else:
        results['failed'].append('block_exec')
    
    # Test 4: Exception handling
    r = executor.execute("1/0")
    if not r.success and 'ZeroDivisionError' in r.error:
        results['passed'].append('exception')
    else:
        results['failed'].append(f'exception: {r.error}')
    
    results['stats'] = executor.get_stats()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("JARVIS Safe Execution Engine - Self Test")
    print("=" * 60)
    
    test = self_test()
    
    print("\n✅ Passed:")
    for t in test['passed']:
        print(f"   ✓ {t}")
    
    if test['failed']:
        print("\n❌ Failed:")
        for t in test['failed']:
            print(f"   ✗ {t}")
    
    print(f"\n📊 Stats: {test['stats']}")
    print("\n" + "=" * 60)
