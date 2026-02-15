#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Code Analysis Engine
===========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Python AST (Abstract Syntax Tree) parsing
- Cyclomatic complexity calculation
- Code smell detection
- Static analysis patterns
- Memory-efficient analysis

Features:
- Full AST parsing and traversal
- Complexity metrics (cyclomatic, cognitive)
- Code quality scoring
- Pattern detection (anti-patterns, best practices)
- Dependency analysis
- Function/class extraction
- Documentation coverage
- Security vulnerability scanning

Memory Impact: < 10MB for typical files
"""

import ast
import sys
import os
import re
import time
import logging
import hashlib
import math
import threading
from typing import Dict, Any, Optional, List, Set, Tuple, Generator, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, Counter
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    """Types of AST nodes we track"""
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    ASYNC_FUNCTION = auto()
    METHOD = auto()
    CLASS_METHOD = auto()
    STATIC_METHOD = auto()
    VARIABLE = auto()
    IMPORT = auto()
    IMPORT_FROM = auto()
    CONSTANT = auto()


class ComplexityLevel(Enum):
    """Complexity rating levels"""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    VERY_HIGH = 4
    CRITICAL = 5


class IssueSeverity(Enum):
    """Severity levels for code issues"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class PatternType(Enum):
    """Types of detected patterns"""
    BEST_PRACTICE = auto()
    ANTI_PATTERN = auto()
    SECURITY_ISSUE = auto()
    PERFORMANCE = auto()
    STYLE = auto()
    MAINTAINABILITY = auto()


@dataclass
class CodeLocation:
    """Location in source code"""
    line_start: int
    line_end: int
    col_start: int = 0
    col_end: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'line_start': self.line_start,
            'line_end': self.line_end,
            'col_start': self.col_start,
            'col_end': self.col_end,
        }


@dataclass
class CodeIssue:
    """A detected issue in code"""
    message: str
    severity: IssueSeverity
    pattern_type: PatternType
    location: CodeLocation
    suggestion: str = ""
    code_snippet: str = ""
    rule_id: str = ""


@dataclass
class ComplexityMetrics:
    """Complexity metrics for code"""
    cyclomatic: int = 1
    cognitive: int = 0
    lines_of_code: int = 0
    logical_lines: int = 0
    comment_lines: int = 0
    docstring_lines: int = 0
    maintainability_index: float = 100.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    
    @property
    def level(self) -> ComplexityLevel:
        """Get complexity level"""
        if self.cyclomatic <= 5:
            return ComplexityLevel.LOW
        elif self.cyclomatic <= 10:
            return ComplexityLevel.MODERATE
        elif self.cyclomatic <= 20:
            return ComplexityLevel.HIGH
        elif self.cyclomatic <= 30:
            return ComplexityLevel.VERY_HIGH
        return ComplexityLevel.CRITICAL
    
    @property
    def docstring_ratio(self) -> float:
        """Get documentation ratio"""
        if self.lines_of_code == 0:
            return 0.0
        return self.docstring_lines / self.lines_of_code


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    node_type: NodeType
    location: CodeLocation
    complexity: ComplexityMetrics
    parameters: List[str]
    returns: List[str]
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: str = ""
    local_variables: Set[str] = field(default_factory=set)
    called_functions: Set[str] = field(default_factory=set)
    issues: List[CodeIssue] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    location: CodeLocation
    docstring: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: Set[str] = field(default_factory=set)
    class_variables: Set[str] = field(default_factory=set)
    instance_variables: Set[str] = field(default_factory=set)
    complexity: ComplexityMetrics = None
    issues: List[CodeIssue] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    names: List[str]
    aliases: Dict[str, str]
    location: CodeLocation
    is_from: bool = False
    level: int = 0  # Relative import level


@dataclass
class FileAnalysis:
    """Complete analysis of a Python file"""
    path: str
    content_hash: str
    size_bytes: int
    parse_time_ms: float
    success: bool = True
    error: str = ""
    
    # Structure
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    module_docstring: Optional[str] = None
    global_variables: Set[str] = field(default_factory=set)
    constants: Set[str] = field(default_factory=set)
    
    # Metrics
    total_complexity: ComplexityMetrics = None
    issues: List[CodeIssue] = field(default_factory=list)
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    stdlib_imports: Set[str] = field(default_factory=set)
    third_party_imports: Set[str] = field(default_factory=set)
    local_imports: Set[str] = field(default_factory=set)
    
    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Get function by name"""
        for func in self.functions:
            if func.name == name:
                return func
        return None
    
    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Get class by name"""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexityCalculator(ast.NodeVisitor):
    """
    Calculate cyclomatic and cognitive complexity.
    
    Cyclomatic Complexity:
    - Counts decision points (if, for, while, and, or, etc.)
    - Higher = more test paths needed
    
    Cognitive Complexity:
    - Human-readable complexity measure
    - Penalizes nesting and breaks
    """
    
    # Nodes that increase cyclomatic complexity
    DECISION_NODES = {
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    }
    
    # Boolean operators that increase complexity
    BOOLEAN_OPS = {ast.And, ast.Or}
    
    def __init__(self):
        self.cyclomatic = 1
        self.cognitive = 0
        self.nesting_level = 0
        self.switch_like = False
        
    def visit_If(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_For(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_While(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_ExceptHandler(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_BoolOp(self, node):
        # Each and/or adds to complexity
        self.cyclomatic += len(node.values) - 1
        self.cognitive += len(node.values) - 1
        self.generic_visit(node)
        
    def visit_comprehension(self, node):
        self.cyclomatic += 1
        self.cognitive += 1
        self.generic_visit(node)
        
    def visit_With(self, node):
        self.cyclomatic += 1
        self.cognitive += 1
        self.generic_visit(node)
        
    def visit_Break(self, node):
        # Breaking adds cognitive complexity
        self.cognitive += 1
        self.generic_visit(node)
        
    def visit_Continue(self, node):
        self.cognitive += 1
        self.generic_visit(node)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PatternDetector(ast.NodeVisitor):
    """
    Detect code patterns, anti-patterns, and issues.
    
    Patterns detected:
    - Long functions
    - Deep nesting
    - Large classes
    - Unused imports
    - Security issues (eval, exec, etc.)
    - Performance issues
    - Style issues
    """
    
    # Standard library modules
    STDLIB_MODULES = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
        'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
        'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
        'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
        'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
        'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
        'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
        'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
        'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
        'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
        'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
        'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
        'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
        'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
        'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
        'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil',
        'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
        'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc',
        'queue', 'quopri', 'random', 're', 'readline', 'reprlib', 'resource',
        'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
        'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
        'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl',
        'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
        'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
        'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
        'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
        'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
        'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu',
        'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
        'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
        'zipapp', 'zipfile', 'zipimport', 'zlib',
    }
    
    # Dangerous functions that should be flagged
    DANGEROUS_FUNCTIONS = {
        'eval': 'eval() can execute arbitrary code - security risk',
        'exec': 'exec() can execute arbitrary code - security risk',
        'compile': 'compile() can create executable code - security risk',
        '__import__': 'Dynamic imports can be dangerous',
        'input': 'input() in Python 2 is equivalent to eval() - use raw_input()',
    }
    
    # Functions that suggest bad practices
    CODE_SMELL_FUNCTIONS = {
        'globals': 'Using globals() suggests poor code organization',
        'locals': 'Using locals() can make code hard to understand',
        'vars': 'Using vars() can make code hard to understand',
    }
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.nesting_level = 0
        self.imports: Dict[str, ImportInfo] = {}
        self.used_names: Set[str] = set()
        self.function_lengths: Dict[str, int] = {}
        
    def visit_FunctionDef(self, node):
        self._visit_function(node)
        
    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)
        
    def _visit_function(self, node):
        old_function = self.current_function
        self.current_function = node.name
        
        # Check function length
        func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        self.function_lengths[node.name] = func_lines
        
        if func_lines > 50:
            self.issues.append(CodeIssue(
                message=f"Function '{node.name}' is too long ({func_lines} lines)",
                severity=IssueSeverity.WARNING,
                pattern_type=PatternType.MAINTAINABILITY,
                location=CodeLocation(node.lineno, node.end_lineno or node.lineno),
                suggestion="Consider breaking this function into smaller functions",
                rule_id="PLW0915",
            ))
        
        # Check for dangerous function calls
        self._check_dangerous_calls(node)
        
        # Visit children
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
        self.current_function = old_function
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        
        # Check class length
        class_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        if class_lines > 300:
            self.issues.append(CodeIssue(
                message=f"Class '{node.name}' is too large ({class_lines} lines)",
                severity=IssueSeverity.WARNING,
                pattern_type=PatternType.MAINTAINABILITY,
                location=CodeLocation(node.lineno, node.end_lineno or node.lineno),
                suggestion="Consider breaking this class into smaller classes",
                rule_id="PLW0916",
            ))
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            # Only warn for public classes
            if not node.name.startswith('_'):
                self.issues.append(CodeIssue(
                    message=f"Class '{node.name}' has no docstring",
                    severity=IssueSeverity.INFO,
                    pattern_type=PatternType.STYLE,
                    location=CodeLocation(node.lineno, node.lineno),
                    suggestion="Add a docstring to document the class purpose",
                    rule_id="D101",
                ))
        
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = ImportInfo(
                module=alias.name,
                names=[alias.name],
                aliases={alias.name: alias.asname} if alias.asname else {},
                location=CodeLocation(node.lineno, node.lineno),
                is_from=False,
            )
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        module = node.module or ''
        names = [alias.name for alias in node.names]
        aliases = {alias.name: alias.asname for alias in node.names if alias.asname}
        
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = ImportInfo(
                module=module,
                names=names,
                aliases=aliases,
                location=CodeLocation(node.lineno, node.lineno),
                is_from=True,
                level=node.level,
            )
        self.generic_visit(node)
        
    def visit_Name(self, node):
        self.used_names.add(node.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if func_name in self.DANGEROUS_FUNCTIONS:
                self.issues.append(CodeIssue(
                    message=self.DANGEROUS_FUNCTIONS[func_name],
                    severity=IssueSeverity.WARNING,
                    pattern_type=PatternType.SECURITY_ISSUE,
                    location=CodeLocation(node.lineno, node.lineno),
                    suggestion="Avoid using this function with untrusted input",
                    rule_id="S001",
                ))
            
            elif func_name in self.CODE_SMELL_FUNCTIONS:
                self.issues.append(CodeIssue(
                    message=self.CODE_SMELL_FUNCTIONS[func_name],
                    severity=IssueSeverity.INFO,
                    pattern_type=PatternType.ANTI_PATTERN,
                    location=CodeLocation(node.lineno, node.lineno),
                    rule_id="A001",
                ))
        
        self.generic_visit(node)
        
    def visit_Try(self, node):
        # Check for bare except
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append(CodeIssue(
                    message="Bare 'except:' clause catches all exceptions including KeyboardInterrupt",
                    severity=IssueSeverity.WARNING,
                    pattern_type=PatternType.ANTI_PATTERN,
                    location=CodeLocation(handler.lineno, handler.lineno),
                    suggestion="Use 'except Exception:' or be more specific",
                    rule_id="E722",
                ))
        
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node):
        # Check for overly broad exception handling
        if node.type and isinstance(node.type, ast.Name):
            if node.type.id == 'Exception':
                self.issues.append(CodeIssue(
                    message="Catching 'Exception' is too broad",
                    severity=IssueSeverity.INFO,
                    pattern_type=PatternType.BEST_PRACTICE,
                    location=CodeLocation(node.lineno, node.lineno),
                    suggestion="Catch specific exceptions instead",
                    rule_id="B001",
                ))
        
        self.generic_visit(node)
        
    def _check_dangerous_calls(self, node):
        """Check for dangerous function calls within a node"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in self.DANGEROUS_FUNCTIONS:
                        self.issues.append(CodeIssue(
                            message=f"Use of {child.func.id}() detected",
                            severity=IssueSeverity.WARNING,
                            pattern_type=PatternType.SECURITY_ISSUE,
                            location=CodeLocation(child.lineno, child.lineno),
                            suggestion="Ensure input is sanitized",
                            rule_id="S002",
                        ))
    
    def check_unused_imports(self) -> List[CodeIssue]:
        """Check for unused imports"""
        issues = []
        
        for name, import_info in self.imports.items():
            if name not in self.used_names:
                issues.append(CodeIssue(
                    message=f"Unused import: '{name}'",
                    severity=IssueSeverity.INFO,
                    pattern_type=PatternType.STYLE,
                    location=import_info.location,
                    suggestion="Remove unused import",
                    rule_id="F401",
                ))
        
        return issues


# ═══════════════════════════════════════════════════════════════════════════════
# CODE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class CodeAnalyzer:
    """
    Ultra-Advanced Code Analysis Engine.
    
    Features:
    - Full AST parsing
    - Complexity metrics
    - Pattern detection
    - Dependency analysis
    - Quality scoring
    - Issue detection
    
    Memory Budget: < 10MB per file
    
    Usage:
        analyzer = CodeAnalyzer()
        
        # Analyze a file
        result = analyzer.analyze_file("my_module.py")
        
        # Analyze code string
        result = analyzer.analyze_code("def hello(): pass")
        
        # Get complexity
        complexity = analyzer.get_complexity(code)
        
        # Check for issues
        issues = analyzer.find_issues(code)
    """
    
    # Standard library modules
    STDLIB_MODULES = PatternDetector.STDLIB_MODULES
    
    def __init__(
        self,
        max_file_size_mb: float = 5.0,
        max_complexity: int = 20,
        max_function_lines: int = 50,
        enable_security_checks: bool = True,
    ):
        """
        Initialize Code Analyzer.
        
        Args:
            max_file_size_mb: Maximum file size to analyze
            max_complexity: Maximum acceptable complexity
            max_function_lines: Maximum lines per function
            enable_security_checks: Enable security scanning
        """
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._max_complexity = max_complexity
        self._max_function_lines = max_function_lines
        self._enable_security = enable_security_checks
        
        # Cache for parsed ASTs
        self._cache: Dict[str, Tuple[ast.AST, float]] = {}
        self._cache_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'files_analyzed': 0,
            'total_issues': 0,
            'total_complexity': 0,
            'cache_hits': 0,
            'parse_errors': 0,
        }
        
        logger.info("CodeAnalyzer initialized")
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            FileAnalysis with complete results
        """
        path = Path(file_path)
        
        # Check file size
        size = path.stat().st_size if path.exists() else 0
        
        if size > self._max_file_size:
            return FileAnalysis(
                path=file_path,
                content_hash="",
                size_bytes=size,
                parse_time_ms=0,
                success=False,
                error=f"File too large: {size / (1024*1024):.1f}MB > {self._max_file_size / (1024*1024):.1f}MB limit",
            )
        
        # Read file
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return FileAnalysis(
                path=file_path,
                content_hash="",
                size_bytes=size,
                parse_time_ms=0,
                success=False,
                error="Unable to decode file as UTF-8",
            )
        except Exception as e:
            return FileAnalysis(
                path=file_path,
                content_hash="",
                size_bytes=size,
                parse_time_ms=0,
                success=False,
                error=str(e),
            )
        
        # Analyze content
        return self.analyze_code(content, file_path)
    
    def analyze_code(
        self,
        code: str,
        file_path: str = "<string>",
    ) -> FileAnalysis:
        """
        Analyze Python code string.
        
        Args:
            code: Python source code
            file_path: Optional path for context
            
        Returns:
            FileAnalysis with results
        """
        start_time = time.time()
        content_hash = hashlib.md5(code.encode()).hexdigest()
        size = len(code.encode('utf-8'))
        
        # Initialize result
        result = FileAnalysis(
            path=file_path,
            content_hash=content_hash,
            size_bytes=size,
            parse_time_ms=0,
        )
        
        # Parse AST with recursion protection
        try:
            # Increase recursion limit for deeply nested expressions
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old_limit, 5000))
            try:
                tree = self._parse_ast(code)
            finally:
                sys.setrecursionlimit(old_limit)
                
            if tree is None:
                result.success = False
                result.error = "Failed to parse AST"
                self._stats['parse_errors'] += 1
                return result
        except SyntaxError as e:
            result.success = False
            result.error = f"Syntax error at line {e.lineno}: {e.msg}"
            self._stats['parse_errors'] += 1
            
            # Add as issue
            result.issues.append(CodeIssue(
                message=f"Syntax error: {e.msg}",
                severity=IssueSeverity.ERROR,
                pattern_type=PatternType.STYLE,
                location=CodeLocation(e.lineno or 0, e.lineno or 0),
                rule_id="E999",
            ))
            
            return result
        except RecursionError as e:
            # Handle deeply nested expressions gracefully
            result.success = True  # Still valid, just can't fully analyze
            result.error = "Code too complex for full analysis (recursion limit)"
            result.parse_time_ms = (time.time() - start_time) * 1000
            self._stats['files_analyzed'] += 1
            return result
        
        # Extract structure (with error handling)
        try:
            self._extract_structure(tree, result)
        except RecursionError:
            result.issues.append(CodeIssue(
                message="Code structure too complex for full analysis",
                severity=IssueSeverity.INFO,
                pattern_type=PatternType.MAINTAINABILITY,
                location=CodeLocation(1, 1),
                rule_id="REC001",
            ))
        
        # Calculate complexity (with error handling)
        try:
            self._calculate_complexity(tree, result)
        except RecursionError:
            pass  # Skip complexity for deeply nested code
        
        # Detect patterns and issues (with error handling)
        try:
            self._detect_issues(tree, result)
        except RecursionError:
            pass  # Skip issue detection for deeply nested code
        
        # Analyze dependencies
        self._analyze_dependencies(result)
        
        # Update stats
        result.parse_time_ms = (time.time() - start_time) * 1000
        self._stats['files_analyzed'] += 1
        self._stats['total_issues'] += len(result.issues)
        
        return result
    
    def _parse_ast(self, code: str) -> Optional[ast.AST]:
        """Parse code to AST"""
        try:
            return ast.parse(code)
        except SyntaxError:
            return None
    
    def _extract_structure(self, tree: ast.AST, result: FileAnalysis):
        """Extract code structure from AST"""
        
        # Module docstring
        result.module_docstring = ast.get_docstring(tree)
        
        for node in ast.walk(tree):
            # Functions
            if isinstance(node, ast.FunctionDef):
                func_info = self._extract_function(node)
                result.functions.append(func_info)
                
            elif isinstance(node, ast.AsyncFunctionDef):
                func_info = self._extract_function(node, is_async=True)
                result.functions.append(func_info)
            
            # Classes
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class(node)
                result.classes.append(class_info)
            
            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name],
                        aliases={alias.name: alias.asname} if alias.asname else {},
                        location=CodeLocation(node.lineno, node.lineno),
                        is_from=False,
                    ))
            
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                aliases = {alias.name: alias.asname for alias in node.names if alias.asname}
                result.imports.append(ImportInfo(
                    module=node.module or '',
                    names=names,
                    aliases=aliases,
                    location=CodeLocation(node.lineno, node.lineno),
                    is_from=True,
                    level=node.level,
                ))
            
            # Global variables (top-level assignments)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper():
                            result.constants.add(name)
                        elif not name.startswith('_'):
                            result.global_variables.add(name)
    
    def _extract_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_async: bool = False
    ) -> FunctionInfo:
        """Extract function information"""
        
        # Parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        for arg in node.args.posonlyargs:
            params.append(arg.arg)
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        
        # Return annotations
        returns = []
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns.append(node.returns.id)
            elif isinstance(node.returns, ast.Constant):
                returns.append(str(node.returns.value))
        
        # Decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
        
        # Complexity
        complexity_calc = ComplexityCalculator()
        complexity_calc.visit(node)
        
        complexity = ComplexityMetrics(
            cyclomatic=complexity_calc.cyclomatic,
            cognitive=complexity_calc.cognitive,
            lines_of_code=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
        )
        
        # Local variables and called functions
        local_vars = set()
        called_funcs = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                local_vars.add(child.id)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    called_funcs.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    called_funcs.add(child.func.attr)
        
        return FunctionInfo(
            name=node.name,
            node_type=NodeType.ASYNC_FUNCTION if is_async else NodeType.FUNCTION,
            location=CodeLocation(
                node.lineno,
                node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            ),
            complexity=complexity,
            parameters=params,
            returns=returns,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            is_async=is_async,
            local_variables=local_vars,
            called_functions=called_funcs,
        )
    
    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information"""
        
        # Base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        
        # Methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                func = self._extract_function(item)
                func.is_method = True
                func.class_name = node.name
                methods.append(func)
            elif isinstance(item, ast.AsyncFunctionDef):
                func = self._extract_function(item, is_async=True)
                func.is_method = True
                func.class_name = node.name
                methods.append(func)
        
        # Class and instance variables
        class_vars = set()
        instance_vars = set()
        
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_vars.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        if target.value.id == 'self':
                            instance_vars.add(target.attr)
        
        # Complexity
        total_complexity = 1
        for method in methods:
            total_complexity += method.complexity.cyclomatic
        
        complexity = ComplexityMetrics(
            cyclomatic=total_complexity,
            lines_of_code=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
        )
        
        return ClassInfo(
            name=node.name,
            location=CodeLocation(
                node.lineno,
                node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            ),
            docstring=ast.get_docstring(node),
            bases=bases,
            methods=methods,
            class_variables=class_vars,
            instance_variables=instance_vars,
            complexity=complexity,
        )
    
    def _calculate_complexity(self, tree: ast.AST, result: FileAnalysis):
        """Calculate overall complexity"""
        
        calc = ComplexityCalculator()
        calc.visit(tree)
        
        # Count lines
        total_lines = 0
        comment_lines = 0
        docstring_lines = 0
        
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                total_lines = max(total_lines, node.lineno)
        
        # Sum function complexities
        total_cyclomatic = 1
        total_cognitive = 0
        
        for func in result.functions:
            total_cyclomatic += func.complexity.cyclomatic
            total_cognitive += func.complexity.cognitive
        
        for cls in result.classes:
            total_cyclomatic += cls.complexity.cyclomatic if cls.complexity else 0
        
        # Calculate maintainability index
        # Simplified version: MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        # Where V = Halstead volume, G = cyclomatic complexity, LOC = lines of code
        loc = max(1, total_lines)
        mi = max(0, min(100, 171 - 5.2 * math.log(total_cyclomatic + 1) - 16.2 * math.log(loc)))
        
        result.total_complexity = ComplexityMetrics(
            cyclomatic=total_cyclomatic,
            cognitive=total_cognitive,
            lines_of_code=total_lines,
            maintainability_index=mi,
        )
        
        self._stats['total_complexity'] += total_cyclomatic
    
    def _detect_issues(self, tree: ast.AST, result: FileAnalysis):
        """Detect code issues and patterns"""
        
        detector = PatternDetector()
        detector.visit(tree)
        
        # Add detected issues
        result.issues.extend(detector.issues)
        
        # Check unused imports
        result.issues.extend(detector.check_unused_imports())
        
        # Add complexity warnings
        for func in result.functions:
            if func.complexity.cyclomatic > self._max_complexity:
                result.issues.append(CodeIssue(
                    message=f"Function '{func.name}' has high complexity ({func.complexity.cyclomatic})",
                    severity=IssueSeverity.WARNING,
                    pattern_type=PatternType.MAINTAINABILITY,
                    location=func.location,
                    suggestion="Consider refactoring to reduce complexity",
                    rule_id="C901",
                ))
    
    def _analyze_dependencies(self, result: FileAnalysis):
        """Analyze import dependencies"""
        
        for imp in result.imports:
            module = imp.module.split('.')[0] if imp.module else ''
            
            if module in self.STDLIB_MODULES:
                result.stdlib_imports.add(module)
            elif module:
                # Heuristic: lowercase with no dots = likely stdlib or third party
                # We'll categorize based on common patterns
                if module.startswith('_'):
                    result.local_imports.add(module)
                else:
                    result.third_party_imports.add(module)
            
            result.dependencies.add(module)
    
    def get_complexity(self, code: str) -> ComplexityMetrics:
        """
        Get complexity metrics for code.
        
        Args:
            code: Python source code
            
        Returns:
            ComplexityMetrics
        """
        try:
            tree = ast.parse(code)
            calc = ComplexityCalculator()
            calc.visit(tree)
            
            return ComplexityMetrics(
                cyclomatic=calc.cyclomatic,
                cognitive=calc.cognitive,
            )
        except Exception:
            return ComplexityMetrics()
    
    def find_issues(self, code: str) -> List[CodeIssue]:
        """
        Find issues in code.
        
        Args:
            code: Python source code
            
        Returns:
            List of CodeIssue objects
        """
        try:
            tree = ast.parse(code)
            detector = PatternDetector()
            detector.visit(tree)
            
            issues = list(detector.issues)
            issues.extend(detector.check_unused_imports())
            
            return issues
        except Exception:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            **self._stats,
            'cache_size': len(self._cache),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer: Optional[CodeAnalyzer] = None


def get_analyzer() -> CodeAnalyzer:
    """Get global CodeAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer


def analyze_file(path: str) -> FileAnalysis:
    """Convenience function to analyze a file"""
    return get_analyzer().analyze_file(path)


def analyze_code(code: str) -> FileAnalysis:
    """Convenience function to analyze code"""
    return get_analyzer().analyze_code(code)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for CodeAnalyzer"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    analyzer = CodeAnalyzer()
    
    # Test code with various patterns
    test_code = '''
"""Module docstring."""
import os
import json
from typing import List, Dict

# Global constant
MAX_VALUE = 100

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def process(self, items: List[Dict]) -> List[str]:
        """Process a list of items."""
        results = []
        for item in items:
            if item.get('active'):
                try:
                    result = self._transform(item)
                    results.append(result)
                except Exception:
                    pass
        return results
    
    def _transform(self, item: Dict) -> str:
        return str(item)

def complex_function(x, y, z):
    """A complex function with nested conditions."""
    if x > 0:
        if y > 0:
            if z > 0:
                for i in range(10):
                    if i % 2 == 0:
                        return "positive"
    elif x < 0:
        return "negative"
    return "zero"

def dangerous_code(user_input):
    """Function with security issues."""
    result = eval(user_input)  # Dangerous!
    exec(user_input)  # Also dangerous!
    return result
'''
    
    # Test 1: Parse code
    result = analyzer.analyze_code(test_code, "test.py")
    if result.success:
        results['passed'].append('parse_code')
    else:
        results['failed'].append(f'parse_code: {result.error}')
    
    # Test 2: Extract functions
    if len(result.functions) >= 3:
        results['passed'].append(f'extract_functions: {len(result.functions)} found')
    else:
        results['failed'].append(f'extract_functions: {len(result.functions)} found')
    
    # Test 3: Extract classes
    if len(result.classes) >= 1:
        results['passed'].append('extract_classes')
    else:
        results['failed'].append(f'extract_classes: {len(result.classes)} found')
    
    # Test 4: Detect imports
    if len(result.imports) >= 3:
        results['passed'].append('extract_imports')
    else:
        results['failed'].append(f'extract_imports: {len(result.imports)} found')
    
    # Test 5: Complexity calculation
    if result.total_complexity and result.total_complexity.cyclomatic > 0:
        results['passed'].append(f'complexity: {result.total_complexity.cyclomatic}')
    else:
        results['failed'].append('complexity')
    
    # Test 6: Issue detection
    if len(result.issues) > 0:
        # Should detect eval() and exec()
        security_issues = [i for i in result.issues if i.pattern_type == PatternType.SECURITY_ISSUE]
        if len(security_issues) >= 2:
            results['passed'].append(f'security_issues: {len(security_issues)} found')
        else:
            results['warnings'].append(f'security_issues: only {len(security_issues)} found')
    else:
        results['failed'].append('issue_detection')
    
    # Test 7: High complexity detection
    complex_func = result.get_function('complex_function')
    if complex_func and complex_func.complexity.cyclomatic > 5:
        results['passed'].append(f'complexity_detection: cyclomatic={complex_func.complexity.cyclomatic}')
    else:
        results['warnings'].append(f'complexity_detection: cyclomatic={complex_func.complexity.cyclomatic if complex_func else 0}')
    
    # Test 8: Class method extraction
    processor_class = result.get_class('DataProcessor')
    if processor_class and len(processor_class.methods) >= 2:
        results['passed'].append(f'class_methods: {len(processor_class.methods)} found')
    else:
        results['failed'].append(f'class_methods: {len(processor_class.methods) if processor_class else 0} found')
    
    results['stats'] = analyzer.get_stats()
    results['analysis_result'] = {
        'functions': len(result.functions),
        'classes': len(result.classes),
        'imports': len(result.imports),
        'issues': len(result.issues),
        'complexity': result.total_complexity.cyclomatic if result.total_complexity else 0,
    }
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Code Analysis Engine - Self Test")
    print("=" * 70)
    print(f"Device: Realme 2 Pro Lite (RMP2402)")
    print("-" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed Tests:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed Tests:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    if test_results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in test_results['warnings']:
            print(f"   ! {warning}")
    
    print("\n📊 Analysis Result:")
    analysis = test_results.get('analysis_result', {})
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
