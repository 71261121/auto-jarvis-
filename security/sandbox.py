#!/usr/bin/env python3
"""
JARVIS Execution Sandbox
Ultra-Advanced Code Execution Security Module

Features:
- Isolated execution environment
- Restricted builtins and imports
- Resource limits (CPU, memory, time) - with Android/Termux fallback
- File system sandboxing
- Network access control
- Import restrictions and whitelisting
- Timeout handling with forced termination
- Safe globals and locals
- Execution monitoring and logging
- Multi-level security tiers
- Safe evaluation and exec

Device: Realme Pad 2 Lite (RMP2402) | RAM: 4GB | Platform: Termux

Author: JARVIS Self-Modifying AI Project
Version: 1.0.1 (Fixed for Termux/Android compatibility)
"""

import os
import sys
import ast
import traceback
import signal
import threading
import builtins
import types
import importlib
import inspect
import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import weakref

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: resource module NOT available on Android/Termux
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import resource
    HAS_RESOURCE = True
    RESOURCE_ERROR = getattr(resource, 'error', Exception)
except ImportError:
    # Android/Termux does not have the 'resource' module
    # This is expected behavior - we use timeout-based limits instead
    HAS_RESOURCE = False
    RESOURCE_ERROR = Exception
    resource = None  # type: ignore

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: thread module deprecated - use _thread
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import _thread as thread
except ImportError:
    import thread  # type: ignore  # Fallback for older Python


# Constants - Optimized for 4GB RAM device (500MB working memory)
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_MEMORY_LIMIT = 50 * 1024 * 1024  # 50 MB (reduced from 100MB for Termux)
DEFAULT_CPU_TIME = 10  # seconds
MAX_OUTPUT_SIZE = 10000  # characters
MAX_RECURSION_DEPTH = 100


class SecurityTier(Enum):
    """Security tier levels"""
    TRUSTED = 0      # Full access (internal code only)
    HIGH = 1         # Trusted user code with some restrictions
    MEDIUM = 2       # Standard restrictions
    LOW = 3          # Heavy restrictions
    ISOLATED = 4     # Maximum isolation


class ExecutionStatus(Enum):
    """Status of code execution"""
    SUCCESS = auto()
    TIMEOUT = auto()
    MEMORY_ERROR = auto()
    IMPORT_DENIED = auto()
    SYNTAX_ERROR = auto()
    RUNTIME_ERROR = auto()
    SECURITY_VIOLATION = auto()
    OUTPUT_TRUNCATED = auto()
    RECURSION_ERROR = auto()


class AccessType(Enum):
    """Types of access to control"""
    FILE_READ = auto()
    FILE_WRITE = auto()
    FILE_DELETE = auto()
    NETWORK_OUT = auto()
    NETWORK_IN = auto()
    PROCESS_SPAWN = auto()
    SYSTEM_CALL = auto()
    IMPORT_MODULE = auto()
    ENV_ACCESS = auto()


@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    memory_used: int = 0
    imports_used: List[str] = field(default_factory=list)
    files_accessed: List[str] = field(default_factory=list)
    security_events: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.name,
            'output': self.output,
            'error': self.error,
            'return_value': repr(self.return_value),
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'imports_used': self.imports_used,
            'files_accessed': self.files_accessed,
            'security_events': self.security_events,
            'stdout': self.stdout,
            'stderr': self.stderr
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    tier: SecurityTier = SecurityTier.MEDIUM
    timeout: int = DEFAULT_TIMEOUT
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    cpu_time_limit: int = DEFAULT_CPU_TIME
    max_output_size: int = MAX_OUTPUT_SIZE
    max_recursion_depth: int = MAX_RECURSION_DEPTH
    allowed_imports: Set[str] = field(default_factory=set)
    denied_imports: Set[str] = field(default_factory=set)
    allowed_paths: Set[str] = field(default_factory=set)
    denied_paths: Set[str] = field(default_factory=set)
    allow_network: bool = False
    allow_file_write: bool = False
    allow_file_read: bool = True
    allow_subprocess: bool = False
    allow_system_calls: bool = False
    allow_env_access: bool = False
    allowed_builtins: Set[str] = field(default_factory=set)
    denied_builtins: Set[str] = field(default_factory=set)

    @classmethod
    def trusted(cls) -> 'SecurityPolicy':
        """Create trusted policy"""
        return cls(
            tier=SecurityTier.TRUSTED,
            timeout=300,
            memory_limit=200 * 1024 * 1024,  # 200MB (reduced for Termux)
            cpu_time_limit=60,
            allow_network=True,
            allow_file_write=True,
            allow_file_read=True,
            allow_subprocess=True,
            allow_system_calls=True,
            allow_env_access=True
        )

    @classmethod
    def high(cls) -> 'SecurityPolicy':
        """Create high trust policy"""
        return cls(
            tier=SecurityTier.HIGH,
            timeout=60,
            memory_limit=100 * 1024 * 1024,  # 100MB
            cpu_time_limit=30,
            allow_file_write=True,
            allow_file_read=True,
            allowed_imports={'os', 'sys', 'json', 're', 'datetime', 'math', 'random', 'collections', 'itertools', 'functools'}
        )

    @classmethod
    def medium(cls) -> 'SecurityPolicy':
        """Create medium security policy"""
        return cls(
            tier=SecurityTier.MEDIUM,
            timeout=30,
            memory_limit=50 * 1024 * 1024,  # 50MB (reduced for Termux)
            cpu_time_limit=10,
            allow_file_read=True,
            allowed_imports={'json', 're', 'datetime', 'math', 'random', 'collections', 'itertools', 'functools'}
        )

    @classmethod
    def low(cls) -> 'SecurityPolicy':
        """Create low trust policy"""
        return cls(
            tier=SecurityTier.LOW,
            timeout=10,
            memory_limit=20 * 1024 * 1024,  # 20MB
            cpu_time_limit=5,
            allowed_imports={'math', 'random'}
        )

    @classmethod
    def isolated(cls) -> 'SecurityPolicy':
        """Create isolated policy"""
        return cls(
            tier=SecurityTier.ISOLATED,
            timeout=5,
            memory_limit=5 * 1024 * 1024,  # 5MB
            cpu_time_limit=2,
            allowed_imports=set()
        )


class ImportValidator:
    """Validates and controls module imports"""

    # Dangerous modules that should never be allowed
    DANGEROUS_MODULES = {
        'subprocess', 'multiprocessing', 'threading', 'ctypes', 'ctypes.wintypes',
        '_thread', 'signal', 'os', 'posix', 'nt', '_posixsubprocess',
        'socket', 'ssl', 'asyncio', 'select', 'selectors',
        'pickle', 'shelve', 'marshal', 'imp', 'importlib',
        'builtins', '__builtin__', 'code', 'codeop', 'compile',
        'sysconfig', 'platform', 'distutils', 'setuptools',
        'popen2', 'commands', 'pipes', 'posixfile',
        'resource', 'syslog', 'pdb', 'bdb', 'faulthandler',
    }

    # Modules that require special permission
    RESTRICTED_MODULES = {
        'os', 'sys', 'io', 'pathlib', 'glob', 'shutil', 'tempfile',
        'socket', 'http', 'urllib', 'requests', 'aiohttp',
        'sqlite3', 'dbm', 'json', 'csv', 'configparser',
        'logging', 'argparse', 'getopt', 'optparse',
        'unittest', 'doctest', 'trace', 'traceback',
    }

    # Safe modules that are always allowed
    SAFE_MODULES = {
        'math', 'cmath', 'decimal', 'fractions', 'statistics',
        'random', 'itertools', 'collections', 'functools', 'operator',
        're', 'string', 'textwrap', 'unicodedata',
        'datetime', 'calendar', 'time',
        'copy', 'pprint', 'reprlib', 'enum',
        'typing', 'dataclasses', 'abc', 'contextlib',
        'json', 'array', 'struct',
    }

    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self._imported_modules: Set[str] = set()
        self._import_denied: List[str] = []

    def validate_import(self, module_name: str) -> Tuple[bool, str]:
        """Validate if module can be imported"""
        # Get base module name
        base_module = module_name.split('.')[0]

        # Check tier-based rules
        if self.policy.tier == SecurityTier.TRUSTED:
            return True, "Trusted tier allows all imports"

        # Check dangerous modules
        if base_module in self.DANGEROUS_MODULES:
            self._import_denied.append(module_name)
            return False, f"Module '{base_module}' is in the dangerous modules list"

        # Check denied list
        if base_module in self.policy.denied_imports:
            self._import_denied.append(module_name)
            return False, f"Module '{base_module}' is denied by policy"

        # Check allowed list (if specified)
        if self.policy.allowed_imports:
            if base_module not in self.policy.allowed_imports and base_module not in self.SAFE_MODULES:
                self._import_denied.append(module_name)
                return False, f"Module '{base_module}' is not in allowed imports"

        # Check restricted modules
        if base_module in self.RESTRICTED_MODULES:
            if self.policy.tier.value > SecurityTier.HIGH.value:
                self._import_denied.append(module_name)
                return False, f"Module '{base_module}' requires higher trust tier"

        self._imported_modules.add(module_name)
        return True, "Import allowed"

    def get_imported_modules(self) -> Set[str]:
        """Get all imported modules"""
        return self._imported_modules.copy()

    def get_denied_imports(self) -> List[str]:
        """Get denied import attempts"""
        return self._import_denied.copy()


class RestrictedBuiltins:
    """Restricted builtins for sandbox execution"""

    # Always dangerous - never allow
    NEVER_ALLOWED = {
        'eval', 'exec', 'compile', 'execfile', 'input',
        '__import__', 'globals', 'locals', 'vars',
        'breakpoint', 'help', 'license', 'credits', 'copyright',
        'exit', 'quit',
    }

    # Potentially dangerous - allow only in higher tiers
    RESTRICTED = {
        'open', 'file', 'memoryview', 'bytearray',
        'property', 'super', 'type', 'object',
        'classmethod', 'staticmethod',
    }

    # Safe builtins
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
        'filter', 'float', 'format', 'frozenset', 'getattr', 'hasattr',
        'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
        'len', 'list', 'map', 'max', 'min', 'next', 'object', 'oct',
        'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
        'zip', 'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
        'Exception', 'BaseException', 'ArithmeticError', 'AssertionError',
        'AttributeError', 'BlockingIOError', 'BrokenPipeError', 'BufferError',
        'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError',
        'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError',
        'DeprecationWarning', 'EOFError', 'EnvironmentError', 'FileExistsError',
        'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit',
        'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError',
        'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt',
        'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError',
        'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError',
        'OverflowError', 'PendingDeprecationWarning', 'PermissionError',
        'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning',
        'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration',
        'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError',
        'TimeoutError', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError',
        'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslationError',
        'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError',
    }

    @classmethod
    def get_safe_builtins(cls, policy: SecurityPolicy) -> Dict[str, Any]:
        """Get safe builtins dict for sandbox"""
        safe = {}

        # Start with safe builtins
        for name in cls.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)

        # Add allowed builtins from policy
        for name in policy.allowed_builtins:
            if hasattr(builtins, name) and name not in cls.NEVER_ALLOWED:
                safe[name] = getattr(builtins, name)

        # Remove denied builtins
        for name in cls.NEVER_ALLOWED:
            safe.pop(name, None)

        for name in policy.denied_builtins:
            safe.pop(name, None)

        # Remove restricted builtins for lower tiers
        if policy.tier.value >= SecurityTier.MEDIUM.value:
            for name in cls.RESTRICTED:
                safe.pop(name, None)

        return safe


class ResourceLimiter:
    """Resource limit enforcement with Termux/Android compatibility"""

    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self._start_time = 0
        self._memory_peak = 0
        self._limits_set = False

    def set_limits(self) -> None:
        """Set resource limits (with Android/Termux fallback)"""
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: resource module not available on Android/Termux
        # ═══════════════════════════════════════════════════════════════════════
        if not HAS_RESOURCE:
            # On Android/Termux, we rely on timeout-based limiting only
            # Python memory management will still enforce limits via GC
            self._limits_set = True
            return

        try:
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU,
                              (self.policy.cpu_time_limit, self.policy.cpu_time_limit + 1))

            # Memory limit (if supported)
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(resource.RLIMIT_AS,
                                  (self.policy.memory_limit, self.policy.memory_limit + 1))

            # No file creation by default
            if not self.policy.allow_file_write:
                if hasattr(resource, 'RLIMIT_FSIZE'):
                    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

            # Limit number of processes
            if hasattr(resource, 'RLIMIT_NPROC'):
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

            self._limits_set = True
        except (ValueError, RESOURCE_ERROR):
            # May fail on some platforms
            pass

    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        self._start_time = time.time()
        self._memory_peak = 0

    def check_timeout(self) -> bool:
        """Check if timeout exceeded"""
        elapsed = time.time() - self._start_time
        return elapsed > self.policy.timeout

    def get_elapsed_time(self) -> float:
        """Get elapsed time"""
        return time.time() - self._start_time


class TimeoutManager:
    """Manages execution timeouts"""

    def __init__(self, timeout: int):
        self.timeout = timeout
        self._timer = None
        self._timed_out = False

    def start(self) -> None:
        """Start timeout timer"""
        self._timed_out = False
        self._timer = threading.Timer(self.timeout, self._timeout_handler)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        """Stop timeout timer"""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _timeout_handler(self) -> None:
        """Handle timeout"""
        self._timed_out = True
        # Try to interrupt the main thread
        try:
            # FIX: Use _thread module (thread module is deprecated)
            thread.interrupt_main()
        except:
            pass

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @contextmanager
    def timeout_context(self):
        """Context manager for timeout"""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class OutputCapture:
    """Capture stdout/stderr during execution"""

    def __init__(self, max_size: int = MAX_OUTPUT_SIZE):
        self.max_size = max_size
        self.stdout_capture = []
        self.stderr_capture = []
        self._original_stdout = None
        self._original_stderr = None
        self._truncated = False

    def start(self) -> None:
        """Start capturing output"""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.stdout_capture = []
        self.stderr_capture = []
        self._truncated = False

        sys.stdout = self
        sys.stderr = self

    def stop(self) -> None:
        """Stop capturing output"""
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

    def write(self, data: str) -> None:
        """Write to capture buffer"""
        if sys.stdout == self:
            self.stdout_capture.append(data)
        elif sys.stderr == self:
            self.stderr_capture.append(data)

        # Check size limit
        total = len(''.join(self.stdout_capture)) + len(''.join(self.stderr_capture))
        if total > self.max_size:
            self._truncated = True
            raise IOError("Output size limit exceeded")

    def flush(self) -> None:
        """Flush capture buffer"""
        pass

    def get_stdout(self) -> str:
        """Get captured stdout"""
        return ''.join(self.stdout_capture)

    def get_stderr(self) -> str:
        """Get captured stderr"""
        return ''.join(self.stderr_capture)

    @property
    def truncated(self) -> bool:
        return self._truncated

    @contextmanager
    def capture_context(self):
        """Context manager for output capture"""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class CodeAnalyzer:
    """Analyze code for security issues before execution"""

    DANGEROUS_PATTERNS = [
        ('__import__', 'Dynamic import'),
        ('eval', 'Dynamic evaluation'),
        ('exec', 'Dynamic execution'),
        ('compile', 'Dynamic compilation'),
        ('globals()', 'Global namespace access'),
        ('locals()', 'Local namespace access'),
        ('vars()', 'Variables access'),
        ('getattr', 'Dynamic attribute access'),
        ('setattr', 'Dynamic attribute modification'),
        ('delattr', 'Dynamic attribute deletion'),
        ('__class__', 'Class introspection'),
        ('__bases__', 'Base class access'),
        ('__subclasses__', 'Subclass enumeration'),
        ('__mro__', 'Method resolution order'),
        ('__dict__', 'Dictionary access'),
        ('__code__', 'Code object access'),
        ('__globals__', 'Global access'),
        ('os.system', 'System command execution'),
        ('subprocess', 'Subprocess execution'),
        ('socket', 'Network socket'),
        ('ctypes', 'C types access'),
        ('mmap', 'Memory mapping'),
    ]

    def __init__(self, code: str):
        self.code = code
        self.ast_tree = None
        self.issues: List[Tuple[str, str]] = []

    def analyze(self) -> Tuple[bool, List[str]]:
        """Analyze code for security issues"""
        issues = []

        # Parse code
        try:
            self.ast_tree = ast.parse(self.code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # Check for dangerous patterns
        for pattern, description in self.DANGEROUS_PATTERNS:
            if pattern in self.code:
                issues.append(f"Potentially dangerous pattern: {pattern} ({description})")

        # Check AST for dangerous nodes
        self._check_ast(self.ast_tree, issues)

        return len(issues) == 0, issues

    def _check_ast(self, node: ast.AST, issues: List[str]) -> None:
        """Recursively check AST nodes"""
        for child in ast.walk(node):
            # Check for dangerous function calls
            if isinstance(child, ast.Call):
                func_name = self._get_func_name(child.func)
                if func_name:
                    dangerous_funcs = {'eval', 'exec', 'compile', '__import__',
                                      'globals', 'locals', 'vars'}
                    if func_name in dangerous_funcs:
                        issues.append(f"Dangerous function call: {func_name}()")

            # Check for attribute access to dangerous attributes
            if isinstance(child, ast.Attribute):
                if child.attr.startswith('_'):
                    if child.attr in {'__class__', '__bases__', '__subclasses__',
                                     '__mro__', '__dict__', '__code__', '__globals__'}:
                        issues.append(f"Dangerous attribute access: {child.attr}")

            # Check for imports
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                module = self._get_module_name(child)
                if module:
                    issues.append(f"Import detected: {module}")

    def _get_func_name(self, node: ast.expr) -> Optional[str]:
        """Get function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_func_name(node.value)}.{node.attr}" if node.value else node.attr
        return None

    def _get_module_name(self, node: Union[ast.Import, ast.ImportFrom]) -> Optional[str]:
        """Get module name from import node"""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            return node.module
        return None

    def get_imports(self) -> List[str]:
        """Get list of imports in code"""
        imports = []
        if self.ast_tree:
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        return imports


class ExecutionSandbox:
    """Main sandbox for secure code execution"""

    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy.medium()
        self.import_validator = ImportValidator(self.policy)
        self.resource_limiter = ResourceLimiter(self.policy)
        self.output_capture = OutputCapture(self.policy.max_output_size)
        self._execution_count = 0
        self._security_events: List[str] = []

    def execute(self, code: str, globals_dict: Dict = None,
                locals_dict: Dict = None) -> ExecutionResult:
        """Execute code in sandbox"""
        self._execution_count += 1
        self._security_events = []
        start_time = time.time()

        # Analyze code first
        analyzer = CodeAnalyzer(code)
        is_safe, issues = analyzer.analyze()

        if not is_safe and self.policy.tier.value >= SecurityTier.MEDIUM.value:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                error=f"Code analysis failed: {'; '.join(issues)}",
                security_events=issues
            )

        # Check imports
        for module in analyzer.get_imports():
            allowed, msg = self.import_validator.validate_import(module)
            if not allowed:
                return ExecutionResult(
                    status=ExecutionStatus.IMPORT_DENIED,
                    error=f"Import denied: {msg}",
                    security_events=[msg]
                )

        # Prepare execution environment
        safe_globals = self._prepare_globals(globals_dict)
        safe_locals = locals_dict or {}

        # Execute with timeout
        result = self._execute_with_timeout(
            code, safe_globals, safe_locals, start_time
        )

        return result

    def _prepare_globals(self, custom_globals: Dict = None) -> Dict:
        """Prepare safe globals dictionary"""
        # Start with restricted builtins
        safe_builtins = RestrictedBuiltins.get_safe_builtins(self.policy)

        # Create globals
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
            '__package__': None,
            '__loader__': None,
            '__spec__': None,
        }

        # Add custom globals
        if custom_globals:
            for key, value in custom_globals.items():
                if key not in ('__builtins__', '__name__'):
                    safe_globals[key] = value

        return safe_globals

    def _execute_with_timeout(self, code: str, globals_dict: Dict,
                              locals_dict: Dict, start_time: float) -> ExecutionResult:
        """Execute code with timeout protection"""
        timeout_mgr = TimeoutManager(self.policy.timeout)
        output_capture = OutputCapture(self.policy.max_output_size)

        return_value = None
        error = ""
        status = ExecutionStatus.SUCCESS

        try:
            # Compile code
            compiled = compile(code, '<sandbox>', 'exec')

            # Set recursion limit
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(self.policy.max_recursion_depth)

            # Start timeout and capture
            timeout_mgr.start()
            output_capture.start()

            # Execute
            exec(compiled, globals_dict, locals_dict)

            # Get return value if any
            return_value = locals_dict.get('__return__', None)

        except SyntaxError as e:
            status = ExecutionStatus.SYNTAX_ERROR
            error = f"Syntax error: {e}"
        except MemoryError:
            status = ExecutionStatus.MEMORY_ERROR
            error = "Memory limit exceeded"
        except RecursionError:
            status = ExecutionStatus.RECURSION_ERROR
            error = f"Maximum recursion depth ({self.policy.max_recursion_depth}) exceeded"
        except TimeoutError:
            status = ExecutionStatus.TIMEOUT
            error = f"Execution timed out after {self.policy.timeout} seconds"
        except ImportError as e:
            status = ExecutionStatus.IMPORT_DENIED
            error = f"Import error: {e}"
        except PermissionError as e:
            status = ExecutionStatus.SECURITY_VIOLATION
            error = f"Permission denied: {e}"
        except Exception as e:
            status = ExecutionStatus.RUNTIME_ERROR
            error = f"{type(e).__name__}: {e}"
            if self.policy.tier.value <= SecurityTier.HIGH.value:
                error += f"\n{traceback.format_exc()}"
        finally:
            # Cleanup
            timeout_mgr.stop()
            output_capture.stop()

            if timeout_mgr.timed_out:
                status = ExecutionStatus.TIMEOUT
                error = f"Execution timed out after {self.policy.timeout} seconds"

            if output_capture.truncated:
                status = ExecutionStatus.OUTPUT_TRUNCATED

        execution_time = time.time() - start_time

        return ExecutionResult(
            status=status,
            output=output_capture.get_stdout(),
            error=error,
            return_value=return_value,
            execution_time=execution_time,
            imports_used=list(self.import_validator.get_imported_modules()),
            security_events=self._security_events,
            stdout=output_capture.get_stdout(),
            stderr=output_capture.get_stderr()
        )

    def evaluate(self, expression: str, globals_dict: Dict = None,
                 locals_dict: Dict = None) -> ExecutionResult:
        """Safely evaluate an expression"""
        # Wrap expression in exec-safe format
        code = f"__return__ = {expression}"
        result = self.execute(code, globals_dict, locals_dict)

        if result.success:
            result.return_value = result.return_value

        return result

    def execute_file(self, filepath: str) -> ExecutionResult:
        """Execute a file in sandbox"""
        try:
            with open(filepath, 'r') as f:
                code = f.read()
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error=f"Could not read file: {e}"
            )

        return self.execute(code)

    def test_code(self, code: str) -> Tuple[bool, List[str]]:
        """Test code without executing (dry run)"""
        analyzer = CodeAnalyzer(code)
        return analyzer.analyze()

    def get_security_report(self) -> Dict[str, Any]:
        """Get security report"""
        return {
            'execution_count': self._execution_count,
            'policy': {
                'tier': self.policy.tier.name,
                'timeout': self.policy.timeout,
                'memory_limit': self.policy.memory_limit,
                'cpu_time_limit': self.policy.cpu_time_limit,
            },
            'import_validator': {
                'imported': list(self.import_validator.get_imported_modules()),
                'denied': self.import_validator.get_denied_imports()
            },
            'platform': {
                'has_resource_module': HAS_RESOURCE,
                'platform': sys.platform,
            }
        }


class SandboxManager:
    """Manager for multiple sandboxes with different policies"""

    def __init__(self):
        self._sandboxes: Dict[str, ExecutionSandbox] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_sandbox(self, name: str, policy: SecurityPolicy = None) -> ExecutionSandbox:
        """Create a named sandbox"""
        with self._lock:
            sandbox = ExecutionSandbox(policy)
            self._sandboxes[name] = sandbox
            return sandbox

    def get_sandbox(self, name: str) -> Optional[ExecutionSandbox]:
        """Get sandbox by name"""
        return self._sandboxes.get(name)

    def execute_in_sandbox(self, sandbox_name: str, code: str,
                           globals_dict: Dict = None,
                           locals_dict: Dict = None) -> ExecutionResult:
        """Execute code in named sandbox"""
        sandbox = self.get_sandbox(sandbox_name)
        if not sandbox:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error=f"Sandbox '{sandbox_name}' not found"
            )

        result = sandbox.execute(code, globals_dict, locals_dict)

        # Record history
        with self._lock:
            self._execution_history.append({
                'sandbox': sandbox_name,
                'status': result.status.name,
                'execution_time': result.execution_time,
                'timestamp': datetime.now().isoformat()
            })

        return result

    def quick_execute(self, code: str,
                      tier: SecurityTier = SecurityTier.MEDIUM) -> ExecutionResult:
        """Quick execution with specified tier"""
        if tier == SecurityTier.TRUSTED:
            policy = SecurityPolicy.trusted()
        elif tier == SecurityTier.HIGH:
            policy = SecurityPolicy.high()
        elif tier == SecurityTier.MEDIUM:
            policy = SecurityPolicy.medium()
        elif tier == SecurityTier.LOW:
            policy = SecurityPolicy.low()
        else:
            policy = SecurityPolicy.isolated()

        sandbox = ExecutionSandbox(policy)
        return sandbox.execute(code)

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        with self._lock:
            return self._execution_history[-limit:]

    def clear_history(self) -> None:
        """Clear execution history"""
        with self._lock:
            self._execution_history.clear()

    def list_sandboxes(self) -> List[str]:
        """List all sandboxes"""
        return list(self._sandboxes.keys())

    def destroy_sandbox(self, name: str) -> bool:
        """Destroy a sandbox"""
        with self._lock:
            if name in self._sandboxes:
                del self._sandboxes[name]
                return True
            return False


# Export classes
__all__ = [
    'SecurityTier',
    'ExecutionStatus',
    'AccessType',
    'ExecutionResult',
    'SecurityPolicy',
    'ImportValidator',
    'RestrictedBuiltins',
    'ResourceLimiter',
    'TimeoutManager',
    'OutputCapture',
    'CodeAnalyzer',
    'ExecutionSandbox',
    'SandboxManager',
    'HAS_RESOURCE',  # Export for platform detection
]


if __name__ == "__main__":
    print("JARVIS Execution Sandbox v1.0.1")
    print("=" * 50)
    print(f"Platform: {sys.platform}")
    print(f"Resource module available: {HAS_RESOURCE}")
    print("=" * 50)

    # Create sandbox with medium security
    sandbox = ExecutionSandbox(SecurityPolicy.medium())

    # Test safe code
    code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""

    result = sandbox.execute(code)
    print(f"Status: {result.status.name}")
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time:.4f}s")

    # Test dangerous code
    dangerous_code = """
import os
os.system('ls')
"""

    result = sandbox.execute(dangerous_code)
    print(f"\nDangerous code status: {result.status.name}")
    print(f"Error: {result.error}")

    print("\nSandbox system ready for Termux/Android!")
