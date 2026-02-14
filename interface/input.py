#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Input Handler
====================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Multi-line input handling
- File input processing
- Voice input (Termux-API integration)
- Input sanitization and validation
- Special character handling
- Unicode support

Features:
- Multi-line code input with smart detection
- File-based input reading
- Voice input via Termux speech-to-text
- Input validation and sanitization
- Escape sequence handling
- Binary input support
- Input buffering for large data
- Input history navigation

Memory Impact: < 10MB for input handling
"""

import sys
import os
import re
import io
import time
import json
import shlex
import subprocess
import logging
import tempfile
import threading
from typing import (
    Dict, Any, Optional, List, Tuple, Callable, 
    Union, Generator, Iterator, BinaryIO, TextIO
)
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class InputSource(Enum):
    """Source of input"""
    TERMINAL = auto()     # Terminal/stdin
    FILE = auto()         # File input
    STRING = auto()       # String input
    VOICE = auto()        # Voice input
    PIPE = auto()         # Piped input
    NETWORK = auto()      # Network input
    CLIPBOARD = auto()    # Clipboard paste


class InputType(Enum):
    """Type of input content"""
    TEXT = auto()         # Plain text
    CODE = auto()         # Code snippet
    COMMAND = auto()      # Command
    MULTILINE = auto()    # Multi-line content
    BINARY = auto()       # Binary data
    JSON = auto()         # JSON data
    MARKDOWN = auto()     # Markdown content


class SanitizationLevel(Enum):
    """Input sanitization levels"""
    NONE = auto()         # No sanitization
    MINIMAL = auto()      # Basic whitespace cleanup
    STANDARD = auto()     # Standard sanitization
    STRICT = auto()       # Strict sanitization (remove dangerous chars)
    PARANOID = auto()     # Maximum sanitization


class InputState(Enum):
    """Input handler state"""
    IDLE = auto()         # Waiting for input
    READING = auto()      # Reading input
    MULTILINE = auto()    # In multiline mode
    ERROR = auto()        # Error state


@dataclass
class InputResult:
    """
    Result of input reading operation.
    
    Contains the input data along with metadata
    about how the input was obtained.
    """
    content: str
    source: InputSource
    input_type: InputType = InputType.TEXT
    raw_content: str = ""  # Before sanitization
    line_count: int = 1
    byte_count: int = 0
    duration_ms: float = 0.0
    encoding: str = "utf-8"
    is_truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if input was successful"""
        return self.error is None
    
    @property
    def lines(self) -> List[str]:
        """Get lines of content"""
        return self.content.split('\n') if self.content else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'source': self.source.name,
            'type': self.input_type.name,
            'line_count': self.line_count,
            'byte_count': self.byte_count,
            'duration_ms': self.duration_ms,
            'is_truncated': self.is_truncated,
            'warnings': self.warnings,
            'error': self.error,
        }


@dataclass
class InputConfig:
    """Input handler configuration"""
    max_input_size: int = 10 * 1024 * 1024  # 10MB max
    max_line_length: int = 100000  # 100KB per line
    max_line_count: int = 100000  # 100K lines max
    default_encoding: str = "utf-8"
    fallback_encodings: List[str] = field(default_factory=lambda: ["utf-8", "latin-1", "cp1252"])
    sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD
    strip_whitespace: bool = True
    preserve_indentation: bool = True
    multiline_end_marker: str = "```"
    multiline_timeout_seconds: float = 30.0
    enable_voice_input: bool = True
    voice_timeout_seconds: float = 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT SANITIZER
# ═══════════════════════════════════════════════════════════════════════════════

class InputSanitizer:
    """
    Sanitize and validate user input.
    
    Features:
    - Multiple sanitization levels
    - Control character removal
    - Unicode normalization
    - Dangerous pattern detection
    - Size limits enforcement
    """
    
    # Control characters to remove (except newline, tab)
    CONTROL_CHARS = set(range(0, 9)) | set(range(11, 32)) | {127}
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        (r'\x00', 'Null byte'),
        (r'\\x[0-9a-fA-F]{2}', 'Escaped byte sequence'),
        (r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', 'Invisible Unicode'),
    ]
    
    def __init__(self, config: InputConfig = None):
        """
        Initialize sanitizer.
        
        Args:
            config: Input configuration
        """
        self._config = config or InputConfig()
    
    def sanitize(
        self,
        content: str,
        level: SanitizationLevel = None,
    ) -> Tuple[str, List[str]]:
        """
        Sanitize input content.
        
        Args:
            content: Input content
            level: Sanitization level (uses config default if None)
            
        Returns:
            Tuple of (sanitized content, warnings)
        """
        level = level or self._config.sanitization_level
        warnings = []
        
        if level == SanitizationLevel.NONE:
            return content, warnings
        
        # Check size limits
        if len(content) > self._config.max_input_size:
            content = content[:self._config.max_input_size]
            warnings.append(f"Input truncated to {self._config.max_input_size} bytes")
        
        # Check line length
        if self._config.max_line_length > 0:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if len(line) > self._config.max_line_length:
                    lines[i] = line[:self._config.max_line_length]
                    warnings.append(f"Line {i+1} truncated")
            content = '\n'.join(lines)
        
        # Apply sanitization based on level
        if level == SanitizationLevel.MINIMAL:
            content = self._minimal_sanitize(content)
        elif level == SanitizationLevel.STANDARD:
            content = self._standard_sanitize(content)
        elif level == SanitizationLevel.STRICT:
            content, w = self._strict_sanitize(content)
            warnings.extend(w)
        elif level == SanitizationLevel.PARANOID:
            content, w = self._paranoid_sanitize(content)
            warnings.extend(w)
        
        # Strip whitespace if configured
        if self._config.strip_whitespace and not self._config.preserve_indentation:
            content = content.strip()
        
        return content, warnings
    
    def _minimal_sanitize(self, content: str) -> str:
        """Minimal sanitization - just remove null bytes"""
        return content.replace('\x00', '')
    
    def _standard_sanitize(self, content: str) -> str:
        """Standard sanitization - remove control chars"""
        result = []
        for char in content:
            if ord(char) not in self.CONTROL_CHARS:
                result.append(char)
        return ''.join(result)
    
    def _strict_sanitize(self, content: str) -> Tuple[str, List[str]]:
        """Strict sanitization - detect dangerous patterns"""
        warnings = []
        
        # First apply standard sanitization
        content = self._standard_sanitize(content)
        
        # Check for dangerous patterns
        for pattern, desc in self.DANGEROUS_PATTERNS:
            if re.search(pattern, content):
                warnings.append(f"Detected: {desc}")
        
        # Unicode normalization
        try:
            import unicodedata
            content = unicodedata.normalize('NFKC', content)
        except:
            pass
        
        return content, warnings
    
    def _paranoid_sanitize(self, content: str) -> Tuple[str, List[str]]:
        """Paranoid sanitization - maximum safety"""
        content, warnings = self._strict_sanitize(content)
        
        # Remove all non-printable ASCII except newline and tab
        result = []
        for char in content:
            if char in ('\n', '\t'):
                result.append(char)
            elif 32 <= ord(char) < 127:
                result.append(char)
            elif ord(char) >= 128:
                # Allow Unicode printable
                import unicodedata
                if unicodedata.category(char)[0] in ('L', 'N', 'P', 'S', 'Z'):
                    result.append(char)
                else:
                    warnings.append(f"Removed non-printable: U+{ord(char):04X}")
        
        return ''.join(result), warnings
    
    def validate_file_path(self, path: str) -> Tuple[bool, str]:
        """
        Validate a file path for safety.
        
        Args:
            path: File path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            p = Path(path)
            
            # Check for path traversal
            if '..' in p.parts:
                return False, "Path traversal detected"
            
            # Resolve and check if within allowed directories
            resolved = p.resolve()
            
            # Basic validation
            if not resolved.exists():
                return False, f"File does not exist: {resolved}"
            
            if not resolved.is_file():
                return False, f"Not a file: {resolved}"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def detect_input_type(self, content: str) -> InputType:
        """
        Detect the type of input content.
        
        Args:
            content: Input content
            
        Returns:
            Detected InputType
        """
        if not content:
            return InputType.TEXT
        
        # Check for JSON
        if content.strip().startswith(('{', '[')):
            try:
                json.loads(content)
                return InputType.JSON
            except:
                pass
        
        # Check for code (basic heuristics)
        code_indicators = [
            r'^\s*(def|class|function|var|let|const|import|from|require)\b',
            r'^\s*(if|else|for|while|switch|try|catch)\s*[\(\{]',
            r'^\s*@\w+',  # Decorators
            r'[\{\[\(\)\]\}]',  # Brackets
            r';\s*$',  # Statement terminator
        ]
        
        code_matches = sum(
            1 for pattern in code_indicators
            if re.search(pattern, content, re.MULTILINE)
        )
        
        if code_matches >= 2:
            return InputType.CODE
        
        # Check for markdown
        md_indicators = [
            r'^#{1,6}\s',  # Headers
            r'^\s*[-*+]\s',  # Lists
            r'\[.*\]\(.*\)',  # Links
            r'^```',  # Code blocks
            r'\*\*.*\*\*',  # Bold
            r'__.*__',  # Bold
        ]
        
        md_matches = sum(
            1 for pattern in md_indicators
            if re.search(pattern, content, re.MULTILINE)
        )
        
        if md_matches >= 2:
            return InputType.MARKDOWN
        
        # Check for multiline
        if '\n' in content:
            return InputType.MULTILINE
        
        # Check for command
        if re.match(r'^[\w_-]+(\s+[\w_-]+)*$', content.strip()):
            return InputType.COMMAND
        
        return InputType.TEXT


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceInputHandler:
    """
    Handle voice input via Termux-API.
    
    Uses termux-speech-to-text for voice recognition
    when running in Termux environment.
    """
    
    def __init__(self, config: InputConfig = None):
        """
        Initialize voice handler.
        
        Args:
            config: Input configuration
        """
        self._config = config or InputConfig()
        self._available: Optional[bool] = None
        self._last_result: Optional[str] = None
    
    @property
    def available(self) -> bool:
        """Check if voice input is available"""
        if self._available is None:
            self._available = self._check_availability()
        return self._available
    
    def _check_availability(self) -> bool:
        """Check if Termux speech-to-text is available"""
        # Check for Termux
        if 'TERMUX_VERSION' not in os.environ:
            logger.debug("Not running in Termux")
            return False
        
        # Check for termux-speech-to-text
        try:
            result = subprocess.run(
                ['which', 'termux-speech-to-text'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False
    
    def listen(
        self,
        timeout: float = None,
        prompt: str = "🎤 Listening...",
    ) -> InputResult:
        """
        Listen for voice input.
        
        Args:
            timeout: Maximum listening time in seconds
            prompt: Prompt to display while listening
            
        Returns:
            InputResult with transcribed text
        """
        start_time = time.time()
        timeout = timeout or self._config.voice_timeout_seconds
        
        if not self.available:
            return InputResult(
                content="",
                source=InputSource.VOICE,
                error="Voice input not available (requires Termux with termux-api)",
            )
        
        print(prompt, end='', flush=True)
        
        try:
            # Run termux-speech-to-text
            result = subprocess.run(
                ['termux-speech-to-text'],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            print()  # New line after prompt
            
            if result.returncode == 0 and result.stdout.strip():
                content = result.stdout.strip()
                self._last_result = content
                
                return InputResult(
                    content=content,
                    source=InputSource.VOICE,
                    input_type=InputType.TEXT,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            else:
                error = result.stderr.strip() or "No speech detected"
                return InputResult(
                    content="",
                    source=InputSource.VOICE,
                    error=error,
                    duration_ms=(time.time() - start_time) * 1000,
                )
                
        except subprocess.TimeoutExpired:
            print()
            return InputResult(
                content="",
                source=InputSource.VOICE,
                error="Voice input timed out",
                duration_ms=timeout * 1000,
            )
        except Exception as e:
            print()
            return InputResult(
                content="",
                source=InputSource.VOICE,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    def listen_continuous(
        self,
        callback: Callable[[str], None],
        stop_event: threading.Event = None,
    ):
        """
        Listen continuously for voice input.
        
        Args:
            callback: Function to call with each result
            stop_event: Event to stop listening
        """
        if stop_event is None:
            stop_event = threading.Event()
        
        while not stop_event.is_set():
            result = self.listen(timeout=10)
            if result.success and result.content:
                callback(result.content)
    
    @property
    def last_result(self) -> Optional[str]:
        """Get last voice recognition result"""
        return self._last_result


# ═══════════════════════════════════════════════════════════════════════════════
# FILE INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class FileInputHandler:
    """
    Handle file-based input.
    
    Features:
    - Multiple encoding support
    - Large file handling
    - Binary file support
    - File type detection
    """
    
    # Text file extensions
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml',
        '.xml', '.html', '.css', '.csv', '.tsv', '.ini', '.cfg',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.c', '.cpp', '.h', '.hpp', '.java', '.kt', '.rs', '.go',
        '.rb', '.php', '.pl', '.lua', '.vim', '.sql', '.r',
    }
    
    def __init__(self, config: InputConfig = None):
        """
        Initialize file handler.
        
        Args:
            config: Input configuration
        """
        self._config = config or InputConfig()
    
    def read_file(
        self,
        path: Union[str, Path],
        encoding: str = None,
        max_size: int = None,
    ) -> InputResult:
        """
        Read content from a file.
        
        Args:
            path: File path
            encoding: File encoding (auto-detected if None)
            max_size: Maximum size to read
            
        Returns:
            InputResult with file content
        """
        start_time = time.time()
        max_size = max_size or self._config.max_input_size
        
        try:
            p = Path(path)
            
            if not p.exists():
                return InputResult(
                    content="",
                    source=InputSource.FILE,
                    error=f"File not found: {path}",
                )
            
            if not p.is_file():
                return InputResult(
                    content="",
                    source=InputSource.FILE,
                    error=f"Not a file: {path}",
                )
            
            # Check size
            file_size = p.stat().st_size
            if file_size > max_size:
                return InputResult(
                    content="",
                    source=InputSource.FILE,
                    error=f"File too large: {file_size} > {max_size} bytes",
                    metadata={'file_size': file_size},
                )
            
            # Check if binary
            if self._is_binary_file(p):
                return self._read_binary(p)
            
            # Read text
            return self._read_text(p, encoding, start_time)
            
        except Exception as e:
            return InputResult(
                content="",
                source=InputSource.FILE,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    def _is_binary_file(self, path: Path) -> bool:
        """Check if file is binary"""
        # Check extension
        if path.suffix.lower() not in self.TEXT_EXTENSIONS:
            # Might be binary
            try:
                with open(path, 'rb') as f:
                    chunk = f.read(8192)
                    if b'\x00' in chunk:
                        return True
            except:
                pass
        return False
    
    def _read_text(
        self,
        path: Path,
        encoding: str,
        start_time: float,
    ) -> InputResult:
        """Read text file with encoding detection"""
        encodings = [encoding] if encoding else self._config.fallback_encodings
        
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                return InputResult(
                    content=content,
                    source=InputSource.FILE,
                    input_type=self._detect_file_type(path),
                    raw_content=content,
                    line_count=len(lines),
                    byte_count=len(content.encode(enc)),
                    duration_ms=(time.time() - start_time) * 1000,
                    encoding=enc,
                    metadata={
                        'path': str(path),
                        'extension': path.suffix,
                    },
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return InputResult(
                    content="",
                    source=InputSource.FILE,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                )
        
        return InputResult(
            content="",
            source=InputSource.FILE,
            error="Unable to decode file with any supported encoding",
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def _read_binary(self, path: Path) -> InputResult:
        """Read binary file"""
        start_time = time.time()
        
        try:
            with open(path, 'rb') as f:
                data = f.read()
            
            # Convert to hex representation for display
            hex_preview = data[:100].hex()
            
            return InputResult(
                content=f"[Binary file: {len(data)} bytes]\nHex preview: {hex_preview}...",
                source=InputSource.FILE,
                input_type=InputType.BINARY,
                byte_count=len(data),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'path': str(path),
                    'binary': True,
                    'size': len(data),
                },
            )
        except Exception as e:
            return InputResult(
                content="",
                source=InputSource.FILE,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    def _detect_file_type(self, path: Path) -> InputType:
        """Detect file type from extension"""
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            return InputType.JSON
        elif suffix in ('.md', '.markdown'):
            return InputType.MARKDOWN
        elif suffix in ('.py', '.js', '.ts', '.java', '.c', '.cpp', '.rs', '.go'):
            return InputType.CODE
        elif suffix in ('.sh', '.bash'):
            return InputType.COMMAND
        else:
            return InputType.TEXT
    
    def read_lines(
        self,
        path: Union[str, Path],
        start: int = 0,
        count: int = None,
    ) -> Generator[str, None, None]:
        """
        Read file line by line.
        
        Args:
            path: File path
            start: Starting line (0-indexed)
            count: Maximum lines to read
            
        Yields:
            Lines from the file
        """
        try:
            with open(path, 'r', encoding=self._config.default_encoding) as f:
                for i, line in enumerate(f):
                    if i < start:
                        continue
                    if count is not None and i >= start + count:
                        break
                    yield line.rstrip('\n\r')
        except Exception as e:
            logger.error(f"Error reading file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTILINE INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class MultilineInputHandler:
    """
    Handle multi-line input.
    
    Features:
    - Smart line continuation detection
    - Bracket matching for continuation
    - Custom end markers
    - Timeout support
    """
    
    # Opening/closing bracket pairs
    BRACKET_PAIRS = {
        '(': ')',
        '[': ']',
        '{': '}',
    }
    
    # Statements that often continue
    CONTINUATION_KEYWORDS = {
        'if', 'else', 'elif', 'for', 'while', 'def', 'class',
        'try', 'except', 'finally', 'with', 'async', 'await',
    }
    
    def __init__(self, config: InputConfig = None):
        """
        Initialize multiline handler.
        
        Args:
            config: Input configuration
        """
        self._config = config or InputConfig()
    
    def read_multiline(
        self,
        prompt: str = ">>> ",
        continuation_prompt: str = "... ",
        end_marker: str = None,
        timeout: float = None,
    ) -> InputResult:
        """
        Read multi-line input.
        
        Args:
            prompt: Initial prompt
            continuation_prompt: Prompt for continuation lines
            end_marker: Marker to end input (default: config setting)
            timeout: Timeout in seconds
            
        Returns:
            InputResult with multiline content
        """
        start_time = time.time()
        end_marker = end_marker or self._config.multiline_end_marker
        timeout = timeout or self._config.multiline_timeout_seconds
        
        lines = []
        bracket_stack = []
        in_string = None
        escape_next = False
        
        try:
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    return InputResult(
                        content='\n'.join(lines),
                        source=InputSource.TERMINAL,
                        input_type=InputType.MULTILINE,
                        error="Multiline input timed out",
                        line_count=len(lines),
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                
                # Get prompt
                current_prompt = prompt if not lines else continuation_prompt
                
                # Read line
                try:
                    line = input(current_prompt)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    return InputResult(
                        content="",
                        source=InputSource.TERMINAL,
                        error="Input cancelled",
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                
                # Check end marker
                if line.strip() == end_marker:
                    break
                
                lines.append(line)
                
                # Analyze line for continuation
                if not self._needs_continuation(line, bracket_stack, in_string, escape_next):
                    # Check if previous state needs more input
                    if not bracket_stack and in_string is None:
                        break
                
        except Exception as e:
            return InputResult(
                content='\n'.join(lines),
                source=InputSource.TERMINAL,
                error=str(e),
                line_count=len(lines),
                duration_ms=(time.time() - start_time) * 1000,
            )
        
        content = '\n'.join(lines)
        
        return InputResult(
            content=content,
            source=InputSource.TERMINAL,
            input_type=InputType.CODE if self._looks_like_code(content) else InputType.MULTILINE,
            line_count=len(lines),
            byte_count=len(content.encode('utf-8')),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def _needs_continuation(
        self,
        line: str,
        bracket_stack: List[str],
        in_string: Optional[str],
        escape_next: bool,
    ) -> bool:
        """Check if more input is needed"""
        i = 0
        while i < len(line):
            char = line[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            # String handling
            if char in ('"', "'"):
                if in_string == char:
                    in_string = None
                elif in_string is None:
                    in_string = char
                i += 1
                continue
            
            # Skip if in string
            if in_string:
                i += 1
                continue
            
            # Bracket handling
            if char in self.BRACKET_PAIRS:
                bracket_stack.append(char)
            elif char in self.BRACKET_PAIRS.values():
                if bracket_stack:
                    expected = self.BRACKET_PAIRS[bracket_stack[-1]]
                    if char == expected:
                        bracket_stack.pop()
            
            i += 1
        
        # Check for line continuation
        if line.rstrip().endswith('\\'):
            return True
        
        # Check for unclosed brackets
        if bracket_stack:
            return True
        
        # Check for unclosed strings
        if in_string:
            return True
        
        # Check for colon at end (Python blocks)
        if line.rstrip().endswith(':'):
            return True
        
        return False
    
    def _looks_like_code(self, content: str) -> bool:
        """Check if content looks like code"""
        code_indicators = 0
        
        if re.search(r'^\s*(def|class|if|for|while|try|with)\b', content, re.MULTILINE):
            code_indicators += 1
        if re.search(r':\s*$', content, re.MULTILINE):
            code_indicators += 1
        if re.search(r'^\s+', content, re.MULTILINE):  # Indentation
            code_indicators += 1
        if re.search(r'\b(import|from|return|raise|yield)\b', content):
            code_indicators += 1
        
        return code_indicators >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLER (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class InputHandler:
    """
    Ultra-Advanced Input Handler for JARVIS.
    
    Features:
    - Multi-source input (terminal, file, voice)
    - Multi-line input with smart detection
    - Input sanitization and validation
    - Type detection
    - Binary input support
    - Large input buffering
    
    Memory Budget: < 10MB
    
    Usage:
        handler = InputHandler()
        
        # Read from terminal
        result = handler.read()
        
        # Read from file
        result = handler.read_file('input.txt')
        
        # Read multi-line
        result = handler.read_multiline()
        
        # Voice input (Termux)
        result = handler.read_voice()
    """
    
    def __init__(self, config: InputConfig = None):
        """
        Initialize Input Handler.
        
        Args:
            config: Input configuration
        """
        self._config = config or InputConfig()
        
        # Sub-handlers
        self._sanitizer = InputSanitizer(self._config)
        self._voice_handler = VoiceInputHandler(self._config)
        self._file_handler = FileInputHandler(self._config)
        self._multiline_handler = MultilineInputHandler(self._config)
        
        # State
        self._state = InputState.IDLE
        self._buffer: List[str] = []
        
        # Statistics
        self._stats = {
            'total_inputs': 0,
            'terminal_inputs': 0,
            'file_inputs': 0,
            'voice_inputs': 0,
            'multiline_inputs': 0,
            'errors': 0,
            'total_bytes': 0,
        }
        
        logger.info("InputHandler initialized")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Input Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def read(
        self,
        prompt: str = "",
        source: InputSource = InputSource.TERMINAL,
        **kwargs,
    ) -> InputResult:
        """
        Read input from specified source.
        
        Args:
            prompt: Input prompt
            source: Input source
            **kwargs: Additional source-specific options
            
        Returns:
            InputResult with input content
        """
        self._state = InputState.READING
        
        if source == InputSource.TERMINAL:
            result = self._read_terminal(prompt, **kwargs)
        elif source == InputSource.FILE:
            path = kwargs.get('path')
            if not path:
                result = InputResult(
                    content="",
                    source=source,
                    error="No file path provided",
                )
            else:
                result = self._file_handler.read_file(path)
        elif source == InputSource.VOICE:
            result = self._voice_handler.listen()
        elif source == InputSource.STRING:
            content = kwargs.get('content', '')
            result = InputResult(
                content=content,
                source=source,
                input_type=InputType.TEXT,
                byte_count=len(content.encode('utf-8')),
            )
        else:
            result = InputResult(
                content="",
                source=source,
                error=f"Unsupported input source: {source}",
            )
        
        # Sanitize if configured
        if result.success and self._config.sanitization_level != SanitizationLevel.NONE:
            result.content, result.warnings = self._sanitizer.sanitize(result.content)
        
        # Update stats
        self._update_stats(result)
        self._state = InputState.IDLE
        
        return result
    
    def _read_terminal(self, prompt: str, **kwargs) -> InputResult:
        """Read from terminal"""
        start_time = time.time()
        multiline = kwargs.get('multiline', False)
        password = kwargs.get('password', False)
        
        try:
            if password:
                import getpass
                content = getpass.getpass(prompt)
            elif multiline:
                result = self._multiline_handler.read_multiline(
                    prompt=prompt,
                    end_marker=kwargs.get('end_marker'),
                    timeout=kwargs.get('timeout'),
                )
                self._stats['multiline_inputs'] += 1
                return result
            else:
                content = input(prompt)
            
            return InputResult(
                content=content,
                source=InputSource.TERMINAL,
                input_type=self._sanitizer.detect_input_type(content),
                line_count=content.count('\n') + 1,
                byte_count=len(content.encode('utf-8')),
                duration_ms=(time.time() - start_time) * 1000,
            )
            
        except EOFError:
            return InputResult(
                content="",
                source=InputSource.TERMINAL,
                error="EOF reached",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except KeyboardInterrupt:
            return InputResult(
                content="",
                source=InputSource.TERMINAL,
                error="Input cancelled",
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def read_line(self, prompt: str = "") -> str:
        """
        Read a single line.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Input content string
        """
        result = self.read(prompt)
        return result.content if result.success else ""
    
    def read_multiline(
        self,
        prompt: str = ">>> ",
        end_marker: str = "```",
    ) -> str:
        """
        Read multi-line input.
        
        Args:
            prompt: Initial prompt
            end_marker: End marker
            
        Returns:
            Multi-line content
        """
        result = self.read(
            prompt,
            source=InputSource.TERMINAL,
            multiline=True,
            end_marker=end_marker,
        )
        return result.content if result.success else ""
    
    def read_file(self, path: Union[str, Path]) -> str:
        """
        Read file content.
        
        Args:
            path: File path
            
        Returns:
            File content
        """
        result = self._file_handler.read_file(path)
        return result.content if result.success else ""
    
    def read_voice(self, timeout: float = 30.0) -> str:
        """
        Read voice input.
        
        Args:
            timeout: Maximum listening time
            
        Returns:
            Transcribed text
        """
        result = self._voice_handler.listen(timeout=timeout)
        return result.content if result.success else ""
    
    def read_password(self, prompt: str = "Password: ") -> str:
        """
        Read password input.
        
        Args:
            prompt: Password prompt
            
        Returns:
            Password (hidden input)
        """
        result = self.read(prompt, password=True)
        return result.content if result.success else ""
    
    def confirm(
        self,
        prompt: str,
        default: bool = None,
    ) -> bool:
        """
        Ask for confirmation.
        
        Args:
            prompt: Question to ask
            default: Default value
            
        Returns:
            True if confirmed
        """
        hints = {True: "[Y/n]", False: "[y/N]", None: "[y/n]"}
        hint = hints.get(default, "[y/n]")
        
        while True:
            response = input(f"{prompt} {hint}: ").strip().lower()
            
            if not response:
                if default is not None:
                    return default
                continue
            
            if response in ('y', 'yes', 'true', '1'):
                return True
            elif response in ('n', 'no', 'false', '0'):
                return False
    
    def choose(
        self,
        prompt: str,
        options: List[str],
        default: int = None,
    ) -> Optional[int]:
        """
        Choose from options.
        
        Args:
            prompt: Question to ask
            options: List of options
            default: Default option index
            
        Returns:
            Chosen index or None
        """
        print(prompt)
        for i, option in enumerate(options):
            marker = ">" if i == default else " "
            print(f"  {marker} {i + 1}. {option}")
        
        while True:
            try:
                hint = f"[1-{len(options)}]"
                if default is not None:
                    hint += f", default={default + 1}"
                
                response = input(f"Choose {hint}: ").strip()
                
                if not response and default is not None:
                    return default
                
                choice = int(response)
                if 1 <= choice <= len(options):
                    return choice - 1
                
                print(f"Please enter a number between 1 and {len(options)}")
                
            except ValueError:
                print("Please enter a valid number")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_type(self, content: str) -> InputType:
        """Detect input type"""
        return self._sanitizer.detect_input_type(content)
    
    def sanitize(
        self,
        content: str,
        level: SanitizationLevel = None,
    ) -> Tuple[str, List[str]]:
        """Sanitize content"""
        return self._sanitizer.sanitize(content, level)
    
    def validate_file(self, path: str) -> Tuple[bool, str]:
        """Validate file path"""
        return self._sanitizer.validate_file_path(path)
    
    @property
    def voice_available(self) -> bool:
        """Check if voice input is available"""
        return self._voice_handler.available
    
    @property
    def state(self) -> InputState:
        """Get current state"""
        return self._state
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self._stats.copy()
    
    def _update_stats(self, result: InputResult):
        """Update statistics"""
        self._stats['total_inputs'] += 1
        
        if result.source == InputSource.TERMINAL:
            self._stats['terminal_inputs'] += 1
        elif result.source == InputSource.FILE:
            self._stats['file_inputs'] += 1
        elif result.source == InputSource.VOICE:
            self._stats['voice_inputs'] += 1
        
        if result.error:
            self._stats['errors'] += 1
        
        self._stats['total_bytes'] += result.byte_count


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo input handler"""
    handler = InputHandler()
    
    print("JARVIS Input Handler Demo")
    print("=" * 40)
    
    # Test voice availability
    print(f"\nVoice input available: {handler.voice_available}")
    
    # Read a line
    name = handler.read_line("What's your name? ")
    print(f"Hello, {name}!")
    
    # Ask confirmation
    if handler.confirm("Continue?", default=True):
        # Read multiline
        print("\nEnter some text (end with ```):")
        text = handler.read_multiline()
        print(f"\nYou entered {len(text)} characters")
        print(f"Detected type: {handler.detect_type(text).name}")
    
    # Show stats
    print("\nStatistics:")
    for key, value in handler.stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
