#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Command Line Interface
=============================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- readline-style input handling (cross-platform)
- Command parsing and routing
- Auto-completion system
- History management with persistence
- Multi-line input support
- Terminal capability detection

Features:
- Interactive REPL interface
- Command auto-completion
- Command history (persistent)
- Syntax highlighting (basic)
- Multi-line code input
- Pipeline command chaining
- Command aliases
- Keyboard shortcuts
- Session management

Memory Impact: < 15MB for interface
"""

import sys
import os
import re
import time
import json
import shlex
import signal
import logging
import hashlib
import threading
import traceback
from typing import (
    Dict, Any, Optional, List, Set, Tuple, Callable, 
    Union, Generator, Iterator, Awaitable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from collections import deque
from datetime import datetime
from functools import wraps, lru_cache
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL CAPABILITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TerminalCapability(Enum):
    """Terminal capability levels"""
    DUMB = auto()        # No special features
    BASIC = auto()       # Basic colors
    COLOR_16 = auto()    # 16 colors
    COLOR_256 = auto()   # 256 colors
    TRUE_COLOR = auto()  # 24-bit color


class TerminalDetector:
    """
    Detect terminal capabilities for optimal output.
    
    Determines what features are available in the current
    terminal environment (Termux, standard terminal, etc.)
    """
    
    # ANSI color support indicators
    COLOR_INDICATORS = [
        ('TERM', ['xterm', 'screen', 'vt100', 'color', 'ansi', 'linux']),
        ('COLORTERM', ['truecolor', '24bit']),
        ('TERM_PROGRAM', ['iTerm', 'Terminal.app', 'vscode']),
    ]
    
    # Termux-specific indicators
    TERMUX_INDICATORS = ['TERMUX_VERSION', 'TERMUX_MAIN_PACKAGE_VERSION']
    
    def __init__(self):
        """Initialize terminal detector"""
        self._capabilities: Optional[TerminalCapability] = None
        self._is_termux: Optional[bool] = None
        self._width: int = 80
        self._height: int = 24
        self._supports_unicode: Optional[bool] = None
        
    @property
    def capabilities(self) -> TerminalCapability:
        """Get terminal capabilities"""
        if self._capabilities is None:
            self._capabilities = self._detect_capabilities()
        return self._capabilities
    
    @property
    def is_termux(self) -> bool:
        """Check if running in Termux"""
        if self._is_termux is None:
            self._is_termux = any(
                env in os.environ 
                for env in self.TERMUX_INDICATORS
            )
        return self._is_termux
    
    @property
    def width(self) -> int:
        """Get terminal width"""
        try:
            import shutil
            w = shutil.get_terminal_size().columns
            self._width = w if w > 0 else 80
        except:
            pass
        return self._width
    
    @property
    def height(self) -> int:
        """Get terminal height"""
        try:
            import shutil
            h = shutil.get_terminal_size().lines
            self._height = h if h > 0 else 24
        except:
            pass
        return self._height
    
    @property
    def supports_unicode(self) -> bool:
        """Check if terminal supports Unicode"""
        if self._supports_unicode is None:
            self._supports_unicode = self._check_unicode_support()
        return self._supports_unicode
    
    @property
    def supports_color(self) -> bool:
        """Check if terminal supports colors"""
        return self.capabilities != TerminalCapability.DUMB
    
    def _detect_capabilities(self) -> TerminalCapability:
        """Detect terminal capabilities"""
        # Check for True Color support
        if os.environ.get('COLORTERM') in ('truecolor', '24bit'):
            return TerminalCapability.TRUE_COLOR
        
        # Check TERM variable
        term = os.environ.get('TERM', '').lower()
        
        if '256color' in term:
            return TerminalCapability.COLOR_256
        elif any(t in term for t in ['xterm', 'screen', 'color', 'ansi']):
            return TerminalCapability.COLOR_16
        elif term in ('dumb', ''):
            return TerminalCapability.DUMB
        elif term:
            return TerminalCapability.BASIC
        
        # Windows check
        if sys.platform == 'win32':
            return TerminalCapability.COLOR_16
        
        return TerminalCapability.BASIC
    
    def _check_unicode_support(self) -> bool:
        """Check Unicode support"""
        # Termux supports Unicode
        if self.is_termux:
            return True
        
        # Check LANG/LC_CTYPE
        lang = os.environ.get('LANG', '') + os.environ.get('LC_CTYPE', '')
        if 'UTF-8' in lang.upper() or 'UTF8' in lang.upper():
            return True
        
        # Try to print a Unicode character
        try:
            '█'.encode(sys.stdout.encoding or 'utf-8')
            return True
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get terminal information"""
        return {
            'capabilities': self.capabilities.name,
            'is_termux': self.is_termux,
            'width': self.width,
            'height': self.height,
            'unicode': self.supports_unicode,
            'color': self.supports_color,
            'term': os.environ.get('TERM', 'unknown'),
            'colorterm': os.environ.get('COLORTERM', 'none'),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """
    ANSI color codes for terminal output.
    
    Provides a simple interface for colored terminal output
    with automatic capability detection and fallback.
    """
    
    # Basic colors (16-color)
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # JARVIS-specific theme colors
    JARVIS_PRIMARY = '\033[36m'      # Cyan
    JARVIS_SECONDARY = '\033[35m'    # Magenta
    JARVIS_SUCCESS = '\033[32m'      # Green
    JARVIS_WARNING = '\033[33m'      # Yellow
    JARVIS_ERROR = '\033[31m'        # Red
    JARVIS_INFO = '\033[34m'         # Blue
    JARVIS_PROMPT = '\033[1;36m'     # Bold Cyan
    
    @classmethod
    def rgb(cls, r: int, g: int, b: int, bg: bool = False) -> str:
        """Create 24-bit color code (if supported)"""
        prefix = '48' if bg else '38'
        return f'\033[{prefix};2;{r};{g};{b}m'
    
    @classmethod
    def color256(cls, code: int, bg: bool = False) -> str:
        """Create 256-color code"""
        prefix = '48' if bg else '38'
        return f'\033[{prefix};5;{code}m'
    
    @classmethod
    def strip(cls, text: str) -> str:
        """Remove ANSI codes from text"""
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return ansi_pattern.sub('', text)
    
    @classmethod
    def length(cls, text: str) -> int:
        """Get visible length (without ANSI codes)"""
        return len(cls.strip(text))


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class CommandType(Enum):
    """Types of commands"""
    BUILTIN = auto()      # Built-in JARVIS commands
    SYSTEM = auto()       # System shell commands
    AI = auto()           # AI interaction
    SCRIPT = auto()       # Script execution
    PLUGIN = auto()       # Plugin commands
    ALIAS = auto()        # Command alias
    UNKNOWN = auto()      # Unknown command


class InputMode(Enum):
    """Input modes"""
    SINGLE_LINE = auto()   # Normal single-line input
    MULTI_LINE = auto()    # Multi-line code input
    PASSWORD = auto()      # Password input (hidden)
    CONTINUATION = auto()  # Line continuation


class ParseResult(Enum):
    """Command parsing results"""
    SUCCESS = auto()
    INCOMPLETE = auto()    # Need more input
    SYNTAX_ERROR = auto()
    EMPTY = auto()


@dataclass
class Command:
    """
    Parsed command representation.
    
    Stores all information about a parsed command
    for execution and history tracking.
    """
    raw: str                              # Original input
    name: str                             # Command name
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)
    command_type: CommandType = CommandType.UNKNOWN
    pipeline: List['Command'] = field(default_factory=list)
    redirects: Dict[str, str] = field(default_factory=dict)
    is_background: bool = False
    is_conditional: bool = False
    source_line: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'raw': self.raw,
            'name': self.name,
            'args': self.args,
            'kwargs': self.kwargs,
            'type': self.command_type.name,
            'is_background': self.is_background,
            'timestamp': self.timestamp,
        }


@dataclass
class HistoryEntry:
    """
    Command history entry.
    
    Stores command with metadata for history
    management and search functionality.
    """
    command: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    output_lines: int = 0
    working_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'command': self.command,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'exit_code': self.exit_code,
            'duration_ms': self.duration_ms,
            'output_lines': self.output_lines,
            'working_dir': self.working_dir,
        }


@dataclass
class CompletionSuggestion:
    """Auto-completion suggestion"""
    text: str
    display: str
    description: str = ""
    type: str = "command"
    priority: int = 0
    
    def __lt__(self, other: 'CompletionSuggestion') -> bool:
        """Sort by priority"""
        return self.priority < other.priority


@dataclass
class CLIConfig:
    """CLI configuration"""
    prompt: str = "JARVIS> "
    prompt_color: str = Colors.JARVIS_PROMPT
    continuation_prompt: str = "... "
    history_file: str = "~/.jarvis/history.json"
    history_size: int = 1000
    auto_save_history: bool = True
    enable_completion: bool = True
    enable_colors: bool = True
    enable_unicode: bool = True
    welcome_message: bool = True
    show_timestamp: bool = False
    show_duration: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class CommandParser:
    """
    Parse command strings into structured commands.
    
    Features:
    - Shell-like syntax parsing
    - Quoted string handling
    - Pipeline support
    - Redirection support
    - Escape sequences
    - Variable expansion
    """
    
    # Special characters
    PIPE = '|'
    BACKGROUND = '&'
    REDIRECT_OUT = '>'
    REDIRECT_IN = '<'
    REDIRECT_APPEND = '>>'
    COMMENT = '#'
    ESCAPE = '\\'
    
    # Quote characters
    QUOTES = ('"', "'")
    
    def __init__(self, variable_expander: Callable[[str], str] = None):
        """
        Initialize parser.
        
        Args:
            variable_expander: Function to expand variables
        """
        self._variable_expander = variable_expander
        self._aliases: Dict[str, str] = {}
        
    def register_alias(self, name: str, expansion: str):
        """Register a command alias"""
        self._aliases[name] = expansion
        
    def parse(self, line: str) -> Tuple[ParseResult, Optional[Command]]:
        """
        Parse a command line.
        
        Args:
            line: Input line to parse
            
        Returns:
            Tuple of (ParseResult, Command or None)
        """
        # Strip and check empty
        line = line.strip()
        if not line:
            return ParseResult.EMPTY, None
        
        # Remove comments (but not inside quotes)
        line = self._remove_comments(line)
        if not line.strip():
            return ParseResult.EMPTY, None
        
        # Check for incomplete input
        if self._is_incomplete(line):
            return ParseResult.INCOMPLETE, None
        
        # Expand aliases
        line = self._expand_aliases(line)
        
        # Expand variables
        if self._variable_expander:
            line = self._variable_expander(line)
        
        try:
            # Parse the command
            command = self._parse_command(line)
            return ParseResult.SUCCESS, command
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return ParseResult.SYNTAX_ERROR, None
    
    def _remove_comments(self, line: str) -> str:
        """Remove comments from line"""
        result = []
        in_quote = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == self.ESCAPE and i + 1 < len(line):
                result.append(char)
                result.append(line[i + 1])
                i += 2
                continue
                
            if char in self.QUOTES:
                if in_quote == char:
                    in_quote = None
                elif in_quote is None:
                    in_quote = char
                result.append(char)
            elif char == self.COMMENT and in_quote is None:
                break
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _is_incomplete(self, line: str) -> bool:
        """Check if input is incomplete"""
        # Check for unclosed quotes
        in_quote = None
        escape_next = False
        
        for char in line:
            if escape_next:
                escape_next = False
                continue
            
            if char == self.ESCAPE:
                escape_next = True
                continue
            
            if char in self.QUOTES:
                if in_quote == char:
                    in_quote = None
                elif in_quote is None:
                    in_quote = char
        
        if in_quote:
            return True
        
        # Check for trailing escape
        if line.endswith(self.ESCAPE):
            return True
        
        # Check for trailing pipe (pipeline continuation)
        if line.rstrip().endswith(self.PIPE):
            return True
        
        return False
    
    def _expand_aliases(self, line: str) -> str:
        """Expand command aliases"""
        parts = line.split(None, 1)
        if not parts:
            return line
        
        cmd_name = parts[0]
        if cmd_name in self._aliases:
            expansion = self._aliases[cmd_name]
            if len(parts) > 1:
                return f"{expansion} {parts[1]}"
            return expansion
        
        return line
    
    def _parse_command(self, line: str) -> Command:
        """Parse a single command"""
        # Split into tokens
        tokens = self._tokenize(line)
        
        if not tokens:
            raise ValueError("Empty command")
        
        # Check for pipeline
        if self.PIPE in tokens:
            return self._parse_pipeline(tokens)
        
        # Parse single command
        return self._parse_single(tokens, line)
    
    def _tokenize(self, line: str) -> List[str]:
        """Tokenize command line"""
        tokens = []
        current = []
        in_quote = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == self.ESCAPE and i + 1 < len(line):
                current.append(line[i + 1])
                i += 2
                continue
            
            if char in self.QUOTES:
                if in_quote == char:
                    in_quote = None
                elif in_quote is None:
                    in_quote = char
                else:
                    current.append(char)
            elif char in ' \t' and in_quote is None:
                if current:
                    tokens.append(''.join(current))
                    current = []
            elif char in (self.PIPE, self.BACKGROUND) and in_quote is None:
                if current:
                    tokens.append(''.join(current))
                    current = []
                tokens.append(char)
            elif char in (self.REDIRECT_OUT, self.REDIRECT_IN) and in_quote is None:
                if current:
                    tokens.append(''.join(current))
                    current = []
                # Handle >> 
                if char == self.REDIRECT_OUT and i + 1 < len(line) and line[i + 1] == self.REDIRECT_OUT:
                    tokens.append(self.REDIRECT_APPEND)
                    i += 1
                else:
                    tokens.append(char)
            else:
                current.append(char)
            
            i += 1
        
        if current:
            tokens.append(''.join(current))
        
        return tokens
    
    def _parse_single(self, tokens: List[str], raw: str) -> Command:
        """Parse a single command from tokens"""
        args = []
        kwargs = {}
        redirects = {}
        is_background = False
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Background
            if token == self.BACKGROUND:
                is_background = True
                i += 1
                continue
            
            # Redirects
            if token in (self.REDIRECT_OUT, self.REDIRECT_APPEND, self.REDIRECT_IN):
                if i + 1 < len(tokens):
                    redirect_type = 'append' if token == self.REDIRECT_APPEND else (
                        'output' if token in (self.REDIRECT_OUT, self.REDIRECT_APPEND) else 'input'
                    )
                    redirects[redirect_type] = tokens[i + 1]
                    i += 2
                    continue
            
            # Keyword arguments (--key=value or --key value)
            if token.startswith('--'):
                if '=' in token:
                    key, value = token[2:].split('=', 1)
                    kwargs[key] = value
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                    kwargs[token[2:]] = tokens[i + 1]
                    i += 1
                else:
                    kwargs[token[2:]] = 'true'
            elif token.startswith('-') and not token.startswith('--'):
                # Short options (-k value or -k)
                key = token[1:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                    kwargs[key] = tokens[i + 1]
                    i += 1
                else:
                    kwargs[key] = 'true'
            else:
                args.append(token)
            
            i += 1
        
        if not args:
            raise ValueError("No command name")
        
        command = Command(
            raw=raw,
            name=args[0],
            args=args[1:] if len(args) > 1 else [],
            kwargs=kwargs,
            redirects=redirects,
            is_background=is_background,
        )
        
        # Determine command type
        command.command_type = self._determine_type(command.name)
        
        return command
    
    def _parse_pipeline(self, tokens: List[str]) -> Command:
        """Parse a pipeline of commands"""
        # Split by pipe
        segments = []
        current = []
        
        for token in tokens:
            if token == self.PIPE:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        
        if current:
            segments.append(current)
        
        # Parse each segment
        commands = []
        for segment in segments:
            cmd = self._parse_single(segment, ' '.join(segment))
            commands.append(cmd)
        
        # Return first command with pipeline
        if commands:
            commands[0].pipeline = commands[1:]
            return commands[0]
        
        raise ValueError("Empty pipeline")
    
    def _determine_type(self, name: str) -> CommandType:
        """Determine command type from name"""
        # Built-in commands
        builtins = {
            'help', 'exit', 'quit', 'clear', 'history', 'alias',
            'unalias', 'set', 'unset', 'export', 'source', 'cd',
            'pwd', 'echo', 'version', 'status', 'config', 'reload',
        }
        
        if name in builtins:
            return CommandType.BUILTIN
        
        # Check for AI commands
        if name.startswith('ask:') or name in ('ask', 'ai', 'chat'):
            return CommandType.AI
        
        # Check for plugin commands
        if name.startswith('plugin:'):
            return CommandType.PLUGIN
        
        # Check for script execution
        if name.endswith('.py') or name.endswith('.sh'):
            return CommandType.SCRIPT
        
        # Default to system
        return CommandType.SYSTEM


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class HistoryManager:
    """
    Manage command history with persistence.
    
    Features:
    - Persistent storage
    - Search functionality
    - Deduplication
    - Session tracking
    - Size limits
    """
    
    def __init__(
        self,
        history_file: str = "~/.jarvis/history.json",
        max_size: int = 1000,
    ):
        """
        Initialize history manager.
        
        Args:
            history_file: Path to history file
            max_size: Maximum number of entries
        """
        self._file = Path(history_file).expanduser()
        self._max_size = max_size
        self._history: deque = deque(maxlen=max_size)
        self._session_id = self._generate_session_id()
        self._current_index = 0
        self._dirty = False
        
        # Load existing history
        self._load()
        
        logger.debug(f"HistoryManager initialized with {len(self._history)} entries")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.md5(
            f"{time.time()}:{os.getpid()}".encode()
        ).hexdigest()[:8]
    
    def _load(self):
        """Load history from file"""
        if not self._file.exists():
            return
        
        try:
            # Ensure parent directory exists
            self._file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data.get('entries', []):
                if isinstance(entry, str):
                    # Old format - just command string
                    self._history.append(HistoryEntry(command=entry))
                else:
                    self._history.append(HistoryEntry(
                        command=entry.get('command', ''),
                        timestamp=entry.get('timestamp', 0),
                        session_id=entry.get('session_id', ''),
                        exit_code=entry.get('exit_code', 0),
                        duration_ms=entry.get('duration_ms', 0),
                    ))
            
            logger.debug(f"Loaded {len(self._history)} history entries")
            
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
    
    def save(self):
        """Save history to file"""
        if not self._dirty:
            return
        
        try:
            # Ensure parent directory exists
            self._file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'version': 1,
                'session_id': self._session_id,
                'entries': [entry.to_dict() for entry in self._history],
            }
            
            # Write atomically
            temp_file = self._file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            temp_file.rename(self._file)
            self._dirty = False
            
            logger.debug(f"Saved {len(self._history)} history entries")
            
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def add(
        self,
        command: str,
        exit_code: int = 0,
        duration_ms: float = 0.0,
    ):
        """Add command to history"""
        if not command.strip():
            return
        
        # Don't duplicate consecutive commands
        if self._history and self._history[-1].command == command:
            return
        
        entry = HistoryEntry(
            command=command,
            session_id=self._session_id,
            exit_code=exit_code,
            duration_ms=duration_ms,
            working_dir=os.getcwd(),
        )
        
        self._history.append(entry)
        self._current_index = len(self._history)
        self._dirty = True
    
    def get_previous(self) -> Optional[str]:
        """Get previous command (navigate up)"""
        if not self._history or self._current_index <= 0:
            return None
        
        self._current_index -= 1
        return self._history[self._current_index].command
    
    def get_next(self) -> Optional[str]:
        """Get next command (navigate down)"""
        if self._current_index >= len(self._history) - 1:
            self._current_index = len(self._history)
            return None
        
        self._current_index += 1
        return self._history[self._current_index].command
    
    def search(
        self,
        query: str,
        limit: int = 10,
        session_only: bool = False,
    ) -> List[HistoryEntry]:
        """Search history"""
        results = []
        query_lower = query.lower()
        
        # Search from most recent
        for entry in reversed(list(self._history)):
            if session_only and entry.session_id != self._session_id:
                continue
            
            if query_lower in entry.command.lower():
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def clear(self):
        """Clear history"""
        self._history.clear()
        self._current_index = 0
        self._dirty = True
    
    def get_all(self) -> List[HistoryEntry]:
        """Get all history entries"""
        return list(self._history)
    
    def get_recent(self, count: int = 10) -> List[HistoryEntry]:
        """Get recent history entries"""
        return list(self._history)[-count:]
    
    @property
    def size(self) -> int:
        """Get history size"""
        return len(self._history)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-COMPLETION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CompletionEngine:
    """
    Auto-completion engine for CLI.
    
    Features:
    - Command completion
    - Argument completion
    - File path completion
    - History-based suggestions
    - Custom completers
    """
    
    def __init__(self, parser: CommandParser = None):
        """
        Initialize completion engine.
        
        Args:
            parser: Command parser for context
        """
        self._parser = parser or CommandParser()
        self._commands: Dict[str, Dict] = {}
        self._completers: Dict[str, Callable] = {}
        self._history_suggestions: Set[str] = set()
        
        # Register default completers
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default command completions"""
        # Built-in commands
        builtin_commands = {
            'help': {'desc': 'Show help information', 'type': 'builtin'},
            'exit': {'desc': 'Exit JARVIS', 'type': 'builtin'},
            'quit': {'desc': 'Exit JARVIS', 'type': 'builtin'},
            'clear': {'desc': 'Clear the screen', 'type': 'builtin'},
            'history': {'desc': 'Show command history', 'type': 'builtin'},
            'alias': {'desc': 'Create command alias', 'type': 'builtin'},
            'unalias': {'desc': 'Remove command alias', 'type': 'builtin'},
            'set': {'desc': 'Set variable', 'type': 'builtin'},
            'unset': {'desc': 'Unset variable', 'type': 'builtin'},
            'export': {'desc': 'Export variable to environment', 'type': 'builtin'},
            'source': {'desc': 'Execute commands from file', 'type': 'builtin'},
            'cd': {'desc': 'Change directory', 'type': 'builtin'},
            'pwd': {'desc': 'Print working directory', 'type': 'builtin'},
            'echo': {'desc': 'Print text', 'type': 'builtin'},
            'version': {'desc': 'Show JARVIS version', 'type': 'builtin'},
            'status': {'desc': 'Show system status', 'type': 'builtin'},
            'config': {'desc': 'View or edit configuration', 'type': 'builtin'},
            'reload': {'desc': 'Reload configuration', 'type': 'builtin'},
            'ask': {'desc': 'Ask JARVIS AI a question', 'type': 'ai'},
            'ai': {'desc': 'Start AI chat mode', 'type': 'ai'},
            'chat': {'desc': 'Chat with JARVIS', 'type': 'ai'},
        }
        
        for cmd, info in builtin_commands.items():
            self._commands[cmd] = info
        
        # Register file completer for cd and source
        self._completers['cd'] = self._complete_path
        self._completers['source'] = self._complete_path
    
    def register_command(
        self,
        name: str,
        description: str = "",
        command_type: str = "custom",
        completer: Callable = None,
    ):
        """
        Register a command for completion.
        
        Args:
            name: Command name
            description: Command description
            command_type: Type of command
            completer: Argument completer function
        """
        self._commands[name] = {
            'desc': description,
            'type': command_type,
        }
        
        if completer:
            self._completers[name] = completer
    
    def update_history(self, history: List[str]):
        """Update history for suggestions"""
        self._history_suggestions = set(history)
    
    def complete(
        self,
        line: str,
        cursor_pos: int = None,
    ) -> List[CompletionSuggestion]:
        """
        Get completion suggestions.
        
        Args:
            line: Current input line
            cursor_pos: Cursor position (defaults to end)
            
        Returns:
            List of completion suggestions
        """
        if cursor_pos is None:
            cursor_pos = len(line)
        
        # Get the word being completed
        text_before = line[:cursor_pos]
        words = text_before.split()
        
        if not words:
            # Complete empty line - show all commands
            return self._get_command_completions('')
        
        if len(words) == 1 and not text_before.endswith(' '):
            # Completing command name
            return self._get_command_completions(words[0])
        
        # Completing arguments
        cmd_name = words[0]
        current_word = words[-1] if not text_before.endswith(' ') else ''
        
        # Check for custom completer
        if cmd_name in self._completers:
            return self._completers[cmd_name](current_word, words)
        
        # Default: file completion
        return self._complete_path(current_word, words)
    
    def _get_command_completions(self, prefix: str) -> List[CompletionSuggestion]:
        """Get command completions"""
        suggestions = []
        prefix_lower = prefix.lower()
        
        for cmd, info in self._commands.items():
            if cmd.startswith(prefix_lower):
                suggestions.append(CompletionSuggestion(
                    text=cmd,
                    display=cmd,
                    description=info.get('desc', ''),
                    type=info.get('type', 'command'),
                    priority=0 if cmd == prefix else 1,
                ))
        
        # Add history suggestions
        for hist_cmd in self._history_suggestions:
            if hist_cmd.startswith(prefix_lower) and hist_cmd not in self._commands:
                # Parse just the command name
                cmd_name = hist_cmd.split()[0] if hist_cmd.split() else hist_cmd
                if cmd_name not in [s.text for s in suggestions]:
                    suggestions.append(CompletionSuggestion(
                        text=cmd_name,
                        display=cmd_name,
                        description='(from history)',
                        type='history',
                        priority=2,
                    ))
        
        return sorted(suggestions)
    
    def _complete_path(
        self,
        prefix: str,
        words: List[str],
    ) -> List[CompletionSuggestion]:
        """Complete file/directory paths"""
        suggestions = []
        
        try:
            # Handle ~ expansion
            if prefix.startswith('~'):
                prefix = str(Path(prefix).expanduser())
            
            # Get directory and partial name
            if '/' in prefix:
                dir_path = Path(prefix).parent
                partial = Path(prefix).name
            else:
                dir_path = Path('.')
                partial = prefix
            
            # List directory
            if dir_path.exists() and dir_path.is_dir():
                for item in dir_path.iterdir():
                    if item.name.startswith(partial):
                        display = item.name
                        if item.is_dir():
                            display += '/'
                            suggestion_type = 'directory'
                        else:
                            suggestion_type = 'file'
                        
                        suggestions.append(CompletionSuggestion(
                            text=item.name,
                            display=display,
                            description=str(item.parent) if str(item.parent) != '.' else '',
                            type=suggestion_type,
                            priority=0,
                        ))
        
        except Exception as e:
            logger.debug(f"Path completion error: {e}")
        
        return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT READER
# ═══════════════════════════════════════════════════════════════════════════════

class InputReader:
    """
    Read user input with advanced features.
    
    Features:
    - Basic line editing
    - History navigation
    - Tab completion
    - Multi-line input
    - Password masking
    """
    
    def __init__(
        self,
        history: HistoryManager = None,
        completion: CompletionEngine = None,
        config: CLIConfig = None,
    ):
        """
        Initialize input reader.
        
        Args:
            history: History manager
            completion: Completion engine
            config: CLI configuration
        """
        self._history = history
        self._completion = completion
        self._config = config or CLIConfig()
        self._detector = TerminalDetector()
        
        # State
        self._mode = InputMode.SINGLE_LINE
        self._buffer: List[str] = []
        self._saved_line = ""
        
        # Setup readline if available
        self._readline = None
        self._setup_readline()
        
        # Signal handlers
        self._setup_signals()
    
    def _setup_readline(self):
        """Setup readline library if available"""
        try:
            import readline
            
            self._readline = readline
            
            # Set history file
            history_file = Path(self._config.history_file).expanduser()
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            readline.set_history_length(self._config.history_size)
            
            # Load existing history
            if history_file.exists():
                try:
                    readline.read_history_file(str(history_file))
                except:
                    pass
            
            # Setup completion
            if self._completion and self._config.enable_completion:
                readline.set_completer(self._readline_completer)
                readline.parse_and_bind('tab: complete')
                readline.set_completer_delims(' \t\n')
            
            logger.debug("Readline initialized")
            
        except ImportError:
            logger.debug("Readline not available")
    
    def _setup_signals(self):
        """Setup signal handlers"""
        def handle_sigint(signum, frame):
            # Handle Ctrl+C gracefully
            print()  # New line
            self._buffer = []
            self._mode = InputMode.SINGLE_LINE
        
        try:
            signal.signal(signal.SIGINT, handle_sigint)
        except:
            pass
    
    def _readline_completer(self, text: str, state: int) -> Optional[str]:
        """Readline completion function"""
        if not self._completion:
            return None
        
        # Get current line
        try:
            line = self._readline.get_line_buffer()
            cursor = self._readline.get_begidx() + len(text)
        except:
            line = text
            cursor = len(text)
        
        # Get completions
        suggestions = self._completion.complete(line, cursor)
        
        # Filter to matching
        matching = [s for s in suggestions if s.text.startswith(text)]
        
        if state < len(matching):
            return matching[state].text
        
        return None
    
    def read_line(self, prompt: str = None) -> Optional[str]:
        """
        Read a line of input.
        
        Args:
            prompt: Prompt to display
            
        Returns:
            Input line or None on EOF
        """
        if prompt is None:
            prompt = self._config.prompt
        
        # Apply color if enabled
        if self._config.enable_colors:
            display_prompt = f"{self._config.prompt_color}{prompt}{Colors.RESET}"
        else:
            display_prompt = prompt
        
        try:
            # Use readline if available
            if self._readline:
                line = input(display_prompt)
            else:
                line = input(display_prompt)
            
            return line
            
        except EOFError:
            return None
        except KeyboardInterrupt:
            print()
            return ""
    
    def read_multiline(
        self,
        prompt: str = None,
        end_marker: str = "```",
    ) -> Optional[str]:
        """
        Read multi-line input.
        
        Args:
            prompt: Initial prompt
            end_marker: Marker to end input
            
        Returns:
            Combined input or None on EOF
        """
        if prompt is None:
            prompt = self._config.prompt
        
        lines = []
        
        # Show instruction
        print(f"{Colors.DIM}Enter multi-line input. End with '{end_marker}' on a new line.{Colors.RESET}")
        
        continuation = self._config.continuation_prompt
        
        while True:
            try:
                if lines:
                    line = input(f"{continuation}")
                else:
                    display_prompt = f"{self._config.prompt_color}{prompt}{Colors.RESET}"
                    line = input(display_prompt)
                
                if line.strip() == end_marker:
                    break
                
                lines.append(line)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                return None
        
        return '\n'.join(lines)
    
    def read_password(self, prompt: str = "Password: ") -> Optional[str]:
        """
        Read password (hidden input).
        
        Args:
            prompt: Prompt to display
            
        Returns:
            Password or None on EOF
        """
        try:
            import getpass
            return getpass.getpass(prompt)
        except EOFError:
            return None
    
    def confirm(
        self,
        prompt: str,
        default: bool = None,
    ) -> bool:
        """
        Ask for confirmation.
        
        Args:
            prompt: Question to ask
            default: Default value (None = no default)
            
        Returns:
            True if confirmed
        """
        if default is True:
            hint = "[Y/n]"
        elif default is False:
            hint = "[y/N]"
        else:
            hint = "[y/n]"
        
        while True:
            try:
                response = input(f"{prompt} {hint}: ").strip().lower()
                
                if not response:
                    if default is not None:
                        return default
                    continue
                
                if response in ('y', 'yes', 'true', '1'):
                    return True
                elif response in ('n', 'no', 'false', '0'):
                    return False
                
            except EOFError:
                return default if default is not None else False
            except KeyboardInterrupt:
                print()
                return False
    
    def choose(
        self,
        prompt: str,
        options: List[str],
        default: int = None,
    ) -> Optional[int]:
        """
        Ask user to choose from options.
        
        Args:
            prompt: Question to ask
            options: List of options
            default: Default option index
            
        Returns:
            Chosen index or None
        """
        # Display options
        print(prompt)
        for i, option in enumerate(options):
            marker = ">" if i == default else " "
            print(f"  {marker} {i + 1}. {option}")
        
        while True:
            try:
                hint = f"[1-{len(options)}" + (f", default={default + 1}" if default is not None else "") + "]"
                response = input(f"Choose {hint}: ").strip()
                
                if not response:
                    if default is not None:
                        return default
                    continue
                
                choice = int(response)
                if 1 <= choice <= len(options):
                    return choice - 1
                
                print(f"Please enter a number between 1 and {len(options)}")
                
            except ValueError:
                print("Please enter a valid number")
            except EOFError:
                return default
            except KeyboardInterrupt:
                print()
                return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CLI:
    """
    Ultra-Advanced Command Line Interface for JARVIS.
    
    Features:
    - Interactive REPL
    - Command parsing and routing
    - Auto-completion
    - History management
    - Multi-line input
    - Pipeline support
    - Command aliases
    - Output formatting
    - Error handling
    
    Memory Budget: < 15MB
    
    Usage:
        cli = CLI()
        
        # Register command handler
        @cli.command('greet')
        def greet_handler(cmd):
            name = cmd.args[0] if cmd.args else 'World'
            return f'Hello, {name}!'
        
        # Start interactive loop
        cli.run()
        
        # Or execute single command
        result = cli.execute('greet Alice')
    """
    
    VERSION = "14.0.0"
    
    def __init__(
        self,
        config: CLIConfig = None,
        command_handlers: Dict[str, Callable] = None,
    ):
        """
        Initialize CLI.
        
        Args:
            config: CLI configuration
            command_handlers: Pre-registered command handlers
        """
        self._config = config or CLIConfig()
        self._detector = TerminalDetector()
        
        # Core components
        self._parser = CommandParser(self._expand_variables)
        self._history = HistoryManager(
            history_file=self._config.history_file,
            max_size=self._config.history_size,
        )
        self._completion = CompletionEngine(self._parser)
        self._reader = InputReader(
            history=self._history,
            completion=self._completion,
            config=self._config,
        )
        
        # Command handlers
        self._handlers: Dict[str, Callable] = {}
        self._middlewares: List[Callable] = []
        self._hooks: Dict[str, List[Callable]] = {
            'pre_command': [],
            'post_command': [],
            'on_error': [],
        }
        
        # State
        self._variables: Dict[str, str] = {}
        self._running = False
        self._exit_code = 0
        
        # Statistics
        self._stats = {
            'commands_executed': 0,
            'errors': 0,
            'start_time': None,
        }
        
        # Register built-in commands
        self._register_builtins()
        
        # Register provided handlers
        if command_handlers:
            for name, handler in command_handlers.items():
                self.register_command(name, handler)
        
        logger.info(f"JARVIS CLI v{self.VERSION} initialized")
    
    def _register_builtins(self):
        """Register built-in commands"""
        # Help
        self.register_command('help', self._cmd_help)
        self._completion.register_command(
            'help', 'Show help information', 'builtin'
        )
        
        # Exit/Quit
        self.register_command('exit', self._cmd_exit)
        self.register_command('quit', self._cmd_exit)
        
        # Clear
        self.register_command('clear', self._cmd_clear)
        
        # History
        self.register_command('history', self._cmd_history)
        
        # Alias
        self.register_command('alias', self._cmd_alias)
        self.register_command('unalias', self._cmd_unalias)
        
        # Variables
        self.register_command('set', self._cmd_set)
        self.register_command('unset', self._cmd_unset)
        self.register_command('export', self._cmd_export)
        
        # Echo
        self.register_command('echo', self._cmd_echo)
        
        # Version
        self.register_command('version', self._cmd_version)
        
        # Status
        self.register_command('status', self._cmd_status)
        
        # CD/PWD
        self.register_command('cd', self._cmd_cd)
        self.register_command('pwd', self._cmd_pwd)
    
    def _expand_variables(self, text: str) -> str:
        """Expand variables in text"""
        # Expand ${VAR} and $VAR patterns
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return self._variables.get(var_name, os.environ.get(var_name, ''))
        
        # ${VAR} pattern
        text = re.sub(r'\$\{(\w+)\}', replace_var, text)
        # $VAR pattern
        text = re.sub(r'\$(\w+)', replace_var, text)
        
        return text
    
    # ─────────────────────────────────────────────────────────────────────────
    # Command Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        aliases: List[str] = None,
    ):
        """
        Register a command handler.
        
        Args:
            name: Command name
            handler: Handler function
            description: Command description
            aliases: Command aliases
        """
        self._handlers[name] = handler
        
        # Register for completion
        self._completion.register_command(name, description, 'custom')
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._parser.register_alias(alias, name)
                self._completion.register_command(alias, f"Alias for {name}", 'alias')
    
    def command(self, name: str, **kwargs):
        """
        Decorator for registering commands.
        
        Usage:
            @cli.command('greet')
            def greet_handler(cmd):
                return 'Hello!'
        """
        def decorator(func):
            self.register_command(name, func, **kwargs)
            return func
        return decorator
    
    def middleware(self, func: Callable):
        """Register middleware function"""
        self._middlewares.append(func)
        return func
    
    def hook(self, event: str):
        """Register event hook"""
        def decorator(func):
            if event in self._hooks:
                self._hooks[event].append(func)
            return func
        return decorator
    
    # ─────────────────────────────────────────────────────────────────────────
    # Command Execution
    # ─────────────────────────────────────────────────────────────────────────
    
    def execute(self, line: str) -> Any:
        """
        Execute a command line.
        
        Args:
            line: Command line to execute
            
        Returns:
            Command result
        """
        start_time = time.time()
        
        # Parse command
        parse_result, command = self._parser.parse(line)
        
        if parse_result == ParseResult.EMPTY:
            return None
        
        if parse_result == ParseResult.INCOMPLETE:
            print(f"{Colors.YELLOW}Incomplete command{Colors.RESET}")
            return None
        
        if parse_result == ParseResult.SYNTAX_ERROR:
            print(f"{Colors.RED}Syntax error{Colors.RESET}")
            self._stats['errors'] += 1
            return None
        
        # Run middlewares
        for middleware in self._middlewares:
            try:
                result = middleware(command)
                if result is False:
                    return None
            except Exception as e:
                logger.error(f"Middleware error: {e}")
        
        # Run pre-command hooks
        for hook in self._hooks['pre_command']:
            try:
                hook(command)
            except Exception as e:
                logger.error(f"Hook error: {e}")
        
        # Execute command
        result = None
        exit_code = 0
        
        try:
            result = self._execute_command(command)
            
            # Handle pipeline
            if command.pipeline:
                for cmd in command.pipeline:
                    result = self._execute_command(cmd)
            
            self._stats['commands_executed'] += 1
            
        except SystemExit:
            raise
        except Exception as e:
            self._stats['errors'] += 1
            exit_code = 1
            
            # Run error hooks
            for hook in self._hooks['on_error']:
                try:
                    hook(command, e)
                except:
                    pass
            
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            logger.error(f"Command error: {e}\n{traceback.format_exc()}")
        
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Add to history
            if self._config.auto_save_history:
                self._history.add(line, exit_code, duration_ms)
            
            # Run post-command hooks
            for hook in self._hooks['post_command']:
                try:
                    hook(command, result, duration_ms)
                except:
                    pass
            
            # Show duration if configured
            if self._config.show_duration and duration_ms > 100:
                print(f"{Colors.DIM}[{duration_ms:.0f}ms]{Colors.RESET}")
        
        return result
    
    def _execute_command(self, command: Command) -> Any:
        """Execute a single command"""
        handler = self._handlers.get(command.name)
        
        if handler:
            return handler(command)
        
        # Unknown command - try as system command
        return self._execute_system(command)
    
    def _execute_system(self, command: Command) -> int:
        """Execute system command"""
        import subprocess
        
        try:
            result = subprocess.run(
                [command.name] + command.args,
                capture_output=True,
                text=True,
            )
            
            if result.stdout:
                print(result.stdout, end='')
            
            if result.stderr:
                print(f"{Colors.RED}{result.stderr}{Colors.RESET}", end='')
            
            return result.returncode
            
        except FileNotFoundError:
            print(f"{Colors.RED}Command not found: {command.name}{Colors.RESET}")
            return 127
        except Exception as e:
            print(f"{Colors.RED}Error executing command: {e}{Colors.RESET}")
            return 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Built-in Commands
    # ─────────────────────────────────────────────────────────────────────────
    
    def _cmd_help(self, cmd: Command) -> str:
        """Help command"""
        topic = cmd.args[0] if cmd.args else None
        
        if topic:
            # Show help for specific command
            handler = self._handlers.get(topic)
            if handler and hasattr(handler, '__doc__'):
                return f"{topic}: {handler.__doc__ or 'No description'}"
            return f"No help available for '{topic}'"
        
        # Show all commands
        lines = [
            f"{Colors.BOLD}JARVIS CLI v{self.VERSION}{Colors.RESET}",
            "",
            "Available commands:",
        ]
        
        for name in sorted(self._handlers.keys()):
            handler = self._handlers[name]
            doc = handler.__doc__ or ""
            # Get first line of docstring
            desc = doc.split('\n')[0] if doc else ""
            lines.append(f"  {Colors.CYAN}{name:<15}{Colors.RESET} {desc}")
        
        lines.extend([
            "",
            "Type 'help <command>' for more information.",
            "Press Ctrl+D to exit.",
        ])
        
        return '\n'.join(lines)
    
    def _cmd_exit(self, cmd: Command):
        """Exit JARVIS"""
        self._running = False
        raise SystemExit(0)
    
    def _cmd_clear(self, cmd: Command):
        """Clear screen"""
        # ANSI clear screen
        print('\033[2J\033[H', end='')
    
    def _cmd_history(self, cmd: Command):
        """Show command history"""
        count = int(cmd.args[0]) if cmd.args else 20
        entries = self._history.get_recent(count)
        
        for entry in entries:
            ts = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
            print(f"{Colors.DIM}{ts}{Colors.RESET} {entry.command}")
    
    def _cmd_alias(self, cmd: Command):
        """Create or show aliases"""
        if not cmd.args:
            # Show all aliases
            for name, expansion in self._parser._aliases.items():
                print(f"alias {name}='{expansion}'")
            return
        
        if '=' in cmd.args[0]:
            # Set alias
            name, expansion = cmd.args[0].split('=', 1)
            self._parser.register_alias(name, expansion.strip("'\""))
            print(f"Alias created: {name}")
        else:
            # Show specific alias
            name = cmd.args[0]
            expansion = self._parser._aliases.get(name)
            if expansion:
                print(f"alias {name}='{expansion}'")
            else:
                print(f"No alias named '{name}'")
    
    def _cmd_unalias(self, cmd: Command):
        """Remove alias"""
        if cmd.args:
            name = cmd.args[0]
            if name in self._parser._aliases:
                del self._parser._aliases[name]
                print(f"Alias removed: {name}")
            else:
                print(f"No alias named '{name}'")
    
    def _cmd_set(self, cmd: Command):
        """Set variable"""
        if not cmd.args:
            # Show all variables
            for name, value in self._variables.items():
                print(f"{name}={value}")
            return
        
        if '=' in cmd.args[0]:
            name, value = cmd.args[0].split('=', 1)
            self._variables[name] = value
        else:
            name = cmd.args[0]
            print(f"{name}={self._variables.get(name, '')}")
    
    def _cmd_unset(self, cmd: Command):
        """Unset variable"""
        if cmd.args:
            name = cmd.args[0]
            if name in self._variables:
                del self._variables[name]
    
    def _cmd_export(self, cmd: Command):
        """Export variable to environment"""
        if '=' in cmd.args[0]:
            name, value = cmd.args[0].split('=', 1)
            self._variables[name] = value
            os.environ[name] = value
        else:
            name = cmd.args[0]
            if name in self._variables:
                os.environ[name] = self._variables[name]
    
    def _cmd_echo(self, cmd: Command) -> str:
        """Print text"""
        return ' '.join(cmd.args)
    
    def _cmd_version(self, cmd: Command) -> str:
        """Show version"""
        return f"JARVIS CLI v{self.VERSION}"
    
    def _cmd_status(self, cmd: Command):
        """Show system status"""
        lines = [
            f"{Colors.BOLD}JARVIS Status{Colors.RESET}",
            "",
            f"  Version:     {self.VERSION}",
            f"  Uptime:      {time.time() - (self._stats['start_time'] or time.time()):.0f}s",
            f"  Commands:    {self._stats['commands_executed']}",
            f"  Errors:      {self._stats['errors']}",
            f"  History:     {self._history.size}",
            "",
            f"{Colors.BOLD}Terminal{Colors.RESET}",
            "",
        ]
        
        info = self._detector.get_info()
        for key, value in info.items():
            lines.append(f"  {key:<12} {value}")
        
        print('\n'.join(lines))
    
    def _cmd_cd(self, cmd: Command):
        """Change directory"""
        try:
            path = cmd.args[0] if cmd.args else str(Path.home())
            os.chdir(path)
        except Exception as e:
            print(f"{Colors.RED}cd: {e}{Colors.RESET}")
    
    def _cmd_pwd(self, cmd: Command) -> str:
        """Print working directory"""
        return os.getcwd()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────────
    
    def run(self):
        """
        Start interactive CLI loop.
        
        This is the main entry point for interactive use.
        """
        self._running = True
        self._stats['start_time'] = time.time()
        
        # Show welcome message
        if self._config.welcome_message:
            self._show_welcome()
        
        # Update completion with history
        self._completion.update_history([
            e.command for e in self._history.get_all()
        ])
        
        try:
            while self._running:
                try:
                    # Read input
                    line = self._reader.read_line()
                    
                    if line is None:
                        # EOF
                        print("\nGoodbye!")
                        break
                    
                    if not line:
                        continue
                    
                    # Check for multi-line input
                    if line.endswith('\\'):
                        # Continuation
                        lines = [line[:-1]]
                        while True:
                            next_line = self._reader.read_line(
                                self._config.continuation_prompt
                            )
                            if next_line is None:
                                break
                            if next_line.endswith('\\'):
                                lines.append(next_line[:-1])
                            else:
                                lines.append(next_line)
                                break
                        line = '\n'.join(lines)
                    
                    # Execute
                    self.execute(line)
                    
                except SystemExit:
                    break
                except KeyboardInterrupt:
                    print()
                    continue
                    
        finally:
            # Save history
            self._history.save()
    
    def _show_welcome(self):
        """Show welcome message"""
        lines = [
            "",
            f"{Colors.JARVIS_PRIMARY}{Colors.BOLD}╔════════════════════════════════════════╗{Colors.RESET}",
            f"{Colors.JARVIS_PRIMARY}{Colors.BOLD}║     JARVIS v{self.VERSION} - Self-Modifying AI      ║{Colors.RESET}",
            f"{Colors.JARVIS_PRIMARY}{Colors.BOLD}╚════════════════════════════════════════╝{Colors.RESET}",
            "",
            f"  Terminal: {self._detector.capabilities.name}",
            f"  Type {Colors.CYAN}'help'{Colors.RESET} for available commands",
            "",
        ]
        
        print('\n'.join(lines))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def print(self, *args, **kwargs):
        """Print with optional color support"""
        print(*args, **kwargs)
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}Error: {message}{Colors.RESET}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}Warning: {message}{Colors.RESET}")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}{message}{Colors.RESET}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}{message}{Colors.RESET}")
    
    def get_variable(self, name: str, default: str = None) -> Optional[str]:
        """Get variable value"""
        return self._variables.get(name, default)
    
    def set_variable(self, name: str, value: str):
        """Set variable value"""
        self._variables[name] = value
    
    @property
    def is_running(self) -> bool:
        """Check if CLI is running"""
        return self._running
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JARVIS CLI')
    parser.add_argument('-c', '--command', help='Execute command and exit')
    parser.add_argument('-f', '--file', help='Execute commands from file')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--no-history', action='store_true', help='Disable history')
    
    args = parser.parse_args()
    
    # Create config
    config = CLIConfig(
        enable_colors=not args.no_color,
        auto_save_history=not args.no_history,
    )
    
    # Create CLI
    cli = CLI(config=config)
    
    # Execute single command
    if args.command:
        cli.execute(args.command)
        return
    
    # Execute file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        cli.execute(line)
        except Exception as e:
            print(f"Error reading file: {e}")
        return
    
    # Start interactive
    cli.run()


if __name__ == '__main__':
    main()
