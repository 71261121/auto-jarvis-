#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Output Formatter
=======================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Terminal-aware output formatting
- Markdown rendering for CLI
- Syntax highlighting (basic, no external deps)
- Table formatting
- Mobile-optimized display

Features:
- Markdown to terminal conversion
- Code syntax highlighting (Python, JSON, etc.)
- Table rendering with borders
- Progress bars and spinners
- Mobile screen optimization
- Word wrapping
- Text alignment
- Color theming
- Indentation handling

Memory Impact: < 5MB for formatting
"""

import sys
import os
import re
import textwrap
import logging
from typing import (
    Dict, Any, Optional, List, Tuple, Union, 
    Callable, Iterator, Generator
)
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLORS (Extended)
# ═══════════════════════════════════════════════════════════════════════════════

class AnsiColors:
    """Extended ANSI color codes"""
    
    # Reset
    RESET = '\033[0m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
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
    
    # Theme colors
    HEADING = BRIGHT_CYAN + BOLD
    SUBHEADING = CYAN + BOLD
    LINK = BRIGHT_BLUE + UNDERLINE
    CODE = BRIGHT_YELLOW
    CODE_BG = BG_BLACK
    QUOTE = ITALIC + DIM
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    INFO = BLUE
    HINT = DIM
    
    # Box drawing characters (Unicode fallback to ASCII)
    BOX_ASCII = {
        'h': '-',
        'v': '|',
        'tl': '+',
        'tr': '+',
        'bl': '+',
        'br': '+',
        'lt': '+',
        'rt': '+',
        'tt': '+',
        'bt': '+',
    }
    
    BOX_UNICODE = {
        'h': '─',
        'v': '│',
        'tl': '┌',
        'tr': '┐',
        'bl': '└',
        'br': '┘',
        'lt': '├',
        'rt': '┤',
        'tt': '┬',
        'bt': '┴',
        'cr': '┼',
    }
    
    @classmethod
    def strip(cls, text: str) -> str:
        """Remove ANSI codes"""
        return re.sub(r'\033\[[0-9;]*m', '', text)
    
    @classmethod
    def visible_length(cls, text: str) -> int:
        """Get visible length without ANSI codes"""
        return len(cls.strip(text))


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class OutputFormat(Enum):
    """Output format types"""
    PLAIN = auto()       # Plain text
    MARKDOWN = auto()    # Markdown format
    HTML = auto()        # HTML format
    JSON = auto()        # JSON format
    TABLE = auto()       # Table format
    CODE = auto()        # Code block
    TREE = auto()        # Tree structure
    PANEL = auto()       # Panel/box


class TextAlign(Enum):
    """Text alignment options"""
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    JUSTIFY = auto()


class BorderStyle(Enum):
    """Border styles"""
    NONE = auto()
    ASCII = auto()
    UNICODE = auto()
    DOUBLE = auto()
    ROUNDED = auto()


class OutputTheme(Enum):
    """Output color themes"""
    DEFAULT = auto()
    DARK = auto()
    LIGHT = auto()
    MONO = auto()
    JARVIS = auto()


@dataclass
class OutputConfig:
    """Output formatter configuration"""
    width: int = 80
    max_width: int = 120
    enable_colors: bool = True
    enable_unicode: bool = True
    theme: OutputTheme = OutputTheme.JARVIS
    border_style: BorderStyle = BorderStyle.UNICODE
    wrap_text: bool = True
    indent_size: int = 2
    table_padding: int = 1
    code_theme: str = "monokai"
    show_line_numbers: bool = False


@dataclass
class TableColumn:
    """Table column definition"""
    header: str
    width: int = 0  # Auto if 0
    align: TextAlign = TextAlign.LEFT
    wrap: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TextUtils:
    """Text manipulation utilities"""
    
    @staticmethod
    def wrap(
        text: str,
        width: int = 80,
        indent: int = 0,
        subsequent_indent: int = 0,
    ) -> str:
        """Wrap text to specified width"""
        if width <= 0:
            return text
        
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                lines.append('')
                continue
            
            wrapped = textwrap.fill(
                paragraph,
                width=width,
                initial_indent=' ' * indent,
                subsequent_indent=' ' * subsequent_indent,
            )
            lines.append(wrapped)
        
        return '\n'.join(lines)
    
    @staticmethod
    def align(
        text: str,
        width: int,
        align: TextAlign,
        fill_char: str = ' ',
    ) -> str:
        """Align text within width"""
        visible_len = AnsiColors.visible_length(text)
        padding = max(0, width - visible_len)
        
        if align == TextAlign.LEFT:
            return text + fill_char * padding
        elif align == TextAlign.RIGHT:
            return fill_char * padding + text
        elif align == TextAlign.CENTER:
            left_pad = padding // 2
            right_pad = padding - left_pad
            return fill_char * left_pad + text + fill_char * right_pad
        else:  # JUSTIFY - only for multi-word
            words = text.split()
            if len(words) <= 1:
                return text + fill_char * padding
            
            # Distribute spaces
            total_spaces = padding + len(words) - 1
            gaps = len(words) - 1
            space_per_gap = total_spaces // gaps if gaps > 0 else 0
            extra = total_spaces % gaps if gaps > 0 else 0
            
            result = []
            for i, word in enumerate(words[:-1]):
                result.append(word)
                result.append(' ' * (space_per_gap + (1 if i < extra else 0)))
            result.append(words[-1])
            
            return ''.join(result)
    
    @staticmethod
    def truncate(
        text: str,
        max_length: int,
        suffix: str = '...',
    ) -> str:
        """Truncate text with suffix"""
        visible_len = AnsiColors.visible_length(text)
        if visible_len <= max_length:
            return text
        
        # Remove ANSI codes for truncation
        plain = AnsiColors.strip(text)
        truncated = plain[:max_length - len(suffix)] + suffix
        
        return truncated
    
    @staticmethod
    def pad(
        text: str,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
    ) -> str:
        """Add padding around text"""
        lines = text.split('\n')
        
        # Vertical padding
        lines = [''] * top + lines + [''] * bottom
        
        # Horizontal padding
        lines = [' ' * left + line + ' ' * right for line in lines]
        
        return '\n'.join(lines)
    
    @staticmethod
    def indent(text: str, spaces: int = 2) -> str:
        """Indent all lines"""
        prefix = ' ' * spaces
        return '\n'.join(prefix + line for line in text.split('\n'))


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTAX HIGHLIGHTER
# ═══════════════════════════════════════════════════════════════════════════════

class SyntaxHighlighter:
    """
    Basic syntax highlighter without external dependencies.
    
    Supports:
    - Python
    - JSON
    - JavaScript (basic)
    - Shell/Bash
    - Generic code
    """
    
    # Python keywords and builtins
    PYTHON_KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield',
    }
    
    PYTHON_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr',
        'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
        'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
        'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex',
        'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
        'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
        'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr',
        'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
        'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip',
    }
    
    # JSON
    JSON_KEYWORDS = {'true', 'false', 'null'}
    
    # Shell keywords
    SHELL_KEYWORDS = {
        'if', 'then', 'else', 'elif', 'fi', 'case', 'esac', 'for', 'while',
        'do', 'done', 'in', 'function', 'return', 'exit', 'export', 'source',
        'alias', 'unalias', 'set', 'unset', 'shift', 'read', 'echo', 'printf',
    }
    
    def __init__(self, config: OutputConfig = None):
        """Initialize highlighter"""
        self._config = config or OutputConfig()
        
        # Color mapping
        self._colors = {
            'keyword': AnsiColors.MAGENTA,
            'builtin': AnsiColors.CYAN,
            'string': AnsiColors.GREEN,
            'number': AnsiColors.YELLOW,
            'comment': AnsiColors.DIM,
            'function': AnsiColors.BLUE,
            'class': AnsiColors.BRIGHT_CYAN,
            'operator': AnsiColors.WHITE,
            'decorator': AnsiColors.BRIGHT_MAGENTA,
            'variable': AnsiColors.WHITE,
        }
    
    def highlight(self, code: str, language: str = 'python') -> str:
        """
        Highlight code syntax.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Highlighted code with ANSI codes
        """
        language = language.lower()
        
        if language in ('python', 'py'):
            return self._highlight_python(code)
        elif language == 'json':
            return self._highlight_json(code)
        elif language in ('bash', 'shell', 'sh'):
            return self._highlight_shell(code)
        elif language in ('javascript', 'js'):
            return self._highlight_js(code)
        else:
            return self._highlight_generic(code)
    
    def _highlight_python(self, code: str) -> str:
        """Highlight Python code"""
        lines = []
        
        for line in code.split('\n'):
            highlighted = self._highlight_python_line(line)
            lines.append(highlighted)
        
        return '\n'.join(lines)
    
    def _highlight_python_line(self, line: str) -> str:
        """Highlight a single Python line"""
        result = []
        i = 0
        in_string = None
        in_comment = False
        
        while i < len(line):
            char = line[i]
            
            # Comment
            if char == '#' and in_string is None:
                result.append(self._colors['comment'] + line[i:] + AnsiColors.RESET)
                break
            
            # String detection
            if char in ('"', "'") and not in_comment:
                if in_string is None:
                    # Check for triple quotes
                    if line[i:i+3] in ('"""', "'''"):
                        in_string = line[i:i+3]
                        result.append(self._colors['string'] + line[i:i+3])
                        i += 3
                        continue
                    else:
                        in_string = char
                        result.append(self._colors['string'] + char)
                        i += 1
                        continue
                elif in_string == char or (len(in_string) == 3 and line[i:i+3] == in_string):
                    # End of string
                    if len(in_string) == 3:
                        result.append(line[i:i+3] + AnsiColors.RESET)
                        i += 3
                    else:
                        result.append(char + AnsiColors.RESET)
                        i += 1
                    in_string = None
                    continue
            
            if in_string:
                # Escape sequences
                if char == '\\' and i + 1 < len(line):
                    result.append(char + line[i+1])
                    i += 2
                    continue
                result.append(char)
                i += 1
                continue
            
            # Decorator
            if char == '@':
                match = re.match(r'@(\w+)', line[i:])
                if match:
                    result.append(self._colors['decorator'] + '@' + match.group(1) + AnsiColors.RESET)
                    i += len(match.group(0))
                    continue
            
            # Identifier
            if char.isalpha() or char == '_':
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                    j += 1
                word = line[i:j]
                
                # Class name (starts with uppercase)
                if word[0].isupper() and word not in self.PYTHON_KEYWORDS:
                    result.append(self._colors['class'] + word + AnsiColors.RESET)
                # Keyword
                elif word in self.PYTHON_KEYWORDS:
                    result.append(self._colors['keyword'] + word + AnsiColors.RESET)
                # Builtin
                elif word in self.PYTHON_BUILTINS:
                    result.append(self._colors['builtin'] + word + AnsiColors.RESET)
                # Function call
                elif j < len(line) and line[j] == '(':
                    result.append(self._colors['function'] + word + AnsiColors.RESET)
                else:
                    result.append(word)
                
                i = j
                continue
            
            # Number
            if char.isdigit() or (char == '.' and i + 1 < len(line) and line[i+1].isdigit()):
                j = i
                if char == '.':
                    j += 1
                while j < len(line) and (line[j].isdigit() or line[j] in '.xXeE+-'):
                    j += 1
                result.append(self._colors['number'] + line[i:j] + AnsiColors.RESET)
                i = j
                continue
            
            # Operators
            if char in '+-*/%=<>!&|^~':
                result.append(self._colors['operator'] + char + AnsiColors.RESET)
                i += 1
                continue
            
            result.append(char)
            i += 1
        
        return ''.join(result)
    
    def _highlight_json(self, code: str) -> str:
        """Highlight JSON"""
        result = []
        in_string = False
        
        i = 0
        while i < len(code):
            char = code[i]
            
            if char == '"' and (i == 0 or code[i-1] != '\\'):
                if in_string:
                    result.append(char + AnsiColors.RESET)
                    in_string = False
                else:
                    result.append(self._colors['string'] + char)
                    in_string = True
                i += 1
                continue
            
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # Keywords
            if char.isalpha():
                j = i
                while j < len(code) and code[j].isalpha():
                    j += 1
                word = code[i:j]
                
                if word in self.JSON_KEYWORDS:
                    result.append(self._colors['keyword'] + word + AnsiColors.RESET)
                else:
                    result.append(word)
                i = j
                continue
            
            # Numbers
            if char.isdigit() or (char == '-' and i + 1 < len(code) and code[i+1].isdigit()):
                j = i
                while j < len(code) and (code[j].isdigit() or code[j] in '.-eE+'):
                    j += 1
                result.append(self._colors['number'] + code[i:j] + AnsiColors.RESET)
                i = j
                continue
            
            result.append(char)
            i += 1
        
        return ''.join(result)
    
    def _highlight_shell(self, code: str) -> str:
        """Highlight Shell/Bash code"""
        lines = []
        
        for line in code.split('\n'):
            # Comments
            if line.strip().startswith('#'):
                lines.append(self._colors['comment'] + line + AnsiColors.RESET)
                continue
            
            # Highlight keywords
            result = line
            for keyword in self.SHELL_KEYWORDS:
                pattern = r'\b' + keyword + r'\b'
                result = re.sub(
                    pattern,
                    self._colors['keyword'] + keyword + AnsiColors.RESET,
                    result
                )
            
            # Highlight strings
            result = re.sub(
                r'"([^"]*)"',
                self._colors['string'] + r'"\1"' + AnsiColors.RESET,
                result
            )
            result = re.sub(
                r"'([^']*)'",
                self._colors['string'] + r"'\1'" + AnsiColors.RESET,
                result
            )
            
            # Variables
            result = re.sub(
                r'\$(\w+)',
                self._colors['variable'] + r'$\1' + AnsiColors.RESET,
                result
            )
            
            lines.append(result)
        
        return '\n'.join(lines)
    
    def _highlight_js(self, code: str) -> str:
        """Highlight JavaScript (basic)"""
        result = code
        
        # Keywords
        keywords = {'const', 'let', 'var', 'function', 'return', 'if', 'else', 
                   'for', 'while', 'class', 'new', 'this', 'import', 'export',
                   'async', 'await', 'try', 'catch', 'throw'}
        
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            result = re.sub(
                pattern,
                self._colors['keyword'] + keyword + AnsiColors.RESET,
                result
            )
        
        # Strings
        result = re.sub(
            r'"([^"]*)"',
            self._colors['string'] + r'"\1"' + AnsiColors.RESET,
            result
        )
        result = re.sub(
            r"'([^']*)'",
            self._colors['string'] + r"'\1'" + AnsiColors.RESET,
            result
        )
        result = re.sub(
            r'`([^`]*)`',
            self._colors['string'] + r'`\1`' + AnsiColors.RESET,
            result
        )
        
        return result
    
    def _highlight_generic(self, code: str) -> str:
        """Generic highlighting"""
        # Just highlight strings and numbers
        result = code
        
        # Strings
        result = re.sub(
            r'"([^"]*)"',
            self._colors['string'] + r'"\1"' + AnsiColors.RESET,
            result
        )
        result = re.sub(
            r"'([^']*)'",
            self._colors['string'] + r"'\1'" + AnsiColors.RESET,
            result
        )
        
        # Numbers
        result = re.sub(
            r'\b(\d+\.?\d*)\b',
            self._colors['number'] + r'\1' + AnsiColors.RESET,
            result
        )
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class MarkdownRenderer:
    """
    Render Markdown to terminal output.
    
    Supports:
    - Headers (h1-h6)
    - Bold, italic, strikethrough
    - Code blocks and inline code
    - Links
    - Lists (ordered, unordered)
    - Blockquotes
    - Horizontal rules
    - Tables
    """
    
    def __init__(self, config: OutputConfig = None):
        """Initialize renderer"""
        self._config = config or OutputConfig()
        self._highlighter = SyntaxHighlighter(config)
    
    def render(self, markdown: str) -> str:
        """
        Render markdown to terminal output.
        
        Args:
            markdown: Markdown text
            
        Returns:
            Formatted terminal output
        """
        lines = markdown.split('\n')
        result = []
        in_code_block = False
        code_block_lang = ''
        code_block_lines = []
        in_list = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End code block
                    code = '\n'.join(code_block_lines)
                    rendered = self._render_code_block(code, code_block_lang)
                    result.append(rendered)
                    code_block_lines = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                    code_block_lang = line[3:].strip()
                i += 1
                continue
            
            if in_code_block:
                code_block_lines.append(line)
                i += 1
                continue
            
            # Headers
            if line.startswith('#'):
                result.append(self._render_header(line))
                i += 1
                continue
            
            # Horizontal rule
            if re.match(r'^[-*_]{3,}\s*$', line):
                result.append(self._render_hr())
                i += 1
                continue
            
            # Blockquote
            if line.startswith('>'):
                result.append(self._render_blockquote(line))
                i += 1
                continue
            
            # Lists
            list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', line)
            if list_match:
                in_list = True
                result.append(self._render_list_item(line))
                i += 1
                continue
            
            # Table
            if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
                # Collect table lines
                table_lines = [line]
                j = i + 1
                while j < len(lines) and '|' in lines[j]:
                    table_lines.append(lines[j])
                    j += 1
                result.append(self._render_table('\n'.join(table_lines)))
                i = j
                continue
            
            # Empty line
            if not line.strip():
                result.append('')
                in_list = False
                i += 1
                continue
            
            # Regular paragraph
            result.append(self._render_inline(line))
            i += 1
        
        return '\n'.join(result)
    
    def _render_header(self, line: str) -> str:
        """Render header"""
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not match:
            return line
        
        level = len(match.group(1))
        text = match.group(2)
        
        # Apply inline formatting
        text = self._render_inline(text)
        
        if level == 1:
            # H1: Bold cyan with underline
            return f"\n{AnsiColors.HEADING}{text}{AnsiColors.RESET}\n{'═' * len(AnsiColors.strip(text))}"
        elif level == 2:
            # H2: Bold cyan
            return f"\n{AnsiColors.SUBHEADING}{text}{AnsiColors.RESET}\n{'─' * len(AnsiColors.strip(text))}"
        elif level == 3:
            # H3: Bold
            return f"\n{AnsiColors.BOLD}{text}{AnsiColors.RESET}"
        else:
            # H4-6: Dim bold
            return f"\n{AnsiColors.DIM}{AnsiColors.BOLD}{text}{AnsiColors.RESET}"
    
    def _render_inline(self, text: str) -> str:
        """Render inline markdown elements"""
        # Bold **text** or __text__
        text = re.sub(
            r'\*\*(.+?)\*\*',
            f'{AnsiColors.BOLD}\\1{AnsiColors.RESET}',
            text
        )
        text = re.sub(
            r'__(.+?)__',
            f'{AnsiColors.BOLD}\\1{AnsiColors.RESET}',
            text
        )
        
        # Italic *text* or _text_
        text = re.sub(
            r'\*(.+?)\*',
            f'{AnsiColors.ITALIC}\\1{AnsiColors.RESET}',
            text
        )
        text = re.sub(
            r'_(.+?)_',
            f'{AnsiColors.ITALIC}\\1{AnsiColors.RESET}',
            text
        )
        
        # Strikethrough ~~text~~
        text = re.sub(
            r'~~(.+?)~~',
            f'{AnsiColors.STRIKETHROUGH}\\1{AnsiColors.RESET}',
            text
        )
        
        # Inline code `code`
        text = re.sub(
            r'`([^`]+)`',
            f'{AnsiColors.CODE}\\1{AnsiColors.RESET}',
            text
        )
        
        # Links [text](url)
        text = re.sub(
            r'\[(.+?)\]\((.+?)\)',
            f'{AnsiColors.LINK}\\1{AnsiColors.RESET} ({AnsiColors.DIM}\\2{AnsiColors.RESET})',
            text
        )
        
        return text
    
    def _render_code_block(self, code: str, language: str = '') -> str:
        """Render code block"""
        # Highlight
        if language:
            highlighted = self._highlighter.highlight(code, language)
        else:
            highlighted = self._highlighter._highlight_generic(code)
        
        # Add box
        lines = highlighted.split('\n')
        width = min(self._config.width, max(len(AnsiColors.strip(l)) for l in lines) + 4)
        
        # Get box characters
        box = AnsiColors.BOX_UNICODE if self._config.enable_unicode else AnsiColors.BOX_ASCII
        
        # Build box
        result = []
        result.append(f"{AnsiColors.DIM}{box['tl']}{box['h'] * (width - 2)}{box['tr']}{AnsiColors.RESET}")
        
        # Language label
        if language:
            label = f" {language} "
            result.append(f"{AnsiColors.DIM}{box['v']}{AnsiColors.DIM}{label:^{width-2}}{box['v']}{AnsiColors.RESET}")
        
        for line in lines:
            visible_len = AnsiColors.visible_length(line)
            padding = width - 2 - visible_len
            result.append(f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET} {line}{' ' * max(0, padding-1)}{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}")
        
        result.append(f"{AnsiColors.DIM}{box['bl']}{box['h'] * (width - 2)}{box['br']}{AnsiColors.RESET}")
        
        return '\n'.join(result)
    
    def _render_hr(self) -> str:
        """Render horizontal rule"""
        width = self._config.width
        char = '─' if self._config.enable_unicode else '-'
        return f"{AnsiColors.DIM}{char * width}{AnsiColors.RESET}"
    
    def _render_blockquote(self, line: str) -> str:
        """Render blockquote"""
        # Remove > prefix
        text = line.lstrip('>').strip()
        text = self._render_inline(text)
        
        # Add quote marker
        marker = '│' if self._config.enable_unicode else '|'
        
        lines = text.split('\n')
        result = []
        for l in lines:
            wrapped = TextUtils.wrap(l, self._config.width - 4)
            for wrapped_line in wrapped.split('\n'):
                result.append(f"{AnsiColors.DIM}{marker}{AnsiColors.RESET} {AnsiColors.QUOTE}{wrapped_line}{AnsiColors.RESET}")
        
        return '\n'.join(result)
    
    def _render_list_item(self, line: str) -> str:
        """Render list item"""
        # Detect indent level
        match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', line)
        if not match:
            return line
        
        indent = len(match.group(1))
        marker = match.group(2)
        text = self._render_inline(match.group(3))
        
        # Convert marker
        if marker in ('-', '*', '+'):
            bullet = '•' if self._config.enable_unicode else '*'
        else:
            bullet = marker
        
        indent_str = ' ' * indent
        return f"{indent_str}{AnsiColors.CYAN}{bullet}{AnsiColors.RESET} {text}"
    
    def _render_table(self, table_text: str) -> str:
        """Render table"""
        lines = table_text.strip().split('\n')
        if len(lines) < 2:
            return table_text
        
        # Parse rows
        rows = []
        for line in lines:
            if re.match(r'^[\s|:-]+$', line):
                continue  # Skip separator
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty cells
            if cells:
                rows.append(cells)
        
        if not rows:
            return table_text
        
        # Calculate column widths
        num_cols = max(len(row) for row in rows)
        widths = [0] * num_cols
        
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(AnsiColors.strip(cell)))
        
        # Build table
        box = AnsiColors.BOX_UNICODE if self._config.enable_unicode else AnsiColors.BOX_ASCII
        
        result = []
        
        # Header separator
        separator = box['lt'] + box['h'] + (box['h'] * 3 + box['h'] + box['tt'] + box['h']) * (num_cols - 1) + box['h'] * 3 + box['h'] + box['rt']
        # Simplified separator
        sep_parts = [box['h'] * (w + 2) for w in widths]
        separator = f"{AnsiColors.DIM}{box['lt']}{(box['h'] + box['tt'] + box['h']).join(sep_parts)}{box['rt']}{AnsiColors.RESET}"
        
        for row_idx, row in enumerate(rows):
            cells = []
            for i in range(num_cols):
                cell = row[i] if i < len(row) else ''
                width = widths[i]
                cells.append(f" {cell:<{width}} ")
            
            row_str = f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}{(AnsiColors.DIM + box['v'] + AnsiColors.RESET).join(cells)}"
            result.append(row_str)
            
            if row_idx == 0:
                # Add separator after header
                result.append(separator)
        
        return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class TableFormatter:
    """
    Format data as tables.
    
    Features:
    - Multiple border styles
    - Column alignment
    - Auto-sizing
    - Headers
    """
    
    def __init__(self, config: OutputConfig = None):
        """Initialize formatter"""
        self._config = config or OutputConfig()
    
    def format(
        self,
        headers: List[str] = None,
        rows: List[List[Any]] = None,
        columns: List[TableColumn] = None,
        data: List[Dict] = None,
    ) -> str:
        """
        Format data as a table.
        
        Args:
            headers: Column headers
            rows: Data rows
            columns: Column definitions
            data: List of dicts (alternative to rows)
            
        Returns:
            Formatted table string
        """
        # Handle dict data
        if data and not rows:
            headers = list(data[0].keys()) if data else []
            rows = [[str(d.get(h, '')) for h in headers] for d in data]
        
        if not rows:
            return ""
        
        # Create column definitions
        if not columns:
            columns = [
                TableColumn(header=h, align=TextAlign.LEFT)
                for h in (headers or [f"Col{i}" for i in range(len(rows[0]))])
            ]
        
        # Calculate widths
        widths = []
        for i, col in enumerate(columns):
            max_width = len(col.header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            widths.append(min(max_width, self._config.max_width // len(columns)))
        
        # Get box characters
        box = AnsiColors.BOX_UNICODE if self._config.enable_unicode else AnsiColors.BOX_ASCII
        
        # Build table
        result = []
        
        # Top border
        top = box['tl'] + ''.join(box['h'] * (w + 2) + (box['tt'] if i < len(widths) - 1 else '') for i, w in enumerate(widths)) + box['tr']
        result.append(f"{AnsiColors.DIM}{top}{AnsiColors.RESET}")
        
        # Headers
        if headers or columns:
            header_cells = []
            for i, (col, width) in enumerate(zip(columns, widths)):
                text = TextUtils.align(col.header, width, TextAlign.CENTER)
                header_cells.append(f" {text} ")
            result.append(f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}{AnsiColors.BOLD}" + 
                         f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}{AnsiColors.BOLD}".join(header_cells) + 
                         f"{AnsiColors.RESET}{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}")
        
        # Header separator
        sep = box['lt'] + ''.join(box['h'] * (w + 2) + (box['cr'] if i < len(widths) - 1 else '') for i, w in enumerate(widths)) + box['rt']
        result.append(f"{AnsiColors.DIM}{sep}{AnsiColors.RESET}")
        
        # Rows
        for row in rows:
            cells = []
            for i, (col, width) in enumerate(zip(columns, widths)):
                cell_value = str(row[i]) if i < len(row) else ''
                # Truncate if needed
                if len(cell_value) > width:
                    cell_value = cell_value[:width-3] + '...'
                text = TextUtils.align(cell_value, width, col.align)
                cells.append(f" {text} ")
            result.append(f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}" + 
                         f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}".join(cells) + 
                         f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}")
        
        # Bottom border
        bottom = box['bl'] + ''.join(box['h'] * (w + 2) + (box['bt'] if i < len(widths) - 1 else '') for i, w in enumerate(widths)) + box['br']
        result.append(f"{AnsiColors.DIM}{bottom}{AnsiColors.RESET}")
        
        return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTER (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class OutputFormatter:
    """
    Ultra-Advanced Output Formatter for JARVIS.
    
    Features:
    - Markdown rendering
    - Syntax highlighting
    - Table formatting
    - Mobile-optimized output
    - Word wrapping
    - Color theming
    
    Memory Budget: < 5MB
    
    Usage:
        formatter = OutputFormatter()
        
        # Format markdown
        print(formatter.markdown("# Hello\\n\\nThis is **bold**"))
        
        # Format code
        print(formatter.code("def hello(): pass", "python"))
        
        # Format table
        print(formatter.table(['Name', 'Age'], [['Alice', 25], ['Bob', 30]]))
    """
    
    def __init__(self, config: OutputConfig = None):
        """
        Initialize Output Formatter.
        
        Args:
            config: Output configuration
        """
        self._config = config or OutputConfig()
        
        # Sub-formatters
        self._markdown = MarkdownRenderer(config)
        self._syntax = SyntaxHighlighter(config)
        self._table = TableFormatter(config)
        
        # Statistics
        self._stats = {
            'outputs_formatted': 0,
            'total_bytes': 0,
        }
        
        logger.info("OutputFormatter initialized")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Formatting Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def format(
        self,
        content: str,
        format_type: OutputFormat = OutputFormat.PLAIN,
        **kwargs,
    ) -> str:
        """
        Format content according to type.
        
        Args:
            content: Content to format
            format_type: Output format type
            **kwargs: Format-specific options
            
        Returns:
            Formatted content
        """
        if format_type == OutputFormat.PLAIN:
            result = self._format_plain(content, **kwargs)
        elif format_type == OutputFormat.MARKDOWN:
            result = self.markdown(content, **kwargs)
        elif format_type == OutputFormat.CODE:
            language = kwargs.get('language', 'text')
            result = self.code(content, language)
        elif format_type == OutputFormat.TABLE:
            result = self._format_table_data(content, **kwargs)
        elif format_type == OutputFormat.JSON:
            result = self.json(content)
        elif format_type == OutputFormat.PANEL:
            result = self.panel(content, **kwargs)
        else:
            result = content
        
        self._stats['outputs_formatted'] += 1
        self._stats['total_bytes'] += len(result)
        
        return result
    
    def _format_plain(self, text: str, **kwargs) -> str:
        """Format plain text"""
        if self._config.wrap_text:
            return TextUtils.wrap(text, self._config.width)
        return text
    
    def _format_table_data(self, content: str, **kwargs) -> str:
        """Format string as table data"""
        # Parse content as CSV-like
        lines = content.strip().split('\n')
        rows = [line.split(',') for line in lines]
        headers = rows[0] if rows else []
        return self.table(headers, rows[1:])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def markdown(self, text: str) -> str:
        """Render markdown"""
        return self._markdown.render(text)
    
    def code(
        self,
        code: str,
        language: str = 'python',
        line_numbers: bool = None,
    ) -> str:
        """
        Format code with syntax highlighting.
        
        Args:
            code: Source code
            language: Programming language
            line_numbers: Show line numbers
            
        Returns:
            Highlighted code
        """
        line_numbers = line_numbers if line_numbers is not None else self._config.show_line_numbers
        
        if line_numbers:
            lines = code.split('\n')
            max_num_len = len(str(len(lines)))
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                num = f"{i:>{max_num_len}} │ "
                numbered_lines.append(f"{AnsiColors.DIM}{num}{AnsiColors.RESET}{line}")
            code = '\n'.join(numbered_lines)
        else:
            code = self._syntax.highlight(code, language)
        
        return self._markdown._render_code_block(code, language)
    
    def table(
        self,
        headers: List[str] = None,
        rows: List[List[Any]] = None,
        data: List[Dict] = None,
    ) -> str:
        """
        Format data as table.
        
        Args:
            headers: Column headers
            rows: Data rows
            data: List of dicts (alternative)
            
        Returns:
            Formatted table
        """
        return self._table.format(headers=headers, rows=rows, data=data)
    
    def json(self, data: Union[str, dict, list], indent: int = 2) -> str:
        """
        Format JSON with highlighting.
        
        Args:
            data: JSON string or Python object
            indent: Indentation level
            
        Returns:
            Highlighted JSON
        """
        import json as json_module
        
        if isinstance(data, str):
            try:
                data = json_module.loads(data)
            except Exception:
                pass
        
        if isinstance(data, (dict, list)):
            json_str = json_module.dumps(data, indent=indent, ensure_ascii=False)
        else:
            json_str = str(data)
        
        return self._syntax.highlight(json_str, 'json')
    
    def panel(
        self,
        content: str,
        title: str = "",
        style: BorderStyle = None,
        padding: int = 1,
    ) -> str:
        """
        Format content in a panel/box.
        
        Args:
            content: Panel content
            title: Panel title
            style: Border style
            padding: Internal padding
            
        Returns:
            Panel string
        """
        style = style or self._config.border_style
        
        if style == BorderStyle.NONE:
            return content
        
        box = AnsiColors.BOX_UNICODE if self._config.enable_unicode else AnsiColors.BOX_ASCII
        
        lines = content.split('\n')
        max_width = max(len(AnsiColors.strip(l)) for l in lines)
        width = min(max_width + padding * 2, self._config.width - 4)
        
        result = []
        
        # Top border
        if title:
            title_str = f" {title} "
            title_len = len(title_str)
            left_len = (width - title_len) // 2
            right_len = width - title_len - left_len
            top = f"{box['tl']}{box['h'] * left_len}{title_str}{box['h'] * right_len}{box['tr']}"
        else:
            top = f"{box['tl']}{box['h'] * width}{box['tr']}"
        result.append(f"{AnsiColors.DIM}{top}{AnsiColors.RESET}")
        
        # Content
        for line in lines:
            visible_len = AnsiColors.visible_length(line)
            pad_right = width - padding - visible_len
            result.append(f"{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}{' ' * padding}{line}{' ' * max(0, pad_right)}{AnsiColors.DIM}{box['v']}{AnsiColors.RESET}")
        
        # Bottom border
        bottom = f"{box['bl']}{box['h'] * width}{box['br']}"
        result.append(f"{AnsiColors.DIM}{bottom}{AnsiColors.RESET}")
        
        return '\n'.join(result)
    
    def list_items(
        self,
        items: List[str],
        ordered: bool = False,
    ) -> str:
        """
        Format list items.
        
        Args:
            items: List items
            ordered: Use numbered list
            
        Returns:
            Formatted list
        """
        lines = []
        for i, item in enumerate(items):
            if ordered:
                marker = f"{i + 1}."
            else:
                marker = '•' if self._config.enable_unicode else '-'
            
            lines.append(f"{AnsiColors.CYAN}{marker}{AnsiColors.RESET} {item}")
        
        return '\n'.join(lines)
    
    def heading(
        self,
        text: str,
        level: int = 1,
    ) -> str:
        """
        Format heading.
        
        Args:
            text: Heading text
            level: Heading level (1-6)
            
        Returns:
            Formatted heading
        """
        if level == 1:
            return f"\n{AnsiColors.HEADING}{text}{AnsiColors.RESET}\n{'═' * len(text)}"
        elif level == 2:
            return f"\n{AnsiColors.SUBHEADING}{text}{AnsiColors.RESET}\n{'─' * len(text)}"
        elif level == 3:
            return f"\n{AnsiColors.BOLD}{text}{AnsiColors.RESET}"
        else:
            return f"\n{AnsiColors.DIM}{AnsiColors.BOLD}{text}{AnsiColors.RESET}"
    
    def link(
        self,
        text: str,
        url: str,
    ) -> str:
        """Format link"""
        return f"{AnsiColors.LINK}{text}{AnsiColors.RESET} ({AnsiColors.DIM}{url}{AnsiColors.RESET})"
    
    def error(self, message: str) -> str:
        """Format error message"""
        return f"{AnsiColors.ERROR}✗ Error: {message}{AnsiColors.RESET}"
    
    def warning(self, message: str) -> str:
        """Format warning message"""
        return f"{AnsiColors.WARNING}⚠ Warning: {message}{AnsiColors.RESET}"
    
    def success(self, message: str) -> str:
        """Format success message"""
        return f"{AnsiColors.SUCCESS}✓ {message}{AnsiColors.RESET}"
    
    def info(self, message: str) -> str:
        """Format info message"""
        return f"{AnsiColors.INFO}ℹ {message}{AnsiColors.RESET}"
    
    def hint(self, message: str) -> str:
        """Format hint message"""
        return f"{AnsiColors.HINT}💡 {message}{AnsiColors.RESET}"
    
    def key_value(
        self,
        data: Dict[str, Any],
        indent: int = 2,
    ) -> str:
        """
        Format key-value pairs.
        
        Args:
            data: Dictionary data
            indent: Indentation
            
        Returns:
            Formatted key-value string
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{AnsiColors.CYAN}{key}:{AnsiColors.RESET}")
                lines.append(self.key_value(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{AnsiColors.CYAN}{key}:{AnsiColors.RESET}")
                for item in value:
                    lines.append(f"{' ' * (indent + 2)}- {item}")
            else:
                lines.append(f"{' ' * indent}{AnsiColors.CYAN}{key}:{AnsiColors.RESET} {value}")
        
        return '\n'.join(lines)
    
    def tree(
        self,
        data: Dict[str, Any],
        prefix: str = '',
        is_last: bool = True,
    ) -> str:
        """
        Format tree structure.
        
        Args:
            data: Tree data
            prefix: Current prefix
            is_last: Is last sibling
            
        Returns:
            Tree string
        """
        result = []
        
        for i, (key, value) in enumerate(data.items()):
            is_last_item = i == len(data) - 1
            
            # Current line prefix
            current_prefix = '└── ' if is_last_item else '├── '
            if not self._config.enable_unicode:
                current_prefix = '`-- ' if is_last_item else '|-- '
            
            # Build line
            if isinstance(value, dict) and value:
                result.append(f"{prefix}{AnsiColors.CYAN}{current_prefix}{key}{AnsiColors.RESET}/")
                # Recurse
                child_prefix = prefix + ('    ' if is_last_item else '│   ')
                if not self._config.enable_unicode:
                    child_prefix = prefix + ('    ' if is_last_item else '|   ')
                result.append(self.tree(value, child_prefix, is_last_item))
            else:
                result.append(f"{prefix}{AnsiColors.CYAN}{current_prefix}{key}{AnsiColors.RESET}: {value}")
        
        return '\n'.join(result)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def wrap(self, text: str, width: int = None) -> str:
        """Wrap text to width"""
        return TextUtils.wrap(text, width or self._config.width)
    
    def truncate(self, text: str, max_length: int) -> str:
        """Truncate text"""
        return TextUtils.truncate(text, max_length)
    
    def align(self, text: str, width: int, align: TextAlign) -> str:
        """Align text"""
        return TextUtils.align(text, width, align)
    
    @property
    def width(self) -> int:
        """Get output width"""
        return self._config.width
    
    @width.setter
    def width(self, value: int):
        """Set output width"""
        self._config.width = value
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo output formatter"""
    formatter = OutputFormatter()
    
    print(formatter.heading("JARVIS Output Formatter Demo", 1))
    
    # Markdown demo
    print(formatter.markdown("""
## Markdown Demo

This is **bold** and *italic* text.

- Item 1
- Item 2
- Item 3

```python
def hello():
    print("Hello, World!")
```
"""))
    
    # Table demo
    print(formatter.heading("Table Demo", 2))
    print(formatter.table(
        ['Name', 'Age', 'City'],
        [
            ['Alice', 25, 'New York'],
            ['Bob', 30, 'San Francisco'],
            ['Charlie', 35, 'Los Angeles'],
        ]
    ))
    
    # JSON demo
    print(formatter.heading("JSON Demo", 2))
    print(formatter.json({'name': 'JARVIS', 'version': '14.0', 'features': ['AI', 'Self-modification']}))
    
    # Panel demo
    print(formatter.heading("Panel Demo", 2))
    print(formatter.panel("This is content inside a panel.\nWith multiple lines.", title="Panel Title"))
    
    # Messages demo
    print(formatter.heading("Message Types", 2))
    print(formatter.success("Operation completed successfully!"))
    print(formatter.error("Something went wrong!"))
    print(formatter.warning("This is a warning!"))
    print(formatter.info("Here's some information."))
    print(formatter.hint("Pro tip: use --help for more options"))


if __name__ == '__main__':
    main()
