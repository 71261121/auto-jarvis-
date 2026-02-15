#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - User Interface Module
============================================

This module provides the user interface components for JARVIS:

- CLI: Command Line Interface
- Input: Input handling (terminal, file, voice)
- Output: Output formatting (markdown, tables, syntax highlighting)
- Commands: Command processing and routing
- Session: Session management and persistence
- Progress: Progress indicators and spinners
- Notify: Notification system (Termux integration)
- Help: Context-sensitive help system

Memory Budget: < 50MB total for UI
"""

from interface.cli import (
    CLI,
    CLIConfig,
    CommandParser,
    HistoryManager,
    CompletionEngine,
    TerminalDetector,
    Colors,
)

from interface.input import (
    InputHandler,
    InputConfig,
    InputResult,
    InputSource,
    InputType,
    InputSanitizer,
    SanitizationLevel,
    VoiceInputHandler,
    FileInputHandler,
    MultilineInputHandler,
)

from interface.output import (
    OutputFormatter,
    OutputConfig,
    OutputFormat,
    TextAlign,
    BorderStyle,
    OutputTheme,
    SyntaxHighlighter,
    MarkdownRenderer,
    TableFormatter,
    TextUtils,
    AnsiColors,
)

from interface.commands import (
    CommandProcessor,
    CommandRegistry,
    CommandExecutor,
    CommandDefinition,
    CommandContext,
    CommandResult,
    PermissionLevel,
    CommandType,
    ErrorCode,
)

from interface.session import (
    SessionManager,
    SessionConfig,
    SessionData,
    SessionState,
    SessionPriority,
)

from interface.progress import (
    ProgressIndicator,
    ProgressConfig,
    ProgressBar,
    Spinner,
    SpinnerStyle,
    ProgressStyle,
)

from interface.notify import (
    Notifier,
    NotificationConfig,
    Notification,
    NotificationPriority,
    NotificationType,
    TermuxNotifier,
)

from interface.help import (
    HelpSystem,
    HelpTopic,
    HelpCategory,
    HelpFormat,
    Tutorial,
)

__all__ = [
    # CLI
    'CLI',
    'CLIConfig',
    'CommandParser',
    'HistoryManager',
    'CompletionEngine',
    'TerminalDetector',
    'Colors',
    
    # Input
    'InputHandler',
    'InputConfig',
    'InputResult',
    'InputSource',
    'InputType',
    'InputSanitizer',
    'SanitizationLevel',
    'VoiceInputHandler',
    'FileInputHandler',
    'MultilineInputHandler',
    
    # Output
    'OutputFormatter',
    'OutputConfig',
    'OutputFormat',
    'TextAlign',
    'BorderStyle',
    'OutputTheme',
    'SyntaxHighlighter',
    'MarkdownRenderer',
    'TableFormatter',
    'TextUtils',
    'AnsiColors',
    
    # Commands
    'CommandProcessor',
    'CommandRegistry',
    'CommandExecutor',
    'CommandDefinition',
    'CommandContext',
    'CommandResult',
    'PermissionLevel',
    'CommandType',
    'ErrorCode',
    
    # Session
    'SessionManager',
    'SessionConfig',
    'SessionData',
    'SessionState',
    'SessionPriority',
    
    # Progress
    'ProgressIndicator',
    'ProgressConfig',
    'ProgressBar',
    'Spinner',
    'SpinnerStyle',
    'ProgressStyle',
    
    # Notify
    'Notifier',
    'NotificationConfig',
    'Notification',
    'NotificationPriority',
    'NotificationType',
    'TermuxNotifier',
    
    # Help
    'HelpSystem',
    'HelpTopic',
    'HelpCategory',
    'HelpFormat',
    'Tutorial',
]

__version__ = '14.0.0'
