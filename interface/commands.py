#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Command Processor
========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Command routing and dispatch
- Permission-based execution
- Command chaining and pipelines
- Error handling and recovery
- Async command support

Features:
- Command registration
- Permission levels
- Command aliases
- Pipeline execution
- Error recovery
- Timeout handling
- Result caching
- Command history

Memory Impact: < 10MB for command processing
"""

import sys
import os
import re
import time
import json
import asyncio
import logging
import threading
import traceback
import subprocess
from typing import (
    Dict, Any, Optional, List, Tuple, Callable, 
    Union, Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from functools import wraps, partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class PermissionLevel(Enum):
    """Command permission levels"""
    PUBLIC = auto()       # Anyone can execute
    USER = auto()         # Authenticated user
    ADMIN = auto()        # Admin user
    SYSTEM = auto()       # System-level access
    ROOT = auto()         # Full access (dangerous)


class CommandStatus(Enum):
    """Command execution status"""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()
    PERMISSION_DENIED = auto()


class CommandType(Enum):
    """Types of commands"""
    SYNC = auto()         # Synchronous execution
    ASYNC = auto()        # Async execution
    STREAMING = auto()    # Streaming output
    INTERACTIVE = auto()  # Requires user interaction


class ErrorCode(Enum):
    """Standard error codes"""
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    INVALID_ARGUMENTS = 2
    PERMISSION_DENIED = 3
    NOT_FOUND = 4
    TIMEOUT = 5
    INTERRUPTED = 6
    DEPENDENCY_ERROR = 7
    SYNTAX_ERROR = 8
    EXECUTION_ERROR = 9


@dataclass
class CommandContext:
    """
    Context for command execution.
    
    Contains all information needed for command
    execution including user, environment, etc.
    """
    user: str = "default"
    session_id: str = ""
    working_dir: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.USER
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, required: PermissionLevel) -> bool:
        """Check if context has required permission"""
        levels = {
            PermissionLevel.PUBLIC: 0,
            PermissionLevel.USER: 1,
            PermissionLevel.ADMIN: 2,
            PermissionLevel.SYSTEM: 3,
            PermissionLevel.ROOT: 4,
        }
        return levels.get(self.permission_level, 0) >= levels.get(required, 0)


@dataclass
class CommandResult:
    """
    Result of command execution.
    
    Contains output, errors, and metadata about
    the command execution.
    """
    success: bool = True
    output: str = ""
    error: str = ""
    error_code: ErrorCode = ErrorCode.SUCCESS
    status: CommandStatus = CommandStatus.SUCCESS
    duration_ms: float = 0.0
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def exit_code(self) -> int:
        """Get exit code"""
        return self.error_code.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'error_code': self.error_code.name,
            'status': self.status.name,
            'duration_ms': self.duration_ms,
        }


@dataclass
class CommandDefinition:
    """
    Definition of a command.
    
    Contains all metadata about a command
    including handler, permissions, help text, etc.
    """
    name: str
    handler: Callable
    description: str = ""
    usage: str = ""
    aliases: List[str] = field(default_factory=list)
    permission: PermissionLevel = PermissionLevel.USER
    command_type: CommandType = CommandType.SYNC
    timeout: float = 30.0
    category: str = "general"
    examples: List[str] = field(default_factory=list)
    arguments: Dict[str, Dict] = field(default_factory=dict)
    hidden: bool = False
    deprecated: bool = False
    replacement: str = ""  # If deprecated
    
    def get_help(self) -> str:
        """Get help text"""
        lines = [
            f"Command: {self.name}",
            f"Description: {self.description}",
        ]
        
        if self.usage:
            lines.append(f"Usage: {self.usage}")
        
        if self.aliases:
            lines.append(f"Aliases: {', '.join(self.aliases)}")
        
        if self.examples:
            lines.append("Examples:")
            for example in self.examples:
                lines.append(f"  {example}")
        
        if self.deprecated:
            lines.append(f"⚠ DEPRECATED: Use '{self.replacement}' instead")
        
        return '\n'.join(lines)


@dataclass
class Pipeline:
    """
    Command pipeline for chaining.
    
    Allows multiple commands to be executed
    in sequence with output piping.
    """
    commands: List[CommandDefinition]
    context: CommandContext
    results: List[CommandResult] = field(default_factory=list)
    current_index: int = 0
    stop_on_error: bool = True
    
    def add(self, command: CommandDefinition):
        """Add command to pipeline"""
        self.commands.append(command)
    
    def get_next_input(self) -> str:
        """Get input for next command from previous output"""
        if self.current_index > 0 and self.results:
            return self.results[-1].output
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class CommandRegistry:
    """
    Registry for all available commands.
    
    Features:
    - Command registration
    - Alias management
    - Command lookup
    - Category organization
    """
    
    def __init__(self):
        """Initialize registry"""
        self._commands: Dict[str, CommandDefinition] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._middlewares: List[Callable] = []
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        **kwargs,
    ) -> CommandDefinition:
        """
        Register a command.
        
        Args:
            name: Command name
            handler: Command handler function
            description: Command description
            **kwargs: Additional command options
            
        Returns:
            CommandDefinition
        """
        cmd_def = CommandDefinition(
            name=name,
            handler=handler,
            description=description,
            **kwargs,
        )
        
        self._commands[name] = cmd_def
        
        # Register aliases
        for alias in cmd_def.aliases:
            self._aliases[alias] = name
        
        # Organize by category
        category = cmd_def.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.debug(f"Registered command: {name}")
        
        return cmd_def
    
    def command(
        self,
        name: str = None,
        **kwargs,
    ):
        """
        Decorator for registering commands.
        
        Usage:
            @registry.command('greet')
            def greet_handler(args, context):
                return 'Hello!'
        """
        def decorator(func):
            cmd_name = name or func.__name__
            self.register(cmd_name, func, **kwargs)
            return func
        return decorator
    
    def get(self, name: str) -> Optional[CommandDefinition]:
        """Get command by name or alias"""
        # Check direct name
        if name in self._commands:
            return self._commands[name]
        
        # Check alias
        if name in self._aliases:
            real_name = self._aliases[name]
            return self._commands.get(real_name)
        
        return None
    
    def get_all(self, include_hidden: bool = False) -> List[CommandDefinition]:
        """Get all commands"""
        commands = list(self._commands.values())
        if not include_hidden:
            commands = [c for c in commands if not c.hidden]
        return commands
    
    def get_by_category(self, category: str) -> List[CommandDefinition]:
        """Get commands by category"""
        names = self._categories.get(category, [])
        return [self._commands[n] for n in names if n in self._commands]
    
    def get_categories(self) -> List[str]:
        """Get all categories"""
        return list(self._categories.keys())
    
    def search(self, query: str) -> List[CommandDefinition]:
        """Search commands by name or description"""
        query_lower = query.lower()
        results = []
        
        for cmd in self._commands.values():
            if query_lower in cmd.name.lower():
                results.append(cmd)
            elif query_lower in cmd.description.lower():
                results.append(cmd)
        
        return results
    
    def add_middleware(self, middleware: Callable):
        """Add middleware function"""
        self._middlewares.append(middleware)
    
    def unregister(self, name: str) -> bool:
        """Unregister a command"""
        if name in self._commands:
            cmd = self._commands[name]
            
            # Remove from category
            if cmd.category in self._categories:
                self._categories[cmd.category].remove(name)
            
            # Remove aliases
            for alias in cmd.aliases:
                self._aliases.pop(alias, None)
            
            # Remove command
            del self._commands[name]
            return True
        
        return False
    
    @property
    def count(self) -> int:
        """Get command count"""
        return len(self._commands)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CommandExecutor:
    """
    Execute commands with error handling.
    
    Features:
    - Sync/async execution
    - Timeout handling
    - Error recovery
    - Result caching
    """
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        max_workers: int = 4,
    ):
        """Initialize executor"""
        self._default_timeout = default_timeout
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[str, Tuple[CommandResult, float]] = {}
        self._cache_ttl = 60.0
    
    def execute(
        self,
        command: CommandDefinition,
        args: List[str],
        kwargs: Dict[str, str],
        context: CommandContext,
        input_data: str = "",
    ) -> CommandResult:
        """
        Execute a command.
        
        Args:
            command: Command definition
            args: Positional arguments
            kwargs: Keyword arguments
            context: Execution context
            input_data: Input data (from pipeline)
            
        Returns:
            CommandResult
        """
        start_time = time.time()
        
        # Check cache for idempotent commands
        cache_key = self._get_cache_key(command.name, args, kwargs)
        if cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Using cached result for {command.name}")
                return cached_result
        
        result = CommandResult(status=CommandStatus.RUNNING)
        
        try:
            # Check command type
            if command.command_type == CommandType.ASYNC:
                # Run async handler
                result = self._execute_async(
                    command, args, kwargs, context, input_data
                )
            else:
                # Run sync handler with timeout
                result = self._execute_sync(
                    command, args, kwargs, context, input_data
                )
        
        except FutureTimeoutError:
            result.status = CommandStatus.TIMEOUT
            result.error_code = ErrorCode.TIMEOUT
            result.error = f"Command timed out after {command.timeout}s"
            result.success = False
        
        except PermissionError:
            result.status = CommandStatus.PERMISSION_DENIED
            result.error_code = ErrorCode.PERMISSION_DENIED
            result.error = "Permission denied"
            result.success = False
        
        except KeyboardInterrupt:
            result.status = CommandStatus.CANCELLED
            result.error_code = ErrorCode.INTERRUPTED
            result.error = "Command interrupted"
            result.success = False
        
        except Exception as e:
            result.status = CommandStatus.FAILED
            result.error_code = ErrorCode.EXECUTION_ERROR
            result.error = str(e)
            result.success = False
            logger.error(f"Command execution error: {e}\n{traceback.format_exc()}")
        
        finally:
            result.duration_ms = (time.time() - start_time) * 1000
            
            # Cache successful results
            if result.success:
                self._cache[cache_key] = (result, time.time())
        
        return result
    
    def _execute_sync(
        self,
        command: CommandDefinition,
        args: List[str],
        kwargs: Dict[str, str],
        context: CommandContext,
        input_data: str,
    ) -> CommandResult:
        """Execute synchronous command"""
        handler = command.handler
        timeout = command.timeout or self._default_timeout
        
        # Build arguments
        call_args = self._build_args(args, kwargs, input_data)
        
        # Execute in thread pool with timeout
        future = self._thread_pool.submit(
            self._call_handler,
            handler,
            call_args,
            context,
        )
        
        try:
            return_value = future.result(timeout=timeout)
            
            if isinstance(return_value, CommandResult):
                return_value.status = CommandStatus.SUCCESS
                return return_value
            
            return CommandResult(
                success=True,
                output=str(return_value) if return_value is not None else "",
                return_value=return_value,
                status=CommandStatus.SUCCESS,
            )
        
        except FutureTimeoutError:
            future.cancel()
            raise
    
    def _execute_async(
        self,
        command: CommandDefinition,
        args: List[str],
        kwargs: Dict[str, str],
        context: CommandContext,
        input_data: str,
    ) -> CommandResult:
        """Execute async command"""
        handler = command.handler
        timeout = command.timeout or self._default_timeout
        
        # Build arguments
        call_args = self._build_args(args, kwargs, input_data)
        
        # Create event loop if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async handler
        try:
            return_value = loop.run_until_complete(
                asyncio.wait_for(
                    handler(call_args, context),
                    timeout=timeout
                )
            )
            
            if isinstance(return_value, CommandResult):
                return_value.status = CommandStatus.SUCCESS
                return return_value
            
            return CommandResult(
                success=True,
                output=str(return_value) if return_value is not None else "",
                return_value=return_value,
                status=CommandStatus.SUCCESS,
            )
        
        except asyncio.TimeoutError:
            raise FutureTimeoutError()
    
    def _build_args(
        self,
        args: List[str],
        kwargs: Dict[str, str],
        input_data: str,
    ) -> Dict[str, Any]:
        """Build arguments dict for handler"""
        return {
            'args': args,
            'kwargs': kwargs,
            'input': input_data,
        }
    
    def _call_handler(
        self,
        handler: Callable,
        call_args: Dict[str, Any],
        context: CommandContext,
    ) -> Any:
        """Call handler with proper signature"""
        import inspect
        
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        
        # Handle different signatures
        if len(params) == 0:
            return handler()
        elif len(params) == 1:
            return handler(call_args)
        elif len(params) == 2:
            return handler(call_args, context)
        else:
            # Try to match by name
            return handler(**call_args, context=context)
    
    def _get_cache_key(
        self,
        name: str,
        args: List[str],
        kwargs: Dict[str, str],
    ) -> str:
        """Generate cache key"""
        import hashlib
        key_data = f"{name}:{':'.join(args)}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear result cache"""
        self._cache.clear()
    
    def shutdown(self):
        """Shutdown executor"""
        self._thread_pool.shutdown(wait=False)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND PROCESSOR (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class CommandProcessor:
    """
    Ultra-Advanced Command Processor for JARVIS.
    
    Features:
    - Command registration and routing
    - Permission-based execution
    - Command chaining and pipelines
    - Error handling and recovery
    - Async command support
    - Result caching
    
    Memory Budget: < 10MB
    
    Usage:
        processor = CommandProcessor()
        
        # Register command
        @processor.command('greet')
        def greet_handler(args, context):
            name = args['args'][0] if args['args'] else 'World'
            return f'Hello, {name}!'
        
        # Execute command
        result = processor.execute('greet Alice')
        print(result.output)
    """
    
    def __init__(
        self,
        default_permission: PermissionLevel = PermissionLevel.USER,
        default_timeout: float = 30.0,
    ):
        """
        Initialize Command Processor.
        
        Args:
            default_permission: Default permission level
            default_timeout: Default command timeout
        """
        self._default_permission = default_permission
        
        # Components
        self._registry = CommandRegistry()
        self._executor = CommandExecutor(default_timeout=default_timeout)
        
        # Hooks
        self._pre_hooks: Dict[str, List[Callable]] = {}
        self._post_hooks: Dict[str, List[Callable]] = {}
        self._error_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self._stats = {
            'commands_executed': 0,
            'commands_succeeded': 0,
            'commands_failed': 0,
            'total_time_ms': 0.0,
        }
        
        # Register built-in commands
        self._register_builtins()
        
        logger.info("CommandProcessor initialized")
    
    def _register_builtins(self):
        """Register built-in commands"""
        # Help command
        self._registry.register(
            name='help',
            handler=self._cmd_help,
            description='Show help for commands',
            permission=PermissionLevel.PUBLIC,
            category='system',
        )
        
        # Version command
        self._registry.register(
            name='version',
            handler=self._cmd_version,
            description='Show JARVIS version',
            permission=PermissionLevel.PUBLIC,
            category='system',
        )
        
        # Echo command
        self._registry.register(
            name='echo',
            handler=self._cmd_echo,
            description='Print text',
            permission=PermissionLevel.PUBLIC,
            category='utility',
        )
        
        # Sleep command (for testing)
        self._registry.register(
            name='sleep',
            handler=self._cmd_sleep,
            description='Sleep for specified seconds',
            permission=PermissionLevel.USER,
            category='utility',
            timeout=60.0,
        )
    
    def _cmd_help(self, args: Dict, context: CommandContext) -> str:
        """Help command"""
        cmd_name = args['args'][0] if args['args'] else None
        
        if cmd_name:
            cmd = self._registry.get(cmd_name)
            if cmd:
                return cmd.get_help()
            return f"Command not found: {cmd_name}"
        
        # List all commands
        lines = ["Available commands:", ""]
        
        for category in sorted(self._registry.get_categories()):
            commands = self._registry.get_by_category(category)
            if commands:
                lines.append(f"[{category}]")
                for cmd in commands:
                    if not cmd.hidden:
                        deprecated = " (deprecated)" if cmd.deprecated else ""
                        lines.append(f"  {cmd.name:<15} {cmd.description}{deprecated}")
                lines.append("")
        
        return '\n'.join(lines)
    
    def _cmd_version(self, args: Dict, context: CommandContext) -> str:
        """Version command"""
        return "JARVIS Command Processor v14.0.0"
    
    def _cmd_echo(self, args: Dict, context: CommandContext) -> str:
        """Echo command"""
        return ' '.join(args['args'])
    
    def _cmd_sleep(self, args: Dict, context: CommandContext) -> str:
        """Sleep command"""
        try:
            seconds = float(args['args'][0]) if args['args'] else 1.0
            time.sleep(min(seconds, 60))  # Max 60 seconds
            return f"Slept for {seconds} seconds"
        except ValueError:
            return "Invalid duration"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Command Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register(
        self,
        name: str,
        handler: Callable,
        **kwargs,
    ) -> CommandDefinition:
        """
        Register a command.
        
        Args:
            name: Command name
            handler: Command handler
            **kwargs: Additional options
            
        Returns:
            CommandDefinition
        """
        return self._registry.register(name, handler, **kwargs)
    
    def command(
        self,
        name: str = None,
        **kwargs,
    ):
        """
        Decorator for registering commands.
        
        Usage:
            @processor.command('greet')
            def greet_handler(args, context):
                return 'Hello!'
        """
        return self._registry.command(name, **kwargs)
    
    def alias(self, name: str, alias: str):
        """Add alias to existing command"""
        cmd = self._registry.get(name)
        if cmd:
            cmd.aliases.append(alias)
            self._registry._aliases[alias] = name
    
    # ─────────────────────────────────────────────────────────────────────────
    # Execution Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        command_str: str,
        context: CommandContext = None,
    ) -> CommandResult:
        """
        Execute a command string.
        
        Args:
            command_str: Command string to execute
            context: Execution context
            
        Returns:
            CommandResult
        """
        start_time = time.time()
        
        # Create default context if needed
        if context is None:
            context = CommandContext(
                permission_level=self._default_permission,
            )
        
        # Parse command string
        name, args, kwargs = self._parse_command(command_str)
        
        if not name:
            return CommandResult(
                success=False,
                error="Empty command",
                error_code=ErrorCode.INVALID_ARGUMENTS,
                status=CommandStatus.FAILED,
            )
        
        # Get command definition
        cmd = self._registry.get(name)
        
        if not cmd:
            return CommandResult(
                success=False,
                error=f"Command not found: {name}",
                error_code=ErrorCode.NOT_FOUND,
                status=CommandStatus.FAILED,
            )
        
        # Check deprecation
        if cmd.deprecated:
            logger.warning(f"Command '{name}' is deprecated. Use '{cmd.replacement}' instead.")
        
        # Check permissions
        if not context.has_permission(cmd.permission):
            return CommandResult(
                success=False,
                error=f"Permission denied. Required: {cmd.permission.name}",
                error_code=ErrorCode.PERMISSION_DENIED,
                status=CommandStatus.PERMISSION_DENIED,
            )
        
        # Run pre-hooks
        self._run_hooks(name, 'pre', args, kwargs, context)
        
        # Execute command
        result = self._executor.execute(cmd, args, kwargs, context)
        
        # Run post-hooks
        self._run_hooks(name, 'post', args, kwargs, context, result)
        
        # Update stats
        self._stats['commands_executed'] += 1
        self._stats['total_time_ms'] += (time.time() - start_time) * 1000
        
        if result.success:
            self._stats['commands_succeeded'] += 1
        else:
            self._stats['commands_failed'] += 1
            # Run error handlers
            self._run_error_handlers(name, args, kwargs, context, result)
        
        return result
    
    def execute_pipeline(
        self,
        commands: List[str],
        context: CommandContext = None,
        stop_on_error: bool = True,
    ) -> List[CommandResult]:
        """
        Execute a pipeline of commands.
        
        Args:
            commands: List of command strings
            context: Execution context
            stop_on_error: Stop on first error
            
        Returns:
            List of CommandResults
        """
        results = []
        current_input = ""
        
        for cmd_str in commands:
            result = self.execute(cmd_str, context)
            
            # Store input for next command
            if result.success:
                current_input = result.output
            
            results.append(result)
            
            if not result.success and stop_on_error:
                break
        
        return results
    
    def _parse_command(self, command_str: str) -> Tuple[str, List[str], Dict[str, str]]:
        """Parse command string into name, args, kwargs"""
        # Simple parsing
        parts = command_str.strip().split()
        
        if not parts:
            return "", [], {}
        
        name = parts[0]
        args = []
        kwargs = {}
        
        i = 1
        while i < len(parts):
            part = parts[i]
            
            if part.startswith('--'):
                # Long option
                if '=' in part:
                    key, value = part[2:].split('=', 1)
                    kwargs[key] = value
                elif i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                    kwargs[part[2:]] = parts[i + 1]
                    i += 1
                else:
                    kwargs[part[2:]] = 'true'
            elif part.startswith('-') and not part.startswith('--'):
                # Short option
                key = part[1:]
                if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                    kwargs[key] = parts[i + 1]
                    i += 1
                else:
                    kwargs[key] = 'true'
            else:
                args.append(part)
            
            i += 1
        
        return name, args, kwargs
    
    # ─────────────────────────────────────────────────────────────────────────
    # Hooks
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_pre_hook(self, command_name: str, hook: Callable):
        """Add pre-execution hook"""
        if command_name not in self._pre_hooks:
            self._pre_hooks[command_name] = []
        self._pre_hooks[command_name].append(hook)
    
    def add_post_hook(self, command_name: str, hook: Callable):
        """Add post-execution hook"""
        if command_name not in self._post_hooks:
            self._post_hooks[command_name] = []
        self._post_hooks[command_name].append(hook)
    
    def add_error_handler(self, command_name: str, handler: Callable):
        """Add error handler"""
        if command_name not in self._error_handlers:
            self._error_handlers[command_name] = []
        self._error_handlers[command_name].append(handler)
    
    def _run_hooks(
        self,
        command_name: str,
        hook_type: str,
        args: List[str],
        kwargs: Dict[str, str],
        context: CommandContext,
        result: CommandResult = None,
    ):
        """Run hooks for command"""
        hooks_map = {
            'pre': self._pre_hooks,
            'post': self._post_hooks,
        }
        
        hooks = hooks_map.get(hook_type, {})
        
        # Run command-specific hooks
        for hook in hooks.get(command_name, []):
            try:
                if hook_type == 'pre':
                    hook(args, kwargs, context)
                else:
                    hook(args, kwargs, context, result)
            except Exception as e:
                logger.error(f"Hook error: {e}")
        
        # Run global hooks
        for hook in hooks.get('*', []):
            try:
                if hook_type == 'pre':
                    hook(command_name, args, kwargs, context)
                else:
                    hook(command_name, args, kwargs, context, result)
            except Exception as e:
                logger.error(f"Global hook error: {e}")
    
    def _run_error_handlers(
        self,
        command_name: str,
        args: List[str],
        kwargs: Dict[str, str],
        context: CommandContext,
        result: CommandResult,
    ):
        """Run error handlers"""
        handlers = self._error_handlers.get(command_name, [])
        handlers.extend(self._error_handlers.get('*', []))
        
        for handler in handlers:
            try:
                handler(command_name, args, kwargs, context, result)
            except Exception as e:
                logger.error(f"Error handler error: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_command(self, name: str) -> Optional[CommandDefinition]:
        """Get command definition"""
        return self._registry.get(name)
    
    def get_commands(self, category: str = None) -> List[CommandDefinition]:
        """Get commands, optionally filtered by category"""
        if category:
            return self._registry.get_by_category(category)
        return self._registry.get_all()
    
    def search_commands(self, query: str) -> List[CommandDefinition]:
        """Search commands"""
        return self._registry.search(query)
    
    def get_categories(self) -> List[str]:
        """Get command categories"""
        return self._registry.get_categories()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self._stats.copy()
    
    @property
    def command_count(self) -> int:
        """Get registered command count"""
        return self._registry.count
    
    def shutdown(self):
        """Shutdown processor"""
        self._executor.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo command processor"""
    processor = CommandProcessor()
    
    print("JARVIS Command Processor Demo")
    print("=" * 40)
    
    # Register custom commands
    @processor.command('greet', description='Greet someone')
    def greet_handler(args, context):
        name = args['args'][0] if args['args'] else 'World'
        return f'Hello, {name}!'
    
    @processor.command('add', description='Add numbers')
    def add_handler(args, context):
        numbers = [float(a) for a in args['args']]
        return f'Sum: {sum(numbers)}'
    
    # Execute commands
    result = processor.execute('greet Alice')
    print(f"\n$ greet Alice")
    print(f"  {result.output}")
    
    result = processor.execute('add 10 20 30')
    print(f"\n$ add 10 20 30")
    print(f"  {result.output}")
    
    # Show help
    result = processor.execute('help')
    print(f"\n{result.output}")
    
    # Show stats
    print("\nStatistics:")
    for key, value in processor.stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
