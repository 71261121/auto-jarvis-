#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Configuration Management System
======================================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- 12 Factor App methodology for config management
- Python-dotenv for environment variables
- YAML/JSON fallback chain for config files
- Pydantic-inspired validation (without heavy dependency)

Features:
- Multi-source configuration (env vars, files, defaults)
- Type coercion and validation
- Hot-reload support
- Secure credential handling
- Configuration inheritance and overrides
- Memory-efficient singleton pattern

Memory Impact: < 500KB
Configuration Sources Priority:
1. Environment Variables (highest priority)
2. Config Files (YAML/JSON)
3. Default Values (lowest priority)
"""

import os
import sys
import json
import logging
import threading
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigSource(Enum):
    """Source of configuration value"""
    DEFAULT = auto()
    FILE = auto()
    ENVIRONMENT = auto()
    RUNTIME = auto()


class ConfigStatus(Enum):
    """Status of configuration"""
    NOT_LOADED = auto()
    LOADED = auto()
    ERROR = auto()
    RELOADING = auto()


@dataclass
class ConfigValue:
    """
    A configuration value with metadata.
    
    Tracks the source and history of each configuration value
    for debugging and auditing purposes.
    """
    key: str
    value: Any
    source: ConfigSource
    type_hint: str = "str"
    description: str = ""
    default_value: Any = None
    is_secret: bool = False
    validation_regex: str = ""
    last_modified: float = field(default_factory=time.time)
    modification_count: int = 0
    
    def __str__(self) -> str:
        if self.is_secret:
            return f"{self.key}=***HIDDEN***"
        return f"{self.key}={self.value} ({self.source.name})"


@dataclass
class ConfigSchema:
    """
    Schema definition for a configuration key.
    
    Used for validation and documentation.
    """
    key: str
    type: str  # 'str', 'int', 'float', 'bool', 'list', 'dict'
    default: Any = None
    required: bool = False
    description: str = ""
    env_var: str = ""
    is_secret: bool = False
    validation_pattern: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: List[Any] = field(default_factory=list)
    
    def validate(self, value: Any) -> tuple:
        """
        Validate a value against this schema.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if self.required and value is None:
            return False, f"Required configuration '{self.key}' is missing"
        
        if value is None:
            return True, None
        
        # Type checking
        type_mapping = {
            'str': str,
            'int': int,
            'float': (int, float),
            'bool': bool,
            'list': list,
            'dict': dict,
        }
        
        expected_type = type_mapping.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            # Try type coercion
            try:
                if self.type == 'int':
                    value = int(value)
                elif self.type == 'float':
                    value = float(value)
                elif self.type == 'bool':
                    if isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = bool(value)
            except (ValueError, TypeError):
                return False, f"'{self.key}' must be {self.type}, got {type(value).__name__}"
        
        # Range checking
        if self.type in ('int', 'float'):
            if self.min_value is not None and value < self.min_value:
                return False, f"'{self.key}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"'{self.key}' must be <= {self.max_value}"
        
        # Allowed values
        if self.allowed_values and value not in self.allowed_values:
            return False, f"'{self.key}' must be one of {self.allowed_values}"
        
        return True, None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION SCHEMAS (Device-Specific)
# ═══════════════════════════════════════════════════════════════════════════════

JARVIS_CONFIG_SCHEMAS = [
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="app_name",
        type="str",
        default="JARVIS v14 Ultimate",
        description="Application name",
    ),
    ConfigSchema(
        key="app_version",
        type="str",
        default="14.0.0",
        description="Application version",
    ),
    ConfigSchema(
        key="device_model",
        type="str",
        default="Realme 2 Pro Lite (RMP2402)",
        description="Device model identifier",
    ),
    ConfigSchema(
        key="device_ram_mb",
        type="int",
        default=4096,
        description="Device RAM in MB",
        min_value=512,
        max_value=16384,
    ),
    ConfigSchema(
        key="platform",
        type="str",
        default="termux",
        description="Platform identifier",
        allowed_values=["termux", "linux", "windows", "macos", "android"],
    ),
    ConfigSchema(
        key="debug_mode",
        type="bool",
        default=False,
        description="Enable debug mode",
        env_var="JARVIS_DEBUG",
    ),
    ConfigSchema(
        key="log_level",
        type="str",
        default="INFO",
        description="Logging level",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        env_var="JARVIS_LOG_LEVEL",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AI PROVIDER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="openrouter_api_key",
        type="str",
        default="",
        description="OpenRouter API key for AI access",
        env_var="OPENROUTER_API_KEY",
        is_secret=True,
        required=True,
    ),
    ConfigSchema(
        key="default_ai_model",
        type="str",
        default="openrouter/free",
        description="Default AI model to use",
    ),
    ConfigSchema(
        key="ai_timeout_seconds",
        type="int",
        default=120,
        description="Timeout for AI requests",
        min_value=10,
        max_value=600,
    ),
    ConfigSchema(
        key="ai_max_retries",
        type="int",
        default=3,
        description="Maximum retry attempts for AI requests",
        min_value=0,
        max_value=10,
    ),
    ConfigSchema(
        key="ai_temperature",
        type="float",
        default=0.7,
        description="AI response randomness",
        min_value=0.0,
        max_value=2.0,
    ),
    ConfigSchema(
        key="ai_max_tokens",
        type="int",
        default=4096,
        description="Maximum tokens in AI response",
        min_value=100,
        max_value=128000,
    ),
    ConfigSchema(
        key="enable_ai_cache",
        type="bool",
        default=True,
        description="Enable AI response caching",
    ),
    ConfigSchema(
        key="ai_cache_ttl_seconds",
        type="int",
        default=3600,
        description="AI cache TTL in seconds",
        min_value=60,
        max_value=86400,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY SYSTEM CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="memory_db_path",
        type="str",
        default="~/.jarvis/memory.db",
        description="Path to memory database",
    ),
    ConfigSchema(
        key="memory_max_entries",
        type="int",
        default=100000,
        description="Maximum entries in memory",
        min_value=1000,
        max_value=1000000,
    ),
    ConfigSchema(
        key="memory_cleanup_interval_hours",
        type="int",
        default=24,
        description="Memory cleanup interval in hours",
        min_value=1,
        max_value=168,
    ),
    ConfigSchema(
        key="enable_memory_compression",
        type="bool",
        default=True,
        description="Enable memory compression for storage",
    ),
    ConfigSchema(
        key="conversation_max_history",
        type="int",
        default=100,
        description="Maximum conversation history per chat",
        min_value=10,
        max_value=1000,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SELF-MODIFICATION CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="enable_self_modification",
        type="bool",
        default=True,
        description="Enable self-modification capabilities",
        env_var="JARVIS_SELF_MOD",
    ),
    ConfigSchema(
        key="self_mod_backup_dir",
        type="str",
        default="~/.jarvis/backups",
        description="Directory for code backups",
    ),
    ConfigSchema(
        key="self_mod_max_backups",
        type="int",
        default=50,
        description="Maximum backup files to keep",
        min_value=5,
        max_value=200,
    ),
    ConfigSchema(
        key="self_mod_require_approval",
        type="bool",
        default=True,
        description="Require user approval for modifications",
    ),
    ConfigSchema(
        key="self_mod_sandbox_mode",
        type="bool",
        default=True,
        description="Run modifications in sandbox first",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="max_memory_usage_mb",
        type="int",
        default=512,
        description="Maximum memory usage in MB",
        min_value=128,
        max_value=2048,
    ),
    ConfigSchema(
        key="gc_threshold_percent",
        type="int",
        default=80,
        description="Garbage collection trigger threshold",
        min_value=50,
        max_value=95,
    ),
    ConfigSchema(
        key="thread_pool_size",
        type="int",
        default=4,
        description="Thread pool size for concurrent operations",
        min_value=1,
        max_value=16,
    ),
    ConfigSchema(
        key="enable_lazy_loading",
        type="bool",
        default=True,
        description="Enable lazy loading for memory efficiency",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GITHUB INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="github_token",
        type="str",
        default="",
        description="GitHub personal access token",
        env_var="GITHUB_TOKEN",
        is_secret=True,
    ),
    ConfigSchema(
        key="github_repo",
        type="str",
        default="",
        description="GitHub repository (owner/repo format)",
    ),
    ConfigSchema(
        key="github_auto_sync",
        type="bool",
        default=False,
        description="Automatically sync changes to GitHub",
    ),
    ConfigSchema(
        key="github_sync_interval_minutes",
        type="int",
        default=30,
        description="GitHub sync interval in minutes",
        min_value=5,
        max_value=1440,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RATE LIMITING
    # ═══════════════════════════════════════════════════════════════════════════
    ConfigSchema(
        key="rate_limit_requests_per_minute",
        type="int",
        default=60,
        description="API rate limit per minute",
        min_value=10,
        max_value=1000,
    ),
    ConfigSchema(
        key="rate_limit_burst_size",
        type="int",
        default=10,
        description="Burst size for rate limiting",
        min_value=1,
        max_value=50,
    ),
    ConfigSchema(
        key="rate_limit_backoff_factor",
        type="float",
        default=2.0,
        description="Exponential backoff factor",
        min_value=1.0,
        max_value=5.0,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """
    Ultra-Advanced Configuration Management System.
    
    Design Goals:
    1. Never crash due to missing config - always have defaults
    2. Multiple configuration sources with priority
    3. Type-safe value access with coercion
    4. Hot-reload support for runtime updates
    5. Secure handling of secrets
    6. Memory-efficient for 4GB RAM device
    7. Full validation and error reporting
    
    Configuration Priority (highest to lowest):
    1. Runtime overrides (set() method)
    2. Environment variables
    3. Configuration files (YAML preferred, JSON fallback)
    4. Default values from schema
    
    Memory Budget: < 1MB for entire configuration
    
    Usage:
        # Initialize
        config = ConfigManager()
        config.load()
        
        # Simple access
        api_key = config.get('openrouter_api_key')
        timeout = config.get_int('ai_timeout_seconds', default=60)
        
        # With validation
        config.require('openrouter_api_key')  # Raises if missing
        
        # Runtime override
        config.set('debug_mode', True)
        
        # Check config status
        if config.is_loaded():
            print("Configuration loaded successfully")
        
        # Export for debugging
        export = config.export_safe()  # Hides secrets
    """
    
    DEFAULT_CONFIG_DIR = "~/.jarvis"
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    def __init__(
        self,
        config_dir: str = None,
        config_file: str = None,
        schemas: List[ConfigSchema] = None,
        auto_load: bool = True,
        enable_hot_reload: bool = False,
        hot_reload_interval: int = 60,
    ):
        """
        Initialize Configuration Manager.
        
        Args:
            config_dir: Directory for configuration files
            config_file: Configuration file name
            schemas: List of configuration schemas
            auto_load: Automatically load configuration on init
            enable_hot_reload: Enable hot-reload for config files
            hot_reload_interval: Interval for hot-reload checks in seconds
        """
        self._config_dir = Path(config_dir or self.DEFAULT_CONFIG_DIR).expanduser()
        self._config_file = config_file or self.DEFAULT_CONFIG_FILE
        self._schemas: Dict[str, ConfigSchema] = {}
        self._values: Dict[str, ConfigValue] = {}
        self._env_prefix = "JARVIS_"
        self._status = ConfigStatus.NOT_LOADED
        self._load_time: Optional[float] = None
        self._file_hash: Optional[str] = None
        
        # Hot reload
        self._enable_hot_reload = enable_hot_reload
        self._hot_reload_interval = hot_reload_interval
        self._last_reload_check: float = 0
        self._reload_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'loads': 0,
            'env_overrides': 0,
            'file_overrides': 0,
            'validation_errors': 0,
            'hot_reloads': 0,
        }
        
        # Initialize schemas
        schemas = schemas or JARVIS_CONFIG_SCHEMAS
        for schema in schemas:
            self._schemas[schema.key] = schema
        
        # Auto load
        if auto_load:
            try:
                self.load()
            except Exception as e:
                logger.warning(f"Auto-load failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CORE METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def load(self, config_path: str = None) -> bool:
        """
        Load configuration from all sources.
        
        Priority:
        1. Load defaults from schema
        2. Load from config file
        3. Apply environment variable overrides
        
        Args:
            config_path: Optional specific config file path
            
        Returns:
            True if loaded successfully
        """
        with self._lock:
            self._status = ConfigStatus.RELOADING
            self._stats['loads'] += 1
            
            try:
                # Step 1: Load defaults
                self._load_defaults()
                
                # Step 2: Load from file
                if config_path:
                    self._load_from_file(config_path)
                else:
                    self._load_from_default_file()
                
                # Step 3: Apply environment variables
                self._apply_env_overrides()
                
                # Step 4: Validate all values
                self._validate_all()
                
                self._status = ConfigStatus.LOADED
                self._load_time = time.time()
                
                logger.info(f"Configuration loaded successfully ({len(self._values)} values)")
                return True
                
            except Exception as e:
                self._status = ConfigStatus.ERROR
                logger.error(f"Configuration load failed: {e}")
                return False
    
    def _load_defaults(self):
        """Load default values from schemas"""
        for key, schema in self._schemas.items():
            if schema.default is not None:
                self._set_value(
                    key=key,
                    value=schema.default,
                    source=ConfigSource.DEFAULT,
                    validate=False
                )
    
    def _load_from_default_file(self):
        """Load configuration from default file location"""
        # Try YAML first, then JSON
        config_path = self._config_dir / self._config_file
        json_path = self._config_dir / "config.json"
        
        if config_path.exists():
            self._load_from_file(str(config_path))
        elif json_path.exists():
            self._load_from_file(str(json_path))
        else:
            logger.debug("No configuration file found, using defaults")
    
    def _load_from_file(self, config_path: str):
        """Load configuration from a specific file"""
        path = Path(config_path).expanduser()
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return
        
        try:
            content = path.read_text(encoding='utf-8')
            self._file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Try YAML first, then JSON
            data = None
            
            if path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    logger.debug("YAML not available, trying JSON parse")
            
            if data is None:
                data = json.loads(content)
            
            if isinstance(data, dict):
                self._apply_file_config(data)
                self._stats['file_overrides'] += len(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _apply_file_config(self, data: Dict[str, Any], prefix: str = ""):
        """Recursively apply configuration from file data"""
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Nested configuration
                self._apply_file_config(value, f"{full_key}.")
            else:
                self._set_value(
                    key=full_key,
                    value=value,
                    source=ConfigSource.FILE,
                    validate=True
                )
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for key, schema in self._schemas.items():
            env_var = schema.env_var or f"{self._env_prefix}{key.upper()}"
            
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type coercion based on schema
                coerced_value = self._coerce_value(value, schema.type)
                
                self._set_value(
                    key=key,
                    value=coerced_value,
                    source=ConfigSource.ENVIRONMENT,
                    validate=True
                )
                self._stats['env_overrides'] += 1
    
    def _coerce_value(self, value: str, type_hint: str) -> Any:
        """Coerce string value to appropriate type"""
        if type_hint == 'int':
            try:
                return int(value)
            except ValueError:
                return value
        
        elif type_hint == 'float':
            try:
                return float(value)
            except ValueError:
                return value
        
        elif type_hint == 'bool':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            return bool(value)
        
        elif type_hint == 'list':
            if isinstance(value, str):
                # Try JSON parse
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Fallback to comma-separated
                    return [v.strip() for v in value.split(',')]
            return value
        
        elif type_hint == 'dict':
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
        
        return value
    
    def _set_value(
        self,
        key: str,
        value: Any,
        source: ConfigSource,
        validate: bool = True
    ) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            source: Source of the value
            validate: Whether to validate the value
            
        Returns:
            True if value was set successfully
        """
        # Get schema
        schema = self._schemas.get(key)
        
        # Validate if schema exists
        if validate and schema:
            is_valid, error = schema.validate(value)
            if not is_valid:
                logger.warning(f"Validation failed for '{key}': {error}")
                self._stats['validation_errors'] += 1
                return False
        
        # Determine type hint
        type_hint = schema.type if schema else type(value).__name__
        is_secret = schema.is_secret if schema else False
        
        # Create config value
        config_value = ConfigValue(
            key=key,
            value=value,
            source=source,
            type_hint=type_hint,
            is_secret=is_secret,
            default_value=schema.default if schema else None,
        )
        
        # Store value
        old_value = self._values.get(key)
        self._values[key] = config_value
        
        # Notify callbacks if value changed
        if old_value and old_value.value != value:
            self._notify_change(key, old_value.value, value)
        
        return True
    
    def _validate_all(self):
        """Validate all configuration values"""
        errors = []
        
        for key, schema in self._schemas.items():
            value = self._values.get(key)
            
            if schema.required and (value is None or value.value is None or value.value == ""):
                errors.append(f"Required configuration '{key}' is missing")
                continue
            
            if value is not None:
                is_valid, error = schema.validate(value.value)
                if not is_valid:
                    errors.append(error)

        
        if errors:
            self._stats['validation_errors'] += len(errors)
            for error in errors:
                logger.warning(f"Configuration validation: {error}")
    
    def _notify_change(self, key: str, old_value: Any, new_value: Any):
        """Notify registered callbacks of configuration change"""
        for callback in self._reload_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.warning(f"Config callback error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ACCESSOR METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            # Check hot reload
            if self._enable_hot_reload:
                self._check_hot_reload()
            
            value = self._values.get(key)
            if value is not None:
                return value.value
            
            # Check schema for default
            schema = self._schemas.get(key)
            if schema and schema.default is not None:
                return schema.default
            
            return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        return bool(value)
    
    def get_list(self, key: str, default: List = None) -> List:
        """Get a list configuration value"""
        default = default or []
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(',')]
        return default
    
    def get_dict(self, key: str, default: Dict = None) -> Dict:
        """Get a dictionary configuration value"""
        default = default or {}
        value = self.get(key, default)
        return value if isinstance(value, dict) else default
    
    def require(self, key: str) -> Any:
        """
        Get a required configuration value.
        
        Raises:
            ValueError: If the value is missing
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required configuration '{key}' is missing")
        return value
    
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """
        Set a configuration value at runtime.
        
        Args:
            key: Configuration key
            value: Configuration value
            validate: Whether to validate the value
            
        Returns:
            True if value was set successfully
        """
        with self._lock:
            return self._set_value(
                key=key,
                value=value,
                source=ConfigSource.RUNTIME,
                validate=validate
            )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HOT RELOAD
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _check_hot_reload(self):
        """Check if configuration file has changed and reload if needed"""
        current_time = time.time()
        
        if current_time - self._last_reload_check < self._hot_reload_interval:
            return
        
        self._last_reload_check = current_time
        
        config_path = self._config_dir / self._config_file
        if not config_path.exists():
            return
        
        try:
            content = config_path.read_text(encoding='utf-8')
            new_hash = hashlib.md5(content.encode()).hexdigest()
            
            if new_hash != self._file_hash:
                logger.info("Configuration file changed, reloading...")
                self._stats['hot_reloads'] += 1
                self.load(str(config_path))
        
        except Exception as e:
            logger.warning(f"Hot reload check failed: {e}")
    
    def on_change(self, callback: Callable[[str, Any, Any], None]):
        """Register a callback for configuration changes"""
        self._reload_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        return self._status == ConfigStatus.LOADED
    
    def get_status(self) -> ConfigStatus:
        """Get current configuration status"""
        return self._status
    
    def get_all_keys(self) -> List[str]:
        """Get all configuration keys"""
        with self._lock:
            return list(self._values.keys())
    
    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for a configuration key"""
        return self._schemas.get(key)
    
    def get_source(self, key: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value"""
        with self._lock:
            value = self._values.get(key)
            return value.source if value else None
    
    def export(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export all configuration values.
        
        Args:
            include_secrets: Whether to include secret values
            
        Returns:
            Dictionary of all configuration values
        """
        with self._lock:
            result = {}
            for key, value in self._values.items():
                if value.is_secret and not include_secrets:
                    result[key] = "***HIDDEN***"
                else:
                    result[key] = value.value
            return result
    
    def export_safe(self) -> Dict[str, Any]:
        """Export configuration without secrets"""
        return self.export(include_secrets=False)
    
    def export_for_file(self) -> str:
        """Export configuration as JSON for saving to file"""
        data = self.export(include_secrets=False)
        return json.dumps(data, indent=2, default=str)
    
    def save(self, path: str = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            path: File path (default: config directory)
            
        Returns:
            True if saved successfully
        """
        path = path or str(self._config_dir / "config.json")
        
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            Path(path).write_text(self.export_for_file(), encoding='utf-8')
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        return {
            **self._stats,
            'status': self._status.name,
            'values_count': len(self._values),
            'schemas_count': len(self._schemas),
            'load_time': datetime.fromtimestamp(self._load_time).isoformat() if self._load_time else None,
        }
    
    def reset(self):
        """Reset configuration to defaults"""
        with self._lock:
            self._values.clear()
            self._status = ConfigStatus.NOT_LOADED
            self._load_time = None
            self._file_hash = None
            logger.info("Configuration reset to defaults")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


def initialize_config(**kwargs) -> ConfigManager:
    """Initialize global configuration with custom settings"""
    global _config
    _config = ConfigManager(**kwargs)
    return _config


# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get a configuration value"""
    return get_config().get(key, default)


def get_int(key: str, default: int = 0) -> int:
    """Get an integer configuration value"""
    return get_config().get_int(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    """Get a boolean configuration value"""
    return get_config().get_bool(key, default)


def require(key: str) -> Any:
    """Get a required configuration value"""
    return get_config().require(key)


def set_value(key: str, value: Any) -> bool:
    """Set a configuration value"""
    return get_config().set(key, value)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """
    Run self-test for configuration system.
    
    Tests:
    1. Default value loading
    2. Type coercion
    3. Environment variable override
    4. Validation
    5. Runtime override
    """
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    # Create test config
    config = ConfigManager(auto_load=False)
    
    # Test 1: Default values
    config._load_defaults()
    if config.get('app_name') == "JARVIS v14 Ultimate":
        results['passed'].append('default_values')
    else:
        results['failed'].append('default_values')
    
    # Test 2: Type coercion
    coerced = config._coerce_value("123", "int")
    if coerced == 123 and isinstance(coerced, int):
        results['passed'].append('type_coercion_int')
    else:
        results['failed'].append('type_coercion_int')
    
    coerced = config._coerce_value("true", "bool")
    if coerced is True:
        results['passed'].append('type_coercion_bool')
    else:
        results['failed'].append('type_coercion_bool')
    
    # Test 3: Set and get
    config.set('debug_mode', True)
    if config.get_bool('debug_mode') is True:
        results['passed'].append('set_get')
    else:
        results['failed'].append('set_get')
    
    # Test 4: Validation
    schema = ConfigSchema(
        key="test_value",
        type="int",
        min_value=1,
        max_value=100
    )
    
    valid, _ = schema.validate(50)
    if valid:
        results['passed'].append('validation_pass')
    else:
        results['failed'].append('validation_pass')
    
    valid, _ = schema.validate(200)
    if not valid:
        results['passed'].append('validation_fail')
    else:
        results['failed'].append('validation_fail')
    
    # Test 5: Required value
    try:
        config.require('app_name')
        results['passed'].append('require_exists')
    except ValueError:
        results['failed'].append('require_exists')
    
    # Test 6: Export
    export = config.export_safe()
    if isinstance(export, dict) and 'app_name' in export:
        results['passed'].append('export')
    else:
        results['failed'].append('export')
    
    results['stats'] = config.get_stats()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Configuration System - Self Test")
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
    
    print("\n📊 Statistics:")
    stats = test_results['stats']
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Demo
    print("\n" + "=" * 70)
    print("Configuration Demo:")
    print("-" * 70)
    
    config = get_config()
    print(f"App Name: {config.get('app_name')}")
    print(f"Device: {config.get('device_model')}")
    print(f"Debug Mode: {config.get_bool('debug_mode')}")
    print(f"AI Timeout: {config.get_int('ai_timeout_seconds')}s")
    print(f"Max Memory: {config.get_int('max_memory_usage_mb')}MB")
    
    print("\n" + "=" * 70)
