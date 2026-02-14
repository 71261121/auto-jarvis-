#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Configuration Generator
==============================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Interactive configuration wizard
- Default configuration generation
- API key setup
- Path configuration

Features:
- Interactive configuration wizard
- Default configurations
- API key management
- Path configuration
- Configuration validation
- Import/Export functionality

Memory Impact: < 5MB for configuration
"""

import os
import sys
import json
import yaml
import getpass
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigMode(Enum):
    """Configuration modes"""
    INTERACTIVE = auto()
    DEFAULT = auto()
    MINIMAL = auto()
    ADVANCED = auto()


class ConfigSection(Enum):
    """Configuration sections"""
    GENERAL = "general"
    AI = "ai"
    SELF_MOD = "self_modification"
    INTERFACE = "interface"
    STORAGE = "storage"
    NETWORK = "network"
    LOGGING = "logging"


@dataclass
class GeneralConfig:
    """General configuration"""
    app_name: str = "JARVIS"
    version: str = "14.0.0"
    debug_mode: bool = False
    quiet_mode: bool = False
    locale: str = "en_US"
    timezone: str = "UTC"


@dataclass
class AIConfig:
    """AI configuration"""
    provider: str = "local"
    model: str = "meta-llama/llama-3.1-8b-instruct:free"
    api_key: str = ""
    fallback_model: str = "local"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    
    # Rate limiting
    rate_limit_per_minute: int = 20
    rate_limit_per_day: int = 500


@dataclass
class SelfModConfig:
    """Self-modification configuration"""
    enabled: bool = True
    auto_backup: bool = True
    backup_dir: str = "~/.jarvis/backups"
    max_backups: int = 50
    validation_level: str = "standard"
    require_confirmation: bool = True
    sandbox_enabled: bool = True


@dataclass
class InterfaceConfig:
    """Interface configuration"""
    theme: str = "dark"
    show_timestamps: bool = True
    history_size: int = 1000
    auto_complete: bool = True
    color_output: bool = True
    unicode_output: bool = True


@dataclass
class StorageConfig:
    """Storage configuration"""
    data_dir: str = "~/.jarvis/data"
    cache_dir: str = "~/.jarvis/cache"
    log_dir: str = "~/.jarvis/logs"
    max_cache_size_mb: int = 100
    max_log_size_mb: int = 50


@dataclass
class NetworkConfig:
    """Network configuration"""
    proxy: str = ""
    timeout: int = 30
    retry_count: int = 3
    user_agent: str = "JARVIS/14.0.0"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class JARVISConfig:
    """Complete JARVIS configuration"""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    self_mod: SelfModConfig = field(default_factory=SelfModConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    config_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'general': asdict(self.general),
            'ai': asdict(self.ai),
            'self_modification': asdict(self.self_mod),
            'interface': asdict(self.interface),
            'storage': asdict(self.storage),
            'network': asdict(self.network),
            'logging': asdict(self.logging),
            'metadata': {
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'config_version': self.config_version,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JARVISConfig':
        """Create from dictionary"""
        config = cls()
        
        if 'general' in data:
            config.general = GeneralConfig(**data['general'])
        if 'ai' in data:
            config.ai = AIConfig(**data['ai'])
        if 'self_modification' in data:
            config.self_mod = SelfModConfig(**data['self_modification'])
        if 'interface' in data:
            config.interface = InterfaceConfig(**data['interface'])
        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])
        if 'network' in data:
            config.network = NetworkConfig(**data['network'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        if 'metadata' in data:
            config.created_at = data['metadata'].get('created_at', '')
            config.updated_at = data['metadata'].get('updated_at', '')
            config.config_version = data['metadata'].get('config_version', '1.0')
        
        return config


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigGenerator:
    """
    Ultra-Advanced Configuration Generator.
    
    Features:
    - Interactive configuration wizard
    - Default configurations
    - API key setup
    - Validation
    
    Memory Budget: < 5MB
    
    Usage:
        generator = ConfigGenerator()
        
        # Generate default config
        config = generator.generate_default()
        
        # Interactive wizard
        config = generator.run_wizard()
        
        # Save config
        generator.save(config, "config.json")
    """
    
    DEFAULT_CONFIG_PATH = "~/.jarvis/config.json"
    
    def __init__(self, config_path: str = None):
        """
        Initialize Config Generator.
        
        Args:
            config_path: Path to configuration file
        """
        self._config_path = Path(config_path or self.DEFAULT_CONFIG_PATH).expanduser()
    
    def generate_default(self) -> JARVISConfig:
        """
        Generate default configuration.
        
        Returns:
            JARVISConfig with defaults
        """
        config = JARVISConfig()
        config.created_at = datetime.now().isoformat()
        config.updated_at = config.created_at
        
        # Expand paths
        config.storage.data_dir = str(Path(config.storage.data_dir).expanduser())
        config.storage.cache_dir = str(Path(config.storage.cache_dir).expanduser())
        config.storage.log_dir = str(Path(config.storage.log_dir).expanduser())
        config.self_mod.backup_dir = str(Path(config.self_mod.backup_dir).expanduser())
        
        return config
    
    def generate_minimal(self) -> JARVISConfig:
        """Generate minimal configuration"""
        config = self.generate_default()
        
        # Disable optional features
        config.self_mod.enabled = False
        config.interface.auto_complete = False
        config.interface.unicode_output = False
        config.logging.file_logging = False
        
        return config
    
    def run_wizard(self, mode: ConfigMode = ConfigMode.INTERACTIVE) -> JARVISConfig:
        """
        Run interactive configuration wizard.
        
        Args:
            mode: Wizard mode
            
        Returns:
            JARVISConfig
        """
        print("\n" + "=" * 60)
        print("  JARVIS Configuration Wizard")
        print("=" * 60 + "\n")
        
        config = self.generate_default()
        
        if mode == ConfigMode.DEFAULT:
            return config
        
        # General settings
        print("\n[General Settings]")
        config.general.debug_mode = self._ask_bool("Enable debug mode?", False)
        config.general.locale = self._ask("Locale", config.general.locale)
        
        # AI settings
        print("\n[AI Settings]")
        self._configure_ai(config)
        
        # Self-modification settings
        print("\n[Self-Modification Settings]")
        config.self_mod.enabled = self._ask_bool("Enable self-modification?", True)
        if config.self_mod.enabled:
            config.self_mod.auto_backup = self._ask_bool("Enable auto-backup?", True)
            config.self_mod.require_confirmation = self._ask_bool("Require confirmation for changes?", True)
        
        # Interface settings
        print("\n[Interface Settings]")
        config.interface.theme = self._ask_choice("Theme", ["dark", "light"], config.interface.theme)
        config.interface.show_timestamps = self._ask_bool("Show timestamps?", True)
        config.interface.color_output = self._ask_bool("Enable colors?", True)
        
        # Logging settings
        print("\n[Logging Settings]")
        config.logging.level = self._ask_choice(
            "Log level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            config.logging.level
        )
        
        config.updated_at = datetime.now().isoformat()
        
        print("\n" + "=" * 60)
        print("  Configuration complete!")
        print("=" * 60 + "\n")
        
        return config
    
    def _configure_ai(self, config: JARVISConfig):
        """Configure AI settings"""
        # Provider selection
        providers = ["openrouter", "openai", "anthropic", "local"]
        config.ai.provider = self._ask_choice("AI Provider", providers, config.ai.provider)
        
        # API key
        if config.ai.provider != "local":
            key_name = f"{config.ai.provider.upper()}_API_KEY"
            existing_key = os.environ.get(key_name, "")
            
            if existing_key:
                use_existing = self._ask_bool(f"Use existing {key_name}?", True)
                if use_existing:
                    config.ai.api_key = f"${key_name}"
                else:
                    config.ai.api_key = getpass.getpass(f"Enter {config.ai.provider} API key: ")
            else:
                config.ai.api_key = getpass.getpass(f"Enter {config.ai.provider} API key: ")
        
        # Model selection (provider-specific)
        if config.ai.provider == "openrouter":
            models = [
                "meta-llama/llama-3.1-8b-instruct:free",
                "google/gemma-2-9b-it:free",
                "mistralai/mistral-7b-instruct:free",
            ]
            config.ai.model = self._ask_choice("Model", models, config.ai.model)
        
        # Temperature
        temp_str = self._ask("Temperature (0.0-2.0)", str(config.ai.temperature))
        try:
            config.ai.temperature = max(0.0, min(2.0, float(temp_str)))
        except:
            pass
    
    def _ask(self, prompt: str, default: str = "") -> str:
        """Ask for input with default"""
        default_str = f" [{default}]" if default else ""
        response = input(f"{prompt}{default_str}: ").strip()
        return response if response else default
    
    def _ask_bool(self, prompt: str, default: bool) -> bool:
        """Ask for boolean input"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        return response in ('y', 'yes', 'true', '1')
    
    def _ask_choice(self, prompt: str, choices: List[str], default: str) -> str:
        """Ask to choose from options"""
        print(f"{prompt}:")
        for i, choice in enumerate(choices):
            marker = ">" if choice == default else " "
            print(f"  {marker} {i + 1}. {choice}")
        
        response = input(f"Choice [1-{len(choices)}]: ").strip()
        
        try:
            index = int(response) - 1
            if 0 <= index < len(choices):
                return choices[index]
        except:
            pass
        
        return default
    
    def save(self, config: JARVISConfig, path: str = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            path: File path (uses default if None)
            
        Returns:
            True if successful
        """
        path = Path(path or self._config_path)
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            config.updated_at = datetime.now().isoformat()
            
            # Write JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load(self, path: str = None) -> Optional[JARVISConfig]:
        """
        Load configuration from file.
        
        Args:
            path: File path (uses default if None)
            
        Returns:
            JARVISConfig or None
        """
        path = Path(path or self._config_path)
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return JARVISConfig.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def validate(self, config: JARVISConfig) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (valid, errors)
        """
        errors = []
        
        # AI validation
        if config.ai.provider != "local":
            if not config.ai.api_key and not config.ai.api_key.startswith('$'):
                errors.append("AI API key is required for non-local provider")
        
        if not 0 <= config.ai.temperature <= 2:
            errors.append("AI temperature must be between 0 and 2")
        
        if config.ai.timeout < 1:
            errors.append("AI timeout must be at least 1 second")
        
        # Storage validation
        for dir_path in [config.storage.data_dir, config.storage.cache_dir, config.storage.log_dir]:
            if not dir_path:
                errors.append(f"Storage directory cannot be empty")
        
        # Logging validation
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_levels:
            errors.append(f"Invalid log level: {config.logging.level}")
        
        return len(errors) == 0, errors
    
    def export_env(self, config: JARVISConfig) -> Dict[str, str]:
        """
        Export configuration as environment variables.
        
        Args:
            config: Configuration to export
            
        Returns:
            Dict of environment variables
        """
        env = {}
        
        # AI settings
        if config.ai.api_key and not config.ai.api_key.startswith('$'):
            env[f"JARVIS_{config.ai.provider.upper()}_API_KEY"] = config.ai.api_key
        
        env["JARVIS_MODEL"] = config.ai.model
        env["JARVIS_DEBUG"] = "1" if config.general.debug_mode else "0"
        env["JARVIS_LOG_LEVEL"] = config.logging.level
        
        return env


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo configuration generator"""
    generator = ConfigGenerator()
    
    print("JARVIS Configuration Generator")
    print("=" * 40)
    
    # Generate default
    config = generator.generate_default()
    print("\nDefault configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Validate
    valid, errors = generator.validate(config)
    if valid:
        print("\n✓ Configuration is valid")
    else:
        print(f"\n✗ Validation errors: {errors}")
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    if generator.save(config, temp_path):
        print(f"\n✓ Saved to {temp_path}")


if __name__ == '__main__':
    main()
