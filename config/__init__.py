#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Configuration Package
============================================

Configuration management for JARVIS AI system.

Exports:
    - ConfigManager: Main configuration manager
    - ConfigSchema: Schema definition for config values
    - ConfigValue: Configuration value with metadata
    - get_config: Get global config instance
    - get: Get a config value
    - require: Get required config value
"""

from .config_manager import (
    ConfigManager,
    ConfigSchema,
    ConfigValue,
    ConfigSource,
    ConfigStatus,
    JARVIS_CONFIG_SCHEMAS,
    get_config,
    initialize_config,
    get as config_get,
    get_int,
    get_bool,
    require,
    set_value,
)

__all__ = [
    'ConfigManager',
    'ConfigSchema',
    'ConfigValue',
    'ConfigSource',
    'ConfigStatus',
    'JARVIS_CONFIG_SCHEMAS',
    'get_config',
    'initialize_config',
    'config_get',
    'get_int',
    'get_bool',
    'require',
    'set_value',
]
