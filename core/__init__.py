#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Core Package
===================================

Device: Realme 2 Pro Lite (RMP2402)
RAM: 4GB
Platform: Termux (No Root)

This package contains all core functionality for the JARVIS AI system.
All modules are designed with:
- Memory optimization for 4GB RAM
- Graceful fallbacks for missing dependencies
- Termux compatibility
- Zero-error operation

Modules:
    - bulletproof_imports: Safe import system with fallbacks
    - http_client: HTTP client with layered fallback
    - safe_exec: Safe code execution engine
    - events: Event system with pub/sub pattern
    - cache: Memory and disk caching
    - plugins: Plugin management system
    - state_machine: Finite state machine
    - error_handler: Global error handling
    - ai: AI provider modules (OpenRouter, rate limiting, model selection)
    - memory: Memory system (storage, context, optimization, indexing)
    - self_mod: Self-modification engine (Phase 4)

Author: JARVIS AI System
Version: 14.0.0
"""

# Core version
__version__ = "14.0.0"
__device__ = "RMP2402"
__platform__ = "Termux"

# Subpackages
from . import ai
from . import memory
from . import self_mod

# Core modules - lazy imports for memory
__all__ = [
    'ai',
    'memory',
    'self_mod',
    'bulletproof_imports',
    'http_client', 
    'safe_exec',
    'events',
    'cache',
    'plugins',
    'state_machine',
    'error_handler',
]

# Convenience imports for common operations
def get_importer():
    """Get bulletproof importer instance"""
    from .bulletproof_imports import get_importer
    return get_importer()

def get_http_client():
    """Get HTTP client instance"""
    from .http_client import get_client
    return get_client()

def get_safe_executor():
    """Get safe code executor"""
    from .safe_exec import get_executor
    return get_executor()

def get_event_emitter():
    """Get event emitter instance"""
    from .events import get_event_emitter
    return get_event_emitter()

def get_cache():
    """Get cache instance"""
    from .cache import get_cache
    return get_cache()

def get_plugin_manager():
    """Get plugin manager instance"""
    from .plugins import get_plugin_manager
    return get_plugin_manager()

def get_state_machine():
    """Get state machine instance"""
    from .state_machine import create_jarvis_state_machine
    return create_jarvis_state_machine()

def get_error_handler():
    """Get error handler instance"""
    from .error_handler import get_error_handler
    return get_error_handler()

def get_ai_client(api_key: str = None):
    """Get OpenRouter AI client"""
    from .ai import OpenRouterClient
    if api_key:
        return OpenRouterClient(api_key=api_key)
    return OpenRouterClient()

def get_memory_system():
    """Get memory storage instance"""
    from .memory import get_storage
    return get_storage()

def get_context_system():
    """Get context manager instance"""
    from .memory import get_context_manager
    return get_context_manager()

def get_memory_monitor():
    """Get memory optimizer instance"""
    from .memory import get_memory_optimizer
    return get_memory_optimizer()
