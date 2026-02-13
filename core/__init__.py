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
    - ai: AI provider modules (OpenRouter, rate limiting, model selection)

Author: JARVIS AI System
Version: 14.0.0
"""

# Core version
__version__ = "14.0.0"
__device__ = "RMP2402"
__platform__ = "Termux"

# Subpackages
from . import ai

# Core modules - lazy imports for memory
__all__ = [
    'ai',
    'bulletproof_imports',
    'http_client', 
    'safe_exec',
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

def get_ai_client(api_key: str = None):
    """Get OpenRouter AI client"""
    from .ai import OpenRouterClient
    if api_key:
        return OpenRouterClient(api_key=api_key)
    return OpenRouterClient()
