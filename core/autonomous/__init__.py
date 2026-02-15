#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Autonomous Engine Module
================================================

This module provides FULL AUTONOMOUS CONTROL over Termux.
No more passive waiting for AI commands - WE DETECT AND EXECUTE.

Components:
- IntentDetector: Detects what user wants from natural language
- AutonomousExecutor: Executes operations directly
- AutonomousEngine: Main orchestrator
- SafetyManager: Protects against dangerous operations

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux

USAGE:
    from core.autonomous import AutonomousEngine
    
    engine = AutonomousEngine(jarvis_instance)
    result = engine.process("read main.py")
    print(result.formatted_output)
"""

from .intent_detector import IntentDetector, IntentType, ParsedIntent
from .executor import AutonomousExecutor, ExecutionResult
from .engine import AutonomousEngine, EngineResult
from .safety_manager import SafetyManager, SafetyLevel, SafetyResult

__all__ = [
    # Core classes
    'AutonomousEngine',
    'IntentDetector',
    'AutonomousExecutor',
    'SafetyManager',
    
    # Data classes
    'ParsedIntent',
    'ExecutionResult',
    'EngineResult',
    'SafetyResult',
    
    # Enums
    'IntentType',
    'SafetyLevel',
]

# Version
__version__ = "1.0.0"

# Convenience function
def create_engine(jarvis_instance=None, project_root: str = None) -> AutonomousEngine:
    """Create an autonomous engine instance"""
    return AutonomousEngine(jarvis_instance, project_root)
