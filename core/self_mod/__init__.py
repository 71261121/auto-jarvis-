#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Modification Package
================================================

The HEART of JARVIS - Self-modification capabilities.

This package enables JARVIS to analyze, modify, backup, and improve
its own code in a safe, controlled manner.

Modules:
    - code_analyzer: AST parsing, complexity analysis, pattern detection
    - safe_modifier: Validated modifications with sandbox testing
    - backup_manager: Version control and rollback system
    - improvement_engine: Learning and self-improvement system

Safety Guarantees:
    1. All modifications are validated before application
    2. Backups are created before every change
    3. Automatic rollback on failure
    4. Learning from outcomes for continuous improvement
    5. Risk assessment for every change

Device: Realme 2 Pro Lite (RMP2402)
Memory: Optimized for 4GB RAM
Platform: Termux compatible
"""

from typing import Dict, Any

from .code_analyzer import (
    CodeAnalyzer,
    FileAnalysis,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    CodeIssue,
    CodeLocation,
    ComplexityMetrics,
    ComplexityLevel,
    NodeType,
    IssueSeverity,
    PatternType,
    get_analyzer,
    analyze_file,
    analyze_code,
)

from .safe_modifier import (
    ModificationEngine,
    Modification,
    ModificationType,
    ModificationStatus,
    ValidationLevel,
    RiskLevel,
    ValidationResult,
    TestResult,
    CodeValidator,
    SandboxExecutor,
    CodeDiff,
    get_modification_engine,
    create_modification,
)

from .backup_manager import (
    BackupManager,
    BackupMetadata,
    BackupType,
    BackupStatus,
    RollbackStatus,
    RollbackResult,
    FileBackup,
    get_backup_manager,
    create_backup,
    rollback,
)

from .improvement_engine import (
    SelfImprovementEngine,
    ModificationOutcome,
    PerformanceMetric,
    ImprovementSuggestion,
    LearningPattern,
    LearningMode,
    OutcomeType,
    ImprovementCategory,
    SuggestionPriority,
    LearningStats,
    get_learning_engine,
    record_outcome,
    get_suggestions,
)

__all__ = [
    # Code Analyzer
    'CodeAnalyzer',
    'FileAnalysis',
    'FunctionInfo',
    'ClassInfo',
    'ImportInfo',
    'CodeIssue',
    'CodeLocation',
    'ComplexityMetrics',
    'ComplexityLevel',
    'NodeType',
    'IssueSeverity',
    'PatternType',
    'get_analyzer',
    'analyze_file',
    'analyze_code',
    
    # Safe Modifier
    'ModificationEngine',
    'Modification',
    'ModificationType',
    'ModificationStatus',
    'ValidationLevel',
    'RiskLevel',
    'ValidationResult',
    'TestResult',
    'CodeValidator',
    'SandboxExecutor',
    'CodeDiff',
    'get_modification_engine',
    'create_modification',
    
    # Backup Manager
    'BackupManager',
    'BackupMetadata',
    'BackupType',
    'BackupStatus',
    'RollbackStatus',
    'RollbackResult',
    'FileBackup',
    'get_backup_manager',
    'create_backup',
    'rollback',
    
    # Improvement Engine
    'SelfImprovementEngine',
    'ModificationOutcome',
    'PerformanceMetric',
    'ImprovementSuggestion',
    'LearningPattern',
    'LearningMode',
    'OutcomeType',
    'ImprovementCategory',
    'SuggestionPriority',
    'LearningStats',
    'get_learning_engine',
    'record_outcome',
    'get_suggestions',
]


def get_self_mod_system() -> Dict[str, Any]:
    """
    Get all self-modification components.
    
    Returns:
        Dict with all components initialized
    """
    return {
        'analyzer': get_analyzer(),
        'modifier': get_modification_engine(),
        'backup': get_backup_manager(),
        'learning': get_learning_engine(),
    }
