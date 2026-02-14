#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Research Package
======================================

Phase 1 Research Documents for JARVIS AI system.

Documents:
    - github_self_modifying_analysis.md: Analysis of self-modifying AI patterns
    - dependency_patterns_analysis.md: Python dependency handling patterns
    - openrouter_free_models.md: OpenRouter free models research
    - termux_package_matrix.md: Termux package compatibility matrix
    - memory_optimization.md: Memory optimization techniques
    - jarvis_dependency_audit.md: Dependency audit for JARVIS codebase
    - safety_framework_research.md: Safety framework best practices
    - api_key_security.md: API key security guidelines
    - performance_benchmarks.md: Performance benchmark results
    - ux_best_practices.md: User experience best practices
"""

from pathlib import Path

RESEARCH_DIR = Path(__file__).parent

RESEARCH_DOCUMENTS = {
    'github_self_modifying_analysis': 'Analysis of self-modifying AI systems on GitHub',
    'dependency_patterns_analysis': 'Python dependency handling with fallbacks',
    'openrouter_free_models': 'OpenRouter free models research and implementation',
    'termux_package_matrix': 'Termux package compatibility matrix',
    'memory_optimization': 'Memory optimization for 4GB devices',
    'jarvis_dependency_audit': 'Dependency audit for JARVIS codebase',
    'safety_framework_research': 'Safety framework for self-modifying AI',
    'api_key_security': 'API key security best practices',
    'performance_benchmarks': 'Performance benchmark results',
    'ux_best_practices': 'User experience best practices',
}

def get_document_path(name: str) -> Path:
    """Get path to a research document"""
    return RESEARCH_DIR / f"{name}.md"

def list_documents() -> dict:
    """List all available research documents"""
    return RESEARCH_DOCUMENTS.copy()

__all__ = [
    'RESEARCH_DIR',
    'RESEARCH_DOCUMENTS',
    'get_document_path',
    'list_documents',
]
