#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 1: Research Documents Test Suite
============================================================

This test suite validates all Phase 1 research documents.

Tests:
1. All research documents exist
2. All documents have required sections
3. All documents have sufficient content
4. All documents have valid markdown
"""

import sys
import os
import re
from pathlib import Path

# Setup paths
RESEARCH_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "research"

class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = 0
    
    def add_pass(self, name):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name, error):
        self.failed.append((name, error))
        print(f"  ✗ {name}: {error[:50]}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        rate = len(self.passed) / total * 100 if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"PHASE 1 RESEARCH TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}%")
        print(f"{'='*60}")
        return len(self.failed) == 0

results = TestResult()

print("="*60)
print("PHASE 1: RESEARCH DOCUMENTS TEST SUITE")
print("="*60)

# Required research documents
REQUIRED_DOCS = [
    "github_self_modifying_analysis.md",
    "dependency_patterns_analysis.md",
    "openrouter_free_models.md",
    "termux_package_matrix.md",
    "memory_optimization.md",
]

# Test 1: Research directory exists
print("\n--- Directory Tests ---")
try:
    assert RESEARCH_DIR.exists(), f"Research directory not found: {RESEARCH_DIR}"
    results.add_pass("Research directory exists")
except Exception as e:
    results.add_fail("Research directory", str(e))

# Test 2: All required documents exist
print("\n--- Document Existence Tests ---")
for doc in REQUIRED_DOCS:
    try:
        doc_path = RESEARCH_DIR / doc
        assert doc_path.exists(), f"Document not found: {doc}"
        results.add_pass(f"Document exists: {doc}")
    except Exception as e:
        results.add_fail(f"Document: {doc}", str(e))

# Test 3: Documents have required sections
print("\n--- Content Validation Tests ---")
REQUIRED_SECTIONS = {
    "github_self_modifying_analysis.md": ["EXECUTIVE SUMMARY", "GITHUB REPOSITORY", "ARCHITECTURE"],
    "dependency_patterns_analysis.md": ["EXECUTIVE SUMMARY", "DEPENDENCY CLASSIFICATION", "FALLBACK"],
    "openrouter_free_models.md": ["EXECUTIVE SUMMARY", "FREE MODELS", "API"],
    "termux_package_matrix.md": ["EXECUTIVE SUMMARY", "COMPATIBILITY", "TERMUX"],
    "memory_optimization.md": ["EXECUTIVE SUMMARY", "MEMORY", "OPTIMIZATION"],
}

for doc, sections in REQUIRED_SECTIONS.items():
    try:
        doc_path = RESEARCH_DIR / doc
        if not doc_path.exists():
            results.add_fail(f"Sections: {doc}", "File not found")
            continue
        
        content = doc_path.read_text()
        missing = []
        for section in sections:
            if section.upper() not in content.upper():
                missing.append(section)
        
        if missing:
            results.add_fail(f"Sections: {doc}", f"Missing: {missing}")
        else:
            results.add_pass(f"All sections present: {doc}")
    except Exception as e:
        results.add_fail(f"Sections: {doc}", str(e))

# Test 4: Minimum content length
print("\n--- Content Length Tests ---")
MIN_LENGTH = 5000  # Minimum characters

for doc in REQUIRED_DOCS:
    try:
        doc_path = RESEARCH_DIR / doc
        if not doc_path.exists():
            results.add_fail(f"Length: {doc}", "File not found")
            continue
        
        content = doc_path.read_text()
        if len(content) >= MIN_LENGTH:
            results.add_pass(f"Content length OK: {doc} ({len(content)} chars)")
        else:
            results.add_fail(f"Length: {doc}", f"Too short: {len(content)} chars")
    except Exception as e:
        results.add_fail(f"Length: {doc}", str(e))

# Test 5: Valid markdown structure
print("\n--- Markdown Validation Tests ---")
for doc in REQUIRED_DOCS:
    try:
        doc_path = RESEARCH_DIR / doc
        if not doc_path.exists():
            results.add_fail(f"Markdown: {doc}", "File not found")
            continue
        
        content = doc_path.read_text()
        
        # Check for headers
        headers = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
        if len(headers) >= 3:
            results.add_pass(f"Valid markdown: {doc} ({len(headers)} headers)")
        else:
            results.add_fail(f"Markdown: {doc}", f"Only {len(headers)} headers")
    except Exception as e:
        results.add_fail(f"Markdown: {doc}", str(e))

# Summary
success = results.summary()
if success:
    print("\n🎉 VERDICT: ALL PHASE 1 RESEARCH TESTS PASSED!")
else:
    print("\n⚠️ VERDICT: SOME TESTS FAILED - CHECK ABOVE")

sys.exit(0 if success else 1)
