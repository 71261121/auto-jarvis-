#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 8 Documentation Tests
==================================================

Tests for validating the documentation system.

Run with: python3 test_phase8.py

Author: JARVIS Self-Modifying AI Project
Version: 14.0.0
"""

import os
import re
import sys
import unittest
from pathlib import Path
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentation(unittest.TestCase):
    """Tests for JARVIS documentation."""

    def setUp(self):
        """Set up test fixtures."""
        self.docs_dir = DOCS_DIR
        self.required_docs = [
            "INSTALLATION.md",
            "USER_GUIDE.md",
            "API.md",
            "CONFIGURATION.md",
            "DEVELOPER.md",
            "TROUBLESHOOTING.md",
            "FAQ.md",
        ]
        self.readme_path = PROJECT_ROOT / "README.md"

    # ─────────────────────────────────────────────────────────────────────────
    # File Existence Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        self.assertTrue(
            self.docs_dir.exists(),
            f"Docs directory not found: {self.docs_dir}"
        )

    def test_readme_exists(self):
        """Test that README.md exists."""
        self.assertTrue(
            self.readme_path.exists(),
            f"README.md not found: {self.readme_path}"
        )

    def test_all_required_docs_exist(self):
        """Test that all required documentation files exist."""
        for doc_name in self.required_docs:
            doc_path = self.docs_dir / doc_name
            self.assertTrue(
                doc_path.exists(),
                f"Required doc not found: {doc_path}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Content Validation Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_readme_has_required_sections(self):
        """Test that README has all required sections."""
        content = self.readme_path.read_text()

        required_sections = [
            "# JARVIS",
            "## Features",
            "## Quick Start",
            "## Documentation",
            "## Installation",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"README missing section: {section}"
            )

    def test_installation_guide_has_required_sections(self):
        """Test that Installation Guide has all required sections."""
        doc_path = self.docs_dir / "INSTALLATION.md"
        content = doc_path.read_text()

        required_sections = [
            "## System Requirements",
            "## Quick Start",
            "## Detailed Installation",
            "## Verification",
            "## Troubleshooting",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"INSTALLATION.md missing section: {section}"
            )

    def test_user_guide_has_required_sections(self):
        """Test that User Guide has all required sections."""
        doc_path = self.docs_dir / "USER_GUIDE.md"
        content = doc_path.read_text()

        required_sections = [
            "## Getting Started",
            "## Commands Reference",
            "## AI Features",
            "## Self-Modification",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"USER_GUIDE.md missing section: {section}"
            )

    def test_api_documentation_has_required_sections(self):
        """Test that API documentation has all required sections."""
        doc_path = self.docs_dir / "API.md"
        content = doc_path.read_text()

        required_sections = [
            "## Core Module",
            "## AI Module",
            "## Interface Module",
            "## Security Module",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"API.md missing section: {section}"
            )

    def test_configuration_guide_has_required_sections(self):
        """Test that Configuration Guide has all required sections."""
        doc_path = self.docs_dir / "CONFIGURATION.md"
        content = doc_path.read_text()

        required_sections = [
            "## General Settings",
            "## AI Settings",
            "## Security Settings",
            "## Environment Variables",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"CONFIGURATION.md missing section: {section}"
            )

    def test_developer_guide_has_required_sections(self):
        """Test that Developer Guide has all required sections."""
        doc_path = self.docs_dir / "DEVELOPER.md"
        content = doc_path.read_text()

        required_sections = [
            "## Architecture Overview",
            "## Project Structure",
            "## Code Style Guide",
            "## Testing",
            "## Contributing",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"DEVELOPER.md missing section: {section}"
            )

    def test_troubleshooting_has_required_sections(self):
        """Test that Troubleshooting has all required sections."""
        doc_path = self.docs_dir / "TROUBLESHOOTING.md"
        content = doc_path.read_text()

        required_sections = [
            "## Installation Issues",
            "## AI/API Issues",
            "## Error Codes Reference",
            "## Recovery Procedures",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"TROUBLESHOOTING.md missing section: {section}"
            )

    def test_faq_has_required_sections(self):
        """Test that FAQ has required sections."""
        doc_path = self.docs_dir / "FAQ.md"
        content = doc_path.read_text()

        # Check for Q&A format
        self.assertIn(
            "###",
            content,
            "FAQ should have questions (### headers)"
        )
        self.assertIn(
            "**",
            content,
            "FAQ should have formatted answers"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Markdown Validation Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_valid_markdown_syntax(self):
        """Test that all markdown files have valid syntax."""
        for doc_name in self.required_docs + ["../README.md"]:
            if doc_name == "../README.md":
                doc_path = self.readme_path
            else:
                doc_path = self.docs_dir / doc_name

            content = doc_path.read_text()

            # Check for balanced headers
            header_pattern = r'^(#{1,6})\s+.+$'
            headers = re.findall(header_pattern, content, re.MULTILINE)
            self.assertTrue(
                len(headers) > 0,
                f"No headers found in {doc_name}"
            )

            # Check for unclosed code blocks
            code_blocks = content.count('```')
            self.assertEqual(
                code_blocks % 2,
                0,
                f"Unclosed code block in {doc_name}"
            )

    def test_links_are_valid_format(self):
        """Test that markdown links have valid format."""
        for doc_name in self.required_docs:
            doc_path = self.docs_dir / doc_name
            content = doc_path.read_text()

            # Find all markdown links
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, content)

            for text, url in links:
                # Skip external links and anchors
                if url.startswith(('http', '#', 'mailto:')):
                    continue

                # Check internal doc links
                if url.endswith('.md'):
                    linked_path = self.docs_dir / url
                    self.assertTrue(
                        linked_path.exists() or (PROJECT_ROOT / url).exists(),
                        f"Broken link in {doc_name}: {url}"
                    )

    def test_code_blocks_have_language(self):
        """Test that code blocks specify language."""
        for doc_name in self.required_docs:
            doc_path = self.docs_dir / doc_name
            content = doc_path.read_text()

            # Find code block starts
            code_block_pattern = r'^```\s*(\w*)'
            matches = re.findall(code_block_pattern, content, re.MULTILINE)

            # Most code blocks should have a language specified
            with_language = sum(1 for m in matches if m)
            total_blocks = len(matches)

            if total_blocks > 0:
                ratio = with_language / total_blocks
                self.assertGreaterEqual(
                    ratio,
                    0.4,
                    f"Many code blocks in {doc_name} missing language specifier (ratio: {ratio:.2%})"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Content Quality Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_minimum_content_length(self):
        """Test that docs have sufficient content."""
        min_lengths = {
            "INSTALLATION.md": 3000,
            "USER_GUIDE.md": 4000,
            "API.md": 5000,
            "CONFIGURATION.md": 4000,
            "DEVELOPER.md": 4000,
            "TROUBLESHOOTING.md": 3000,
            "FAQ.md": 2000,
        }

        for doc_name, min_length in min_lengths.items():
            doc_path = self.docs_dir / doc_name
            content = doc_path.read_text()

            self.assertGreater(
                len(content),
                min_length,
                f"{doc_name} too short ({len(content)} chars, expected {min_length})"
            )

    def test_no_placeholder_content(self):
        """Test that docs don't contain placeholder text."""
        placeholders = [
            "TODO:",
            "FIXME:",
            "PLACEHOLDER",
            "Lorem ipsum",
            "[Insert",
            "[Add",
        ]

        for doc_name in self.required_docs:
            doc_path = self.docs_dir / doc_name
            content = doc_path.read_text()

            for placeholder in placeholders:
                self.assertNotIn(
                    placeholder,
                    content,
                    f"Placeholder '{placeholder}' found in {doc_name}"
                )

    def test_version_references(self):
        """Test that version references are consistent."""
        for doc_name in self.required_docs:
            doc_path = self.docs_dir / doc_name
            content = doc_path.read_text()

            # Check for version at end of document
            if "Version 14.0.0" in content or "v14" in content:
                self.assertIn(
                    "14.0.0",
                    content,
                    f"Inconsistent version in {doc_name}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("JARVIS v14 Ultimate - Phase 8 Documentation Tests")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDocumentation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests Run:    {result.testsRun}")
    print(f"Failures:     {len(result.failures)}")
    print(f"Errors:       {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
