#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Intent Detector
======================================

This module detects user intent from natural language input.
NO MORE PASSIVE WAITING - We actively determine what to do.

Key Innovation:
- Uses pattern matching for fast detection (<50ms)
- Extracts targets, actions, and content from natural language
- Falls back to AI classification for ambiguous inputs

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
"""

import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents - what does user want to DO?"""
    
    # File operations (DIRECT - no AI needed)
    READ_FILE = auto()        # "read main.py"
    LIST_DIR = auto()         # "list files in core/"
    SEARCH_FILES = auto()     # "search for 'def hello'"
    DELETE_FILE = auto()      # "delete test.py"
    
    # File operations (AI-ASSISTED)
    MODIFY_FILE = auto()      # "modify main.py to add debug"
    CREATE_FILE = auto()      # "create utils.py with helpers"
    ANALYZE_CODE = auto()     # "analyze main.py for bugs"
    
    # Terminal operations
    EXECUTE_CMD = auto()      # "run python test.py"
    INSTALL_PKG = auto()      # "install requests"
    
    # Git operations
    GIT_STATUS = auto()       # "git status"
    GIT_COMMIT = auto()       # "commit changes"
    GIT_LOG = auto()          # "show git log"
    
    # System operations
    SHOW_STATUS = auto()      # "show status"
    SHOW_HELP = auto()        # "help"
    CLEAR_SCREEN = auto()     # "clear"
    
    # Chat (AI handles this)
    CHAT = auto()             # General conversation
    
    # Unknown
    UNKNOWN = auto()


@dataclass
class ParsedIntent:
    """Parsed user intent with all extracted information"""
    intent_type: IntentType
    target: str = ""                    # File path, directory, command, etc.
    action: str = ""                    # Specific action (add, remove, change)
    content: str = ""                   # Content to write/modify
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    original_input: str = ""
    needs_ai: bool = False              # Does this need AI assistance?
    
    def __str__(self) -> str:
        return f"Intent({self.intent_type.name}, target={self.target}, confidence={self.confidence:.2f})"


class IntentDetector:
    """
    Detect user intent from natural language.
    
    This is the BRAIN of the autonomous system.
    It figures out WHAT the user wants to do.
    
    Usage:
        detector = IntentDetector()
        intent = detector.detect("read main.py")
        # intent.intent_type == IntentType.READ_FILE
        # intent.target == "main.py"
    """
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INTENT PATTERNS - English + Hindi/Hinglish Support
    # ═══════════════════════════════════════════════════════════════════════════════
    
    INTENT_PATTERNS = {
        # READ_FILE patterns (English + Hindi)
        IntentType.READ_FILE: [
            # English
            (r"^read\s+['\"]?([^'\"\s]+)['\"]?(?:\s+file)?$", 0.95),
            (r"^show\s+(?:me\s+)?(?:the\s+)?(?:content\s+of\s+)?['\"]?([^'\"\s]+)['\"]?$", 0.90),
            (r"^open\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^cat\s+['\"]?([^'\"\s]+)['\"]?$", 0.95),
            (r"^display\s+['\"]?([^'\"\s]+)['\"]?$", 0.90),
            (r"^view\s+['\"]?([^'\"\s]+)['\"]?$", 0.90),
            # Hindi/Hinglish
            (r"^['\"]?([^'\"\s]+\.py)['\"]?\s+(?:ko\s+)?padh[oa]?$", 0.90),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+dikha?o?$", 0.90),  # main.py dikhao
            (r"^['\"]?([^'\"\s]+\.py)['\"]?\s+(?:ka\s+)?content\s+dikha?o?$", 0.90),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+me\s+kya\s+hai$", 0.80),
        ],
        
        # LIST_DIR patterns
        IntentType.LIST_DIR: [
            (r"^list\s*(?:files)?\s*(?:in\s+)?['\"]?([^'\"\s]*)['\"]?$", 0.90),
            (r"^ls\s*['\"]?([^'\"\s]*)['\"]?$", 0.95),
            (r"^dir\s*['\"]?([^'\"\s]*)['\"]?$", 0.90),
            (r"^show\s+(?:me\s+)?(?:files\s+)?(?:in\s+)?['\"]?([^'\"\s]*)['\"]?$", 0.85),
            # Hindi
            (r"^['\"]?([^'\"\s]*)['\"]?\s+me\s+files?\s+dikh[ao]?$", 0.85),
            (r"^files?\s+dikh[ao]?$", 0.90),
        ],
        
        # MODIFY_FILE patterns (English + Hindi)
        IntentType.MODIFY_FILE: [
            # English - specific patterns
            (r"^modify\s+['\"]?([^'\"\s]+)['\"]?(?:\s+to\s+(.+))?$", 0.90),
            (r"^modify\s+(?:yourself|jarvis)(?:[,\.]?\s*(?:and\s+)?(.+))?$", 0.90),  # modify yourself
            (r"^change\s+['\"]?([^'\"\s]+)['\"]?(?:\s+to\s+(.+))?$", 0.85),
            (r"^update\s+['\"]?([^'\"\s]+)['\"]?(?:\s+to\s+(.+))?$", 0.85),
            (r"^edit\s+['\"]?([^'\"\s]+)['\"]?(?:\s+to\s+(.+))?$", 0.90),
            (r"^add\s+(?:a\s+)?(.+?)\s+(?:to\s+)?['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^fix\s+(?:the\s+)?(.+?)\s+(?:in\s+)?['\"]?([^'\"\s]+)['\"]?$", 0.80),
            (r"^improve\s+['\"]?([^'\"\s]+)['\"]?$", 0.80),
            # Hindi/Hinglish
            (r"^['\"]?([^'\"\s]+)['\"]?\s+(?:ko\s+)?modify\s+kar[oa]?$", 0.90),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+(?:ko\s+)?change\s+kar[oa]?$", 0.85),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+me\s+(.+?)\s+add\s+kar[oa]?$", 0.85),
            (r"^apne\s+aap\s+ko\s+modify\s+kar[oa]?$", 0.90),  # apne aap ko modify karo
        ],
        
        # CREATE_FILE patterns
        IntentType.CREATE_FILE: [
            (r"^create\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?['\"]?([^'\"\s]+)['\"]?(?:\s+(?:with\s+)?(.+))?$", 0.90),
            (r"^make\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?['\"]?([^'\"\s]+)['\"]?(?:\s+(?:with\s+)?(.+))?$", 0.85),
            (r"^new\s+file\s+['\"]?([^'\"\s]+)['\"]?(?:\s+(?:with\s+)?(.+))?$", 0.90),
            (r"^touch\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            # Hindi
            (r"^['\"]?([^'\"\s]+)['\"]?\s+(?:file\s+)?bana[oa]?$", 0.85),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+create\s+kar[oa]?$", 0.85),
            # Feature creation
            (r"^create\s+(?:a\s+)?(.+?)\s+(?:feature|module|function)", 0.85),
        ],
        
        # DELETE_FILE patterns
        IntentType.DELETE_FILE: [
            (r"^delete\s+['\"]?([^'\"\s]+)['\"]?$", 0.90),
            (r"^remove\s+['\"]?([^'\"\s]+)['\"]?$", 0.90),
            (r"^rm\s+['\"]?([^'\"\s]+)['\"]?$", 0.95),
            # Hindi
            (r"^['\"]?([^'\"\s]+)['\"]?\s+(?:ko\s+)?delete\s+kar[oa]?$", 0.90),
            (r"^['\"]?([^'\"\s]+)['\"]?\s+hata[oa]?$", 0.85),
        ],
        
        # SEARCH_FILES patterns
        IntentType.SEARCH_FILES: [
            (r"^search\s+(?:for\s+)?['\"]?([^'\"]+)['\"]?$", 0.85),
            (r"^find\s+(?:files\s+)?(?:containing\s+)?['\"]?([^'\"]+)['\"]?$", 0.85),
            (r"^grep\s+['\"]?([^'\"]+)['\"]?$", 0.90),
            # Hindi
            (r"^['\"]?([^'\"]+)['\"]?\s+dhoond[ho]?$", 0.85),
            (r"^['\"]?([^'\"]+)['\"]?\s+search\s+kar[oa]?$", 0.85),
        ],
        
        # EXECUTE_CMD patterns
        IntentType.EXECUTE_CMD: [
            (r"^run\s+(.+)$", 0.90),
            (r"^execute\s+(.+)$", 0.90),
            (r"^(?:python|python3)\s+(.+)$", 0.95),
            (r"^pip\s+(.+)$", 0.95),
            (r"^pkg\s+(.+)$", 0.95),
            (r"^bash\s+(.+)$", 0.90),
            # Hindi
            (r"^(.+)\s+chala[oa]?$", 0.85),  # python test.py chalao
        ],
        
        # INSTALL_PKG patterns
        IntentType.INSTALL_PKG: [
            (r"^install\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^pip\s+install\s+['\"]?([^'\"\s]+)['\"]?$", 0.95),
            (r"^pkg\s+install\s+['\"]?([^'\"\s]+)['\"]?$", 0.95),
            # Hindi
            (r"^['\"]?([^'\"\s]+)['\"]?\s+install\s+kar[oa]?$", 0.85),
        ],
        
        # ANALYZE_CODE patterns (English + Hindi)
        IntentType.ANALYZE_CODE: [
            # English
            (r"^analyze\s+['\"]?([^'\"\s]+)['\"]?(?:\s+file)?$", 0.90),
            (r"^analyze\s+(?:the\s+)?['\"]?([^'\"\s]+)['\"]?$", 0.90),
            (r"^check\s+['\"]?([^'\"\s]+)['\"]?(?:\s+for\s+(.+))?$", 0.85),
            (r"^debug\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^review\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^inspect\s+['\"]?([^'\"\s]+)['\"]?$", 0.85),
            (r"^find\s+(?:bugs?|issues?|errors?)\s+(?:in\s+)?['\"]?([^'\"\s]+)['\"]?$", 0.80),
            # Hindi/Hinglish - KEY PATTERNS
            (r"^['\"]?([^'\"\s]+)['\"]?\s+me\s+(?:error|bug|issue)\s+(?:find|dhoondh)(?:\s+kar[oa])?$", 0.90),  # main.py me error find karo
            (r"^['\"]?([^'\"\s]+)['\"]?\s+me\s+(?:errors?|bugs?|issues?)\s+dhoondh[oa]?$", 0.90),  # main.py me error dhoondho
            (r"^['\"]?([^'\"\s]+)['\"]?\s+(?:me\s+)?kya\s+(?:error|bug)\s+hai$", 0.85),  # main.py me kya error hai
            (r"^['\"]?([^'\"\s]+)['\"]?\s+analyze\s+kar[oa]?$", 0.90),  # main.py analyze karo
            (r"^['\"]?([^'\"\s]+)['\"]?\s+check\s+kar[oa]?$", 0.85),
        ],
        
        # GIT_STATUS patterns
        IntentType.GIT_STATUS: [
            (r"^git\s+status$", 0.95),
            (r"^show\s+git\s+status$", 0.90),
        ],
        
        # GIT_COMMIT patterns
        IntentType.GIT_COMMIT: [
            (r"^git\s+commit(?:\s+(.+))?$", 0.95),
            (r"^commit\s+(?:changes\s+)?(?:with\s+message\s+)?['\"]?([^'\"]+)['\"]?$", 0.85),
        ],
        
        # SHOW_STATUS patterns
        IntentType.SHOW_STATUS: [
            (r"^status$", 0.90),
            (r"^show\s+status$", 0.95),
        ],
        
        # SHOW_HELP patterns
        IntentType.SHOW_HELP: [
            (r"^help$", 0.95),
            (r"^show\s+help$", 0.95),
            (r"^help\s+dikh[ao]?$", 0.90),
        ],
        
        # CLEAR_SCREEN patterns
        IntentType.CLEAR_SCREEN: [
            (r"^clear$", 0.95),
            (r"^cls$", 0.95),
        ],
    }
    
    # Intents that need AI assistance
    AI_ASSISTED_INTENTS = {
        IntentType.MODIFY_FILE,
        IntentType.CREATE_FILE,
        IntentType.ANALYZE_CODE,
        IntentType.GIT_COMMIT,
    }
    
    # Common file extensions to help with path detection
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.zsh',
        '.html', '.css', '.scss', '.json', '.xml', '.yaml', '.yml',
        '.md', '.txt', '.sql', '.r', '.swift', '.kt'
    }
    
    def __init__(self, project_root: str = None):
        """
        Initialize Intent Detector.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._compiled_patterns = self._compile_patterns()
        
        logger.info(f"IntentDetector initialized (project: {self.project_root})")
    
    def _compile_patterns(self) -> Dict[IntentType, List[Tuple[re.Pattern, float]]]:
        """Compile regex patterns for faster matching"""
        compiled = {}
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            compiled[intent_type] = [
                (re.compile(p, re.IGNORECASE), conf) 
                for p, conf in patterns
            ]
        return compiled
    
    def detect(self, user_input: str) -> ParsedIntent:
        """
        Detect user intent from input string.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            ParsedIntent with detected type, target, and confidence
        """
        start_time = time.time()
        
        # Normalize input
        input_clean = user_input.strip()
        input_lower = input_clean.lower()
        
        # Try each intent type
        best_match = ParsedIntent(
            intent_type=IntentType.UNKNOWN,
            original_input=input_clean,
            confidence=0.0
        )
        
        for intent_type, patterns in self._compiled_patterns.items():
            for pattern, confidence in patterns:
                match = pattern.match(input_clean)
                if match:
                    # Found a match
                    parsed = self._extract_intent(intent_type, match, input_clean, confidence)
                    if parsed.confidence > best_match.confidence:
                        best_match = parsed
                    break
        
        # If no pattern matched, treat as chat
        if best_match.intent_type == IntentType.UNKNOWN:
            best_match = ParsedIntent(
                intent_type=IntentType.CHAT,
                original_input=input_clean,
                confidence=0.5,
                needs_ai=True
            )
        
        # Mark if AI assistance is needed
        best_match.needs_ai = best_match.intent_type in self.AI_ASSISTED_INTENTS
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Intent detected in {elapsed:.1f}ms: {best_match}")
        
        return best_match
    
    def _extract_intent(
        self, 
        intent_type: IntentType, 
        match: re.Match, 
        original: str, 
        confidence: float
    ) -> ParsedIntent:
        """Extract detailed intent from regex match"""
        
        groups = match.groups()
        target = ""
        action = ""
        content = ""
        
        # Extract based on intent type
        if intent_type == IntentType.READ_FILE:
            target = groups[0] if groups else ""
            # Find last group that looks like a file path
            for g in groups:
                if g and ('.' in g or '/' in g or g.endswith('.py')):
                    target = g
                    break
        
        elif intent_type == IntentType.LIST_DIR:
            target = groups[0] if groups else "."
            if not target:
                target = "."
        
        elif intent_type == IntentType.MODIFY_FILE:
            # "modify main.py to add debug" -> target=main.py, action=add debug
            if len(groups) >= 2:
                if groups[1]:  # Has "to X" part
                    target = groups[0]
                    action = groups[1]
                else:
                    target = groups[0]
            else:
                target = groups[0] if groups else ""
        
        elif intent_type == IntentType.CREATE_FILE:
            # "create utils.py with helpers" -> target=utils.py, content=helpers
            if len(groups) >= 2 and groups[1]:
                target = groups[0]
                content = groups[1]
            else:
                target = groups[0] if groups else ""
        
        elif intent_type == IntentType.DELETE_FILE:
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.SEARCH_FILES:
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.EXECUTE_CMD:
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.INSTALL_PKG:
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.ANALYZE_CODE:
            target = groups[0] if groups else ""
            if len(groups) >= 2 and groups[1]:
                action = groups[1]
        
        # Clean up target
        target = target.strip().strip('"\'')
        
        return ParsedIntent(
            intent_type=intent_type,
            target=target,
            action=action,
            content=content,
            confidence=confidence,
            original_input=original,
            needs_ai=intent_type in self.AI_ASSISTED_INTENTS
        )
    
    def resolve_path(self, target: str) -> Path:
        """
        Resolve a target string to an absolute path.
        
        Args:
            target: Target string (can be relative, absolute, or just filename)
            
        Returns:
            Resolved Path object
        """
        if not target:
            return self.project_root
        
        target_path = Path(target)
        
        # Already absolute
        if target_path.is_absolute():
            return target_path
        
        # Check if exists relative to project root
        full_path = self.project_root / target
        if full_path.exists():
            return full_path
        
        # Try to find the file in project
        for ext in self.CODE_EXTENSIONS:
            matches = list(self.project_root.glob(f"**/{target}{ext}"))
            if matches:
                return matches[0]
        
        # Return the path as-is (might not exist yet)
        return full_path
    
    def is_file_operation(self, intent: ParsedIntent) -> bool:
        """Check if intent is a file operation"""
        return intent.intent_type in {
            IntentType.READ_FILE,
            IntentType.MODIFY_FILE,
            IntentType.CREATE_FILE,
            IntentType.DELETE_FILE,
            IntentType.ANALYZE_CODE,
        }
    
    def is_dangerous(self, intent: ParsedIntent) -> bool:
        """Check if intent might be dangerous"""
        return intent.intent_type in {
            IntentType.DELETE_FILE,
            IntentType.EXECUTE_CMD,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for IntentDetector"""
    results = {
        'passed': [],
        'failed': [],
    }
    
    detector = IntentDetector()
    
    # Test cases
    test_cases = [
        ("read main.py", IntentType.READ_FILE, "main.py"),
        ("list files in core/", IntentType.LIST_DIR, "core/"),
        ("modify main.py to add debug", IntentType.MODIFY_FILE, "main.py"),
        ("create utils.py with helper functions", IntentType.CREATE_FILE, "utils.py"),
        ("delete test.py", IntentType.DELETE_FILE, "test.py"),
        ("search for def hello", IntentType.SEARCH_FILES, "def hello"),
        ("run python test.py", IntentType.EXECUTE_CMD, "python test.py"),
        ("install requests", IntentType.INSTALL_PKG, "requests"),
        ("analyze main.py", IntentType.ANALYZE_CODE, "main.py"),
        ("git status", IntentType.GIT_STATUS, ""),
        ("help", IntentType.SHOW_HELP, ""),
        ("clear", IntentType.CLEAR_SCREEN, ""),
        ("What is the meaning of life?", IntentType.CHAT, ""),
    ]
    
    for user_input, expected_type, expected_target in test_cases:
        intent = detector.detect(user_input)
        
        if intent.intent_type == expected_type:
            results['passed'].append(f"'{user_input}' → {expected_type.name}")
        else:
            results['failed'].append(
                f"'{user_input}' → Expected {expected_type.name}, got {intent.intent_type.name}"
            )
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Intent Detector - Self Test")
    print("=" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    print(f"\nTotal: {len(test_results['passed'])}/{len(test_results['passed']) + len(test_results['failed'])} passed")
