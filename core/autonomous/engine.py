#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Autonomous Engine
========================================

THE BRAIN that makes JARVIS autonomous.

Flow:
1. Receive user input
2. Detect intent (local pattern matching, <50ms)
3. Check safety
4. Execute directly OR get AI assistance
5. Return formatted result

Key Innovation:
- NOT passive (waiting for AI commands)
- ACTIVELY detects and executes
- AI is a TOOL, not the driver

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto

# Import local modules
from .intent_detector import IntentDetector, IntentType, ParsedIntent
from .executor import AutonomousExecutor, ExecutionResult
from .safety_manager import SafetyManager, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """
    Result from the autonomous engine.
    
    This is what gets returned to the user.
    """
    success: bool
    output: str = ""
    error: str = ""
    intent_type: str = ""
    target: str = ""
    requires_ai: bool = False
    ai_explanation: str = ""
    execution_time_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_output(self) -> str:
        """Get formatted output for display"""
        if self.success:
            return self.output
        else:
            return f"❌ Error: {self.error}"


class AutonomousEngine:
    """
    The Autonomous Engine - JARVIS's Brain.
    
    This is the MAIN ENTRY POINT for all user interactions.
    It determines WHAT to do and EXECUTES it.
    
    Usage:
        engine = AutonomousEngine(jarvis_instance)
        result = engine.process("read main.py")
        print(result.formatted_output)
    
    Architecture:
        User Input
            ↓
        IntentDetector.detect()  → ParsedIntent
            ↓
        SafetyManager.check()    → SafetyResult
            ↓
        ┌────────────────────┐
        │                    │
        │  Direct Operation  │  AI-Assisted
        │  (read, list,      │  (modify, create,
        │   execute)         │   analyze)
        │         ↓          │       ↓
        │    Executor        │   AI + Executor
        │                    │
        └────────────────────┘
                    ↓
              EngineResult
    """
    
    # Intents that can be handled directly (no AI)
    DIRECT_INTENTS = {
        IntentType.READ_FILE,
        IntentType.LIST_DIR,
        IntentType.SEARCH_FILES,
        IntentType.EXECUTE_CMD,
        IntentType.INSTALL_PKG,
        IntentType.DELETE_FILE,
        IntentType.GIT_STATUS,
        IntentType.SHOW_STATUS,
        IntentType.SHOW_HELP,
        IntentType.CLEAR_SCREEN,
    }
    
    # Intents that need AI assistance
    AI_ASSISTED_INTENTS = {
        IntentType.MODIFY_FILE,
        IntentType.CREATE_FILE,
        IntentType.ANALYZE_CODE,
        IntentType.GIT_COMMIT,
    }
    
    def __init__(self, jarvis_instance=None, project_root: str = None):
        """
        Initialize Autonomous Engine.
        
        Args:
            jarvis_instance: Main JARVIS instance (for AI client, backup manager)
            project_root: Root directory of the project
        """
        self.jarvis = jarvis_instance
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Initialize components
        self.intent_detector = IntentDetector(project_root=str(self.project_root))
        self.executor = AutonomousExecutor(jarvis_instance, str(self.project_root))
        self.safety = SafetyManager(project_root=str(self.project_root))
        
        # Get AI client from JARVIS
        self.ai_client = None
        if jarvis_instance:
            self.ai_client = getattr(jarvis_instance, 'ai_client', None)
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'direct_executions': 0,
            'ai_assisted': 0,
            'chat_fallbacks': 0,
            'errors': 0,
            'total_time_ms': 0,
        }
        
        logger.info(f"AutonomousEngine initialized (project: {self.project_root})")
    
    def process(self, user_input: str) -> EngineResult:
        """
        Main entry point - Process user input.
        
        This is what main.py calls for every user input.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            EngineResult with output
        """
        start_time = time.time()
        self._stats['total_requests'] += 1
        
        # Step 1: Detect intent
        intent = self.intent_detector.detect(user_input)
        logger.info(f"Detected intent: {intent.intent_type.name} (confidence: {intent.confidence:.2f})")
        
        # Step 2: Handle based on intent type
        if intent.intent_type == IntentType.CHAT:
            # Pure chat - needs AI
            self._stats['chat_fallbacks'] += 1
            return self._handle_chat(user_input, intent)
        
        elif intent.intent_type == IntentType.SHOW_HELP:
            return self._handle_help()
        
        elif intent.intent_type == IntentType.SHOW_STATUS:
            return self._handle_status()
        
        elif intent.intent_type == IntentType.CLEAR_SCREEN:
            os.system('clear' if os.name != 'nt' else 'cls')
            return EngineResult(success=True, output="")
        
        # Step 3: Check safety for operations
        safety_result = self._check_safety(intent)
        if not safety_result.allowed:
            elapsed = (time.time() - start_time) * 1000
            return EngineResult(
                success=False,
                error=safety_result.message,
                intent_type=intent.intent_type.name,
                target=intent.target,
                execution_time_ms=elapsed
            )
        
        # Step 4: Execute
        if intent.intent_type in self.DIRECT_INTENTS:
            # Direct execution
            result = self._execute_direct(intent)
            self._stats['direct_executions'] += 1
        else:
            # AI-assisted execution
            result = self._execute_ai_assisted(intent)
            self._stats['ai_assisted'] += 1
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self._stats['total_time_ms'] += elapsed
        
        if not result.success:
            self._stats['errors'] += 1
        
        # Build engine result
        return EngineResult(
            success=result.success,
            output=result.output,
            error=result.error,
            intent_type=intent.intent_type.name,
            target=intent.target,
            requires_ai=intent.intent_type in self.AI_ASSISTED_INTENTS,
            execution_time_ms=elapsed,
            data=result.data
        )
    
    def _check_safety(self, intent: ParsedIntent) -> SafetyResult:
        """Check if operation is safe"""
        intent_type = intent.intent_type
        
        if intent_type == IntentType.READ_FILE:
            return self.safety.check_file_read(intent.target)
        
        elif intent_type in (IntentType.MODIFY_FILE, IntentType.CREATE_FILE):
            return self.safety.check_file_write(intent.target)
        
        elif intent_type == IntentType.DELETE_FILE:
            return self.safety.check_file_delete(intent.target)
        
        elif intent_type == IntentType.EXECUTE_CMD:
            return self.safety.check_command(intent.target)
        
        elif intent_type == IntentType.INSTALL_PKG:
            return self.safety.check_install(intent.target)
        
        # Default: safe
        return SafetyResult(level=SafetyLevel.SAFE, allowed=True)
    
    def _execute_direct(self, intent: ParsedIntent) -> ExecutionResult:
        """Execute operations that don't need AI"""
        intent_type = intent.intent_type
        target = intent.target
        
        if intent_type == IntentType.READ_FILE:
            return self.executor.read_file(target)
        
        elif intent_type == IntentType.LIST_DIR:
            return self.executor.list_directory(target or ".")
        
        elif intent_type == IntentType.SEARCH_FILES:
            return self.executor.search_files(target)
        
        elif intent_type == IntentType.EXECUTE_CMD:
            return self.executor.execute_command(target)
        
        elif intent_type == IntentType.INSTALL_PKG:
            return self.executor.install_package(target)
        
        elif intent_type == IntentType.DELETE_FILE:
            return self.executor.delete_file(target)
        
        elif intent_type == IntentType.GIT_STATUS:
            return self.executor.execute_command("git status")
        
        else:
            return ExecutionResult(
                success=False,
                operation="UNKNOWN",
                error=f"Unknown intent: {intent_type.name}"
            )
    
    def _execute_ai_assisted(self, intent: ParsedIntent) -> ExecutionResult:
        """Execute operations that need AI help"""
        intent_type = intent.intent_type
        target = intent.target
        
        if not self.ai_client:
            return ExecutionResult(
                success=False,
                operation=intent_type.name,
                target=target,
                error="AI client not available for this operation"
            )
        
        if intent_type == IntentType.MODIFY_FILE:
            return self.executor.modify_file(target, intent.action or intent.content)
        
        elif intent_type == IntentType.CREATE_FILE:
            return self.executor.create_file(target, intent.content or intent.action)
        
        elif intent_type == IntentType.ANALYZE_CODE:
            return self.executor.analyze_code(target, intent.action)
        
        else:
            return ExecutionResult(
                success=False,
                operation=intent_type.name,
                error=f"Unknown AI-assisted intent: {intent_type.name}"
            )
    
    def _handle_chat(self, user_input: str, intent: ParsedIntent) -> EngineResult:
        """Handle general chat - forward to AI"""
        if not self.ai_client:
            return EngineResult(
                success=False,
                error="AI client not available for chat",
                intent_type="CHAT"
            )
        
        try:
            response = self.ai_client.chat(
                message=user_input,
                system="You are JARVIS, a helpful AI assistant on Termux/Android."
            )
            
            if response.success:
                return EngineResult(
                    success=True,
                    output=response.content,
                    intent_type="CHAT",
                    requires_ai=True
                )
            else:
                return EngineResult(
                    success=False,
                    error=response.error or "AI response failed",
                    intent_type="CHAT"
                )
        except Exception as e:
            return EngineResult(
                success=False,
                error=f"Chat error: {e}",
                intent_type="CHAT"
            )
    
    def _handle_help(self) -> EngineResult:
        """Show help information"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                JARVIS v14 - Autonomous AI Assistant           ║
╚══════════════════════════════════════════════════════════════╝

📋 FILE OPERATIONS (Direct - No AI needed):
  read <file>           - Read file content
  list [dir]            - List directory contents
  search <pattern>      - Search for pattern in files
  delete <file>         - Delete a file (with backup)

📝 FILE OPERATIONS (AI-Assisted):
  modify <file> to...   - Modify file with AI help
  create <file> with... - Create new file with AI help
  analyze <file>        - Analyze code for issues

⚡ TERMINAL OPERATIONS:
  run <command>         - Execute terminal command
  install <package>     - Install a package

🔧 GIT OPERATIONS:
  git status            - Show git status

💬 CHAT:
  <any question>        - Ask AI anything

⚡ EXAMPLES:
  "read main.py"
  "list files in core/"
  "modify main.py to add debug command"
  "create utils.py with string helpers"
  "search for def hello"
  "run python test.py"
  "install requests"
  "What is Python?"

Type 'status' for system info, 'help' for this menu.
"""
        return EngineResult(
            success=True,
            output=help_text,
            intent_type="HELP"
        )
    
    def _handle_status(self) -> EngineResult:
        """Show system status"""
        status_lines = [
            "═══ JARVIS Autonomous Engine Status ═══",
            "",
            f"Project Root: {self.project_root}",
            "",
            "Components:",
            f"  Intent Detector: ✓",
            f"  Executor: ✓",
            f"  Safety Manager: ✓",
            f"  AI Client: {'✓' if self.ai_client else '✗'}",
            "",
            "Statistics:",
            f"  Total Requests: {self._stats['total_requests']}",
            f"  Direct Executions: {self._stats['direct_executions']}",
            f"  AI-Assisted: {self._stats['ai_assisted']}",
            f"  Chat Fallbacks: {self._stats['chat_fallbacks']}",
            f"  Errors: {self._stats['errors']}",
            f"  Avg Time: {self._stats['total_time_ms'] / max(1, self._stats['total_requests']):.1f}ms",
        ]
        
        return EngineResult(
            success=True,
            output="\n".join(status_lines),
            intent_type="STATUS"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for AutonomousEngine"""
    results = {
        'passed': [],
        'failed': [],
    }
    
    engine = AutonomousEngine(project_root=os.getcwd())
    
    # Test 1: Help command
    result = engine.process("help")
    if result.success and "FILE OPERATIONS" in result.output:
        results['passed'].append('help_command')
    else:
        results['failed'].append(f'help_command: {result.error}')
    
    # Test 2: List directory
    result = engine.process("list .")
    if result.success:
        results['passed'].append('list_directory')
    else:
        results['failed'].append(f'list_directory: {result.error}')
    
    # Test 3: Read file
    result = engine.process("read main.py")
    if result.success or "not found" in result.error.lower():
        results['passed'].append('read_file')
    else:
        results['failed'].append(f'read_file: {result.error}')
    
    # Test 4: Status
    result = engine.process("status")
    if result.success:
        results['passed'].append('status')
    else:
        results['failed'].append(f'status: {result.error}')
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Autonomous Engine - Self Test")
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
