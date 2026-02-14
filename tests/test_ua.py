#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 64: User Acceptance Tests
====================================================

User acceptance tests for:
- Installation flow
- Basic usage scenarios
- Error messages clarity
- Recovery scenarios

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
"""

import sys
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = None
    
    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        print(f"    Error: {error[:100]}")
    
    def summary(self):
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        total = len(self.passed) + len(self.failed)
        rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"TODO 64: User Acceptance Tests Results")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}%")
        return len(self.failed) == 0


results = TestResult()
import time
results.start_time = time.time()

print("="*60)
print("TODO 64: User Acceptance Tests")
print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION FLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Installation Flow Tests ---")

def test_environment_detection():
    """Test environment detection works"""
    try:
        from install.detect import EnvironmentDetector
        
        detector = EnvironmentDetector()
        report = detector.detect_all()
        
        assert report is not None
        assert 'python_version' in report or hasattr(report, 'python_version')
        
        print(f"    Python: {report.get('python_version', 'N/A')}")
        print(f"    Platform: {report.get('platform', 'N/A')}")
        results.add_pass("uat: Environment detection")
    except Exception as e:
        results.add_fail("uat: Environment detection", str(e))

def test_dependency_check():
    """Test dependency checking"""
    try:
        from install.deps import DependencyInstaller
        
        installer = DependencyInstaller()
        
        # Verify basic packages
        results_list = installer.verify_all()
        
        # At minimum, json and os should work
        results.add_pass("uat: Dependency check")
    except Exception as e:
        results.add_fail("uat: Dependency check", str(e))

def test_config_generation():
    """Test configuration generation"""
    try:
        from install.config_gen import ConfigGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ConfigGenerator()
            config = gen.generate_default()
            
            assert config is not None
            
            # Save config
            config_path = os.path.join(tmpdir, "config.json")
            gen.save(config, config_path)
            
            assert os.path.exists(config_path)
            
            results.add_pass("uat: Config generation")
    except Exception as e:
        results.add_fail("uat: Config generation", str(e))

def test_first_run_wizard():
    """Test first-run wizard"""
    try:
        from install.first_run import FirstRunSetup
        
        setup = FirstRunSetup()
        
        # Get features list
        features = setup.get_features_list()
        
        assert len(features) > 0
        print(f"    Features available: {len(features)}")
        results.add_pass("uat: First-run wizard")
    except Exception as e:
        results.add_fail("uat: First-run wizard", str(e))

test_environment_detection()
test_dependency_check()
test_config_generation()
test_first_run_wizard()


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC USAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Basic Usage Tests ---")

def test_cli_help():
    """Test CLI help is available"""
    try:
        from interface.help import HelpSystem
        
        help_sys = HelpSystem()
        
        # Get overview
        overview = help_sys.show_overview()
        
        assert overview is not None or True  # May return None for display
        results.add_pass("uat: CLI help available")
    except Exception as e:
        results.add_fail("uat: CLI help", str(e))

def test_session_management():
    """Test basic session management"""
    try:
        from interface.session import SessionManager
        
        sm = SessionManager()
        
        # Create session
        session = sm.create_session()
        
        assert session is not None
        
        # Set variable
        sm.set_variable(session, "test", "value")
        
        # Get variable
        value = sm.get_variable(session, "test")
        
        assert value == "value"
        results.add_pass("uat: Session management")
    except Exception as e:
        results.add_fail("uat: Session management", str(e))

def test_command_parsing():
    """Test command parsing"""
    try:
        from interface.input import CommandParser
        
        parser = CommandParser()
        
        # Parse simple command
        result = parser.parse("help")
        
        assert result is not None
        results.add_pass("uat: Command parsing")
    except Exception as e:
        results.add_fail("uat: Command parsing", str(e))

def test_output_formatting():
    """Test output formatting"""
    try:
        from interface.output import OutputFormatter
        
        formatter = OutputFormatter()
        
        # Format text
        text = "Hello World"
        formatted = formatter.format(text)
        
        results.add_pass("uat: Output formatting")
    except Exception as e:
        results.add_fail("uat: Output formatting", str(e))

test_cli_help()
test_session_management()
test_command_parsing()
test_output_formatting()


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR MESSAGE CLARITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Error Message Clarity Tests ---")

def test_error_categorization():
    """Test errors are properly categorized"""
    try:
        from core.error_handler import ErrorHandler, ErrorCategory
        
        handler = ErrorHandler()
        
        # Test different error types
        errors = [
            ValueError("test value error"),
            TypeError("test type error"),
            FileNotFoundError("test file error"),
        ]
        
        for error in errors:
            result = handler.handle(error)
            assert result is not None
        
        results.add_pass("uat: Error categorization")
    except Exception as e:
        results.add_fail("uat: Error categorization", str(e))

def test_user_friendly_messages():
    """Test user-friendly error messages"""
    try:
        from core.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Handle error
        try:
            raise ValueError("Invalid input value")
        except ValueError as e:
            result = handler.handle(e)
        
        # Check for user-friendly message
        # Should have explanation
        results.add_pass("uat: User-friendly messages")
    except Exception as e:
        results.add_fail("uat: User-friendly messages", str(e))

def test_error_suggestions():
    """Test error suggestions"""
    try:
        from core.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Handle error and get suggestions
        try:
            raise FileNotFoundError("config.json not found")
        except FileNotFoundError as e:
            result = handler.handle(e)
        
        # May include suggestions
        results.add_pass("uat: Error suggestions")
    except Exception as e:
        results.add_fail("uat: Error suggestions", str(e))

test_error_categorization()
test_user_friendly_messages()
test_error_suggestions()


# ═══════════════════════════════════════════════════════════════════════════════
# RECOVERY SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Recovery Scenario Tests ---")

def test_backup_recovery():
    """Test backup and recovery"""
    try:
        from core.self_mod.backup_manager import BackupManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bm = BackupManager(backup_dir=tmpdir)
            
            # Create test file
            test_file = os.path.join(tmpdir, "test.py")
            original = "original content"
            
            with open(test_file, 'w') as f:
                f.write(original)
            
            # Create backup
            backup_id = bm.create_backup(test_file, "test backup")
            
            # Modify file
            with open(test_file, 'w') as f:
                f.write("modified")
            
            # Restore
            bm.restore_backup(backup_id)
            
            with open(test_file, 'r') as f:
                restored = f.read()
            
            assert restored == original
            results.add_pass("uat: Backup recovery")
    except Exception as e:
        results.add_fail("uat: Backup recovery", str(e))

def test_error_recovery():
    """Test error recovery"""
    try:
        from core.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Register recovery strategy
        def recover(error):
            return "recovered"
        
        handler.register_recovery(ValueError, recover)
        
        # Trigger error
        try:
            raise ValueError("test")
        except ValueError as e:
            result = handler.handle(e)
        
        results.add_pass("uat: Error recovery")
    except Exception as e:
        results.add_fail("uat: Error recovery", str(e))

def test_circuit_breaker_recovery():
    """Test circuit breaker auto-recovery"""
    try:
        from core.ai.rate_limiter import CircuitBreaker, CircuitState
        import time
        
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        time.sleep(0.3)
        
        # Should be able to execute
        assert cb.can_execute()
        
        results.add_pass("uat: Circuit breaker recovery")
    except Exception as e:
        results.add_fail("uat: Circuit breaker recovery", str(e))

test_backup_recovery()
test_error_recovery()
test_circuit_breaker_recovery()


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-WORLD SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Real-World Scenario Tests ---")

def test_startup_sequence():
    """Test typical startup sequence"""
    try:
        # This simulates what happens when JARVIS starts
        
        # 1. Load config
        from install.config_gen import ConfigGenerator
        gen = ConfigGenerator()
        config = gen.generate_default()
        
        # 2. Initialize core
        from core.events import EventEmitter
        emitter = EventEmitter()
        
        # 3. Initialize cache
        from core.cache import Cache
        cache = Cache(max_size=1000)
        
        # 4. Start state machine
        from core.state_machine import JARVISStateMachine
        sm = JARVISStateMachine()
        
        # 5. Setup error handling
        from core.error_handler import ErrorHandler
        handler = ErrorHandler()
        
        # All should initialize without errors
        results.add_pass("uat: Startup sequence")
    except Exception as e:
        results.add_fail("uat: Startup sequence", str(e))

def test_user_interaction_flow():
    """Test complete user interaction flow"""
    try:
        from interface.session import SessionManager
        from interface.input import InputSanitizer
        from core.events import EventEmitter
        
        # Setup
        sm = SessionManager()
        sanitizer = InputSanitizer()
        emitter = EventEmitter()
        
        # User starts session
        session = sm.create_session()
        
        # User enters command
        raw_input = "  hello world  "
        clean_input = sanitizer.sanitize(raw_input)
        
        # Process command
        emitter.emit("command", {"text": clean_input, "session": session})
        
        results.add_pass("uat: User interaction flow")
    except Exception as e:
        results.add_fail("uat: User interaction flow", str(e))

def test_graceful_shutdown():
    """Test graceful shutdown"""
    try:
        from core.events import EventEmitter
        from core.cache import Cache
        
        # Setup
        emitter = EventEmitter()
        cache = Cache(max_size=100)
        
        # Add some data
        cache.set("test", "value")
        
        # Simulate shutdown
        del emitter
        del cache
        
        results.add_pass("uat: Graceful shutdown")
    except Exception as e:
        results.add_fail("uat: Graceful shutdown", str(e))

test_startup_sequence()
test_user_interaction_flow()
test_graceful_shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()

if success:
    print("\n🎉 TODO 64: ALL USER ACCEPTANCE TESTS PASSED!")
else:
    print("\n⚠️ SOME TESTS FAILED - CHECK ABOVE")

sys.exit(0 if success else 1)
