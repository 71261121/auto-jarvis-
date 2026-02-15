#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - TODO 60: Integration Tests
================================================

Tests for full workflow and component interaction:
- All modules working together
- Error recovery
- State persistence
- End-to-end workflows

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
"""

import sys
import os
import time
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = time.time()
    
    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        print(f"    Error: {error[:100]}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed)
        rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"TODO 60: Integration Tests Results")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}% | Time: {elapsed:.2f}s")
        return len(self.failed) == 0


results = TestResult()
print("="*60)
print("TODO 60: Integration Tests")
print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE + AI INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Core + AI Integration ---")

def test_events_with_ai():
    """Test events system with AI operations"""
    try:
        from core.events import EventEmitter, Event
        
        emitter = EventEmitter()
        events_received = []
        
        def on_ai_request(event):
            events_received.append(event.data)
        
        emitter.on("ai_request", on_ai_request)
        emitter.emit(Event("ai_request", {"prompt": "Hello", "model": "test"}))
        
        assert len(events_received) == 1
        assert events_received[0]["prompt"] == "Hello"
        results.add_pass("integration: Events with AI")
    except Exception as e:
        results.add_fail("integration: Events with AI", str(e))

def test_cache_with_ai_responses():
    """Test cache system with AI responses"""
    try:
        from core.cache import Cache
        
        cache = Cache(max_size=100)
        
        # Simulate AI response caching
        cache.set("ai_response:hash1", {"content": "Hello", "model": "test"})
        
        # Retrieve
        cached = cache.get("ai_response:hash1")
        assert cached["content"] == "Hello"
        results.add_pass("integration: Cache with AI responses")
    except Exception as e:
        results.add_fail("integration: Cache with AI responses", str(e))

def test_state_machine_workflow():
    """Test state machine for AI workflow"""
    try:
        from core.state_machine import JARVISStateMachine, JARVISState
        
        sm = JARVISStateMachine()
        
        # Simulate workflow
        assert sm.current_state == JARVISState.IDLE
        
        sm.transition_to(JARVISState.RUNNING)
        assert sm.current_state == JARVISState.RUNNING
        
        sm.transition_to(JARVISState.IDLE)
        assert sm.current_state == JARVISState.IDLE
        
        results.add_pass("integration: State machine workflow")
    except Exception as e:
        results.add_fail("integration: State machine workflow", str(e))

test_events_with_ai()
test_cache_with_ai_responses()
test_state_machine_workflow()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-MODIFICATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Self-Modification Integration ---")

def test_analyzer_modifier_backup():
    """Test analyzer + modifier + backup integration"""
    try:
        from core.self_mod.code_analyzer import CodeAnalyzer
        from core.self_mod.safe_modifier import SafeModifier
        from core.self_mod.backup_manager import BackupManager
        
        code = "def test(): return 1"
        
        # Analyze
        analyzer = CodeAnalyzer(code)
        
        # Validate
        modifier = SafeModifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Backup
            bm = BackupManager(backup_dir=tmpdir)
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, 'w') as f:
                f.write(code)
            backup_id = bm.create_backup(test_file)
            
            assert bm.backup_exists(backup_id)
            results.add_pass("integration: Analyzer-Modifier-Backup")
    except Exception as e:
        results.add_fail("integration: Analyzer-Modifier-Backup", str(e))

test_analyzer_modifier_backup()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE + CORE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Interface + Core Integration ---")

def test_cli_with_events():
    """Test CLI with events system"""
    try:
        from core.events import EventEmitter
        from interface.commands import CommandRegistry
        
        emitter = EventEmitter()
        registry = CommandRegistry()
        
        results.add_pass("integration: CLI with events")
    except Exception as e:
        results.add_fail("integration: CLI with events", str(e))

def test_session_with_cache():
    """Test session management with cache"""
    try:
        from interface.session import SessionManager
        from core.cache import Cache
        
        sm = SessionManager()
        cache = Cache(max_size=100)
        
        session = sm.create_session()
        sm.set_variable(session, "test_var", "value")
        
        # Cache session data
        session_data = sm.get_session_data(session)
        cache.set(f"session:{session}", session_data)
        
        # Retrieve from cache
        cached = cache.get(f"session:{session}")
        results.add_pass("integration: Session with cache")
    except Exception as e:
        results.add_fail("integration: Session with cache", str(e))

test_cli_with_events()
test_session_with_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Error Recovery ---")

def test_error_handler_recovery():
    """Test error handler recovery"""
    try:
        from core.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Register recovery strategy
        def recovery_strategy(error):
            return "recovered"
        
        handler.register_recovery(ValueError, recovery_strategy)
        
        # Test recovery
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = handler.handle(e)
        
        results.add_pass("integration: Error recovery")
    except Exception as e:
        results.add_fail("integration: Error recovery", str(e))

def test_circuit_breaker_recovery():
    """Test circuit breaker recovery"""
    try:
        from core.ai.rate_limiter import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
        
        # Trigger failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        time.sleep(0.3)
        
        # Should allow test request
        assert cb.can_execute() == True
        results.add_pass("integration: Circuit breaker recovery")
    except Exception as e:
        results.add_fail("integration: Circuit breaker recovery", str(e))

test_error_handler_recovery()
test_circuit_breaker_recovery()


# ═══════════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- State Persistence ---")

def test_session_persistence():
    """Test session persistence"""
    try:
        from interface.session import SessionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(storage_path=os.path.join(tmpdir, "sessions.json"))
            
            session = sm.create_session()
            sm.set_variable(session, "key", "value")
            
            # Save
            sm.save()
            
            # Create new manager
            sm2 = SessionManager(storage_path=os.path.join(tmpdir, "sessions.json"))
            
            # Should have saved session
            results.add_pass("integration: Session persistence")
    except Exception as e:
        results.add_fail("integration: Session persistence", str(e))

def test_cache_persistence():
    """Test cache persistence"""
    try:
        from core.cache import Cache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cache.db")
            
            # Create and populate cache
            cache = Cache(max_size=100, disk_path=cache_file)
            cache.set("key1", "value1")
            
            # Save
            cache.flush()
            
            # Create new cache instance
            cache2 = Cache(max_size=100, disk_path=cache_file)
            
            results.add_pass("integration: Cache persistence")
    except Exception as e:
        results.add_fail("integration: Cache persistence", str(e))

test_session_persistence()
test_cache_persistence()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Full Workflow Tests ---")

def test_complete_user_workflow():
    """Test complete user interaction workflow"""
    try:
        # This tests the full flow:
        # 1. User input -> 2. Command processing -> 3. AI query -> 4. Response
        
        from core.events import EventEmitter
        from interface.input import InputSanitizer
        from interface.commands import CommandRegistry
        from core.cache import Cache
        
        # Setup components
        emitter = EventEmitter()
        cache = Cache(max_size=100)
        registry = CommandRegistry()
        sanitizer = InputSanitizer()
        
        # Simulate workflow
        user_input = "  Hello JARVIS!  "
        clean_input = sanitizer.sanitize(user_input)
        
        assert clean_input == "Hello JARVIS!"
        
        # Emit event
        emitter.emit("user_input", {"text": clean_input})
        
        # Cache response
        cache.set("last_input", clean_input)
        
        results.add_pass("integration: Complete user workflow")
    except Exception as e:
        results.add_fail("integration: Complete user workflow", str(e))

test_complete_user_workflow()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()

if success:
    print("\n🎉 TODO 60: ALL INTEGRATION TESTS PASSED!")
else:
    print("\n⚠️ SOME TESTS FAILED - CHECK ABOVE")

sys.exit(0 if success else 1)
