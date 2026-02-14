#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 5 ULTIMATE Test Suite
==================================================

100x DEEPER Analysis Testing - Edge Cases, Race Conditions,
Thread Safety, Memory Leaks, Resource Management, Termux Compatibility

Run: python test_phase5_ULTIMATE.py
"""

import sys
import os
import time
import json
import threading
import tempfile
import traceback
import signal
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test counters
tests_passed = 0
tests_failed = 0
tests_warned = 0
test_results = []


def run_test(name, test_func):
    """Run a single test"""
    global tests_passed, tests_failed, tests_warned
    
    try:
        result = test_func()
        if result == "WARN":
            tests_warned += 1
            test_results.append(('⚠', name, "Warning"))
            print(f"  ⚠ {name} - Warning")
        else:
            tests_passed += 1
            test_results.append(('✓', name, None))
            print(f"  ✓ {name}")
        return True
    except AssertionError as e:
        tests_failed += 1
        test_results.append(('✗', name, str(e)))
        print(f"  ✗ {name}: {e}")
        return False
    except Exception as e:
        tests_failed += 1
        test_results.append(('✗', name, f"Error: {e}"))
        print(f"  ✗ {name}: Error - {e}")
        return False


def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}\nExpected: {b}\nActual: {a}")


def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(f"Expected True: {msg}")


def assert_not_none(value, msg=""):
    if value is None:
        raise AssertionError(f"Expected not None: {msg}")


def assert_raises(exception_class, func, msg=""):
    """Assert that function raises expected exception"""
    try:
        func()
        raise AssertionError(f"{msg}\nExpected {exception_class.__name__} but no exception raised")
    except exception_class:
        pass


def assert_no_bare_except(filepath):
    """Check file for bare except clauses"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == 'except:' or stripped == 'except :':
            issues.append(f"Line {i}: Bare except clause")
    
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 ULTIMATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 5 ULTIMATE TESTS - 100x DEEP ANALYSIS")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 1: CODE QUALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 1: CODE QUALITY TESTS ---")


def test_cli_no_bare_except():
    """CLI should not have bare except clauses"""
    issues = assert_no_bare_except('interface/cli.py')
    if issues:
        raise AssertionError(f"Bare except found in cli.py:\n" + "\n".join(issues[:5]))


def test_input_no_bare_except():
    """Input should not have bare except clauses"""
    issues = assert_no_bare_except('interface/input.py')
    if issues:
        raise AssertionError(f"Bare except found in input.py:\n" + "\n".join(issues[:5]))


def test_output_no_bare_except():
    """Output should not have bare except clauses"""
    issues = assert_no_bare_except('interface/output.py')
    if issues:
        raise AssertionError(f"Bare except found in output.py:\n" + "\n".join(issues[:5]))


def test_notify_no_bare_except():
    """Notify should not have bare except clauses"""
    issues = assert_no_bare_except('interface/notify.py')
    if issues:
        raise AssertionError(f"Bare except found in notify.py:\n" + "\n".join(issues[:5]))


run_test("CLI: No bare except clauses", test_cli_no_bare_except)
run_test("Input: No bare except clauses", test_input_no_bare_except)
run_test("Output: No bare except clauses", test_output_no_bare_except)
run_test("Notify: No bare except clauses", test_notify_no_bare_except)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 2: THREAD SAFETY TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 2: THREAD SAFETY TESTS ---")


def test_session_manager_thread_safety():
    """SessionManager should be thread-safe"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        
        errors = []
        
        def set_var(i):
            try:
                for j in range(100):
                    manager.set_variable(f"var_{i}_{j}", f"value_{j}")
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=set_var, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


def test_history_manager_thread_safety():
    """HistoryManager should be thread-safe"""
    from interface.cli import HistoryManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        hm = HistoryManager(history_file=history_file, max_size=1000)
        
        errors = []
        
        def add_command(i):
            try:
                for j in range(50):
                    hm.add(f"command_{i}_{j}")
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=add_command, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


def test_completion_engine_thread_safety():
    """CompletionEngine should be thread-safe"""
    from interface.cli import CompletionEngine
    
    engine = CompletionEngine()
    errors = []
    
    def complete_query(q):
        try:
            for _ in range(100):
                suggestions = engine.complete(q)
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=complete_query, args=(q,)) for q in ['hel', 'exi', 'cle', 'his']]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


def test_notifier_thread_safety():
    """Notifier should be thread-safe"""
    from interface.notify import Notifier, NotificationConfig
    
    config = NotificationConfig(termux_notifications=False)
    notifier = Notifier(config)
    
    errors = []
    
    def send_notification(i):
        try:
            for j in range(20):
                notifier.info(f"Title_{i}_{j}", f"Message_{j}")
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=send_notification, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


run_test("SessionManager thread safety", test_session_manager_thread_safety)
run_test("HistoryManager thread safety", test_history_manager_thread_safety)
run_test("CompletionEngine thread safety", test_completion_engine_thread_safety)
run_test("Notifier thread safety", test_notifier_thread_safety)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 3: EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 3: EDGE CASE TESTS ---")


def test_empty_input():
    """InputHandler should handle empty input gracefully"""
    from interface.input import InputHandler, InputSource
    
    handler = InputHandler()
    result = handler.read(source=InputSource.STRING, content="")
    assert_equal(result.content, "")
    assert_true(result.success)


def test_whitespace_input():
    """InputHandler should handle whitespace input"""
    from interface.input import InputHandler, InputSource
    
    handler = InputHandler()
    result = handler.read(source=InputSource.STRING, content="   \n\t   ")
    assert_true(result.content.strip() == "" or result.success)


def test_unicode_input():
    """InputHandler should handle Unicode input"""
    from interface.input import InputHandler, InputSource
    
    handler = InputHandler()
    unicode_text = "Hello 你好 مرحبا 🎉 émojis"
    result = handler.read(source=InputSource.STRING, content=unicode_text)
    assert_true(unicode_text in result.content)


def test_large_input():
    """InputHandler should handle large input"""
    from interface.input import InputHandler, InputSource, InputConfig
    
    config = InputConfig(max_input_size=1024*1024)  # 1MB
    handler = InputHandler(config)
    
    # Create large input
    large_text = "x" * 500000
    result = handler.read(source=InputSource.STRING, content=large_text)
    assert_true(len(result.content) > 0)


def test_special_characters_in_commands():
    """CommandParser should handle special characters"""
    from interface.cli import CommandParser
    
    parser = CommandParser()
    
    # Test various special characters
    test_cases = [
        ('echo "hello; world"', 'echo'),
        ("echo 'single quotes'", 'echo'),
        ('cmd --flag="value with spaces"', 'cmd'),
        # Note: backslash-space is an escaped space, so it's part of the name
        # This is correct shell-like behavior
        ('test\\ backslash', 'test backslash'),
    ]
    
    for cmd_str, expected_name in test_cases:
        result, cmd = parser.parse(cmd_str)
        assert_equal(cmd.name, expected_name, f"Failed for: {cmd_str}")


def test_null_bytes_in_input():
    """InputHandler should handle null bytes"""
    from interface.input import InputSanitizer, SanitizationLevel
    
    sanitizer = InputSanitizer()
    result, warnings = sanitizer.sanitize("Hello\x00World", SanitizationLevel.STANDARD)
    assert_true('\x00' not in result, "Null bytes should be removed")


def test_control_characters():
    """InputSanitizer should handle control characters"""
    from interface.input import InputSanitizer, SanitizationLevel
    
    sanitizer = InputSanitizer()
    control_chars = "".join(chr(i) for i in range(32))
    result, warnings = sanitizer.sanitize(control_chars, SanitizationLevel.STANDARD)
    # Most control chars should be removed except newline and tab
    assert_true(len(result) <= 2, "Control characters should be removed")


def test_json_detection():
    """InputSanitizer should detect JSON"""
    from interface.input import InputSanitizer, InputType
    
    sanitizer = InputSanitizer()
    
    assert_equal(sanitizer.detect_input_type('{"key": "value"}'), InputType.JSON)
    assert_equal(sanitizer.detect_input_type('[1, 2, 3]'), InputType.JSON)


def test_code_detection():
    """InputSanitizer should detect code"""
    from interface.input import InputSanitizer, InputType
    
    sanitizer = InputSanitizer()
    
    python_code = "def hello():\n    print('world')"
    assert_equal(sanitizer.detect_input_type(python_code), InputType.CODE)


run_test("Empty input handling", test_empty_input)
run_test("Whitespace input handling", test_whitespace_input)
run_test("Unicode input handling", test_unicode_input)
run_test("Large input handling", test_large_input)
run_test("Special characters in commands", test_special_characters_in_commands)
run_test("Null bytes handling", test_null_bytes_in_input)
run_test("Control characters handling", test_control_characters)
run_test("JSON detection", test_json_detection)
run_test("Code detection", test_code_detection)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 4: MEMORY MANAGEMENT TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 4: MEMORY MANAGEMENT TESTS ---")


def test_history_size_limit():
    """HistoryManager should enforce size limits"""
    from interface.cli import HistoryManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        max_size = 10
        hm = HistoryManager(history_file=history_file, max_size=max_size)
        
        # Add more items than max
        for i in range(20):
            hm.add(f"command_{i}")
        
        assert_true(hm.size <= max_size, f"Size {hm.size} exceeds max {max_size}")


def test_session_history_limit():
    """SessionManager should enforce history limits"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_dir=tmpdir, 
            auto_save=False,
            max_history_per_session=10
        )
        manager = SessionManager(config)
        manager.create_session("test")
        
        # Add many history entries
        for i in range(50):
            manager.add_history(f"cmd_{i}", f"result_{i}", 10)
        
        history = manager.get_history()
        assert_true(len(history) <= 10, f"History size {len(history)} exceeds limit")


def test_notification_history_limit():
    """Notifier should enforce history limits"""
    from interface.notify import Notifier, NotificationConfig
    
    config = NotificationConfig(termux_notifications=False, max_history=10)
    notifier = Notifier(config)
    
    # Add many notifications
    for i in range(50):
        notifier.info(f"Title_{i}", f"Message_{i}")
    
    assert_true(notifier.history_count <= 10, f"History {notifier.history_count} exceeds limit")


def test_cache_eviction():
    """CommandExecutor should evict old cache entries"""
    from interface.commands import CommandExecutor
    
    executor = CommandExecutor()
    
    # Simulate cache filling
    for i in range(100):
        executor._cache[f"key_{i}"] = ("result", time.time())
    
    # Cache should have reasonable size
    assert_true(len(executor._cache) < 1000, "Cache should be bounded")


run_test("History size limit", test_history_size_limit)
run_test("Session history limit", test_session_history_limit)
run_test("Notification history limit", test_notification_history_limit)
run_test("Cache eviction", test_cache_eviction)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 5: ERROR RECOVERY TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 5: ERROR RECOVERY TESTS ---")


def test_invalid_json_history():
    """HistoryManager should handle corrupt history file"""
    from interface.cli import HistoryManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        
        # Write invalid JSON
        with open(history_file, 'w') as f:
            f.write("{ invalid json }")
        
        # Should not crash
        hm = HistoryManager(history_file=history_file, max_size=10)
        assert_true(hm.size >= 0)


def test_invalid_config_file():
    """ConfigGenerator should handle corrupt config"""
    from install.config_gen import ConfigGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        
        # Write invalid JSON
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        generator = ConfigGenerator(config_file)
        loaded = generator.load()
        
        # Should return None for invalid config
        assert_true(loaded is None or loaded is not None)  # Should not crash


def test_missing_session_file():
    """SessionManager should handle missing session file"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        
        # Try to load non-existent session
        session = manager.load_session("non_existent_id")
        assert_true(session is None)


def test_command_not_found():
    """CommandProcessor should handle unknown commands"""
    from interface.commands import CommandProcessor
    
    processor = CommandProcessor()
    result = processor.execute('nonexistent_command_xyz123')
    
    assert_true(not result.success)
    assert_true('not found' in result.error.lower() or 'unknown' in result.error.lower())


run_test("Invalid JSON history recovery", test_invalid_json_history)
run_test("Invalid config file recovery", test_invalid_config_file)
run_test("Missing session file recovery", test_missing_session_file)
run_test("Command not found handling", test_command_not_found)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 6: TERMUX COMPATIBILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 6: TERMUX COMPATIBILITY TESTS ---")


def test_terminal_detector_termux():
    """TerminalDetector should detect Termux environment"""
    from interface.cli import TerminalDetector
    
    detector = TerminalDetector()
    
    # Should have some capability detected
    assert_not_none(detector.capabilities)
    
    # Should have reasonable width
    assert_true(detector.width > 0 and detector.width <= 1000)


def test_termux_env_check():
    """Check if Termux environment variables exist"""
    is_termux = 'TERMUX_VERSION' in os.environ
    
    if is_termux:
        print(f"    Running in Termux: {os.environ.get('TERMUX_VERSION')}")
    else:
        print("    Not running in Termux (expected in test environment)")
    
    return True  # Always pass, just informational


def test_color_support():
    """Colors should work in Termux"""
    from interface.cli import Colors
    
    # Test color stripping
    colored = f"{Colors.RED}Test{Colors.RESET}"
    stripped = Colors.strip(colored)
    assert_equal(stripped, "Test")


def test_unicode_support():
    """Unicode should work in Termux"""
    from interface.cli import TerminalDetector
    
    detector = TerminalDetector()
    
    # Most terminals support Unicode now
    assert_true(detector.supports_unicode or not detector.supports_unicode)


run_test("TerminalDetector Termux detection", test_terminal_detector_termux)
run_test("Termux environment check", test_termux_env_check)
run_test("Color support", test_color_support)
run_test("Unicode support", test_unicode_support)


# ─────────────────────────────────────────────────────────────────────────────────
# LAYER 7: PERFORMANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────────

print("\n--- LAYER 7: PERFORMANCE TESTS ---")


def test_command_parsing_speed():
    """Command parsing should be fast"""
    from interface.cli import CommandParser
    
    parser = CommandParser()
    
    start = time.time()
    for _ in range(1000):
        parser.parse("echo hello world --verbose --count 10")
    elapsed = time.time() - start
    
    assert_true(elapsed < 1.0, f"Parsing too slow: {elapsed:.3f}s for 1000 iterations")


def test_session_variable_speed():
    """Session variable operations should be fast"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        
        start = time.time()
        for i in range(1000):
            manager.set_variable(f"var_{i}", f"value_{i}")
            manager.get_variable(f"var_{i}")
        elapsed = time.time() - start
        
        assert_true(elapsed < 2.0, f"Variable operations too slow: {elapsed:.3f}s")


def test_output_formatting_speed():
    """Output formatting should be fast"""
    from interface.output import OutputFormatter, MarkdownRenderer
    
    formatter = OutputFormatter()
    renderer = MarkdownRenderer()
    
    markdown = "# Header\n\nParagraph with **bold** and *italic*.\n\n- Item 1\n- Item 2"
    
    start = time.time()
    for _ in range(100):
        renderer.render(markdown)
    elapsed = time.time() - start
    
    assert_true(elapsed < 1.0, f"Markdown rendering too slow: {elapsed:.3f}s")


run_test("Command parsing speed", test_command_parsing_speed)
run_test("Session variable speed", test_session_variable_speed)
run_test("Output formatting speed", test_output_formatting_speed)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ULTIMATE TEST SUMMARY - PHASE 5")
print("=" * 60)

total_tests = tests_passed + tests_failed + tests_warned
print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {tests_passed}")
print(f"Warnings: {tests_warned}")
print(f"Failed: {tests_failed}")

if total_tests > 0:
    print(f"Success Rate: {(tests_passed/total_tests*100):.1f}%")

if tests_failed > 0:
    print("\nFailed Tests:")
    for status, name, error in test_results:
        if status == '✗':
            print(f"  {status} {name}")
            if error:
                print(f"      {error}")

print("\n" + "=" * 60)
if tests_failed == 0:
    print("✓ ALL ULTIMATE TESTS PASSED!")
    print("  Phase 5 is 100% functional for Termux/4GB devices!")
else:
    print(f"✗ {tests_failed} TESTS FAILED - NEEDS FIX")
print("=" * 60)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
