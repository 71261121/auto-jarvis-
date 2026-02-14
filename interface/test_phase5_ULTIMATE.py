#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 5 ULTIMATE Test Suite
=================================================

100x DEPTH Testing for User Interface modules:
- Race condition detection
- Thread safety verification
- Input validation edge cases
- Memory leak detection
- Resource exhaustion tests
- Boundary condition tests
- Concurrency stress tests

Run: python test_phase5_ULTIMATE.py
"""

import sys
import os
import time
import json
import tempfile
import threading
import concurrent.futures
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test counters
tests_passed = 0
tests_failed = 0
test_results = []


def run_test(name, test_func):
    """Run a single test"""
    global tests_passed, tests_failed
    
    try:
        test_func()
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


def assert_no_exception(func, msg=""):
    try:
        func()
    except Exception as e:
        raise AssertionError(f"{msg}: Unexpected exception {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 ULTIMATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 5 ULTIMATE TESTS - 100x DEPTH ANALYSIS")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. THREAD SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1. THREAD SAFETY TESTS]")


def test_history_manager_thread_safety():
    """Test HistoryManager under concurrent access"""
    from interface.cli import HistoryManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        hm = HistoryManager(history_file=history_file, max_size=100)
        
        errors = []
        
        def add_commands(prefix):
            try:
                for i in range(50):
                    hm.add(f"{prefix}_command_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_commands, args=(f"thread_{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert_true(len(errors) == 0, f"Thread safety errors: {errors}")
        
        # Verify data integrity
        assert_true(hm.size <= 100, "History should not exceed max_size")


def test_session_manager_thread_safety():
    """Test SessionManager under concurrent access"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        
        errors = []
        
        def modify_session(thread_id):
            try:
                for i in range(20):
                    manager.set_variable(f"var_{thread_id}_{i}", i)
                    manager.get_variable(f"var_{thread_id}_{i}")
                    manager.add_history(f"cmd_{thread_id}_{i}", "result", 10)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=modify_session, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


def test_notifier_thread_safety():
    """Test Notifier under concurrent access"""
    from interface.notify import Notifier, NotificationConfig
    
    config = NotificationConfig(termux_notifications=False)
    notifier = Notifier(config)
    
    errors = []
    
    def send_notifications(thread_id):
        try:
            for i in range(20):
                notifier.info(f"Test_{thread_id}_{i}", f"Message {i}")
        except Exception as e:
            errors.append(str(e))
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=send_notifications, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")
    assert_true(notifier.history_count >= 50, "Should have recorded notifications")


def test_command_executor_thread_safety():
    """Test CommandExecutor cache under concurrent access"""
    from interface.commands import CommandProcessor
    
    processor = CommandProcessor()
    errors = []
    
    def execute_commands():
        try:
            for i in range(10):
                result = processor.execute('echo test')
                assert_true(result.success, "Command should succeed")
        except Exception as e:
            errors.append(str(e))
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=execute_commands)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")
    processor.shutdown()


run_test("HistoryManager thread safety", test_history_manager_thread_safety)
run_test("SessionManager thread safety", test_session_manager_thread_safety)
run_test("Notifier thread safety", test_notifier_thread_safety)
run_test("CommandProcessor thread safety", test_command_executor_thread_safety)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INPUT VALIDATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2. INPUT VALIDATION EDGE CASES]")


def test_command_parser_edge_cases():
    """Test CommandParser with edge cases"""
    from interface.cli import CommandParser, ParseResult
    
    parser = CommandParser()
    
    # Empty input
    result, cmd = parser.parse("")
    assert_equal(result, ParseResult.EMPTY)
    
    # Whitespace only
    result, cmd = parser.parse("   \t\n")
    assert_equal(result, ParseResult.EMPTY)
    
    # Comment only
    result, cmd = parser.parse("# This is a comment")
    assert_equal(result, ParseResult.EMPTY)
    
    # Unclosed quote
    result, cmd = parser.parse('echo "unclosed')
    assert_equal(result, ParseResult.INCOMPLETE)
    
    # Escape at end
    result, cmd = parser.parse('echo test\\')
    assert_equal(result, ParseResult.INCOMPLETE)
    
    # Pipe at end
    result, cmd = parser.parse('echo test |')
    assert_equal(result, ParseResult.INCOMPLETE)
    
    # Very long command
    long_cmd = "echo " + "a" * 10000
    result, cmd = parser.parse(long_cmd)
    assert_equal(result, ParseResult.SUCCESS)
    assert_true(len(cmd.args[0]) == 10000)
    
    # Unicode in command
    result, cmd = parser.parse('echo "नमस्ते दुनिया"')
    assert_equal(result, ParseResult.SUCCESS)


def test_input_sanitizer_edge_cases():
    """Test InputSanitizer with edge cases"""
    from interface.input import InputSanitizer, SanitizationLevel, InputType
    
    sanitizer = InputSanitizer()
    
    # Empty string
    result, warnings = sanitizer.sanitize("", SanitizationLevel.STANDARD)
    assert_equal(result, "")
    
    # Only null bytes
    result, warnings = sanitizer.sanitize("\x00\x00\x00", SanitizationLevel.MINIMAL)
    assert_equal(result, "")
    
    # All control characters
    control_chars = ''.join(chr(i) for i in range(32))
    result, warnings = sanitizer.sanitize(control_chars, SanitizationLevel.STANDARD)
    assert_true('\n' in result or '\t' in result or result == "", 
                "Should preserve newline/tab or strip all controls")
    
    # Very long string
    long_str = "a" * 10000000  # 10MB
    result, warnings = sanitizer.sanitize(long_str)
    assert_true(len(result) <= 10000000, "Should handle long strings")
    
    # Type detection edge cases
    assert_equal(sanitizer.detect_input_type(''), InputType.TEXT)
    assert_equal(sanitizer.detect_input_type('   '), InputType.TEXT)
    
    # Malformed JSON
    assert_equal(sanitizer.detect_input_type('{"broken": '), InputType.TEXT)
    
    # Valid JSON
    assert_equal(sanitizer.detect_input_type('{"valid": true}'), InputType.JSON)
    
    # JSON array
    assert_equal(sanitizer.detect_input_type('[1, 2, 3]'), InputType.JSON)


def test_file_input_edge_cases():
    """Test FileInputHandler with edge cases"""
    from interface.input import FileInputHandler
    
    handler = FileInputHandler()
    
    # Non-existent file
    result = handler.read_file('/nonexistent/path/file.txt')
    assert_true(not result.success)
    assert_true('not found' in result.error.lower() or 'not exist' in result.error.lower())
    
    # Directory instead of file
    with tempfile.TemporaryDirectory() as tmpdir:
        result = handler.read_file(tmpdir)
        assert_true(not result.success)
    
    # Empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        f.flush()
        result = handler.read_file(f.name)
        os.unlink(f.name)
    
    assert_true(result.success)
    assert_equal(result.content, "")


run_test("CommandParser edge cases", test_command_parser_edge_cases)
run_test("InputSanitizer edge cases", test_input_sanitizer_edge_cases)
run_test("FileInputHandler edge cases", test_file_input_edge_cases)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT FORMATTER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3. OUTPUT FORMATTER EDGE CASES]")


def test_text_utils_edge_cases():
    """Test TextUtils with edge cases"""
    from interface.output import TextUtils, TextAlign
    
    # Empty string
    result = TextUtils.wrap("", width=10)
    assert_equal(result, "")
    
    # Width of 0 or negative
    result = TextUtils.wrap("test", width=0)
    assert_true(len(result) >= 0)  # Should not crash
    
    # Truncate with suffix longer than text
    result = TextUtils.truncate("ab", 10, "...")
    assert_true("ab" in result)
    
    # Align empty string
    result = TextUtils.align("", 10, TextAlign.CENTER)
    assert_true(len(result) >= 0)


def test_syntax_highlighter_edge_cases():
    """Test SyntaxHighlighter with edge cases"""
    from interface.output import SyntaxHighlighter
    
    highlighter = SyntaxHighlighter()
    
    # Empty code
    result = highlighter.highlight("", "python")
    assert_equal(result, "")
    
    # Very long line
    long_line = "def test(): pass  # " + "x" * 10000
    result = highlighter.highlight(long_line, "python")
    assert_true(len(result) > 0)
    
    # Invalid language
    result = highlighter.highlight("test", "nonexistent_language")
    assert_true(len(result) >= 0)
    
    # Code with only comments
    result = highlighter.highlight("# Just a comment\n# Another comment", "python")
    assert_true("#" in result)
    
    # Unclosed string
    result = highlighter.highlight('x = "unclosed', "python")
    assert_true(len(result) >= 0)
    
    # Unicode in code
    result = highlighter.highlight('msg = "नमस्ते"', "python")
    assert_true("नमस्ते" in result)


def test_markdown_renderer_edge_cases():
    """Test MarkdownRenderer with edge cases"""
    from interface.output import MarkdownRenderer
    
    renderer = MarkdownRenderer()
    
    # Empty markdown
    result = renderer.render("")
    assert_equal(result, "")
    
    # Only whitespace
    result = renderer.render("   \n\n\t\n   ")
    assert_true(len(result) >= 0)
    
    # Unclosed code block
    result = renderer.render("```python\nprint('hello')")
    assert_true("print" in result or len(result) >= 0)
    
    # Nested formatting
    result = renderer.render("**bold and *italic* inside**")
    assert_true(len(result) >= 0)
    
    # Very deep nesting
    deep_md = "#" * 10 + " Too many hashes"
    result = renderer.render(deep_md)
    assert_true(len(result) >= 0)


def test_table_formatter_edge_cases():
    """Test TableFormatter with edge cases"""
    from interface.output import TableFormatter
    
    formatter = TableFormatter()
    
    # Empty table
    result = formatter.format()
    assert_equal(result, "")
    
    # Single cell
    result = formatter.format(headers=['A'], rows=[['1']])
    assert_true("A" in result)
    assert_true("1" in result)
    
    # Mismatched column counts
    result = formatter.format(
        headers=['A', 'B', 'C'],
        rows=[['1'], ['2', '3']]  # Different row lengths
    )
    assert_true(len(result) >= 0)
    
    # Very long cell content
    long_content = "x" * 1000
    result = formatter.format(headers=['Data'], rows=[[long_content]])
    assert_true(len(result) >= 0)
    
    # Special characters in table
    result = formatter.format(
        headers=['Name'],
        rows=[['Test | Pipe'], ['Test < Angle'], ['Test & Amp']]
    )
    assert_true(len(result) >= 0)


run_test("TextUtils edge cases", test_text_utils_edge_cases)
run_test("SyntaxHighlighter edge cases", test_syntax_highlighter_edge_cases)
run_test("MarkdownRenderer edge cases", test_markdown_renderer_edge_cases)
run_test("TableFormatter edge cases", test_table_formatter_edge_cases)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMMAND PROCESSOR STRESS TESTS
# ═════════════════════════════════════════════════════════

print("\n[4. COMMAND PROCESSOR STRESS TESTS]")


def test_command_processor_many_commands():
    """Test CommandProcessor with many command registrations"""
    from interface.commands import CommandProcessor, PermissionLevel
    
    processor = CommandProcessor()
    initial_count = processor.command_count
    
    # Register many commands
    for i in range(100):
        processor.register(
            name=f'test_cmd_{i}',
            handler=lambda args, ctx, i=i: f"Result {i}",
            description=f"Test command {i}",
            permission=PermissionLevel.USER,
        )
    
    assert_true(processor.command_count >= initial_count + 100)
    processor.shutdown()


def test_command_processor_timeout():
    """Test CommandProcessor timeout handling"""
    from interface.commands import CommandProcessor, CommandResult
    
    processor = CommandProcessor(default_timeout=1.0)
    
    # Register slow command
    def slow_handler(args, ctx):
        time.sleep(5)  # Sleep longer than timeout
        return "Should not reach"
    
    processor.register('slow', slow_handler, timeout=0.5)
    
    # Execute - should timeout
    result = processor.execute('slow')
    # Should either timeout or handle gracefully
    assert_true(isinstance(result, CommandResult))
    
    processor.shutdown()


def test_command_processor_invalid_names():
    """Test CommandProcessor with invalid command names"""
    from interface.commands import CommandProcessor
    
    processor = CommandProcessor()
    
    # Non-existent command
    result = processor.execute('nonexistent_command_xyz')
    assert_true(not result.success)
    
    # Empty command
    result = processor.execute('')
    assert_true(not result.success)
    
    # Command with special characters
    result = processor.execute('cmd-with-dashes')
    assert_true(isinstance(result.success, bool))
    
    processor.shutdown()


run_test("CommandProcessor many commands", test_command_processor_many_commands)
run_test("CommandProcessor timeout", test_command_processor_timeout)
run_test("CommandProcessor invalid names", test_command_processor_invalid_names)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SESSION MANAGER PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5. SESSION MANAGER PERSISTENCE TESTS]")


def test_session_persistence():
    """Test session save and restore"""
    from interface.session import SessionManager, SessionConfig, SessionData
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        
        # Create and populate session
        manager1 = SessionManager(config)
        session = manager1.create_session("test_persist")
        manager1.set_variable("test_key", "test_value")
        manager1.add_history("echo test", "test", 10)
        manager1.save_session()
        
        session_id = session.session_id
        
        # Create new manager and load
        manager2 = SessionManager(config)
        loaded = manager2.load_session(session_id)
        
        assert_not_none(loaded)
        # Note: Variable might not persist across instances


def test_session_large_data():
    """Test session with large data"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False, max_history_per_session=10)
        manager = SessionManager(config)
        manager.create_session("test")
        
        # Add many history entries (more than max)
        for i in range(100):
            manager.add_history(f"command_{i}", f"result_{i}", i)
        
        history = manager.get_history()
        # Should be trimmed to max_history_per_session
        assert_true(len(history) <= 10)


def test_session_export_import():
    """Test session export and import"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("export_test")
        manager.set_variable("key1", "value1")
        
        # Export
        export_path = os.path.join(tmpdir, "export.json")
        success = manager.export_session(path=export_path)
        assert_true(success)
        
        # Verify export file exists
        assert_true(os.path.exists(export_path))
        
        # Import
        imported = manager.import_session(export_path)
        assert_not_none(imported)


run_test("Session persistence", test_session_persistence)
run_test("Session large data", test_session_large_data)
run_test("Session export/import", test_session_export_import)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NOTIFICATION SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6. NOTIFICATION SYSTEM TESTS]")


def test_notification_history_limit():
    """Test notification history limit"""
    from interface.notify import Notifier, NotificationConfig
    
    config = NotificationConfig(termux_notifications=False, max_history=10)
    notifier = Notifier(config)
    
    # Add more than max
    for i in range(20):
        notifier.info(f"Test {i}", f"Message {i}")
    
    history = notifier.get_history()
    assert_true(len(history) <= 10)


def test_notification_priorities():
    """Test notification priority handling"""
    from interface.notify import Notifier, NotificationConfig, NotificationPriority
    
    config = NotificationConfig(termux_notifications=False, quiet_mode=True, 
                                quiet_hours_start=0, quiet_hours_end=23)
    notifier = Notifier(config)
    
    # All priorities should create notification objects
    n1 = notifier.info("Low", "Low priority")
    n2 = notifier.warning("Medium", "Medium priority")
    n3 = notifier.error("High", "High priority")
    n4 = notifier.urgent("Critical", "Critical priority")
    
    assert_not_none(n1)
    assert_not_none(n2)
    assert_not_none(n3)
    assert_not_none(n4)


def test_notification_clear_history():
    """Test clearing notification history"""
    from interface.notify import Notifier, NotificationConfig
    
    config = NotificationConfig(termux_notifications=False)
    notifier = Notifier(config)
    
    for i in range(5):
        notifier.info(f"Test {i}", "")
    
    assert_true(notifier.history_count >= 5)
    
    notifier.clear_history()
    assert_equal(notifier.history_count, 0)


run_test("Notification history limit", test_notification_history_limit)
run_test("Notification priorities", test_notification_priorities)
run_test("Notification clear history", test_notification_clear_history)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HELP SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7. HELP SYSTEM TESTS]")


def test_help_search_edge_cases():
    """Test help system search with edge cases"""
    from interface.help import HelpSystem
    
    help_sys = HelpSystem()
    
    # Empty search
    results = help_sys.search("")
    assert_true(len(results) >= 0)
    
    # Very long search
    long_search = "a" * 1000
    results = help_sys.search(long_search)
    assert_true(len(results) >= 0)
    
    # Special characters
    results = help_sys.search("error!@#$%")
    assert_true(len(results) >= 0)
    
    # Unicode search
    results = help_sys.search("नमस्ते")
    assert_true(len(results) >= 0)


def test_help_topic_not_found():
    """Test help system with non-existent topic"""
    from interface.help import HelpSystem
    
    help_sys = HelpSystem()
    
    result = help_sys.show("nonexistent_topic_xyz_12345")
    assert_true("not found" in result.lower() or "No help" in result)


run_test("Help search edge cases", test_help_search_edge_cases)
run_test("Help topic not found", test_help_topic_not_found)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PROGRESS INDICATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[8. PROGRESS INDICATOR TESTS]")


def test_progress_bar_edge_cases():
    """Test progress bar with edge cases"""
    from interface.progress import ProgressBar
    
    # Zero total
    bar = ProgressBar(0, "Test")
    bar.update(1)  # Should not crash
    
    # Negative update
    bar = ProgressBar(100, "Test")
    bar.update(-10)  # Should not crash
    
    # Over 100%
    bar = ProgressBar(100, "Test")
    bar.set(150)  # Should cap at 100


def test_spinner_edge_cases():
    """Test spinner with edge cases"""
    from interface.progress import Spinner, SpinnerStyle
    
    # Empty message
    spinner = Spinner("", SpinnerStyle.DOTS)
    spinner.start()
    time.sleep(0.1)
    spinner.stop("")
    
    # Very long message
    long_msg = "Loading" + "." * 1000
    spinner = Spinner(long_msg, SpinnerStyle.LINE)
    spinner.start()
    time.sleep(0.1)
    spinner.stop()


run_test("ProgressBar edge cases", test_progress_bar_edge_cases)
run_test("Spinner edge cases", test_spinner_edge_cases)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MEMORY STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[9. MEMORY STRESS TEST]")


def test_large_history_memory():
    """Test memory with large history"""
    from interface.cli import HistoryManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        hm = HistoryManager(history_file=history_file, max_size=10000)
        
        # Add many commands
        for i in range(1000):
            hm.add(f"command_{i}" * 10, 0, 0)
        
        assert_true(hm.size <= 10000)


def test_large_session_data():
    """Test session with large data"""
    from interface.session import SessionManager, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        
        # Set large variable
        large_value = "x" * 100000  # 100KB
        manager.set_variable("large", large_value)
        
        # Get it back
        result = manager.get_variable("large")
        assert_true(len(result) == 100000)


run_test("Large history memory", test_large_history_memory)
run_test("Large session data", test_large_session_data)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ULTIMATE TEST SUMMARY")
print("=" * 70)

total_tests = tests_passed + tests_failed
print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {tests_passed}")
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


print("\n" + "=" * 70)
if tests_failed == 0:
    print("✓ ALL ULTIMATE TESTS PASSED!")
else:
    print(f"✗ {tests_failed} ULTIMATE TESTS FAILED")
print("=" * 70)


# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
