#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 5 Test Suite
========================================

Comprehensive tests for User Interface modules.

Run: python test_phase5.py
"""

import sys
import os
import time
import tempfile
from pathlib import Path

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


def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{msg}\nExpected {item} in {container}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLI INTERFACE TESTS")
print("=" * 60)


def test_terminal_detector_init():
    from interface.cli import TerminalDetector
    detector = TerminalDetector()
    assert_not_none(detector.capabilities)
    assert_not_none(detector.width)


def test_colors_strip():
    from interface.cli import Colors
    colored = f"{Colors.RED}Hello{Colors.RESET}"
    stripped = Colors.strip(colored)
    assert_equal(stripped, "Hello")


def test_colors_length():
    from interface.cli import Colors
    colored = f"{Colors.RED}Hello{Colors.RESET}"
    length = Colors.length(colored)
    assert_equal(length, 5)


def test_command_parser_basic():
    from interface.cli import CommandParser
    parser = CommandParser()
    result, cmd = parser.parse("echo hello world")
    assert_equal(cmd.name, "echo")
    assert_equal(cmd.args, ["hello", "world"])


def test_command_parser_kwargs():
    from interface.cli import CommandParser
    parser = CommandParser()
    result, cmd = parser.parse("cmd --verbose --count 5")
    assert_equal(cmd.kwargs.get('verbose'), 'true')
    assert_equal(cmd.kwargs.get('count'), '5')


def test_command_parser_quoted():
    from interface.cli import CommandParser
    parser = CommandParser()
    result, cmd = parser.parse('echo "hello world"')
    assert_equal(cmd.args, ['hello world'])


def test_history_manager():
    from interface.cli import HistoryManager
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = os.path.join(tmpdir, 'history.json')
        hm = HistoryManager(history_file=history_file, max_size=10)
        hm.add("command1")
        hm.add("command2")
        assert_equal(hm.size, 2)


def test_completion_engine():
    from interface.cli import CompletionEngine
    engine = CompletionEngine()
    suggestions = engine.complete("hel")
    found_help = any(s.text == 'help' for s in suggestions)
    assert_true(found_help, "Should suggest 'help'")


def test_cli_init():
    from interface.cli import CLI
    cli = CLI()
    assert_not_none(cli)
    assert_equal(cli.VERSION, "14.0.0")


# Run CLI tests
run_test("TerminalDetector initialization", test_terminal_detector_init)
run_test("Colors strip ANSI codes", test_colors_strip)
run_test("Colors visible length", test_colors_length)
run_test("CommandParser basic parsing", test_command_parser_basic)
run_test("CommandParser with kwargs", test_command_parser_kwargs)
run_test("CommandParser quoted strings", test_command_parser_quoted)
run_test("HistoryManager basic operations", test_history_manager)
run_test("CompletionEngine basic completion", test_completion_engine)
run_test("CLI initialization", test_cli_init)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("INPUT HANDLER TESTS")
print("=" * 60)


def test_sanitizer_minimal():
    from interface.input import InputSanitizer, SanitizationLevel
    sanitizer = InputSanitizer()
    result, warnings = sanitizer.sanitize("Hello\x00World", SanitizationLevel.MINIMAL)
    assert_equal(result, "HelloWorld")


def test_sanitizer_detect_type():
    from interface.input import InputSanitizer, InputType
    sanitizer = InputSanitizer()
    json_type = sanitizer.detect_input_type('{"key": "value"}')
    assert_equal(json_type, InputType.JSON)


def test_input_handler_init():
    from interface.input import InputHandler
    handler = InputHandler()
    assert_not_none(handler)


def test_input_handler_string():
    from interface.input import InputHandler, InputSource
    handler = InputHandler()
    result = handler.read(source=InputSource.STRING, content="test content")
    assert_equal(result.content, "test content")


def test_file_handler():
    from interface.input import FileInputHandler
    handler = FileInputHandler()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        f.flush()
        result = handler.read_file(f.name)
        os.unlink(f.name)
    assert_equal(result.content, "Test content")


# Run Input tests
run_test("InputSanitizer minimal sanitization", test_sanitizer_minimal)
run_test("InputSanitizer detect input type", test_sanitizer_detect_type)
run_test("InputHandler initialization", test_input_handler_init)
run_test("InputHandler string input", test_input_handler_string)
run_test("FileInputHandler read file", test_file_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("OUTPUT FORMATTER TESTS")
print("=" * 60)


def test_text_utils_wrap():
    from interface.output import TextUtils
    wrapped = TextUtils.wrap("Hello World", width=5)
    assert_true(len(wrapped) > 0)


def test_text_utils_truncate():
    from interface.output import TextUtils
    truncated = TextUtils.truncate("Hello World This Is Long", 10)
    assert_true("..." in truncated)


def test_syntax_highlighter_python():
    from interface.output import SyntaxHighlighter
    highlighter = SyntaxHighlighter()
    code = "def hello():\n    pass"
    highlighted = highlighter.highlight(code, 'python')
    assert_true('def' in highlighted)


def test_markdown_headers():
    from interface.output import MarkdownRenderer
    renderer = MarkdownRenderer()
    result = renderer.render("# Header One\n## Header Two")
    assert_true("Header One" in result)


def test_table_formatter():
    from interface.output import TableFormatter
    formatter = TableFormatter()
    result = formatter.format(
        headers=['Name', 'Age'],
        rows=[['Alice', 25], ['Bob', 30]]
    )
    assert_true("Name" in result)
    assert_true("Alice" in result)


def test_output_formatter_init():
    from interface.output import OutputFormatter
    formatter = OutputFormatter()
    assert_not_none(formatter)


# Run Output tests
run_test("TextUtils wrap text", test_text_utils_wrap)
run_test("TextUtils truncate", test_text_utils_truncate)
run_test("SyntaxHighlighter Python", test_syntax_highlighter_python)
run_test("MarkdownRenderer headers", test_markdown_headers)
run_test("TableFormatter basic table", test_table_formatter)
run_test("OutputFormatter initialization", test_output_formatter_init)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND PROCESSOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COMMAND PROCESSOR TESTS")
print("=" * 60)


def test_command_registry():
    from interface.commands import CommandRegistry
    registry = CommandRegistry()
    registry.register('test', lambda a, c: "result", description="Test command")
    cmd = registry.get('test')
    assert_not_none(cmd)
    assert_equal(cmd.name, 'test')


def test_command_processor_init():
    from interface.commands import CommandProcessor
    processor = CommandProcessor()
    assert_not_none(processor)
    assert_true(processor.command_count > 0)


def test_command_processor_execute():
    from interface.commands import CommandProcessor
    processor = CommandProcessor()
    result = processor.execute('help')
    assert_true(result.success)


# Run Command tests
run_test("CommandRegistry registration", test_command_registry)
run_test("CommandProcessor initialization", test_command_processor_init)
run_test("CommandProcessor execute", test_command_processor_execute)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SESSION MANAGER TESTS")
print("=" * 60)


def test_session_create():
    from interface.session import SessionManager, SessionConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        session = manager.create_session("test_session")
        assert_not_none(session)
        assert_equal(session.name, "test_session")


def test_session_variables():
    from interface.session import SessionManager, SessionConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        manager.set_variable("test_key", "test_value")
        value = manager.get_variable("test_key")
        assert_equal(value, "test_value")


def test_session_history():
    from interface.session import SessionManager, SessionConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(session_dir=tmpdir, auto_save=False)
        manager = SessionManager(config)
        manager.create_session("test")
        manager.add_history("echo hello", "hello", 10)
        history = manager.get_history()
        assert_equal(len(history), 1)


# Run Session tests
run_test("SessionManager create session", test_session_create)
run_test("SessionManager variables", test_session_variables)
run_test("SessionManager history", test_session_history)


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS INDICATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PROGRESS INDICATOR TESTS")
print("=" * 60)


def test_spinner_init():
    from interface.progress import Spinner, SpinnerStyle
    spinner = Spinner("Loading...", SpinnerStyle.DOTS)
    assert_not_none(spinner)


def test_progress_bar_init():
    from interface.progress import ProgressBar
    bar = ProgressBar(100, "Processing")
    assert_not_none(bar)


def test_progress_indicator_init():
    from interface.progress import ProgressIndicator
    progress = ProgressIndicator()
    assert_not_none(progress)


# Run Progress tests
run_test("Spinner initialization", test_spinner_init)
run_test("ProgressBar initialization", test_progress_bar_init)
run_test("ProgressIndicator initialization", test_progress_indicator_init)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("NOTIFICATION SYSTEM TESTS")
print("=" * 60)


def test_notifier_init():
    from interface.notify import Notifier
    notifier = Notifier()
    assert_not_none(notifier)


def test_notifier_info():
    from interface.notify import Notifier, NotificationConfig
    config = NotificationConfig(termux_notifications=False)
    notifier = Notifier(config)
    notification = notifier.info("Test", "Info message")
    assert_equal(notification.title, "Test")


def test_notifier_history():
    from interface.notify import Notifier, NotificationConfig
    config = NotificationConfig(termux_notifications=False)
    notifier = Notifier(config)
    notifier.info("Test1", "Message1")
    history = notifier.get_history()
    assert_true(len(history) >= 1)


# Run Notification tests
run_test("Notifier initialization", test_notifier_init)
run_test("Notifier send info", test_notifier_info)
run_test("Notifier history", test_notifier_history)


# ═══════════════════════════════════════════════════════════════════════════════
# HELP SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("HELP SYSTEM TESTS")
print("=" * 60)


def test_help_system_init():
    from interface.help import HelpSystem
    help_sys = HelpSystem()
    assert_not_none(help_sys)
    assert_true(help_sys.topic_count > 0)


def test_help_overview():
    from interface.help import HelpSystem
    help_sys = HelpSystem()
    overview = help_sys.show()
    assert_true(len(overview) > 0)


def test_help_topic():
    from interface.help import HelpSystem
    help_sys = HelpSystem()
    topic = help_sys.show("commands")
    assert_true("command" in topic.lower() or "Command" in topic)


def test_help_search():
    from interface.help import HelpSystem
    help_sys = HelpSystem()
    results = help_sys.search("error")
    assert_true(len(results) > 0 or "Found" in results)


# Run Help tests
run_test("HelpSystem initialization", test_help_system_init)
run_test("HelpSystem show overview", test_help_overview)
run_test("HelpSystem show topic", test_help_topic)
run_test("HelpSystem search", test_help_search)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

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


print("\n" + "=" * 60)
if tests_failed == 0:
    print("✓ ALL TESTS PASSED!")
else:
    print(f"✗ {tests_failed} TESTS FAILED")
print("=" * 60)


# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
