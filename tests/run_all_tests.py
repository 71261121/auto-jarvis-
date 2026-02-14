#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 7: Unified Test Runner
==================================================

Runs all Phase 7 tests (TODO 57-64) and generates comprehensive report.

Phase 7: Testing & Validation
- TODO 57: Unit Tests - Core
- TODO 58: Unit Tests - AI Engine
- TODO 59: Unit Tests - Self-Modification
- TODO 60: Integration Tests
- TODO 61: Performance Tests
- TODO 62: Compatibility Tests
- TODO 63: Security Tests
- TODO 64: User Acceptance Tests

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test files in order (TODO 57-64)
TEST_FILES = [
    ("TODO 57: Core Unit Tests", "tests/test_core.py"),
    ("TODO 58: AI Engine Tests", "tests/test_ai.py"),
    ("TODO 59: Self-Modification Tests", "tests/test_selfmod.py"),
    ("TODO 60: Integration Tests", "tests/test_integration.py"),
    ("TODO 61: Performance Tests", "tests/test_performance.py"),
    ("TODO 62: Compatibility Tests", "tests/test_compat.py"),
    ("TODO 63: Security Tests", "tests/test_security.py"),
    ("TODO 64: User Acceptance Tests", "tests/test_ua.py"),
]


def print_banner():
    """Print test suite banner"""
    print()
    print("╔" + "═"*60 + "╗")
    print("║" + " "*60 + "║")
    print("║" + "   JARVIS v14 Ultimate - Phase 7 Test Suite".center(60) + "║")
    print("║" + " "*60 + "║")
    print("║" + "   Testing & Validation (TODO 57-64)".center(60) + "║")
    print("║" + " "*60 + "║")
    print("╚" + "═"*60 + "╝")
    print()


def run_test(name: str, test_file: str) -> dict:
    """Run a single test file and return results"""
    result = {
        'name': name,
        'file': test_file,
        'passed': 0,
        'failed': 0,
        'total': 0,
        'success': False,
        'time': 0,
        'output': ''
    }
    
    test_path = PROJECT_ROOT / test_file
    
    if not test_path.exists():
        result['output'] = f"Test file not found: {test_file}"
        result['failed'] = 1
        return result
    
    print(f"\n{'─'*60}")
    print(f"Running: {name}")
    print(f"File: {test_file}")
    print(f"{'─'*60}")
    
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)},
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        result['time'] = time.time() - start_time
        result['output'] = proc.stdout + proc.stderr
        result['success'] = proc.returncode == 0
        
        # Parse output for test counts
        output = proc.stdout.lower()
        
        # Look for test summary patterns
        import re
        
        # Pattern: "Total: X | Passed: Y | Failed: Z"
        match = re.search(r'total:\s*(\d+).*passed:\s*(\d+).*failed:\s*(\d+)', output)
        if match:
            result['total'] = int(match.group(1))
            result['passed'] = int(match.group(2))
            result['failed'] = int(match.group(3))
        else:
            # Alternative patterns
            if 'all tests passed' in output or 'all.*passed' in output:
                result['passed'] = 1
                result['total'] = 1
            elif proc.returncode == 0:
                result['passed'] = 1
                result['total'] = 1
            else:
                result['failed'] = 1
                result['total'] = 1
        
    except subprocess.TimeoutExpired:
        result['time'] = time.time() - start_time
        result['output'] = "Test timed out (300s)"
        result['failed'] = 1
        result['total'] = 1
        result['success'] = False
    
    except Exception as e:
        result['time'] = time.time() - start_time
        result['output'] = f"Error running test: {str(e)}"
        result['failed'] = 1
        result['total'] = 1
        result['success'] = False
    
    return result


def print_results(results: list):
    """Print summary results"""
    print()
    print("╔" + "═"*60 + "╗")
    print("║" + " "*60 + "║")
    print("║" + "   PHASE 7 TEST RESULTS SUMMARY".center(60) + "║")
    print("║" + " "*60 + "║")
    print("╠" + "═"*60 + "╣")
    
    total_passed = 0
    total_failed = 0
    total_tests = 0
    
    for r in results:
        total_passed += r['passed']
        total_failed += r['failed']
        total_tests += r['total']
        
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        time_str = f"{r['time']:.2f}s"
        
        print(f"║ {r['name'][:45]:<45} {status:<8} {time_str:>6} ║")
    
    print("╠" + "═"*60 + "╣")
    
    total_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"║ {'TOTALS':<45} {total_passed:>5}/{total_tests:<5} {total_rate:>6.1f}% ║")
    print("║" + " "*60 + "║")
    
    if total_failed == 0:
        print("║" + "   🎉 ALL PHASE 7 TESTS PASSED! 🎉".center(60) + "║")
    else:
        print("║" + f"   ⚠️ {total_failed} TEST(S) FAILED".center(60) + "║")
    
    print("║" + " "*60 + "║")
    print("╚" + "═"*60 + "╝")
    print()


def generate_report(results: list) -> str:
    """Generate detailed text report"""
    report = []
    report.append("="*60)
    report.append("JARVIS v14 Ultimate - Phase 7 Test Report")
    report.append("="*60)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    for r in results:
        report.append("-"*60)
        report.append(f"Test: {r['name']}")
        report.append(f"File: {r['file']}")
        report.append(f"Status: {'PASS' if r['success'] else 'FAIL'}")
        report.append(f"Tests: {r['passed']}/{r['total']} passed")
        report.append(f"Time: {r['time']:.2f}s")
        report.append("")
        
        # Add last 20 lines of output
        if r['output']:
            lines = r['output'].strip().split('\n')
            if len(lines) > 20:
                report.append("... (truncated) ...")
                report.extend(lines[-20:])
            else:
                report.extend(lines)
        report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point"""
    print_banner()
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    all_results = []
    start_time = time.time()
    
    for name, test_file in TEST_FILES:
        result = run_test(name, test_file)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    print_results(all_results)
    
    # Generate report
    report = generate_report(all_results)
    report_file = PROJECT_ROOT / "tests" / "phase7_test_report.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"Could not save report: {e}")
    
    # Return exit code
    total_failed = sum(r['failed'] for r in all_results)
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
