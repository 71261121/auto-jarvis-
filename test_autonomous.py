#!/usr/bin/env python3
"""
JARVIS Autonomous Engine - Quick Test
======================================

Run this to verify autonomous engine is working.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import correctly"""
    print("=" * 60)
    print("1. Testing Imports...")
    print("=" * 60)
    
    try:
        from core.autonomous import AutonomousEngine
        print("  ✓ AutonomousEngine imported")
        
        from core.autonomous.intent_detector import IntentDetector, IntentType
        print("  ✓ IntentDetector imported")
        
        from core.autonomous.executor import AutonomousExecutor
        print("  ✓ AutonomousExecutor imported")
        
        from core.autonomous.safety_manager import SafetyManager
        print("  ✓ SafetyManager imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_intent_detection():
    """Test intent detection"""
    print("\n" + "=" * 60)
    print("2. Testing Intent Detection...")
    print("=" * 60)
    
    from core.autonomous.intent_detector import IntentDetector
    
    detector = IntentDetector(project_root=os.getcwd())
    
    tests = [
        ("read main.py", "READ_FILE"),
        ("list core/", "LIST_DIR"),
        ("modify main.py to add debug", "MODIFY_FILE"),
        ("create utils.py with helpers", "CREATE_FILE"),
        ("search for import", "SEARCH_FILES"),
        ("run python test.py", "EXECUTE_CMD"),
        ("install requests", "INSTALL_PKG"),
        ("What is Python?", "CHAT"),
    ]
    
    passed = 0
    for user_input, expected in tests:
        intent = detector.detect(user_input)
        if intent.intent_type.name == expected:
            print(f"  ✓ '{user_input[:30]}...' → {expected}")
            passed += 1
        else:
            print(f"  ✗ '{user_input[:30]}...' → Expected {expected}, got {intent.intent_type.name}")
    
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_direct_operations():
    """Test direct operations"""
    print("\n" + "=" * 60)
    print("3. Testing Direct Operations...")
    print("=" * 60)
    
    from core.autonomous.engine import AutonomousEngine
    
    engine = AutonomousEngine(project_root=os.getcwd())
    
    tests = [
        ("read main.py", "READ_FILE"),
        ("list core/", "LIST_DIR"),
        ("search for def", "SEARCH_FILES"),
        ("help", "HELP"),
        ("status", "STATUS"),
    ]
    
    passed = 0
    for cmd, expected_intent in tests:
        result = engine.process(cmd)
        if result.success:
            print(f"  ✓ '{cmd}' succeeded")
            passed += 1
        else:
            print(f"  ✗ '{cmd}' failed: {result.error}")
    
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_safety():
    """Test safety manager"""
    print("\n" + "=" * 60)
    print("4. Testing Safety Manager...")
    print("=" * 60)
    
    from core.autonomous.safety_manager import SafetyManager
    
    safety = SafetyManager()
    
    # Test dangerous command blocking
    result = safety.check_command("rm -rf /")
    if not result.allowed:
        print("  ✓ Blocked dangerous command: 'rm -rf /'")
        cmd_safe = True
    else:
        print("  ✗ Failed to block: 'rm -rf /'")
        cmd_safe = False
    
    # Test protected file detection
    result = safety.check_file_write(".env")
    if result.level.name in ("WARNING", "DANGEROUS"):
        print("  ✓ Protected file detected: '.env'")
        file_safe = True
    else:
        print("  ✗ Failed to detect protected file: '.env'")
        file_safe = False
    
    # Test delete confirmation
    result = safety.check_file_delete("test.py")
    if result.requires_confirmation:
        print("  ✓ Delete requires confirmation")
        del_safe = True
    else:
        print("  ✗ Delete doesn't require confirmation")
        del_safe = False
    
    return cmd_safe and file_safe and del_safe


def test_jarvis_integration():
    """Test JARVIS integration"""
    print("\n" + "=" * 60)
    print("5. Testing JARVIS Integration...")
    print("=" * 60)
    
    try:
        from main import JARVIS, AUTONOMOUS_AVAILABLE
        
        print(f"  ✓ AUTONOMOUS_AVAILABLE = {AUTONOMOUS_AVAILABLE}")
        
        if AUTONOMOUS_AVAILABLE:
            # Create JARVIS instance
            jarvis = JARVIS(debug=False)
            
            if jarvis.autonomous_engine:
                print("  ✓ autonomous_engine initialized")
                
                # Test through JARVIS
                result = jarvis.autonomous_engine.process("list .")
                if result.success:
                    print("  ✓ Operations working through JARVIS")
                    return True
                else:
                    print(f"  ✗ Operation failed: {result.error}")
                    return False
            else:
                print("  ✗ autonomous_engine not initialized")
                return False
        else:
            print("  ✗ AUTONOMOUS_AVAILABLE is False")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " JARVIS Autonomous Engine - Quick Test ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Intent Detection", test_intent_detection()))
    results.append(("Direct Operations", test_direct_operations()))
    results.append(("Safety Manager", test_safety()))
    results.append(("JARVIS Integration", test_jarvis_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print()
    if passed == total:
        print("🎉 ALL TESTS PASSED! Autonomous engine is fully operational!")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some issues detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
