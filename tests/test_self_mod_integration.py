#!/usr/bin/env python3
"""
JARVIS Self-Modification Integration Test
=========================================

This script tests if the bridge is properly integrated and working.

Run: python tests/test_self_mod_integration.py
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_bridge_import():
    """Test 1: Can import the bridge module"""
    print("\n📋 Test 1: Bridge Import")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge, ModificationResult
        print("✓ SelfModificationBridge imported successfully")
        print("✓ ModificationResult imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_bridge_initialization():
    """Test 2: Can initialize bridge with mock JARVIS"""
    print("\n📋 Test 2: Bridge Initialization")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge
        
        # Create mock JARVIS instance
        class MockJarvis:
            def __init__(self):
                self.code_analyzer = None
                self.backup_manager = None
                self.improvement_engine = None
                self.sandbox = None
        
        mock_jarvis = MockJarvis()
        bridge = SelfModificationBridge(mock_jarvis)
        
        print("✓ Bridge initialized successfully")
        print(f"✓ Project root: {bridge.project_root}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_system_prompt():
    """Test 3: System prompt contains tool definitions"""
    print("\n📋 Test 3: System Prompt Generation")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge
        
        class MockJarvis:
            code_analyzer = None
            backup_manager = None
            improvement_engine = None
            sandbox = None
        
        bridge = SelfModificationBridge(MockJarvis())
        prompt = bridge.get_system_prompt()
        
        checks = [
            ('READ' in prompt, "READ command"),
            ('MODIFY' in prompt, "MODIFY command"),
            ('CREATE' in prompt, "CREATE command"),
            ('DELETE' in prompt, "DELETE command"),
            ('SAFETY RULES' in prompt, "Safety rules section"),
            ('Project root' in prompt, "Project context"),
        ]
        
        all_passed = True
        for check, name in checks:
            if check:
                print(f"✓ {name} present")
            else:
                print(f"✗ {name} missing")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"✗ System prompt test failed: {e}")
        return False


def test_command_parsing():
    """Test 4: Can parse AI commands from responses"""
    print("\n📋 Test 4: Command Parsing")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge
        
        class MockJarvis:
            code_analyzer = None
            backup_manager = None
            improvement_engine = None
            sandbox = None
        
        bridge = SelfModificationBridge(MockJarvis())
        
        # Test response with multiple commands
        test_response = """
Let me first read the file:
[READ:main.py]

Now I'll modify it:
[MODIFY:main.py]
```python
def new_function():
    return "Hello"
```

And create a new file:
[CREATE:utils.py]
```python
def helper():
    pass
```
"""
        
        commands = bridge.parse_commands(test_response)
        
        print(f"✓ Found {len(commands)} commands")
        
        for cmd in commands:
            print(f"  - {cmd['command']}: {cmd['target']}")
            if 'code' in cmd:
                print(f"    Code: {len(cmd['code'])} chars")
        
        if len(commands) == 3:
            print("✓ All commands parsed correctly")
            return True
        else:
            print(f"✗ Expected 3 commands, got {len(commands)}")
            return False
            
    except Exception as e:
        print(f"✗ Command parsing test failed: {e}")
        return False


def test_protected_files():
    """Test 5: Protected files are detected"""
    print("\n📋 Test 5: Protected File Detection")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge
        
        class MockJarvis:
            code_analyzer = None
            backup_manager = None
            improvement_engine = None
            sandbox = None
        
        bridge = SelfModificationBridge(MockJarvis())
        
        protected = [
            '.env',
            'credentials.json',
            'secrets.yaml',
        ]
        
        not_protected = [
            'main.py',
            'utils.py',
            'core/ai/client.py',
        ]
        
        all_passed = True
        
        for f in protected:
            if bridge._is_protected(Path(f)):
                print(f"✓ {f} is protected")
            else:
                print(f"✗ {f} should be protected")
                all_passed = False
        
        for f in not_protected:
            if not bridge._is_protected(Path(f)):
                print(f"✓ {f} is not protected")
            else:
                print(f"✗ {f} should not be protected")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"✗ Protected file test failed: {e}")
        return False


def test_file_operations():
    """Test 6: File read/write operations work"""
    print("\n📋 Test 6: File Operations")
    print("-" * 40)
    
    try:
        from core.self_mod.bridge import SelfModificationBridge
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        class MockJarvis:
            code_analyzer = None
            backup_manager = None
            improvement_engine = None
            sandbox = None
        
        bridge = SelfModificationBridge(MockJarvis())
        bridge.project_root = temp_dir
        
        # Test CREATE
        create_result = bridge._execute_create(
            temp_dir / "test.py",
            "def hello():\n    print('Hello')\n"
        )
        
        if create_result.success:
            print("✓ CREATE operation successful")
        else:
            print(f"✗ CREATE failed: {create_result.error}")
            return False
        
        # Test READ
        read_result = bridge._execute_read(temp_dir / "test.py")
        
        if read_result.success:
            print("✓ READ operation successful")
            print(f"  Lines: {read_result.details.get('lines', 0)}")
        else:
            print(f"✗ READ failed: {read_result.error}")
            return False
        
        # Test MODIFY
        modify_result = bridge._execute_modify(
            temp_dir / "test.py",
            "def goodbye():\n    print('Goodbye')\n"
        )
        
        if modify_result.success:
            print("✓ MODIFY operation successful")
        else:
            print(f"✗ MODIFY failed: {modify_result.error}")
            return False
        
        # Test DELETE
        delete_result = bridge._execute_delete(temp_dir / "test.py")
        
        if delete_result.success:
            print("✓ DELETE operation successful")
        else:
            print(f"✗ DELETE failed: {delete_result.error}")
            return False
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False


def test_main_py_integration():
    """Test 7: main.py has bridge integration"""
    print("\n📋 Test 7: main.py Integration Check")
    print("-" * 40)
    
    try:
        main_py_path = PROJECT_ROOT / "main.py"
        
        if not main_py_path.exists():
            print("✗ main.py not found")
            return False
        
        content = main_py_path.read_text()
        
        checks = [
            ("SelfModificationBridge" in content, "Bridge import"),
            ("mod_bridge" in content, "Bridge attribute"),
            ("process_response" in content, "Response processing"),
            ("get_system_prompt" in content, "System prompt usage"),
        ]
        
        all_passed = True
        for check, name in checks:
            if check:
                print(f"✓ {name} present in main.py")
            else:
                print(f"✗ {name} missing in main.py")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"✗ main.py integration check failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("JARVIS Self-Modification Integration Tests")
    print("=" * 60)
    
    tests = [
        test_bridge_import,
        test_bridge_initialization,
        test_system_prompt,
        test_command_parsing,
        test_protected_files,
        test_file_operations,
        test_main_py_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nJARVIS is now capable of:")
        print("  • Reading files")
        print("  • Modifying files (with auto-backup)")
        print("  • Creating new files")
        print("  • Deleting files")
        print("  • Analyzing code")
        print("  • Searching codebase")
        print("\nThe AI can now modify itself!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease check the errors above and fix them.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
