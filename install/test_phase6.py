#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 6 Test Suite
========================================

Comprehensive tests for Installation System modules:
- Environment Detector
- Dependency Installer
- Configuration Generator
- First Run Setup
- Update System
- Repair System
- Uninstall System

Run: python test_phase6.py
"""

import sys
import os
import time
import json
import tempfile
import shutil
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
# ENVIRONMENT DETECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ENVIRONMENT DETECTOR TESTS")
print("=" * 60)


def test_env_detector_init():
    from install.detect import EnvironmentDetector
    detector = EnvironmentDetector()
    assert_not_none(detector)


def test_env_detector_detect():
    from install.detect import EnvironmentDetector
    detector = EnvironmentDetector()
    info = detector.detect_all()
    assert_not_none(info)
    assert_not_none(info.python_version)
    assert_true(len(info.checks) > 0)


def test_env_detector_report():
    from install.detect import EnvironmentDetector
    detector = EnvironmentDetector()
    detector.detect_all()
    report = detector.get_report()
    assert_true(len(report) > 0)
    assert_true("Python" in report)


def test_version_comparison():
    from install.detect import Platform, Architecture
    assert_true(Platform.TERMUX is not None)
    assert_true(Architecture.ARM64 is not None)


# Run Environment Detector tests
run_test("EnvironmentDetector initialization", test_env_detector_init)
run_test("EnvironmentDetector detect_all", test_env_detector_detect)
run_test("EnvironmentDetector get_report", test_env_detector_report)
run_test("Platform and Architecture enums", test_version_comparison)


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DEPENDENCY INSTALLER TESTS")
print("=" * 60)


def test_package_registry():
    from install.deps import PackageRegistry, PackageClass
    registry = PackageRegistry()
    
    # Check Class 0 packages
    class0 = registry.get_by_class(PackageClass.CLASS_0_GUARANTEED)
    assert_true(len(class0) > 0)


def test_package_registry_get():
    from install.deps import PackageRegistry
    registry = PackageRegistry()
    
    pkg = registry.get('click')
    assert_not_none(pkg)
    assert_equal(pkg.name, 'click')


def test_dependency_installer_init():
    from install.deps import DependencyInstaller
    installer = DependencyInstaller()
    assert_not_none(installer)


def test_dependency_installer_verify():
    from install.deps import DependencyInstaller
    installer = DependencyInstaller()
    
    success, missing = installer.verify_all()
    # Should return a tuple
    assert_true(isinstance(success, bool))
    assert_true(isinstance(missing, list))


def test_install_config():
    from install.deps import InstallConfig, InstallStrategy
    config = InstallConfig()
    assert_equal(config.strategy, InstallStrategy.STANDARD)
    assert_equal(config.timeout_per_package, 120)


# Run Dependency tests
run_test("PackageRegistry initialization", test_package_registry)
run_test("PackageRegistry get package", test_package_registry_get)
run_test("DependencyInstaller initialization", test_dependency_installer_init)
run_test("DependencyInstaller verify_all", test_dependency_installer_verify)
run_test("InstallConfig defaults", test_install_config)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CONFIGURATION GENERATOR TESTS")
print("=" * 60)


def test_config_generator_init():
    from install.config_gen import ConfigGenerator
    generator = ConfigGenerator()
    assert_not_none(generator)


def test_config_generator_default():
    from install.config_gen import ConfigGenerator
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    assert_not_none(config)
    assert_equal(config.general.app_name, "JARVIS")
    assert_not_none(config.ai)


def test_config_to_dict():
    from install.config_gen import ConfigGenerator
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    data = config.to_dict()
    assert_true('general' in data)
    assert_true('ai' in data)


def test_config_validation():
    from install.config_gen import ConfigGenerator
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    valid, errors = generator.validate(config)
    assert_true(valid)
    assert_equal(len(errors), 0)


def test_config_save_load():
    from install.config_gen import ConfigGenerator
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        success = generator.save(config, temp_path)
        assert_true(success)
        
        # Load
        loaded = generator.load(temp_path)
        assert_not_none(loaded)
        assert_equal(loaded.general.app_name, "JARVIS")
    finally:
        os.unlink(temp_path)


# Run Config tests
run_test("ConfigGenerator initialization", test_config_generator_init)
run_test("ConfigGenerator generate_default", test_config_generator_default)
run_test("JARVISConfig to_dict", test_config_to_dict)
run_test("ConfigGenerator validate", test_config_validation)
run_test("ConfigGenerator save and load", test_config_save_load)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST RUN SETUP TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FIRST RUN SETUP TESTS")
print("=" * 60)


def test_first_run_init():
    from install.first_run import FirstRunSetup
    setup = FirstRunSetup()
    assert_not_none(setup)


def test_first_run_features():
    from install.first_run import FirstRunSetup, Feature
    setup = FirstRunSetup()
    
    assert_true(len(setup.FEATURES) > 0)


def test_first_run_is_complete():
    from install.first_run import FirstRunSetup
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_file = Path(tmpdir) / ".setup_complete"
        setup = FirstRunSetup()
        setup._setup_file = setup_file
        
        assert_true(not setup.is_complete())


def test_setup_state():
    from install.first_run import SetupState, SetupStep, SetupStatus
    state = SetupState()
    
    assert_equal(state.current_step, SetupStep.WELCOME)
    assert_equal(state.status, SetupStatus.NOT_STARTED)


# Run First Run tests
run_test("FirstRunSetup initialization", test_first_run_init)
run_test("FirstRunSetup features list", test_first_run_features)
run_test("FirstRunSetup is_complete", test_first_run_is_complete)
run_test("SetupState defaults", test_setup_state)


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("UPDATE SYSTEM TESTS")
print("=" * 60)


def test_version_parse():
    from install.updater import Version
    
    v = Version.parse("1.2.3")
    assert_equal(v.major, 1)
    assert_equal(v.minor, 2)
    assert_equal(v.patch, 3)


def test_version_comparison():
    from install.updater import Version
    
    v1 = Version.parse("1.0.0")
    v2 = Version.parse("2.0.0")
    v3 = Version.parse("1.1.0")
    
    assert_true(v1 < v2)
    assert_true(v1 < v3)
    assert_true(v2 > v3)


def test_updater_init():
    from install.updater import Updater
    updater = Updater()
    assert_not_none(updater)
    assert_not_none(updater.current_version)


def test_updater_current_version():
    from install.updater import Updater
    updater = Updater(current_version="14.0.0")
    
    assert_equal(str(updater.current_version), "14.0.0")


# Run Update tests
run_test("Version parse", test_version_parse)
run_test("Version comparison", test_version_comparison)
run_test("Updater initialization", test_updater_init)
run_test("Updater current_version", test_updater_current_version)


# ═══════════════════════════════════════════════════════════════════════════════
# REPAIR SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("REPAIR SYSTEM TESTS")
print("=" * 60)


def test_repair_system_init():
    from install.repair import RepairSystem
    repair = RepairSystem()
    assert_not_none(repair)


def test_repair_diagnose():
    from install.repair import RepairSystem
    with tempfile.TemporaryDirectory() as tmpdir:
        repair = RepairSystem(jarvis_dir=Path(tmpdir))
        results = repair.diagnose()
        
        assert_true(len(results) > 0)


def test_repair_type():
    from install.repair import RepairType
    assert_true(RepairType.DEPENDENCIES is not None)
    assert_true(RepairType.CONFIGURATION is not None)
    assert_true(RepairType.FULL_RESET is not None)


def test_diagnostic_result():
    from install.repair import DiagnosticResult
    result = DiagnosticResult(name="Test", healthy=True)
    
    assert_equal(result.name, "Test")
    assert_true(result.healthy)


# Run Repair tests
run_test("RepairSystem initialization", test_repair_system_init)
run_test("RepairSystem diagnose", test_repair_diagnose)
run_test("RepairType enum", test_repair_type)
run_test("DiagnosticResult", test_diagnostic_result)


# ═══════════════════════════════════════════════════════════════════════════════
# UNINSTALL SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("UNINSTALL SYSTEM TESTS")
print("=" * 60)


def test_uninstaller_init():
    from install.uninstall import Uninstaller
    uninstaller = Uninstaller()
    assert_not_none(uninstaller)


def test_uninstall_scope():
    from install.uninstall import UninstallScope
    assert_true(UninstallScope.FULL is not None)
    assert_true(UninstallScope.KEEP_DATA is not None)


def test_uninstaller_get_size():
    from install.uninstall import Uninstaller
    with tempfile.TemporaryDirectory() as tmpdir:
        uninstaller = Uninstaller(jarvis_dir=Path(tmpdir))
        size = uninstaller.get_install_size()
        assert_true(size >= 0)


def test_uninstaller_list_backups():
    from install.uninstall import Uninstaller
    uninstaller = Uninstaller()
    backups = uninstaller.list_backups()
    assert_true(isinstance(backups, list))


# Run Uninstall tests
run_test("Uninstaller initialization", test_uninstaller_init)
run_test("UninstallScope enum", test_uninstall_scope)
run_test("Uninstaller get_install_size", test_uninstaller_get_size)
run_test("Uninstaller list_backups", test_uninstaller_list_backups)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("INTEGRATION TESTS")
print("=" * 60)


def test_detector_config_integration():
    from install.detect import EnvironmentDetector
    from install.config_gen import ConfigGenerator
    
    detector = EnvironmentDetector()
    info = detector.detect_all()
    
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    # Adjust config based on environment
    if info.available_memory_mb < 1024:
        config.logging.level = "WARNING"
    
    assert_not_none(config)


def test_repair_config_integration():
    from install.repair import RepairSystem, RepairType
    from install.config_gen import ConfigGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repair = RepairSystem(jarvis_dir=Path(tmpdir))
        
        # Repair configuration
        result = repair.repair(RepairType.CONFIGURATION, backup=False)
        
        assert_true(result.success or result.status.name == "NOT_NEEDED")


# Run Integration tests
run_test("Detector-Config integration", test_detector_config_integration)
run_test("Repair-Config integration", test_repair_config_integration)


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
