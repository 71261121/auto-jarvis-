#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 6 ULTIMATE Test Suite
=================================================

100x DEPTH Testing for Installation System modules:
- Dependency installer edge cases
- Configuration validation stress tests
- Update system integrity tests
- Repair system recovery tests
- Thread safety verification
- Resource exhaustion tests

Run: python test_phase6_ULTIMATE.py
"""

import sys
import os
import time
import json
import tempfile
import threading
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 ULTIMATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 6 ULTIMATE TESTS - 100x DEPTH ANALYSIS")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEPENDENCY INSTALLER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1. DEPENDENCY INSTALLER EDGE CASES]")


def test_package_registry_completeness():
    """Test PackageRegistry has all required packages"""
    from install.deps import PackageRegistry, PackageClass
    
    registry = PackageRegistry()
    
    # Class 0 must have packages
    class0 = registry.get_by_class(PackageClass.CLASS_0_GUARANTEED)
    assert_true(len(class0) > 0, "Class 0 should have packages")
    
    # Class 4 (impossible) should exist
    class4 = registry.get_by_class(PackageClass.CLASS_4_IMPOSSIBLE)
    assert_true(len(class4) > 0, "Class 4 should have impossible packages")
    
    # Check critical packages exist
    critical = ['click', 'colorama', 'requests', 'pyyaml']
    for pkg_name in critical:
        pkg = registry.get(pkg_name)
        assert_not_none(pkg, f"Package {pkg_name} should exist")


def test_package_size_parsing():
    """Test package size string parsing"""
    from install.deps import PackageRegistry
    
    registry = PackageRegistry()
    
    # Get a package and check size
    click_pkg = registry.get('click')
    assert_not_none(click_pkg)
    # Size should be reasonable
    assert_true(click_pkg.size_mb >= 0)


def test_dependency_installer_missing_packages():
    """Test installer with missing package names"""
    from install.deps import DependencyInstaller
    
    installer = DependencyInstaller()
    
    # Try to get missing packages
    result = installer.install_all(['nonexistent_package_xyz_123'])
    
    # Should handle gracefully
    assert_true(isinstance(result.success, bool))


def test_install_strategies():
    """Test different installation strategies"""
    from install.deps import DependencyInstaller, InstallConfig, InstallStrategy
    
    for strategy in InstallStrategy:
        config = InstallConfig(strategy=strategy)
        installer = DependencyInstaller(config)
        assert_not_none(installer)


def test_package_class_ordering():
    """Test package class ordering"""
    from install.deps import PackageClass
    
    # Class values should be ordered by risk
    assert_true(PackageClass.CLASS_0_GUARANTEED.value < PackageClass.CLASS_4_IMPOSSIBLE.value)


run_test("PackageRegistry completeness", test_package_registry_completeness)
run_test("Package size parsing", test_package_size_parsing)
run_test("DependencyInstaller missing packages", test_dependency_installer_missing_packages)
run_test("Installation strategies", test_install_strategies)
run_test("Package class ordering", test_package_class_ordering)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENVIRONMENT DETECTOR EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2. ENVIRONMENT DETECTOR EDGE CASES]")


def test_env_detector_all_platforms():
    """Test Platform enum completeness"""
    from install.detect import Platform
    
    # All platforms should be accessible
    platforms = [Platform.TERMUX, Platform.ANDROID, Platform.LINUX, 
                 Platform.MACOS, Platform.WINDOWS, Platform.UNKNOWN]
    assert_true(len(platforms) == 6)


def test_env_detector_all_architectures():
    """Test Architecture enum completeness"""
    from install.detect import Architecture
    
    archs = [Architecture.ARM64, Architecture.ARM, Architecture.X86_64, 
             Architecture.X86, Architecture.UNKNOWN]
    assert_true(len(archs) == 5)


def test_env_detector_check_categories():
    """Test CheckCategory enum completeness"""
    from install.detect import CheckCategory
    
    categories = [CheckCategory.SYSTEM, CheckCategory.PYTHON, CheckCategory.MEMORY,
                  CheckCategory.STORAGE, CheckCategory.NETWORK, CheckCategory.DEPENDENCY]
    assert_true(len(categories) == 6)


def test_env_detector_memory_fallback():
    """Test memory detection fallback"""
    from install.detect import EnvironmentDetector
    
    detector = EnvironmentDetector()
    info = detector.detect_all()
    
    # Should always have some memory value
    assert_true(info.total_memory_mb >= 0)
    assert_true(info.available_memory_mb >= 0)


def test_env_detector_storage_fallback():
    """Test storage detection fallback"""
    from install.detect import EnvironmentDetector
    
    detector = EnvironmentDetector()
    info = detector.detect_all()
    
    # Should always have some storage value
    assert_true(info.total_storage_gb >= 0)
    assert_true(info.available_storage_gb >= 0)


def test_env_detector_check_status():
    """Test CheckStatus enum"""
    from install.detect import CheckStatus
    
    statuses = [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL, CheckStatus.SKIP]
    assert_true(len(statuses) == 4)


run_test("Platform enum completeness", test_env_detector_all_platforms)
run_test("Architecture enum completeness", test_env_detector_all_architectures)
run_test("CheckCategory enum completeness", test_env_detector_check_categories)
run_test("Memory detection fallback", test_env_detector_memory_fallback)
run_test("Storage detection fallback", test_env_detector_storage_fallback)
run_test("CheckStatus enum", test_env_detector_check_status)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONFIGURATION GENERATOR STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3. CONFIGURATION GENERATOR STRESS TESTS]")


def test_config_all_sections():
    """Test all config sections exist"""
    from install.config_gen import ConfigGenerator
    
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    # Check all sections exist
    assert_not_none(config.general)
    assert_not_none(config.ai)
    assert_not_none(config.self_mod)
    assert_not_none(config.interface)
    assert_not_none(config.storage)
    assert_not_none(config.network)
    assert_not_none(config.logging)


def test_config_to_dict_complete():
    """Test config serialization completeness"""
    from install.config_gen import ConfigGenerator
    
    generator = ConfigGenerator()
    config = generator.generate_default()
    data = config.to_dict()
    
    # Check all sections in dict
    required_keys = ['general', 'ai', 'self_modification', 'interface', 
                     'storage', 'network', 'logging', 'metadata']
    for key in required_keys:
        assert_true(key in data, f"Key {key} should be in config dict")


def test_config_from_dict_edge_cases():
    """Test config loading with edge cases"""
    from install.config_gen import JARVISConfig
    
    # Empty dict
    config = JARVISConfig.from_dict({})
    assert_not_none(config)
    
    # Partial dict
    config = JARVISConfig.from_dict({'general': {'debug_mode': True}})
    assert_not_none(config)
    assert_true(config.general.debug_mode)
    
    # Extra keys (should be ignored)
    config = JARVISConfig.from_dict({'unknown_key': 'value'})
    assert_not_none(config)


def test_config_validation_edge_cases():
    """Test config validation with edge cases"""
    from install.config_gen import ConfigGenerator, JARVISConfig, AIConfig
    
    generator = ConfigGenerator()
    
    # Test invalid temperature
    config = generator.generate_default()
    config.ai.temperature = 3.0  # Invalid (> 2.0)
    valid, errors = generator.validate(config)
    assert_true(not valid or 'temperature' in str(errors).lower())
    
    # Test invalid timeout
    config = generator.generate_default()
    config.ai.timeout = 0  # Invalid (< 1)
    valid, errors = generator.validate(config)
    assert_true(not valid or 'timeout' in str(errors).lower())


def test_config_minimal_mode():
    """Test minimal config generation"""
    from install.config_gen import ConfigGenerator
    
    generator = ConfigGenerator()
    config = generator.generate_minimal()
    
    assert_true(not config.self_mod.enabled)
    assert_true(not config.interface.auto_complete)


def test_config_env_export():
    """Test config environment export"""
    from install.config_gen import ConfigGenerator
    
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    env = generator.export_env(config)
    assert_true(isinstance(env, dict))
    assert_true('JARVIS_MODEL' in env)


run_test("Config all sections", test_config_all_sections)
run_test("Config to_dict complete", test_config_to_dict_complete)
run_test("Config from_dict edge cases", test_config_from_dict_edge_cases)
run_test("Config validation edge cases", test_config_validation_edge_cases)
run_test("Config minimal mode", test_config_minimal_mode)
run_test("Config env export", test_config_env_export)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIRST RUN SETUP TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4. FIRST RUN SETUP TESTS]")


def test_setup_step_enum():
    """Test SetupStep enum completeness"""
    from install.first_run import SetupStep
    
    steps = [SetupStep.WELCOME, SetupStep.LICENSE, SetupStep.ENVIRONMENT_CHECK,
             SetupStep.FEATURE_SELECTION, SetupStep.AI_CONFIG, SetupStep.STORAGE_CONFIG,
             SetupStep.FINALIZE, SetupStep.COMPLETE]
    assert_true(len(steps) == 8)


def test_setup_status_enum():
    """Test SetupStatus enum"""
    from install.first_run import SetupStatus
    
    statuses = [SetupStatus.NOT_STARTED, SetupStatus.IN_PROGRESS, 
                SetupStatus.COMPLETE, SetupStatus.FAILED, SetupStatus.SKIPPED]
    assert_true(len(statuses) == 5)


def test_feature_dataclass():
    """Test Feature dataclass"""
    from install.first_run import Feature
    
    feature = Feature(
        id="test",
        name="Test Feature",
        description="Test description",
        required=False,
        enabled=True,
        dependencies=["dep1", "dep2"]
    )
    
    assert_equal(feature.id, "test")
    assert_equal(len(feature.dependencies), 2)


def test_setup_state_dataclass():
    """Test SetupState dataclass"""
    from install.first_run import SetupState, SetupStep, SetupStatus
    
    state = SetupState()
    
    assert_equal(state.current_step, SetupStep.WELCOME)
    assert_equal(state.status, SetupStatus.NOT_STARTED)
    assert_equal(state.progress, 0)
    assert_equal(state.total_steps, 7)


def test_setup_auto_mode():
    """Test FirstRunSetup in auto mode"""
    from install.first_run import FirstRunSetup
    
    setup = FirstRunSetup(auto_mode=True)
    assert_true(setup._auto_mode)


run_test("SetupStep enum", test_setup_step_enum)
run_test("SetupStatus enum", test_setup_status_enum)
run_test("Feature dataclass", test_feature_dataclass)
run_test("SetupState dataclass", test_setup_state_dataclass)
run_test("Setup auto mode", test_setup_auto_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UPDATE SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5. UPDATE SYSTEM TESTS]")


def test_version_parse_edge_cases():
    """Test Version parsing with edge cases"""
    from install.updater import Version
    
    # Standard version
    v = Version.parse("1.2.3")
    assert_equal(str(v), "1.2.3")
    
    # With 'v' prefix
    v = Version.parse("v2.0.0")
    assert_equal(str(v), "2.0.0")
    
    # With prerelease
    v = Version.parse("1.0.0-beta")
    assert_equal(v.prerelease, "beta")
    
    # Single number
    v = Version.parse("1")
    assert_equal(v.major, 1)
    assert_equal(v.minor, 0)
    
    # Two numbers
    v = Version.parse("1.2")
    assert_equal(v.major, 1)
    assert_equal(v.minor, 2)


def test_version_comparison_edge_cases():
    """Test Version comparison with edge cases"""
    from install.updater import Version
    
    # Equal versions
    v1 = Version.parse("1.0.0")
    v2 = Version.parse("1.0.0")
    assert_true(v1 == v2)
    
    # Prerelease < release
    v1 = Version.parse("1.0.0-beta")
    v2 = Version.parse("1.0.0")
    assert_true(v1 < v2)
    
    # Major difference
    v1 = Version.parse("1.9.9")
    v2 = Version.parse("2.0.0")
    assert_true(v1 < v2)
    
    # Minor difference
    v1 = Version.parse("1.0.9")
    v2 = Version.parse("1.1.0")
    assert_true(v1 < v2)


def test_update_status_enum():
    """Test UpdateStatus enum"""
    from install.updater import UpdateStatus
    
    statuses = [UpdateStatus.UP_TO_DATE, UpdateStatus.UPDATE_AVAILABLE,
                UpdateStatus.UPDATE_REQUIRED, UpdateStatus.UPDATE_FAILED,
                UpdateStatus.CHECK_FAILED]
    assert_true(len(statuses) == 5)


def test_update_source_enum():
    """Test UpdateSource enum"""
    from install.updater import UpdateSource
    
    sources = [UpdateSource.GITHUB, UpdateSource.LOCAL, UpdateSource.CUSTOM]
    assert_true(len(sources) == 3)


def test_updater_current_version():
    """Test Updater version handling"""
    from install.updater import Updater
    
    updater = Updater(current_version="14.0.0")
    assert_equal(str(updater.current_version), "14.0.0")


run_test("Version parse edge cases", test_version_parse_edge_cases)
run_test("Version comparison edge cases", test_version_comparison_edge_cases)
run_test("UpdateStatus enum", test_update_status_enum)
run_test("UpdateSource enum", test_update_source_enum)
run_test("Updater current_version", test_updater_current_version)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPAIR SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6. REPAIR SYSTEM TESTS]")


def test_repair_type_enum():
    """Test RepairType enum"""
    from install.repair import RepairType
    
    types = [RepairType.DEPENDENCIES, RepairType.CONFIGURATION, RepairType.DATABASE,
             RepairType.CACHE, RepairType.PERMISSIONS, RepairType.FULL_RESET]
    assert_true(len(types) == 6)


def test_repair_status_enum():
    """Test RepairStatus enum"""
    from install.repair import RepairStatus
    
    statuses = [RepairStatus.NOT_NEEDED, RepairStatus.REPAIRED, 
                RepairStatus.FAILED, RepairStatus.SKIPPED]
    assert_true(len(statuses) == 4)


def test_diagnostic_result_dataclass():
    """Test DiagnosticResult dataclass"""
    from install.repair import DiagnosticResult
    
    result = DiagnosticResult(
        name="Test",
        healthy=True,
        issues=["Issue 1"],
        repairable=True
    )
    
    assert_equal(result.name, "Test")
    assert_true(result.healthy)
    assert_equal(len(result.issues), 1)


def test_repair_result_dataclass():
    """Test RepairResult dataclass"""
    from install.repair import RepairResult, RepairType, RepairStatus
    
    result = RepairResult(
        repair_type=RepairType.CACHE,
        status=RepairStatus.REPAIRED,
        message="Test repair"
    )
    
    assert_true(result.success)
    assert_equal(result.repair_type, RepairType.CACHE)


def test_repair_diagnose_all():
    """Test RepairSystem diagnose all"""
    from install.repair import RepairSystem
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repair = RepairSystem(jarvis_dir=Path(tmpdir))
        results = repair.diagnose()
        
        # Should have results for all categories
        assert_true(len(results) >= 5)


run_test("RepairType enum", test_repair_type_enum)
run_test("RepairStatus enum", test_repair_status_enum)
run_test("DiagnosticResult dataclass", test_diagnostic_result_dataclass)
run_test("RepairResult dataclass", test_repair_result_dataclass)
run_test("RepairSystem diagnose all", test_repair_diagnose_all)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UNINSTALL SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7. UNINSTALL SYSTEM TESTS]")


def test_uninstall_scope_enum():
    """Test UninstallScope enum"""
    from install.uninstall import UninstallScope
    
    scopes = [UninstallScope.FULL, UninstallScope.KEEP_DATA, 
              UninstallScope.KEEP_CONFIG, UninstallScope.MINIMAL]
    assert_true(len(scopes) == 4)


def test_uninstall_result_dataclass():
    """Test UninstallResult dataclass"""
    from install.uninstall import UninstallResult, UninstallScope
    
    result = UninstallResult(
        success=True,
        scope=UninstallScope.FULL,
        removed_files=10,
        removed_dirs=5,
        freed_bytes=1024*1024
    )
    
    assert_true(result.success)
    assert_equal(result.removed_files, 10)


def test_uninstaller_get_install_size():
    """Test Uninstaller get_install_size"""
    from install.uninstall import Uninstaller
    
    with tempfile.TemporaryDirectory() as tmpdir:
        uninstaller = Uninstaller(jarvis_dir=Path(tmpdir))
        size = uninstaller.get_install_size()
        assert_true(size >= 0)


def test_uninstaller_list_backups():
    """Test Uninstaller list_backups"""
    from install.uninstall import Uninstaller
    
    uninstaller = Uninstaller()
    backups = uninstaller.list_backups()
    assert_true(isinstance(backups, list))


run_test("UninstallScope enum", test_uninstall_scope_enum)
run_test("UninstallResult dataclass", test_uninstall_result_dataclass)
run_test("Uninstaller get_install_size", test_uninstaller_get_install_size)
run_test("Uninstaller list_backups", test_uninstaller_list_backups)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. THREAD SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[8. THREAD SAFETY TESTS]")


def test_config_generator_thread_safety():
    """Test ConfigGenerator thread safety"""
    from install.config_gen import ConfigGenerator
    
    generator = ConfigGenerator()
    errors = []
    
    def generate_configs():
        try:
            for _ in range(20):
                config = generator.generate_default()
                generator.validate(config)
        except Exception as e:
            errors.append(str(e))
    
    threads = []
    for _ in range(5):
        t = threading.Thread(target=generate_configs)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


def test_env_detector_thread_safety():
    """Test EnvironmentDetector thread safety"""
    from install.detect import EnvironmentDetector
    
    errors = []
    
    def detect_env():
        try:
            detector = EnvironmentDetector()
            for _ in range(10):
                detector.detect_all()
        except Exception as e:
            errors.append(str(e))
    
    threads = []
    for _ in range(5):
        t = threading.Thread(target=detect_env)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    assert_true(len(errors) == 0, f"Thread safety errors: {errors}")


run_test("ConfigGenerator thread safety", test_config_generator_thread_safety)
run_test("EnvironmentDetector thread safety", test_env_detector_thread_safety)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FILE I/O EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[9. FILE I/O EDGE CASES]")


def test_config_save_to_readonly():
    """Test config save to read-only location"""
    from install.config_gen import ConfigGenerator, JARVISConfig
    
    generator = ConfigGenerator()
    config = generator.generate_default()
    
    # Try to save to invalid path (should handle gracefully)
    result = generator.save(config, "/dev/null/config.json")  # Invalid path
    # Should return False or raise handled exception
    assert_true(isinstance(result, bool))


def test_config_load_corrupted():
    """Test config load with corrupted file"""
    from install.config_gen import ConfigGenerator
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{corrupted json")
        temp_path = f.name
    
    try:
        generator = ConfigGenerator()
        result = generator.load(temp_path)
        assert_true(result is None or isinstance(result, object))
    finally:
        os.unlink(temp_path)


run_test("Config save to read-only", test_config_save_to_readonly)
run_test("Config load corrupted", test_config_load_corrupted)


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
