#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Installation System Module
=================================================

This module provides installation and maintenance functionality:

- detect: Environment detection and validation
- deps: Dependency installation with classification
- config_gen: Configuration generation
- first_run: First run setup wizard
- updater: Version checking and updates
- repair: System repair and diagnostics
- uninstall: Clean uninstallation

Memory Budget: < 50MB total for installation
"""

from install.detect import (
    EnvironmentDetector,
    EnvironmentInfo,
    CheckResult,
    CheckCategory,
    CheckStatus,
    Platform,
    Architecture,
)

from install.deps import (
    DependencyInstaller,
    PackageRegistry,
    Package,
    PackageClass,
    InstallStatus,
    InstallStrategy,
    InstallResult,
    InstallConfig,
)

from install.config_gen import (
    ConfigGenerator,
    JARVISConfig,
    ConfigMode,
    ConfigSection,
    GeneralConfig,
    AIConfig,
    SelfModConfig,
    InterfaceConfig,
    StorageConfig,
    NetworkConfig,
    LoggingConfig,
)

from install.first_run import (
    FirstRunSetup,
    SetupStep,
    SetupStatus,
    SetupState,
    Feature,
)

from install.updater import (
    Updater,
    Version,
    Release,
    UpdateStatus,
    UpdateSource,
    UpdateResult,
)

from install.repair import (
    RepairSystem,
    RepairType,
    RepairStatus,
    RepairResult,
    DiagnosticResult,
)

from install.uninstall import (
    Uninstaller,
    UninstallScope,
    UninstallResult,
)

__all__ = [
    # Detection
    'EnvironmentDetector',
    'EnvironmentInfo',
    'CheckResult',
    'CheckCategory',
    'CheckStatus',
    'Platform',
    'Architecture',
    
    # Dependencies
    'DependencyInstaller',
    'PackageRegistry',
    'Package',
    'PackageClass',
    'InstallStatus',
    'InstallStrategy',
    'InstallResult',
    'InstallConfig',
    
    # Configuration
    'ConfigGenerator',
    'JARVISConfig',
    'ConfigMode',
    'ConfigSection',
    'GeneralConfig',
    'AIConfig',
    'SelfModConfig',
    'InterfaceConfig',
    'StorageConfig',
    'NetworkConfig',
    'LoggingConfig',
    
    # First Run
    'FirstRunSetup',
    'SetupStep',
    'SetupStatus',
    'SetupState',
    'Feature',
    
    # Updater
    'Updater',
    'Version',
    'Release',
    'UpdateStatus',
    'UpdateSource',
    'UpdateResult',
    
    # Repair
    'RepairSystem',
    'RepairType',
    'RepairStatus',
    'RepairResult',
    'DiagnosticResult',
    
    # Uninstall
    'Uninstaller',
    'UninstallScope',
    'UninstallResult',
]

__version__ = '14.0.0'
