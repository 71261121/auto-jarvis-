#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Environment Detector
===========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Termux environment detection
- Python version compatibility
- Memory availability checking
- Storage space analysis
- Network connectivity testing

Features:
- Termux detection and version
- Python version check (3.9+)
- RAM availability analysis
- Storage space check
- Network connectivity test
- CPU architecture detection
- Battery status check
- Device capabilities assessment

Memory Impact: < 5MB for detection
"""

import os
import sys
import re
import platform
import subprocess
import shutil
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class Platform(Enum):
    """Detected platform types"""
    TERMUX = auto()
    ANDROID = auto()
    LINUX = auto()
    MACOS = auto()
    WINDOWS = auto()
    UNKNOWN = auto()


class Architecture(Enum):
    """CPU architectures"""
    ARM64 = auto()
    ARM = auto()
    X86_64 = auto()
    X86 = auto()
    UNKNOWN = auto()


class CheckStatus(Enum):
    """Status of environment checks"""
    PASS = auto()
    WARN = auto()
    FAIL = auto()
    SKIP = auto()


class CheckCategory(Enum):
    """Categories of checks"""
    SYSTEM = auto()
    PYTHON = auto()
    MEMORY = auto()
    STORAGE = auto()
    NETWORK = auto()
    DEPENDENCY = auto()


@dataclass
class CheckResult:
    """Result of an environment check"""
    name: str
    category: CheckCategory
    status: CheckStatus
    message: str = ""
    value: Any = None
    required: bool = True
    suggestion: str = ""
    
    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN)


@dataclass
class EnvironmentInfo:
    """
    Complete environment information.
    
    Contains all detected information about the
    current system environment.
    """
    platform: Platform = Platform.UNKNOWN
    architecture: Architecture = Architecture.UNKNOWN
    is_termux: bool = False
    termux_version: str = ""
    
    # Python
    python_version: str = ""
    python_path: str = ""
    pip_version: str = ""
    
    # System
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""
    username: str = ""
    
    # Hardware
    cpu_count: int = 0
    cpu_arch: str = ""
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    total_storage_gb: float = 0.0
    available_storage_gb: float = 0.0
    
    # Network
    has_network: bool = False
    has_wifi: bool = False
    has_mobile_data: bool = False
    
    # Battery (Termux)
    battery_level: int = 0
    battery_charging: bool = False
    
    # Checks
    checks: List[CheckResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'platform': self.platform.name,
            'architecture': self.architecture.name,
            'is_termux': self.is_termux,
            'termux_version': self.termux_version,
            'python_version': self.python_version,
            'python_path': self.python_path,
            'cpu_count': self.cpu_count,
            'total_memory_mb': self.total_memory_mb,
            'available_memory_mb': self.available_memory_mb,
            'total_storage_gb': self.total_storage_gb,
            'available_storage_gb': self.available_storage_gb,
            'has_network': self.has_network,
            'battery_level': self.battery_level,
        }
    
    @property
    def all_passed(self) -> bool:
        """Check if all required checks passed"""
        return all(c.passed or not c.required for c in self.checks)
    
    @property
    def warnings(self) -> List[CheckResult]:
        """Get warning checks"""
        return [c for c in self.checks if c.status == CheckStatus.WARN]
    
    @property
    def failures(self) -> List[CheckResult]:
        """Get failed checks"""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentDetector:
    """
    Ultra-Advanced Environment Detection System.
    
    Features:
    - Termux detection
    - Python version check
    - Memory analysis
    - Storage analysis
    - Network testing
    - Battery status
    - Comprehensive reporting
    
    Memory Budget: < 5MB
    
    Usage:
        detector = EnvironmentDetector()
        info = detector.detect_all()
        
        if info.all_passed:
            print("Environment is ready!")
        else:
            for failure in info.failures:
                print(f"Failed: {failure.name} - {failure.message}")
    """
    
    # Minimum requirements
    MIN_PYTHON_VERSION = (3, 9)
    MIN_MEMORY_MB = 512
    MIN_STORAGE_GB = 0.5
    RECOMMENDED_MEMORY_MB = 2048
    
    # Network test endpoints
    NETWORK_TEST_URLS = [
        "https://api.openrouter.ai",
        "https://api.github.com",
        "https://google.com",
    ]
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Environment Detector.
        
        Args:
            verbose: Enable verbose logging
        """
        self._verbose = verbose
        self._info = EnvironmentInfo()
    
    def detect_all(self) -> EnvironmentInfo:
        """
        Run all detection checks.
        
        Returns:
            EnvironmentInfo with all results
        """
        logger.info("Starting environment detection...")
        
        # Detect platform
        self._detect_platform()
        
        # Detect Python
        self._detect_python()
        
        # Detect hardware
        self._detect_hardware()
        
        # Detect storage
        self._detect_storage()
        
        # Detect network
        self._detect_network()
        
        # Termux-specific checks
        if self._info.is_termux:
            self._detect_termux_specifics()
        
        # Run validation checks
        self._run_checks()
        
        logger.info(f"Environment detection complete. Platform: {self._info.platform.name}")
        
        return self._info
    
    # ─────────────────────────────────────────────────────────────────────────
    # Platform Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_platform(self):
        """Detect operating platform"""
        # Check for Termux
        if 'TERMUX_VERSION' in os.environ:
            self._info.is_termux = True
            self._info.platform = Platform.TERMUX
            self._info.termux_version = os.environ.get('TERMUX_VERSION', '')
            self._info.os_name = 'Android (Termux)'
            logger.debug(f"Termux detected: v{self._info.termux_version}")
        elif 'ANDROID_ROOT' in os.environ:
            self._info.platform = Platform.ANDROID
            self._info.os_name = 'Android'
        elif sys.platform == 'linux':
            self._info.platform = Platform.LINUX
            self._info.os_name = 'Linux'
            self._detect_linux_distro()
        elif sys.platform == 'darwin':
            self._info.platform = Platform.MACOS
            self._info.os_name = 'macOS'
        elif sys.platform == 'win32':
            self._info.platform = Platform.WINDOWS
            self._info.os_name = 'Windows'
        else:
            self._info.platform = Platform.UNKNOWN
            self._info.os_name = 'Unknown'
        
        # Detect architecture
        self._detect_architecture()
        
        # Basic system info
        self._info.hostname = platform.node()
        self._info.username = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))
    
    def _detect_architecture(self):
        """Detect CPU architecture"""
        machine = platform.machine().lower()
        
        if machine in ('aarch64', 'arm64'):
            self._info.architecture = Architecture.ARM64
            self._info.cpu_arch = 'arm64'
        elif machine in ('arm', 'armv7l', 'armv8l'):
            self._info.architecture = Architecture.ARM
            self._info.cpu_arch = 'arm'
        elif machine in ('x86_64', 'amd64'):
            self._info.architecture = Architecture.X86_64
            self._info.cpu_arch = 'x86_64'
        elif machine in ('x86', 'i386', 'i686'):
            self._info.architecture = Architecture.X86
            self._info.cpu_arch = 'x86'
        else:
            self._info.architecture = Architecture.UNKNOWN
            self._info.cpu_arch = machine
    
    def _detect_linux_distro(self):
        """Detect Linux distribution"""
        try:
            # Try os-release
            release_file = Path('/etc/os-release')
            if release_file.exists():
                content = release_file.read_text()
                for line in content.split('\n'):
                    if line.startswith('PRETTY_NAME='):
                        self._info.os_name = line.split('=')[1].strip('"')
                        break
        except:
            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Python Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_python(self):
        """Detect Python installation"""
        # Version
        self._info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self._info.python_path = sys.executable
        
        # Pip version
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                match = re.search(r'pip\s+(\d+\.\d+\.?\d*)', result.stdout)
                if match:
                    self._info.pip_version = match.group(1)
        except:
            pass
        
        logger.debug(f"Python {self._info.python_version} at {self._info.python_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Hardware Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_hardware(self):
        """Detect hardware resources"""
        # CPU count
        self._info.cpu_count = os.cpu_count() or 1
        
        # Memory detection
        self._detect_memory()
    
    def _detect_memory(self):
        """Detect available memory"""
        try:
            # Try psutil first
            import psutil
            mem = psutil.virtual_memory()
            self._info.total_memory_mb = mem.total // (1024 * 1024)
            self._info.available_memory_mb = mem.available // (1024 * 1024)
            logger.debug(f"Memory: {self._info.total_memory_mb}MB total, {self._info.available_memory_mb}MB available")
            return
        except ImportError:
            pass
        
        # Fallback methods
        if self._info.is_termux:
            self._detect_memory_termux()
        elif sys.platform == 'linux':
            self._detect_memory_linux()
        else:
            # Default estimate
            self._info.total_memory_mb = 2048
            self._info.available_memory_mb = 1024
    
    def _detect_memory_termux(self):
        """Detect memory on Termux"""
        try:
            # Try termux-api
            result = subprocess.run(
                ['termux-memory-info'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse output
                data = json.loads(result.stdout)
                self._info.total_memory_mb = data.get('total', 0) // (1024 * 1024)
                self._info.available_memory_mb = data.get('available', 0) // (1024 * 1024)
                return
        except:
            pass
        
        # Fallback to /proc/meminfo
        self._detect_memory_linux()
    
    def _detect_memory_linux(self):
        """Detect memory on Linux"""
        try:
            with open('/proc/meminfo', 'r') as f:
                content = f.read()
            
            for line in content.split('\n'):
                if line.startswith('MemTotal:'):
                    self._info.total_memory_mb = int(line.split()[1]) // 1024
                elif line.startswith('MemAvailable:'):
                    self._info.available_memory_mb = int(line.split()[1]) // 1024
            
            if self._info.available_memory_mb == 0:
                # Calculate from free memory
                for line in content.split('\n'):
                    if line.startswith('MemFree:'):
                        self._info.available_memory_mb = int(line.split()[1]) // 1024
        except:
            # Default for unknown
            self._info.total_memory_mb = 2048
            self._info.available_memory_mb = 1024
    
    # ─────────────────────────────────────────────────────────────────────────
    # Storage Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_storage(self):
        """Detect available storage"""
        try:
            # Get storage for home directory
            home = Path.home()
            usage = shutil.disk_usage(home)
            
            self._info.total_storage_gb = usage.total / (1024 ** 3)
            self._info.available_storage_gb = usage.free / (1024 ** 3)
            
            logger.debug(f"Storage: {self._info.total_storage_gb:.1f}GB total, {self._info.available_storage_gb:.1f}GB available")
        except:
            self._info.total_storage_gb = 16.0
            self._info.available_storage_gb = 8.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Network Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_network(self):
        """Detect network connectivity"""
        self._info.has_network = self._test_network()
        
        if self._info.is_termux:
            self._detect_termux_network()
    
    def _test_network(self, timeout: float = 5.0) -> bool:
        """Test network connectivity"""
        import urllib.request
        import urllib.error
        
        for url in self.NETWORK_TEST_URLS:
            try:
                urllib.request.urlopen(url, timeout=timeout)
                logger.debug(f"Network test passed: {url}")
                return True
            except:
                continue
        
        logger.debug("Network test failed")
        return False
    
    def _detect_termux_network(self):
        """Detect Termux network details"""
        try:
            result = subprocess.run(
                ['termux-telephony-deviceinfo'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Check for mobile data
                self._info.has_mobile_data = data.get('data_state', '') == 'connected'
        except:
            pass
        
        try:
            result = subprocess.run(
                ['termux-wifi-connectioninfo'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self._info.has_wifi = True
        except:
            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Termux-Specific Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_termux_specifics(self):
        """Detect Termux-specific information"""
        # Battery status
        self._detect_battery()
        
        # Android version
        self._info.os_version = os.environ.get('ANDROID_VERSION', '')
    
    def _detect_battery(self):
        """Detect battery status on Termux"""
        try:
            result = subprocess.run(
                ['termux-battery-status'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                self._info.battery_level = data.get('percentage', 0)
                self._info.battery_charging = data.get('status', '') == 'CHARGING'
        except:
            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation Checks
    # ─────────────────────────────────────────────────────────────────────────
    
    def _run_checks(self):
        """Run all validation checks"""
        self._info.checks = []
        
        # Python version check
        self._check_python_version()
        
        # Memory check
        self._check_memory()
        
        # Storage check
        self._check_storage()
        
        # Network check
        self._check_network()
        
        # Architecture check
        self._check_architecture()
        
        # Pip check
        self._check_pip()
    
    def _check_python_version(self):
        """Check Python version requirement"""
        version = sys.version_info[:2]
        required = self.MIN_PYTHON_VERSION
        
        if version >= required:
            self._info.checks.append(CheckResult(
                name="Python Version",
                category=CheckCategory.PYTHON,
                status=CheckStatus.PASS,
                message=f"Python {self._info.python_version} meets requirement (>= {required[0]}.{required[1]})",
                value=version,
            ))
        else:
            self._info.checks.append(CheckResult(
                name="Python Version",
                category=CheckCategory.PYTHON,
                status=CheckStatus.FAIL,
                message=f"Python {self._info.python_version} does not meet requirement (>= {required[0]}.{required[1]})",
                value=version,
                suggestion=f"Upgrade to Python {required[0]}.{required[1]} or higher",
            ))
    
    def _check_memory(self):
        """Check available memory"""
        available = self._info.available_memory_mb
        minimum = self.MIN_MEMORY_MB
        recommended = self.RECOMMENDED_MEMORY_MB
        
        if available >= recommended:
            self._info.checks.append(CheckResult(
                name="Available Memory",
                category=CheckCategory.MEMORY,
                status=CheckStatus.PASS,
                message=f"{available}MB available (recommended: {recommended}MB)",
                value=available,
            ))
        elif available >= minimum:
            self._info.checks.append(CheckResult(
                name="Available Memory",
                category=CheckCategory.MEMORY,
                status=CheckStatus.WARN,
                message=f"{available}MB available (recommended: {recommended}MB)",
                value=available,
                suggestion="Close other applications for better performance",
            ))
        else:
            self._info.checks.append(CheckResult(
                name="Available Memory",
                category=CheckCategory.MEMORY,
                status=CheckStatus.FAIL,
                message=f"Only {available}MB available (minimum: {minimum}MB)",
                value=available,
                suggestion="Free up memory before running JARVIS",
            ))
    
    def _check_storage(self):
        """Check available storage"""
        available = self._info.available_storage_gb
        minimum = self.MIN_STORAGE_GB
        
        if available >= minimum:
            self._info.checks.append(CheckResult(
                name="Available Storage",
                category=CheckCategory.STORAGE,
                status=CheckStatus.PASS,
                message=f"{available:.1f}GB available",
                value=available,
            ))
        else:
            self._info.checks.append(CheckResult(
                name="Available Storage",
                category=CheckCategory.STORAGE,
                status=CheckStatus.FAIL,
                message=f"Only {available:.1f}GB available (minimum: {minimum}GB)",
                value=available,
                suggestion="Free up storage space",
            ))
    
    def _check_network(self):
        """Check network connectivity"""
        if self._info.has_network:
            self._info.checks.append(CheckResult(
                name="Network Connectivity",
                category=CheckCategory.NETWORK,
                status=CheckStatus.PASS,
                message="Network connection available",
                value=True,
            ))
        else:
            self._info.checks.append(CheckResult(
                name="Network Connectivity",
                category=CheckCategory.NETWORK,
                status=CheckStatus.WARN,
                message="No network connection",
                value=False,
                required=False,  # Not strictly required
                suggestion="Connect to network for AI features",
            ))
    
    def _check_architecture(self):
        """Check CPU architecture compatibility"""
        if self._info.architecture in (Architecture.ARM64, Architecture.ARM, Architecture.X86_64):
            self._info.checks.append(CheckResult(
                name="CPU Architecture",
                category=CheckCategory.SYSTEM,
                status=CheckStatus.PASS,
                message=f"{self._info.cpu_arch} architecture supported",
                value=self._info.architecture.name,
            ))
        else:
            self._info.checks.append(CheckResult(
                name="CPU Architecture",
                category=CheckCategory.SYSTEM,
                status=CheckStatus.WARN,
                message=f"{self._info.cpu_arch} architecture may have compatibility issues",
                value=self._info.architecture.name,
                required=False,
            ))
    
    def _check_pip(self):
        """Check pip availability"""
        if self._info.pip_version:
            self._info.checks.append(CheckResult(
                name="Pip Installation",
                category=CheckCategory.DEPENDENCY,
                status=CheckStatus.PASS,
                message=f"pip {self._info.pip_version} available",
                value=self._info.pip_version,
            ))
        else:
            self._info.checks.append(CheckResult(
                name="Pip Installation",
                category=CheckCategory.DEPENDENCY,
                status=CheckStatus.WARN,
                message="pip version could not be determined",
                required=False,
            ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_report(self) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 60,
            "JARVIS Environment Report",
            "=" * 60,
            "",
            f"Platform: {self._info.os_name} ({self._info.platform.name})",
            f"Architecture: {self._info.cpu_arch}",
            f"Hostname: {self._info.hostname}",
            "",
            "Python:",
            f"  Version: {self._info.python_version}",
            f"  Path: {self._info.python_path}",
            f"  Pip: {self._info.pip_version or 'Unknown'}",
            "",
            "Hardware:",
            f"  CPU Cores: {self._info.cpu_count}",
            f"  Total Memory: {self._info.total_memory_mb}MB",
            f"  Available Memory: {self._info.available_memory_mb}MB",
            "",
            "Storage:",
            f"  Total: {self._info.total_storage_gb:.1f}GB",
            f"  Available: {self._info.available_storage_gb:.1f}GB",
            "",
            "Network:",
            f"  Connected: {self._info.has_network}",
        ]
        
        if self._info.is_termux:
            lines.extend([
                "",
                "Termux:",
                f"  Version: {self._info.termux_version}",
                f"  Battery: {self._info.battery_level}%",
                f"  Charging: {self._info.battery_charging}",
            ])
        
        lines.extend([
            "",
            "=" * 60,
            "Checks:",
            "=" * 60,
        ])
        
        for check in self._info.checks:
            status_icon = {
                CheckStatus.PASS: "✓",
                CheckStatus.WARN: "⚠",
                CheckStatus.FAIL: "✗",
                CheckStatus.SKIP: "○",
            }.get(check.status, "?")
            
            lines.append(f"  {status_icon} {check.name}: {check.message}")
            
            if check.suggestion:
                lines.append(f"      Suggestion: {check.suggestion}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        if self._info.all_passed:
            lines.append("✓ Environment is ready for JARVIS!")
        else:
            lines.append("✗ Environment has issues that need to be resolved")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    @property
    def info(self) -> EnvironmentInfo:
        """Get environment info"""
        return self._info


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo environment detection"""
    detector = EnvironmentDetector(verbose=True)
    info = detector.detect_all()
    
    print(detector.get_report())
    
    print("\nJSON Output:")
    print(json.dumps(info.to_dict(), indent=2))


if __name__ == '__main__':
    import json
    main()
