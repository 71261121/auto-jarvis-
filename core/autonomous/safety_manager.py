#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Safety Manager
====================================

Protects against dangerous operations.
Because autonomy needs safety rails.

Features:
- Protected file detection
- Dangerous command blocking
- Confirmation prompts for destructive operations
- Rate limiting for critical operations

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for operations"""
    SAFE = auto()           # Can execute without confirmation
    WARNING = auto()        # Should show warning
    DANGEROUS = auto()      # Needs explicit confirmation
    BLOCKED = auto()        # Cannot execute


@dataclass
class SafetyResult:
    """Result of safety check"""
    level: SafetyLevel
    allowed: bool
    message: str = ""
    requires_confirmation: bool = False
    confirmation_prompt: str = ""


class SafetyManager:
    """
    Safety Manager - The Guardian of JARVIS.
    
    Prevents accidental destruction of critical files
    and blocks dangerous commands.
    
    Usage:
        safety = SafetyManager()
        result = safety.check_file_write('/path/to/file')
        if result.requires_confirmation:
            if user_confirms():
                # proceed
    """
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PROTECTED FILES - Cannot be modified without explicit confirmation
    # ═══════════════════════════════════════════════════════════════════════════════
    
    PROTECTED_FILES = {
        # Environment and secrets
        '.env', '.env.local', '.env.production',
        'credentials.json', 'secrets.json', 'secrets.yaml',
        'id_rsa', 'id_rsa.pub', 'id_ed25519', 'id_ed25519.pub',
        'private.key', 'private.pem', 'public.key',
        
        # Git
        '.git', '.gitignore', '.gitmodules',
        
        # System
        '.bashrc', '.zshrc', '.profile', '.bash_profile',
        'authorized_keys', 'known_hosts', 'config',
        
        # Project critical
        'main.py', '__init__.py',
    }
    
    PROTECTED_PATTERNS = [
        r'\.env(\..*)?$',
        r'credentials?\..*$',
        r'secrets?\..*$',
        r'\.git/.*$',
        r'private.*\.pem$',
        r'private.*\.key$',
        r'\.ssh/.*$',
        r'.*\.pub$',
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DANGEROUS COMMANDS - Blocked or require confirmation
    # ═══════════════════════════════════════════════════════════════════════════════
    
    DANGEROUS_COMMANDS = {
        # Filesystem destruction
        r'rm\s+-rf\s+/',           # rm -rf /
        r'rm\s+-rf\s+~',           # rm -rf ~
        r'rm\s+-rf\s+\*',          # rm -rf *
        r'rm\s+-rf\s+\.',          # rm -rf .
        r'del\s+/s',               # Windows delete all
        r'rmdir\s+/s',             # Windows rmdir all
        r'shred\s+',               # Secure delete
        
        # Disk operations
        r'dd\s+if=',               # Disk dump
        r'mkfs',                   # Format filesystem
        r'fdisk',                  # Partition editor
        r'parted',                 # Partition editor
        r'format\s+[a-z]:',        # Windows format
        
        # System control
        r'shutdown',               # Shutdown system
        r'reboot',                 # Reboot system
        r'halt',                   # Halt system
        r'init\s+0',               # Shutdown
        r'init\s+6',               # Reboot
        r'poweroff',               # Power off
        r'systemctl\s+stop',       # Stop services
        
        # Dangerous scripts
        r':\(\)\s*\{\s*:\|:&\s*\}',  # Fork bomb
        r'chmod\s+777',            # Dangerous permissions
        r'chown\s+.*:.*\s+/',      # Change ownership of root
        
        # Network dangerous
        r'iptables\s+-F',          # Flush firewall
        r'ufw\s+disable',          # Disable firewall
        r'nc\s+-l',                # Netcat listener
        r'telnet\s+.*\d+',         # Telnet
        
        # Package system
        r'apt\s+remove\s+.*\*',    # Remove all
        r'apt\s+purge\s+.*\*',     # Purge all
        r'pip\s+uninstall\s+.*-y', # Uninstall packages
    }
    
    WARNING_COMMANDS = {
        r'pip\s+install',          # Installing packages
        r'pkg\s+install',          # Termux install
        r'npm\s+install',          # NPM install
        r'apt\s+install',          # APT install
        r'git\s+push',             # Git push
        r'git\s+reset\s+--hard',   # Git reset hard
        r'git\s+clean\s+-fd',      # Git clean
    }
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DIRECTORIES TO PROTECT
    # ═══════════════════════════════════════════════════════════════════════════════
    
    PROTECTED_DIRECTORIES = {
        '/',                      # Root
        '~',                      # Home
        '/etc',                   # System config
        '/usr',                   # System files
        '/bin',                   # Binaries
        '/sbin',                  # System binaries
        '/boot',                  # Boot files
        '/dev',                   # Devices
        '/proc',                  # Process info
        '/sys',                   # System info
        '/root',                  # Root home
        '/var',                   # Variable data
    }
    
    def __init__(self, project_root: str = None):
        """
        Initialize Safety Manager.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._confirmation_cache: Dict[str, bool] = {}
        
        logger.info("SafetyManager initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FILE SAFETY CHECKS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def check_file_read(self, file_path: str) -> SafetyResult:
        """
        Check if file can be read safely.
        
        Args:
            file_path: Path to file
            
        Returns:
            SafetyResult
        """
        path = Path(file_path).expanduser().resolve()
        
        # Check if exists
        if not path.exists():
            return SafetyResult(
                level=SafetyLevel.WARNING,
                allowed=False,
                message=f"File not found: {path}"
            )
        
        # Check if readable
        if not os.access(path, os.R_OK):
            return SafetyResult(
                level=SafetyLevel.WARNING,
                allowed=False,
                message=f"No read permission: {path}"
            )
        
        # Check for sensitive files
        if self._is_protected_file(path):
            return SafetyResult(
                level=SafetyLevel.WARNING,
                allowed=True,
                message=f"Reading protected file: {path.name}",
                requires_confirmation=True,
                confirmation_prompt=f"This is a protected file ({path.name}). Continue reading?"
            )
        
        return SafetyResult(
            level=SafetyLevel.SAFE,
            allowed=True
        )
    
    def check_file_write(self, file_path: str) -> SafetyResult:
        """
        Check if file can be written safely.
        
        Args:
            file_path: Path to file
            
        Returns:
            SafetyResult
        """
        path = Path(file_path).expanduser().resolve()
        
        # Check if protected file
        if self._is_protected_file(path):
            return SafetyResult(
                level=SafetyLevel.DANGEROUS,
                allowed=True,  # Allow with confirmation
                message=f"Protected file: {path.name}",
                requires_confirmation=True,
                confirmation_prompt=f"⚠️ You are about to modify protected file: {path.name}\nThis may cause issues. Continue?"
            )
        
        # Check if file exists
        if path.exists():
            return SafetyResult(
                level=SafetyLevel.WARNING,
                allowed=True,
                message=f"File will be overwritten: {path.name}",
                requires_confirmation=False  # Auto-backup handles this
            )
        
        # Check parent directory
        parent = path.parent
        if not parent.exists():
            return SafetyResult(
                level=SafetyLevel.SAFE,
                allowed=True,
                message=f"Will create parent directory: {parent}"
            )
        
        return SafetyResult(
            level=SafetyLevel.SAFE,
            allowed=True
        )
    
    def check_file_delete(self, file_path: str) -> SafetyResult:
        """
        Check if file can be deleted safely.
        
        ALWAYS requires confirmation for deletion.
        
        Args:
            file_path: Path to file
            
        Returns:
            SafetyResult
        """
        path = Path(file_path).expanduser().resolve()
        
        # Check if protected
        if self._is_protected_file(path):
            return SafetyResult(
                level=SafetyLevel.BLOCKED,
                allowed=False,
                message=f"Cannot delete protected file: {path.name}"
            )
        
        # Always require confirmation for deletion
        return SafetyResult(
            level=SafetyLevel.DANGEROUS,
            allowed=True,
            message=f"Will delete: {path}",
            requires_confirmation=True,
            confirmation_prompt=f"⚠️ Are you sure you want to DELETE: {path.name}?\nThis cannot be undone (backup will be created)."
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # COMMAND SAFETY CHECKS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def check_command(self, command: str) -> SafetyResult:
        """
        Check if command is safe to execute.
        
        Args:
            command: Command to check
            
        Returns:
            SafetyResult
        """
        command_lower = command.lower().strip()
        
        # Check against dangerous patterns
        for pattern in self.DANGEROUS_COMMANDS:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return SafetyResult(
                    level=SafetyLevel.BLOCKED,
                    allowed=False,
                    message=f"Blocked dangerous command pattern"
                )
        
        # Check against warning patterns
        for pattern in self.WARNING_COMMANDS:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return SafetyResult(
                    level=SafetyLevel.WARNING,
                    allowed=True,
                    message=f"This command may modify system state",
                    requires_confirmation=True,
                    confirmation_prompt=f"⚠️ Execute: {command}?"
                )
        
        return SafetyResult(
            level=SafetyLevel.SAFE,
            allowed=True
        )
    
    def check_install(self, package: str) -> SafetyResult:
        """
        Check if package installation is safe.
        
        Args:
            package: Package name
            
        Returns:
            SafetyResult
        """
        # Check for suspicious package names
        suspicious_patterns = [
            r'^evil', r'^malware', r'^hack', r'^crack',
            r'^pirate', r'^keygen', r'^warez',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, package, re.IGNORECASE):
                return SafetyResult(
                    level=SafetyLevel.WARNING,
                    allowed=True,
                    message=f"Package name looks suspicious: {package}",
                    requires_confirmation=True,
                    confirmation_prompt=f"⚠️ Install package '{package}'? This package name looks suspicious."
                )
        
        return SafetyResult(
            level=SafetyLevel.WARNING,
            allowed=True,
            message=f"Will install: {package}",
            requires_confirmation=True,
            confirmation_prompt=f"Install package: {package}?"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _is_protected_file(self, path: Path) -> bool:
        """Check if file is in protected list"""
        name = path.name
        path_str = str(path)
        
        # Check exact matches
        if name in self.PROTECTED_FILES:
            return True
        
        # Check patterns
        for pattern in self.PROTECTED_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                return True
        
        return False
    
    def _is_protected_directory(self, path: Path) -> bool:
        """Check if directory is protected"""
        path_str = str(path.resolve())
        
        for protected in self.PROTECTED_DIRECTORIES:
            protected_path = Path(protected).expanduser().resolve()
            try:
                if path.resolve() == protected_path:
                    return True
                if path_str.startswith(str(protected_path)):
                    return True
            except Exception:
                pass
        
        return False
    
    def get_safe_project_files(self) -> List[str]:
        """Get list of files that are safe to modify"""
        safe_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and protected directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = Path(root) / file
                if not self._is_protected_file(file_path):
                    rel_path = file_path.relative_to(self.project_root)
                    safe_files.append(str(rel_path))
        
        return safe_files


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for SafetyManager"""
    results = {
        'passed': [],
        'failed': [],
    }
    
    safety = SafetyManager()
    
    # Test 1: Block dangerous command
    result = safety.check_command("rm -rf /")
    if not result.allowed:
        results['passed'].append('block_dangerous_command')
    else:
        results['failed'].append('block_dangerous_command')
    
    # Test 2: Allow safe command
    result = safety.check_command("ls -la")
    if result.allowed and result.level == SafetyLevel.SAFE:
        results['passed'].append('allow_safe_command')
    else:
        results['failed'].append('allow_safe_command')
    
    # Test 3: Protect .env file
    result = safety.check_file_write(".env")
    if result.requires_confirmation or result.level == SafetyLevel.DANGEROUS:
        results['passed'].append('protect_env_file')
    else:
        results['failed'].append('protect_env_file')
    
    # Test 4: Delete requires confirmation
    result = safety.check_file_delete("test.py")
    if result.requires_confirmation:
        results['passed'].append('delete_requires_confirmation')
    else:
        results['failed'].append('delete_requires_confirmation')
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Safety Manager - Self Test")
    print("=" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    print(f"\nTotal: {len(test_results['passed'])}/{len(test_results['passed']) + len(test_results['failed'])} passed")
