#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Uninstall System
=======================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Clean uninstall
- Data backup before uninstall
- Optional dependency cleanup

Features:
- Complete uninstallation
- Data backup
- Configuration removal
- Optional dependency cleanup
- Uninstall confirmation

Memory Impact: < 5MB for uninstall
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class UninstallScope(Enum):
    """Uninstall scope"""
    FULL = auto()          # Remove everything
    KEEP_DATA = auto()     # Keep user data
    KEEP_CONFIG = auto()   # Keep configuration
    MINIMAL = auto()       # Remove only program files


@dataclass
class UninstallResult:
    """Result of uninstall operation"""
    success: bool
    scope: UninstallScope
    backup_path: str = ""
    removed_files: int = 0
    removed_dirs: int = 0
    freed_bytes: int = 0
    message: str = ""
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# UNINSTALLER
# ═══════════════════════════════════════════════════════════════════════════════

class Uninstaller:
    """
    Ultra-Advanced Uninstall System.
    
    Features:
    - Complete uninstallation
    - Data backup
    - Configuration removal
    - Optional dependency cleanup
    
    Memory Budget: < 5MB
    
    Usage:
        uninstaller = Uninstaller()
        
        # Uninstall with backup
        result = uninstaller.uninstall(
            scope=UninstallScope.FULL,
            backup=True
        )
    """
    
    JARVIS_DIR = Path.home() / ".jarvis"
    BACKUP_DIR = Path.home() / ".jarvis_backups"
    
    def __init__(self, jarvis_dir: Path = None):
        """
        Initialize Uninstaller.
        
        Args:
            jarvis_dir: JARVIS installation directory
        """
        self._jarvis_dir = jarvis_dir or self.JARVIS_DIR
        self._backup_dir = self.BACKUP_DIR
    
    def uninstall(
        self,
        scope: UninstallScope = UninstallScope.FULL,
        backup: bool = True,
        remove_deps: bool = False,
        confirm: bool = True,
    ) -> UninstallResult:
        """
        Perform uninstallation.
        
        Args:
            scope: Uninstall scope
            backup: Create backup before uninstall
            remove_deps: Remove installed dependencies
            confirm: Require user confirmation
            
        Returns:
            UninstallResult
        """
        logger.info(f"Starting uninstall (scope: {scope.name})")
        
        result = UninstallResult(
            success=False,
            scope=scope,
        )
        
        # Check if JARVIS is installed
        if not self._jarvis_dir.exists():
            result.message = "JARVIS is not installed"
            result.success = True
            return result
        
        # Confirm
        if confirm:
            if not self._confirm_uninstall(scope):
                result.message = "Uninstall cancelled by user"
                return result
        
        # Create backup
        if backup:
            backup_path = self._create_backup()
            if backup_path:
                result.backup_path = str(backup_path)
            else:
                result.errors.append("Failed to create backup")
                if scope == UninstallScope.FULL:
                    result.message = "Uninstall failed: could not create backup"
                    return result
        
        # Calculate what to remove
        to_remove = self._get_removal_list(scope)
        
        # Remove files
        for item in to_remove:
            try:
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    result.removed_dirs += 1
                    result.freed_bytes += size
                else:
                    size = item.stat().st_size
                    item.unlink()
                    result.removed_files += 1
                    result.freed_bytes += size
            except Exception as e:
                result.errors.append(f"Failed to remove {item}: {e}")
        
        # Remove dependencies
        if remove_deps:
            self._remove_dependencies()
        
        # Remove setup marker
        setup_file = self._jarvis_dir / ".setup_complete"
        if setup_file.exists():
            setup_file.unlink()
        
        result.success = True
        result.message = f"Uninstall complete. Freed {result.freed_bytes / 1024 / 1024:.1f}MB"
        
        logger.info(result.message)
        
        return result
    
    def _confirm_uninstall(self, scope: UninstallScope) -> bool:
        """Get user confirmation for uninstall"""
        print("\n" + "=" * 60)
        print("  JARVIS Uninstall Confirmation")
        print("=" * 60 + "\n")
        
        print(f"Uninstall scope: {scope.name}\n")
        
        if scope == UninstallScope.FULL:
            print("WARNING: This will remove ALL JARVIS data including:")
            print("  - Configuration files")
            print("  - User data")
            print("  - Logs")
            print("  - Cache")
        elif scope == UninstallScope.KEEP_DATA:
            print("This will remove:")
            print("  - Program files")
            print("  - Configuration")
            print("  - Logs")
            print("Keeping:")
            print("  - User data")
        elif scope == UninstallScope.KEEP_CONFIG:
            print("This will remove:")
            print("  - Program files")
            print("  - User data")
            print("Keeping:")
            print("  - Configuration")
        elif scope == UninstallScope.MINIMAL:
            print("This will remove only program files.")
        
        print()
        response = input("Continue with uninstall? [y/N]: ").strip().lower()
        return response in ('y', 'yes')
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup before uninstall"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self._backup_dir / f"jarvis_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy important files
            for item in self._jarvis_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, backup_path / item.name)
                else:
                    shutil.copy2(item, backup_path / item.name)
            
            # Write backup info
            with open(backup_path / "backup_info.json", 'w') as f:
                json.dump({
                    'created_at': datetime.now().isoformat(),
                    'original_path': str(self._jarvis_dir),
                }, f, indent=2)
            
            logger.info(f"Backup created at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _get_removal_list(self, scope: UninstallScope) -> List[Path]:
        """Get list of items to remove based on scope"""
        to_remove = []
        
        if scope == UninstallScope.FULL:
            # Remove everything
            if self._jarvis_dir.exists():
                to_remove.append(self._jarvis_dir)
        
        elif scope == UninstallScope.KEEP_DATA:
            # Keep data directory
            for item in self._jarvis_dir.iterdir():
                if item.name == 'data':
                    continue
                to_remove.append(item)
        
        elif scope == UninstallScope.KEEP_CONFIG:
            # Keep config file
            for item in self._jarvis_dir.iterdir():
                if item.name == 'config.json':
                    continue
                to_remove.append(item)
        
        elif scope == UninstallScope.MINIMAL:
            # Remove only specific directories
            for subdir in ['cache', 'logs', 'backups']:
                path = self._jarvis_dir / subdir
                if path.exists():
                    to_remove.append(path)
        
        return to_remove
    
    def _remove_dependencies(self):
        """Remove installed dependencies"""
        # List of packages to remove
        packages = [
            'click', 'colorama', 'python-dotenv', 'pyyaml',
            'requests', 'tqdm', 'schedule', 'typing-extensions',
            'psutil', 'httpx', 'rich', 'loguru', 'beautifulsoup4',
        ]
        
        for package in packages:
            try:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'uninstall', '-y', package],
                    capture_output=True,
                    timeout=30,
                )
            except:
                pass
    
    def get_install_size(self) -> int:
        """Get total installation size in bytes"""
        if not self._jarvis_dir.exists():
            return 0
        
        total = 0
        for item in self._jarvis_dir.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        
        return total
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        if not self._backup_dir.exists():
            return backups
        
        for backup in sorted(self._backup_dir.iterdir(), reverse=True):
            if backup.is_dir() and backup.name.startswith('jarvis_backup_'):
                info_file = backup / 'backup_info.json'
                info = {}
                
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                    except:
                        pass
                
                # Calculate backup size
                size = sum(f.stat().st_size for f in backup.rglob('*') if f.is_file())
                
                backups.append({
                    'path': str(backup),
                    'name': backup.name,
                    'created_at': info.get('created_at', ''),
                    'size_bytes': size,
                })
        
        return backups


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo uninstall system"""
    uninstaller = Uninstaller()
    
    print("JARVIS Uninstall System")
    print("=" * 40)
    
    # Show install size
    size = uninstaller.get_install_size()
    print(f"\nInstallation size: {size / 1024 / 1024:.1f}MB")
    
    # List backups
    backups = uninstaller.list_backups()
    if backups:
        print(f"\nAvailable backups: {len(backups)}")
        for backup in backups:
            print(f"  - {backup['name']}: {backup['size_bytes'] / 1024 / 1024:.1f}MB")
    
    print("\nTo uninstall, run with --uninstall flag")


if __name__ == '__main__':
    if '--uninstall' in sys.argv:
        uninstaller = Uninstaller()
        result = uninstaller.uninstall(confirm=True)
        
        if result.success:
            print(f"\n✓ {result.message}")
            if result.backup_path:
                print(f"Backup saved to: {result.backup_path}")
        else:
            print(f"\n✗ {result.message}")
    else:
        main()
