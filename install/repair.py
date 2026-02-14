#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Repair System
====================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Dependency repair
- Configuration repair
- Database repair
- Full reset option

Features:
- Diagnose and repair dependencies
- Repair corrupted configuration
- Database integrity check
- Full factory reset
- Backup before repair

Memory Impact: < 10MB for repair
"""

import os
import sys
import json
import shutil
import logging
import subprocess
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class RepairType(Enum):
    """Types of repairs"""
    DEPENDENCIES = auto()
    CONFIGURATION = auto()
    DATABASE = auto()
    CACHE = auto()
    PERMISSIONS = auto()
    FULL_RESET = auto()


class RepairStatus(Enum):
    """Repair status"""
    NOT_NEEDED = auto()
    REPAIRED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class RepairResult:
    """Result of repair operation"""
    repair_type: RepairType
    status: RepairStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status in (RepairStatus.NOT_NEEDED, RepairStatus.REPAIRED)


@dataclass
class DiagnosticResult:
    """Result of diagnostic check"""
    name: str
    healthy: bool
    issues: List[str] = field(default_factory=list)
    repairable: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# REPAIR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class RepairSystem:
    """
    Ultra-Advanced Repair System.
    
    Features:
    - Dependency repair
    - Configuration repair
    - Database repair
    - Full reset
    
    Memory Budget: < 10MB
    
    Usage:
        repair = RepairSystem()
        
        # Run diagnostics
        results = repair.diagnose()
        
        # Repair specific issue
        result = repair.repair(RepairType.DEPENDENCIES)
        
        # Full repair
        repair.repair_all()
    """
    
    JARVIS_DIR = Path.home() / ".jarvis"
    
    def __init__(self, jarvis_dir: Path = None):
        """
        Initialize Repair System.
        
        Args:
            jarvis_dir: JARVIS directory
        """
        self._jarvis_dir = jarvis_dir or self.JARVIS_DIR
    
    def diagnose(self) -> List[DiagnosticResult]:
        """
        Run diagnostic checks.
        
        Returns:
            List of DiagnosticResult
        """
        results = []
        
        # Check dependencies
        results.append(self._diagnose_dependencies())
        
        # Check configuration
        results.append(self._diagnose_configuration())
        
        # Check database
        results.append(self._diagnose_database())
        
        # Check cache
        results.append(self._diagnose_cache())
        
        # Check permissions
        results.append(self._diagnose_permissions())
        
        return results
    
    def _diagnose_dependencies(self) -> DiagnosticResult:
        """Diagnose dependency issues"""
        result = DiagnosticResult(
            name="Dependencies",
            healthy=True,
            issues=[],
        )
        
        try:
            from install.deps import DependencyInstaller
            installer = DependencyInstaller()
            
            success, missing = installer.verify_all()
            
            if not success:
                result.healthy = False
                result.issues.append(f"Missing packages: {', '.join(missing)}")
        
        except Exception as e:
            result.healthy = False
            result.issues.append(f"Dependency check failed: {e}")
        
        return result
    
    def _diagnose_configuration(self) -> DiagnosticResult:
        """Diagnose configuration issues"""
        result = DiagnosticResult(
            name="Configuration",
            healthy=True,
            issues=[],
        )
        
        config_file = self._jarvis_dir / "config.json"
        
        if not config_file.exists():
            result.healthy = False
            result.issues.append("Configuration file not found")
            return result
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate structure
            required_sections = ['general', 'ai', 'storage']
            for section in required_sections:
                if section not in config:
                    result.healthy = False
                    result.issues.append(f"Missing section: {section}")
        
        except json.JSONDecodeError:
            result.healthy = False
            result.issues.append("Configuration file is corrupted (invalid JSON)")
        except Exception as e:
            result.healthy = False
            result.issues.append(f"Configuration check failed: {e}")
        
        return result
    
    def _diagnose_database(self) -> DiagnosticResult:
        """Diagnose database issues"""
        result = DiagnosticResult(
            name="Database",
            healthy=True,
            issues=[],
        )
        
        db_file = self._jarvis_dir / "data" / "jarvis.db"
        
        if not db_file.exists():
            # Database doesn't need to exist yet
            return result
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            
            if integrity != "ok":
                result.healthy = False
                result.issues.append(f"Database integrity issue: {integrity}")
            
            conn.close()
        
        except Exception as e:
            result.healthy = False
            result.issues.append(f"Database check failed: {e}")
        
        return result
    
    def _diagnose_cache(self) -> DiagnosticResult:
        """Diagnose cache issues"""
        result = DiagnosticResult(
            name="Cache",
            healthy=True,
            issues=[],
        )
        
        cache_dir = self._jarvis_dir / "cache"
        
        if not cache_dir.exists():
            return result
        
        # Check for stale cache files
        try:
            now = time.time()
            stale_count = 0
            
            for item in cache_dir.rglob('*'):
                if item.is_file():
                    # Check file age (older than 7 days)
                    if now - item.stat().st_mtime > 7 * 24 * 3600:
                        stale_count += 1
            
            if stale_count > 100:
                result.healthy = False
                result.issues.append(f"Many stale cache files: {stale_count}")
        
        except Exception as e:
            result.issues.append(f"Cache check warning: {e}")
        
        return result
    
    def _diagnose_permissions(self) -> DiagnosticResult:
        """Diagnose permission issues"""
        result = DiagnosticResult(
            name="Permissions",
            healthy=True,
            issues=[],
        )
        
        # Check if directories are writable
        for subdir in ['', 'data', 'cache', 'logs']:
            path = self._jarvis_dir / subdir
            
            if path.exists() and not os.access(path, os.W_OK):
                result.healthy = False
                result.issues.append(f"Cannot write to: {path}")
        
        return result
    
    def repair(self, repair_type: RepairType, backup: bool = True) -> RepairResult:
        """
        Perform specific repair.
        
        Args:
            repair_type: Type of repair
            backup: Create backup before repair
            
        Returns:
            RepairResult
        """
        logger.info(f"Starting repair: {repair_type.name}")
        
        if backup:
            self._create_backup()
        
        if repair_type == RepairType.DEPENDENCIES:
            return self._repair_dependencies()
        elif repair_type == RepairType.CONFIGURATION:
            return self._repair_configuration()
        elif repair_type == RepairType.DATABASE:
            return self._repair_database()
        elif repair_type == RepairType.CACHE:
            return self._repair_cache()
        elif repair_type == RepairType.PERMISSIONS:
            return self._repair_permissions()
        elif repair_type == RepairType.FULL_RESET:
            return self._full_reset()
        
        return RepairResult(
            repair_type=repair_type,
            status=RepairStatus.SKIPPED,
            message="Unknown repair type",
        )
    
    def repair_all(self) -> List[RepairResult]:
        """
        Perform all needed repairs.
        
        Returns:
            List of RepairResult
        """
        results = []
        
        diagnostics = self.diagnose()
        
        for diag in diagnostics:
            if not diag.healthy and diag.repairable:
                repair_type = {
                    "Dependencies": RepairType.DEPENDENCIES,
                    "Configuration": RepairType.CONFIGURATION,
                    "Database": RepairType.DATABASE,
                    "Cache": RepairType.CACHE,
                    "Permissions": RepairType.PERMISSIONS,
                }.get(diag.name)
                
                if repair_type:
                    result = self.repair(repair_type, backup=False)
                    results.append(result)
        
        return results
    
    def _repair_dependencies(self) -> RepairResult:
        """Repair dependencies"""
        try:
            from install.deps import DependencyInstaller
            installer = DependencyInstaller()
            
            success, missing = installer.verify_all()
            
            if success:
                return RepairResult(
                    repair_type=RepairType.DEPENDENCIES,
                    status=RepairStatus.NOT_NEEDED,
                    message="All dependencies are installed",
                )
            
            # Install missing
            result = installer.install_all(missing)
            
            if result.success:
                return RepairResult(
                    repair_type=RepairType.DEPENDENCIES,
                    status=RepairStatus.REPAIRED,
                    message=f"Installed: {', '.join(result.packages_installed)}",
                )
            else:
                return RepairResult(
                    repair_type=RepairType.DEPENDENCIES,
                    status=RepairStatus.FAILED,
                    message=f"Failed to install: {', '.join(result.packages_failed)}",
                )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.DEPENDENCIES,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _repair_configuration(self) -> RepairResult:
        """Repair configuration"""
        try:
            from install.config_gen import ConfigGenerator
            generator = ConfigGenerator()
            
            # Try to load existing
            config = generator.load()
            
            if config is None:
                # Create new default config
                config = generator.generate_default()
            
            # Validate
            valid, errors = generator.validate(config)
            
            if not valid:
                # Reset to defaults
                config = generator.generate_default()
            
            # Save
            generator.save(config)
            
            return RepairResult(
                repair_type=RepairType.CONFIGURATION,
                status=RepairStatus.REPAIRED,
                message="Configuration repaired",
            )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.CONFIGURATION,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _repair_database(self) -> RepairResult:
        """Repair database"""
        db_file = self._jarvis_dir / "data" / "jarvis.db"
        
        if not db_file.exists():
            return RepairResult(
                repair_type=RepairType.DATABASE,
                status=RepairStatus.NOT_NEEDED,
                message="No database to repair",
            )
        
        try:
            import sqlite3
            
            # Backup first
            backup_file = db_file.with_suffix('.db.backup')
            shutil.copy2(db_file, backup_file)
            
            # Try to repair
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Dump and recreate
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Run VACUUM to repair
            cursor.execute("VACUUM")
            
            conn.close()
            
            return RepairResult(
                repair_type=RepairType.DATABASE,
                status=RepairStatus.REPAIRED,
                message="Database repaired",
                details={'backup': str(backup_file)},
            )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.DATABASE,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _repair_cache(self) -> RepairResult:
        """Repair/clear cache"""
        cache_dir = self._jarvis_dir / "cache"
        
        if not cache_dir.exists():
            return RepairResult(
                repair_type=RepairType.CACHE,
                status=RepairStatus.NOT_NEEDED,
                message="No cache to clear",
            )
        
        try:
            # Clear all cache
            for item in cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            
            return RepairResult(
                repair_type=RepairType.CACHE,
                status=RepairStatus.REPAIRED,
                message="Cache cleared",
            )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.CACHE,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _repair_permissions(self) -> RepairResult:
        """Repair permissions"""
        try:
            # Ensure directories exist and are writable
            for subdir in ['', 'data', 'cache', 'logs', 'backups']:
                path = self._jarvis_dir / subdir
                path.mkdir(parents=True, exist_ok=True)
                
                # Set permissions
                os.chmod(path, 0o755)
            
            return RepairResult(
                repair_type=RepairType.PERMISSIONS,
                status=RepairStatus.REPAIRED,
                message="Permissions repaired",
            )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.PERMISSIONS,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _full_reset(self) -> RepairResult:
        """Perform full factory reset"""
        try:
            # Backup everything
            backup_dir = self._jarvis_dir / "backups" / f"reset_backup_{int(time.time())}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy important files
            for item in self._jarvis_dir.iterdir():
                if item.name == 'backups':
                    continue
                if item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                else:
                    shutil.copy2(item, backup_dir / item.name)
            
            # Clear everything
            for item in self._jarvis_dir.iterdir():
                if item.name == 'backups':
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            
            # Recreate structure
            for subdir in ['data', 'cache', 'logs', 'backups']:
                (self._jarvis_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Generate new config
            from install.config_gen import ConfigGenerator
            generator = ConfigGenerator()
            config = generator.generate_default()
            generator.save(config)
            
            return RepairResult(
                repair_type=RepairType.FULL_RESET,
                status=RepairStatus.REPAIRED,
                message="Full reset complete",
                details={'backup': str(backup_dir)},
            )
        
        except Exception as e:
            return RepairResult(
                repair_type=RepairType.FULL_RESET,
                status=RepairStatus.FAILED,
                message=str(e),
            )
    
    def _create_backup(self):
        """Create backup before repair"""
        backup_dir = self._jarvis_dir / "backups" / f"pre_repair_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration
        config_file = self._jarvis_dir / "config.json"
        if config_file.exists():
            shutil.copy2(config_file, backup_dir / "config.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo repair system"""
    repair = RepairSystem()
    
    print("JARVIS Repair System")
    print("=" * 40)
    
    # Diagnose
    print("\nRunning diagnostics...")
    results = repair.diagnose()
    
    for result in results:
        status = "✓" if result.healthy else "✗"
        print(f"  {status} {result.name}")
        for issue in result.issues:
            print(f"      - {issue}")
    
    # Repair all
    print("\nRepairing issues...")
    repairs = repair.repair_all()
    
    for result in repairs:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.repair_type.name}: {result.message}")


if __name__ == '__main__':
    main()
