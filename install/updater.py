#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Update System
====================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Version checking
- GitHub release checking
- Self-update capability
- Rollback on failure

Features:
- Version comparison
- GitHub API integration
- Download and install updates
- Automatic rollback
- Update notifications
- Changelog display

Memory Impact: < 10MB for updates
"""

import os
import sys
import re
import json
import time
import shutil
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class UpdateStatus(Enum):
    """Update status"""
    UP_TO_DATE = auto()
    UPDATE_AVAILABLE = auto()
    UPDATE_REQUIRED = auto()
    UPDATE_FAILED = auto()
    CHECK_FAILED = auto()


class UpdateSource(Enum):
    """Update sources"""
    GITHUB = auto()
    LOCAL = auto()
    CUSTOM = auto()


@dataclass
class Version:
    """Version representation"""
    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: str = ""
    
    @classmethod
    def parse(cls, version_str: str) -> 'Version':
        """Parse version string"""
        # Handle formats: 1.0.0, v1.0.0, 1.0.0-beta
        version_str = version_str.lstrip('v')
        
        parts = version_str.split('-', 1)
        main = parts[0]
        prerelease = parts[1] if len(parts) > 1 else ""
        
        version_parts = main.split('.')
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        
        return cls(major, minor, patch, prerelease)
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version
    
    def __lt__(self, other: 'Version') -> bool:
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Prerelease versions are less than release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return self.prerelease < other.prerelease
    
    def __gt__(self, other: 'Version') -> bool:
        return other < self
    
    def __eq__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)
    
    def __le__(self, other: 'Version') -> bool:
        return self < other or self == other
    
    def __ge__(self, other: 'Version') -> bool:
        return self > other or self == other


@dataclass
class Release:
    """Release information"""
    version: Version
    name: str = ""
    published_at: str = ""
    body: str = ""  # Changelog
    assets: List[Dict] = field(default_factory=list)
    download_url: str = ""
    size_bytes: int = 0
    
    @classmethod
    def from_github(cls, data: Dict) -> 'Release':
        """Create from GitHub API response"""
        version = Version.parse(data.get('tag_name', '0.0.0'))
        
        assets = data.get('assets', [])
        download_url = ""
        size_bytes = 0
        
        # Find zipball or tarball
        for asset in assets:
            if asset.get('name', '').endswith(('.zip', '.tar.gz')):
                download_url = asset.get('browser_download_url', '')
                size_bytes = asset.get('size', 0)
                break
        
        # Fallback to zipball_url
        if not download_url:
            download_url = data.get('zipball_url', '')
        
        return cls(
            version=version,
            name=data.get('name', ''),
            published_at=data.get('published_at', ''),
            body=data.get('body', ''),
            assets=assets,
            download_url=download_url,
            size_bytes=size_bytes,
        )


@dataclass
class UpdateResult:
    """Result of update operation"""
    success: bool
    status: UpdateStatus = UpdateStatus.UP_TO_DATE
    current_version: str = ""
    new_version: str = ""
    message: str = ""
    rollback_available: bool = False
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Updater:
    """
    Ultra-Advanced Update System.
    
    Features:
    - Version checking
    - GitHub release checking
    - Self-update capability
    - Automatic rollback
    
    Memory Budget: < 10MB
    
    Usage:
        updater = Updater()
        
        # Check for updates
        status, release = updater.check()
        
        if status == UpdateStatus.UPDATE_AVAILABLE:
            print(f"Update available: {release.version}")
            
            # Install update
            result = updater.update()
    """
    
    CURRENT_VERSION = "14.0.0"
    GITHUB_REPO = "jarvis/jarvis-v14"  # Example repo
    GITHUB_API = "https://api.github.com"
    
    def __init__(
        self,
        github_repo: str = None,
        current_version: str = None,
    ):
        """
        Initialize Updater.
        
        Args:
            github_repo: GitHub repository (owner/repo)
            current_version: Current version string
        """
        self._github_repo = github_repo or self.GITHUB_REPO
        self._current_version = Version.parse(current_version or self.CURRENT_VERSION)
        self._backup_dir = Path.home() / ".jarvis" / "backups" / "pre_update"
    
    @property
    def current_version(self) -> Version:
        """Get current version"""
        return self._current_version
    
    def check(self, timeout: float = 30.0) -> Tuple[UpdateStatus, Optional[Release]]:
        """
        Check for updates.
        
        Args:
            timeout: Request timeout
            
        Returns:
            Tuple of (UpdateStatus, Release or None)
        """
        logger.info("Checking for updates...")
        
        try:
            release = self._fetch_latest_release(timeout)
            
            if release is None:
                return UpdateStatus.CHECK_FAILED, None
            
            logger.debug(f"Latest release: {release.version}")
            
            if release.version > self._current_version:
                return UpdateStatus.UPDATE_AVAILABLE, release
            else:
                return UpdateStatus.UP_TO_DATE, release
                
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return UpdateStatus.CHECK_FAILED, None
    
    def _fetch_latest_release(self, timeout: float) -> Optional[Release]:
        """Fetch latest release from GitHub"""
        url = f"{self.GITHUB_API}/repos/{self._github_repo}/releases/latest"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/vnd.github.v3+json')
            request.add_header('User-Agent', f'JARVIS/{self._current_version}')
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
                return Release.from_github(data)
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning("No releases found")
            else:
                logger.error(f"HTTP error: {e.code}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch release: {e}")
            return None
    
    def update(self, backup: bool = True) -> UpdateResult:
        """
        Perform update.
        
        Args:
            backup: Create backup before update
            
        Returns:
            UpdateResult
        """
        start_time = time.time()
        
        # Check for update
        status, release = self.check()
        
        if status != UpdateStatus.UPDATE_AVAILABLE:
            return UpdateResult(
                success=False,
                status=status,
                current_version=str(self._current_version),
                message="No update available",
            )
        
        if release is None:
            return UpdateResult(
                success=False,
                status=UpdateStatus.CHECK_FAILED,
                message="Could not fetch release information",
            )
        
        logger.info(f"Updating from {self._current_version} to {release.version}...")
        
        try:
            # Create backup
            if backup:
                self._create_backup()
            
            # Download update
            update_file = self._download_update(release)
            
            if update_file is None:
                return UpdateResult(
                    success=False,
                    status=UpdateStatus.UPDATE_FAILED,
                    message="Failed to download update",
                )
            
            # Install update
            success = self._install_update(update_file)
            
            if success:
                duration_ms = (time.time() - start_time) * 1000
                return UpdateResult(
                    success=True,
                    status=UpdateStatus.UPDATE_AVAILABLE,
                    current_version=str(self._current_version),
                    new_version=str(release.version),
                    message=f"Successfully updated to {release.version}",
                    rollback_available=backup,
                    duration_ms=duration_ms,
                )
            else:
                # Rollback
                if backup:
                    self._rollback()
                
                return UpdateResult(
                    success=False,
                    status=UpdateStatus.UPDATE_FAILED,
                    message="Failed to install update",
                    rollback_available=False,
                )
                
        except Exception as e:
            logger.error(f"Update failed: {e}")
            
            # Rollback
            if backup:
                self._rollback()
            
            return UpdateResult(
                success=False,
                status=UpdateStatus.UPDATE_FAILED,
                message=str(e),
            )
    
    def _create_backup(self):
        """Create backup before update"""
        logger.info("Creating backup...")
        
        try:
            self._backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy current installation
            jarvis_dir = Path(__file__).parent.parent
            
            for item in jarvis_dir.iterdir():
                if item.name in ('__pycache__', '.git', 'backups'):
                    continue
                
                dest = self._backup_dir / item.name
                
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            # Save backup metadata
            with open(self._backup_dir / 'backup_info.json', 'w') as f:
                json.dump({
                    'version': str(self._current_version),
                    'created_at': datetime.now().isoformat(),
                }, f, indent=2)
            
            logger.info(f"Backup created at {self._backup_dir}")
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def _download_update(self, release: Release) -> Optional[Path]:
        """Download update package"""
        if not release.download_url:
            logger.error("No download URL available")
            return None
        
        logger.info(f"Downloading update from {release.download_url}...")
        
        try:
            # Create temp directory
            temp_dir = Path(tempfile.mkdtemp())
            update_file = temp_dir / f"update-{release.version}.zip"
            
            # Download
            request = urllib.request.Request(release.download_url)
            request.add_header('User-Agent', f'JARVIS/{self._current_version}')
            
            with urllib.request.urlopen(request, timeout=300) as response:
                with open(update_file, 'wb') as f:
                    f.write(response.read())
            
            logger.info(f"Downloaded {update_file.stat().st_size / 1024 / 1024:.1f}MB")
            return update_file
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def _install_update(self, update_file: Path) -> bool:
        """Install update package"""
        logger.info(f"Installing update from {update_file}...")
        
        try:
            import zipfile
            
            # Extract
            extract_dir = update_file.parent / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(update_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find extracted content
            extracted_items = list(extract_dir.iterdir())
            if not extracted_items:
                logger.error("No files extracted")
                return False
            
            # Usually GitHub zipballs have a single directory
            source_dir = extracted_items[0] if len(extracted_items) == 1 else extract_dir
            
            # Copy files to installation directory
            jarvis_dir = Path(__file__).parent.parent
            
            for item in source_dir.rglob('*'):
                if item.is_file():
                    relative = item.relative_to(source_dir)
                    dest = jarvis_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
            
            logger.info("Update installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _rollback(self) -> bool:
        """Rollback to previous version"""
        logger.info("Rolling back to previous version...")
        
        if not self._backup_dir.exists():
            logger.error("No backup available for rollback")
            return False
        
        try:
            jarvis_dir = Path(__file__).parent.parent
            
            # Restore from backup
            for item in self._backup_dir.iterdir():
                if item.name == 'backup_info.json':
                    continue
                
                dest = jarvis_dir / item.name
                
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            logger.info("Rollback complete")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_changelog(self, limit: int = 5) -> List[Release]:
        """
        Get recent changelog.
        
        Args:
            limit: Number of releases to fetch
            
        Returns:
            List of Release objects
        """
        url = f"{self.GITHUB_API}/repos/{self._github_repo}/releases"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('Accept', 'application/vnd.github.v3+json')
            request.add_header('User-Agent', f'JARVIS/{self._current_version}')
            
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode())
                return [Release.from_github(r) for r in data[:limit]]
                
        except Exception as e:
            logger.error(f"Failed to fetch changelog: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo update system"""
    updater = Updater()
    
    print("JARVIS Update System")
    print("=" * 40)
    print(f"Current version: {updater.current_version}")
    print()
    
    # Check for updates
    status, release = updater.check()
    
    if status == UpdateStatus.UPDATE_AVAILABLE:
        print(f"Update available: {release.version}")
        print(f"Published: {release.published_at}")
        print()
        print("Changelog:")
        print(release.body[:500] if release.body else "No changelog available")
    elif status == UpdateStatus.UP_TO_DATE:
        print("✓ Already up to date")
    else:
        print(f"✗ Update check failed: {status.name}")


if __name__ == '__main__':
    main()
