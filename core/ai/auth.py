#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - API Key Authentication & Storage
=======================================================

Device: Realme Pad 2 Lite (RMP2402) | RAM: 4GB | Platform: Termux

Purpose:
- Secure API key storage
- Environment variable management
- Key validation
- Encrypted storage option

Security:
- Uses stdlib hashlib (no external crypto dependencies)
- Keys never logged or displayed in full
- Supports multiple key providers

Memory Impact: < 1MB
"""

import os
import hashlib
import base64
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class KeyProvider(Enum):
    """Supported key providers"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class KeyStatus(Enum):
    """Status of API key"""
    VALID = auto()
    INVALID = auto()
    EXPIRED = auto()
    MISSING = auto()
    UNKNOWN = auto()


@dataclass
class KeyInfo:
    """Information about an API key"""
    provider: KeyProvider
    key_prefix: str  # First 8 characters only
    status: KeyStatus = KeyStatus.UNKNOWN
    last_validated: Optional[float] = None
    validation_count: int = 0
    created_at: float = field(default_factory=time.time)


class SecureStorage:
    """
    Simple secure storage using stdlib.

    Note: This is NOT cryptographically secure encryption.
    It provides obfuscation to prevent accidental key exposure.
    For production, use proper encryption libraries.

    For Termux/4GB devices, we avoid heavy crypto libraries.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize secure storage.

        Args:
            storage_path: Path to storage file
        """
        self._storage_path = Path(storage_path or os.path.expanduser("~/.jarvis/keys.dat"))
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate a device-specific salt
        self._salt = self._generate_salt()

    def _generate_salt(self) -> bytes:
        """Generate a device-specific salt"""
        # Use device-specific info for salt
        salt_source = f"{os.environ.get('USER', 'user')}:{os.environ.get('HOME', '/home')}:jarvis-v14"
        return hashlib.sha256(salt_source.encode()).digest()[:16]

    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            self._salt,
            100000,  # iterations
            dklen=32
        )

    def encrypt(self, data: str, password: str = "jarvis-default") -> str:
        """
        Encrypt data (obfuscation, not cryptographically secure).

        Args:
            data: Data to encrypt
            password: Password for encryption

        Returns:
            Base64 encoded encrypted data
        """
        key = self._derive_key(password)

        # Simple XOR encryption (obfuscation only)
        data_bytes = data.encode('utf-8')
        key_bytes = key[:len(data_bytes)] if len(key) >= len(data_bytes) else key * (len(data_bytes) // len(key) + 1)
        key_bytes = key_bytes[:len(data_bytes)]

        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_bytes))

        return base64.b64encode(encrypted).decode('ascii')

    def decrypt(self, encrypted_data: str, password: str = "jarvis-default") -> str:
        """
        Decrypt data.

        Args:
            encrypted_data: Base64 encoded encrypted data
            password: Password for decryption

        Returns:
            Decrypted string
        """
        key = self._derive_key(password)

        encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))

        # XOR decryption (same as encryption)
        key_bytes = key[:len(encrypted_bytes)] if len(key) >= len(encrypted_bytes) else key * (len(encrypted_bytes) // len(key) + 1)
        key_bytes = key_bytes[:len(encrypted_bytes)]

        decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key_bytes))

        return decrypted.decode('utf-8')

    def save(self, data: Dict[str, str], password: str = None):
        """Save encrypted data to file"""
        password = password or "jarvis-default"

        encrypted_data = {}
        for key, value in data.items():
            encrypted_data[key] = self.encrypt(value, password)

        with open(self._storage_path, 'w') as f:
            json.dump({
                'version': 1,
                'data': encrypted_data,
                'timestamp': time.time()
            }, f)

        # Set restrictive permissions
        try:
            os.chmod(self._storage_path, 0o600)
        except Exception:
            pass

    def load(self, password: str = None) -> Dict[str, str]:
        """Load and decrypt data from file"""
        password = password or "jarvis-default"

        if not self._storage_path.exists():
            return {}

        try:
            with open(self._storage_path, 'r') as f:
                stored = json.load(f)

            if 'data' not in stored:
                return {}

            decrypted_data = {}
            for key, value in stored['data'].items():
                decrypted_data[key] = self.decrypt(value, password)

            return decrypted_data

        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}


class AuthManager:
    """
    Manage API key authentication.

    Features:
    - Multiple provider support
    - Environment variable integration
    - Encrypted file storage
    - Key validation
    - Safe key display

    Usage:
        auth = AuthManager()

        # Set key
        auth.set_key(KeyProvider.OPENROUTER, "sk-or-v1-xxx")

        # Get key
        key = auth.get_key(KeyProvider.OPENROUTER)

        # Validate
        is_valid = auth.validate_key(KeyProvider.OPENROUTER)
    """

    # Environment variable names
    ENV_VAR_MAP = {
        KeyProvider.OPENROUTER: "OPENROUTER_API_KEY",
        KeyProvider.OPENAI: "OPENAI_API_KEY",
        KeyProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }

    # Key prefixes for validation
    KEY_PREFIX_MAP = {
        KeyProvider.OPENROUTER: ("sk-or-", "sk-or-v1-"),
        KeyProvider.OPENAI: ("sk-",),
        KeyProvider.ANTHROPIC: ("sk-ant-",),
    }

    def __init__(self, storage_path: str = None, enable_encrypted_storage: bool = True):
        """
        Initialize Auth Manager.

        Args:
            storage_path: Path for encrypted storage
            enable_encrypted_storage: Whether to use encrypted file storage
        """
        self._storage = SecureStorage(storage_path) if enable_encrypted_storage else None
        self._keys: Dict[KeyProvider, str] = {}
        self._key_info: Dict[KeyProvider, KeyInfo] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Load from storage
        self._load_keys()

        # Load from environment
        self._load_from_env()

        logger.info("Auth Manager initialized")

    def _load_from_env(self):
        """Load keys from environment variables"""
        for provider, env_var in self.ENV_VAR_MAP.items():
            key = os.environ.get(env_var)
            if key and provider not in self._keys:
                self._keys[provider] = key
                self._key_info[provider] = KeyInfo(
                    provider=provider,
                    key_prefix=key[:8] + "..." if len(key) > 8 else "***"
                )

    def _load_keys(self):
        """Load keys from encrypted storage"""
        if not self._storage:
            return

        try:
            stored = self._storage.load()
            for key_str, key_value in stored.items():
                try:
                    provider = KeyProvider(key_str)
                    self._keys[provider] = key_value
                    self._key_info[provider] = KeyInfo(
                        provider=provider,
                        key_prefix=key_value[:8] + "..." if len(key_value) > 8 else "***"
                    )
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Could not load stored keys: {e}")

    def set_key(
        self,
        provider: KeyProvider,
        key: str,
        persist: bool = True,
        set_env: bool = True
    ):
        """
        Set an API key.

        Args:
            provider: Key provider
            key: API key value
            persist: Whether to save to encrypted storage
            set_env: Whether to set environment variable
        """
        with self._lock:
            # Store in memory
            self._keys[provider] = key
            self._key_info[provider] = KeyInfo(
                provider=provider,
                key_prefix=key[:8] + "..." if len(key) > 8 else "***"
            )

            # Set environment variable
            if set_env:
                env_var = self.ENV_VAR_MAP.get(provider)
                if env_var:
                    os.environ[env_var] = key

            # Persist to storage
            if persist and self._storage:
                self._save_keys()

        logger.info(f"Key set for {provider.value}")

    def get_key(self, provider: KeyProvider) -> Optional[str]:
        """
        Get an API key.

        Args:
            provider: Key provider

        Returns:
            API key or None
        """
        with self._lock:
            return self._keys.get(provider)

    def has_key(self, provider: KeyProvider) -> bool:
        """Check if a key is set"""
        return provider in self._keys and bool(self._keys[provider])

    def remove_key(self, provider: KeyProvider):
        """Remove an API key"""
        with self._lock:
            self._keys.pop(provider, None)
            self._key_info.pop(provider, None)

            # Remove from environment
            env_var = self.ENV_VAR_MAP.get(provider)
            if env_var and env_var in os.environ:
                del os.environ[env_var]

            # Update storage
            if self._storage:
                self._save_keys()

    def validate_key(self, provider: KeyProvider) -> KeyStatus:
        """
        Validate an API key format.

        Note: This only validates the format, not whether the key actually works.

        Args:
            provider: Key provider

        Returns:
            KeyStatus indicating validity
        """
        key = self.get_key(provider)

        if not key:
            return KeyStatus.MISSING

        # Check key format
        prefixes = self.KEY_PREFIX_MAP.get(provider, ())
        if prefixes and not any(key.startswith(p) for p in prefixes):
            return KeyStatus.INVALID

        # Check minimum length
        if len(key) < 20:
            return KeyStatus.INVALID

        # Update info
        if provider in self._key_info:
            self._key_info[provider].status = KeyStatus.VALID
            self._key_info[provider].last_validated = time.time()
            self._key_info[provider].validation_count += 1

        return KeyStatus.VALID

    def get_key_info(self, provider: KeyProvider) -> Optional[KeyInfo]:
        """Get information about a key (without revealing it)"""
        return self._key_info.get(provider)

    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all stored keys (safe display)"""
        result = {}
        for provider, info in self._key_info.items():
            result[provider.value] = {
                'prefix': info.key_prefix,
                'status': self.validate_key(provider).name,
                'last_validated': datetime.fromtimestamp(info.last_validated).isoformat() if info.last_validated else None,
                'has_key': self.has_key(provider),
            }
        return result

    def _save_keys(self):
        """Save keys to encrypted storage"""
        if not self._storage:
            return

        data = {provider.value: key for provider, key in self._keys.items()}
        self._storage.save(data)

    def get_any_valid_key(self) -> Tuple[Optional[KeyProvider], Optional[str]]:
        """
        Get any valid API key.

        Returns:
            Tuple of (provider, key) or (None, None)
        """
        priorities = [
            KeyProvider.OPENROUTER,
            KeyProvider.OPENAI,
            KeyProvider.ANTHROPIC,
        ]

        for provider in priorities:
            if self.has_key(provider) and self.validate_key(provider) == KeyStatus.VALID:
                return provider, self.get_key(provider)

        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_auth: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance"""
    global _auth
    if _auth is None:
        _auth = AuthManager()
    return _auth


def get_api_key(provider: KeyProvider = KeyProvider.OPENROUTER) -> Optional[str]:
    """Convenience function to get API key"""
    return get_auth_manager().get_key(provider)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for Auth Manager"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }

    # Use temp storage for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "test_keys.dat")
        auth = AuthManager(storage_path=storage_path)

        # Test 1: Set and get key
        auth.set_key(KeyProvider.OPENROUTER, "sk-or-v1-test123456789", persist=False)
        key = auth.get_key(KeyProvider.OPENROUTER)
        if key == "sk-or-v1-test123456789":
            results['passed'].append('set_get_key')
        else:
            results['failed'].append('set_get_key')

        # Test 2: Validate key
        status = auth.validate_key(KeyProvider.OPENROUTER)
        if status == KeyStatus.VALID:
            results['passed'].append('validate_key')
        else:
            results['failed'].append(f'validate_key: {status.name}')

        # Test 3: Invalid key
        auth.set_key(KeyProvider.OPENROUTER, "invalid", persist=False)
        status = auth.validate_key(KeyProvider.OPENROUTER)
        if status == KeyStatus.INVALID:
            results['passed'].append('detect_invalid_key')
        else:
            results['failed'].append(f'detect_invalid: {status.name}')

        # Test 4: Remove key
        auth.set_key(KeyProvider.OPENROUTER, "sk-or-v1-test123456789", persist=False)
        auth.remove_key(KeyProvider.OPENROUTER)
        if not auth.has_key(KeyProvider.OPENROUTER):
            results['passed'].append('remove_key')
        else:
            results['failed'].append('remove_key')

        # Test 5: Missing key
        status = auth.validate_key(KeyProvider.OPENROUTER)
        if status == KeyStatus.MISSING:
            results['passed'].append('detect_missing_key')
        else:
            results['failed'].append(f'detect_missing: {status.name}')

        # Test 6: List keys
        auth.set_key(KeyProvider.OPENROUTER, "sk-or-v1-test123456789", persist=False)
        keys = auth.list_keys()
        if KeyProvider.OPENROUTER.value in keys:
            results['passed'].append('list_keys')
        else:
            results['failed'].append('list_keys')

        # Test 7: Key info
        info = auth.get_key_info(KeyProvider.OPENROUTER)
        if info and info.key_prefix:
            results['passed'].append(f'key_info: {info.key_prefix}')
        else:
            results['failed'].append('key_info')

        results['keys'] = auth.list_keys()

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Auth Manager - Self Test")
    print("=" * 70)

    test_results = self_test()

    print("\n✅ Passed Tests:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")

    if test_results['failed']:
        print("\n❌ Failed Tests:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")

    print("\n📋 Stored Keys:")
    for provider, info in test_results.get('keys', {}).items():
        print(f"   {provider}: {info['prefix']} ({info['status']})")

    print("\n" + "=" * 70)
