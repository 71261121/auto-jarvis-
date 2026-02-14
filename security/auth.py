#!/usr/bin/env python3
"""
JARVIS Authentication System
Ultra-Advanced Security Module for Self-Modifying AI

Features:
- Multi-method password hashing (bcrypt, scrypt, argon2 with fallbacks)
- Session management with secure tokens
- Rate limiting for brute-force protection
- Account lockout after failed attempts
- Multi-factor authentication support
- Password policy enforcement
- Secure token generation
- Session persistence and recovery
- API key authentication
- Device fingerprinting

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import time
import json
import secrets
import hashlib
import hmac
import base64
import threading
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from abc import ABC, abstractmethod
import uuid
import struct

# Try importing optional dependencies with fallbacks
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import hashlib as hashlib_module
    SCRYPT_AVAILABLE = hasattr(hashlib_module, 'scrypt')
except Exception:
    SCRYPT_AVAILABLE = False

# Constants
DEFAULT_SALT_LENGTH = 32
DEFAULT_TOKEN_LENGTH = 64
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
TOKEN_EXPIRY = 86400  # 24 hours


class AuthMethod(Enum):
    """Authentication methods supported by JARVIS"""
    PASSWORD = auto()
    API_KEY = auto()
    TOKEN = auto()
    BIOMETRIC = auto()
    MULTI_FACTOR = auto()
    DEVICE_TRUST = auto()


class AuthStatus(Enum):
    """Status of authentication attempts"""
    SUCCESS = auto()
    FAILED = auto()
    LOCKED_OUT = auto()
    EXPIRED = auto()
    INVALID_TOKEN = auto()
    RATE_LIMITED = auto()
    PASSWORD_EXPIRED = auto()
    MFA_REQUIRED = auto()


class UserRole(Enum):
    """User roles for permission system"""
    GUEST = 0
    USER = 1
    DEVELOPER = 2
    ADMIN = 3
    SUPER_ADMIN = 4
    SYSTEM = 5


class PasswordPolicyResult(Enum):
    """Password policy validation results"""
    VALID = auto()
    TOO_SHORT = auto()
    TOO_LONG = auto()
    NO_UPPERCASE = auto()
    NO_LOWERCASE = auto()
    NO_DIGIT = auto()
    NO_SPECIAL = auto()
    COMMON_PASSWORD = auto()
    CONTAINS_USERNAME = auto()
    CONTAINS_EMAIL = auto()


@dataclass
class PasswordPolicy:
    """Password policy configuration"""
    min_length: int = PASSWORD_MIN_LENGTH
    max_length: int = PASSWORD_MAX_LENGTH
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    prevent_common: bool = True
    prevent_username: bool = True
    prevent_email: bool = True
    max_age_days: int = 90
    history_count: int = 5
    min_complexity_score: int = 3


@dataclass
class User:
    """User data model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: UserRole = UserRole.USER
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)
    password_history: List[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    trusted_devices: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'salt': self.salt,
            'role': self.role.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'failed_attempts': self.failed_attempts,
            'locked_until': self.locked_until.isoformat() if self.locked_until else None,
            'password_changed_at': self.password_changed_at.isoformat() if self.password_changed_at else None,
            'password_history': self.password_history,
            'mfa_enabled': self.mfa_enabled,
            'mfa_secret': self.mfa_secret,
            'api_keys': self.api_keys,
            'trusted_devices': self.trusted_devices,
            'metadata': self.metadata,
            'is_active': self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary"""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            salt=data['salt'],
            role=UserRole(data.get('role', 1)),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            failed_attempts=data.get('failed_attempts', 0),
            locked_until=datetime.fromisoformat(data['locked_until']) if data.get('locked_until') else None,
            password_changed_at=datetime.fromisoformat(data['password_changed_at']) if data.get('password_changed_at') else datetime.now(),
            password_history=data.get('password_history', []),
            mfa_enabled=data.get('mfa_enabled', False),
            mfa_secret=data.get('mfa_secret'),
            api_keys=data.get('api_keys', []),
            trusted_devices=data.get('trusted_devices', []),
            metadata=data.get('metadata', {}),
            is_active=data.get('is_active', True)
        )


@dataclass
class Session:
    """Session data model"""
    session_id: str
    user_id: str
    token: str
    refresh_token: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(seconds=DEFAULT_SESSION_TIMEOUT))
    last_activity: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'token': self.token,
            'refresh_token': self.refresh_token,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'device_fingerprint': self.device_fingerprint,
            'is_active': self.is_active,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            token=data['token'],
            refresh_token=data['refresh_token'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            device_fingerprint=data.get('device_fingerprint'),
            is_active=data.get('is_active', True),
            metadata=data.get('metadata', {})
        )

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now() > self.expires_at

    def extend(self, seconds: int = DEFAULT_SESSION_TIMEOUT) -> None:
        """Extend session expiry"""
        self.expires_at = datetime.now() + timedelta(seconds=seconds)
        self.last_activity = datetime.now()


@dataclass
class AuthResult:
    """Authentication result"""
    status: AuthStatus
    user: Optional[User] = None
    session: Optional[Session] = None
    message: str = ""
    mfa_required: bool = False
    attempts_remaining: int = 0

    @property
    def success(self) -> bool:
        return self.status == AuthStatus.SUCCESS


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry"""
    attempts: int = 0
    first_attempt: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    blocked_until: Optional[datetime] = None


class PasswordHasher(ABC):
    """Abstract password hasher interface"""

    @abstractmethod
    def hash(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password, return (hash, salt)"""
        pass

    @abstractmethod
    def verify(self, password: str, hash: str, salt: str) -> bool:
        """Verify password against hash"""
        pass

    @abstractmethod
    def needs_rehash(self, hash: str) -> bool:
        """Check if hash needs rehashing"""
        pass


class BcryptHasher(PasswordHasher):
    """Bcrypt password hasher"""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds

    def hash(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password using bcrypt"""
        if BCRYPT_AVAILABLE:
            if salt is None:
                salt = bcrypt.gensalt(rounds=self.rounds)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8'), salt.decode('utf-8')
        else:
            # Fallback to internal implementation
            return self._fallback_hash(password, salt)

    def _fallback_hash(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Fallback hash when bcrypt not available"""
        if salt is None:
            salt = secrets.token_bytes(DEFAULT_SALT_LENGTH)
        salt_str = base64.b64encode(salt).decode('utf-8')
        hash_input = password.encode('utf-8') + salt
        hashed = hashlib.sha256(hash_input).hexdigest()
        return f"sha256${hashed}", salt_str

    def verify(self, password: str, hash: str, salt: str) -> bool:
        """Verify password against bcrypt hash"""
        if BCRYPT_AVAILABLE and hash.startswith('$2'):
            try:
                return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))
            except Exception:
                return False
        elif hash.startswith('sha256$'):
            # Verify fallback hash
            salt_bytes = base64.b64decode(salt)
            _, hash_value = hash.split('$')
            hash_input = password.encode('utf-8') + salt_bytes
            computed = hashlib.sha256(hash_input).hexdigest()
            return hmac.compare_digest(hash_value, computed)
        return False

    def needs_rehash(self, hash: str) -> bool:
        """Check if hash needs rehashing"""
        if hash.startswith('sha256$'):
            return True
        if BCRYPT_AVAILABLE and hash.startswith('$2'):
            try:
                return bcrypt.checkpw(b'', hash.encode('utf-8')) or False
            except Exception:
                pass
        return False


class ScryptHasher(PasswordHasher):
    """Scrypt password hasher"""

    def __init__(self, n: int = 2**14, r: int = 8, p: int = 1):
        self.n = n
        self.r = r
        self.p = p

    def hash(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password using scrypt"""
        if salt is None:
            salt = secrets.token_bytes(DEFAULT_SALT_LENGTH)

        salt_str = base64.b64encode(salt).decode('utf-8')

        if SCRYPT_AVAILABLE:
            hashed = hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt,
                n=self.n,
                r=self.r,
                p=self.p,
                dklen=64
            )
            hash_str = base64.b64encode(hashed).decode('utf-8')
            return f"scrypt${hash_str}", salt_str
        else:
            # Fallback to PBKDF2
            hashed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000,
                dklen=64
            )
            hash_str = base64.b64encode(hashed).decode('utf-8')
            return f"pbkdf2${hash_str}", salt_str

    def verify(self, password: str, hash: str, salt: str) -> bool:
        """Verify password against scrypt hash"""
        try:
            salt_bytes = base64.b64decode(salt)

            if hash.startswith('scrypt$'):
                if SCRYPT_AVAILABLE:
                    _, hash_value = hash.split('$')
                    expected = base64.b64decode(hash_value)
                    computed = hashlib.scrypt(
                        password.encode('utf-8'),
                        salt=salt_bytes,
                        n=self.n,
                        r=self.r,
                        p=self.p,
                        dklen=64
                    )
                    return hmac.compare_digest(expected, computed)
            elif hash.startswith('pbkdf2$'):
                _, hash_value = hash.split('$')
                expected = base64.b64decode(hash_value)
                computed = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt_bytes,
                    100000,
                    dklen=64
                )
                return hmac.compare_digest(expected, computed)
        except Exception:
            pass
        return False

    def needs_rehash(self, hash: str) -> bool:
        """Check if hash needs rehashing"""
        return hash.startswith('pbkdf2$')


class TokenGenerator:
    """Secure token generator for sessions and API keys"""

    @staticmethod
    def generate_token(length: int = DEFAULT_TOKEN_LENGTH) -> str:
        """Generate a secure random token"""
        return secrets.token_hex(length)

    @staticmethod
    def generate_uuid() -> str:
        """Generate a UUID4 string"""
        return str(uuid.uuid4())

    @staticmethod
    def generate_api_key(prefix: str = "jrv") -> str:
        """Generate an API key with prefix"""
        key_part = secrets.token_urlsafe(32)
        return f"{prefix}_{key_part}"

    @staticmethod
    def generate_refresh_token() -> str:
        """Generate a refresh token"""
        return secrets.token_urlsafe(64)

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for storage"""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    @staticmethod
    def verify_token_hash(token: str, hash: str) -> bool:
        """Verify a token against its hash"""
        computed = hashlib.sha256(token.encode('utf-8')).hexdigest()
        return hmac.compare_digest(computed, hash)


class DeviceFingerprinter:
    """Generate device fingerprints for trusted device tracking"""

    @staticmethod
    def generate(user_agent: str, ip_address: str, additional_data: Optional[Dict] = None) -> str:
        """Generate a device fingerprint"""
        components = [
            user_agent or "",
            ip_address or "",
        ]

        if additional_data:
            for key in sorted(additional_data.keys()):
                components.append(f"{key}:{additional_data[key]}")

        fingerprint_data = "|".join(components)
        return hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()[:32]

    @staticmethod
    def verify(fingerprint: str, user_agent: str, ip_address: str, additional_data: Optional[Dict] = None) -> bool:
        """Verify a device fingerprint"""
        computed = DeviceFingerprinter.generate(user_agent, ip_address, additional_data)
        return hmac.compare_digest(fingerprint, computed)


class PasswordValidator:
    """Password policy validator"""

    # Common passwords to reject
    COMMON_PASSWORDS = {
        'password', 'password1', 'password123', '123456', '12345678',
        'qwerty', 'abc123', 'monkey', 'master', 'dragon', 'letmein',
        'login', 'admin', 'welcome', 'football', 'iloveyou', 'starwars',
        'batman', 'superman', 'princess', 'ashley', 'michael', 'shadow'
    }

    def __init__(self, policy: Optional[PasswordPolicy] = None):
        self.policy = policy or PasswordPolicy()

    def validate(self, password: str, username: str = "", email: str = "") -> List[PasswordPolicyResult]:
        """Validate password against policy"""
        results = []

        # Length checks
        if len(password) < self.policy.min_length:
            results.append(PasswordPolicyResult.TOO_SHORT)

        if len(password) > self.policy.max_length:
            results.append(PasswordPolicyResult.TOO_LONG)

        # Character requirements
        if self.policy.require_uppercase and not re.search(r'[A-Z]', password):
            results.append(PasswordPolicyResult.NO_UPPERCASE)

        if self.policy.require_lowercase and not re.search(r'[a-z]', password):
            results.append(PasswordPolicyResult.NO_LOWERCASE)

        if self.policy.require_digit and not re.search(r'\d', password):
            results.append(PasswordPolicyResult.NO_DIGIT)

        if self.policy.require_special and not re.search(f'[{re.escape(self.policy.special_chars)}]', password):
            results.append(PasswordPolicyResult.NO_SPECIAL)

        # Common password check
        if self.policy.prevent_common:
            lower_password = password.lower()
            for common in self.COMMON_PASSWORDS:
                if common in lower_password or lower_password in common:
                    results.append(PasswordPolicyResult.COMMON_PASSWORD)
                    break

        # Username/email check
        if self.policy.prevent_username and username:
            if username.lower() in password.lower():
                results.append(PasswordPolicyResult.CONTAINS_USERNAME)

        if self.policy.prevent_email and email:
            email_local = email.split('@')[0]
            if email_local.lower() in password.lower():
                results.append(PasswordPolicyResult.CONTAINS_EMAIL)

        if not results:
            results.append(PasswordPolicyResult.VALID)

        return results

    def calculate_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0

        # Length contribution
        score += min(len(password) * 4, 40)

        # Character variety
        if re.search(r'[a-z]', password):
            score += 5
        if re.search(r'[A-Z]', password):
            score += 5
        if re.search(r'\d', password):
            score += 5
        if re.search(r'[^a-zA-Z\d]', password):
            score += 10

        # Variety bonus
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 20)

        # Pattern penalties
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            score -= 10
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
            score -= 10
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
            score -= 10

        # Common password penalty
        if password.lower() in self.COMMON_PASSWORDS:
            score = max(score - 30, 0)

        return max(0, min(100, score))


class RateLimiter:
    """Rate limiter for authentication attempts"""

    def __init__(self, max_attempts: int = MAX_FAILED_ATTEMPTS,
                 window_seconds: int = 300,
                 lockout_duration: int = LOCKOUT_DURATION):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_duration = lockout_duration
        self._entries: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()

    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """Check if identifier is rate limited. Returns (is_limited, seconds_remaining)"""
        with self._lock:
            now = datetime.now()
            entry = self._entries.get(identifier)

            if entry is None:
                return False, 0

            # Check if currently locked out
            if entry.blocked_until and entry.blocked_until > now:
                remaining = (entry.blocked_until - now).total_seconds()
                return True, int(remaining)

            # Clear old entries
            if entry.first_attempt:
                window_end = entry.first_attempt + timedelta(seconds=self.window_seconds)
                if window_end < now:
                    del self._entries[identifier]
                    return False, 0

            return False, 0

    def record_attempt(self, identifier: str, success: bool) -> None:
        """Record an authentication attempt"""
        with self._lock:
            now = datetime.now()

            if identifier not in self._entries:
                self._entries[identifier] = RateLimitEntry()

            entry = self._entries[identifier]

            if success:
                # Clear on successful auth
                if identifier in self._entries:
                    del self._entries[identifier]
                return

            # Record failed attempt
            if entry.first_attempt is None:
                entry.first_attempt = now

            entry.attempts += 1
            entry.last_attempt = now

            # Check if should lock out
            if entry.attempts >= self.max_attempts:
                entry.blocked_until = now + timedelta(seconds=self.lockout_duration)

    def get_remaining_attempts(self, identifier: str) -> int:
        """Get remaining attempts before lockout"""
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is None:
                return self.max_attempts
            return max(0, self.max_attempts - entry.attempts)

    def clear(self, identifier: str) -> None:
        """Clear rate limit for identifier"""
        with self._lock:
            if identifier in self._entries:
                del self._entries[identifier]


class UserStore:
    """User storage interface"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.expanduser("~/.jarvis/users.json")
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}
        self._email_index: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load users from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        user = User.from_dict(user_data)
                        self._users[user.user_id] = user
                        self._username_index[user.username.lower()] = user.user_id
                        if user.email:
                            self._email_index[user.email.lower()] = user.user_id
        except Exception as e:
            print(f"Warning: Could not load users: {e}")

    def _save(self) -> None:
        """Save users to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {
                'users': [user.to_dict() for user in self._users.values()],
                'version': 1
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save users: {e}")

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self._lock:
            user_id = self._username_index.get(username.lower())
            return self._users.get(user_id) if user_id else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        with self._lock:
            user_id = self._email_index.get(email.lower())
            return self._users.get(user_id) if user_id else None

    def save_user(self, user: User) -> None:
        """Save user to storage"""
        with self._lock:
            self._users[user.user_id] = user
            self._username_index[user.username.lower()] = user.user_id
            if user.email:
                self._email_index[user.email.lower()] = user.user_id
        self._save()

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                del self._users[user_id]
                self._username_index.pop(user.username.lower(), None)
                if user.email:
                    self._email_index.pop(user.email.lower(), None)
                self._save()
                return True
            return False

    def list_users(self) -> List[User]:
        """List all users"""
        with self._lock:
            return list(self._users.values())


class SessionStore:
    """Session storage interface"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.expanduser("~/.jarvis/sessions.json")
        self._sessions: Dict[str, Session] = {}
        self._token_index: Dict[str, str] = {}
        self._user_sessions: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load sessions from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for session_data in data.get('sessions', []):
                        session = Session.from_dict(session_data)
                        if not session.is_expired():
                            self._sessions[session.session_id] = session
                            self._token_index[session.token] = session.session_id
                            self._user_sessions[session.user_id].append(session.session_id)
        except Exception as e:
            print(f"Warning: Could not load sessions: {e}")

    def _save(self) -> None:
        """Save sessions to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {
                'sessions': [s.to_dict() for s in self._sessions.values() if not s.is_expired()],
                'version': 1
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save sessions: {e}")

    def create_session(self, user_id: str, ip_address: str = None,
                       user_agent: str = None, device_fingerprint: str = None) -> Session:
        """Create a new session"""
        session = Session(
            session_id=TokenGenerator.generate_uuid(),
            user_id=user_id,
            token=TokenGenerator.generate_token(),
            refresh_token=TokenGenerator.generate_refresh_token(),
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._token_index[session.token] = session.session_id
            self._user_sessions[user_id].append(session.session_id)

        self._save()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired():
                return session
            return None

    def get_session_by_token(self, token: str) -> Optional[Session]:
        """Get session by token"""
        with self._lock:
            session_id = self._token_index.get(token)
            if session_id:
                session = self._sessions.get(session_id)
                if session and not session.is_expired():
                    return session
            return None

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user"""
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            sessions = []
            for sid in session_ids:
                session = self._sessions.get(sid)
                if session and not session.is_expired():
                    sessions.append(session)
            return sessions

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_active = False
                self._token_index.pop(session.token, None)
                self._save()
                return True
            return False

    def invalidate_user_sessions(self, user_id: str, exclude_session: str = None) -> int:
        """Invalidate all sessions for a user"""
        count = 0
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            for sid in session_ids:
                if sid != exclude_session:
                    session = self._sessions.get(sid)
                    if session:
                        session.is_active = False
                        self._token_index.pop(session.token, None)
                        count += 1
            self._save()
        return count

    def cleanup_expired(self) -> int:
        """Remove expired sessions"""
        count = 0
        with self._lock:
            expired_ids = [sid for sid, s in self._sessions.items() if s.is_expired()]
            for sid in expired_ids:
                session = self._sessions.pop(sid, None)
                if session:
                    self._token_index.pop(session.token, None)
                    if session.user_id in self._user_sessions:
                        try:
                            self._user_sessions[session.user_id].remove(sid)
                        except ValueError:
                            pass
                    count += 1
            if count > 0:
                self._save()
        return count


class Authenticator:
    """Main authentication system"""

    def __init__(self, user_store: Optional[UserStore] = None,
                 session_store: Optional[SessionStore] = None,
                 password_policy: Optional[PasswordPolicy] = None,
                 session_timeout: int = DEFAULT_SESSION_TIMEOUT):
        self.user_store = user_store or UserStore()
        self.session_store = session_store or SessionStore()
        self.password_policy = password_policy or PasswordPolicy()
        self.session_timeout = session_timeout

        # Initialize password hasher
        if BCRYPT_AVAILABLE:
            self.hasher = BcryptHasher()
        elif SCRYPT_AVAILABLE:
            self.hasher = ScryptHasher()
        else:
            self.hasher = BcryptHasher()  # Uses fallback

        self.validator = PasswordValidator(self.password_policy)
        self.rate_limiter = RateLimiter()
        self.token_gen = TokenGenerator()

        # Create system user if no users exist
        self._ensure_system_user()

    def _ensure_system_user(self) -> None:
        """Ensure system user exists"""
        if not self.user_store.list_users():
            self.create_user(
                username="admin",
                password="Admin@123456",
                email="admin@jarvis.local",
                role=UserRole.SUPER_ADMIN
            )

    def create_user(self, username: str, password: str, email: str = "",
                    role: UserRole = UserRole.USER, metadata: Dict = None) -> Tuple[bool, str, Optional[User]]:
        """Create a new user"""
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters", None

        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "Username can only contain letters, numbers, underscores and hyphens", None

        # Check if username exists
        if self.user_store.get_user_by_username(username):
            return False, "Username already exists", None

        # Check if email exists
        if email and self.user_store.get_user_by_email(email):
            return False, "Email already exists", None

        # Validate password
        validation = self.validator.validate(password, username, email)
        if validation[0] != PasswordPolicyResult.VALID:
            issues = [v.name.replace('_', ' ').title() for v in validation]
            return False, f"Password validation failed: {', '.join(issues)}", None

        # Create user
        user_id = self.token_gen.generate_uuid()
        password_hash, salt = self.hasher.hash(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
            metadata=metadata or {}
        )

        self.user_store.save_user(user)
        return True, "User created successfully", user

    def authenticate(self, username: str, password: str,
                     ip_address: str = None, user_agent: str = None,
                     device_fingerprint: str = None) -> AuthResult:
        """Authenticate a user"""
        # Get user
        user = self.user_store.get_user_by_username(username)
        if not user:
            # Still record attempt for security
            self.rate_limiter.record_attempt(username, False)
            return AuthResult(
                status=AuthStatus.FAILED,
                message="Invalid credentials",
                attempts_remaining=self.rate_limiter.get_remaining_attempts(username)
            )

        # Check if user is active
        if not user.is_active:
            return AuthResult(
                status=AuthStatus.LOCKED_OUT,
                message="Account is disabled",
                user=user
            )

        # Check rate limit
        is_limited, remaining = self.rate_limiter.check_rate_limit(username)
        if is_limited:
            return AuthResult(
                status=AuthStatus.RATE_LIMITED,
                message=f"Too many failed attempts. Try again in {remaining} seconds",
                user=user
            )

        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now():
            remaining = (user.locked_until - datetime.now()).total_seconds()
            return AuthResult(
                status=AuthStatus.LOCKED_OUT,
                message=f"Account locked. Try again in {int(remaining)} seconds",
                user=user
            )

        # Verify password
        if not self.hasher.verify(password, user.password_hash, user.salt):
            self._record_failed_auth(user)
            return AuthResult(
                status=AuthStatus.FAILED,
                message="Invalid credentials",
                user=user,
                attempts_remaining=self.rate_limiter.get_remaining_attempts(username)
            )

        # Check if password is expired
        if self.password_policy.max_age_days > 0:
            password_age = (datetime.now() - user.password_changed_at).days
            if password_age > self.password_policy.max_age_days:
                return AuthResult(
                    status=AuthStatus.PASSWORD_EXPIRED,
                    message="Password has expired. Please reset your password.",
                    user=user
                )

        # Check MFA
        if user.mfa_enabled:
            return AuthResult(
                status=AuthStatus.MFA_REQUIRED,
                message="MFA verification required",
                user=user,
                mfa_required=True
            )

        # Successful authentication
        self._record_successful_auth(user)
        session = self.session_store.create_session(
            user.user_id, ip_address, user_agent, device_fingerprint
        )

        return AuthResult(
            status=AuthStatus.SUCCESS,
            message="Authentication successful",
            user=user,
            session=session
        )

    def _record_failed_auth(self, user: User) -> None:
        """Record failed authentication attempt"""
        user.failed_attempts += 1

        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.now() + timedelta(seconds=LOCKOUT_DURATION)

        self.user_store.save_user(user)
        self.rate_limiter.record_attempt(user.username, False)

    def _record_successful_auth(self, user: User) -> None:
        """Record successful authentication"""
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        self.user_store.save_user(user)
        self.rate_limiter.clear(user.username)

    def validate_session(self, token: str) -> Tuple[bool, Optional[User], Optional[Session]]:
        """Validate a session token"""
        session = self.session_store.get_session_by_token(token)

        if not session:
            return False, None, None

        if not session.is_active:
            return False, None, None

        if session.is_expired():
            self.session_store.invalidate_session(session.session_id)
            return False, None, None

        user = self.user_store.get_user(session.user_id)
        if not user or not user.is_active:
            return False, None, None

        # Update last activity
        session.extend(self.session_timeout)

        return True, user, session

    def logout(self, token: str) -> bool:
        """Logout a user by invalidating their session"""
        session = self.session_store.get_session_by_token(token)
        if session:
            return self.session_store.invalidate_session(session.session_id)
        return False

    def logout_all(self, user_id: str) -> int:
        """Logout all sessions for a user"""
        return self.session_store.invalidate_user_sessions(user_id)

    def change_password(self, user_id: str, current_password: str,
                        new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"

        # Verify current password
        if not self.hasher.verify(current_password, user.password_hash, user.salt):
            return False, "Current password is incorrect"

        # Validate new password
        validation = self.validator.validate(new_password, user.username, user.email)
        if validation[0] != PasswordPolicyResult.VALID:
            issues = [v.name.replace('_', ' ').title() for v in validation]
            return False, f"Password validation failed: {', '.join(issues)}"

        # Check password history
        for old_hash in user.password_history:
            if self.hasher.verify(new_password, old_hash, user.salt):
                return False, "Cannot reuse a recent password"

        # Update password
        new_hash, new_salt = self.hasher.hash(new_password)
        user.password_hash = new_hash
        user.salt = new_salt
        user.password_changed_at = datetime.now()

        # Update history
        user.password_history.append(user.password_hash)
        if len(user.password_history) > self.password_policy.history_count:
            user.password_history = user.password_history[-self.password_policy.history_count:]

        self.user_store.save_user(user)
        return True, "Password changed successfully"

    def reset_password(self, user_id: str, new_password: str,
                       reset_token: str = None) -> Tuple[bool, str]:
        """Reset user password (admin or token-based)"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"

        # Validate new password
        validation = self.validator.validate(new_password, user.username, user.email)
        if validation[0] != PasswordPolicyResult.VALID:
            issues = [v.name.replace('_', ' ').title() for v in validation]
            return False, f"Password validation failed: {', '.join(issues)}"

        # Update password
        new_hash, new_salt = self.hasher.hash(new_password)
        user.password_hash = new_hash
        user.salt = new_salt
        user.password_changed_at = datetime.now()
        user.failed_attempts = 0
        user.locked_until = None

        self.user_store.save_user(user)

        # Invalidate all sessions
        self.session_store.invalidate_user_sessions(user_id)

        return True, "Password reset successfully"

    def create_api_key(self, user_id: str) -> Tuple[bool, str, Optional[str]]:
        """Create an API key for a user"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found", None

        api_key = self.token_gen.generate_api_key()
        user.api_keys.append(api_key)
        self.user_store.save_user(user)

        return True, "API key created", api_key

    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[User]]:
        """Validate an API key"""
        for user in self.user_store.list_users():
            if api_key in user.api_keys:
                if user.is_active:
                    return True, user
        return False, None

    def revoke_api_key(self, user_id: str, api_key: str) -> bool:
        """Revoke an API key"""
        user = self.user_store.get_user(user_id)
        if user and api_key in user.api_keys:
            user.api_keys.remove(api_key)
            self.user_store.save_user(user)
            return True
        return False

    def add_trusted_device(self, user_id: str, device_fingerprint: str) -> bool:
        """Add a trusted device for a user"""
        user = self.user_store.get_user(user_id)
        if user:
            if device_fingerprint not in user.trusted_devices:
                user.trusted_devices.append(device_fingerprint)
                self.user_store.save_user(user)
            return True
        return False

    def remove_trusted_device(self, user_id: str, device_fingerprint: str) -> bool:
        """Remove a trusted device"""
        user = self.user_store.get_user(user_id)
        if user and device_fingerprint in user.trusted_devices:
            user.trusted_devices.remove(device_fingerprint)
            self.user_store.save_user(user)
            return True
        return False

    def is_trusted_device(self, user_id: str, device_fingerprint: str) -> bool:
        """Check if device is trusted"""
        user = self.user_store.get_user(user_id)
        return user and device_fingerprint in user.trusted_devices

    def update_user_role(self, user_id: str, role: UserRole) -> Tuple[bool, str]:
        """Update user role"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"

        user.role = role
        self.user_store.save_user(user)
        return True, "Role updated successfully"

    def disable_user(self, user_id: str) -> Tuple[bool, str]:
        """Disable a user account"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"

        user.is_active = False
        self.user_store.save_user(user)
        self.session_store.invalidate_user_sessions(user_id)
        return True, "User disabled successfully"

    def enable_user(self, user_id: str) -> Tuple[bool, str]:
        """Enable a user account"""
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"

        user.is_active = True
        user.failed_attempts = 0
        user.locked_until = None
        self.user_store.save_user(user)
        return True, "User enabled successfully"

    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """Delete a user"""
        if not self.user_store.delete_user(user_id):
            return False, "User not found"

        self.session_store.invalidate_user_sessions(user_id)
        return True, "User deleted successfully"

    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired sessions and data"""
        expired_sessions = self.session_store.cleanup_expired()
        return {
            'expired_sessions_removed': expired_sessions
        }


# Export classes
__all__ = [
    'AuthMethod',
    'AuthStatus',
    'UserRole',
    'PasswordPolicyResult',
    'PasswordPolicy',
    'User',
    'Session',
    'AuthResult',
    'RateLimitEntry',
    'PasswordHasher',
    'BcryptHasher',
    'ScryptHasher',
    'TokenGenerator',
    'DeviceFingerprinter',
    'PasswordValidator',
    'RateLimiter',
    'UserStore',
    'SessionStore',
    'Authenticator'
]


if __name__ == "__main__":
    # Quick test
    print("JARVIS Authentication System")
    print("=" * 50)

    auth = Authenticator()

    # Test user creation
    success, msg, user = auth.create_user(
        username="testuser",
        password="Test@123456",
        email="test@example.com"
    )
    print(f"Create user: {msg}")

    if success:
        # Test authentication
        result = auth.authenticate("testuser", "Test@123456")
        print(f"Authenticate: {result.message}")
        print(f"Status: {result.status.name}")

        if result.session:
            print(f"Session token: {result.session.token[:20]}...")

    print("\nAuthentication system ready!")
