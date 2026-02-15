#!/usr/bin/env python3
"""
JARVIS Security System
Ultra-Advanced Security Module for Self-Modifying AI

This package provides comprehensive security features including:
- Authentication and Authorization
- Data Encryption and Hashing
- Execution Sandboxing
- Audit Logging
- Threat Detection
- Permission Management
- Key Management

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

from security.auth import (
    AuthMethod,
    AuthStatus,
    UserRole,
    PasswordPolicyResult,
    PasswordPolicy,
    User,
    Session,
    AuthResult,
    RateLimitEntry,
    PasswordHasher,
    BcryptHasher,
    ScryptHasher,
    TokenGenerator,
    DeviceFingerprinter,
    PasswordValidator,
    RateLimiter,
    UserStore,
    SessionStore,
    Authenticator,
)

from security.encryption import (
    EncryptionAlgorithm,
    HashAlgorithm,
    KeyDerivationFunction,
    EncryptedData,
    SecureRandom,
    HashFunction,
    HMACUtils,
    KeyDerivation,
    SymmetricCipher,
    AESGCMCipher,
    ChaCha20Cipher,
    FernetCipher,
    AsymmetricEncryption,
    EncryptionManager,
    SecureStorage,
)

from security.sandbox import (
    SecurityTier,
    ExecutionStatus,
    AccessType,
    ExecutionResult,
    SecurityPolicy,
    ImportValidator,
    RestrictedBuiltins,
    ResourceLimiter,
    TimeoutManager,
    OutputCapture,
    CodeAnalyzer,
    ExecutionSandbox,
    SandboxManager,
)

from security.audit import (
    AuditEventType,
    AuditSeverity,
    AuditCategory,
    AuditEvent,
    AuditFilter,
    AuditStorage,
    FileAuditStorage,
    MemoryAuditStorage,
    AuditAlert,
    AuditIntegrityChecker,
    AuditReporter,
    AuditLogger,
    get_audit_logger,
)

from security.threat_detect import (
    ThreatType,
    ThreatSeverity,
    ThreatStatus,
    ThreatIndicator,
    Threat,
    ThreatPattern,
    AttackSignatureDatabase,
    IPReputationChecker,
    AnomalyDetector,
    BruteForceDetector,
    ThreatScorer,
    ThreatDetector,
)

from security.permissions import (
    PermissionAction,
    ResourceScope,
    Permission,
    Role,
    PermissionCheckResult,
    PermissionCondition,
    TimeCondition,
    IPCondition,
    AttributeCondition,
    PermissionCache,
    RoleStore,
    PermissionManager,
    require_permission,
    require_role,
    get_permission_manager,
)

from security.keys import (
    KeyType,
    KeyStatus,
    KeyAlgorithm,
    KeyMetadata,
    Key,
    KeyGenerator,
    KeyStorage,
    KeyRotationManager,
    KeyAccessAudit,
    KeyManager,
    get_key_manager,
)


__all__ = [
    # Authentication
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
    'Authenticator',

    # Encryption
    'EncryptionAlgorithm',
    'HashAlgorithm',
    'KeyDerivationFunction',
    'EncryptedData',
    'SecureRandom',
    'HashFunction',
    'HMACUtils',
    'KeyDerivation',
    'SymmetricCipher',
    'AESGCMCipher',
    'ChaCha20Cipher',
    'FernetCipher',
    'AsymmetricEncryption',
    'EncryptionManager',
    'SecureStorage',

    # Sandbox
    'SecurityTier',
    'ExecutionStatus',
    'AccessType',
    'ExecutionResult',
    'SecurityPolicy',
    'ImportValidator',
    'RestrictedBuiltins',
    'ResourceLimiter',
    'TimeoutManager',
    'OutputCapture',
    'CodeAnalyzer',
    'ExecutionSandbox',
    'SandboxManager',

    # Audit
    'AuditEventType',
    'AuditSeverity',
    'AuditCategory',
    'AuditEvent',
    'AuditFilter',
    'AuditStorage',
    'FileAuditStorage',
    'MemoryAuditStorage',
    'AuditAlert',
    'AuditIntegrityChecker',
    'AuditReporter',
    'AuditLogger',
    'get_audit_logger',

    # Threat Detection
    'ThreatType',
    'ThreatSeverity',
    'ThreatStatus',
    'ThreatIndicator',
    'Threat',
    'ThreatPattern',
    'AttackSignatureDatabase',
    'IPReputationChecker',
    'AnomalyDetector',
    'BruteForceDetector',
    'ThreatScorer',
    'ThreatDetector',

    # Permissions
    'PermissionAction',
    'ResourceScope',
    'Permission',
    'Role',
    'PermissionCheckResult',
    'PermissionCondition',
    'TimeCondition',
    'IPCondition',
    'AttributeCondition',
    'PermissionCache',
    'RoleStore',
    'PermissionManager',
    'require_permission',
    'require_role',
    'get_permission_manager',

    # Keys
    'KeyType',
    'KeyStatus',
    'KeyAlgorithm',
    'KeyMetadata',
    'Key',
    'KeyGenerator',
    'KeyStorage',
    'KeyRotationManager',
    'KeyAccessAudit',
    'KeyManager',
    'get_key_manager',
]

__version__ = "1.0.0"
__author__ = "JARVIS Self-Modifying AI Project"
