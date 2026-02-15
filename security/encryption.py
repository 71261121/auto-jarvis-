#!/usr/bin/env python3
"""
JARVIS Encryption System
Ultra-Advanced Data Encryption Module for Self-Modifying AI

Features:
- Multiple encryption algorithms with fallbacks
- Symmetric encryption (AES-256-GCM, ChaCha20-Poly1305)
- Asymmetric encryption (RSA-like operations)
- Key derivation functions (PBKDF2, HKDF, Argon2-like)
- Hash functions with multiple algorithms
- HMAC for message authentication
- Secure random number generation
- Data encoding/decoding (Base64, Hex)
- Encrypted file storage
- Encrypted message passing

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import json
import base64
import hashlib
import hmac
import secrets
import struct
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Union, Callable, ByteString
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import zlib

# Try importing optional crypto libraries
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives import serialization
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import nacl.secret
    import nacl.public
    import nacl.utils
    import nacl.pwhash
    import nacl.encoding
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False


# Constants
DEFAULT_KEY_SIZE = 32  # 256 bits
DEFAULT_IV_SIZE = 16   # 128 bits
DEFAULT_SALT_SIZE = 32
DEFAULT_TAG_SIZE = 16  # 128 bits for GCM
PBKDF2_ITERATIONS = 100000
HKDF_INFO = b"jarvis_encryption"
MAX_ENCRYPTION_SIZE = 100 * 1024 * 1024  # 100 MB


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    XOR_FALLBACK = "xor-fallback"
    FERNET = "fernet"
    NACL_SECRET = "nacl-secret"


class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3-256"
    SHA3_512 = "sha3-512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class KeyDerivationFunction(Enum):
    """Key derivation functions"""
    PBKDF2 = "pbkdf2"
    HKDF = "hkdf"
    SCRYPT = "scrypt"
    ARGON2ID = "argon2id"


@dataclass
class EncryptedData:
    """Container for encrypted data"""
    ciphertext: bytes
    iv: bytes
    tag: bytes = b""
    algorithm: str = "aes-256-gcm"
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'ciphertext': base64.b64encode(self.ciphertext).decode('utf-8'),
            'iv': base64.b64encode(self.iv).decode('utf-8'),
            'tag': base64.b64encode(self.tag).decode('utf-8') if self.tag else "",
            'algorithm': self.algorithm,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage"""
        data = self.to_dict()
        return json.dumps(data).encode('utf-8')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary"""
        return cls(
            ciphertext=base64.b64decode(data['ciphertext']),
            iv=base64.b64decode(data['iv']),
            tag=base64.b64decode(data['tag']) if data.get('tag') else b"",
            algorithm=data.get('algorithm', 'aes-256-gcm'),
            version=data.get('version', 1),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(),
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncryptedData':
        """Deserialize from bytes"""
        decoded = json.loads(data.decode('utf-8'))
        return cls.from_dict(decoded)


class SecureRandom:
    """Cryptographically secure random number generator"""

    @staticmethod
    def bytes(length: int) -> bytes:
        """Generate secure random bytes"""
        return secrets.token_bytes(length)

    @staticmethod
    def hex(length: int) -> str:
        """Generate secure random hex string"""
        return secrets.token_hex(length)

    @staticmethod
    def url_safe(length: int) -> str:
        """Generate URL-safe random string"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def integer(min_val: int, max_val: int) -> int:
        """Generate secure random integer in range"""
        return secrets.randbelow(max_val - min_val + 1) + min_val

    @staticmethod
    def choice(sequence: List[Any]) -> Any:
        """Select random element from sequence"""
        return secrets.choice(sequence)

    @staticmethod
    def uuid() -> str:
        """Generate UUID4"""
        import uuid
        return str(uuid.uuid4())


class HashFunction:
    """Hash function utilities"""

    ALGORITHM_MAP = {
        HashAlgorithm.SHA256: hashlib.sha256,
        HashAlgorithm.SHA384: hashlib.sha384,
        HashAlgorithm.SHA512: hashlib.sha512,
        HashAlgorithm.SHA3_256: hashlib.sha3_256,
        HashAlgorithm.SHA3_512: hashlib.sha3_512,
        HashAlgorithm.BLAKE2B: hashlib.blake2b,
        HashAlgorithm.BLAKE2S: hashlib.blake2s,
    }

    @classmethod
    def hash(cls, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data with specified algorithm"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        hasher = cls.ALGORITHM_MAP.get(algorithm, hashlib.sha256)
        return hasher(data).hexdigest()

    @classmethod
    def hash_bytes(cls, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
        """Hash data and return bytes"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        hasher = cls.ALGORITHM_MAP.get(algorithm, hashlib.sha256)
        return hasher(data).digest()

    @classmethod
    def file_hash(cls, filepath: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash a file"""
        hasher = cls.ALGORITHM_MAP.get(algorithm, hashlib.sha256)()

        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    @classmethod
    def verify_hash(cls, data: Union[str, bytes], expected_hash: str,
                    algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify data against expected hash"""
        computed = cls.hash(data, algorithm)
        return hmac.compare_digest(computed, expected_hash)

    @classmethod
    def double_hash(cls, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Double hash for extra security"""
        first_hash = cls.hash(data, algorithm)
        return cls.hash(first_hash, algorithm)


class HMACUtils:
    """HMAC utilities for message authentication"""

    @staticmethod
    def sign(data: Union[str, bytes], key: Union[str, bytes],
             algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Sign data with HMAC"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')

        algo_name = algorithm.value
        computed = hmac.new(key, data, algo_name).hexdigest()
        return computed

    @staticmethod
    def verify(data: Union[str, bytes], signature: str, key: Union[str, bytes],
               algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature"""
        expected = HMACUtils.sign(data, key, algorithm)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def sign_bytes(data: bytes, key: bytes,
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
        """Sign data with HMAC, return bytes"""
        algo_name = algorithm.value
        return hmac.new(key, data, algo_name).digest()


class KeyDerivation:
    """Key derivation functions"""

    @staticmethod
    def pbkdf2(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS,
                key_length: int = DEFAULT_KEY_SIZE) -> bytes:
        """Derive key using PBKDF2"""
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            return kdf.derive(password.encode('utf-8'))
        else:
            # Fallback implementation
            return hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations,
                dklen=key_length
            )

    @staticmethod
    def hkdf(input_key: bytes, salt: bytes, info: bytes = HKDF_INFO,
             key_length: int = DEFAULT_KEY_SIZE) -> bytes:
        """Derive key using HKDF"""
        if CRYPTOGRAPHY_AVAILABLE:
            hkdf_obj = HKDF(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                info=info,
                backend=default_backend()
            )
            return hkdf_obj.derive(input_key)
        else:
            # Fallback using HMAC
            prk = hmac.new(salt, input_key, hashlib.sha256).digest()
            okm = b""
            t = b""
            for i in range((key_length + 31) // 32):
                t = hmac.new(prk, t + info + bytes([i + 1]), hashlib.sha256).digest()
                okm += t
            return okm[:key_length]

    @staticmethod
    def scrypt(password: str, salt: bytes, n: int = 2**14, r: int = 8,
               p: int = 1, key_length: int = DEFAULT_KEY_SIZE) -> bytes:
        """Derive key using scrypt"""
        if hasattr(hashlib, 'scrypt'):
            return hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt,
                n=n,
                r=r,
                p=p,
                dklen=key_length
            )
        else:
            # Fallback to PBKDF2
            return KeyDerivation.pbkdf2(password, salt, iterations=PBKDF2_ITERATIONS,
                                        key_length=key_length)

    @staticmethod
    def derive_from_password(password: str, salt: bytes = None,
                             kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2) -> Tuple[bytes, bytes]:
        """Derive encryption key from password"""
        if salt is None:
            salt = SecureRandom.bytes(DEFAULT_SALT_SIZE)

        if kdf == KeyDerivationFunction.PBKDF2:
            key = KeyDerivation.pbkdf2(password, salt)
        elif kdf == KeyDerivationFunction.SCRYPT:
            key = KeyDerivation.scrypt(password, salt)
        else:
            key = KeyDerivation.pbkdf2(password, salt)

        return key, salt


class SymmetricCipher(ABC):
    """Abstract base class for symmetric ciphers"""

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Encrypt data"""
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data"""
        pass


class AESGCMCipher(SymmetricCipher):
    """AES-256-GCM cipher implementation"""

    def encrypt(self, plaintext: bytes, key: bytes, associated_data: bytes = None) -> EncryptedData:
        """Encrypt using AES-256-GCM"""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")

        if CRYPTOGRAPHY_AVAILABLE:
            iv = SecureRandom.bytes(12)  # 96 bits for GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            if associated_data:
                encryptor.authenticate_additional_data(associated_data)

            ciphertext = encryptor.update(plaintext) + encryptor.finalize()

            return EncryptedData(
                ciphertext=ciphertext,
                iv=iv,
                tag=encryptor.tag,
                algorithm="aes-256-gcm",
                metadata={'aad': base64.b64encode(associated_data).decode() if associated_data else None}
            )
        else:
            # Fallback to XOR with HMAC
            return self._fallback_encrypt(plaintext, key)

    def _fallback_encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Fallback encryption when cryptography not available"""
        iv = SecureRandom.bytes(DEFAULT_IV_SIZE)

        # Simple XOR-based encryption with HMAC
        key_stream = hashlib.sha256(iv + key).digest()
        while len(key_stream) < len(plaintext) + DEFAULT_IV_SIZE:
            key_stream += hashlib.sha256(key_stream + key).digest()

        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream[:len(plaintext)]))
        tag = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()[:DEFAULT_TAG_SIZE]

        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv,
            tag=tag,
            algorithm="xor-fallback",
            metadata={'warning': 'Using fallback encryption'}
        )

    def decrypt(self, encrypted_data: EncryptedData, key: bytes,
                associated_data: bytes = None) -> bytes:
        """Decrypt using AES-256-GCM"""
        if encrypted_data.algorithm == "xor-fallback":
            return self._fallback_decrypt(encrypted_data, key)

        if CRYPTOGRAPHY_AVAILABLE:
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(encrypted_data.iv, encrypted_data.tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            if associated_data:
                decryptor.authenticate_additional_data(associated_data)

            return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
        else:
            return self._fallback_decrypt(encrypted_data, key)

    def _fallback_decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Fallback decryption"""
        # Verify tag
        expected_tag = hmac.new(key, encrypted_data.iv + encrypted_data.ciphertext,
                                hashlib.sha256).digest()[:DEFAULT_TAG_SIZE]
        if not hmac.compare_digest(encrypted_data.tag, expected_tag):
            raise ValueError("Authentication tag verification failed")

        # XOR decryption
        key_stream = hashlib.sha256(encrypted_data.iv + key).digest()
        while len(key_stream) < len(encrypted_data.ciphertext) + DEFAULT_IV_SIZE:
            key_stream += hashlib.sha256(key_stream + key).digest()

        return bytes(a ^ b for a, b in zip(encrypted_data.ciphertext, key_stream[:len(encrypted_data.ciphertext)]))


class ChaCha20Cipher(SymmetricCipher):
    """ChaCha20-Poly1305 cipher implementation"""

    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Encrypt using ChaCha20-Poly1305"""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")

        if NACL_AVAILABLE:
            box = nacl.secret.SecretBox(key)
            nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
            ciphertext = box.encrypt(plaintext, nonce)
            # nacl prepends nonce to ciphertext
            return EncryptedData(
                ciphertext=ciphertext[box.NONCE_SIZE:],
                iv=ciphertext[:box.NONCE_SIZE],
                algorithm="chacha20-poly1305"
            )
        else:
            # Fallback to AES-GCM
            return AESGCMCipher().encrypt(plaintext, key)

    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        if NACL_AVAILABLE:
            box = nacl.secret.SecretBox(key)
            # Combine nonce and ciphertext
            combined = encrypted_data.iv + encrypted_data.ciphertext
            return box.decrypt(combined)
        else:
            return AESGCMCipher().decrypt(encrypted_data, key)


class FernetCipher(SymmetricCipher):
    """Fernet cipher implementation (AES-128-CBC with HMAC)"""

    def __init__(self):
        self._key = None

    def generate_key(self) -> bytes:
        """Generate a Fernet-compatible key"""
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.fernet import Fernet
            return Fernet.generate_key()
        else:
            # Fallback key
            return base64.urlsafe_b64encode(SecureRandom.bytes(32))

    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Encrypt using Fernet"""
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.fernet import Fernet
            f = Fernet(key)
            ciphertext = f.encrypt(plaintext)
            return EncryptedData(
                ciphertext=ciphertext,
                iv=b"",  # Fernet handles IV internally
                tag=b"",
                algorithm="fernet"
            )
        else:
            return AESGCMCipher().encrypt(plaintext, key[:32])

    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt using Fernet"""
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.fernet import Fernet
            f = Fernet(key)
            return f.decrypt(encrypted_data.ciphertext)
        else:
            return AESGCMCipher().decrypt(encrypted_data, key[:32])


class AsymmetricEncryption:
    """Asymmetric encryption utilities"""

    def __init__(self):
        self._private_key = None
        self._public_key = None

    def generate_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA keypair"""
        if CRYPTOGRAPHY_AVAILABLE:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )

            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            self._private_key = private_key
            self._public_key = private_key.public_key()

            return private_pem, public_pem
        elif NACL_AVAILABLE:
            private_key = nacl.public.PrivateKey.generate()
            public_key = private_key.public_key

            return bytes(private_key), bytes(public_key)
        else:
            # Fallback - not truly asymmetric
            key = SecureRandom.bytes(32)
            return key, key

    def encrypt_public(self, plaintext: bytes, public_key_bytes: bytes) -> bytes:
        """Encrypt with public key"""
        if CRYPTOGRAPHY_AVAILABLE:
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=default_backend()
            )
            ciphertext = public_key.encrypt(
                plaintext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return ciphertext
        elif NACL_AVAILABLE:
            public_key = nacl.public.PublicKey(public_key_bytes)
            sealed_box = nacl.public.SealedBox(public_key)
            return sealed_box.encrypt(plaintext)
        else:
            # Fallback to symmetric
            key = hashlib.sha256(public_key_bytes).digest()
            return AESGCMCipher().encrypt(plaintext, key).ciphertext

    def decrypt_private(self, ciphertext: bytes, private_key_bytes: bytes) -> bytes:
        """Decrypt with private key"""
        if CRYPTOGRAPHY_AVAILABLE:
            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=None,
                backend=default_backend()
            )
            plaintext = private_key.decrypt(
                ciphertext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return plaintext
        elif NACL_AVAILABLE:
            private_key = nacl.public.PrivateKey(private_key_bytes)
            sealed_box = nacl.public.SealedBox(private_key)
            return sealed_box.decrypt(ciphertext)
        else:
            # Fallback
            key = hashlib.sha256(private_key_bytes).digest()
            return AESGCMCipher().decrypt(
                EncryptedData(ciphertext=ciphertext, iv=b"\x00"*16),
                key
            )


class EncryptionManager:
    """Main encryption manager for JARVIS"""

    def __init__(self, default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM):
        self.default_algorithm = default_algorithm
        self._ciphers = {
            EncryptionAlgorithm.AES_256_GCM: AESGCMCipher(),
            EncryptionAlgorithm.CHACHA20_POLY1305: ChaCha20Cipher(),
            EncryptionAlgorithm.FERNET: FernetCipher(),
        }
        self._key_cache: Dict[str, bytes] = {}
        self._lock = threading.Lock()

    def generate_key(self) -> bytes:
        """Generate a new encryption key"""
        return SecureRandom.bytes(DEFAULT_KEY_SIZE)

    def encrypt(self, plaintext: Union[str, bytes], key: bytes,
                algorithm: EncryptionAlgorithm = None,
                associated_data: bytes = None) -> EncryptedData:
        """Encrypt data with specified algorithm"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        if len(plaintext) > MAX_ENCRYPTION_SIZE:
            raise ValueError(f"Data too large. Max size: {MAX_ENCRYPTION_SIZE} bytes")

        algorithm = algorithm or self.default_algorithm
        cipher = self._ciphers.get(algorithm, AESGCMCipher())

        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return cipher.encrypt(plaintext, key, associated_data)
        else:
            return cipher.encrypt(plaintext, key)

    def decrypt(self, encrypted_data: EncryptedData, key: bytes,
                associated_data: bytes = None) -> bytes:
        """Decrypt data"""
        algorithm = EncryptionAlgorithm(encrypted_data.algorithm)
        cipher = self._ciphers.get(algorithm)

        if cipher is None:
            # Try all ciphers
            for cipher in self._ciphers.values():
                try:
                    return cipher.decrypt(encrypted_data, key)
                except Exception:
                    continue
            raise ValueError("Could not decrypt with any available cipher")

        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return cipher.decrypt(encrypted_data, key, associated_data)
        else:
            return cipher.decrypt(encrypted_data, key)

    def encrypt_string(self, plaintext: str, key: bytes) -> str:
        """Encrypt string and return base64 encoded result"""
        encrypted = self.encrypt(plaintext, key)
        return base64.b64encode(encrypted.to_bytes()).decode('utf-8')

    def decrypt_string(self, encrypted_b64: str, key: bytes) -> str:
        """Decrypt base64 encoded encrypted string"""
        encrypted_bytes = base64.b64decode(encrypted_b64)
        encrypted = EncryptedData.from_bytes(encrypted_bytes)
        return self.decrypt(encrypted, key).decode('utf-8')

    def encrypt_file(self, input_path: str, output_path: str, key: bytes) -> None:
        """Encrypt a file"""
        with open(input_path, 'rb') as f:
            plaintext = f.read()

        encrypted = self.encrypt(plaintext, key)
        encrypted_bytes = encrypted.to_bytes()

        with open(output_path, 'wb') as f:
            f.write(encrypted_bytes)

    def decrypt_file(self, input_path: str, output_path: str, key: bytes) -> None:
        """Decrypt a file"""
        with open(input_path, 'rb') as f:
            encrypted_bytes = f.read()

        encrypted = EncryptedData.from_bytes(encrypted_bytes)
        plaintext = self.decrypt(encrypted, key)

        with open(output_path, 'wb') as f:
            f.write(plaintext)

    def encrypt_dict(self, data: Dict[str, Any], key: bytes) -> str:
        """Encrypt a dictionary as JSON"""
        json_str = json.dumps(data)
        return self.encrypt_string(json_str, key)

    def decrypt_dict(self, encrypted: str, key: bytes) -> Dict[str, Any]:
        """Decrypt to dictionary"""
        json_str = self.decrypt_string(encrypted, key)
        return json.loads(json_str)

    def derive_key(self, password: str, salt: bytes = None,
                   kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2) -> Tuple[bytes, bytes]:
        """Derive encryption key from password"""
        return KeyDerivation.derive_from_password(password, salt, kdf)

    def hash_password(self, password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """Hash password with salt"""
        if salt is None:
            salt = SecureRandom.bytes(DEFAULT_SALT_SIZE)

        key, _ = KeyDerivation.derive_from_password(password, salt)
        return base64.b64encode(key).decode('utf-8'), salt

    def verify_password(self, password: str, password_hash: str, salt: bytes) -> bool:
        """Verify password against hash"""
        expected_key, _ = KeyDerivation.derive_from_password(password, salt)
        expected_hash = base64.b64encode(expected_key).decode('utf-8')
        return hmac.compare_digest(password_hash, expected_hash)

    def cache_key(self, key_id: str, key: bytes) -> None:
        """Cache a key for later use"""
        with self._lock:
            self._key_cache[key_id] = key

    def get_cached_key(self, key_id: str) -> Optional[bytes]:
        """Get cached key"""
        with self._lock:
            return self._key_cache.get(key_id)

    def clear_key_cache(self) -> None:
        """Clear all cached keys"""
        with self._lock:
            self._key_cache.clear()

    @staticmethod
    def hash_data(data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data with specified algorithm"""
        return HashFunction.hash(data, algorithm)

    @staticmethod
    def sign_data(data: Union[str, bytes], key: Union[str, bytes],
                  algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Sign data with HMAC"""
        return HMACUtils.sign(data, key, algorithm)

    @staticmethod
    def verify_signature(data: Union[str, bytes], signature: str,
                         key: Union[str, bytes],
                         algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature"""
        return HMACUtils.verify(data, signature, key, algorithm)


class SecureStorage:
    """Encrypted storage for sensitive data"""

    def __init__(self, storage_path: str, master_key: bytes = None,
                 password: str = None):
        self.storage_path = storage_path
        self.manager = EncryptionManager()

        if master_key:
            self.key = master_key
        elif password:
            self.key, _ = self.manager.derive_key(password)
        else:
            self.key = self.manager.generate_key()

        self._data: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load encrypted storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        """Save encrypted storage"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self._data, f)

    def set(self, key: str, value: Any) -> None:
        """Store encrypted value"""
        json_value = json.dumps(value)
        encrypted = self.manager.encrypt_string(json_value, self.key)
        self._data[key] = encrypted
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve and decrypt value"""
        encrypted = self._data.get(key)
        if encrypted is None:
            return default

        try:
            json_value = self.manager.decrypt_string(encrypted, self.key)
            return json.loads(json_value)
        except Exception:
            return default

    def delete(self, key: str) -> bool:
        """Delete a key"""
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all keys"""
        return list(self._data.keys())

    def clear(self) -> None:
        """Clear all data"""
        self._data.clear()
        self._save()


# Export classes
__all__ = [
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
    'SecureStorage'
]


if __name__ == "__main__":
    print("JARVIS Encryption System")
    print("=" * 50)

    manager = EncryptionManager()

    # Generate key
    key = manager.generate_key()
    print(f"Generated key: {base64.b64encode(key).decode()[:30]}...")

    # Test encryption
    plaintext = "Hello, JARVIS! This is a secret message."
    encrypted = manager.encrypt(plaintext, key)
    print(f"\nEncrypted: {encrypted.to_dict()}")

    # Test decryption
    decrypted = manager.decrypt(encrypted, key)
    print(f"Decrypted: {decrypted.decode()}")

    # Test string encryption
    encrypted_str = manager.encrypt_string(plaintext, key)
    decrypted_str = manager.decrypt_string(encrypted_str, key)
    print(f"\nString encryption test: {decrypted_str}")

    # Test hashing
    hash_result = EncryptionManager.hash_data("test data")
    print(f"\nHash: {hash_result}")

    print("\nEncryption system ready!")
