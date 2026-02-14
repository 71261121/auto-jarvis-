# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS v14 Ultimate - API Key Security Research
# ═══════════════════════════════════════════════════════════════════════════════

## SECTION A: EXECUTIVE SUMMARY

This document covers API key security for JARVIS on Android/Termux.

### A.1 Security Principles

1. Never store API keys in code
2. Never log API keys
3. Use environment variables
4. Encrypt stored keys
5. Rotate keys regularly

---

## SECTION B: KEY STORAGE

### B.1 Environment Variables (Recommended)

```bash
# In ~/.bashrc or Termux
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
```

### B.2 Encrypted Storage

```python
# For persistent storage with encryption
import hashlib
import base64

def encrypt_key(key: str, password: str) -> str:
    """Simple encryption for API keys"""
    key_bytes = key.encode()
    pass_bytes = password.encode()
    
    # Derive key from password
    derived = hashlib.sha256(pass_bytes).digest()
    
    # Simple XOR encryption
    encrypted = bytes(a ^ b for a, b in zip(key_bytes, derived * len(key_bytes)))
    return base64.b64encode(encrypted).decode()
```

---

## SECTION C: KEY USAGE

### C.1 Loading Keys

```python
import os

def get_api_key(service: str) -> str:
    """Get API key from environment"""
    key_name = f"{service.upper()}_API_KEY"
    key = os.environ.get(key_name)
    
    if not key:
        raise ValueError(f"API key not found: {key_name}")
    
    return key
```

### C.2 Never Log Keys

```python
def log_request(url: str, headers: dict):
    """Log request without exposing keys"""
    safe_headers = {
        k: '***' if 'auth' in k.lower() or 'key' in k.lower() else v
        for k, v in headers.items()
    }
    logging.info(f"Request to {url} with headers: {safe_headers}")
```

---

## SECTION D: ANDROID-SPECIFIC

### D.1 Termux Storage

- Keys stored in `~/.env` file
- File permissions: 600 (user read/write only)
- Never backed up to cloud

### D.2 Key Rotation

```python
def rotate_key(service: str, new_key: str):
    """Rotate an API key"""
    old_key = get_api_key(service)
    
    # Store old key temporarily
    backup = encrypt_key(old_key, get_device_id())
    
    # Update key
    os.environ[f"{service.upper()}_API_KEY"] = new_key
    
    # Log rotation (without keys)
    logging.info(f"Rotated API key for {service}")
```

---

**Document Version: 1.0**
