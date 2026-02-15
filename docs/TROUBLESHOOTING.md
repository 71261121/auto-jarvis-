# JARVIS v14 Ultimate - Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Startup Issues](#startup-issues)
3. [AI/API Issues](#aiapi-issues)
4. [Memory Issues](#memory-issues)
5. [Performance Issues](#performance-issues)
6. [Self-Modification Issues](#self-modification-issues)
7. [Security Issues](#security-issues)
8. [Network Issues](#network-issues)
9. [Error Codes Reference](#error-codes-reference)
10. [Recovery Procedures](#recovery-procedures)

---

## Installation Issues

### Python Version Errors

**Error:**
```
Error: Python 3.9+ required, found 3.8.x
```

**Solution:**
```bash
# Termux
pkg upgrade python

# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# Create venv with new Python
python3.10 -m venv venv
source venv/bin/activate
```

### Permission Denied

**Error:**
```
Permission denied: '/data/data/com.termux/files/home/.jarvis'
```

**Solution:**
```bash
# Termux - grant storage permission
termux-setup-storage

# Click "Allow" when prompted

# If still failing, reset permissions
pkg install termux-api
termux-setup-storage
```

### Package Installation Failures

**Error:**
```
ERROR: Could not find a version that satisfies the requirement xyz
```

**Solution:**
```bash
# Update pip
pip install --upgrade pip

# Install package with specific version
pip install "package>=1.0,<2.0"

# If package fails, try alternatives
pip install --no-cache-dir package

# For compilation errors, install build tools
pkg install build-essential libffi openssl  # Termux
```

### Memory Errors During Install

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Close other applications
# Install dependencies one at a time
pip install click
pip install colorama
pip install requests
# etc.

# Or use --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

---

## Startup Issues

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'core'
```

**Solution:**
```bash
# Make sure you're in the correct directory
cd ~/jarvis_v14_ultimate

# Or set PYTHONPATH
export PYTHONPATH=/path/to/jarvis_v14_ultimate:$PYTHONPATH

# Run with correct path
python3 main.py
```

### Configuration Errors

**Error:**
```
ConfigError: Invalid configuration at ~/.jarvis/config.json
```

**Solution:**
```bash
# Validate configuration
python3 -c "import json; json.load(open('$HOME/.jarvis/config.json'))"

# Reset to defaults
rm ~/.jarvis/config.json
python3 main.py --setup

# Or manually fix JSON
python3 -m json.tool ~/.jarvis/config.json
```

### Dependency Errors

**Error:**
```
ImportError: cannot import name 'x' from 'y'
```

**Solution:**
```bash
# Reinstall problematic package
pip uninstall package
pip install package

# Check for version conflicts
pip check

# Update all packages
pip install --upgrade -r requirements.txt
```

---

## AI/API Issues

### API Key Errors

**Error:**
```
AuthenticationError: Invalid API key
```

**Solution:**
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY

# Set API key
export OPENROUTER_API_KEY='sk-or-v1-your-key'

# Verify key format (should start with sk-or-)
# Get new key from https://openrouter.ai/keys
```

### Rate Limiting

**Error:**
```
RateLimitError: Too many requests
```

**Solution:**
```bash
# Wait and retry
# JARVIS has automatic retry with backoff

# Reduce request frequency in config
jarvis> /config set rate_limiting.requests_per_minute 10

# Or enable local fallback
jarvis> /ai off  # Use local mode temporarily
```

### Model Not Available

**Error:**
```
ModelError: Model 'xyz' not available
```

**Solution:**
```bash
# List available models
jarvis> /models

# Switch to available model
jarvis> /model meta-llama/llama-3.1-8b-instruct:free

# Check model status
curl -s https://openrouter.ai/api/v1/models | jq '.data[].id'
```

### Timeout Errors

**Error:**
```
TimeoutError: Request timed out
```

**Solution:**
```bash
# Increase timeout
jarvis> /config set ai.timeout 60

# Check network connection
ping -c 3 openrouter.ai

# Use proxy if needed
export HTTPS_PROXY=http://proxy:8080
```

### Empty AI Responses

**Error:**
```
AI returned empty response
```

**Solution:**
```bash
# Check API status
curl -s https://status.openrouter.ai/

# Try different model
jarvis> /model mistralai/mistral-7b-instruct:free

# Enable debug mode for details
jarvis> /debug on
```

---

## Memory Issues

### Out of Memory

**Error:**
```
MemoryError: Cannot allocate memory
```

**Solution:**
```bash
# Check memory usage
jarvis> /memory

# Clear caches
jarvis> /memory clear cache

# Reduce history size
jarvis> /config set memory.max_history 50

# Enable aggressive cleanup
jarvis> /config set memory.gc_threshold 0.7

# Restart JARVIS
```

### High Memory Usage

**Symptoms:**
- Slow responses
- Device lagging
- Crashes

**Solution:**
```bash
# Check what's using memory
jarvis> /memory details

# Reduce context window
jarvis> /config set ai.context_window 2048

# Disable unnecessary features
jarvis> /config set interface.syntax_highlighting false

# Clear conversation
jarvis> /context clear
```

### Memory Leaks

**Symptoms:**
- Memory usage grows over time
- Performance degrades

**Solution:**
```python
# Debug memory usage
import tracemalloc
tracemalloc.start()

# Use JARVIS normally, then check
snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics('lineno')
for stat in top[:10]:
    print(stat)
```

---

## Performance Issues

### Slow Startup

**Symptoms:**
- Takes >10 seconds to start

**Solution:**
```bash
# Enable lazy loading
jarvis> /config set performance.lazy_loading true

# Reduce startup imports
# Edit main.py to use minimal imports

# Profile startup
python3 -X importtime main.py 2>&1 | head -50
```

### Slow Responses

**Symptoms:**
- AI responses take >30 seconds

**Solution:**
```bash
# Check network speed
speedtest-cli

# Use faster model
jarvis> /model mistralai/mistral-7b-instruct:free

# Enable streaming
jarvis> /config set ai.streaming true

# Check API latency
curl -w "@curl-format.txt" -o /dev/null -s https://openrouter.ai/api/v1/chat/completions
```

### Slow File Operations

**Symptoms:**
- File reads/writes are slow

**Solution:**
```bash
# Check disk space
df -h

# Check storage health
# On Termux, device storage speed varies

# Reduce cache size
jarvis> /config set storage.cache_max_size_mb 50

# Clear old caches
rm -rf ~/.jarvis/cache/*
```

---

## Self-Modification Issues

### Modification Rejected

**Error:**
```
SafetyError: Modification rejected - dangerous pattern detected
```

**Solution:**
```bash
# Review what was flagged
jarvis> /debug on
jarvis> /modify file.py "description"

# If safe, use force flag (carefully!)
jarvis> /modify file.py "description" --force

# Or adjust safety settings
jarvis> /config set self_modification.safety_level moderate
```

### Backup Restore Failed

**Error:**
```
RestoreError: Could not restore from backup_xxx
```

**Solution:**
```bash
# Check backup exists
jarvis> /backup list

# Verify backup integrity
ls -la ~/.jarvis/backups/backup_xxx/

# Manual restore
cp -r ~/.jarvis/backups/backup_xxx/* .

# Create fresh backup
jarvis> /backup create "manual backup"
```

### Test Failures After Modification

**Error:**
```
TestError: 3 tests failed after modification
```

**Solution:**
```bash
# View failed tests
jarvis> /test failed

# Rollback modification
jarvis> /rollback last

# Fix issues manually and re-run tests
python3 -m pytest tests/ -v

# Re-attempt modification with fixes
```

---

## Security Issues

### Login Failures

**Error:**
```
AuthError: Invalid credentials
```

**Solution:**
```bash
# Reset password
jarvis> /auth reset-password

# Or manually reset
python3 -c "
from security import Authenticator
auth = Authenticator()
auth.reset_user('admin')
"

# Check for account lockout
jarvis> /auth status
```

### Encryption Errors

**Error:**
```
DecryptionError: Failed to decrypt data
```

**Solution:**
```bash
# Verify password
# Check if key file exists
ls ~/.jarvis/keys/

# Reset encryption keys (WARNING: loses encrypted data)
jarvis> /security reset-keys

# Restore from backup if needed
jarvis> /restore backup_before_encryption
```

### Audit Log Issues

**Error:**
```
AuditError: Audit log corrupted
```

**Solution:**
```bash
# Repair audit log
jarvis> /audit repair

# Or manually fix
python3 -c "
from security.audit import AuditIntegrityChecker
checker = AuditIntegrityChecker()
checker.repair('~/.jarvis/logs/audit.log')
"
```

---

## Network Issues

### Connection Refused

**Error:**
```
ConnectionRefusedError: Connection refused
```

**Solution:**
```bash
# Check internet
ping -c 3 google.com

# Check DNS
nslookup openrouter.ai

# Try alternative DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Check firewall
# On Termux, usually no firewall issues
```

### SSL Errors

**Error:**
```
SSLError: Certificate verify failed
```

**Solution:**
```bash
# Update certificates
pip install --upgrade certifi

# On Termux
pkg install ca-certificates

# Temporary workaround (not recommended for production)
export PYTHONHTTPSVERIFY=0
```

### Proxy Issues

**Error:**
```
ProxyError: Unable to connect to proxy
```

**Solution:**
```bash
# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Clear proxy
unset HTTP_PROXY HTTPS_PROXY

# Or configure correct proxy
export HTTPS_PROXY=http://correct-proxy:8080

# Update config
jarvis> /config set network.proxy http://proxy:8080
```

---

## Error Codes Reference

### Error Code Format

```
JARVIS-[MODULE]-[CODE]: [Message]
```

### Core Errors (JARVIS-CORE-xxx)

| Code | Message | Solution |
|------|---------|----------|
| 001 | Import failed | Install missing package |
| 002 | Configuration invalid | Fix or reset config |
| 003 | Cache corrupted | Clear cache |
| 004 | State machine error | Reset state |

### AI Errors (JARVIS-AI-xxx)

| Code | Message | Solution |
|------|---------|----------|
| 101 | API key invalid | Set valid API key |
| 102 | Rate limited | Wait or reduce requests |
| 103 | Model unavailable | Switch model |
| 104 | Request timeout | Increase timeout |
| 105 | Response parse error | Enable debug mode |

### Self-Modification Errors (JARVIS-MOD-xxx)

| Code | Message | Solution |
|------|---------|----------|
| 201 | Dangerous pattern | Review or force |
| 202 | Backup failed | Check disk space |
| 203 | Test failed | Rollback or fix |
| 204 | Syntax error | Fix code |
| 205 | Import validation failed | Fix imports |

### Security Errors (JARVIS-SEC-xxx)

| Code | Message | Solution |
|------|---------|----------|
| 301 | Authentication failed | Check credentials |
| 302 | Permission denied | Check user role |
| 303 | Encryption failed | Check key |
| 304 | Audit failed | Repair logs |
| 305 | Sandbox violation | Adjust policy |

---

## Recovery Procedures

### Full System Reset

```bash
# Backup data first
cp -r ~/.jarvis ~/.jarvis_backup

# Uninstall
rm -rf ~/jarvis_v14_ultimate

# Clear configuration
rm -rf ~/.jarvis

# Reinstall
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash

# Restore data
cp -r ~/.jarvis_backup/data ~/.jarvis/
```

### Configuration Recovery

```bash
# Reset to defaults
jarvis> /config reset

# Or manually
rm ~/.jarvis/config.json
python3 main.py --setup
```

### Backup Recovery

```bash
# List backups
jarvis> /backup list

# Restore specific backup
jarvis> /backup restore backup_20240114_001

# Manual restore
cd ~/.jarvis/backups/backup_xxx
tar -xzf backup.tar.gz -C ~/jarvis_v14_ultimate
```

### Emergency Mode

```bash
# Start in safe mode
python3 main.py --safe

# This disables:
# - Self-modification
# - Network features
# - Plugins
# - Background tasks

# Minimal startup
python3 main.py --minimal
```

### Diagnostic Mode

```bash
# Run diagnostics
python3 main.py --diagnose

# Output example:
# ✓ Python 3.11.0
# ✓ Memory: 1.8GB available
# ✓ Network: Connected
# ✓ API key: Set
# ✗ Cache: Corrupted (run /cache repair)
# ✓ Security: All checks passed
```

---

## Getting Help

### Collect Debug Information

```bash
# Generate debug report
python3 main.py --debug-report > debug_report.txt

# Include this in bug reports
# Contains:
# - System info
# - Python version
# - Installed packages
# - Configuration (sanitized)
# - Recent logs
# - Memory status
```

### Support Channels

1. **GitHub Issues**: https://github.com/jarvis/jarvis-v14/issues
2. **Documentation**: Check other docs in `docs/`
3. **Community**: GitHub Discussions

### When Reporting Issues

Include:
1. Error message (full text)
2. Steps to reproduce
3. Debug report output
4. Your environment (Termux version, Android version)
5. What you've already tried

---

*Last Updated: Version 14.0.0*
