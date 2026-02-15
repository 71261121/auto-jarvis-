# JARVIS v14 Ultimate - Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Post-Installation Setup](#post-installation-setup)
6. [Verification](#verification)
7. [Uninstallation](#uninstallation)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **RAM** | 512MB | 1GB+ |
| **Storage** | 100MB | 500MB |
| **Python** | 3.9+ | 3.11+ |
| **Platform** | Termux/Linux | Termux on Android |

### Supported Platforms

JARVIS v14 is designed primarily for **Termux** on Android devices, specifically optimized for:

- **Device**: Realme 2 Pro Lite (RMP2402)
- **RAM**: 4GB (with ~1.5-2GB usable after Android overhead)
- **Architecture**: ARM64 (aarch64)
- **Android**: 8.1+ (Oreo)

The system also works on:
- Any Android device with Termux
- Linux distributions (Ubuntu, Debian, etc.)
- macOS (with some limitations)
- Windows (via WSL or Git Bash)

### Network Requirements

- Internet connection for AI features (OpenRouter API)
- HTTPS access to `openrouter.ai`
- GitHub access for updates

---

## Quick Start

### One-Line Installation

The fastest way to install JARVIS is using the installation script:

```bash
# Using curl
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash

# Or using wget
wget -qO- https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash
```

This script will:
1. Check your system requirements
2. Download JARVIS to `~/jarvis_v14_ultimate`
3. Install required dependencies
4. Create configuration files
5. Set up the launcher script

### Manual Quick Start

If you prefer manual installation:

```bash
# Clone the repository
git clone https://github.com/jarvis/jarvis-v14.git ~/jarvis_v14_ultimate
cd ~/jarvis_v14_ultimate

# Install dependencies
pip install -r requirements.txt

# Run JARVIS
python3 main.py
```

---

## Detailed Installation

### Step 1: Prepare Your Environment

#### On Termux (Android)

```bash
# Update packages
pkg update && pkg upgrade

# Install Python
pkg install python

# Install git (optional, for cloning)
pkg install git
```

#### On Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip git

# Fedora
sudo dnf install python3 python3-pip git

# Arch Linux
sudo pacman -S python python-pip git
```

### Step 2: Download JARVIS

#### Option A: Using Git (Recommended)

```bash
git clone https://github.com/jarvis/jarvis-v14.git ~/jarvis_v14_ultimate
cd ~/jarvis_v14_ultimate
```

#### Option B: Download Archive

```bash
# Download
curl -L https://github.com/jarvis/jarvis-v14/archive/refs/heads/main.zip -o jarvis.zip

# Extract
unzip jarvis.zip
mv jarvis-v14-main ~/jarvis_v14_ultimate
cd ~/jarvis_v14_ultimate
```

### Step 3: Install Dependencies

JARVIS uses a layered dependency system with fallbacks. Core dependencies are minimal:

```bash
# Install core dependencies (Class 0 - guaranteed safe)
pip install click colorama python-dotenv pyyaml requests tqdm schedule typing-extensions
```

For enhanced features, optionally install:

```bash
# Enhanced features (Class 1 - high probability)
pip install rich loguru httpx
```

### Step 4: Create Configuration

```bash
# Create JARVIS directory
mkdir -p ~/.jarvis/{data,cache,logs,backups}

# Create default configuration
python3 -c "
import json
config = {
    'general': {
        'app_name': 'JARVIS',
        'debug_mode': False
    },
    'ai': {
        'provider': 'openrouter',
        'model': 'meta-llama/llama-3.1-8b-instruct:free',
        'temperature': 0.7,
        'max_tokens': 2048
    }
}
with open('$HOME/.jarvis/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

### Step 5: Configure API Key

For AI features, you need an OpenRouter API key:

```bash
# Set environment variable (temporary)
export OPENROUTER_API_KEY='your-api-key-here'

# Or add to shell config (permanent)
echo "export OPENROUTER_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

To get an OpenRouter API key:
1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up for a free account
3. Generate an API key from the dashboard
4. Many models are available for free

---

## Platform-Specific Instructions

### Termux (Android)

Termux is the primary target platform. Additional setup:

```bash
# Install Termux from F-Droid (recommended) or Play Store
# Note: Play Store version is outdated, use F-Droid

# Grant storage permission (optional, for file access)
termux-setup-storage

# Install additional packages for best experience
pkg install termux-api  # For notifications

# Configure Termux settings
mkdir -p ~/.termux
echo "extra-keys = [['ESC','/','-','HOME','UP','END','DEL'],['TAB','CTRL','ALT','LEFT','DOWN','RIGHT','ENTER']]" > ~/.termux/termux.properties
```

### Linux Desktop

For Linux desktop systems:

```bash
# Create desktop entry (optional)
cat > ~/.local/share/applications/jarvis.desktop << EOF
[Desktop Entry]
Name=JARVIS
Comment=Self-Modifying AI Assistant
Exec=python3 ~/jarvis_v14_ultimate/main.py
Terminal=true
Type=Application
Categories=Utility;
EOF
```

### macOS

On macOS, some packages may require Homebrew:

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3

# Continue with standard installation
```

### Windows (WSL)

For Windows users, WSL is recommended:

```powershell
# In PowerShell (as Administrator)
wsl --install

# After WSL is set up, follow Linux instructions
```

---

## Post-Installation Setup

### First Run Wizard

On first run, JARVIS will guide you through setup:

```bash
python3 main.py --setup
```

This will:
1. Detect your environment
2. Install missing dependencies
3. Configure AI settings
4. Set up API keys
5. Create initial backup

### API Key Configuration

Configure your OpenRouter API key:

```bash
# Method 1: Environment variable
export OPENROUTER_API_KEY='sk-or-v1-...'

# Method 2: Configuration file
python3 -c "
import json
with open('$HOME/.jarvis/config.json', 'r') as f:
    config = json.load(f)
config['ai']['api_key'] = 'sk-or-v1-...'
with open('$HOME/.jarvis/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"

# Method 3: .env file
echo "OPENROUTER_API_KEY=sk-or-v1-..." > ~/jarvis_v14_ultimate/.env
```

### Shell Integration

Add JARVIS to your shell:

```bash
# Add alias
echo "alias jarvis='python3 ~/jarvis_v14_ultimate/main.py'" >> ~/.bashrc
source ~/.bashrc

# Now you can run
jarvis
```

---

## Verification

### Verify Installation

Run the verification script:

```bash
python3 ~/jarvis_v14_ultimate/main.py --verify
```

Or manually verify:

```python
# Test imports
python3 -c "
from core import get_importer, get_http_client
from core.ai import OpenRouterClient
print('✓ Core imports successful')

from interface import CLI, InputHandler
print('✓ Interface imports successful')

from install import EnvironmentDetector
print('✓ Install module successful')

from security import Authenticator, EncryptionManager
print('✓ Security module successful')

print('\\nAll modules verified!')
"
```

### Test AI Connection

```bash
# Test OpenRouter connection
python3 -c "
from core.ai import OpenRouterClient
client = OpenRouterClient()
response = client.chat('Hello, test message')
print('AI Response:', response.content[:100])
"
```

---

## Uninstallation

### Quick Uninstall

```bash
# Using the installation script
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash -s -- --uninstall
```

### Manual Uninstall

```bash
# Remove installation
rm -rf ~/jarvis_v14_ultimate
rm -f ~/jarvis

# Optional: Remove configuration and data
rm -rf ~/.jarvis

# Optional: Remove dependencies
pip uninstall -y click colorama python-dotenv pyyaml requests tqdm schedule
```

### Preserve Data

To keep your data while uninstalling:

```bash
# Backup data
cp -r ~/.jarvis ~/jarvis_backup

# Uninstall
rm -rf ~/jarvis_v14_ultimate

# Restore later
mv ~/jarvis_backup ~/.jarvis
```

---

## Troubleshooting

### Common Installation Issues

#### Python Version Too Old

```
Error: Python 3.9+ required, found 3.8.x
```

**Solution**: Update Python:
```bash
# Termux
pkg upgrade python

# Ubuntu
sudo apt install python3.10
```

#### Permission Denied

```
Error: Permission denied: '/data/data/com.termux/...'
```

**Solution**: 
```bash
# In Termux
termux-setup-storage
# Grant permission when prompted
```

#### Memory Error During Install

```
Error: Cannot allocate memory
```

**Solution**:
```bash
# Close other apps
# Install dependencies one at a time
pip install --no-cache-dir click
pip install --no-cache-dir requests
# etc.
```

#### Network Timeout

```
Error: Connection timed out
```

**Solution**:
```bash
# Check connection
ping -c 3 google.com

# Use different DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Or use offline mode
python3 main.py --offline
```

### Dependency Installation Issues

#### Package Not Found

Some packages may not be available on all platforms. JARVIS handles this gracefully with fallbacks.

```python
# Test if a package works
python3 -c "import rich; print('rich available')"
```

#### Compilation Errors

For packages requiring compilation (like cryptography):

```bash
# Termux - install build dependencies
pkg install build-essential libffi openssl

# Then retry
pip install cryptography
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Check the [FAQ](FAQ.md)
3. Run diagnostics: `python3 main.py --diagnose`
4. Open an issue on GitHub with:
   - Your platform (Termux version, Android version)
   - Python version (`python3 --version`)
   - Error message
   - Steps to reproduce

---

## Next Steps

After installation:

1. Read the [User Guide](USER_GUIDE.md) to learn JARVIS features
2. Configure advanced settings in [Configuration Guide](CONFIGURATION.md)
3. Explore the [API Documentation](API.md) for development
4. Join the community for support and updates

---

*Last Updated: Version 14.0.0*
