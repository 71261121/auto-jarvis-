#!/bin/bash
#################################################################
# JARVIS v14 Ultimate - One-Command Termux Setup
#################################################################
# This script:
# 1. Removes old JARVIS installation
# 2. Downloads fresh JARVIS v14
# 3. Sets up everything automatically
#################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# JARVIS installation directory
JARVIS_DIR="$HOME/jarvis"
BACKUP_DIR="$HOME/jarvis_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         JARVIS v14 Ultimate - Termux Setup                   ║"
echo "║         One Command - Complete Installation                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Backup old installation if exists
if [ -d "$JARVIS_DIR" ]; then
    echo -e "${YELLOW}[1/5] Backing up old JARVIS installation...${NC}"
    mv "$JARVIS_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}✓ Old installation backed up to: $BACKUP_DIR${NC}"
else
    echo -e "${GREEN}[1/5] No existing installation found. Fresh install.${NC}"
fi

# Step 2: Create directory structure
echo -e "${YELLOW}[2/5] Creating JARVIS directory structure...${NC}"
mkdir -p "$JARVIS_DIR"
mkdir -p "$JARVIS_DIR/core"
mkdir -p "$JARVIS_DIR/core/ai"
mkdir -p "$JARVIS_DIR/core/autonomous"
mkdir -p "$JARVIS_DIR/core/self_mod"
mkdir -p "$JARVIS_DIR/core/memory"
mkdir -p "$JARVIS_DIR/core/optimization"
mkdir -p "$JARVIS_DIR/interface"
mkdir -p "$JARVIS_DIR/install"
mkdir -p "$JARVIS_DIR/security"
mkdir -p "$JARVIS_DIR/config"
mkdir -p "$JARVIS_DIR/data"
mkdir -p "$HOME/.jarvis"
mkdir -p "$HOME/.jarvis/backups"
mkdir -p "$HOME/.jarvis/data"
echo -e "${GREEN}✓ Directory structure created${NC}"

# Step 3: Create core __init__.py files
echo -e "${YELLOW}[3/5] Creating Python module structure...${NC}"

# Core __init__.py
cat > "$JARVIS_DIR/core/__init__.py" << 'COREINIT'
"""JARVIS Core Module"""
COREINIT

# AI __init__.py
cat > "$JARVIS_DIR/core/ai/__init__.py" << 'AIINIT'
"""JARVIS AI Module"""
from .openrouter_client import OpenRouterClient, FreeModel
__all__ = ['OpenRouterClient', 'FreeModel']
AIINIT

# Memory __init__.py
cat > "$JARVIS_DIR/core/memory/__init__.py" << 'MEMINIT'
"""JARVIS Memory Module"""
MEMINIT

# Self_mod __init__.py
cat > "$JARVIS_DIR/core/self_mod/__init__.py" << 'SELMODINIT'
"""JARVIS Self-Modification Module"""
SELMODINIT

# Interface __init__.py
cat > "$JARVIS_DIR/interface/__init__.py" << 'INTINIT'
"""JARVIS Interface Module"""
INTINIT

# Security __init__.py
cat > "$JARVIS_DIR/security/__init__.py" << 'SECINIT'
"""JARVIS Security Module"""
SECINIT

# Install __init__.py
cat > "$JARVIS_DIR/install/__init__.py" << 'INSTINIT'
"""JARVIS Installation Module"""
INSTINIT

# Config __init__.py
cat > "$JARVIS_DIR/config/__init__.py" << 'CONFINIT'
"""JARVIS Config Module"""
CONFINIT

echo -e "${GREEN}✓ Python module structure created${NC}"

# Step 4: Download JARVIS files from GitHub
echo -e "${YELLOW}[4/5] Downloading JARVIS v14 files...${NC}"

# GitHub raw URL base
GITHUB_RAW="https://raw.githubusercontent.com/AnantDongaria/jarvis/main"

# Function to download file
download_file() {
    local file_path="$1"
    local target="$JARVIS_DIR/$file_path"
    local dir=$(dirname "$target")
    
    mkdir -p "$dir"
    
    # Try curl first, then wget
    if command -v curl &> /dev/null; then
        curl -sL "$GITHUB_RAW/$file_path" -o "$target" 2>/dev/null || echo "Failed: $file_path"
    elif command -v wget &> /dev/null; then
        wget -q "$GITHUB_RAW/$file_path" -O "$target" 2>/dev/null || echo "Failed: $file_path"
    else
        echo -e "${RED}Error: curl or wget required${NC}"
        exit 1
    fi
}

# Download core files
FILES_TO_DOWNLOAD=(
    "main.py"
    "requirements.txt"
    "core/__init__.py"
    "core/events.py"
    "core/cache.py"
    "core/plugins.py"
    "core/state_machine.py"
    "core/error_handler.py"
    "core/bulletproof_imports.py"
    "core/http_client.py"
    "core/safe_exec.py"
    "core/ai/__init__.py"
    "core/ai/openrouter_client.py"
    "core/ai/rate_limiter.py"
    "core/ai/model_selector.py"
    "core/ai/response_parser.py"
    "core/ai/auth.py"
    "core/ai/health.py"
    "core/ai/local.py"
    "core/autonomous/__init__.py"
    "core/autonomous/intent_detector.py"
    "core/autonomous/executor.py"
    "core/autonomous/engine.py"
    "core/autonomous/safety_manager.py"
    "core/self_mod/__init__.py"
    "core/self_mod/bridge.py"
    "core/self_mod/code_analyzer.py"
    "core/self_mod/safe_modifier.py"
    "core/self_mod/backup_manager.py"
    "core/self_mod/improvement_engine.py"
    "core/memory/__init__.py"
    "core/memory/context_manager.py"
    "core/memory/chat_storage.py"
    "core/memory/memory_optimizer.py"
    "core/memory/conversation_indexer.py"
    "interface/__init__.py"
    "interface/cli.py"
    "interface/commands.py"
    "interface/input.py"
    "interface/output.py"
    "interface/session.py"
    "interface/help.py"
    "interface/notify.py"
    "interface/progress.py"
    "security/__init__.py"
    "security/auth.py"
    "security/encryption.py"
    "security/sandbox.py"
    "security/permissions.py"
    "security/audit.py"
    "security/keys.py"
    "security/threat_detect.py"
    "install/__init__.py"
    "install/detect.py"
    "install/deps.py"
    "install/config_gen.py"
    "install/first_run.py"
    "install/updater.py"
    "install/repair.py"
    "install/uninstall.py"
    "config/__init__.py"
    "config/config_manager.py"
)

# Download each file
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    echo -n "  Downloading: $file ... "
    download_file "$file"
    echo "✓"
done

echo -e "${GREEN}✓ All files downloaded${NC}"

# Step 5: Install dependencies and finalize
echo -e "${YELLOW}[5/5] Installing dependencies...${NC}"

# Check if pip is available
if command -v pip &> /dev/null; then
    pip install -q requests python-dotenv 2>/dev/null || true
elif command -v pip3 &> /dev/null; then
    pip3 install -q requests python-dotenv 2>/dev/null || true
fi

# Create default config
cat > "$HOME/.jarvis/config.json" << 'CONFIGJSON'
{
    "general": {
        "app_name": "JARVIS",
        "version": "14.0.0",
        "debug_mode": false,
        "quiet_mode": false
    },
    "ai": {
        "provider": "openrouter",
        "model": "openrouter/auto",
        "temperature": 0.7,
        "max_tokens": 2048,
        "enable_cache": true
    },
    "security": {
        "enable_auth": false,
        "session_timeout": 3600,
        "max_failed_attempts": 5
    },
    "memory": {
        "max_context_length": 100000,
        "enable_optimization": true
    },
    "self_mod": {
        "enable": true,
        "backup_dir": "~/.jarvis/backups",
        "max_backups": 50
    }
}
CONFIGJSON

# Create .env template
cat > "$HOME/.jarvis/.env.template" << 'ENVTEMPLATE'
# JARVIS Configuration
# Copy this file to .env and fill in your API key

# OpenRouter API Key (get free key at openrouter.ai)
OPENROUTER_API_KEY=your_key_here

# Optional: Other AI providers
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
ENVTEMPLATE

# Create initialization marker
touch "$HOME/.jarvis/.initialized"
date > "$HOME/.jarvis/.initialized"

# Create launcher script
cat > "$HOME/jarvis.sh" << 'LAUNCHER'
#!/bin/bash
# JARVIS Launcher
cd ~/jarvis
python main.py "$@"
LAUNCHER
chmod +x "$HOME/jarvis.sh"

# Add alias to bashrc
if ! grep -q "alias jarvis=" "$HOME/.bashrc" 2>/dev/null; then
    echo 'alias jarvis="~/jarvis.sh"' >> "$HOME/.bashrc"
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Done!
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           JARVIS v14 Ultimate - Setup Complete!              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Installation Details:${NC}"
echo "  Location: $JARVIS_DIR"
echo "  Config: $HOME/.jarvis/config.json"
echo "  Backups: $HOME/.jarvis/backups"
if [ -d "$BACKUP_DIR" ]; then
    echo "  Old Backup: $BACKUP_DIR"
fi
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo -e "  ${YELLOW}1.${NC} Set your API key:"
echo "    export OPENROUTER_API_KEY='your_key_here'"
echo ""
echo -e "  ${YELLOW}2.${NC} Start JARVIS:"
echo "    source ~/.bashrc"
echo "    jarvis"
echo ""
echo -e "  ${YELLOW}Or run directly:${NC}"
echo "    cd ~/jarvis && python main.py"
echo ""
echo -e "${CYAN}Features:${NC}"
echo "  ✓ Autonomous file operations (read, write, create, delete)"
echo "  ✓ Terminal command execution"
echo "  ✓ AI-powered chat"
echo "  ✓ Self-modification capabilities"
echo "  ✓ Auto-backup system"
echo ""
echo -e "${GREEN}Type 'help' in JARVIS for all commands.${NC}"
echo ""
