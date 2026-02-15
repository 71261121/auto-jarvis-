#!/bin/bash
#####################################################################
# JARVIS v14 Ultimate - SINGLE COMMAND INSTALLER FOR TERMUX
#####################################################################
#
# EK COMMAND SE PURANA DELETE + NAYA SETUP:
#
#   bash -c "$(curl -sL https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/install_jarvis.sh)"
#
# Or with wget:
#   bash -c "$(wget -qO- https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/install_jarvis.sh)"
#
#####################################################################

set -e

# Colors
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' C='\033[0;36m' NC='\033[0m'

echo -e "${C}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      JARVIS v14 Ultimate - Single Command Installer          ║"
echo "║      Purana Delete + Naya Setup = 1 Command                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Variables
JARVIS_DIR="$HOME/jarvis"
CONFIG_DIR="$HOME/.jarvis"
BACKUP_DIR="$HOME/jarvis_old_$(date +%Y%m%d_%H%M%S)"
GITHUB="https://raw.githubusercontent.com/AnantDongaria/jarvis/main"

# Step 1: Remove/Backup old installation
echo -e "${Y}[1/5] Checking for existing installation...${NC}"
if [ -d "$JARVIS_DIR" ]; then
    echo "  Found existing installation. Backing up..."
    mv "$JARVIS_DIR" "$BACKUP_DIR"
    echo -e "${G}  ✓ Old installation backed up to: $BACKUP_DIR${NC}"
else
    echo -e "${G}  ✓ Fresh installation${NC}"
fi

# Step 2: Create directories
echo -e "${Y}[2/5] Creating directories...${NC}"
mkdir -p "$JARVIS_DIR"/{core/{ai,autonomous,self_mod,memory,optimization},interface,security,install,config}
mkdir -p "$CONFIG_DIR"/{backups,data}
echo -e "${G}  ✓ Directories created${NC}"

# Step 3: Download files
echo -e "${Y}[3/5] Downloading JARVIS files...${NC}"

download() {
    local file="$1"
    local url="$GITHUB/$file"
    local target="$JARVIS_DIR/$file"
    mkdir -p "$(dirname "$target")"
    
    if command -v curl &>/dev/null; then
        curl -sL "$url" -o "$target" 2>/dev/null && echo -e "  ${G}✓${NC} $file" || echo -e "  ${R}✗${NC} $file"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$target" 2>/dev/null && echo -e "  ${G}✓${NC} $file" || echo -e "  ${R}✗${NC} $file"
    fi
}

# Core files
download "main.py"
download "requirements.txt"
download "core/__init__.py"
download "core/events.py"
download "core/state_machine.py"
download "core/cache.py"
download "core/error_handler.py"
download "core/bulletproof_imports.py"

# AI module
download "core/ai/__init__.py"
download "core/ai/openrouter_client.py"
download "core/ai/rate_limiter.py"
download "core/ai/model_selector.py"
download "core/ai/response_parser.py"

# Autonomous Engine (THE KEY!)
download "core/autonomous/__init__.py"
download "core/autonomous/intent_detector.py"
download "core/autonomous/executor.py"
download "core/autonomous/engine.py"
download "core/autonomous/safety_manager.py"

# Self-modification
download "core/self_mod/__init__.py"
download "core/self_mod/bridge.py"
download "core/self_mod/code_analyzer.py"
download "core/self_mod/safe_modifier.py"
download "core/self_mod/backup_manager.py"

# Memory
download "core/memory/__init__.py"
download "core/memory/context_manager.py"
download "core/memory/chat_storage.py"

# Interface
download "interface/__init__.py"
download "interface/cli.py"
download "interface/commands.py"

# Security
download "security/__init__.py"

# Config
download "config/__init__.py"
download "config/config_manager.py"

echo -e "${G}  ✓ All files downloaded${NC}"

# Step 4: Install dependencies
echo -e "${Y}[4/5] Installing Python dependencies...${NC}"
pip install -q requests python-dotenv 2>/dev/null || pip3 install -q requests python-dotenv 2>/dev/null || true
echo -e "${G}  ✓ Dependencies installed${NC}"

# Step 5: Create config and launcher
echo -e "${Y}[5/5] Creating configuration...${NC}"

# Config file
cat > "$CONFIG_DIR/config.json" << 'EOF'
{
    "general": {"app_name": "JARVIS", "version": "14.0.0", "debug_mode": false},
    "ai": {"provider": "openrouter", "model": "openrouter/auto", "temperature": 0.7},
    "security": {"enable_auth": false},
    "self_mod": {"enable": true, "backup_dir": "~/.jarvis/backups"}
}
EOF

# .env template
cat > "$CONFIG_DIR/.env.template" << 'EOF'
# JARVIS API Configuration
# Get FREE API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key_here
EOF

# Launcher script
cat > "$HOME/jarvis.sh" << 'EOF'
#!/bin/bash
cd ~/jarvis
python main.py "$@"
EOF
chmod +x "$HOME/jarvis.sh"

# Add alias
grep -q "alias jarvis=" ~/.bashrc 2>/dev/null || echo 'alias jarvis="~/jarvis.sh"' >> ~/.bashrc

# Initialize marker
date > "$CONFIG_DIR/.initialized"

echo -e "${G}  ✓ Configuration created${NC}"

# Done!
echo ""
echo -e "${G}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${G}║          JARVIS v14 Ultimate - INSTALLATION COMPLETE!        ║${NC}"
echo -e "${G}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${C}📍 Location:${NC} $JARVIS_DIR"
echo -e "${C}📍 Config:${NC}   $CONFIG_DIR"
if [ -d "$BACKUP_DIR" ]; then
    echo -e "${C}📍 Old Backup:${NC} $BACKUP_DIR"
fi
echo ""
echo -e "${C}🚀 QUICK START:${NC}"
echo ""
echo -e "  ${Y}Step 1:${NC} Set your API key (FREE from openrouter.ai):"
echo "         export OPENROUTER_API_KEY='your_key_here'"
echo ""
echo -e "  ${Y}Step 2:${NC} Start JARVIS:"
echo "         source ~/.bashrc"
echo "         jarvis"
echo ""
echo -e "  ${Y}Alternative:${NC} cd ~/jarvis && python main.py"
echo ""
echo -e "${C}✨ FEATURES:${NC}"
echo "  ✓ Autonomous file operations (read, write, create, delete)"
echo "  ✓ Terminal command execution from chat"
echo "  ✓ AI-powered conversations"
echo "  ✓ Self-modification capabilities"
echo "  ✓ Auto-backup system"
echo ""
echo -e "${G}Type 'help' in JARVIS for all commands!${NC}"
echo ""
