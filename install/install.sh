#!/bin/bash
#
# JARVIS v14 Ultimate - Installation Script
# ==========================================
#
# Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
#
# One-line install command:
#   curl -fsSL https://raw.githubusercontent.com/71261121/auto-jarvis-/main/install/install.sh | bash
#
# Or:
#   wget -qO- https://raw.githubusercontent.com/71261121/auto-jarvis-/main/install/install.sh | bash
#

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS AND FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║${NC}"
    echo -e "${CYAN}║   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║${NC}"
    echo -e "${CYAN}║   ██║███████║██████╔╝██║   ██║██║███████╗                   ║${NC}"
    echo -e "${CYAN}║   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║${NC}"
    echo -e "${CYAN}║   ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║${NC}"
    echo -e "${CYAN}║   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║              Self-Modifying AI Assistant v14                 ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}==>${NC} ${1}"
}

print_success() {
    echo -e "${GREEN}✓${NC} ${1}"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} ${1}"
}

print_error() {
    echo -e "${RED}✗${NC} ${1}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

JARVIS_DIR="$HOME/.jarvis"
INSTALL_DIR="$HOME/jarvis_v14_ultimate"
GITHUB_REPO="https://github.com/71261121/auto-jarvis-"
PYTHON_MIN_VERSION="3.9"
MIN_MEMORY_MB=512

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

get_python_version() {
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0"
}

version_ge() {
    # Returns 0 if $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

get_available_memory() {
    if [ -f /proc/meminfo ]; then
        grep -m1 MemAvailable /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' || \
        grep -m1 MemFree /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}'
    else
        echo "1024"  # Default assumption
    fi
}

is_termux() {
    [ -n "$TERMUX_VERSION" ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

check_termux() {
    print_step "Checking Termux environment..."
    
    if is_termux; then
        print_success "Running in Termux v${TERMUX_VERSION}"
        return 0
    fi
    
    print_warning "Not running in Termux. Some features may not work."
    return 0
}

check_python() {
    print_step "Checking Python installation..."
    
    if ! check_command python3; then
        print_error "Python 3 is not installed"
        echo ""
        echo "Please install Python:"
        if is_termux; then
            echo "  pkg install python"
        else
            echo "  Visit: https://www.python.org/downloads/"
        fi
        return 1
    fi
    
    PYTHON_VERSION=$(get_python_version)
    
    if ! version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required (found: $PYTHON_VERSION)"
        return 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
    return 0
}

check_pip() {
    print_step "Checking pip installation..."
    
    if ! python3 -m pip --version &> /dev/null; then
        print_warning "pip not found, installing..."
        
        if is_termux; then
            pkg install python-pip -y
        else
            python3 -m ensurepip --upgrade
        fi
    fi
    
    if python3 -m pip --version &> /dev/null; then
        PIP_VERSION=$(python3 -m pip --version | awk '{print $2}')
        print_success "pip $PIP_VERSION found"
        return 0
    fi
    
    print_error "Failed to install pip"
    return 1
}

check_memory() {
    print_step "Checking available memory..."
    
    AVAILABLE_MEM=$(get_available_memory)
    
    if [ "$AVAILABLE_MEM" -lt "$MIN_MEMORY_MB" ]; then
        print_warning "Low memory: ${AVAILABLE_MEM}MB available (recommended: ${MIN_MEMORY_MB}MB)"
        print_warning "JARVIS may run slowly. Consider closing other applications."
    else
        print_success "${AVAILABLE_MEM}MB available"
    fi
    
    return 0
}

check_storage() {
    print_step "Checking available storage..."
    
    AVAILABLE_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | tr -d 'G')
    
    if [ "${AVAILABLE_GB:-0}" -lt 1 ]; then
        print_error "Insufficient storage space"
        return 1
    fi
    
    print_success "${AVAILABLE_GB}GB available"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

download_jarvis() {
    print_step "Downloading JARVIS..."
    
    # Check if already installed
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory already exists"
        read -p "Reinstall? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
        rm -rf "$INSTALL_DIR"
    fi
    
    # Clone repository
    if check_command git; then
        git clone --depth 1 "$GITHUB_REPO" "$INSTALL_DIR" 2>/dev/null || {
            print_error "Failed to clone repository"
            return 1
        }
    else
        # Download as zip
        print_warning "git not found, downloading as archive..."
        TEMP_FILE=$(mktemp)
        
        if check_command curl; then
            curl -fsSL "${GITHUB_REPO}/archive/refs/heads/main.zip" -o "$TEMP_FILE"
        elif check_command wget; then
            wget -q "${GITHUB_REPO}/archive/refs/heads/main.zip" -O "$TEMP_FILE"
        else
            print_error "Neither curl nor wget found"
            return 1
        fi
        
        # Extract
        if check_command unzip; then
            unzip -q "$TEMP_FILE" -d "$HOME"
            mv "$HOME/jarvis-v14-main" "$INSTALL_DIR"
        else
            print_error "unzip not found"
            rm -f "$TEMP_FILE"
            return 1
        fi
        
        rm -f "$TEMP_FILE"
    fi
    
    print_success "JARVIS downloaded to $INSTALL_DIR"
    return 0
}

install_dependencies() {
    print_step "Installing dependencies..."
    
    cd "$INSTALL_DIR"
    
    # Create requirements file
    cat > requirements.txt << 'EOF'
# Class 0 - Guaranteed safe
click>=8.0.0
colorama>=0.4.0
python-dotenv>=0.19.0
pyyaml>=6.0
requests>=2.26.0
tqdm>=4.62.0
schedule>=1.1.0
typing-extensions>=4.0.0

# Class 1 - High probability
rich>=12.0.0
loguru>=0.6.0

# Additional dependencies (used in code)
cryptography>=3.4.0
httpx>=0.24.0
pynacl>=1.5.0
psutil>=5.9.0
EOF
    
    # Install
    python3 -m pip install --quiet -r requirements.txt 2>/dev/null || {
        print_warning "Some dependencies may not have installed correctly"
    }
    
    print_success "Dependencies installed"
    return 0
}

create_directories() {
    print_step "Creating directories..."
    
    mkdir -p "$JARVIS_DIR"
    mkdir -p "$JARVIS_DIR/data"
    mkdir -p "$JARVIS_DIR/cache"
    mkdir -p "$JARVIS_DIR/logs"
    mkdir -p "$JARVIS_DIR/backups"
    
    print_success "Directories created"
    return 0
}

create_config() {
    print_step "Creating configuration..."
    
    if [ ! -f "$JARVIS_DIR/config.json" ]; then
        python3 -c "
import json
config = {
    'general': {
        'app_name': 'JARVIS',
        'debug_mode': False,
        'quiet_mode': False
    },
    'ai': {
        'provider': 'openrouter',
        'model': 'meta-llama/llama-3.1-8b-instruct:free',
        'temperature': 0.7,
        'max_tokens': 2048
    },
    'storage': {
        'data_dir': '$JARVIS_DIR/data',
        'cache_dir': '$JARVIS_DIR/cache',
        'log_dir': '$JARVIS_DIR/logs'
    },
    'metadata': {
        'created_at': '$(date -Iseconds)',
        'config_version': '1.0'
    }
}
with open('$JARVIS_DIR/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
        print_success "Configuration created"
    else
        print_success "Configuration already exists"
    fi
    
    return 0
}

create_launcher() {
    print_step "Creating launcher script..."
    
    cat > "$HOME/jarvis" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
python3 main.py "\$@"
EOF
    
    chmod +x "$HOME/jarvis"
    
    print_success "Launcher created at ~/jarvis"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

verify_installation() {
    print_step "Verifying installation..."
    
    # Check files
    [ -d "$INSTALL_DIR" ] || return 1
    [ -f "$JARVIS_DIR/config.json" ] || return 1
    [ -x "$HOME/jarvis" ] || return 1
    
    # Check Python imports
    python3 -c "import click, requests, yaml" 2>/dev/null || return 1
    
    print_success "Installation verified"
    return 0
}

show_completion() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation Complete!                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Quick Start:"
    echo "  1. Run: ~/jarvis"
    echo "  2. Or: cd $INSTALL_DIR && python3 main.py"
    echo ""
    echo "Configuration: $JARVIS_DIR/config.json"
    echo ""
    echo "For AI features, configure your API key:"
    echo "  export OPENROUTER_API_KEY='your-key-here'"
    echo ""
}

main() {
    print_header
    
    # Pre-flight checks
    echo -e "${CYAN}Performing pre-installation checks...${NC}"
    echo ""
    
    check_termux || exit 1
    check_python || exit 1
    check_pip || exit 1
    check_memory
    check_storage || exit 1
    
    echo ""
    echo -e "${CYAN}Installing JARVIS...${NC}"
    echo ""
    
    # Installation steps
    download_jarvis || exit 1
    install_dependencies
    create_directories
    create_config
    create_launcher
    
    # Verify
    if ! verify_installation; then
        print_error "Installation verification failed"
        exit 1
    fi
    
    # Done
    show_completion
}

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

# Handle flags
case "${1:-}" in
    --help|-h)
        echo "JARVIS v14 Installation Script"
        echo ""
        echo "Usage: curl -fsSL <url> | bash"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --uninstall   Remove JARVIS installation"
        echo "  --update      Update JARVIS to latest version"
        exit 0
        ;;
    --uninstall)
        echo "Uninstalling JARVIS..."
        rm -rf "$INSTALL_DIR"
        rm -f "$HOME/jarvis"
        echo "JARVIS has been uninstalled. Data preserved at $JARVIS_DIR"
        exit 0
        ;;
    --update)
        echo "Updating JARVIS..."
        cd "$INSTALL_DIR" 2>/dev/null || {
            echo "JARVIS not installed. Run without --update to install."
            exit 1
        }
        git pull origin main
        python3 -m pip install -q -r requirements.txt
        echo "JARVIS updated successfully!"
        exit 0
        ;;
esac

# Run main installation
main
