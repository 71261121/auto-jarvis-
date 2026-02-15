#!/bin/bash
# JARVIS v14 Ultimate - Termux Fix Commands
# Run these commands in Termux

echo "════════════════════════════════════════════════════════════"
echo "JARVIS v14 - TERMUX FIX SCRIPT"
echo "════════════════════════════════════════════════════════════"

# Step 1: Remove old JARVIS folder
echo ""
echo "[1/5] Removing old JARVIS folder..."
rm -rf ~/jarvis

# Step 2: Install libsodium (required for pynacl)
echo ""
echo "[2/5] Installing libsodium..."
pkg install libsodium -y

# Step 3: Install Python dependencies (skip pynacl for now)
echo ""
echo "[3/5] Installing Python dependencies..."
pip install click colorama python-dotenv pyyaml requests tqdm schedule \
            typing-extensions rich loguru cryptography httpx psutil

# Step 4: Try installing pynacl with pre-built binary
echo ""
echo "[4/5] Installing pynacl (with binary wheel)..."
pip install pynacl --only-binary :all: 2>/dev/null || {
    echo "pynacl binary not available, trying alternative..."
    pip install pynacl --no-build-isolation 2>/dev/null || {
        echo "⚠️ pynacl installation failed - using alternative encryption"
    }
}

# Step 5: Clone fresh JARVIS
echo ""
echo "[5/5] Cloning JARVIS v14 Ultimate..."
cd ~
git clone https://github.com/71261121/auto-jarvis-.git jarvis
cd jarvis

echo ""
echo "════════════════════════════════════════════════════════════"
echo "SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To run JARVIS:"
echo "  cd ~/jarvis"
echo "  export OPENROUTER_API_KEY=\"your-api-key\""
echo "  python main.py"
echo ""
