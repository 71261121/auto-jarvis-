#!/bin/bash
# JARVIS v14 Ultimate - Simple Setup Script
# One command installation for Termux/Linux

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         JARVIS v14 Ultimate - Installing...                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Backup old
[ -d "$HOME/jarvis" ] && mv "$HOME/jarvis" "$HOME/jarvis_old_$(date +%s)"

# Clone from GitHub
echo "[1/3] Downloading JARVIS..."
git clone -q https://github.com/71261121/auto-jarvis-.git "$HOME/jarvis"

# Install dependencies
echo "[2/3] Installing dependencies..."
pip install -q requests python-dotenv 2>/dev/null || pip3 install -q requests python-dotenv 2>/dev/null || true

# Create config
echo "[3/3] Setting up..."
mkdir -p "$HOME/.jarvis/backups"
echo 'alias jarvis="cd ~/jarvis && python main.py"' >> "$HOME/.bashrc"

# Done
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              INSTALLATION COMPLETE!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 Location: ~/jarvis"
echo ""
echo "🔑 Set API key (get FREE key from openrouter.ai/keys):"
echo "   export OPENROUTER_API_KEY='your_key_here'"
echo ""
echo "▶️  Start JARVIS:"
echo "   source ~/.bashrc"
echo "   jarvis"
echo ""
echo "   OR: cd ~/jarvis && python main.py"
echo ""
