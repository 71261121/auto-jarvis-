#!/bin/bash
# FORCE UPDATE JARVIS ON TERMUX

echo "=== Force Updating JARVIS ==="

cd ~/jarvis

# 1. Hard reset to match GitHub
git fetch origin
git reset --hard origin/master

# 2. Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 3. Verify fix
echo ""
echo "=== Verifying Fix ==="
grep -A1 "ANALYZE_CODE:" core/autonomous/intent_detector.py | head -2

echo ""
echo "=== Done! Now run: ==="
echo "python main.py"
