# JARVIS v14 Ultimate - Installation Guide

## 🚀 METHOD 1: Copy-Paste (Easiest - 100% Works)

### Step 1: Download Installer

```bash
# Create installer file
cat > ~/jarvis_installer.py << 'INSTALLER_EOF'
```

**Copy the entire content from `jarvis_installer.py` file and paste here, then:**

```bash
INSTALLER_EOF
```

### Step 2: Run Installer

```bash
python3 ~/jarvis_installer.py
```

---

## 🚀 METHOD 2: Manual Setup (If Script Fails)

```bash
# 1. Remove old JARVIS
rm -rf ~/jarvis
rm -rf ~/.jarvis

# 2. Create directories
mkdir -p ~/jarvis/core/{ai,autonomous,self_mod,memory}
mkdir -p ~/.jarvis/backups

# 3. Download installer (if you have internet)
# OR copy the files manually

# 4. Run installer
python3 ~/jarvis_installer.py
```

---

## 🔑 After Installation: Set API Key

```bash
# Get FREE API key from: https://openrouter.ai/keys
export OPENROUTER_API_KEY='sk-or-v1-your-key-here'
```

---

## ▶️ Start JARVIS

```bash
# Method 1: Using alias
source ~/.bashrc
jarvis

# Method 2: Direct
cd ~/jarvis && python main.py
```

---

## 📝 Quick Commands

```
JARVIS> read main.py
JARVIS> list files
JARVIS> search def hello
JARVIS> run ls -la
JARVIS> install requests
JARVIS> help
JARVIS> status
```

---

## ⚠️ Troubleshooting

### Error: "requests not found"
```bash
pip install requests python-dotenv
```

### Error: "AI not available"
```bash
export OPENROUTER_API_KEY='your_key_here'
```

### Error: "Permission denied"
```bash
chmod +x ~/jarvis.sh
```

---

**Version:** 14.0.0 Ultimate
**Platform:** Termux/Android, Linux
**RAM Required:** 4GB+
