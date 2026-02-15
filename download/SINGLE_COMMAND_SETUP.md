# JARVIS v14 Ultimate - Single Command Setup

## 🚀 EK COMMAND SE PURANA DELETE + NAYA SETUP

### Termux/Android ke liye:

```bash
bash -c "$(curl -sL https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/install_jarvis.sh)"
```

### Agar wget use karna ho:

```bash
bash -c "$(wget -qO- https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/install_jarvis.sh)"
```

---

## 📋 Ye command kya karega:

1. **Purana JARVIS delete** (backup ke sath)
2. **Naye directories create**
3. **Saari files download** (from GitHub)
4. **Dependencies install**
5. **Configuration setup**
6. **Launcher script create**

---

## 🔑 API Key Setup (Required for AI):

```bash
# FREE API key lein: https://openrouter.ai/keys
export OPENROUTER_API_KEY='your_key_here'
```

---

## ▶️ JARVIS Start Karne ke liye:

```bash
# Method 1: Using alias
source ~/.bashrc
jarvis

# Method 2: Direct
cd ~/jarvis && python main.py
```

---

## ✨ Features:

| Feature | Description |
|---------|-------------|
| 🗂️ File Operations | read, write, create, delete files |
| ⚡ Terminal | Execute commands from chat |
| 🤖 AI Chat | Natural conversations |
| 🔧 Self-Modify | AI can modify its own code |
| 💾 Auto-Backup | Automatic backups before changes |

---

## 📝 Example Commands:

```
JARVIS> read main.py
JARVIS> list files in core/
JARVIS> create utils.py with helper functions
JARVIS> modify main.py to add debug command
JARVIS> run python test.py
JARVIS> install requests
JARVIS> What is Python?  (AI chat)
```

---

## ⚠️ Requirements:

- Termux (Android) or Linux
- Python 3.8+
- curl or wget
- Internet connection

---

## 🔧 Manual Setup (Alternative):

Agar automatic script na kaam kare:

```bash
# 1. Old backup
mv ~/jarvis ~/jarvis_old_backup

# 2. Clone new
git clone https://github.com/AnantDongaria/jarvis.git ~/jarvis

# 3. Install deps
pip install requests python-dotenv

# 4. Set API key
export OPENROUTER_API_KEY='your_key'

# 5. Run
cd ~/jarvis && python main.py
```

---

**Version:** 14.0.0 Ultimate
**Platform:** Termux/Android, Linux
