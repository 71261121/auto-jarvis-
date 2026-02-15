# JARVIS v14 Ultimate - Frequently Asked Questions

## General Questions

### What is JARVIS?

JARVIS is a self-modifying AI assistant designed to run efficiently on mobile devices (specifically Termux on Android). It combines a powerful AI chat interface with the unique ability to analyze and improve its own code.

### Why "Self-Modifying"?

JARVIS can analyze its own code, suggest improvements, and apply those improvements safely with automatic backups and rollback capabilities. This means JARVIS can:

- Optimize its own performance
- Fix bugs in its code
- Add new features
- Improve code quality over time

### Is JARVIS really free?

Yes! JARVIS uses free AI models through OpenRouter, requiring no paid API keys for basic functionality. You can use free models like:
- Llama 3.1 8B
- Gemma 2 9B
- Mistral 7B
- Qwen 2 7B

### What devices does JARVIS support?

JARVIS is optimized for:
- **Primary**: Android devices with Termux (tested on Realme 2 Pro Lite with 4GB RAM)
- **Secondary**: Any Linux distribution, macOS, Windows (via WSL)

Minimum requirements:
- Python 3.9+
- 512MB RAM (1GB+ recommended)
- 100MB storage

---

## Installation Questions

### How do I install JARVIS?

The easiest way is:
```bash
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash
```

For detailed instructions, see the [Installation Guide](INSTALLATION.md).

### Do I need root access?

No! JARVIS is designed to work without root access on Termux/Android.

### Why is the installation failing?

Common reasons:
1. **Old Python version**: Need Python 3.9+
2. **Insufficient memory**: Close other apps during installation
3. **Network issues**: Check internet connection
4. **Permission denied**: Run `termux-setup-storage` on Termux

See [Troubleshooting](TROUBLESHOOTING.md) for detailed solutions.

### Can I install JARVIS on Windows?

Yes, via WSL (Windows Subsystem for Linux):
```powershell
wsl --install
# Then follow Linux installation instructions
```

---

## AI & API Questions

### Do I need an API key?

For AI features, yes. You need an OpenRouter API key (free tier available):

1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up (free)
3. Generate an API key
4. Set it: `export OPENROUTER_API_KEY='your-key'`

### What if I don't have an API key?

JARVIS has a local fallback mode that works without API access:
- Pattern-based responses
- Rule-based reasoning
- Basic calculations

Enable with: `/ai off` or set `ai.provider: local` in config.

### Which AI model should I use?

| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| Llama 3.1 8B | General use | Medium | High |
| Gemma 2 9B | Reasoning | Medium | High |
| Mistral 7B | Fast responses | Fast | Medium |
| Qwen 2 7B | Multilingual | Fast | Medium |

### Why are AI responses slow?

Possible reasons:
1. **Network latency**: Check your connection
2. **API server load**: Try again later
3. **Large context**: Reduce context window
4. **Model processing**: Some models are slower

Solutions:
- Enable streaming: `/config set ai.streaming true`
- Use faster model: `/model mistralai/mistral-7b-instruct:free`
- Reduce max tokens: `/config set ai.max_tokens 1024`

### How do I use a different AI provider?

Currently supported:
- OpenRouter (default)
- OpenAI (with your key)
- Local (no API)

To use OpenAI:
```bash
export OPENAI_API_KEY='your-openai-key'
jarvis> /config set ai.provider openai
```

---

## Self-Modification Questions

### Is self-modification safe?

Yes! JARVIS has multiple safety measures:

1. **Backup System**: Every modification creates a backup
2. **Safety Validation**: Dangerous patterns are blocked
3. **Test Verification**: Tests run after each modification
4. **Easy Rollback**: One-command rollback to any state

### Can JARVIS break itself?

JARVIS is designed to prevent self-damage:
- Core files are protected
- Dangerous operations are blocked
- Backups are always created
- Failed modifications auto-rollback

### How do I undo a modification?

```bash
# View history
jarvis> /history

# Rollback to specific point
jarvis> /rollback 001

# Or restore from backup
jarvis> /backup restore backup_20240114_001
```

### What kind of modifications can JARVIS make?

JARVIS can:
- Optimize code performance
- Add type hints
- Refactor code
- Fix bugs
- Add documentation
- Improve error handling

JARVIS cannot (by default):
- Delete files
- Modify test files
- Execute arbitrary code
- Change protected files

### Can I disable self-modification?

Yes:
```bash
jarvis> /config set self_modification.enabled false
```

---

## Memory & Performance Questions

### How much memory does JARVIS use?

Typical usage:
- **Startup**: 30-50 MB
- **During AI chat**: 50-80 MB
- **With caching**: 80-150 MB

JARVIS is optimized for devices with as little as 512MB available RAM.

### Why is JARVIS using too much memory?

If memory usage is high:
1. Clear cache: `/memory clear`
2. Reduce history: `/config set memory.max_history 50`
3. Reduce context: `/config set ai.context_window 2048`

### How do I improve performance?

Tips for better performance:
1. Enable caching: `/config set cache.enabled true`
2. Use lazy loading: `/config set performance.lazy_loading true`
3. Disable animations: `/config set interface.progress_animations false`
4. Use faster model: `/model mistralai/mistral-7b-instruct:free`

### Why is startup slow?

Slow startup causes:
1. **Large import chain**: Enable lazy loading
2. **Slow storage**: Check disk health
3. **Network checks**: Disable on startup

Solution:
```bash
# Profile startup
python3 -X importtime main.py 2>&1 | head -20

# Enable lazy loading
jarvis> /config set performance.lazy_loading true
```

---

## Security Questions

### Is JARVIS secure?

JARVIS includes comprehensive security:
- Encrypted storage for sensitive data
- Authentication system
- Audit logging
- Sandbox for code execution

### How do I enable authentication?

```bash
# First time setup
jarvis> /auth setup

# Login
jarvis> /auth login
Username: admin
Password: ********
```

### How do I encrypt my data?

```bash
# Encrypt file
jarvis> /encrypt secrets.txt

# Decrypt file
jarvis> /decrypt secrets.txt.enc
```

### Are my API keys safe?

Yes. API keys are:
- Never logged
- Stored encrypted
- Not included in backups
- Configured via environment variables

### What does JARVIS audit?

Security audit includes:
- Login attempts (successful and failed)
- File access
- Configuration changes
- Self-modifications
- Permission changes

View with: `/audit`

---

## Customization Questions

### How do I change the theme?

```bash
# Available themes: dark, light, mono
jarvis> /config set interface.theme light

# Custom theme
jarvis> /config set interface.theme custom
# Then edit ~/.jarvis/themes/custom.json
```

### How do I add custom commands?

Create a plugin:
```python
# ~/.jarvis/plugins/my_plugin/plugin.py
from core.plugins import Plugin

class MyPlugin(Plugin):
    def on_load(self):
        self.register_command('hello', self.hello_cmd)

    def hello_cmd(self, args, context):
        return "Hello from my plugin!"
```

Then enable: `/plugin enable my_plugin`

### How do I change the prompt?

```bash
jarvis> /config set interface.prompt "[JARVIS]>"
```

Variables available:
- `{name}` - Application name
- `{state}` - Current state
- `{time}` - Current time

### Can I use multiple AI models?

Yes, switch between models:
```bash
# Switch temporarily
jarvis> /model mistralai/mistral-7b-instruct:free

# Set as default
jarvis> /config set ai.model mistralai/mistral-7b-instruct:free
```

---

## Troubleshooting Questions

### Why won't JARVIS start?

Check:
1. Python version: `python3 --version` (need 3.9+)
2. Working directory: `cd ~/jarvis_v14_ultimate`
3. Dependencies: `pip install -r requirements.txt`
4. Configuration: `python3 main.py --validate-config`

### Why do I get import errors?

```bash
# Ensure you're in the right directory
cd ~/jarvis_v14_ultimate

# Or set PYTHONPATH
export PYTHONPATH=~/jarvis_v14_ultimate:$PYTHONPATH
```

### Why is the AI not responding?

Check:
1. API key set: `echo $OPENROUTER_API_KEY`
2. Network: `ping openrouter.ai`
3. API status: Check [status.openrouter.ai](https://status.openrouter.ai)
4. Debug mode: `/debug on` for details

### How do I reset everything?

```bash
# Backup first
cp -r ~/.jarvis ~/.jarvis_backup

# Uninstall
rm -rf ~/jarvis_v14_ultimate ~/.jarvis

# Reinstall
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash
```

### How do I report a bug?

1. Generate debug report: `python3 main.py --debug-report > report.txt`
2. Go to GitHub Issues
3. Create new issue with:
   - Error message
   - Steps to reproduce
   - Debug report
   - Your environment

---

## Advanced Questions

### Can I run JARVIS as a service?

On Linux:
```bash
# Create systemd service
sudo cat > /etc/systemd/system/jarvis.service << EOF
[Unit]
Description=JARVIS AI Assistant

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/jarvis_v14_ultimate
ExecStart=/usr/bin/python3 main.py --daemon

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable jarvis
sudo systemctl start jarvis
```

### Can I use JARVIS programmatically?

Yes:
```python
from core.ai import OpenRouterClient
from core import get_cache

client = OpenRouterClient()
response = client.chat("Hello!")

cache = get_cache()
cache.set('key', 'value')
```

See [API Documentation](API.md) for full reference.

### How do I contribute?

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `pytest`
5. Submit pull request

See [Developer Guide](DEVELOPER.md) for details.

### Where is my data stored?

```
~/.jarvis/
├── config.json       # Configuration
├── data/             # User data
├── cache/            # Cache files
├── logs/             # Log files
├── backups/          # Backups
├── keys/             # Encryption keys
└── plugins/          # Installed plugins
```

---

## Still Have Questions?

- 📖 [User Guide](USER_GUIDE.md)
- 🔧 [Troubleshooting](TROUBLESHOOTING.md)
- 📚 [API Documentation](API.md)
- 💬 [GitHub Discussions](https://github.com/jarvis/jarvis-v14/discussions)

---

*Last Updated: Version 14.0.0*
