# JARVIS v14 Ultimate

<p align="center">
  <img src="docs/images/jarvis-logo.png" alt="JARVIS Logo" width="200">
</p>

<p align="center">
  <strong>Self-Modifying AI Assistant for Mobile Devices</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Platform-Termux%20%7C%20Linux%20%7C%20macOS-green.svg" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/RAM-512MB+-orange.svg" alt="RAM">
</p>

---

## Overview

JARVIS v14 Ultimate is a revolutionary AI assistant designed specifically for mobile devices with limited resources. Built for Termux on Android (optimized for 4GB RAM devices), JARVIS combines powerful AI capabilities with a unique self-modification engine that allows it to improve itself over time.

### What Makes JARVIS Special?

- **🤖 Self-Modifying**: Can analyze and improve its own code safely
- **📱 Mobile-First**: Optimized for devices with as little as 512MB RAM
- **🆓 Free AI**: Uses free models through OpenRouter - no paid API required
- **🔒 Secure**: Built-in encryption, authentication, and audit logging
- **🛡️ Safe**: Automatic backups and rollback for all modifications
- **🔌 Extensible**: Plugin system for custom functionality

---

## Features

### 🧠 AI Engine
- Multiple free AI models (Llama 3.1, Gemma 2, Mistral, Qwen)
- Intelligent model selection based on task type
- Streaming responses for real-time output
- Context management for long conversations
- Local fallback when offline

### 🔧 Self-Modification
- Code analysis with complexity scoring
- Safe modification with pattern validation
- Automatic backup before changes
- Test-driven modification verification
- Easy rollback to any previous state

### 🔐 Security
- User authentication with role-based access
- Data encryption (AES-256-GCM, ChaCha20)
- Execution sandboxing
- Comprehensive audit logging
- Threat detection and prevention

### 💾 Memory System
- Efficient conversation storage
- Context management with compression
- Memory optimization for low-RAM devices
- Conversation search and export

### 🖥️ User Interface
- Clean CLI with syntax highlighting
- Multi-line input support
- Command history and auto-completion
- Markdown rendering
- Progress indicators

### ⚙️ Configuration
- Flexible JSON-based configuration
- Environment variable support
- Multiple configuration profiles
- Hot-reload capabilities

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- 512MB+ RAM (1GB+ recommended)
- Internet connection (for AI features)

### Installation

```bash
# One-line installation
curl -fsSL https://raw.githubusercontent.com/jarvis/jarvis-v14/main/install.sh | bash
```

Or manually:

```bash
# Clone repository
git clone https://github.com/jarvis/jarvis-v14.git ~/jarvis_v14_ultimate
cd ~/jarvis_v14_ultimate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY='your-key-here'

# Run
python3 main.py
```

### First Run

```
╔══════════════════════════════════════════════════════════════╗
║   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║
║   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║
║   ██║███████║██████╔╝██║   ██║██║███████╗                   ║
║   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║
║   ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║
║   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║
║              Self-Modifying AI Assistant v14                 ║
╚══════════════════════════════════════════════════════════════╝

JARVIS is ready. Type 'help' for commands.
jarvis> Hello!
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Detailed installation instructions |
| [User Guide](docs/USER_GUIDE.md) | How to use JARVIS features |
| [API Documentation](docs/API.md) | Programming interface reference |
| [Configuration Guide](docs/CONFIGURATION.md) | All configuration options |
| [Developer Guide](docs/DEVELOPER.md) | Contributing and extending |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [FAQ](docs/FAQ.md) | Frequently asked questions |

---

## Project Structure

```
jarvis_v14_ultimate/
├── main.py              # Entry point
├── core/                # Core functionality
│   ├── ai/              # AI provider modules
│   ├── memory/          # Memory system
│   └── self_mod/        # Self-modification engine
├── interface/           # User interface
├── security/            # Security modules
├── install/             # Installation system
├── config/              # Configuration management
└── docs/                # Documentation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      JARVIS v14                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐ │
│  │   CLI   │──│  Core   │──│     AI Engine           │ │
│  │  Layer  │  │  Layer  │  │  (OpenRouter/Local)     │ │
│  └─────────┘  └────┬────┘  └─────────────────────────┘ │
│                    │                                    │
│     ┌──────────────┼──────────────┐                   │
│     │              │              │                    │
│     ▼              ▼              ▼                    │
│  ┌────────┐  ┌──────────┐  ┌────────────┐             │
│  │ Memory │  │ Security │  │ Self-Mod   │             │
│  │ System │  │  Layer   │  │  Engine    │             │
│  └────────┘  └──────────┘  └────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## Available AI Models

JARVIS uses OpenRouter to access free AI models:

| Model | Context Window | Best For |
|-------|----------------|----------|
| Llama 3.1 8B | 128K tokens | General purpose |
| Gemma 2 9B | 8K tokens | Reasoning |
| Mistral 7B | 32K tokens | Fast responses |
| Qwen 2 7B | 32K tokens | Multilingual |

All models are free to use with an OpenRouter account.

---

## Commands Overview

```bash
# AI Commands
jarvis> /ai on                    # Enable AI
jarvis> /model llama-3.1-8b       # Switch model
jarvis> /models                   # List models

# Self-Modification Commands
jarvis> /analyze file.py          # Analyze code
jarvis> /modify file.py "desc"    # Request modification
jarvis> /rollback 001             # Undo modification
jarvis> /history                  # View modifications

# Configuration Commands
jarvis> /config                   # Show config
jarvis> /config set ai.temp 0.8   # Set value

# Memory Commands
jarvis> /memory                   # Memory status
jarvis> /context clear            # Clear context

# Other Commands
jarvis> /help                     # Show help
jarvis> /status                   # System status
jarvis> /debug on                 # Debug mode
```

---

## Performance

JARVIS is designed for efficiency:

| Metric | Value |
|--------|-------|
| Startup Time | <3 seconds |
| Memory Usage | 30-80 MB |
| Response Time | 1-5 seconds (AI dependent) |
| Disk Footprint | ~50 MB |

---

## Contributing

We welcome contributions! See the [Developer Guide](docs/DEVELOPER.md) for:

- Code style guidelines
- Testing procedures
- Pull request process
- Plugin development

### Quick Contribution Guide

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/jarvis-v14.git

# Create branch
git checkout -b feature/my-feature

# Make changes and test
pytest

# Submit PR
git push origin feature/my-feature
```

---

## Roadmap

- [ ] Voice input/output support
- [ ] Web interface
- [ ] Mobile app (Android)
- [ ] Multi-language support
- [ ] Cloud sync
- [ ] Plugin marketplace

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- OpenRouter for free AI model access
- The open-source community
- All contributors

---

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/jarvis/jarvis-v14/issues)
- 💬 [Discussions](https://github.com/jarvis/jarvis-v14/discussions)

---

<p align="center">
  Made with ❤️ for the Termux community
</p>
