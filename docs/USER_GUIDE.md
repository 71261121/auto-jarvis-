# JARVIS v14 Ultimate - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Commands Reference](#commands-reference)
4. [AI Features](#ai-features)
5. [Self-Modification](#self-modification)
6. [Memory Management](#memory-management)
7. [Security Features](#security-features)
8. [Tips and Tricks](#tips-and-tricks)
9. [Keyboard Shortcuts](#keyboard-shortcuts)

---

## Getting Started

### Starting JARVIS

Launch JARVIS from your terminal:

```bash
# Using the launcher
~/jarvis

# Or directly
cd ~/jarvis_v14_ultimate
python3 main.py
```

### First Launch

On your first launch, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║
║   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║
║   ██║███████║██████╔╝██║   ██║██║███████╗                   ║
║   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║
║   ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║
║   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║
║                                                              ║
║              Self-Modifying AI Assistant v14                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

JARVIS is ready. Type 'help' for commands or start chatting!
jarvis> _
```

### The Interface

JARVIS uses a command-line interface with these components:

1. **Prompt**: `jarvis>` indicates JARVIS is ready for input
2. **Output Area**: Responses and information display here
3. **Status Bar**: Shows current mode, memory usage, connection status

---

## Basic Usage

### Chatting with JARVIS

The simplest way to interact is through natural language:

```
jarvis> What is the capital of France?

The capital of France is Paris. It's known as the "City of Light" 
and is famous for landmarks like the Eiffel Tower, the Louvre Museum,
and Notre-Dame Cathedral.
```

### Multi-line Input

For longer messages, use multi-line mode:

```
jarvis> """
... This is a multi-line message.
... I can write code or long explanations.
... Press Enter twice to send.
... """

I received your multi-line message. How can I help you with this?
```

### Command Mode

Commands start with `/` or `!`:

```
jarvis> /help
jarvis> !status
jarvis> /config set ai.temperature 0.8
```

### File Input

Send files to JARVIS:

```
jarvis> /read myfile.py
jarvis> /analyze code.py
jarvis> Paste the content of document.txt
```

---

## Commands Reference

### Core Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/help` | `/?` | Show help menu |
| `/exit` | `/quit`, `/q` | Exit JARVIS |
| `/clear` | `/cls` | Clear the screen |
| `/status` | `/st` | Show system status |
| `/version` | `/v` | Show version info |

### AI Commands

| Command | Description |
|---------|-------------|
| `/ai on` | Enable AI responses |
| `/ai off` | Disable AI (use local fallback) |
| `/model [name]` | Switch AI model |
| `/models` | List available models |
| `/tokens` | Show token usage |

**Examples:**

```
jarvis> /models
Available AI Models:
  1. meta-llama/llama-3.1-8b-instruct:free (default)
  2. google/gemma-2-9b-it:free
  3. mistralai/mistral-7b-instruct:free

jarvis> /model google/gemma-2-9b-it:free
Model switched to gemma-2-9b-it
```

### Self-Modification Commands

| Command | Description |
|---------|-------------|
| `/analyze [file]` | Analyze code for improvements |
| `/modify [file] [desc]` | Request code modification |
| `/rollback [id]` | Rollback a modification |
| `/history` | Show modification history |
| `/backup` | Create backup |
| `/restore [id]` | Restore from backup |

**Examples:**

```
jarvis> /analyze core/cache.py
Analyzing core/cache.py...
Found 3 potential improvements:
  1. Line 45: Use context manager for file handling
  2. Line 89: Add type hints for better documentation
  3. Line 120: Consider using LRU cache decorator

jarvis> /modify core/cache.py "Add type hints"
Planning modification...
[Safety Check] Passed
[Backup] Created backup_20240114_001
[Modified] core/cache.py
Modification complete!
```

### Configuration Commands

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/config set [key] [value]` | Set a configuration value |
| `/config get [key]` | Get a configuration value |
| `/config reset` | Reset to defaults |
| `/config save` | Save current configuration |

**Examples:**

```
jarvis> /config set ai.temperature 0.5
Configuration updated: ai.temperature = 0.5

jarvis> /config get ai.model
ai.model = meta-llama/llama-3.1-8b-instruct:free
```

### Memory Commands

| Command | Description |
|---------|-------------|
| `/memory` | Show memory statistics |
| `/memory clear` | Clear conversation memory |
| `/memory export [file]` | Export memory to file |
| `/memory import [file]` | Import memory from file |
| `/context` | Show current context |

### Session Commands

| Command | Description |
|---------|-------------|
| `/session save [name]` | Save current session |
| `/session load [name]` | Load a saved session |
| `/session list` | List saved sessions |
| `/session delete [name]` | Delete a session |

### Security Commands

| Command | Description |
|---------|-------------|
| `/auth login` | Authenticate user |
| `/auth logout` | Logout current user |
| `/auth status` | Show authentication status |
| `/encrypt [file]` | Encrypt a file |
| `/decrypt [file]` | Decrypt a file |
| `/audit` | Show security audit log |

---

## AI Features

### Free AI Models

JARVIS uses OpenRouter to access free AI models:

| Model | Best For | Context |
|-------|----------|---------|
| Llama 3.1 8B | General chat, coding | 128K |
| Gemma 2 9B | Reasoning, analysis | 8K |
| Mistral 7B | Fast responses | 32K |
| Qwen 2 7B | Multilingual | 32K |

### Model Selection

JARVIS automatically selects the best model for your task:

```
jarvis> Write a Python function to sort a list

[Model: llama-3.1-8b] Selected for coding task

def sort_list(items, reverse=False):
    '''Sort a list in ascending or descending order.'''
    return sorted(items, reverse=reverse)
```

### Context Management

JARVIS maintains conversation context:

```
jarvis> My name is Alex
Nice to meet you, Alex!

jarvis> What's my name?
Your name is Alex.
```

To manage context:

```
jarvis> /context
Current Context:
  - User: Alex
  - Session: 45 minutes
  - Messages: 23
  - Tokens used: 3,420 / 128,000

jarvis> /context clear
Context cleared. Starting fresh.
```

### Response Modes

Control how JARVIS responds:

```
jarvis> /mode concise    # Brief responses
jarvis> /mode detailed   # Detailed responses  
jarvis> /mode code       # Code-focused responses
jarvis> /mode creative   # Creative responses
```

---

## Self-Modification

### Overview

JARVIS can modify its own code to improve performance, fix bugs, or add features. This is done safely with:

1. **Backup System**: All changes are backed up
2. **Safety Validation**: Code is checked before modification
3. **Rollback Capability**: Easy rollback to any previous state
4. **Audit Trail**: Complete history of all modifications

### Requesting Modifications

```
jarvis> The cache system seems slow. Can you optimize it?

Analyzing cache.py for performance issues...
Found optimization opportunities:
  1. Use dictionary comprehension (faster)
  2. Implement LRU eviction strategy
  3. Add memoization for repeated lookups

Would you like me to apply these optimizations? [y/N]: y

Creating backup... backup_20240114_002
Applying modifications...
  ✓ Line 45: Dictionary comprehension
  ✓ Line 78: LRU eviction
  ✓ Line 92: Memoization

Running tests...
  ✓ All 42 tests passed

Optimization complete! Estimated 23% performance improvement.
```

### Viewing History

```
jarvis> /history
Modification History (last 10):

ID       | Date       | File         | Description
---------|------------|--------------|----------------------------------
002      | 2024-01-14 | cache.py     | Performance optimization
001      | 2024-01-13 | analyzer.py  | Add complexity scoring
000      | 2024-01-12 | __init__.py  | Export new modules
```

### Rolling Back

```
jarvis> /rollback 001
Rollback to modification 001?

Changes that will be reverted:
  - cache.py performance optimizations
  
Continue? [y/N]: y

Restoring from backup_20240114_001...
Rollback complete!
```

### Backup Management

```
jarvis> /backup create "Before major changes"
Backup created: backup_20240114_003

jarvis> /backup list
Backups:
  003 | 2024-01-14 | Before major changes | 2.3 MB
  002 | 2024-01-14 | Pre-optimization      | 2.1 MB
  001 | 2024-01-13 | Daily backup          | 2.0 MB

jarvis> /backup restore 001
Restoring from backup_20240114_001...
Restore complete!
```

---

## Memory Management

### Memory Optimization

JARVIS is optimized for low-memory devices:

```
jarvis> /memory
Memory Statistics:
  Total: 4.0 GB
  Available: 1.8 GB
  JARVIS Usage: 45 MB
  
  Breakdown:
    - Core: 12 MB
    - AI Client: 8 MB
    - Cache: 15 MB
    - Conversation: 10 MB

Memory Status: Healthy ✓
```

### Clearing Memory

```
jarvis> /memory clear
What would you like to clear?
  1. Conversation history
  2. Cache
  3. Temporary files
  4. All of the above
  
Choice: 1

Conversation history cleared. Memory freed: 5 MB
```

### Conversation Export

```
jarvis> /memory export conversation.json
Exporting conversation...
Exported 156 messages to conversation.json

jarvis> /memory import conversation.json
Importing conversation...
Imported 156 messages from conversation.json
```

---

## Security Features

### Authentication

```
jarvis> /auth login
Username: admin
Password: ********
Authenticating...
Login successful! Welcome, admin.

jarvis> /auth status
Logged in as: admin
Role: Administrator
Session expires: 24 hours
```

### Encryption

Encrypt sensitive files:

```
jarvis> /encrypt secrets.txt
Encrypting secrets.txt...
Encrypted file: secrets.txt.enc
Key stored in secure storage.

jarvis> /decrypt secrets.txt.enc
Decrypting secrets.txt.enc...
Enter password: ********
Decrypted to: secrets.txt
```

### Audit Logging

View security events:

```
jarvis> /audit
Security Audit Log (last 24 hours):

Time           | Event           | User   | Details
---------------|-----------------|--------|------------------
14:32:15       | Login           | admin  | IP: 192.168.1.1
14:30:00       | Auth attempt    | ?      | Failed - wrong password
14:15:22       | File encrypt    | admin  | secrets.txt
14:00:00       | Session start   | admin  | New session
```

---

## Tips and Tricks

### Productivity Tips

1. **Use Aliases for Common Commands**
   ```
   jarvis> /alias c /config
   jarvis> /alias m /memory
   jarvis> c get ai.model  # Same as /config get ai.model
   ```

2. **Chain Commands**
   ```
   jarvis> /analyze file.py && /modify file.py "Add type hints"
   ```

3. **Quick Code Execution**
   ```
   jarvis> ```python
   ... print([x**2 for x in range(10)])
   ... ```
   [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
   ```

4. **Use Templates**
   ```
   jarvis> /template function --name my_func --args x,y
   def my_func(x, y):
       """Your function docstring here"""
       pass
   ```

### Performance Tips

1. **Enable Caching**
   ```
   jarvis> /config set cache.enabled true
   ```

2. **Use Streaming Responses**
   ```
   jarvis> /config set ai.streaming true
   ```

3. **Reduce Memory Usage**
   ```
   jarvis> /config set memory.max_history 50
   ```

### Debugging Tips

1. **Enable Debug Mode**
   ```
   jarvis> /debug on
   Debug mode enabled. Detailed logs will be shown.
   ```

2. **Verbose Output**
   ```
   jarvis> /verbose
   Verbose mode enabled.
   ```

3. **Check Logs**
   ```
   jarvis> /logs tail 50
   Showing last 50 log entries...
   ```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete command |
| `Up/Down` | Navigate history |
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` | Exit JARVIS |
| `Ctrl+L` | Clear screen |
| `Ctrl+R` | Search history |
| `Ctrl+A` | Go to line start |
| `Ctrl+E` | Go to line end |
| `Ctrl+K` | Clear to end of line |
| `Ctrl+U` | Clear entire line |

---

## Getting Help

### Built-in Help

```
jarvis> /help
jarvis> /help ai           # AI-specific help
jarvis> /help modify       # Self-modification help
jarvis> /help config       # Configuration help
```

### Online Resources

- [API Documentation](API.md)
- [Configuration Guide](CONFIGURATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [FAQ](FAQ.md)

### Community Support

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share tips
- Wiki: Community-maintained guides

---

*Last Updated: Version 14.0.0*
