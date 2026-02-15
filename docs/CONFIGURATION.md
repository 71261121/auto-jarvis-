# JARVIS v14 Ultimate - Configuration Guide

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Configuration Files](#configuration-files)
3. [General Settings](#general-settings)
4. [AI Settings](#ai-settings)
5. [Self-Modification Settings](#self-modification-settings)
6. [Interface Settings](#interface-settings)
7. [Storage Settings](#storage-settings)
8. [Network Settings](#network-settings)
9. [Security Settings](#security-settings)
10. [Logging Settings](#logging-settings)
11. [Environment Variables](#environment-variables)
12. [Advanced Configuration](#advanced-configuration)

---

## Configuration Overview

JARVIS uses a layered configuration system with the following priority (highest to lowest):

1. **Command-line arguments** - Override everything
2. **Environment variables** - System-level settings
3. **User configuration** - `~/.jarvis/config.json`
4. **Project configuration** - `./jarvis.json`
5. **Default configuration** - Built-in defaults

### Viewing Current Configuration

```bash
# CLI command
jarvis> /config

# Or via API
python3 -c "from config import ConfigManager; print(ConfigManager().get_all())"
```

---

## Configuration Files

### Main Configuration File

**Location:** `~/.jarvis/config.json`

```json
{
  "general": {
    "app_name": "JARVIS",
    "debug_mode": false,
    "quiet_mode": false,
    "locale": "en_US"
  },
  "ai": {
    "provider": "openrouter",
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 30
  },
  "self_modification": {
    "enabled": true,
    "auto_backup": true,
    "max_backups": 10,
    "require_confirmation": true
  },
  "interface": {
    "theme": "dark",
    "prompt": "jarvis>",
    "history_size": 1000,
    "completion_enabled": true
  },
  "storage": {
    "data_dir": "~/.jarvis/data",
    "cache_dir": "~/.jarvis/cache",
    "log_dir": "~/.jarvis/logs",
    "backup_dir": "~/.jarvis/backups"
  },
  "network": {
    "proxy": null,
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 1.0
  },
  "security": {
    "encryption_enabled": true,
    "audit_logging": true,
    "session_timeout": 86400
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": true,
    "console_logging": true
  }
}
```

### Project Configuration

**Location:** `./jarvis.json` (in project directory)

This file allows project-specific overrides:

```json
{
  "ai": {
    "model": "mistralai/mistral-7b-instruct:free"
  },
  "self_modification": {
    "allowed_paths": ["./src", "./lib"],
    "excluded_paths": ["./tests"]
  }
}
```

### Environment File

**Location:** `~/.jarvis/.env` or `./.env`

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx
JARVIS_DEBUG=false
JARVIS_LOG_LEVEL=INFO
```

---

## General Settings

### `general.app_name`

**Type:** `string`  
**Default:** `"JARVIS"`  
**Description:** Application name displayed in prompts and headers.

```json
{
  "general": {
    "app_name": "MyAssistant"
  }
}
```

### `general.debug_mode`

**Type:** `boolean`  
**Default:** `false`  
**Description:** Enable debug mode with verbose logging and additional diagnostics.

```json
{
  "general": {
    "debug_mode": true
  }
}
```

When enabled:
- All API calls are logged
- Stack traces shown for errors
- Timing information displayed
- Additional debug commands available

### `general.quiet_mode`

**Type:** `boolean`  
**Default:** `false`  
**Description:** Minimize output, show only essential information.

### `general.locale`

**Type:** `string`  
**Default:** `"en_US"`  
**Description:** Locale for internationalization.

**Supported locales:**
- `en_US` - English (US)
- `en_GB` - English (UK)
- `hi_IN` - Hindi
- `ur_PK` - Urdu

---

## AI Settings

### `ai.provider`

**Type:** `string`  
**Default:** `"openrouter"`  
**Description:** AI provider to use.

**Options:**
- `openrouter` - OpenRouter API (recommended)
- `openai` - OpenAI API
- `local` - Local fallback (pattern-based)

### `ai.model`

**Type:** `string`  
**Default:** `"meta-llama/llama-3.1-8b-instruct:free"`  
**Description:** Default AI model to use.

**Free Models (OpenRouter):**

| Model | Context | Best For |
|-------|---------|----------|
| `meta-llama/llama-3.1-8b-instruct:free` | 128K | General purpose |
| `google/gemma-2-9b-it:free` | 8K | Reasoning |
| `mistralai/mistral-7b-instruct:free` | 32K | Fast responses |
| `qwen/qwen-2-7b-instruct:free` | 32K | Multilingual |

```json
{
  "ai": {
    "model": "mistralai/mistral-7b-instruct:free"
  }
}
```

### `ai.temperature`

**Type:** `float`  
**Default:** `0.7`  
**Range:** `0.0 - 2.0`  
**Description:** Response randomness. Lower = more focused, higher = more creative.

```json
{
  "ai": {
    "temperature": 0.3  // More focused
  }
}
```

**Guidelines:**
- `0.0 - 0.3`: Code generation, factual answers
- `0.4 - 0.7`: Balanced responses
- `0.8 - 1.0`: Creative writing, brainstorming
- `1.0+`: Highly experimental

### `ai.max_tokens`

**Type:** `integer`  
**Default:** `2048`  
**Range:** `1 - 128000` (model dependent)  
**Description:** Maximum tokens in response.

### `ai.timeout`

**Type:** `integer`  
**Default:** `30`  
**Description:** Request timeout in seconds.

### `ai.system_prompt`

**Type:** `string`  
**Default:** `null`  
**Description:** Custom system prompt for AI interactions.

```json
{
  "ai": {
    "system_prompt": "You are a helpful coding assistant specialized in Python."
  }
}
```

### `ai.fallback_model`

**Type:** `string`  
**Default:** `"local"`  
**Description:** Model to use if primary fails.

### `ai.streaming`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable streaming responses.

### `ai.context_window`

**Type:** `integer`  
**Default:** `4096`  
**Description:** Maximum context tokens to maintain.

---

## Self-Modification Settings

### `self_modification.enabled`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable self-modification features.

### `self_modification.auto_backup`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Automatically backup before modifications.

### `self_modification.max_backups`

**Type:** `integer`  
**Default:** `10`  
**Description:** Maximum number of backups to keep.

### `self_modification.require_confirmation`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Require user confirmation before modifications.

### `self_modification.allowed_paths`

**Type:** `array`  
**Default:** All project paths  
**Description:** Paths where modifications are allowed.

```json
{
  "self_modification": {
    "allowed_paths": [
      "./core",
      "./interface",
      "./utils"
    ]
  }
}
```

### `self_modification.excluded_paths`

**Type:** `array`  
**Default:** `[]`  
**Description:** Paths excluded from modification.

```json
{
  "self_modification": {
    "excluded_paths": [
      "./tests",
      "./docs",
      "./config"
    ]
  }
}
```

### `self_modification.dangerous_patterns`

**Type:** `array`  
**Description:** Additional patterns to flag as dangerous.

```json
{
  "self_modification": {
    "dangerous_patterns": [
      "os.system",
      "subprocess.call",
      "__import__"
    ]
  }
}
```

---

## Interface Settings

### `interface.theme`

**Type:** `string`  
**Default:** `"dark"`  
**Description:** Color theme for terminal output.

**Options:**
- `dark` - Dark background, light text
- `light` - Light background, dark text
- `mono` - Monochrome (no colors)
- `custom` - Custom theme file

### `interface.prompt`

**Type:** `string`  
**Default:** `"jarvis>"`  
**Description:** Command prompt string.

**Variables:**
- `{name}` - Application name
- `{state}` - Current state
- `{time}` - Current time

```json
{
  "interface": {
    "prompt": "[{state}] {name}>"
  }
}
```

### `interface.history_size`

**Type:** `integer`  
**Default:** `1000`  
**Description:** Maximum command history entries.

### `interface.completion_enabled`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable tab completion.

### `interface.syntax_highlighting`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable syntax highlighting for code.

### `interface.markdown_rendering`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Render markdown in responses.

### `interface.progress_animations`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Show animated progress indicators.

---

## Storage Settings

### `storage.data_dir`

**Type:** `string`  
**Default:** `"~/.jarvis/data"`  
**Description:** Directory for data files.

### `storage.cache_dir`

**Type:** `string`  
**Default:** `"~/.jarvis/cache"`  
**Description:** Directory for cache files.

### `storage.log_dir`

**Type:** `string`  
**Default:** `"~/.jarvis/logs"`  
**Description:** Directory for log files.

### `storage.backup_dir`

**Type:** `string`  
**Default:** `"~/.jarvis/backups"`  
**Description:** Directory for backup files.

### `storage.cache_max_size_mb`

**Type:** `integer`  
**Default:** `100`  
**Description:** Maximum cache size in megabytes.

### `storage.cache_ttl`

**Type:** `integer`  
**Default:** `3600`  
**Description:** Default cache TTL in seconds.

---

## Network Settings

### `network.proxy`

**Type:** `string`  
**Default:** `null`  
**Description:** HTTP proxy URL.

```json
{
  "network": {
    "proxy": "http://proxy.example.com:8080"
  }
}
```

### `network.timeout`

**Type:** `integer`  
**Default:** `30`  
**Description:** Default network timeout in seconds.

### `network.retry_count`

**Type:** `integer`  
**Default:** `3`  
**Description:** Number of retries for failed requests.

### `network.retry_delay`

**Type:** `float`  
**Default:** `1.0`  
**Description:** Delay between retries in seconds.

### `network.retry_backoff`

**Type:** `float`  
**Default:** `2.0`  
**Description:** Backoff multiplier for retries.

### `network.user_agent`

**Type:** `string`  
**Default:** `"JARVIS/14.0"`  
**Description:** User-Agent header for HTTP requests.

---

## Security Settings

### `security.encryption_enabled`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable data encryption.

### `security.encryption_algorithm`

**Type:** `string`  
**Default:** `"AES-256-GCM"`  
**Description:** Encryption algorithm to use.

**Options:**
- `AES-256-GCM` - AES with GCM mode
- `ChaCha20-Poly1305` - ChaCha20 cipher
- `Fernet` - Python Fernet (simple)

### `security.audit_logging`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable security audit logging.

### `security.session_timeout`

**Type:** `integer`  
**Default:** `86400` (24 hours)  
**Description:** Session timeout in seconds.

### `security.max_login_attempts`

**Type:** `integer`  
**Default:** `5`  
**Description:** Maximum login attempts before lockout.

### `security.password_policy`

**Type:** `object`  
**Description:** Password requirements.

```json
{
  "security": {
    "password_policy": {
      "min_length": 8,
      "require_uppercase": true,
      "require_lowercase": true,
      "require_numbers": true,
      "require_special": false
    }
  }
}
```

---

## Logging Settings

### `logging.level`

**Type:** `string`  
**Default:** `"INFO"`  
**Description:** Log level.

**Options:**
- `DEBUG` - Detailed debugging info
- `INFO` - General information
- `WARNING` - Warning messages
- `ERROR` - Error messages only
- `CRITICAL` - Critical errors only

### `logging.format`

**Type:** `string`  
**Description:** Log message format.

```json
{
  "logging": {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### `logging.file_logging`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable logging to file.

### `logging.console_logging`

**Type:** `boolean`  
**Default:** `true`  
**Description:** Enable logging to console.

### `logging.max_file_size_mb`

**Type:** `integer`  
**Default:** `10`  
**Description:** Maximum log file size before rotation.

### `logging.backup_count`

**Type:** `integer`  
**Default:** `5`  
**Description:** Number of rotated log files to keep.

---

## Environment Variables

Environment variables override configuration file settings.

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JARVIS_DEBUG` | Enable debug mode | `false` |
| `JARVIS_LOG_LEVEL` | Logging level | `INFO` |
| `JARVIS_CONFIG` | Custom config path | `~/.jarvis/config.json` |
| `JARVIS_DATA_DIR` | Data directory | `~/.jarvis/data` |

### AI Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `AI_MODEL` | Default AI model |
| `AI_TEMPERATURE` | Response temperature |

### Network Variables

| Variable | Description |
|----------|-------------|
| `HTTP_PROXY` | HTTP proxy URL |
| `HTTPS_PROXY` | HTTPS proxy URL |
| `NO_PROXY` | Proxy bypass list |

### Setting Environment Variables

```bash
# Temporary (current session)
export OPENROUTER_API_KEY='sk-or-v1-xxxxx'

# Permanent (add to shell config)
echo 'export OPENROUTER_API_KEY="sk-or-v1-xxxxx"' >> ~/.bashrc
source ~/.bashrc

# In .env file
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxx" > ~/.jarvis/.env
```

---

## Advanced Configuration

### Performance Tuning

```json
{
  "performance": {
    "cache_size_mb": 50,
    "memory_limit_mb": 200,
    "worker_threads": 2,
    "async_enabled": true,
    "lazy_loading": true
  }
}
```

### Memory Optimization

```json
{
  "memory": {
    "max_history": 100,
    "context_compression": true,
    "gc_threshold": 0.8,
    "cleanup_interval": 300
  }
}
```

### Rate Limiting

```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 20,
    "tokens_per_minute": 40000,
    "burst_limit": 5
  }
}
```

### Custom Themes

Create a custom theme file at `~/.jarvis/themes/custom.json`:

```json
{
  "name": "my_theme",
  "colors": {
    "primary": "#00ff00",
    "secondary": "#008800",
    "error": "#ff0000",
    "warning": "#ffff00",
    "info": "#00ffff",
    "text": "#ffffff",
    "background": "#000000"
  },
  "styles": {
    "prompt": "bold primary",
    "response": "text",
    "error": "bold error",
    "code": "secondary"
  }
}
```

Then set in config:

```json
{
  "interface": {
    "theme": "custom"
  }
}
```

### Profiles

Create multiple configuration profiles:

```bash
# Development profile
~/.jarvis/profiles/dev.json

# Production profile  
~/.jarvis/profiles/prod.json
```

Switch profiles:

```bash
# Via environment variable
export JARVIS_PROFILE=dev

# Via CLI
jarvis> /profile dev

# Via command line
python3 main.py --profile prod
```

---

## Configuration Validation

JARVIS validates configuration on startup:

```bash
# Validate configuration
python3 main.py --validate-config

# Output
✓ General settings valid
✓ AI settings valid
✓ Storage paths exist
✓ Network settings valid
⚠ Warning: cache_max_size_mb is high for 4GB device
```

Invalid configurations will show errors:

```
✗ Error: Invalid temperature value '3.0' (must be 0.0-2.0)
✗ Error: Unknown model 'invalid-model-name'
```

---

*Last Updated: Version 14.0.0*
