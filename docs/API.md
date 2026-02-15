# JARVIS v14 Ultimate - API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Module](#core-module)
3. [AI Module](#ai-module)
4. [Interface Module](#interface-module)
5. [Install Module](#install-module)
6. [Security Module](#security-module)
7. [Self-Modification Module](#self-modification-module)
8. [Memory Module](#memory-module)
9. [Events](#events)
10. [Examples](#examples)

---

## Overview

JARVIS provides a comprehensive Python API for programmatic access to all features. This documentation covers all public APIs, their parameters, return types, and usage examples.

### Import Convention

```python
# Core functionality
from core import get_importer, get_http_client, get_cache
from core import get_event_emitter, get_state_machine, get_error_handler

# AI functionality
from core.ai import OpenRouterClient, ModelSelector, ResponseParser

# Interface components
from interface import CLI, InputHandler, OutputFormatter, CommandProcessor

# Installation utilities
from install import EnvironmentDetector, DependencyInstaller, FirstRunSetup

# Security features
from security import Authenticator, EncryptionManager, AuditLogger

# Self-modification
from core.self_mod import CodeAnalyzer, SafeModifier, BackupManager
```

---

## Core Module

### `bulletproof_imports` - Safe Import System

Provides robust import handling with automatic fallbacks.

#### `get_importer()`

Returns the global importer instance.

**Returns:** `ImportManager`

**Example:**
```python
from core import get_importer

importer = get_importer()
numpy = importer.safe_import('numpy', fallback=None)
```

#### `safe_import(module_name, fallback=None, warn=True)`

Safely import a module with optional fallback.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module_name` | `str` | required | Name of module to import |
| `fallback` | `Any` | `None` | Value to return if import fails |
| `warn` | `bool` | `True` | Whether to log warning on failure |

**Returns:** `module` or `fallback`

**Example:**
```python
from core.bulletproof_imports import safe_import

# With fallback
np = safe_import('numpy', fallback=None)
if np is None:
    print("NumPy not available, using pure Python")

# Silent import
rich = safe_import('rich', warn=False)
```

#### `optional_import(module_name)`

Import that returns None on failure without warning.

**Returns:** `module` or `None`

---

### `http_client` - HTTP Client with Fallbacks

Layered HTTP client with automatic fallback chain.

#### `get_client()`

Returns the global HTTP client instance.

**Returns:** `HTTPClient`

#### `HTTPClient.request(method, url, **kwargs)`

Make an HTTP request with automatic fallback.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | required | HTTP method (GET, POST, etc.) |
| `url` | `str` | required | Target URL |
| `**kwargs` | `dict` | - | Additional request options |

**Returns:** `HTTPResponse`

**Example:**
```python
from core import get_http_client

client = get_http_client()
response = client.request('GET', 'https://api.example.com/data')
print(response.json())
```

#### `HTTPClient.get(url, **kwargs)`

Convenience method for GET requests.

**Returns:** `HTTPResponse`

#### `HTTPClient.post(url, data=None, json=None, **kwargs)`

Convenience method for POST requests.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | required | Target URL |
| `data` | `dict` | `None` | Form data |
| `json` | `dict` | `None` | JSON body |

**Returns:** `HTTPResponse`

---

### `cache` - Memory and Disk Cache

High-performance caching system with TTL support.

#### `get_cache()`

Returns the global cache instance.

**Returns:** `Cache`

#### `Cache.get(key, default=None)`

Retrieve a value from cache.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | required | Cache key |
| `default` | `Any` | `None` | Default if not found |

**Returns:** Cached value or default

#### `Cache.set(key, value, ttl=None, tags=None)`

Store a value in cache.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | required | Cache key |
| `value` | `Any` | required | Value to cache |
| `ttl` | `int` | `None` | Time-to-live in seconds |
| `tags` | `list` | `None` | Tags for grouping |

**Returns:** `bool`

**Example:**
```python
from core import get_cache

cache = get_cache()

# Basic caching
cache.set('user:123', {'name': 'Alex'}, ttl=3600)
user = cache.get('user:123')

# With tags
cache.set('config:theme', 'dark', tags=['config', 'ui'])

# Invalidate by tag
cache.invalidate_tag('config')
```

#### `Cache.delete(key)`

Remove a value from cache.

**Returns:** `bool`

#### `Cache.clear()`

Clear all cached values.

**Returns:** `int` (number of items cleared)

---

### `events` - Event System

Pub/sub event system with priorities and async support.

#### `get_event_emitter()`

Returns the global event emitter.

**Returns:** `EventEmitter`

#### `EventEmitter.on(event, handler, priority=0)`

Subscribe to an event.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event` | `str` | required | Event name (supports wildcards) |
| `handler` | `callable` | required | Event handler function |
| `priority` | `int` | `0` | Handler priority (higher = first) |

**Returns:** `str` (subscription ID)

**Example:**
```python
from core import get_event_emitter

emitter = get_event_emitter()

# Subscribe to event
def on_message(data):
    print(f"Received: {data}")

emitter.on('message', on_message)

# With priority
emitter.on('message', high_priority_handler, priority=10)

# Wildcard subscription
emitter.on('user.*', handle_user_events)
```

#### `EventEmitter.emit(event, data=None)`

Emit an event.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event` | `str` | required | Event name |
| `data` | `Any` | `None` | Event data |

**Returns:** `int` (number of handlers called)

#### `EventEmitter.off(subscription_id)`

Unsubscribe from an event.

**Returns:** `bool`

---

### `state_machine` - Finite State Machine

State management for JARVIS.

#### `create_jarvis_state_machine()`

Create a new state machine instance.

**Returns:** `StateMachine`

**Example:**
```python
from core import get_state_machine

sm = get_state_machine()

# Check current state
print(sm.current_state)  # 'idle'

# Transition
sm.transition('processing')
print(sm.current_state)  # 'processing'

# Check if transition allowed
if sm.can_transition('idle'):
    sm.transition('idle')
```

#### `StateMachine.states`

List of valid states.

#### `StateMachine.current_state`

Current state name.

#### `StateMachine.transition(target, **context)`

Transition to a new state.

**Returns:** `bool`

#### `StateMachine.on_enter(state, handler)`

Register handler for state entry.

#### `StateMachine.on_exit(state, handler)`

Register handler for state exit.

---

### `error_handler` - Error Handling

Global error handling and recovery.

#### `get_error_handler()`

Returns the global error handler.

**Returns:** `ErrorHandler`

#### `ErrorHandler.register_strategy(error_type, strategy)`

Register a recovery strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `error_type` | `type` | Exception class |
| `strategy` | `callable` | Recovery function |

**Example:**
```python
from core import get_error_handler

handler = get_error_handler()

def handle_network_error(error):
    print("Network error, retrying...")
    return 'retry'

handler.register_strategy(ConnectionError, handle_network_error)
```

#### `@safe_execute`

Decorator for safe function execution.

```python
from core.error_handler import safe_execute

@safe_execute(default=None, retries=3)
def risky_operation():
    # May raise exception
    return api.call()
```

---

## AI Module

### `OpenRouterClient` - AI API Client

Primary interface for AI interactions.

#### Constructor

```python
from core.ai import OpenRouterClient

client = OpenRouterClient(
    api_key='sk-or-...',      # Optional, uses env var
    model='llama-3.1-8b',     # Default model
    timeout=30                # Request timeout
)
```

#### `chat(message, context=None, **kwargs)`

Send a chat message.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | required | User message |
| `context` | `list` | `None` | Conversation context |
| `**kwargs` | `dict` | - | Model parameters |

**Returns:** `AIResponse`

**Example:**
```python
from core.ai import OpenRouterClient

client = OpenRouterClient()

# Simple chat
response = client.chat("Hello!")
print(response.content)

# With context
context = [
    {"role": "user", "content": "My name is Alex"},
    {"role": "assistant", "content": "Nice to meet you, Alex!"}
]
response = client.chat("What's my name?", context=context)
print(response.content)  # "Your name is Alex."

# With parameters
response = client.chat(
    "Write a poem",
    temperature=0.9,
    max_tokens=500
)
```

#### `stream(message, context=None, **kwargs)`

Stream response chunks.

**Yields:** `str` (text chunks)

```python
for chunk in client.stream("Tell me a story"):
    print(chunk, end='', flush=True)
```

#### `set_model(model_name)`

Switch the active model.

**Returns:** `bool`

#### `get_models()`

List available models.

**Returns:** `list[FreeModel]`

---

### `ModelSelector` - Intelligent Model Selection

Automatically select best model for task.

#### `select_model(task_type, **kwargs)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_type` | `TaskType` | Type of task (CODING, REASONING, etc.) |
| `**kwargs` | - | Task parameters |

**Returns:** `SelectionResult`

**Example:**
```python
from core.ai import ModelSelector, TaskType

selector = ModelSelector()

# Select for coding
result = selector.select_model(TaskType.CODING)
print(result.model)  # Best model for code
print(result.confidence)  # Selection confidence

# Select for creative writing
result = selector.select_model(TaskType.CREATIVE)
```

---

### `ResponseParser` - Parse AI Responses

Parse structured responses from AI.

#### `parse(response_text, expected_type=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_text` | `str` | Raw AI response |
| `expected_type` | `ResponseType` | Expected format |

**Returns:** `ParsedResponse`

**Example:**
```python
from core.ai import ResponseParser

parser = ResponseParser()

response = """
Here's the code:

```python
def hello():
    print("Hello!")
```
"""

parsed = parser.parse(response)
print(parsed.code_blocks)  # [{'language': 'python', 'code': '...'}]
print(parsed.text)  # "Here's the code:"
```

---

## Interface Module

### `CLI` - Command Line Interface

Main CLI component.

#### Constructor

```python
from interface import CLI, CLIConfig

cli = CLI(config=CLIConfig(
    prompt='jarvis>',
    history_file='~/.jarvis/history',
    completion_enabled=True
))
```

#### `run()`

Start the CLI main loop.

**Example:**
```python
from interface import CLI

cli = CLI()
cli.run()
```

#### `register_command(name, handler, help_text='')`

Register a custom command.

```python
def my_command(args):
    """Handle /my command"""
    return "Command executed!"

cli.register_command('my', my_command, help_text='My custom command')

# Now /my works in CLI
```

---

### `InputHandler` - Input Processing

Handle various input types.

#### Constructor

```python
from interface import InputHandler, InputConfig

handler = InputHandler(config=InputConfig(
    multiline_enabled=True,
    sanitization_level='normal'
))
```

#### `get_input(prompt='')`

Get user input.

**Returns:** `InputResult`

**Example:**
```python
from interface import InputHandler

handler = InputHandler()

result = handler.get_input('Enter command: ')
print(result.text)  # The input text
print(result.type)  # InputType (SINGLE_LINE, MULTI_LINE, FILE, etc.)
```

#### `sanitize(text, level='normal')`

Sanitize input text.

| Level | Description |
|-------|-------------|
| `none` | No sanitization |
| `normal` | Basic sanitization |
| `strict` | Maximum sanitization |

**Returns:** `str`

---

### `OutputFormatter` - Output Formatting

Format and display output.

#### `format(data, style='default')`

Format data for display.

```python
from interface import OutputFormatter

formatter = OutputFormatter()

# Format as table
data = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25}
]
print(formatter.format(data, style='table'))

# Format as JSON
print(formatter.format(data, style='json'))
```

#### `markdown(text)`

Render markdown to terminal.

```python
formatter.markdown('# Hello\n**Bold** text')
```

#### `code(code, language='python')`

Syntax highlight code.

```python
formatter.code('def hello(): pass', language='python')
```

---

## Install Module

### `EnvironmentDetector` - Environment Detection

Detect and validate the runtime environment.

#### `detect()`

Run all environment checks.

**Returns:** `EnvironmentInfo`

**Example:**
```python
from install import EnvironmentDetector

detector = EnvironmentDetector()
info = detector.detect()

print(f"Platform: {info.platform}")
print(f"Python: {info.python_version}")
print(f"Memory: {info.memory_available}MB")
print(f"Termux: {info.is_termux}")
```

#### `check_memory()`

Check available memory.

**Returns:** `CheckResult`

#### `check_python_version(min_version='3.9')`

Check Python version.

**Returns:** `CheckResult`

---

### `DependencyInstaller` - Install Dependencies

Install and manage dependencies.

#### `install(packages, strategy='safe')`

Install packages.

| Parameter | Type | Description |
|-----------|------|-------------|
| `packages` | `list` | Package names |
| `strategy` | `str` | 'safe', 'aggressive', 'minimal' |

**Returns:** `InstallResult`

**Example:**
```python
from install import DependencyInstaller

installer = DependencyInstaller()
result = installer.install(['numpy', 'pandas'], strategy='safe')

print(f"Installed: {result.installed}")
print(f"Failed: {result.failed}")
print(f"Fallbacks used: {result.fallbacks}")
```

---

### `FirstRunSetup` - Initial Setup Wizard

Guide user through initial setup.

#### `run()`

Run the setup wizard.

**Returns:** `SetupState`

**Example:**
```python
from install import FirstRunSetup

setup = FirstRunSetup()
state = setup.run()

if state.completed:
    print("Setup complete!")
    print(f"API key configured: {state.api_key_set}")
```

---

## Security Module

### `Authenticator` - User Authentication

Handle user authentication.

#### `login(username, password)`

Authenticate a user.

**Returns:** `AuthResult`

**Example:**
```python
from security import Authenticator

auth = Authenticator()
result = auth.login('admin', 'password123')

if result.success:
    print(f"Welcome, {result.user.name}")
    print(f"Session token: {result.token}")
else:
    print(f"Login failed: {result.error}")
```

#### `logout(token)`

End a session.

**Returns:** `bool`

#### `verify_token(token)`

Verify a session token.

**Returns:** `User` or `None`

---

### `EncryptionManager` - Data Encryption

Encrypt and decrypt data.

#### `encrypt(data, password=None)`

Encrypt data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `bytes/str` | Data to encrypt |
| `password` | `str` | Optional password |

**Returns:** `EncryptedData`

**Example:**
```python
from security import EncryptionManager

enc = EncryptionManager()

# Encrypt with auto-generated key
encrypted = enc.encrypt(b"Secret data")
print(encrypted.data)  # Encrypted bytes
print(encrypted.key_id)  # Key identifier

# Decrypt
decrypted = enc.decrypt(encrypted)
print(decrypted)  # b"Secret data"

# Encrypt with password
encrypted = enc.encrypt("Secret", password="mypass")
decrypted = enc.decrypt(encrypted, password="mypass")
```

#### `decrypt(encrypted_data, password=None)`

Decrypt data.

**Returns:** `bytes`

---

### `AuditLogger` - Security Auditing

Log security-relevant events.

#### `log(event_type, details, severity='info')`

Log an audit event.

```python
from security import AuditLogger, AuditEventType

audit = AuditLogger()

audit.log(
    AuditEventType.LOGIN,
    {'user': 'admin', 'ip': '192.168.1.1'},
    severity='info'
)

audit.log(
    AuditEventType.AUTH_FAILED,
    {'user': 'admin', 'reason': 'wrong_password'},
    severity='warning'
)
```

#### `query(filters=None, limit=100)`

Query audit logs.

**Returns:** `list[AuditEvent]`

---

## Self-Modification Module

### `CodeAnalyzer` - Code Analysis

Analyze code for improvements.

#### `analyze(code, file_path=None)`

Analyze code.

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | `str` | Source code |
| `file_path` | `str` | Optional file path |

**Returns:** `AnalysisResult`

**Example:**
```python
from core.self_mod import CodeAnalyzer

analyzer = CodeAnalyzer()

code = '''
def foo(x):
    return x * 2
'''

result = analyzer.analyze(code)

print(f"Complexity: {result.complexity}")
print(f"Issues: {result.issues}")
print(f"Suggestions: {result.suggestions}")
```

#### `get_metrics(code)`

Get code metrics.

**Returns:** `dict`

---

### `SafeModifier` - Safe Code Modification

Safely modify code with validation.

#### `modify(file_path, changes, backup=True)`

Apply modifications.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | File to modify |
| `changes` | `list` | List of changes |
| `backup` | `bool` | Create backup first |

**Returns:** `ModificationResult`

**Example:**
```python
from core.self_mod import SafeModifier

modifier = SafeModifier()

result = modifier.modify(
    'core/cache.py',
    changes=[
        {'line': 45, 'old': 'x = x + 1', 'new': 'x += 1'}
    ]
)

if result.success:
    print("Modified successfully")
    print(f"Backup: {result.backup_id}")
else:
    print(f"Failed: {result.error}")
```

---

### `BackupManager` - Backup System

Create and manage backups.

#### `create(paths, description='')`

Create a backup.

**Returns:** `str` (backup ID)

**Example:**
```python
from core.self_mod import BackupManager

backup = BackupManager()

# Create backup
backup_id = backup.create(
    ['core/cache.py', 'core/events.py'],
    description='Before optimization'
)
print(f"Backup created: {backup_id}")

# Restore backup
backup.restore(backup_id)

# List backups
backups = backup.list()
for b in backups:
    print(f"{b.id}: {b.description} ({b.date})")
```

#### `restore(backup_id)`

Restore from backup.

**Returns:** `bool`

#### `list(limit=10)`

List available backups.

**Returns:** `list[Backup]`

---

## Memory Module

### `ChatStorage` - Conversation Storage

Store and retrieve conversations.

#### `save(conversation)`

Save a conversation.

```python
from core.memory import ChatStorage

storage = ChatStorage()

storage.save({
    'messages': [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi!'}
    ],
    'metadata': {'model': 'llama-3'}
})
```

#### `load(conversation_id)`

Load a conversation.

**Returns:** `dict`

#### `search(query, limit=10)`

Search conversations.

**Returns:** `list[dict]`

---

### `ContextManager` - Conversation Context

Manage conversation context.

#### `add_message(role, content)`

Add a message to context.

```python
from core.memory import get_context_manager

ctx = get_context_manager()

ctx.add_message('user', 'Hello')
ctx.add_message('assistant', 'Hi there!')
```

#### `get_context(max_tokens=None)`

Get conversation context.

**Returns:** `list[dict]`

#### `clear()`

Clear the context.

---

### `MemoryOptimizer` - Memory Optimization

Optimize memory usage.

#### `optimize()`

Run memory optimization.

**Returns:** `dict` (optimization results)

**Example:**
```python
from core.memory import get_memory_optimizer

opt = get_memory_optimizer()

result = opt.optimize()
print(f"Freed: {result['freed_bytes']} bytes")
print(f"GC collected: {result['gc_collected']} objects")
```

---

## Events

### Event Types

| Event | Trigger | Data |
|-------|---------|------|
| `ai.request` | Before AI request | `{message, model}` |
| `ai.response` | After AI response | `{response, tokens}` |
| `ai.error` | AI error | `{error, retry_count}` |
| `modify.start` | Before modification | `{file, changes}` |
| `modify.complete` | After modification | `{result, backup_id}` |
| `backup.create` | Backup created | `{backup_id, paths}` |
| `backup.restore` | Backup restored | `{backup_id}` |
| `state.change` | State transition | `{from, to}` |
| `error.unhandled` | Unhandled error | `{error, traceback}` |

### Example Usage

```python
from core import get_event_emitter

emitter = get_event_emitter()

# Log all AI requests
@emitter.on('ai.request')
def log_request(data):
    print(f"[AI] Request to {data['model']}")

# Track modifications
@emitter.on('modify.complete')
def track_mod(data):
    print(f"[MOD] Modified: {data['result'].files_changed}")

# Handle errors
@emitter.on('error.unhandled')
def handle_error(data):
    send_alert(f"Error: {data['error']}")
```

---

## Examples

### Complete Example: Building a Custom Command

```python
from interface import CLI, CommandProcessor, CommandDefinition

# Create custom command
def weather_command(args, context):
    """Get weather for a location"""
    location = args[0] if args else 'current location'
    
    # Use HTTP client
    from core import get_http_client
    client = get_http_client()
    
    response = client.get(f'https://api.weather/{location}')
    data = response.json()
    
    return f"Weather in {location}: {data['temp']}°C, {data['condition']}"

# Register command
cli = CLI()
cli.register_command(
    'weather',
    weather_command,
    help_text='Get weather: /weather [location]'
)

# Run CLI
cli.run()
```

### Complete Example: AI-Powered Code Review

```python
from core.ai import OpenRouterClient
from core.self_mod import CodeAnalyzer
from interface import OutputFormatter

def review_code(file_path):
    """AI-powered code review"""
    # Read code
    with open(file_path) as f:
        code = f.read()
    
    # Analyze locally
    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze(code)
    
    # Get AI review
    client = OpenRouterClient()
    
    prompt = f"""
    Review this code and provide suggestions:
    
    ```python
    {code}
    ```
    
    Local analysis found:
    - Complexity: {analysis.complexity}
    - Issues: {len(analysis.issues)}
    """
    
    response = client.chat(prompt, model='llama-3.1-8b')
    
    # Format output
    formatter = OutputFormatter()
    print(formatter.markdown(response.content))
    
    return response.content

# Usage
review_code('my_module.py')
```

---

*Last Updated: Version 14.0.0*
