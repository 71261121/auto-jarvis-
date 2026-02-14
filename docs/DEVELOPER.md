# JARVIS v14 Ultimate - Developer Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Development Setup](#development-setup)
4. [Code Style Guide](#code-style-guide)
5. [Testing](#testing)
6. [Contributing](#contributing)
7. [Extending JARVIS](#extending-jarvis)
8. [Plugin Development](#plugin-development)
9. [Debugging](#debugging)
10. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JARVIS v14 Ultimate                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   CLI/UI    │───►│   Core      │◄──►│   AI Engine             │ │
│  │  Interface  │    │   Layer     │    │   (OpenRouter)          │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────────────┘ │
│                            │                                        │
│         ┌──────────────────┼──────────────────┐                    │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Memory     │    │  Security   │    │ Self-Mod    │             │
│  │  System     │    │  Layer      │    │ Engine      │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Storage & Cache Layer                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
main.py
    │
    ├── interface/
    │   ├── cli.py ──────► commands.py
    │   ├── input.py
    │   └── output.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── bulletproof_imports.py
    │   ├── http_client.py
    │   ├── events.py
    │   ├── cache.py
    │   ├── state_machine.py
    │   ├── error_handler.py
    │   │
    │   ├── ai/
    │   │   ├── openrouter_client.py
    │   │   ├── model_selector.py
    │   │   ├── rate_limiter.py
    │   │   └── response_parser.py
    │   │
    │   ├── memory/
    │   │   ├── chat_storage.py
    │   │   ├── context_manager.py
    │   │   └── memory_optimizer.py
    │   │
    │   └── self_mod/
    │       ├── code_analyzer.py
    │       ├── safe_modifier.py
    │       ├── backup_manager.py
    │       └── improvement_engine.py
    │
    ├── security/
    │   ├── auth.py
    │   ├── encryption.py
    │   ├── sandbox.py
    │   ├── audit.py
    │   └── threat_detect.py
    │
    └── config/
        └── config_manager.py
```

### Design Principles

1. **Graceful Degradation**: All features have fallbacks
2. **Memory Efficiency**: Optimized for 4GB RAM devices
3. **Zero-Error Operation**: Robust error handling throughout
4. **Modularity**: Loose coupling between components
5. **Testability**: Every module is independently testable

---

## Project Structure

```
jarvis_v14_ultimate/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # Project readme
│
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── bulletproof_imports.py
│   ├── http_client.py
│   ├── events.py
│   ├── cache.py
│   ├── plugins.py
│   ├── state_machine.py
│   ├── error_handler.py
│   ├── safe_exec.py
│   ├── test_phase2.py
│   │
│   ├── ai/               # AI provider modules
│   │   ├── __init__.py
│   │   ├── openrouter_client.py
│   │   ├── model_selector.py
│   │   ├── rate_limiter.py
│   │   └── response_parser.py
│   │
│   ├── memory/           # Memory system
│   │   ├── __init__.py
│   │   ├── chat_storage.py
│   │   ├── context_manager.py
│   │   ├── conversation_indexer.py
│   │   └── memory_optimizer.py
│   │
│   └── self_mod/         # Self-modification
│       ├── __init__.py
│       ├── code_analyzer.py
│       ├── safe_modifier.py
│       ├── backup_manager.py
│       ├── improvement_engine.py
│       └── test_phase4.py
│
├── interface/            # User interface
│   ├── __init__.py
│   ├── cli.py
│   ├── input.py
│   ├── output.py
│   ├── commands.py
│   ├── session.py
│   ├── progress.py
│   ├── notify.py
│   ├── help.py
│   └── test_phase5.py
│
├── install/              # Installation system
│   ├── __init__.py
│   ├── detect.py
│   ├── deps.py
│   ├── config_gen.py
│   ├── first_run.py
│   ├── updater.py
│   ├── repair.py
│   ├── uninstall.py
│   ├── install.sh
│   └── test_phase6.py
│
├── security/             # Security modules
│   ├── __init__.py
│   ├── auth.py
│   ├── encryption.py
│   ├── sandbox.py
│   ├── audit.py
│   ├── threat_detect.py
│   ├── permissions.py
│   ├── keys.py
│   └── test_phase7.py
│
├── config/               # Configuration
│   ├── __init__.py
│   └── config_manager.py
│
├── docs/                 # Documentation
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   ├── API.md
│   ├── CONFIGURATION.md
│   ├── DEVELOPER.md
│   ├── TROUBLESHOOTING.md
│   └── FAQ.md
│
└── tests/                # Additional tests
    └── test_*.py
```

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- pip or poetry

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/jarvis/jarvis-v14.git
cd jarvis-v14

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

Create `requirements-dev.txt`:

```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.0.0
mypy>=1.0.0
pre-commit>=3.0.0
```

### IDE Setup

#### VS Code

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Open Settings → Python Interpreter
2. Add interpreter from `./venv/bin/python`
3. Enable pytest as test runner
4. Configure Black as external tool

---

## Code Style Guide

### Python Style

JARVIS follows PEP 8 with some modifications:

```python
# Line length: 100 characters (not 79)
# Use double quotes for strings
# Use trailing commas in multi-line structures

def example_function(
    param1: str,
    param2: int,
    param3: Optional[dict] = None,
) -> ReturnType:
    """
    Function description.

    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Optional parameter description

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
    """
    if not param1:
        raise ValueError("param1 cannot be empty")

    result = do_something(param1, param2)

    return ReturnType(
        field1=result.value,
        field2=param2,
    )
```

### Import Order

```python
# Standard library
import os
import sys
from typing import Optional, List, Dict

# Third-party
import requests
from rich.console import Console

# Local imports
from core import get_cache, get_event_emitter
from core.ai import OpenRouterClient
from interface import OutputFormatter
```

### Naming Conventions

```python
# Classes: PascalCase
class MyClassName:
    pass

# Functions/Methods: snake_case
def my_function_name():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Private: leading underscore
def _internal_function():
    pass

# Protected: single leading underscore
class MyClass:
    def _protected_method(self):
        pass
```

### Type Hints

All public APIs must have type hints:

```python
from typing import Optional, List, Dict, Any, Callable

def process_data(
    data: List[Dict[str, Any]],
    processor: Callable[[Dict], Optional[Dict]],
    max_items: int = 100,
) -> List[Dict[str, Any]]:
    """Process data items."""
    results: List[Dict[str, Any]] = []

    for item in data[:max_items]:
        processed = processor(item)
        if processed is not None:
            results.append(processed)

    return results
```

### Documentation Strings

```python
def complex_function(a: int, b: str, c: Optional[float] = None) -> Dict[str, Any]:
    """
    One-line summary.

    Longer description if needed. Can span multiple
    lines and paragraphs.

    Args:
        a: Description of parameter a.
        b: Description of parameter b.
        c: Optional parameter. Defaults to None.

    Returns:
        Description of return value with example:
        {
            'result': 'value',
            'status': 'success'
        }

    Raises:
        ValueError: If a is negative.
        TypeError: If b is not a string.

    Example:
        >>> result = complex_function(1, "test")
        >>> print(result['status'])
        'success'

    Note:
        Additional notes or warnings.
    """
    pass
```

---

## Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_core.py          # Core module tests
├── test_ai.py            # AI module tests
├── test_self_mod.py      # Self-modification tests
├── test_interface.py     # Interface tests
├── test_security.py      # Security tests
└── test_integration.py   # Integration tests
```

### Writing Tests

```python
import pytest
from core import get_cache

class TestCache:
    """Tests for cache module."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""
        cache = get_cache()
        cache.clear()
        return cache

    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set('key', 'value')
        assert cache.get('key') == 'value'

    def test_get_missing_key(self, cache):
        """Test getting a missing key returns default."""
        assert cache.get('missing') is None
        assert cache.get('missing', 'default') == 'default'

    def test_ttl_expiry(self, cache):
        """Test TTL expiration."""
        import time

        cache.set('key', 'value', ttl=1)
        assert cache.get('key') == 'value'

        time.sleep(1.1)
        assert cache.get('key') is None

    @pytest.mark.parametrize('value', [
        'string',
        123,
        {'key': 'value'},
        ['list', 'of', 'items'],
    ])
    def test_various_value_types(self, cache, value):
        """Test caching various value types."""
        cache.set('key', value)
        assert cache.get('key') == value
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=interface --cov=security

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestCache::test_set_and_get

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

### Test Fixtures

```python
# conftest.py
import pytest
from core import get_cache, get_event_emitter
from core.ai import OpenRouterClient

@pytest.fixture
def cache():
    """Fresh cache instance."""
    cache = get_cache()
    cache.clear()
    yield cache
    cache.clear()

@pytest.fixture
def event_emitter():
    """Fresh event emitter."""
    return get_event_emitter()

@pytest.fixture
def mock_ai_response(monkeypatch):
    """Mock AI responses for testing."""
    def mock_chat(message, **kwargs):
        from dataclasses import dataclass
        @dataclass
        class Response:
            content: str = "Test response"
        return Response()

    monkeypatch.setattr(OpenRouterClient, 'chat', mock_chat)
```

---

## Contributing

### Workflow

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to fork
git push origin feature/my-new-feature

# Create pull request on GitHub
```

### Commit Messages

Follow Conventional Commits:

```
feat: add new feature
fix: resolve bug in module
docs: update documentation
test: add tests for module
refactor: improve code structure
perf: optimize performance
chore: maintenance tasks
```

### Pull Request Guidelines

1. **Small PRs**: Keep changes focused
2. **Tests**: Include tests for new features
3. **Documentation**: Update docs if needed
4. **CI Pass**: All tests must pass

### Code Review Checklist

- [ ] Code follows style guide
- [ ] Type hints are present
- [ ] Docstrings are complete
- [ ] Tests are included
- [ ] No security issues
- [ ] Memory efficient
- [ ] Error handling is robust

---

## Extending JARVIS

### Adding New Commands

```python
# In interface/commands.py or a plugin

from interface import CommandProcessor

def register_commands(processor: CommandProcessor):
    """Register custom commands."""

    @processor.command('mycommand')
    def my_command(args, context):
        """
        Execute my custom command.

        Usage: /mycommand [args]
        """
        # Implementation
        return "Command executed!"

    @processor.command('calculate')
    def calculate_command(args, context):
        """Calculate mathematical expressions."""
        try:
            expression = ' '.join(args)
            result = eval(expression)  # Note: use safe_exec in production
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
```

### Adding New AI Models

```python
# In core/ai/model_selector.py

FREE_MODELS = {
    # Add new model
    'new-provider/new-model:free': ModelInfo(
        name='New Model',
        provider='new-provider',
        context_window=8192,
        capabilities=['chat', 'code'],
        cost_per_token=0.0,
    ),
}
```

### Adding New Events

```python
# Define event type
from core import get_event_emitter

emitter = get_event_emitter()

# Emit custom event
emitter.emit('custom.event', {
    'data': 'value',
    'timestamp': time.time(),
})

# Subscribe to custom event
@emitter.on('custom.event')
def handle_custom_event(data):
    print(f"Received: {data}")
```

---

## Plugin Development

### Plugin Structure

```
my_plugin/
├── __init__.py
├── plugin.py          # Main plugin class
├── commands.py        # Custom commands
├── hooks.py          # Event hooks
└── tests/
    └── test_plugin.py
```

### Plugin Class

```python
# my_plugin/plugin.py
from core.plugins import Plugin, PluginInfo

class MyPlugin(Plugin):
    """My custom JARVIS plugin."""

    info = PluginInfo(
        name='my_plugin',
        version='1.0.0',
        description='Does something useful',
        author='Your Name',
        dependencies=['requests'],  # Optional dependencies
    )

    def on_load(self):
        """Called when plugin is loaded."""
        self.logger.info("MyPlugin loaded!")

        # Register commands
        self.register_command('mycmd', self.my_command)

        # Subscribe to events
        self.on_event('ai.response', self.handle_ai_response)

    def on_unload(self):
        """Called when plugin is unloaded."""
        self.logger.info("MyPlugin unloaded!")

    def my_command(self, args, context):
        """Handle /mycmd command."""
        return "My command executed!"

    def handle_ai_response(self, data):
        """Handle AI response events."""
        self.logger.debug(f"AI response: {data['response'][:50]}...")
```

### Plugin Manifest

```yaml
# my_plugin/plugin.yaml
name: my_plugin
version: 1.0.0
description: Does something useful
author: Your Name

entry_point: plugin:MyPlugin

dependencies:
  - requests>=2.0.0

commands:
  - name: mycmd
    description: My custom command
    usage: /mycmd [args]

hooks:
  - event: ai.response
    handler: handle_ai_response
```

### Installing Plugins

```bash
# From directory
jarvis> /plugin install /path/to/my_plugin

# From GitHub
jarvis> /plugin install https://github.com/user/jarvis-plugin

# List installed plugins
jarvis> /plugin list

# Enable/disable plugin
jarvis> /plugin enable my_plugin
jarvis> /plugin disable my_plugin
```

---

## Debugging

### Debug Mode

```bash
# Enable debug mode
jarvis> /debug on

# Or via environment
export JARVIS_DEBUG=true
python3 main.py
```

### Logging

```python
import logging

# Get logger for module
logger = logging.getLogger('jarvis.core')

# Log levels
logger.debug("Detailed debug info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

### Debugging Tools

```python
# Interactive debugger
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()

# Pretty print
from pprint import pprint
pprint(complex_object)

# Inspect object
import inspect
print(inspect.getsource(some_function))
```

### Memory Debugging

```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Your code here
# ...

# Get snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = some_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Optimization

```python
# Use generators instead of lists
def get_items():
    for item in large_collection:
        yield process(item)  # Memory efficient

# Instead of:
def get_items_bad():
    return [process(item) for item in large_collection]  # Memory heavy

# Use __slots__ for classes with many instances
class OptimizedClass:
    __slots__ = ['field1', 'field2', 'field3']

    def __init__(self, field1, field2, field3):
        self.field1 = field1
        self.field2 = field2
        self.field3 = field3
```

### Async Optimization

```python
import asyncio

async def fetch_multiple(urls):
    """Fetch multiple URLs concurrently."""
    tasks = [fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

async def fetch_one(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### Cache Optimization

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n):
    """Cache results of expensive computations."""
    return complex_calculation(n)

# Clear cache when needed
expensive_computation.cache_clear()
```

---

*Last Updated: Version 14.0.0*
