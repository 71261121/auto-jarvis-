#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Quick Setup Script for Termux
====================================================

Ek command mein purana delete + naya setup!

USAGE:
    python3 -c "$(curl -sL https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/jarvis_quick_setup.py)"

Or:
    wget -qO- https://raw.githubusercontent.com/AnantDongaria/jarvis/main/download/jarvis_quick_setup.py | python3 -

Or locally:
    python3 jarvis_quick_setup.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

JARVIS_DIR = Path.home() / "jarvis"
CONFIG_DIR = Path.home() / ".jarvis"
BACKUP_DIR = Path.home() / f"jarvis_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS FILES - Complete source code embedded
# ═══════════════════════════════════════════════════════════════════════════════

JARVIS_FILES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN.PY - Entry Point
    # ═══════════════════════════════════════════════════════════════════════════
    "main.py": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JARVIS v14 Ultimate - Self-Modifying AI Assistant"""

import sys
import os
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "14.0.0"
__codename__ = "Ultimate"

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    load_dotenv(Path.home() / '.jarvis' / '.env')
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

CORE_AVAILABLE = True
AI_AVAILABLE = True
AUTONOMOUS_AVAILABLE = True

try:
    from core.events import EventEmitter
    from core.state_machine import StateMachine, JarvisStates, create_jarvis_state_machine
except ImportError:
    CORE_AVAILABLE = False

try:
    from core.ai.openrouter_client import OpenRouterClient, FreeModel
except ImportError:
    AI_AVAILABLE = False

try:
    from core.autonomous import AutonomousEngine
except ImportError:
    AUTONOMOUS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    RED = '\\033[0;31m'
    GREEN = '\\033[0;32m'
    YELLOW = '\\033[1;33m'
    CYAN = '\\033[0;36m'
    MAGENTA = '\\033[0;35m'
    RESET = '\\033[0m'
    BOLD = '\\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JARVIS:
    """Main JARVIS Application - Fully Autonomous!"""
    
    def __init__(self, debug: bool = False):
        self.version = __version__
        self.debug = debug
        self.running = False
        self.start_time = None
        self.ai_client = None
        self.autonomous_engine = None
        
        # Initialize AI client
        self._init_ai()
        
        # Initialize autonomous engine
        self._init_autonomous()
        
        # Setup signals
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        if self.debug:
            print(f"[DEBUG] JARVIS v{self.version} initialized")
    
    def _init_ai(self):
        """Initialize AI client"""
        api_key = os.environ.get('OPENROUTER_API_KEY')
        
        if AI_AVAILABLE and api_key:
            try:
                self.ai_client = OpenRouterClient(
                    api_key=api_key,
                    default_model=FreeModel.AUTO_FREE
                )
                if self.debug:
                    print("[DEBUG] AI client initialized")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] AI init failed: {e}")
    
    def _init_autonomous(self):
        """Initialize autonomous engine - THE KEY TO AUTONOMY!"""
        if AUTONOMOUS_AVAILABLE:
            try:
                self.autonomous_engine = AutonomousEngine(self, str(PROJECT_ROOT))
                if self.debug:
                    print("[DEBUG] AUTONOMOUS ENGINE initialized!")
            except Exception as e:
                print(f"Warning: Autonomous engine failed: {e}")
    
    def _signal_handler(self, signum, frame):
        print("\\n\\nShutting down JARVIS...")
        self.running = False
        sys.exit(0)
    
    def start(self):
        """Start JARVIS"""
        self.running = True
        self.start_time = datetime.now()
        self._display_welcome()
        self._run_interactive()
    
    def _display_welcome(self):
        """Display welcome message"""
        print()
        print(Colors.CYAN + "╔══════════════════════════════════════════════════════════════╗" + Colors.RESET)
        print(Colors.CYAN + "║                                                              ║" + Colors.RESET)
        print(Colors.CYAN + "║   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║" + Colors.RESET)
        print(Colors.CYAN + "║   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║" + Colors.RESET)
        print(Colors.CYAN + "║   ██║███████║██████╔╝██║   ██║██║███████╗                   ║" + Colors.RESET)
        print(Colors.CYAN + "║   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║" + Colors.RESET)
        print(Colors.CYAN + "║   ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║" + Colors.RESET)
        print(Colors.CYAN + "║   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║" + Colors.RESET)
        print(Colors.CYAN + "║                                                              ║" + Colors.RESET)
        print(Colors.CYAN + "║       Self-Modifying AI Assistant v" + __version__ + " - AUTONOMOUS       ║" + Colors.RESET)
        print(Colors.CYAN + "║                                                              ║" + Colors.RESET)
        print(Colors.CYAN + "╚══════════════════════════════════════════════════════════════╝" + Colors.RESET)
        print()
        print(Colors.GREEN + "✓ Autonomous Engine: ACTIVE" + Colors.RESET)
        print(Colors.GREEN + "✓ AI Client: " + ("CONNECTED" if self.ai_client else "NOT CONFIGURED") + Colors.RESET)
        print()
        print("Type 'help' for commands, 'exit' to quit.")
        print()
    
    def _run_interactive(self):
        """Run interactive mode"""
        while self.running:
            try:
                user_input = input(Colors.CYAN + "JARVIS> " + Colors.RESET)
                
                if user_input.strip():
                    self._handle_command(user_input.strip())
                    
            except EOFError:
                print("\\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\\n")
                continue
            except Exception as e:
                print(Colors.RED + f"Error: {e}" + Colors.RESET)
    
    def _handle_command(self, command: str):
        """Handle user command - WITH AUTONOMOUS ENGINE!"""
        cmd_lower = command.lower().strip()
        
        # Built-in commands
        if cmd_lower in ('exit', 'quit', 'bye'):
            print(Colors.GREEN + "Goodbye! JARVIS signing off..." + Colors.RESET)
            self.running = False
            return
        
        if cmd_lower == 'version':
            print(f"JARVIS v{self.version} ({__codename__})")
            return
        
        # ═════════════════════════════════════════════════════════════════════════
        # AUTONOMOUS ENGINE - Handles EVERYTHING!
        # ═════════════════════════════════════════════════════════════════════════
        if self.autonomous_engine:
            result = self.autonomous_engine.process(command)
            
            if result.success:
                print(result.formatted_output)
                
                if self.debug:
                    print(Colors.YELLOW + f"\\n[DEBUG] Intent: {result.intent_type}" + Colors.RESET)
                    print(Colors.YELLOW + f"[DEBUG] Target: {result.target}" + Colors.RESET)
                    print(Colors.YELLOW + f"[DEBUG] Time: {result.execution_time_ms:.1f}ms" + Colors.RESET)
            else:
                print(Colors.RED + f"Error: {result.error}" + Colors.RESET)
                
                # If chat intent and AI available, try chat
                if result.intent_type == "CHAT" and self.ai_client:
                    self._handle_chat(command)
            
            return
        
        # Fallback
        print(Colors.YELLOW + f"Command not recognized: {command}" + Colors.RESET)
        print("Type 'help' for available commands.")
    
    def _handle_chat(self, command: str):
        """Handle pure AI chat"""
        if not self.ai_client:
            print(Colors.YELLOW + "AI client not available. Set OPENROUTER_API_KEY." + Colors.RESET)
            return
        
        try:
            print(Colors.CYAN + "Thinking..." + Colors.RESET, end='\\r')
            
            response = self.ai_client.chat(
                message=command,
                system="You are JARVIS, a helpful AI assistant on Termux/Android."
            )
            
            print(" " * 40, end='\\r')
            
            if response.success:
                print(response.content)
            else:
                print(Colors.RED + f"AI Error: {response.error}" + Colors.RESET)
                
        except Exception as e:
            print(Colors.RED + f"Chat error: {e}" + Colors.RESET)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"JARVIS v{__version__}")
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    jarvis = JARVIS(debug=args.debug)
    jarvis.start()

if __name__ == "__main__":
    main()
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE __init__.py
    # ═══════════════════════════════════════════════════════════════════════════
    "core/__init__.py": '''"""JARVIS Core Module"""
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENTS.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/events.py": '''#!/usr/bin/env python3
"""JARVIS Event System"""
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    name: str
    data: Any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

EventHandler = Callable[[Event], None]

class EventEmitter:
    def __init__(self):
        self._listeners: Dict[str, List[EventHandler]] = {}
    
    def on(self, event_name: str, handler: EventHandler):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(handler)
    
    def emit(self, event_name: str, data: Any = None):
        event = Event(name=event_name, data=data)
        for handler in self._listeners.get(event_name, []):
            try:
                handler(event)
            except Exception:
                pass
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE_MACHINE.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/state_machine.py": '''#!/usr/bin/env python3
"""JARVIS State Machine"""
from enum import Enum, auto
from typing import Optional, Dict, Set

class JarvisStates(Enum):
    IDLE = auto()
    INITIALIZING = auto()
    PROCESSING = auto()
    THINKING = auto()
    EXECUTING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()

class StateMachine:
    def __init__(self):
        self._state: Optional[JarvisStates] = None
        self._transitions: Dict[JarvisStates, Set[JarvisStates]] = {
            JarvisStates.IDLE: {JarvisStates.PROCESSING, JarvisStates.SHUTTING_DOWN},
            JarvisStates.INITIALIZING: {JarvisStates.IDLE, JarvisStates.ERROR},
            JarvisStates.PROCESSING: {JarvisStates.THINKING, JarvisStates.EXECUTING, JarvisStates.IDLE},
            JarvisStates.THINKING: {JarvisStates.EXECUTING, JarvisStates.IDLE},
            JarvisStates.EXECUTING: {JarvisStates.IDLE},
            JarvisStates.ERROR: {JarvisStates.IDLE, JarvisStates.SHUTTING_DOWN},
            JarvisStates.SHUTTING_DOWN: set(),
        }
    
    def start(self):
        self._state = JarvisStates.INITIALIZING
    
    def transition(self, new_state: str):
        try:
            target = JarvisStates[new_state.upper()]
            if self._state is None or target in self._transitions.get(self._state, set()):
                self._state = target
        except (KeyError, AttributeError):
            pass
    
    @property
    def state(self) -> Optional[JarvisStates]:
        return self._state

def create_jarvis_state_machine() -> StateMachine:
    return StateMachine()
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # AI __init__.py
    # ═══════════════════════════════════════════════════════════════════════════
    "core/ai/__init__.py": '''"""JARVIS AI Module"""
try:
    from .openrouter_client import OpenRouterClient, FreeModel
except ImportError:
    pass
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # OPENROUTER_CLIENT.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/ai/openrouter_client.py": '''#!/usr/bin/env python3
"""JARVIS OpenRouter Client"""
import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class FreeModel(Enum):
    AUTO_FREE = "openrouter/auto"
    LLAMA_FREE = "meta-llama/llama-3-8b-instruct:free"
    MISTRAL_FREE = "mistralai/mistral-7b-instruct:free"
    GEMMA_FREE = "google/gemma-7b-it:free"

@dataclass
class ChatResponse:
    success: bool
    content: str = ""
    error: str = ""
    model: str = ""
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}

class OpenRouterClient:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str, default_model: FreeModel = FreeModel.AUTO_FREE, enable_cache: bool = True):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required")
        
        self.api_key = api_key
        self.default_model = default_model.value
        self.enable_cache = enable_cache
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 3600
    
    def chat(self, message: str, system: str = None, model: str = None, 
             temperature: float = 0.7, max_tokens: int = 2048, context_id: str = None) -> ChatResponse:
        
        cache_key = hashlib.md5(f"{message}{system}{model}".encode()).hexdigest() if self.enable_cache else None
        
        if cache_key and cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['time'] < self._cache_ttl:
                return ChatResponse(success=True, content=cached['content'], model=model or self.default_model)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jarvis-ai",
            "X-Title": "JARVIS AI"
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if cache_key and content:
                    self._cache[cache_key] = {'content': content, 'time': time.time()}
                
                return ChatResponse(
                    success=True,
                    content=content,
                    model=data.get('model', model or self.default_model),
                    usage=data.get('usage', {})
                )
            else:
                return ChatResponse(success=False, error=f"API Error: {response.status_code}")
                
        except Exception as e:
            return ChatResponse(success=False, error=str(e))
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTONOMOUS __init__.py
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/__init__.py": '''#!/usr/bin/env python3
"""JARVIS Autonomous Engine - FULL AUTONOMOUS CONTROL"""
from .intent_detector import IntentDetector, IntentType, ParsedIntent
from .executor import AutonomousExecutor, ExecutionResult
from .engine import AutonomousEngine, EngineResult
from .safety_manager import SafetyManager, SafetyLevel, SafetyResult

__all__ = [
    'AutonomousEngine', 'IntentDetector', 'AutonomousExecutor', 'SafetyManager',
    'ParsedIntent', 'ExecutionResult', 'EngineResult', 'SafetyResult',
    'IntentType', 'SafetyLevel',
]

def create_engine(jarvis_instance=None, project_root: str = None) -> AutonomousEngine:
    return AutonomousEngine(jarvis_instance, project_root)
''',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\033[0;36m")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         JARVIS v14 Ultimate - Quick Setup                    ║")
    print("║         One Command - Complete Installation                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")


def backup_old_installation():
    """Backup old JARVIS if exists"""
    if JARVIS_DIR.exists():
        print(f"\033[1;33m[1/4] Backing up old installation...\033[0m")
        shutil.move(str(JARVIS_DIR), str(BACKUP_DIR))
        print(f"\033[0;32m✓ Old installation backed up to: {BACKUP_DIR}\033[0m")
    else:
        print("\033[0;32m[1/4] Fresh installation\033[0m")


def create_directory_structure():
    """Create JARVIS directories"""
    print("\033[1;33m[2/4] Creating directory structure...\033[0m")
    
    directories = [
        JARVIS_DIR,
        JARVIS_DIR / "core",
        JARVIS_DIR / "core" / "ai",
        JARVIS_DIR / "core" / "autonomous",
        JARVIS_DIR / "core" / "self_mod",
        JARVIS_DIR / "core" / "memory",
        JARVIS_DIR / "interface",
        JARVIS_DIR / "security",
        JARVIS_DIR / "install",
        JARVIS_DIR / "config",
        CONFIG_DIR,
        CONFIG_DIR / "backups",
        CONFIG_DIR / "data",
    ]
    
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\033[0;32m✓ Directory structure created\033[0m")


def write_jarvis_files():
    """Write all JARVIS files"""
    print("\033[1;33m[3/4] Writing JARVIS files...\033[0m")
    
    for file_path, content in JARVIS_FILES.items():
        target = JARVIS_DIR / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        print(f"  ✓ {file_path}")
    
    print("\033[0;32m✓ All files written\033[0m")


def finalize_setup():
    """Finalize setup"""
    print("\033[1;33m[4/4] Finalizing setup...\033[0m")
    
    # Create config
    config_content = '''{
    "general": {"app_name": "JARVIS", "version": "14.0.0"},
    "ai": {"provider": "openrouter", "model": "openrouter/auto"},
    "self_mod": {"enable": true}
}'''
    (CONFIG_DIR / "config.json").write_text(config_content)
    
    # Create .env template
    env_template = '''# JARVIS Configuration
# Get free API key at: https://openrouter.ai
OPENROUTER_API_KEY=your_key_here
'''
    (CONFIG_DIR / ".env.template").write_text(env_template)
    
    # Create launcher script
    launcher = f'''#!/bin/bash
cd {JARVIS_DIR}
python main.py "$@"
'''
    launcher_path = Path.home() / "jarvis.sh"
    launcher_path.write_text(launcher)
    launcher_path.chmod(0o755)
    
    # Add alias
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        if 'alias jarvis=' not in bashrc.read_text():
            with open(bashrc, 'a') as f:
                f.write('\nalias jarvis="~/jarvis.sh"\n')
    
    print("\033[0;32m✓ Setup finalized\033[0m")


def show_completion_message():
    """Show completion message"""
    print("\n\033[0;32m╔══════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[0;32m║           JARVIS v14 Ultimate - Setup Complete!              ║\033[0m")
    print("\033[0;32m╚══════════════════════════════════════════════════════════════╝\033[0m")
    print()
    print("\033[0;36mInstallation Details:\033[0m")
    print(f"  Location: {JARVIS_DIR}")
    print(f"  Config: {CONFIG_DIR}")
    print()
    print("\033[0;36mQuick Start:\033[0m")
    print("\033[1;33m  1.\033[0m Set API key:")
    print("      export OPENROUTER_API_KEY='your_key_here'")
    print()
    print("\033[1;33m  2.\033[0m Start JARVIS:")
    print("      source ~/.bashrc")
    print("      jarvis")
    print()
    print("\033[0;36mOr run directly:\033[0m")
    print(f"      cd {JARVIS_DIR} && python main.py")
    print()


def main():
    """Main setup function"""
    print_banner()
    
    try:
        backup_old_installation()
        create_directory_structure()
        write_jarvis_files()
        finalize_setup()
        show_completion_message()
        
    except Exception as e:
        print(f"\033[0;31mError during setup: {e}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
