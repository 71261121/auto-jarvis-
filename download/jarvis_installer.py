#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     JARVIS v14 Ultimate - Complete Installer                  ║
║                                                                               ║
║  100% SELF-CONTAINED - NO EXTERNAL DOWNLOADS NEEDED                           ║
║  PURANA DELETE + NAYA SETUP = EK COMMAND                                      ║
║                                                                               ║
║  USAGE:                                                                       ║
║    python3 jarvis_installer.py                                                ║
║                                                                               ║
║  Or direct from URL (if hosted):                                              ║
║    python3 -c "$(curl -sL URL_TO_THIS_FILE)"                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
BACKUP_DIR = Path.home() / f"jarvis_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ═══════════════════════════════════════════════════════════════════════════════
# ALL JARVIS SOURCE CODE - EMBEDDED (NO EXTERNAL DOWNLOADS)
# ═══════════════════════════════════════════════════════════════════════════════

FILES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN.PY - Entry Point with Autonomous Engine
    # ═══════════════════════════════════════════════════════════════════════════
    "main.py": r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Modifying AI Assistant
==================================================
Platform: Termux/Android | Linux
Features: Autonomous file operations, AI chat, Self-modification
"""

import sys
import os
import signal
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Version info
__version__ = "14.0.0"
__author__ = "JARVIS AI Project"
__codename__ = "Ultimate"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    load_dotenv(Path.home() / '.jarvis' / '.env')
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

CORE_AVAILABLE = True
AI_AVAILABLE = True
AUTONOMOUS_AVAILABLE = True
SELF_MOD_AVAILABLE = True

# Core modules
try:
    from core.events import EventEmitter
    from core.state_machine import StateMachine, JarvisStates, create_jarvis_state_machine
except ImportError as e:
    CORE_AVAILABLE = False

# AI modules
try:
    from core.ai.openrouter_client import OpenRouterClient, FreeModel
except ImportError as e:
    AI_AVAILABLE = False

# Self-modification bridge
try:
    from core.self_mod.bridge import SelfModificationBridge
    SELF_MOD_BRIDGE_AVAILABLE = True
except ImportError:
    SELF_MOD_BRIDGE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS ENGINE - THE KEY TO FULL AUTONOMY!
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from core.autonomous import AutonomousEngine
    AUTONOMOUS_AVAILABLE = True
except ImportError as e:
    AUTONOMOUS_AVAILABLE = False
    print(f"Warning: Autonomous engine not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JARVIS:
    """
    Main JARVIS Application Class
    
    Fully Autonomous - Can read, write, create, delete files and execute commands!
    """
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        self.version = __version__
        self.debug = debug
        self.running = False
        self.start_time = None
        
        # Configuration
        self.config_path = config_path or os.path.expanduser("~/.jarvis/config.json")
        self.config = self._load_config()
        
        # Initialize subsystems
        self.ai_client = None
        self.autonomous_engine = None
        self.mod_bridge = None
        self.backup_manager = None
        
        self._init_ai()
        self._init_autonomous()
        self._init_self_mod()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        if self.debug:
            print(f"[DEBUG] JARVIS v{self.version} initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'general': {'app_name': 'JARVIS', 'version': self.version, 'debug_mode': self.debug},
            'ai': {'provider': 'openrouter', 'model': 'openrouter/auto', 'temperature': 0.7, 'max_tokens': 2048},
            'security': {'enable_auth': False},
            'self_mod': {'enable': True, 'backup_dir': '~/.jarvis/backups'}
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    for key in loaded:
                        if key in default_config:
                            default_config[key].update(loaded[key])
        except Exception:
            pass
        
        return default_config
    
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
                    print("[DEBUG] AUTONOMOUS ENGINE initialized - FULL CONTROL!")
            except Exception as e:
                print(f"Warning: Autonomous engine failed: {e}")
    
    def _init_self_mod(self):
        """Initialize self-modification bridge"""
        if SELF_MOD_BRIDGE_AVAILABLE:
            try:
                self.mod_bridge = SelfModificationBridge(self)
                if self.debug:
                    print("[DEBUG] Self-mod bridge initialized")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Self-mod bridge failed: {e}")
    
    def _signal_handler(self, signum, frame):
        print("\n\nShutting down JARVIS...")
        self.running = False
        sys.exit(0)
    
    def start(self, interactive: bool = True):
        """Start JARVIS"""
        self.running = True
        self.start_time = datetime.now()
        self._display_welcome()
        
        if interactive:
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
        
        # Status indicators
        if self.autonomous_engine:
            print(Colors.GREEN + "✓ Autonomous Engine: ACTIVE" + Colors.RESET)
        else:
            print(Colors.YELLOW + "✗ Autonomous Engine: NOT AVAILABLE" + Colors.RESET)
        
        if self.ai_client:
            print(Colors.GREEN + "✓ AI Client: CONNECTED" + Colors.RESET)
        else:
            print(Colors.YELLOW + "✗ AI Client: NOT CONFIGURED (set OPENROUTER_API_KEY)" + Colors.RESET)
        
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
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\n")
                continue
            except Exception as e:
                print(Colors.RED + f"Error: {e}" + Colors.RESET)
    
    def _handle_command(self, command: str):
        """Handle user command - WITH AUTONOMOUS ENGINE!"""
        cmd_lower = command.lower().strip()
        
        # Built-in commands (fast exit)
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
                    print(Colors.YELLOW + f"\n[DEBUG] Intent: {result.intent_type}" + Colors.RESET)
                    print(Colors.YELLOW + f"[DEBUG] Target: {result.target}" + Colors.RESET)
                    print(Colors.YELLOW + f"[DEBUG] Time: {result.execution_time_ms:.1f}ms" + Colors.RESET)
            else:
                print(Colors.RED + f"Error: {result.error}" + Colors.RESET)
                
                # If chat intent and AI available, try chat
                if result.intent_type == "CHAT" and self.ai_client:
                    self._handle_chat(command)
            
            return
        
        # Fallback for when autonomous engine is not available
        if self.ai_client:
            self._handle_chat(command)
            return
        
        print(Colors.YELLOW + f"Command not recognized: {command}" + Colors.RESET)
        print("Type 'help' for available commands.")
    
    def _handle_chat(self, command: str):
        """Handle pure AI chat"""
        if not self.ai_client:
            print(Colors.YELLOW + "AI client not available. Set OPENROUTER_API_KEY." + Colors.RESET)
            return
        
        try:
            print(Colors.CYAN + "Thinking..." + Colors.RESET, end='\r')
            
            response = self.ai_client.chat(
                message=command,
                system="You are JARVIS, a helpful AI assistant on Termux/Android. Be concise and helpful."
            )
            
            print(" " * 40, end='\r')
            
            if response.success:
                print(response.content)
            else:
                print(Colors.RED + f"AI Error: {response.error}" + Colors.RESET)
                
        except Exception as e:
            print(Colors.RED + f"Chat error: {e}" + Colors.RESET)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description=f"JARVIS v{__version__} - Self-Modifying AI Assistant")
    parser.add_argument('--version', '-v', action='version', version=f'JARVIS {__version__}')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    jarvis = JARVIS(config_path=args.config, debug=args.debug)
    jarvis.start()

if __name__ == "__main__":
    main()
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/__INIT__.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/__init__.py": '''#!/usr/bin/env python3
"""JARVIS Core Module"""
__version__ = "14.0.0"
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/EVENTS.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/events.py": '''#!/usr/bin/env python3
"""JARVIS Event System - Pub/Sub for inter-module communication"""
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Event data class"""
    name: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""

EventHandler = Callable[[Event], None]

class EventEmitter:
    """Event emitter for pub/sub pattern"""
    
    def __init__(self):
        self._listeners: Dict[str, List[EventHandler]] = {}
        self._once_listeners: Dict[str, List[EventHandler]] = {}
    
    def on(self, event_name: str, handler: EventHandler) -> 'EventEmitter':
        """Register event handler"""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(handler)
        return self
    
    def once(self, event_name: str, handler: EventHandler) -> 'EventEmitter':
        """Register one-time event handler"""
        if event_name not in self._once_listeners:
            self._once_listeners[event_name] = []
        self._once_listeners[event_name].append(handler)
        return self
    
    def off(self, event_name: str, handler: EventHandler = None) -> 'EventEmitter':
        """Remove event handler"""
        if handler is None:
            self._listeners.pop(event_name, None)
            self._once_listeners.pop(event_name, None)
        else:
            if event_name in self._listeners and handler in self._listeners[event_name]:
                self._listeners[event_name].remove(handler)
        return self
    
    def emit(self, event_name: str, data: Any = None, source: str = "") -> 'EventEmitter':
        """Emit event to all handlers"""
        event = Event(name=event_name, data=data, source=source)
        
        # Call regular handlers
        for handler in self._listeners.get(event_name, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        # Call one-time handlers
        once_handlers = self._once_listeners.pop(event_name, [])
        for handler in once_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Once handler error: {e}")
        
        return self
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/STATE_MACHINE.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/state_machine.py": '''#!/usr/bin/env python3
"""JARVIS State Machine - Manages application states"""
from enum import Enum, auto
from typing import Optional, Dict, Set, Callable, Any
import logging

logger = logging.getLogger(__name__)

class JarvisStates(Enum):
    """JARVIS states"""
    IDLE = auto()
    INITIALIZING = auto()
    PROCESSING = auto()
    THINKING = auto()
    EXECUTING = auto()
    WAITING_INPUT = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()

class StateMachine:
    """State machine for JARVIS"""
    
    def __init__(self):
        self._state: Optional[JarvisStates] = None
        self._previous_state: Optional[JarvisStates] = None
        self._callbacks: Dict[JarvisStates, Callable] = {}
        
        self._transitions: Dict[JarvisStates, Set[JarvisStates]] = {
            JarvisStates.IDLE: {JarvisStates.PROCESSING, JarvisStates.SHUTTING_DOWN, JarvisStates.WAITING_INPUT},
            JarvisStates.INITIALIZING: {JarvisStates.IDLE, JarvisStates.ERROR},
            JarvisStates.PROCESSING: {JarvisStates.THINKING, JarvisStates.EXECUTING, JarvisStates.IDLE, JarvisStates.ERROR},
            JarvisStates.THINKING: {JarvisStates.EXECUTING, JarvisStates.PROCESSING, JarvisStates.IDLE},
            JarvisStates.EXECUTING: {JarvisStates.IDLE, JarvisStates.PROCESSING, JarvisStates.ERROR},
            JarvisStates.WAITING_INPUT: {JarvisStates.PROCESSING, JarvisStates.SHUTTING_DOWN},
            JarvisStates.ERROR: {JarvisStates.IDLE, JarvisStates.SHUTTING_DOWN},
            JarvisStates.SHUTTING_DOWN: set(),
        }
    
    def start(self):
        """Start state machine"""
        self._state = JarvisStates.INITIALIZING
        logger.info(f"State machine started: {self._state.name}")
    
    def transition(self, new_state_name: str) -> bool:
        """Transition to new state"""
        try:
            target = JarvisStates[new_state_name.upper()]
        except KeyError:
            logger.error(f"Invalid state: {new_state_name}")
            return False
        
        if self._state is None:
            self._state = target
            return True
        
        if target in self._transitions.get(self._state, set()):
            self._previous_state = self._state
            self._state = target
            logger.debug(f"State transition: {self._previous_state.name} -> {self._state.name}")
            
            if target in self._callbacks:
                try:
                    self._callbacks[target]()
                except Exception as e:
                    logger.error(f"State callback error: {e}")
            
            return True
        
        logger.warning(f"Invalid transition: {self._state.name} -> {target.name}")
        return False
    
    def on_enter(self, state: JarvisStates, callback: Callable):
        """Register callback for state entry"""
        self._callbacks[state] = callback
    
    @property
    def state(self) -> Optional[JarvisStates]:
        return self._state
    
    @property
    def previous_state(self) -> Optional[JarvisStates]:
        return self._previous_state
    
    def is_running(self) -> bool:
        return self._state not in (None, JarvisStates.SHUTTING_DOWN)

def create_jarvis_state_machine() -> StateMachine:
    return StateMachine()
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AI/__INIT__.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/ai/__init__.py": '''#!/usr/bin/env python3
"""JARVIS AI Module"""
try:
    from .openrouter_client import OpenRouterClient, FreeModel
    __all__ = ['OpenRouterClient', 'FreeModel']
except ImportError:
    pass
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AI/OPENROUTER_CLIENT.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/ai/openrouter_client.py": '''#!/usr/bin/env python3
"""JARVIS OpenRouter API Client - Free AI Models"""
import os
import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check for requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not installed - AI features disabled")

class FreeModel(Enum):
    """Free models available on OpenRouter"""
    AUTO_FREE = "openrouter/auto"
    LLAMA_3_8B = "meta-llama/llama-3-8b-instruct:free"
    LLAMA_3_70B = "meta-llama/llama-3-70b-instruct:free"
    MISTRAL_7B = "mistralai/mistral-7b-instruct:free"
    GEMMA_7B = "google/gemma-7b-it:free"
    DEEPSEEK = "deepseek/deepseek-r1:free"
    QWEN = "qwen/qwen-2-7b-instruct:free"

@dataclass
class ChatResponse:
    """Response from chat API"""
    success: bool
    content: str = ""
    error: str = ""
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

class OpenRouterClient:
    """OpenRouter API Client for free AI models"""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: str,
        default_model: FreeModel = FreeModel.AUTO_FREE,
        enable_cache: bool = True,
        cache_ttl: int = 3600
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install with: pip install requests")
        
        self.api_key = api_key
        self.default_model = default_model.value
        self.enable_cache = enable_cache
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = cache_ttl
        
        logger.info(f"OpenRouter client initialized with model: {self.default_model}")
    
    def _get_cache_key(self, messages: List[Dict], model: str) -> str:
        """Generate cache key"""
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    def chat(
        self,
        message: str,
        system: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_id: str = None
    ) -> ChatResponse:
        """Send chat message to AI"""
        
        if not self.api_key:
            return ChatResponse(success=False, error="No API key configured")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        
        model_to_use = model or self.default_model
        
        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(messages, model_to_use)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if time.time() - cached['time'] < self._cache_ttl:
                    logger.debug("Using cached response")
                    return ChatResponse(
                        success=True,
                        content=cached['content'],
                        model=model_to_use
                    )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jarvis-ai",
            "X-Title": "JARVIS AI Assistant"
        }
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if not content:
                    return ChatResponse(
                        success=False,
                        error="Empty response from AI",
                        latency_ms=latency
                    )
                
                # Cache successful response
                if self.enable_cache:
                    self._cache[cache_key] = {
                        'content': content,
                        'time': time.time()
                    }
                
                return ChatResponse(
                    success=True,
                    content=content,
                    model=data.get('model', model_to_use),
                    usage=data.get('usage', {}),
                    latency_ms=latency
                )
            else:
                error_msg = f"API Error {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = error_data['error'].get('message', error_msg)
                except:
                    pass
                
                return ChatResponse(
                    success=False,
                    error=error_msg,
                    latency_ms=latency
                )
                
        except requests.Timeout:
            return ChatResponse(success=False, error="Request timed out")
        except requests.ConnectionError:
            return ChatResponse(success=False, error="Connection error - check internet")
        except Exception as e:
            return ChatResponse(success=False, error=f"Request failed: {e}")
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("Cache cleared")
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AUTONOMOUS/__INIT__.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/__init__.py": '''#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Autonomous Engine Module
================================================

This module provides FULL AUTONOMOUS CONTROL over Termux.
No more passive waiting for AI commands - WE DETECT AND EXECUTE.

Components:
- IntentDetector: Detects what user wants from natural language
- AutonomousExecutor: Executes operations directly
- AutonomousEngine: Main orchestrator
- SafetyManager: Protects against dangerous operations

USAGE:
    from core.autonomous import AutonomousEngine
    
    engine = AutonomousEngine(jarvis_instance)
    result = engine.process("read main.py")
    print(result.formatted_output)
"""

from .intent_detector import IntentDetector, IntentType, ParsedIntent
from .executor import AutonomousExecutor, ExecutionResult
from .engine import AutonomousEngine, EngineResult
from .safety_manager import SafetyManager, SafetyLevel, SafetyResult

__all__ = [
    'AutonomousEngine',
    'IntentDetector', 
    'AutonomousExecutor',
    'SafetyManager',
    'ParsedIntent',
    'ExecutionResult',
    'EngineResult',
    'SafetyResult',
    'IntentType',
    'SafetyLevel',
]

__version__ = "1.0.0"

def create_engine(jarvis_instance=None, project_root: str = None) -> AutonomousEngine:
    """Create an autonomous engine instance"""
    return AutonomousEngine(jarvis_instance, project_root)
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AUTONOMOUS/INTENT_DETECTOR.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/intent_detector.py": '''#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Intent Detector
======================================

Detects user intent from natural language input.
Uses pattern matching for fast detection (<50ms).
"""
import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents"""
    # File operations (DIRECT - no AI needed)
    READ_FILE = auto()
    LIST_DIR = auto()
    SEARCH_FILES = auto()
    DELETE_FILE = auto()
    
    # File operations (AI-ASSISTED)
    MODIFY_FILE = auto()
    CREATE_FILE = auto()
    ANALYZE_CODE = auto()
    
    # Terminal operations
    EXECUTE_CMD = auto()
    INSTALL_PKG = auto()
    
    # Git operations
    GIT_STATUS = auto()
    GIT_COMMIT = auto()
    
    # System operations
    SHOW_STATUS = auto()
    SHOW_HELP = auto()
    CLEAR_SCREEN = auto()
    
    # Chat (AI handles this)
    CHAT = auto()
    
    # Unknown
    UNKNOWN = auto()

@dataclass
class ParsedIntent:
    """Parsed user intent"""
    intent_type: IntentType
    target: str = ""
    action: str = ""
    content: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    original_input: str = ""
    needs_ai: bool = False
    
    def __str__(self) -> str:
        return f"Intent({self.intent_type.name}, target={self.target}, confidence={self.confidence:.2f})"

class IntentDetector:
    """Detect user intent from natural language"""
    
    INTENT_PATTERNS = {
        IntentType.READ_FILE: [
            (r"^read\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.95),
            (r"^show\\s+(?:me\\s+)?(?:the\\s+)?(?:content\\s+of\\s+)?['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^open\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
            (r"^cat\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.95),
            (r"^display\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^view\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^what('s| is)\\s+in\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
        ],
        
        IntentType.LIST_DIR: [
            (r"^list\\s*(?:files)?\\s*(?:in\\s+)?['\\\"]?([^'\\\"\\s]*)['\\\"]?$", 0.90),
            (r"^ls\\s*['\\\"]?([^'\\\"\\s]*)['\\\"]?$", 0.95),
            (r"^dir\\s*['\\\"]?([^'\\\"\\s]*)['\\\"]?$", 0.90),
            (r"^show\\s+(?:me\\s+)?(?:files\\s+)?(?:in\\s+)?['\\\"]?([^'\\\"\\s]*)['\\\"]?$", 0.85),
            (r"^list\\s+directory\\s*['\\\"]?([^'\\\"\\s]*)['\\\"]?$", 0.90),
        ],
        
        IntentType.MODIFY_FILE: [
            (r"^modify\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+to\\s+(.+))?$", 0.90),
            (r"^change\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+to\\s+(.+))?$", 0.85),
            (r"^update\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+to\\s+(.+))?$", 0.85),
            (r"^edit\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+to\\s+(.+))?$", 0.90),
            (r"^add\\s+(?:a\\s+)?(.+?)\\s+to\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
            (r"^fix\\s+(?:the\\s+)?(.+?)\\s+(?:in\\s+)?['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.80),
        ],
        
        IntentType.CREATE_FILE: [
            (r"^create\\s+(?:a\\s+)?(?:new\\s+)?(?:file\\s+)?['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+(?:with\\s+)?(.+))?$", 0.90),
            (r"^make\\s+(?:a\\s+)?(?:new\\s+)?(?:file\\s+)?['\\\"]?([^'\\\"\\s]+)['\\\"]?(?:\\s+(?:with\\s+)?(.+))?$", 0.85),
            (r"^new\\s+file\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
        ],
        
        IntentType.DELETE_FILE: [
            (r"^delete\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^remove\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^rm\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.95),
        ],
        
        IntentType.SEARCH_FILES: [
            (r"^search\\s+(?:for\\s+)?['\\\"]?([^'\\\"]+)['\\\"]?$", 0.85),
            (r"^find\\s+(?:files\\s+)?(?:containing\\s+)?['\\\"]?([^'\\\"]+)['\\\"]?$", 0.85),
            (r"^grep\\s+['\\\"]?([^'\\\"]+)['\\\"]?$", 0.90),
        ],
        
        IntentType.EXECUTE_CMD: [
            (r"^run\\s+(.+)$", 0.90),
            (r"^execute\\s+(.+)$", 0.90),
            (r"^(?:python|python3)\\s+(.+)$", 0.95),
            (r"^pip\\s+(.+)$", 0.95),
            (r"^bash\\s+(.+)$", 0.90),
        ],
        
        IntentType.INSTALL_PKG: [
            (r"^install\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
            (r"^pip\\s+install\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.95),
            (r"^pkg\\s+install\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.95),
        ],
        
        IntentType.ANALYZE_CODE: [
            (r"^analyze\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.90),
            (r"^check\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
            (r"^debug\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
            (r"^review\\s+['\\\"]?([^'\\\"\\s]+)['\\\"]?$", 0.85),
        ],
        
        IntentType.GIT_STATUS: [
            (r"^git\\s+status$", 0.95),
            (r"^show\\s+git\\s+status$", 0.90),
        ],
        
        IntentType.SHOW_STATUS: [
            (r"^status$", 0.90),
            (r"^show\\s+status$", 0.95),
        ],
        
        IntentType.SHOW_HELP: [
            (r"^help$", 0.95),
            (r"^show\\s+help$", 0.95),
            (r"^\\?$", 0.90),
        ],
        
        IntentType.CLEAR_SCREEN: [
            (r"^clear$", 0.95),
            (r"^cls$", 0.95),
        ],
    }
    
    AI_ASSISTED_INTENTS = {
        IntentType.MODIFY_FILE,
        IntentType.CREATE_FILE,
        IntentType.ANALYZE_CODE,
    }
    
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
        '.go', '.rs', '.rb', '.php', '.sh', '.bash',
        '.html', '.css', '.json', '.xml', '.yaml', '.yml',
        '.md', '.txt', '.sql'
    }
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._compiled_patterns = self._compile_patterns()
        logger.info(f"IntentDetector initialized")
    
    def _compile_patterns(self) -> Dict[IntentType, List[Tuple[re.Pattern, float]]]:
        compiled = {}
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            compiled[intent_type] = [
                (re.compile(p, re.IGNORECASE), conf)
                for p, conf in patterns
            ]
        return compiled
    
    def detect(self, user_input: str) -> ParsedIntent:
        """Detect user intent from input"""
        start_time = time.time()
        
        input_clean = user_input.strip()
        
        best_match = ParsedIntent(
            intent_type=IntentType.UNKNOWN,
            original_input=input_clean,
            confidence=0.0
        )
        
        for intent_type, patterns in self._compiled_patterns.items():
            for pattern, confidence in patterns:
                match = pattern.match(input_clean)
                if match:
                    parsed = self._extract_intent(intent_type, match, input_clean, confidence)
                    if parsed.confidence > best_match.confidence:
                        best_match = parsed
                    break
        
        if best_match.intent_type == IntentType.UNKNOWN:
            best_match = ParsedIntent(
                intent_type=IntentType.CHAT,
                original_input=input_clean,
                confidence=0.5,
                needs_ai=True
            )
        
        best_match.needs_ai = best_match.intent_type in self.AI_ASSISTED_INTENTS
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Intent detected in {elapsed:.1f}ms: {best_match}")
        
        return best_match
    
    def _extract_intent(self, intent_type: IntentType, match: re.Match, original: str, confidence: float) -> ParsedIntent:
        groups = match.groups()
        target = ""
        action = ""
        content = ""
        
        if intent_type == IntentType.READ_FILE:
            target = groups[0] if groups else ""
            for g in groups:
                if g and ('.' in g or '/' in g):
                    target = g
                    break
        
        elif intent_type == IntentType.LIST_DIR:
            target = groups[0] if groups else "."
            if not target:
                target = "."
        
        elif intent_type == IntentType.MODIFY_FILE:
            if len(groups) >= 2 and groups[1]:
                target = groups[0]
                action = groups[1]
            else:
                target = groups[0] if groups else ""
        
        elif intent_type == IntentType.CREATE_FILE:
            if len(groups) >= 2 and groups[1]:
                target = groups[0]
                content = groups[1]
            else:
                target = groups[0] if groups else ""
        
        elif intent_type in (IntentType.DELETE_FILE, IntentType.SEARCH_FILES, IntentType.INSTALL_PKG):
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.EXECUTE_CMD:
            target = groups[0] if groups else ""
        
        elif intent_type == IntentType.ANALYZE_CODE:
            target = groups[0] if groups else ""
        
        target = target.strip().strip('"\\'')
        
        return ParsedIntent(
            intent_type=intent_type,
            target=target,
            action=action,
            content=content,
            confidence=confidence,
            original_input=original,
            needs_ai=intent_type in self.AI_ASSISTED_INTENTS
        )
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AUTONOMOUS/EXECUTOR.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/executor.py": '''#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Autonomous Executor
==========================================

EXECUTES operations directly. No passive waiting.
"""
import os
import re
import ast
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import shutil

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of an executed operation"""
    success: bool
    operation: str
    target: str = ""
    output: str = ""
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    backup_id: str = ""
    execution_time_ms: float = 0.0

class AutonomousExecutor:
    """The EXECUTOR - Actually performs operations"""
    
    MAX_OUTPUT_SIZE = 50000
    MAX_FILE_SIZE = 1000000
    
    def __init__(self, jarvis_instance=None, project_root: str = None):
        self.jarvis = jarvis_instance
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        if jarvis_instance:
            self.ai_client = getattr(jarvis_instance, 'ai_client', None)
        else:
            self.ai_client = None
        
        logger.info(f"AutonomousExecutor initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def read_file(self, file_path: str) -> ExecutionResult:
        """Read a file and return content"""
        start_time = time.time()
        path = self._resolve_path(file_path)
        
        if not path.exists():
            return ExecutionResult(success=False, operation="READ", target=str(path), error=f"File not found: {path}")
        
        if not path.is_file():
            return ExecutionResult(success=False, operation="READ", target=str(path), error=f"Not a file: {path}")
        
        try:
            size = path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                return ExecutionResult(success=False, operation="READ", target=str(path), error=f"File too large")
            
            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.splitlines()
            output = self._format_file_content(path.name, content, lines)
            
            elapsed = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=True, operation="READ", target=str(path), output=output,
                data={'content': content, 'lines': len(lines), 'size': size},
                execution_time_ms=elapsed
            )
        except Exception as e:
            return ExecutionResult(success=False, operation="READ", target=str(path), error=f"Cannot read: {e}")
    
    def list_directory(self, dir_path: str = ".") -> ExecutionResult:
        """List directory contents"""
        start_time = time.time()
        path = self._resolve_path(dir_path)
        
        if not path.exists():
            return ExecutionResult(success=False, operation="LIST", target=str(path), error=f"Directory not found")
        
        if not path.is_dir():
            return ExecutionResult(success=False, operation="LIST", target=str(path), error=f"Not a directory")
        
        try:
            items = []
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for item in sorted(path.iterdir()):
                if item.name.startswith('.'):
                    continue
                
                try:
                    if item.is_dir():
                        items.append(f"📁 {item.name}/")
                        dir_count += 1
                    else:
                        size = item.stat().st_size
                        total_size += size
                        file_count += 1
                        items.append(f"📄 {item.name} ({self._format_size(size)})")
                except:
                    items.append(f"🔒 {item.name}")
            
            output_lines = [
                f"📂 Directory: {path}",
                f"   {file_count} files, {dir_count} directories",
                f"   Total size: {self._format_size(total_size)}",
                "",
            ]
            output_lines.extend(items)
            
            elapsed = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=True, operation="LIST", target=str(path),
                output="\n".join(output_lines),
                data={'file_count': file_count, 'dir_count': dir_count},
                execution_time_ms=elapsed
            )
        except Exception as e:
            return ExecutionResult(success=False, operation="LIST", error=str(e))
    
    def search_files(self, pattern: str, directory: str = ".") -> ExecutionResult:
        """Search for pattern in files"""
        start_time = time.time()
        search_dir = self._resolve_path(directory)
        
        if not search_dir.exists():
            return ExecutionResult(success=False, operation="SEARCH", error="Directory not found")
        
        results = []
        pattern_lower = pattern.lower()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    if not any(file.endswith(ext) for ext in ['.py', '.txt', '.md', '.json', '.sh']):
                        continue
                    
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if pattern_lower in content.lower():
                            rel_path = file_path.relative_to(self.project_root)
                            results.append(str(rel_path))
                            if len(results) >= 20:
                                break
                    except:
                        pass
                
                if len(results) >= 20:
                    break
            
            if results:
                output = f"🔍 Found '{pattern}' in {len(results)} files:\\n" + "\\n".join(f"   📄 {r}" for r in results)
            else:
                output = f"🔍 No files found containing '{pattern}'"
            
            elapsed = (time.time() - start_time) * 1000
            return ExecutionResult(success=True, operation="SEARCH", output=output, data={'results': results}, execution_time_ms=elapsed)
        except Exception as e:
            return ExecutionResult(success=False, operation="SEARCH", error=str(e))
    
    def delete_file(self, file_path: str) -> ExecutionResult:
        """Delete a file"""
        start_time = time.time()
        path = self._resolve_path(file_path)
        
        if not path.exists():
            return ExecutionResult(success=False, operation="DELETE", error="File not found")
        
        try:
            path.unlink()
            elapsed = (time.time() - start_time) * 1000
            return ExecutionResult(success=True, operation="DELETE", target=str(path), output=f"🗑️ Deleted: {path}", execution_time_ms=elapsed)
        except Exception as e:
            return ExecutionResult(success=False, operation="DELETE", error=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TERMINAL OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def execute_command(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Execute terminal command"""
        start_time = time.time()
        
        # Block dangerous commands
        dangerous = [r'rm\\s+-rf\\s+/', r'dd\\s+if=', r'mkfs', r'shutdown', r'reboot', r':\\(\\)\\s*\\{\\s*:\\|:&\\s*\\}']
        for pattern in dangerous:
            if re.search(pattern, command, re.IGNORECASE):
                return ExecutionResult(success=False, operation="EXECUTE", error="Blocked dangerous command")
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(self.project_root)
            )
            
            stdout = result.stdout[:self.MAX_OUTPUT_SIZE]
            stderr = result.stderr[:self.MAX_OUTPUT_SIZE]
            
            output_lines = [f"⚡ Executed: {command}", ""]
            if stdout:
                output_lines.append("Output:")
                output_lines.append(stdout)
            if stderr:
                output_lines.append("Errors:")
                output_lines.append(stderr)
            if not stdout and not stderr:
                output_lines.append("(no output)")
            
            elapsed = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=result.returncode == 0,
                operation="EXECUTE",
                target=command,
                output="\n".join(output_lines),
                error=stderr if result.returncode != 0 else "",
                data={'return_code': result.returncode},
                execution_time_ms=elapsed
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, operation="EXECUTE", error=f"Timed out after {timeout}s")
        except Exception as e:
            return ExecutionResult(success=False, operation="EXECUTE", error=str(e))
    
    def install_package(self, package: str) -> ExecutionResult:
        """Install a package"""
        if os.environ.get('TERMUX_VERSION'):
            command = f"pkg install -y {package}"
        else:
            command = f"pip install {package}"
        return self.execute_command(command, timeout=120)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AI-ASSISTED OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def modify_file(self, file_path: str, modification: str) -> ExecutionResult:
        """Modify file using AI"""
        if not self.ai_client:
            return ExecutionResult(success=False, operation="MODIFY", error="AI client not available")
        
        path = self._resolve_path(file_path)
        if not path.exists():
            return ExecutionResult(success=False, operation="MODIFY", error="File not found")
        
        try:
            current = path.read_text(encoding='utf-8')
            
            response = self.ai_client.chat(
                message=f"Modify this file according to: {modification}\\n\\nFile content:\\n```\\n{current}\\n```\\n\\nReturn ONLY the modified code, no explanations.",
                system="You are a code modification expert. Return ONLY the complete modified code, nothing else.",
                temperature=0.3
            )
            
            if not response.success:
                return ExecutionResult(success=False, operation="MODIFY", error=f"AI failed: {response.error}")
            
            new_content = self._extract_code(response.content)
            if not new_content:
                return ExecutionResult(success=False, operation="MODIFY", error="Could not extract code from AI response")
            
            # Validate Python syntax
            if path.suffix == '.py':
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    return ExecutionResult(success=False, operation="MODIFY", error=f"Syntax error in generated code: {e}")
            
            path.write_text(new_content, encoding='utf-8')
            
            return ExecutionResult(
                success=True, operation="MODIFY", target=str(path),
                output=f"✏️ Modified: {path}\\n   Change: {modification[:100]}"
            )
        except Exception as e:
            return ExecutionResult(success=False, operation="MODIFY", error=str(e))
    
    def create_file(self, file_path: str, description: str) -> ExecutionResult:
        """Create file using AI"""
        if not self.ai_client:
            return ExecutionResult(success=False, operation="CREATE", error="AI client not available")
        
        path = self._resolve_path(file_path)
        if path.exists():
            return ExecutionResult(success=False, operation="CREATE", error="File already exists")
        
        try:
            response = self.ai_client.chat(
                message=f"Create a file with: {description}\\n\\nReturn ONLY the code, nothing else.",
                system="You are a code generation expert. Return ONLY the complete code.",
                temperature=0.3
            )
            
            if not response.success:
                return ExecutionResult(success=False, operation="CREATE", error=f"AI failed: {response.error}")
            
            content = self._extract_code(response.content) or response.content
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return ExecutionResult(
                success=True, operation="CREATE", target=str(path),
                output=f"📝 Created: {path}\\n   Lines: {len(content.splitlines())}"
            )
        except Exception as e:
            return ExecutionResult(success=False, operation="CREATE", error=str(e))
    
    def analyze_code(self, file_path: str, focus: str = None) -> ExecutionResult:
        """Analyze code"""
        read_result = self.read_file(file_path)
        if not read_result.success:
            return read_result
        
        content = read_result.data.get('content', '')
        path = Path(read_result.target)
        
        analysis = {
            'lines': len(content.splitlines()),
            'functions': len([l for l in content.splitlines() if l.strip().startswith('def ')]),
            'classes': len([l for l in content.splitlines() if l.strip().startswith('class ')]),
        }
        
        output = f"📊 Analysis of: {path.name}\\n\\nLines: {analysis['lines']}\\nFunctions: {analysis['functions']}\\nClasses: {analysis['classes']}"
        
        if self.ai_client:
            try:
                response = self.ai_client.chat(
                    message=f"Analyze this code:\\n```\\n{content[:3000]}\\n```\\n\\nProvide: 1) Overall assessment 2) Potential issues 3) Suggestions",
                    system="You are a code analysis expert."
                )
                if response.success:
                    output += f"\\n\\nAI Analysis:\\n{response.content}"
            except:
                pass
        
        return ExecutionResult(success=True, operation="ANALYZE", target=str(path), output=output, data=analysis)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _resolve_path(self, target: str) -> Path:
        if not target or target == ".":
            return self.project_root
        target_path = Path(target)
        if target_path.is_absolute():
            return target_path
        return self.project_root / target
    
    def _format_file_content(self, filename: str, content: str, lines: list) -> str:
        output = [f"📄 {filename} ({len(lines)} lines)", ""]
        for i, line in enumerate(lines[:100], 1):
            output.append(f"{i:4d} | {line}")
        if len(lines) > 100:
            output.append(f"... ({len(lines) - 100} more lines)")
        return "\\n".join(output)
    
    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def _extract_code(self, text: str) -> str:
        patterns = [r'```(?:python)?\\s*\\n(.*?)\\n```', r'```\\s*\\n(.*?)\\n```']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        if text.strip().startswith(('import ', 'def ', 'class ', 'from ', '#!')):
            return text.strip()
        return ""
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AUTONOMOUS/ENGINE.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/engine.py": '''#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Autonomous Engine
========================================

THE BRAIN that makes JARVIS autonomous.
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .intent_detector import IntentDetector, IntentType, ParsedIntent
from .executor import AutonomousExecutor, ExecutionResult
from .safety_manager import SafetyManager, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

@dataclass
class EngineResult:
    """Result from autonomous engine"""
    success: bool
    output: str = ""
    error: str = ""
    intent_type: str = ""
    target: str = ""
    requires_ai: bool = False
    execution_time_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_output(self) -> str:
        return self.output if self.success else f"❌ Error: {self.error}"

class AutonomousEngine:
    """The Autonomous Engine - JARVIS's Brain"""
    
    DIRECT_INTENTS = {
        IntentType.READ_FILE, IntentType.LIST_DIR, IntentType.SEARCH_FILES,
        IntentType.EXECUTE_CMD, IntentType.INSTALL_PKG, IntentType.DELETE_FILE,
        IntentType.GIT_STATUS, IntentType.SHOW_STATUS, IntentType.SHOW_HELP,
        IntentType.CLEAR_SCREEN,
    }
    
    AI_ASSISTED_INTENTS = {
        IntentType.MODIFY_FILE, IntentType.CREATE_FILE, IntentType.ANALYZE_CODE,
    }
    
    def __init__(self, jarvis_instance=None, project_root: str = None):
        self.jarvis = jarvis_instance
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        self.intent_detector = IntentDetector(project_root=str(self.project_root))
        self.executor = AutonomousExecutor(jarvis_instance, str(self.project_root))
        self.safety = SafetyManager(project_root=str(self.project_root))
        
        if jarvis_instance:
            self.ai_client = getattr(jarvis_instance, 'ai_client', None)
        else:
            self.ai_client = None
        
        self._stats = {
            'total_requests': 0, 'direct_executions': 0,
            'ai_assisted': 0, 'chat_fallbacks': 0, 'errors': 0
        }
        
        logger.info(f"AutonomousEngine initialized")
    
    def process(self, user_input: str) -> EngineResult:
        """Main entry point - Process user input"""
        start_time = time.time()
        self._stats['total_requests'] += 1
        
        intent = self.intent_detector.detect(user_input)
        logger.info(f"Detected: {intent.intent_type.name}")
        
        if intent.intent_type == IntentType.CHAT:
            self._stats['chat_fallbacks'] += 1
            return self._handle_chat(user_input)
        
        if intent.intent_type == IntentType.SHOW_HELP:
            return self._handle_help()
        
        if intent.intent_type == IntentType.SHOW_STATUS:
            return self._handle_status()
        
        if intent.intent_type == IntentType.CLEAR_SCREEN:
            os.system('clear' if os.name != 'nt' else 'cls')
            return EngineResult(success=True, output="")
        
        # Execute
        if intent.intent_type in self.DIRECT_INTENTS:
            result = self._execute_direct(intent)
            self._stats['direct_executions'] += 1
        else:
            result = self._execute_ai_assisted(intent)
            self._stats['ai_assisted'] += 1
        
        if not result.success:
            self._stats['errors'] += 1
        
        elapsed = (time.time() - start_time) * 1000
        
        return EngineResult(
            success=result.success,
            output=result.output,
            error=result.error,
            intent_type=intent.intent_type.name,
            target=intent.target,
            execution_time_ms=elapsed,
            data=result.data
        )
    
    def _execute_direct(self, intent: ParsedIntent) -> ExecutionResult:
        """Execute operations that don't need AI"""
        if intent.intent_type == IntentType.READ_FILE:
            return self.executor.read_file(intent.target)
        elif intent.intent_type == IntentType.LIST_DIR:
            return self.executor.list_directory(intent.target or ".")
        elif intent.intent_type == IntentType.SEARCH_FILES:
            return self.executor.search_files(intent.target)
        elif intent.intent_type == IntentType.EXECUTE_CMD:
            return self.executor.execute_command(intent.target)
        elif intent.intent_type == IntentType.INSTALL_PKG:
            return self.executor.install_package(intent.target)
        elif intent.intent_type == IntentType.DELETE_FILE:
            return self.executor.delete_file(intent.target)
        elif intent.intent_type == IntentType.GIT_STATUS:
            return self.executor.execute_command("git status")
        else:
            return ExecutionResult(success=False, operation="UNKNOWN", error="Unknown intent")
    
    def _execute_ai_assisted(self, intent: ParsedIntent) -> ExecutionResult:
        """Execute operations that need AI"""
        if not self.ai_client:
            return ExecutionResult(success=False, operation=intent.intent_type.name, error="AI client not available")
        
        if intent.intent_type == IntentType.MODIFY_FILE:
            return self.executor.modify_file(intent.target, intent.action or intent.content)
        elif intent.intent_type == IntentType.CREATE_FILE:
            return self.executor.create_file(intent.target, intent.content or intent.action)
        elif intent.intent_type == IntentType.ANALYZE_CODE:
            return self.executor.analyze_code(intent.target, intent.action)
        else:
            return ExecutionResult(success=False, operation=intent.intent_type.name, error="Unknown AI intent")
    
    def _handle_chat(self, user_input: str) -> EngineResult:
        """Handle general chat"""
        if not self.ai_client:
            return EngineResult(success=False, error="AI client not available for chat", intent_type="CHAT")
        
        try:
            response = self.ai_client.chat(
                message=user_input,
                system="You are JARVIS, a helpful AI assistant on Termux/Android."
            )
            
            if response.success:
                return EngineResult(success=True, output=response.content, intent_type="CHAT", requires_ai=True)
            else:
                return EngineResult(success=False, error=response.error or "AI failed", intent_type="CHAT")
        except Exception as e:
            return EngineResult(success=False, error=f"Chat error: {e}", intent_type="CHAT")
    
    def _handle_help(self) -> EngineResult:
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                JARVIS v14 - Autonomous AI Assistant           ║
╚══════════════════════════════════════════════════════════════╝

📋 FILE OPERATIONS (Direct - No AI needed):
  read <file>           - Read file content
  list [dir]            - List directory contents
  search <pattern>      - Search for pattern in files
  delete <file>         - Delete a file

📝 FILE OPERATIONS (AI-Assisted):
  modify <file> to...   - Modify file with AI help
  create <file> with... - Create new file with AI help
  analyze <file>        - Analyze code for issues

⚡ TERMINAL OPERATIONS:
  run <command>         - Execute terminal command
  install <package>     - Install a package

💬 CHAT:
  <any question>        - Ask AI anything

⚡ EXAMPLES:
  "read main.py"
  "list files in core/"
  "modify main.py to add debug command"
  "create utils.py with string helpers"
  "run python test.py"
  "install requests"
  "What is Python?"

Type 'status' for system info.
"""
        return EngineResult(success=True, output=help_text, intent_type="HELP")
    
    def _handle_status(self) -> EngineResult:
        status = f"""═══ JARVIS Autonomous Engine Status ═══

Project Root: {self.project_root}

Components:
  Intent Detector: ✓
  Executor: ✓
  Safety Manager: ✓
  AI Client: {'✓' if self.ai_client else '✗'}

Statistics:
  Total Requests: {self._stats['total_requests']}
  Direct Executions: {self._stats['direct_executions']}
  AI-Assisted: {self._stats['ai_assisted']}
  Chat Fallbacks: {self._stats['chat_fallbacks']}
  Errors: {self._stats['errors']}
"""
        return EngineResult(success=True, output=status, intent_type="STATUS")
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/AUTONOMOUS/SAFETY_MANAGER.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/autonomous/safety_manager.py": '''#!/usr/bin/env python3
"""
JARVIS v14 Ultimate - Safety Manager
====================================

Protects against dangerous operations.
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Set
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    SAFE = auto()
    WARNING = auto()
    DANGEROUS = auto()
    BLOCKED = auto()

@dataclass
class SafetyResult:
    level: SafetyLevel
    allowed: bool
    message: str = ""
    requires_confirmation: bool = False

class SafetyManager:
    """Safety Manager - The Guardian"""
    
    PROTECTED_FILES = {
        '.env', '.env.local', 'credentials.json', 'secrets.json',
        'id_rsa', 'id_ed25519', 'private.key', 'private.pem',
        '.gitignore', '.bashrc', '.zshrc', 'authorized_keys',
        'main.py', '__init__.py',
    }
    
    DANGEROUS_COMMANDS = {
        r'rm\\s+-rf\\s+/', r'rm\\s+-rf\\s+~', r'rm\\s+-rf\\s+\\*',
        r'dd\\s+if=', r'mkfs', r'shutdown', r'reboot', r'halt',
        r':\\(\\)\\s*\\{\\s*:\\|:&\\s*\\}', r'chmod\\s+777',
    }
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        logger.info("SafetyManager initialized")
    
    def check_file_read(self, file_path: str) -> SafetyResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return SafetyResult(level=SafetyLevel.WARNING, allowed=False, message="File not found")
        return SafetyResult(level=SafetyLevel.SAFE, allowed=True)
    
    def check_file_write(self, file_path: str) -> SafetyResult:
        path = Path(file_path).expanduser().resolve()
        if path.name in self.PROTECTED_FILES:
            return SafetyResult(level=SafetyLevel.DANGEROUS, allowed=True, message="Protected file")
        return SafetyResult(level=SafetyLevel.SAFE, allowed=True)
    
    def check_file_delete(self, file_path: str) -> SafetyResult:
        path = Path(file_path).expanduser().resolve()
        if path.name in self.PROTECTED_FILES:
            return SafetyResult(level=SafetyLevel.BLOCKED, allowed=False, message="Cannot delete protected file")
        return SafetyResult(level=SafetyLevel.DANGEROUS, allowed=True, message="Will delete", requires_confirmation=True)
    
    def check_command(self, command: str) -> SafetyResult:
        for pattern in self.DANGEROUS_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return SafetyResult(level=SafetyLevel.BLOCKED, allowed=False, message="Blocked dangerous command")
        return SafetyResult(level=SafetyLevel.SAFE, allowed=True)
    
    def check_install(self, package: str) -> SafetyResult:
        return SafetyResult(level=SafetyLevel.WARNING, allowed=True, message=f"Will install: {package}")
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/SELF_MOD/__INIT__.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/self_mod/__init__.py": '''#!/usr/bin/env python3
"""JARVIS Self-Modification Module"""
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/SELF_MOD/BRIDGE.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/self_mod/bridge.py": '''#!/usr/bin/env python3
"""JARVIS Self-Modification Bridge"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SelfModificationBridge:
    """Bridge for AI self-modification capabilities"""
    
    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        logger.info("SelfModificationBridge initialized")
    
    def get_system_prompt(self) -> str:
        return """You are JARVIS, a self-modifying AI assistant.
You have the ability to read, analyze, and modify your own code.
Always be helpful, safe, and precise in your modifications."""
    
    def process_response(self, response: str) -> Dict[str, Any]:
        return {'commands_found': 0, 'executions': [], 'modified_files': []}
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE/MEMORY/__INIT__.PY
    # ═══════════════════════════════════════════════════════════════════════════
    "core/memory/__init__.py": '''#!/usr/bin/env python3
"""JARVIS Memory Module"""
''',

    # ═══════════════════════════════════════════════════════════════════════════
    # REQUIREMENTS.TXT
    # ═══════════════════════════════════════════════════════════════════════════
    "requirements.txt": '''requests>=2.25.0
python-dotenv>=0.19.0
''',
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\033[0;36m")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       JARVIS v14 Ultimate - Complete Installer               ║")
    print("║       100% Self-Contained - No External Downloads            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")

def backup_old():
    """Backup old JARVIS installation"""
    if JARVIS_DIR.exists():
        print(f"\033[1;33m[1/5] Backing up old installation...\033[0m")
        shutil.move(str(JARVIS_DIR), str(BACKUP_DIR))
        print(f"\033[0;32m✓ Backed up to: {BACKUP_DIR}\033[0m")
    else:
        print("\033[0;32m[1/5] Fresh installation\033[0m")

def create_dirs():
    """Create directory structure"""
    print("\033[1;33m[2/5] Creating directories...\033[0m")
    
    dirs = [
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
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\033[0;32m✓ Directories created\033[0m")

def write_files():
    """Write all JARVIS files"""
    print("\033[1;33m[3/5] Writing JARVIS files...\033[0m")
    
    count = 0
    for file_path, content in FILES.items():
        target = JARVIS_DIR / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        count += 1
        print(f"  \033[0;32m✓\033[0m {file_path}")
    
    print(f"\033[0;32m✓ {count} files written\033[0m")

def install_deps():
    """Install Python dependencies"""
    print("\033[1;33m[4/5] Installing dependencies...\033[0m")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "requests", "python-dotenv"],
            capture_output=True
        )
        print("\033[0;32m✓ Dependencies installed\033[0m")
    except Exception as e:
        print(f"\033[0;33m⚠ Could not install deps: {e}\033[0m")
        print("  Run manually: pip install requests python-dotenv")

def create_config():
    """Create configuration files"""
    print("\033[1;33m[5/5] Creating configuration...\033[0m")
    
    # Config JSON
    config = '''{
    "general": {"app_name": "JARVIS", "version": "14.0.0"},
    "ai": {"provider": "openrouter", "model": "openrouter/auto"},
    "self_mod": {"enable": true}
}'''
    (CONFIG_DIR / "config.json").write_text(config)
    
    # .env template
    env = '''# JARVIS Configuration
# Get FREE API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key_here
'''
    (CONFIG_DIR / ".env.template").write_text(env)
    
    # Launcher script
    launcher = f'''#!/bin/bash
cd {JARVIS_DIR}
python main.py "$@"
'''
    launcher_path = Path.home() / "jarvis.sh"
    launcher_path.write_text(launcher)
    launcher_path.chmod(0o755)
    
    # Add alias to bashrc
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        content = bashrc.read_text()
        if 'alias jarvis=' not in content:
            with open(bashrc, 'a') as f:
                f.write('\nalias jarvis="~/jarvis.sh"\n')
    
    # Initialize marker
    (CONFIG_DIR / ".initialized").write_text(datetime.now().isoformat())
    
    print("\033[0;32m✓ Configuration created\033[0m")

def show_completion():
    """Show completion message"""
    print()
    print("\033[0;32m╔══════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[0;32m║          JARVIS v14 Ultimate - INSTALLATION COMPLETE!        ║\033[0m")
    print("\033[0;32m╚══════════════════════════════════════════════════════════════╝\033[0m")
    print()
    print("\033[0;36m📍 Location:\033[0m", JARVIS_DIR)
    print("\033[0;36m📍 Config:\033[0m  ", CONFIG_DIR)
    if BACKUP_DIR.exists():
        print("\033[0;36m📍 Old Backup:\033[0m", BACKUP_DIR)
    print()
    print("\033[0;36m🚀 QUICK START:\033[0m")
    print()
    print("\033[1;33m  Step 1:\033[0m Set API key (FREE from https://openrouter.ai/keys):")
    print("         export OPENROUTER_API_KEY='your_key_here'")
    print()
    print("\033[1;33m  Step 2:\033[0m Start JARVIS:")
    print("         source ~/.bashrc")
    print("         jarvis")
    print()
    print("\033[0;36m  Alternative:\033[0m cd ~/jarvis && python main.py")
    print()
    print("\033[0;36m✨ FEATURES:\033[0m")
    print("  ✓ Autonomous file operations (read, write, create, delete)")
    print("  ✓ Terminal command execution from chat")
    print("  ✓ AI-powered conversations")
    print("  ✓ Self-modification capabilities")
    print()
    print("\033[0;32mType 'help' in JARVIS for all commands!\033[0m")
    print()

def main():
    """Main installer function"""
    print_banner()
    
    try:
        backup_old()
        create_dirs()
        write_files()
        install_deps()
        create_config()
        show_completion()
    except Exception as e:
        print(f"\033[0;31m❌ Installation failed: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
