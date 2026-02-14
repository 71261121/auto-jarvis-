#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Main Entry Point
=======================================

Self-Modifying AI Assistant for Termux/Linux
Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

This is the main entry point that integrates all JARVIS modules:
- Core Infrastructure (Events, Cache, Plugins, State Machine, Error Handler)
- AI Engine (OpenRouter Client, Rate Limiter, Model Selector, Response Parser)
- Self-Modification (Code Analyzer, Safe Modifier, Backup Manager, Improvement Engine)
- User Interface (CLI, Commands, Input, Output, Progress, Help, Notify)
- Installation System (Detect, Deps, Config, First Run, Update, Repair)
- Security System (Auth, Encryption, Sandbox, Permissions, Audit, Keys, Threat Detect)
- Memory System (Context Manager, Chat Storage, Conversation Indexer, Memory Optimizer)

Author: JARVIS Self-Modifying AI Project
Version: 14.0.0
License: MIT
"""

import sys
import os
import argparse
import signal
import json
import time
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

# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF IMPORTS - Graceful fallbacks for all modules
# ═══════════════════════════════════════════════════════════════════════════════

# Track module availability
CORE_AVAILABLE = True
AI_AVAILABLE = True
SELF_MOD_AVAILABLE = True
INTERFACE_AVAILABLE = True
INSTALL_AVAILABLE = True
SECURITY_AVAILABLE = True
MEMORY_AVAILABLE = True
CONFIG_AVAILABLE = True

# Core modules - individual imports with fallbacks
try:
    from core.events import EventEmitter, Event, EventHandler
except ImportError:
    CORE_AVAILABLE = False

try:
    from core.cache import MemoryCache, get_cache
except ImportError:
    pass  # Non-critical

try:
    from core.plugins import PluginManager, PluginContext
except ImportError:
    pass  # Non-critical

try:
    from core.state_machine import StateMachine, JarvisStates, create_jarvis_state_machine
except ImportError:
    CORE_AVAILABLE = False

try:
    from core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
except ImportError:
    pass  # Non-critical

# AI modules
try:
    from core.ai.openrouter_client import OpenRouterClient, FreeModel
    from core.ai.rate_limiter import RateLimiterManager
    from core.ai.model_selector import ModelSelector, TaskType
    from core.ai.response_parser import ResponseParser
except ImportError as e:
    AI_AVAILABLE = False

# Self-modification modules
try:
    from core.self_mod.code_analyzer import CodeAnalyzer
    from core.self_mod.safe_modifier import CodeValidator
    from core.self_mod.backup_manager import BackupManager
except ImportError:
    SELF_MOD_AVAILABLE = False

try:
    from core.self_mod.improvement_engine import SelfImprovementEngine
except ImportError:
    pass  # Non-critical

# Interface modules
try:
    from interface.cli import CLI, Colors, TerminalDetector
    from interface.commands import CommandRegistry, CommandProcessor
    from interface.input import InputHandler
    from interface.output import OutputFormatter
    from interface.session import SessionManager
    from interface.help import HelpSystem
except ImportError as e:
    INTERFACE_AVAILABLE = False

# Installation modules
try:
    from install.detect import EnvironmentDetector
    from install.deps import DependencyInstaller
    from install.config_gen import ConfigGenerator
    from install.first_run import FirstRunSetup
except ImportError:
    INSTALL_AVAILABLE = False

# Security modules
try:
    from security.auth import Authenticator, UserRole
    from security.encryption import EncryptionManager
    from security.sandbox import ExecutionSandbox
    from security.audit import AuditLogger
except ImportError:
    SECURITY_AVAILABLE = False

# Memory modules
try:
    from core.memory.context_manager import ContextManager
    from core.memory.chat_storage import ChatStorage
    from core.memory.memory_optimizer import MemoryOptimizer
except ImportError:
    MEMORY_AVAILABLE = False

# Config
try:
    from config.config_manager import ConfigManager
except ImportError:
    CONFIG_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JARVIS:
    """
    Main JARVIS Application Class
    
    Integrates all modules and provides unified interface for:
    - AI-powered conversations
    - Self-modification capabilities
    - Secure operations
    - Memory management
    """
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize JARVIS with all subsystems"""
        self.version = __version__
        self.debug = debug
        self.running = False
        self.start_time = None
        
        # Configuration
        self.config_path = config_path or os.path.expanduser("~/.jarvis/config.json")
        self.config = self._load_config()
        
        # Initialize subsystems
        self._init_core()
        self._init_ai()
        self._init_self_mod()
        self._init_interface()
        self._init_security()
        self._init_memory()
        
        # Setup signal handlers
        self._setup_signals()
        
        if self.debug:
            print(f"[DEBUG] JARVIS v{self.version} initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'general': {
                'app_name': 'JARVIS',
                'version': self.version,
                'debug_mode': self.debug,
                'quiet_mode': False
            },
            'ai': {
                'provider': 'openrouter',
                'model': 'openrouter/auto',
                'temperature': 0.7,
                'max_tokens': 2048,
                'enable_cache': True
            },
            'security': {
                'enable_auth': True,
                'session_timeout': 3600,
                'max_failed_attempts': 5
            },
            'memory': {
                'max_context_length': 100000,
                'enable_optimization': True
            },
            'self_mod': {
                'enable': True,
                'backup_dir': '~/.jarvis/backups',
                'max_backups': 50
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults
                    for key in loaded:
                        if key in default_config:
                            default_config[key].update(loaded[key])
                        else:
                            default_config[key] = loaded[key]
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Using default config: {e}")
        
        return default_config
    
    def _init_core(self):
        """Initialize core infrastructure"""
        self.event_bus = None
        self.cache = None
        self.plugin_manager = None
        self.state_machine = None
        self.error_handler = None
        
        if not CORE_AVAILABLE:
            return
        
        try:
            # Event bus for inter-module communication
            self.event_bus = EventEmitter()
            
            # Cache system
            cache_dir = os.path.expanduser("~/.jarvis/cache")
            self.cache = MemoryCache()
            
            # Plugin manager
            self.plugin_manager = PluginManager()
            
            # State machine
            self.state_machine = create_jarvis_state_machine()
            
            # Error handler
            self.error_handler = ErrorHandler()
            
            if self.debug:
                print("[DEBUG] Core infrastructure initialized")
        except Exception as e:
            print(f"Error initializing core: {e}")
    
    def _init_ai(self):
        """Initialize AI engine"""
        self.ai_client = None
        self.rate_limiter = None
        self.model_selector = None
        self.response_parser = None
        
        if not AI_AVAILABLE:
            return
        
        try:
            # Get API key
            api_key = os.environ.get('OPENROUTER_API_KEY')
            
            # Rate limiter
            self.rate_limiter = RateLimiterManager()
            
            # Model selector
            self.model_selector = ModelSelector()
            
            # Response parser
            self.response_parser = ResponseParser()
            
            # AI Client (optional - requires API key)
            if api_key:
                self.ai_client = OpenRouterClient(
                    api_key=api_key,
                    default_model=FreeModel.AUTO_FREE,
                    enable_cache=self.config['ai'].get('enable_cache', True)
                )
                if self.debug:
                    print("[DEBUG] AI client initialized with API key")
            else:
                if self.debug:
                    print("[DEBUG] AI client not initialized (no API key)")
                    
        except Exception as e:
            print(f"Error initializing AI: {e}")
    
    def _init_self_mod(self):
        """Initialize self-modification engine"""
        self.code_analyzer = None
        self.safe_modifier = None
        self.backup_manager = None
        self.improvement_engine = None
        
        if not SELF_MOD_AVAILABLE:
            return
        
        try:
            backup_dir = os.path.expanduser(self.config['self_mod'].get('backup_dir', '~/.jarvis/backups'))
            
            self.code_analyzer = CodeAnalyzer()
            self.backup_manager = BackupManager(backup_dir=backup_dir)
            self.safe_modifier = CodeValidator()
            self.improvement_engine = SelfImprovementEngine()
            
            if self.debug:
                print("[DEBUG] Self-modification engine initialized")
        except Exception as e:
            print(f"Error initializing self-mod: {e}")
    
    def _init_interface(self):
        """Initialize user interface"""
        self.cli = None
        self.command_processor = None
        self.input_handler = None
        self.output_formatter = None
        self.session_manager = None
        self.help_system = None
        
        if not INTERFACE_AVAILABLE:
            return
        
        try:
            self.output_formatter = OutputFormatter()
            self.input_handler = InputHandler()
            self.session_manager = SessionManager()
            self.help_system = HelpSystem()
            self.command_processor = CommandProcessor()
            self.cli = CLI()
            
            if self.debug:
                print("[DEBUG] User interface initialized")
        except Exception as e:
            print(f"Error initializing interface: {e}")
    
    def _init_security(self):
        """Initialize security systems"""
        self.authenticator = None
        self.encryption = None
        self.sandbox = None
        self.audit_logger = None
        
        if not SECURITY_AVAILABLE:
            return
        
        try:
            if self.config['security'].get('enable_auth', True):
                self.authenticator = Authenticator()
            
            self.encryption = EncryptionManager()
            self.sandbox = ExecutionSandbox()
            self.audit_logger = AuditLogger()
            
            if self.debug:
                print("[DEBUG] Security systems initialized")
        except Exception as e:
            print(f"Error initializing security: {e}")
    
    def _init_memory(self):
        """Initialize memory systems"""
        self.context_manager = None
        self.chat_storage = None
        self.memory_optimizer = None
        
        if not MEMORY_AVAILABLE:
            return
        
        try:
            data_dir = os.path.expanduser("~/.jarvis/data")
            os.makedirs(data_dir, exist_ok=True)
            
            self.context_manager = ContextManager()
            self.chat_storage = ChatStorage(storage_path=os.path.join(data_dir, "chat.db"))
            self.memory_optimizer = MemoryOptimizer()
            
            if self.debug:
                print("[DEBUG] Memory systems initialized")
        except Exception as e:
            print(f"Error initializing memory: {e}")
    
    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\nShutting down JARVIS...")
        self.shutdown()
        sys.exit(0)
    
    def start(self, interactive: bool = True):
        """Start JARVIS"""
        self.running = True
        self.start_time = datetime.now()
        
        # Set state to initializing
        if self.state_machine:
            self.state_machine.transition(JarvisStates.INITIALIZING.value)
        
        # Display welcome
        self._display_welcome()
        
        # Run first-time setup if needed
        if INSTALL_AVAILABLE and not self._is_first_run_complete():
            self._run_first_time_setup()
        
        # Set state to running
        if self.state_machine:
            self.state_machine.transition(JarvisStates.IDLE.value)
        
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
        print(Colors.CYAN + "║          Self-Modifying AI Assistant v" + __version__ + "                 ║" + Colors.RESET)
        print(Colors.CYAN + "║                                                              ║" + Colors.RESET)
        print(Colors.CYAN + "╚══════════════════════════════════════════════════════════════╝" + Colors.RESET)
        print()
        print(Colors.GREEN + "Welcome! Type 'help' for commands, 'exit' to quit." + Colors.RESET)
        print()
    
    def _is_first_run_complete(self) -> bool:
        """Check if first run setup is complete"""
        marker_file = os.path.expanduser("~/.jarvis/.initialized")
        return os.path.exists(marker_file)
    
    def _run_first_time_setup(self):
        """Run first-time setup"""
        print(Colors.YELLOW + "First time setup..." + Colors.RESET)
        
        try:
            setup = FirstRunSetup()
            setup.run()
            
            # Create marker file
            os.makedirs(os.path.dirname(os.path.expanduser("~/.jarvis/.initialized")), exist_ok=True)
            with open(os.path.expanduser("~/.jarvis/.initialized"), 'w') as f:
                f.write(datetime.now().isoformat())
                
            print(Colors.GREEN + "Setup complete!" + Colors.RESET)
        except Exception as e:
            print(f"Setup warning: {e}")
    
    def _run_interactive(self):
        """Run interactive mode"""
        while self.running:
            try:
                # Get input
                if self.input_handler:
                    user_input = self.input_handler.get_input("JARVIS> ")
                else:
                    user_input = input("JARVIS> ")
                
                # Handle command
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
        """Handle user command"""
        cmd_lower = command.lower()
        
        # Built-in commands
        if cmd_lower in ('exit', 'quit', 'bye'):
            print(Colors.GREEN + "Goodbye! JARVIS signing off..." + Colors.RESET)
            self.shutdown()
            self.running = False
            return
        
        if cmd_lower == 'help':
            self._show_help()
            return
        
        if cmd_lower == 'version':
            print(f"JARVIS v{self.version} ({__codename__})")
            return
        
        if cmd_lower == 'status':
            self._show_status()
            return
        
        if cmd_lower == 'clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            return
        
        # AI conversation
        if self.ai_client and cmd_lower not in ('analyze', 'modify', 'backup', 'restore'):
            self._handle_ai_command(command)
            return
        
        # Command processor for other commands
        if self.command_processor:
            try:
                result = self.command_processor.execute(command)
                if result:
                    print(result)
            except Exception as e:
                print(Colors.RED + f"Command error: {e}" + Colors.RESET)
        else:
            print(Colors.YELLOW + f"Unknown command: {command}" + Colors.RESET)
            print("Type 'help' for available commands.")
    
    def _handle_ai_command(self, command: str):
        """Handle AI conversation"""
        if not self.ai_client:
            print(Colors.YELLOW + "AI client not available. Set OPENROUTER_API_KEY environment variable." + Colors.RESET)
            return
        
        try:
            print(Colors.CYAN + "Thinking..." + Colors.RESET, end='\r')
            
            # Get conversation context
            context_id = "default"
            if self.context_manager:
                context_id = self.context_manager.get_or_create_context()
            
            # Send to AI
            response = self.ai_client.chat(
                message=command,
                context_id=context_id
            )
            
            if response.success:
                # Clear "Thinking..." line
                print(" " * 40, end='\r')
                
                # Display response
                if self.output_formatter:
                    formatted = self.output_formatter.format_markdown(response.content)
                    print(formatted)
                else:
                    print(response.content)
                
                # Store in chat history
                if self.chat_storage:
                    self.chat_storage.store_message(
                        context_id=context_id,
                        role='user',
                        content=command
                    )
                    self.chat_storage.store_message(
                        context_id=context_id,
                        role='assistant',
                        content=response.content
                    )
            else:
                print(Colors.RED + f"AI Error: {response.error or 'Unknown error'}" + Colors.RESET)
                
        except Exception as e:
            print(Colors.RED + f"AI error: {e}" + Colors.RESET)
    
    def _show_help(self):
        """Show help information"""
        if self.help_system:
            self.help_system.show_overview()
        else:
            print("""
Available Commands:
  help          - Show this help message
  version       - Show JARVIS version
  status        - Show system status
  clear         - Clear screen
  exit/quit     - Exit JARVIS
  
AI Commands (requires OPENROUTER_API_KEY):
  <any text>    - Chat with AI assistant

Self-Modification Commands:
  analyze <file>      - Analyze code file
  backup create       - Create backup
  backup list         - List backups
  backup restore <id> - Restore from backup

For more help, see docs/USER_GUIDE.md
""")
    
    def _show_status(self):
        """Show system status"""
        print(Colors.CYAN + "═══ JARVIS System Status ═══" + Colors.RESET)
        print(f"Version: {self.version}")
        print(f"Status: {'Running' if self.running else 'Stopped'}")
        
        if self.start_time:
            uptime = datetime.now() - self.start_time
            print(f"Uptime: {uptime}")
        
        print()
        print("Modules:")
        print(f"  Core Infrastructure: {'✓' if CORE_AVAILABLE else '✗'}")
        print(f"  AI Engine: {'✓' if AI_AVAILABLE else '✗'}")
        print(f"  Self-Modification: {'✓' if SELF_MOD_AVAILABLE else '✗'}")
        print(f"  User Interface: {'✓' if INTERFACE_AVAILABLE else '✗'}")
        print(f"  Security: {'✓' if SECURITY_AVAILABLE else '✗'}")
        print(f"  Memory: {'✓' if MEMORY_AVAILABLE else '✗'}")
        
        print()
        print("Configuration:")
        print(f"  Config file: {self.config_path}")
        print(f"  Debug mode: {self.debug}")
        print(f"  AI Model: {self.config['ai'].get('model', 'not set')}")
        print(f"  API Key: {'configured' if os.environ.get('OPENROUTER_API_KEY') else 'not set'}")
    
    def shutdown(self):
        """Graceful shutdown"""
        if self.debug:
            print("[DEBUG] Shutting down subsystems...")
        
        # Save chat history
        if self.chat_storage:
            try:
                self.chat_storage.flush()
            except:
                pass
        
        # Clear cache if needed
        if self.cache:
            try:
                self.cache.cleanup_expired()
            except:
                pass
        
        # Set state to shutdown
        if self.state_machine:
            try:
                self.state_machine.transition(JarvisStates.SHUTTING_DOWN.value)
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# CLI FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli(jarvis: JARVIS, args):
    """Run CLI mode"""
    jarvis.start(interactive=True)


def run_once(jarvis: JARVIS, args):
    """Run single command mode"""
    jarvis.start(interactive=False)
    
    if args.command:
        jarvis._handle_command(args.command)
    else:
        print("Error: No command specified for --once mode")
        sys.exit(1)


def run_daemon(jarvis: JARVIS, args):
    """Run daemon mode (background process)"""
    print(f"Starting JARVIS daemon...")
    jarvis.start(interactive=False)
    
    # Keep running
    try:
        while jarvis.running:
            time.sleep(1)
    except KeyboardInterrupt:
        jarvis.shutdown()


def show_info():
    """Show system information"""
    print(f"""
JARVIS v{__version__} ({__codename__})
Self-Modifying AI Assistant

Platform: {sys.platform}
Python: {sys.version.split()[0]}
Project Root: {PROJECT_ROOT}

Modules Available:
  Core: {CORE_AVAILABLE}
  AI: {AI_AVAILABLE}
  Self-Mod: {SELF_MOD_AVAILABLE}
  Interface: {INTERFACE_AVAILABLE}
  Security: {SECURITY_AVAILABLE}
  Memory: {MEMORY_AVAILABLE}

For more information, see README.md
""")


def run_tests():
    """Run all test suites"""
    import subprocess
    
    print("=" * 60)
    print("Running JARVIS Test Suite")
    print("=" * 60)
    
    test_files = [
        "research/test_phase1.py",
        "core/test_phase2.py",
        "core/ai/test_phase3.py",
        "core/self_mod/test_phase4.py",
        "interface/test_phase5.py",
        "install/test_phase6.py",
        "security/test_phase7.py",
        "docs/test_phase8.py"
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        test_path = PROJECT_ROOT / test_file
        if test_path.exists():
            print(f"\n--- Running: {test_file} ---")
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(PROJECT_ROOT),
                env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)},
                capture_output=True,
                text=True
            )
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"Test file not found: {test_file}")
    
    print("\n" + "=" * 60)
    print("Test run complete!")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"JARVIS v{__version__} - Self-Modifying AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start interactive mode
  python main.py --debug            Start with debug output
  python main.py --once "hello"     Run single command
  python main.py --test             Run test suite
  python main.py --info             Show system info
        """
    )
    
    parser.add_argument('--version', '-v', action='version', version=f'JARVIS {__version__}')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--info', '-i', action='store_true', help='Show system info')
    parser.add_argument('--test', '-t', action='store_true', help='Run test suite')
    parser.add_argument('--once', '-o', type=str, dest='command', help='Run single command')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Handle info mode
    if args.info:
        show_info()
        return 0
    
    # Handle test mode
    if args.test:
        run_tests()
        return 0
    
    # Create JARVIS instance
    jarvis = JARVIS(
        config_path=args.config,
        debug=args.debug
    )
    
    # Run appropriate mode
    if args.command:
        run_once(jarvis, args)
    elif args.daemon:
        run_daemon(jarvis, args)
    else:
        run_cli(jarvis, args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
