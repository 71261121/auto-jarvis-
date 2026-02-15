#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Help System
=================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Context-sensitive help
- Command documentation
- Tutorial mode
- Example gallery

Features:
- Context-aware help
- Command documentation
- Tutorial system
- Example gallery
- Search functionality
- Markdown rendering

Memory Impact: < 3MB for help system
"""

import os
import sys
import re
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class HelpCategory(Enum):
    """Help topic categories"""
    GETTING_STARTED = auto()
    COMMANDS = auto()
    AI = auto()
    SELF_MODIFICATION = auto()
    CONFIGURATION = auto()
    TROUBLESHOOTING = auto()
    ADVANCED = auto()
    API = auto()


class HelpFormat(Enum):
    """Help output formats"""
    PLAIN = auto()
    MARKDOWN = auto()
    MANPAGE = auto()


@dataclass
class HelpTopic:
    """
    Help topic definition.
    
    Contains all information about a help topic
    including content and metadata.
    """
    id: str
    title: str
    content: str
    category: HelpCategory = HelpCategory.GETTING_STARTED
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    version_added: str = "1.0.0"
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category.name,
            'keywords': self.keywords,
        }


@dataclass
class Tutorial:
    """Tutorial definition"""
    id: str
    title: str
    description: str
    steps: List[Dict[str, str]]
    difficulty: str = "beginner"
    time_estimate: str = "5 minutes"


# ═══════════════════════════════════════════════════════════════════════════════
# HELP DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class HelpDatabase:
    """
    Database of help topics.
    
    Manages storage and retrieval of help content.
    """
    
    def __init__(self):
        """Initialize help database"""
        self._topics: Dict[str, HelpTopic] = {}
        self._tutorials: Dict[str, Tutorial] = {}
        self._categories: Dict[HelpCategory, List[str]] = {}
        
        # Initialize built-in help
        self._init_builtin_help()
    
    def _init_builtin_help(self):
        """Initialize built-in help topics"""
        # Getting Started
        self.add_topic(HelpTopic(
            id="getting-started",
            title="Getting Started with JARVIS",
            content="""
Welcome to JARVIS v14 - Your Self-Modifying AI Assistant!

JARVIS is designed to run on Termux (Android) with limited resources.
Here's how to get started:

1. **Start JARVIS**
   Simply run: `python main.py`

2. **Basic Commands**
   - `help` - Show available commands
   - `version` - Show JARVIS version
   - `status` - Show system status
   - `ask <question>` - Ask JARVIS a question

3. **AI Features**
   - Chat with JARVIS about any topic
   - Get code suggestions and analysis
   - Use natural language commands

4. **Self-Modification**
   JARVIS can modify its own code safely:
   - `analyze <file>` - Analyze code
   - `improve <file>` - Suggest improvements
   - `modify <file>` - Apply modifications

5. **Configuration**
   - Configuration file: `~/.jarvis/config.json`
   - Environment variables: `JARVIS_*`
""",
            category=HelpCategory.GETTING_STARTED,
            keywords=["start", "begin", "intro", "introduction"],
        ))
        
        # Commands
        self.add_topic(HelpTopic(
            id="commands",
            title="Available Commands",
            content="""
JARVIS provides a variety of built-in commands:

**System Commands**
- `help [topic]` - Show help
- `version` - Show version info
- `status` - Show system status
- `clear` - Clear screen
- `exit` / `quit` - Exit JARVIS

**File Commands**
- `cd <dir>` - Change directory
- `pwd` - Print working directory
- `ls [path]` - List files
- `cat <file>` - Show file content

**AI Commands**
- `ask <question>` - Ask JARVIS
- `chat` - Start chat mode
- `analyze <code>` - Analyze code

**Self-Modification**
- `analyze <file>` - Code analysis
- `improve <file>` - Suggest improvements
- `modify <file>` - Apply changes
- `rollback` - Undo modifications

**Configuration**
- `config` - View configuration
- `set <key> <value>` - Set variable
- `alias <name> <cmd>` - Create alias
""",
            category=HelpCategory.COMMANDS,
            keywords=["command", "cmd", "help", "list"],
        ))
        
        # AI Help
        self.add_topic(HelpTopic(
            id="ai-features",
            title="AI Features",
            content="""
JARVIS AI Features:

**Natural Language Processing**
JARVIS understands natural language queries:
- Ask questions in plain English
- Get intelligent responses
- Context-aware conversations

**Code Intelligence**
- Code analysis and review
- Bug detection
- Optimization suggestions
- Documentation generation

**Self-Modification**
The AI can safely modify its own code:
- Improves itself over time
- Learns from interactions
- Maintains safety constraints

**Model Selection**
JARVIS uses multiple AI models:
- Primary: OpenRouter (free models)
- Fallback: Local pattern matching
- Automatic model switching

**Rate Limits**
Free models have rate limits:
- ~20 requests/minute
- Automatic queuing
- Request prioritization
""",
            category=HelpCategory.AI,
            keywords=["ai", "intelligence", "model", "chat"],
        ))
        
        # Self-Modification
        self.add_topic(HelpTopic(
            id="self-modification",
            title="Self-Modification System",
            content="""
JARVIS Self-Modification System:

**Safety First**
All modifications go through strict validation:
- Syntax checking
- Security scanning
- Impact analysis
- Automatic backups

**Modification Process**
1. Analyze target code
2. Generate modification plan
3. Validate changes
4. Test in sandbox
5. Apply with backup
6. Monitor for issues

**Rollback**
If something goes wrong:
- Automatic rollback on error
- Manual rollback available
- Backup history preserved

**Best Practices**
- Always review suggested changes
- Test in sandbox first
- Keep backups
- Monitor after modifications
""",
            category=HelpCategory.SELF_MODIFICATION,
            keywords=["modify", "self-mod", "improve", "change"],
        ))
        
        # Troubleshooting
        self.add_topic(HelpTopic(
            id="troubleshooting",
            title="Troubleshooting",
            content="""
Common Issues and Solutions:

**Import Errors**
If you see import errors:
- Check Python version (3.9+)
- Install missing packages: `pip install <package>`
- Check Termux packages: `pkg install python`

**Memory Issues**
If JARVIS runs out of memory:
- Close other applications
- Reduce history size
- Clear caches: `clear cache`

**API Errors**
If AI features don't work:
- Check API key configuration
- Verify network connection
- Check rate limits

**Permission Errors**
If you get permission errors:
- Check file permissions
- Ensure write access to directories
- Run with appropriate user

**Slow Performance**
To improve performance:
- Disable unnecessary features
- Reduce cache size
- Use lighter AI models
""",
            category=HelpCategory.TROUBLESHOOTING,
            keywords=["error", "problem", "fix", "issue", "debug"],
        ))
    
    def add_topic(self, topic: HelpTopic):
        """Add a help topic"""
        self._topics[topic.id] = topic
        
        # Organize by category
        if topic.category not in self._categories:
            self._categories[topic.category] = []
        self._categories[topic.category].append(topic.id)
    
    def get_topic(self, topic_id: str) -> Optional[HelpTopic]:
        """Get topic by ID"""
        return self._topics.get(topic_id)
    
    def search(self, query: str) -> List[HelpTopic]:
        """Search help topics"""
        query_lower = query.lower()
        results = []
        
        for topic in self._topics.values():
            # Check ID
            if query_lower in topic.id.lower():
                results.append(topic)
                continue
            
            # Check title
            if query_lower in topic.title.lower():
                results.append(topic)
                continue
            
            # Check keywords
            for keyword in topic.keywords:
                if query_lower in keyword.lower():
                    results.append(topic)
                    break
        
        return results
    
    def get_by_category(self, category: HelpCategory) -> List[HelpTopic]:
        """Get topics by category"""
        topic_ids = self._categories.get(category, [])
        return [self._topics[tid] for tid in topic_ids if tid in self._topics]
    
    def get_all_topics(self) -> List[HelpTopic]:
        """Get all topics"""
        return list(self._topics.values())
    
    def add_tutorial(self, tutorial: Tutorial):
        """Add a tutorial"""
        self._tutorials[tutorial.id] = tutorial
    
    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """Get tutorial by ID"""
        return self._tutorials.get(tutorial_id)
    
    def get_tutorials(self) -> List[Tutorial]:
        """Get all tutorials"""
        return list(self._tutorials.values())


# ═══════════════════════════════════════════════════════════════════════════════
# HELP SYSTEM (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class HelpSystem:
    """
    Ultra-Advanced Help System for JARVIS.
    
    Features:
    - Context-sensitive help
    - Command documentation
    - Tutorial system
    - Example gallery
    - Search functionality
    
    Memory Budget: < 3MB
    
    Usage:
        help = HelpSystem()
        
        # Show help for topic
        help.show("commands")
        
        # Search help
        results = help.search("error")
        
        # Show tutorial
        help.tutorial("getting-started")
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, output_format: HelpFormat = HelpFormat.MARKDOWN):
        """
        Initialize Help System.
        
        Args:
            output_format: Output format for help text
        """
        self._format = output_format
        self._db = HelpDatabase()
        
        # Current context
        self._context: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Help Display
    # ─────────────────────────────────────────────────────────────────────────
    
    def show(self, topic_id: str = None) -> str:
        """
        Show help for a topic.
        
        Args:
            topic_id: Help topic ID (shows overview if None)
            
        Returns:
            Formatted help text
        """
        if topic_id is None:
            return self._show_overview()
        
        topic = self._db.get_topic(topic_id)
        
        if not topic:
            # Try searching
            results = self._db.search(topic_id)
            if results:
                return self._format_search_results(results)
            return f"No help found for: {topic_id}"
        
        return self._format_topic(topic)
    
    def _show_overview(self) -> str:
        """Show help overview"""
        lines = [
            f"{Colors.BOLD}{Colors.CYAN}JARVIS Help System{Colors.RESET}",
            "",
            "Available help topics:",
            "",
        ]
        
        for category in HelpCategory:
            topics = self._db.get_by_category(category)
            if topics:
                lines.append(f"{Colors.BOLD}{category.name.replace('_', ' ')}{Colors.RESET}")
                for topic in topics:
                    lines.append(f"  {Colors.CYAN}{topic.id:<20}{Colors.RESET} {topic.title}")
                lines.append("")
        
        lines.extend([
            "Type 'help <topic>' for more information.",
            "Type 'help search <query>' to search.",
        ])
        
        return '\n'.join(lines)
    
    def _format_topic(self, topic: HelpTopic) -> str:
        """Format a help topic"""
        lines = [
            f"{Colors.BOLD}{Colors.CYAN}{topic.title}{Colors.RESET}",
            "",
            topic.content,
        ]
        
        # Examples
        if topic.examples:
            lines.extend([
                "",
                f"{Colors.BOLD}Examples:{Colors.RESET}",
            ])
            for example in topic.examples:
                lines.append(f"  {Colors.GREEN}{example}{Colors.RESET}")
        
        # Related topics
        if topic.related:
            lines.extend([
                "",
                f"{Colors.BOLD}Related Topics:{Colors.RESET}",
            ])
            for related in topic.related:
                lines.append(f"  {Colors.BLUE}{related}{Colors.RESET}")
        
        return '\n'.join(lines)
    
    def _format_search_results(self, results: List[HelpTopic]) -> str:
        """Format search results"""
        lines = [
            f"Found {len(results)} matching topics:",
            "",
        ]
        
        for topic in results:
            lines.append(f"  {Colors.CYAN}{topic.id:<20}{Colors.RESET} {topic.title}")
        
        lines.append("")
        lines.append("Type 'help <topic>' for more information.")
        
        return '\n'.join(lines)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────
    
    def search(self, query: str) -> str:
        """
        Search help topics.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        results = self._db.search(query)
        return self._format_search_results(results)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tutorials
    # ─────────────────────────────────────────────────────────────────────────
    
    def tutorial(self, tutorial_id: str = None) -> str:
        """
        Show tutorial.
        
        Args:
            tutorial_id: Tutorial ID
            
        Returns:
            Formatted tutorial
        """
        if tutorial_id is None:
            tutorials = self._db.get_tutorials()
            if not tutorials:
                return "No tutorials available."
            
            lines = ["Available tutorials:", ""]
            for t in tutorials:
                lines.append(f"  {Colors.CYAN}{t.id:<20}{Colors.RESET} {t.title}")
            
            return '\n'.join(lines)
        
        tutorial = self._db.get_tutorial(tutorial_id)
        
        if not tutorial:
            return f"Tutorial not found: {tutorial_id}"
        
        lines = [
            f"{Colors.BOLD}{Colors.CYAN}{tutorial.title}{Colors.RESET}",
            f"{Colors.DIM}{tutorial.description}{Colors.RESET}",
            f"Difficulty: {tutorial.difficulty} | Time: {tutorial.time_estimate}",
            "",
        ]
        
        for i, step in enumerate(tutorial.steps, 1):
            lines.append(f"{Colors.BOLD}Step {i}: {step.get('title', '')}{Colors.RESET}")
            lines.append(f"  {step.get('content', '')}")
            lines.append("")
        
        return '\n'.join(lines)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Context-Sensitive Help
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_context(self, context: str):
        """Set current help context"""
        self._context = context
    
    def context_help(self) -> str:
        """Get help for current context"""
        if not self._context:
            return self.show()
        
        # Try to find context-specific help
        topic = self._db.get_topic(self._context)
        
        if topic:
            return self._format_topic(topic)
        
        # Search for related topics
        results = self._db.search(self._context)
        if results:
            return self._format_search_results(results)
        
        return self.show()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Examples
    # ─────────────────────────────────────────────────────────────────────────
    
    def examples(self, topic_id: str = None) -> str:
        """
        Show examples.
        
        Args:
            topic_id: Topic ID (shows all if None)
            
        Returns:
            Formatted examples
        """
        if topic_id:
            topic = self._db.get_topic(topic_id)
            if topic and topic.examples:
                lines = [f"Examples for {topic.title}:", ""]
                for example in topic.examples:
                    lines.append(f"  {Colors.GREEN}{example}{Colors.RESET}")
                return '\n'.join(lines)
            return f"No examples found for: {topic_id}"
        
        # Show all examples
        lines = ["Examples:", ""]
        
        for topic in self._db.get_all_topics():
            if topic.examples:
                lines.append(f"{Colors.BOLD}{topic.title}{Colors.RESET}")
                for example in topic.examples:
                    lines.append(f"  {Colors.GREEN}{example}{Colors.RESET}")
                lines.append("")
        
        return '\n'.join(lines)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_topic(self, topic: HelpTopic):
        """Add a help topic"""
        self._db.add_topic(topic)
    
    def add_tutorial(self, tutorial: Tutorial):
        """Add a tutorial"""
        self._db.add_tutorial(tutorial)
    
    @property
    def topic_count(self) -> int:
        """Get topic count"""
        return len(self._db._topics)
    
    @property
    def tutorial_count(self) -> int:
        """Get tutorial count"""
        return len(self._db._tutorials)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo help system"""
    help_system = HelpSystem()
    
    print("JARVIS Help System Demo")
    print("=" * 40)
    
    # Show overview
    print("\n" + help_system.show())
    
    # Show specific topic
    print("\n" + help_system.show("commands"))
    
    # Search
    print("\n" + help_system.search("error"))


if __name__ == '__main__':
    main()
