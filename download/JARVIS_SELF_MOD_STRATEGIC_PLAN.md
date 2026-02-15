# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║           JARVIS v14 Ultimate - SELF-MODIFICATION INTEGRATION                ║
# ║                     DEEPEST STRATEGIC PLAN                                   ║
# ║                                                                              ║
# ║                    Goal: Make JARVIS TRULY Self-Modifying                    ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 0: DEEP ARCHITECTURE ANALYSIS
## ═══════════════════════════════════════════════════════════════════════════════

### Current Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JARVIS ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐  │
│   │   main.py   │────▶│  JARVIS     │────▶│      AI Client              │  │
│   │  (Entry)    │     │   Class     │     │  (OpenRouterClient)         │  │
│   └─────────────┘     └──────┬──────┘     │                             │  │
│                              │            │  chat() ──▶ TEXT ONLY ❌     │  │
│                              │            └─────────────────────────────┘  │
│                              │                                              │
│        ┌─────────────────────┼─────────────────────┐                      │
│        │                     │                     │                       │
│        ▼                     ▼                     ▼                       │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                  │
│   │  Memory  │         │Security  │         │Self-Mod  │                  │
│   │  System  │         │  System  │         │  Engine  │                  │
│   ├──────────┤         ├──────────┤         ├──────────┤                  │
│   │Context   │         │Auth      │         │Code      │ ✅ EXISTS        │
│   │Manager   │         │Encrypt   │         │Analyzer  │   BUT            │
│   │ChatStore │         │Sandbox   │         │Backup    │   NOT            │
│   │Optimizer │         │Audit     │         │Manager   │   CONNECTED!     │
│   └──────────┘         └──────────┘         │Validator │                  │
│                                             │Learning  │                  │
│                                             │  Engine  │                  │
│                                             └──────────┘                  │
│                                                                             │
│   PROBLEM: AI ────────▶ Text Response (DOESN'T USE SELF-MOD ENGINE!)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gap Analysis - What's Missing

| Component | Exists | Connected | Working | Gap |
|-----------|--------|-----------|---------|-----|
| CodeAnalyzer | ✅ | ❌ | ❌ | No bridge to AI |
| BackupManager | ✅ | ❌ | ❌ | Not called by AI |
| CodeValidator | ✅ | ❌ | ❌ | Not used |
| LearningEngine | ✅ | ❌ | ❌ | No feedback loop |
| File Read by AI | ❌ | ❌ | ❌ | Need implementation |
| File Write by AI | ❌ | ❌ | ❌ | Need implementation |
| Command Protocol | ❌ | ❌ | ❌ | Need implementation |
| AI→Engine Bridge | ❌ | ❌ | ❌ | **CRITICAL MISSING PIECE** |

### Data Flow Analysis

**CURRENT (BROKEN):**
```
User: "modify main.py to add feature X"
    ↓
AI Client: Sends to OpenRouter
    ↓
AI Response: "Here's how you could modify..." (TEXT ONLY!)
    ↓
Display: Shows text to user
    ↓
Result: NOTHING ACTUALLY HAPPENS ❌
```

**REQUIRED (FIXED):**
```
User: "modify main.py to add feature X"
    ↓
AI Bridge: Read current main.py content
    ↓
AI System Prompt: Includes file content + modification tools
    ↓
AI Response: "[MODIFY:main.py]\n```python\n...new code...```"
    ↓
Command Parser: Extract modification command
    ↓
BackupManager: Create backup
    ↓
CodeValidator: Validate changes
    ↓
FileWriter: Apply changes
    ↓
Result: FILE ACTUALLY MODIFIED ✅
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 1: INTEGRATION ARCHITECTURE
## ═══════════════════════════════════════════════════════════════════════════════

### New Architecture After Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JARVIS SELF-MOD ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SELF-MODIFICATION BRIDGE (NEW)                   │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │  │ File         │  │ Command      │  │ System Prompt            │  │   │
│   │  │ Operations   │  │ Parser       │  │ Generator                │  │   │
│   │  │ - Read       │  │ - Extract    │  │ - Project Context        │  │   │
│   │  │ - Write      │  │ - Validate   │  │ - Tool Definitions       │  │   │
│   │  │ - Create     │  │ - Execute    │  │ - Safety Rules           │  │   │
│   │  │ - Delete     │  │              │  │                          │  │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐  │
│   │   main.py   │────▶│    JARVIS    │────▶│      AI Client              │  │
│   │  (Entry)    │     │    Class     │     │  (OpenRouterClient)         │  │
│   └─────────────┘     └──────┬───────┘     │                             │  │
│                              │              │  chat() ──▶ MODIFIED PROMPT │  │
│                              │              │           ──▶ PARSED CMDS   │  │
│                              │              └─────────────────────────────┘  │
│                              │                                              │
│        ┌─────────────────────┼─────────────────────┐                       │
│        │                     │                     │                        │
│        ▼                     ▼                     ▼                        │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                   │
│   │  Memory  │         │ Security │         │ Self-Mod │                   │
│   │  System  │         │  System  │         │  Engine  │                   │
│   ├──────────┤         ├──────────┤         ├──────────┤                   │
│   │Context   │◀───────▶│Sandbox   │◀───────▶│Code      │                   │
│   │Manager   │         │Execute   │         │Analyzer  │                   │
│   │ChatStore │         │Validate  │         │Backup    │                   │
│   │Optimizer │         │Audit     │         │Validator │                   │
│   └──────────┘         └──────────┘         │Learning  │                   │
│                                             │  Engine  │                   │
│                                             └──────────┘                   │
│                                                                             │
│   RESULT: AI ────────▶ ACTIONS (FILES MODIFIED, BACKUPS CREATED) ✅        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 2: FILE STRUCTURE
## ═══════════════════════════════════════════════════════════════════════════════

### New Files to Create

```
/home/z/my-project/
├── main.py                          # MODIFY - Add bridge integration
├── core/
│   └── self_mod/
│       ├── __init__.py              # EXISTS
│       ├── code_analyzer.py         # EXISTS
│       ├── safe_modifier.py         # EXISTS
│       ├── backup_manager.py        # EXISTS
│       ├── improvement_engine.py    # EXISTS
│       │
│       ├── bridge.py                # NEW - Main integration bridge
│       ├── file_ops.py              # NEW - Safe file operations
│       ├── command_parser.py        # NEW - Parse AI commands
│       ├── system_prompts.py        # NEW - AI prompts with tools
│       └── safety_checker.py        # NEW - Pre-modification safety
│
└── tests/
    └── test_self_mod_integration.py # NEW - Integration tests
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 3: COMMAND PROTOCOL DESIGN
## ═══════════════════════════════════════════════════════════════════════════════

### AI Command Format (JARVIS Tool Protocol)

```markdown
## JARVIS Self-Modification Commands

### 1. READ - Read File Content
[READ:path/to/file.py]

### 2. MODIFY - Modify Existing File
[MODIFY:path/to/file.py]
```python
# New/modified code here
def new_function():
    pass
```

### 3. CREATE - Create New File
[CREATE:path/to/new_file.py]
```python
# New file content
```

### 4. DELETE - Delete File
[DELETE:path/to/file.py]

### 5. ANALYZE - Run Code Analysis
[ANALYZE:path/to/file.py]

### 6. LIST - List Directory
[LIST:path/to/directory]

### 7. BACKUP - Create Manual Backup
[BACKUP:path/to/file.py]

### 8. ROLLBACK - Restore from Backup
[ROLLBACK:backup_id]

### 9. EXECUTE - Run Code in Sandbox
[EXECUTE]
```python
# Code to execute safely
```

### 10. SEARCH - Search Codebase
[SEARCH:pattern]
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 4: SAFETY MECHANISMS
## ═══════════════════════════════════════════════════════════════════════════════

### 4.1 Pre-Modification Safety Checks

```python
SAFETY_CHECKS = {
    # 1. File Protection
    'protected_files': [
        '.env',           # API keys
        'credentials.*',  # Credentials
        '.git/',          # Git directory
    ],
    
    # 2. Code Validation
    'code_checks': [
        'syntax_valid',       # Must parse as valid Python
        'no_infinite_loops',  # Detect potential infinite loops
        'no_system_calls',    # No os.system, subprocess in sandbox
        'imports_allowed',    # Only whitelisted imports
    ],
    
    # 3. Backup Requirements
    'backup_rules': {
        'always_backup': True,           # Create backup before any modification
        'max_backups': 100,              # Keep last 100 backups
        'backup_age_days': 30,           # Keep backups for 30 days
    },
    
    # 4. Modification Limits
    'limits': {
        'max_file_size': 5 * 1024 * 1024,    # 5MB max file
        'max_changes_per_session': 20,        # Max modifications per session
        'max_lines_changed': 500,             # Max lines per modification
    },
    
    # 5. Rollback Triggers
    'auto_rollback': [
        'syntax_error_after_mod',     # Code doesn't parse
        'import_error_on_restart',    # Missing imports
        'runtime_error_on_test',      # Test fails after mod
        'user_requested',             # User says "rollback"
    ],
}
```

### 4.2 Safety Validation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAFETY VALIDATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Modification Request                                                      │
│         │                                                                   │
│         ▼                                                                   │
│   ┌──────────────────┐                                                     │
│   │ Is file          │──── NO ──▶ BLOCK: "Protected file"                  │
│   │ protected?       │                                                     │
│   └────────┬─────────┘                                                     │
│            │ YES                                                            │
│            ▼                                                                │
│   ┌──────────────────┐                                                     │
│   │ Create backup    │──── FAIL ▶ BLOCK: "Backup failed, aborting"         │
│   │ of original      │                                                     │
│   └────────┬─────────┘                                                     │
│            │ SUCCESS                                                        │
│            ▼                                                                │
│   ┌──────────────────┐                                                     │
│   │ Validate new     │──── FAIL ▶ RESTORE from backup                      │
│   │ code syntax      │                                                     │
│   └────────┬─────────┘                                                     │
│            │ VALID                                                          │
│            ▼                                                                │
│   ┌──────────────────┐                                                     │
│   │ Check for        │──── FOUND ▶ WARN: "Dangerous pattern detected"      │
│   │ dangerous        │           │                                         │
│   │ patterns         │           ▼                                         │
│   └────────┬─────────┘           ┌──────────────────┐                      │
│            │ SAFE                 │ User confirms?   │                      │
│            ▼                      │ [Y/n]            │                      │
│   ┌──────────────────┐           └────────┬─────────┘                      │
│   │ Apply            │                    │                                 │
│   │ modification     │                    │ NO ──▶ RESTORE from backup     │
│   └────────┬─────────┘                    │ YES                              │
│            │                              ▼                                  │
│            │                    ┌──────────────────┐                        │
│            │                    │ Apply anyway     │                        │
│            │                    └────────┬─────────┘                        │
│            │                             │                                   │
│            ▼                             ▼                                   │
│   ┌──────────────────┐                                                     │
│   │ Test modified    │──── FAIL ▶ RESTORE from backup + LOG ERROR          │
│   │ code (optional)  │                                                     │
│   └────────┬─────────┘                                                     │
│            │ PASS                                                           │
│            ▼                                                                │
│   ┌──────────────────┐                                                     │
│   │ Log to Learning  │                                                     │
│   │ Engine           │                                                     │
│   └────────┬─────────┘                                                     │
│            │                                                                │
│            ▼                                                                │
│   ✅ MODIFICATION COMPLETE                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 5: IMPLEMENTATION DETAILS
## ═══════════════════════════════════════════════════════════════════════════════

### 5.1 File: core/self_mod/bridge.py (CRITICAL)

```python
#!/usr/bin/env python3
"""
JARVIS Self-Modification Bridge
===============================

This is the CENTRAL INTEGRATION POINT that connects:
- AI responses → File operations
- AI commands → Self-mod engine
- User requests → Actual modifications

WITHOUT THIS FILE: JARVIS is just a chatbot
WITH THIS FILE: JARVIS can modify itself
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ModificationResult:
    """Result of a modification operation"""
    success: bool
    operation: str
    target: str
    backup_id: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SelfModificationBridge:
    """
    Bridge between AI responses and actual file modifications.
    
    This class:
    1. Generates system prompts with tool definitions
    2. Parses AI responses for commands
    3. Executes file operations safely
    4. Integrates with backup and validation systems
    """
    
    # Command patterns for parsing AI responses
    COMMAND_PATTERNS = {
        'READ': r'\[READ:([^\]]+)\]',
        'MODIFY': r'\[MODIFY:([^\]]+)\]',
        'CREATE': r'\[CREATE:([^\]]+)\]',
        'DELETE': r'\[DELETE:([^\]]+)\]',
        'ANALYZE': r'\[ANALYZE:([^\]]+)\]',
        'LIST': r'\[LIST:([^\]]+)\]',
        'BACKUP': r'\[BACKUP:([^\]]+)\]',
        'ROLLBACK': r'\[ROLLBACK:([^\]]+)\]',
        'EXECUTE': r'\[EXECUTE\]',
        'SEARCH': r'\[SEARCH:([^\]]+)\]',
    }
    
    # Protected files that cannot be modified
    PROTECTED_FILES = {
        '.env',
        'credentials.json',
        'secrets.json',
        '.git',
    }
    
    # Protected patterns
    PROTECTED_PATTERNS = [
        r'\.env(\..*)?$',
        r'credentials?\..*$',
        r'secrets?\..*$',
        r'\.git/.*$',
        r'private.*\.pem$',
    ]
    
    def __init__(self, jarvis_instance):
        """
        Initialize bridge with JARVIS instance.
        
        Args:
            jarvis_instance: Main JARVIS class instance
        """
        self.jarvis = jarvis_instance
        self.project_root = Path(__file__).parent.parent.parent
        
        # Get components from JARVIS
        self.code_analyzer = getattr(jarvis_instance, 'code_analyzer', None)
        self.backup_manager = getattr(jarvis_instance, 'backup_manager', None)
        self.safe_modifier = getattr(jarvis_instance, 'safe_modifier', None)
        self.improvement_engine = getattr(jarvis_instance, 'improvement_engine', None)
        self.ai_client = getattr(jarvis_instance, 'ai_client', None)
        self.sandbox = getattr(jarvis_instance, 'sandbox', None)
        
        # Initialize project context
        self._project_context = None
        self._last_context_update = 0
        self._context_cache_ttl = 300  # 5 minutes
    
    def get_system_prompt(self) -> str:
        """
        Generate system prompt with self-modification tools.
        
        This prompt tells the AI:
        1. It CAN modify files
        2. HOW to modify files (command syntax)
        3. What SAFETY rules to follow
        """
        project_context = self._get_project_context()
        
        return '''You are JARVIS, a SELF-MODIFYING AI assistant with the ability to READ, MODIFY, CREATE, and DELETE files in your own codebase.

## 🔧 AVAILABLE TOOLS

You have access to the following tools. Use them by including the command syntax in your response:

### 1. READ - Read file content
[READ:path/to/file.py]
Returns the complete content of the specified file.

### 2. MODIFY - Modify an existing file
[MODIFY:path/to/file.py]
```python
# Complete new content of the file
# This REPLACES the entire file
```

### 3. CREATE - Create a new file
[CREATE:path/to/new_file.py]
```python
# Content for the new file
```

### 4. DELETE - Delete a file
[DELETE:path/to/file.py]

### 5. ANALYZE - Analyze code for issues
[ANALYZE:path/to/file.py]

### 6. LIST - List directory contents
[LIST:path/to/directory]

### 7. BACKUP - Create manual backup
[BACKUP:path/to/file.py]

### 8. ROLLBACK - Restore from backup
[ROLLBACK:backup_id]

### 9. SEARCH - Search codebase
[SEARCH:search_pattern]

## ⚠️ SAFETY RULES

1. ALWAYS read a file before modifying it
2. Create backups are AUTOMATIC before any modification
3. NEVER modify: .env, credentials, .git directory
4. Test modifications when possible
5. Explain what you're changing and WHY
6. One modification at a time for safety

## 📁 YOUR PROJECT STRUCTURE

''' + project_context + '''

## 💡 BEST PRACTICES

1. When asked to modify code:
   - First [READ:file] to see current content
   - Analyze what needs to change
   - [MODIFY:file] with complete new content
   - Explain the changes

2. When asked to add a feature:
   - Identify the best file to modify
   - Read existing code first
   - Make minimal, focused changes
   - Add proper error handling

3. When asked to debug:
   - Read the relevant files
   - [ANALYZE:file] to find issues
   - Propose specific fixes
   - Apply fixes with MODIFY

REMEMBER: You are not just suggesting - you are actually DOING the modifications. Use the tools above to make real changes to files.
'''
    
    def _get_project_context(self) -> str:
        """Build context about project structure"""
        # Check cache
        now = time.time()
        if self._project_context and (now - self._last_context_update) < self._context_cache_ttl:
            return self._project_context
        
        context_lines = ["Project root: " + str(self.project_root), ""]
        
        # Walk directory tree
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules']
            
            level = root.replace(str(self.project_root), '').count(os.sep)
            if level > 3:  # Limit depth
                continue
            
            indent = '  ' * level
            rel_path = os.path.relpath(root, self.project_root)
            if rel_path != '.':
                context_lines.append(f"{indent}📁 {os.path.basename(root)}/")
            
            subindent = '  ' * (level + 1)
            for file in sorted(files)[:15]:  # Limit files per directory
                if file.endswith('.py') or file.endswith('.json') or file.endswith('.md'):
                    context_lines.append(f"{subindent}📄 {file}")
        
        self._project_context = '\n'.join(context_lines)
        self._last_context_update = now
        return self._project_context
    
    def parse_commands(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse AI response for commands.
        
        Args:
            response: AI's response text
            
        Returns:
            List of parsed commands with targets and content
        """
        commands = []
        
        for cmd_name, pattern in self.COMMAND_PATTERNS.items():
            matches = re.findall(pattern, response)
            for match in matches:
                command = {
                    'command': cmd_name,
                    'target': match if isinstance(match, str) else match[0] if match else '',
                }
                
                # Extract code block for MODIFY/CREATE/EXECUTE
                if cmd_name in ('MODIFY', 'CREATE', 'EXECUTE'):
                    # Find code block after the command
                    cmd_pattern = re.escape(f'[{cmd_name}:{command["target"]}]') if cmd_name != 'EXECUTE' else r'\[EXECUTE\]'
                    code_match = re.search(
                        rf'{cmd_pattern}\s*```(?:python)?\s*\n(.*?)\n```',
                        response,
                        re.DOTALL
                    )
                    if code_match:
                        command['code'] = code_match.group(1)
                    else:
                        command['error'] = 'No code block found'
                
                commands.append(command)
        
        return commands
    
    def execute_command(self, command: Dict[str, Any], response: str) -> ModificationResult:
        """
        Execute a parsed command.
        
        Args:
            command: Parsed command dict
            response: Full AI response for context
            
        Returns:
            ModificationResult with outcome
        """
        cmd_type = command.get('command')
        target = command.get('target', '')
        
        # Resolve path
        if target.startswith('/'):
            file_path = Path(target)
        else:
            file_path = self.project_root / target
        
        # Check if file is protected
        if self._is_protected(file_path):
            return ModificationResult(
                success=False,
                operation=cmd_type,
                target=str(file_path),
                error=f"File is protected: {file_path}"
            )
        
        # Execute based on command type
        if cmd_type == 'READ':
            return self._execute_read(file_path)
        elif cmd_type == 'MODIFY':
            return self._execute_modify(file_path, command.get('code', ''))
        elif cmd_type == 'CREATE':
            return self._execute_create(file_path, command.get('code', ''))
        elif cmd_type == 'DELETE':
            return self._execute_delete(file_path)
        elif cmd_type == 'ANALYZE':
            return self._execute_analyze(file_path)
        elif cmd_type == 'LIST':
            return self._execute_list(file_path)
        elif cmd_type == 'BACKUP':
            return self._execute_backup(file_path)
        elif cmd_type == 'ROLLBACK':
            return self._execute_rollback(target)
        elif cmd_type == 'SEARCH':
            return self._execute_search(target)
        else:
            return ModificationResult(
                success=False,
                operation=cmd_type,
                target=target,
                error=f"Unknown command: {cmd_type}"
            )
    
    def _is_protected(self, file_path: Path) -> bool:
        """Check if file is protected from modification"""
        name = file_path.name
        path_str = str(file_path)
        
        if name in self.PROTECTED_FILES:
            return True
        
        for pattern in self.PROTECTED_PATTERNS:
            if re.search(pattern, path_str):
                return True
        
        return False
    
    def _execute_read(self, file_path: Path) -> ModificationResult:
        """Execute READ command"""
        if not file_path.exists():
            return ModificationResult(
                success=False,
                operation='READ',
                target=str(file_path),
                error=f"File not found: {file_path}"
            )
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return ModificationResult(
                success=True,
                operation='READ',
                target=str(file_path),
                details={
                    'content': content,
                    'lines': len(content.splitlines()),
                    'size': len(content),
                }
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='READ',
                target=str(file_path),
                error=str(e)
            )
    
    def _execute_modify(self, file_path: Path, new_code: str) -> ModificationResult:
        """Execute MODIFY command with backup"""
        if not file_path.exists():
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"File not found: {file_path}"
            )
        
        if not new_code:
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error="No code provided for modification"
            )
        
        # Validate syntax
        try:
            import ast
            ast.parse(new_code)
        except SyntaxError as e:
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"Syntax error in new code: {e}"
            )
        
        # Create backup
        backup_id = None
        if self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(
                    str(file_path),
                    description="Before AI modification"
                )
            except Exception as e:
                return ModificationResult(
                    success=False,
                    operation='MODIFY',
                    target=str(file_path),
                    error=f"Backup failed: {e}"
                )
        
        # Read original for comparison
        try:
            original = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"Cannot read file: {e}",
                backup_id=backup_id
            )
        
        # Write new content
        try:
            file_path.write_text(new_code, encoding='utf-8')
            
            # Log to improvement engine
            if self.improvement_engine:
                from core.self_mod.improvement_engine import ModificationOutcome, OutcomeType
                outcome = ModificationOutcome(
                    modification_id=backup_id or f"mod_{time.time()}",
                    outcome_type=OutcomeType.SUCCESS,
                    timestamp=time.time()
                )
                self.improvement_engine.record_outcome(outcome)
            
            return ModificationResult(
                success=True,
                operation='MODIFY',
                target=str(file_path),
                backup_id=backup_id,
                details={
                    'original_lines': len(original.splitlines()),
                    'new_lines': len(new_code.splitlines()),
                }
            )
        except Exception as e:
            # Attempt rollback
            if backup_id and self.backup_manager:
                try:
                    self.backup_manager.rollback(backup_id)
                except Exception:
                    pass
            
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"Write failed: {e}",
                backup_id=backup_id
            )
    
    def _execute_create(self, file_path: Path, code: str) -> ModificationResult:
        """Execute CREATE command"""
        if not code:
            return ModificationResult(
                success=False,
                operation='CREATE',
                target=str(file_path),
                error="No code provided for new file"
            )
        
        # Validate syntax
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return ModificationResult(
                success=False,
                operation='CREATE',
                target=str(file_path),
                error=f"Syntax error: {e}"
            )
        
        # Create parent directories
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='CREATE',
                target=str(file_path),
                error=f"Cannot create directory: {e}"
            )
        
        # Write file
        try:
            file_path.write_text(code, encoding='utf-8')
            return ModificationResult(
                success=True,
                operation='CREATE',
                target=str(file_path),
                details={'lines': len(code.splitlines())}
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='CREATE',
                target=str(file_path),
                error=str(e)
            )
    
    def _execute_delete(self, file_path: Path) -> ModificationResult:
        """Execute DELETE command with backup"""
        if not file_path.exists():
            return ModificationResult(
                success=False,
                operation='DELETE',
                target=str(file_path),
                error=f"File not found: {file_path}"
            )
        
        # Create backup before deletion
        backup_id = None
        if self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(
                    str(file_path),
                    description="Before deletion"
                )
            except Exception:
                pass  # Continue even if backup fails
        
        # Delete file
        try:
            file_path.unlink()
            return ModificationResult(
                success=True,
                operation='DELETE',
                target=str(file_path),
                backup_id=backup_id
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='DELETE',
                target=str(file_path),
                error=str(e),
                backup_id=backup_id
            )
    
    def _execute_analyze(self, file_path: Path) -> ModificationResult:
        """Execute ANALYZE command"""
        if not file_path.exists():
            return ModificationResult(
                success=False,
                operation='ANALYZE',
                target=str(file_path),
                error=f"File not found: {file_path}"
            )
        
        if self.code_analyzer:
            try:
                analysis = self.code_analyzer.analyze_file(str(file_path))
                return ModificationResult(
                    success=True,
                    operation='ANALYZE',
                    target=str(file_path),
                    details={
                        'functions': len(analysis.functions),
                        'classes': len(analysis.classes),
                        'imports': len(analysis.imports),
                        'issues': len(analysis.issues),
                        'analysis': analysis,
                    }
                )
            except Exception as e:
                return ModificationResult(
                    success=False,
                    operation='ANALYZE',
                    target=str(file_path),
                    error=str(e)
                )
        else:
            # Basic analysis without CodeAnalyzer
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                return ModificationResult(
                    success=True,
                    operation='ANALYZE',
                    target=str(file_path),
                    details={
                        'lines': len(lines),
                        'characters': len(content),
                        'size_bytes': file_path.stat().st_size,
                    }
                )
            except Exception as e:
                return ModificationResult(
                    success=False,
                    operation='ANALYZE',
                    target=str(file_path),
                    error=str(e)
                )
    
    def _execute_list(self, dir_path: Path) -> ModificationResult:
        """Execute LIST command"""
        if not dir_path.exists():
            dir_path = self.project_root / dir_path
        
        if not dir_path.exists():
            return ModificationResult(
                success=False,
                operation='LIST',
                target=str(dir_path),
                error=f"Directory not found: {dir_path}"
            )
        
        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    items.append({'name': item.name, 'type': 'directory'})
                else:
                    items.append({
                        'name': item.name,
                        'type': 'file',
                        'size': item.stat().st_size
                    })
            
            return ModificationResult(
                success=True,
                operation='LIST',
                target=str(dir_path),
                details={'items': items, 'count': len(items)}
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='LIST',
                target=str(dir_path),
                error=str(e)
            )
    
    def _execute_backup(self, file_path: Path) -> ModificationResult:
        """Execute BACKUP command"""
        if not file_path.exists():
            return ModificationResult(
                success=False,
                operation='BACKUP',
                target=str(file_path),
                error=f"File not found: {file_path}"
            )
        
        if not self.backup_manager:
            return ModificationResult(
                success=False,
                operation='BACKUP',
                target=str(file_path),
                error="Backup manager not available"
            )
        
        try:
            backup_id = self.backup_manager.create_backup(
                str(file_path),
                description="Manual backup"
            )
            return ModificationResult(
                success=True,
                operation='BACKUP',
                target=str(file_path),
                backup_id=backup_id
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='BACKUP',
                target=str(file_path),
                error=str(e)
            )
    
    def _execute_rollback(self, backup_id: str) -> ModificationResult:
        """Execute ROLLBACK command"""
        if not self.backup_manager:
            return ModificationResult(
                success=False,
                operation='ROLLBACK',
                target=backup_id,
                error="Backup manager not available"
            )
        
        try:
            result = self.backup_manager.rollback(backup_id)
            return ModificationResult(
                success=result.success,
                operation='ROLLBACK',
                target=backup_id,
                details={
                    'files_restored': result.files_restored,
                    'errors': result.errors,
                }
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='ROLLBACK',
                target=backup_id,
                error=str(e)
            )
    
    def _execute_search(self, pattern: str) -> ModificationResult:
        """Execute SEARCH command"""
        results = []
        
        try:
            for root, dirs, files in os.walk(self.project_root):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            if pattern.lower() in content.lower():
                                rel_path = file_path.relative_to(self.project_root)
                                results.append(str(rel_path))
                        except Exception:
                            pass
            
            return ModificationResult(
                success=True,
                operation='SEARCH',
                target=pattern,
                details={'results': results[:20], 'count': len(results)}
            )
        except Exception as e:
            return ModificationResult(
                success=False,
                operation='SEARCH',
                target=pattern,
                error=str(e)
            )
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        Process AI response and execute all commands.
        
        Args:
            response: AI's response text
            
        Returns:
            Dict with all execution results
        """
        commands = self.parse_commands(response)
        results = {
            'commands_found': len(commands),
            'executions': [],
            'successes': 0,
            'failures': 0,
            'modified_files': [],
        }
        
        for cmd in commands:
            result = self.execute_command(cmd, response)
            results['executions'].append({
                'command': cmd.get('command'),
                'target': cmd.get('target'),
                'success': result.success,
                'backup_id': result.backup_id,
                'error': result.error,
            })
            
            if result.success:
                results['successes'] += 1
                if cmd.get('command') in ('MODIFY', 'CREATE', 'DELETE'):
                    results['modified_files'].append(str(result.target))
            else:
                results['failures'] += 1
        
        return results
```

### 5.2 File: main.py Modifications

```python
# ADD TO IMPORTS SECTION (around line 115)

try:
    from core.self_mod.bridge import SelfModificationBridge, ModificationResult
    SELF_MOD_BRIDGE_AVAILABLE = True
except ImportError:
    SELF_MOD_BRIDGE_AVAILABLE = False


# ADD TO JARVIS.__init__() (after _init_self_mod)

# Self-modification bridge
self.mod_bridge = None
if SELF_MOD_BRIDGE_AVAILABLE and SELF_MOD_AVAILABLE:
    try:
        self.mod_bridge = SelfModificationBridge(self)
        if self.debug:
            print("[DEBUG] Self-modification bridge initialized")
    except Exception as e:
        print(f"Error initializing mod bridge: {e}")


# REPLACE _handle_ai_command() METHOD

def _handle_ai_command(self, command: str):
    """Handle AI conversation with self-modification capabilities"""
    if not self.ai_client:
        print(Colors.YELLOW + "AI client not available. Set OPENROUTER_API_KEY environment variable." + Colors.RESET)
        return
    
    try:
        print(Colors.CYAN + "Thinking..." + Colors.RESET, end='\r')
        
        # Get system prompt with self-modification tools
        if self.mod_bridge:
            system_prompt = self.mod_bridge.get_system_prompt()
        else:
            system_prompt = "You are JARVIS, a helpful AI assistant."
        
        # Get conversation context
        context_id = "default"
        
        # Create conversation in chat_storage if needed
        if self.chat_storage:
            existing_conv = self.chat_storage.get_conversation(context_id)
            if not existing_conv:
                try:
                    self.chat_storage.create_conversation(
                        conversation_id=context_id,
                        title="Default Chat",
                        system_prompt=system_prompt[:500]  # Truncate for storage
                    )
                except Exception:
                    pass
        
        # Create context in context_manager with full system prompt
        if self.context_manager:
            existing = self.context_manager.get_context(context_id)
            if not existing:
                self.context_manager.create_context(context_id)
            self.context_manager.set_system_prompt(context_id, system_prompt)
        
        # Send to AI with enhanced system prompt
        response = self.ai_client.chat(
            message=command,
            system=system_prompt,
            context_id=context_id
        )
        
        if response.success:
            # Clear "Thinking..." line
            print(" " * 40, end='\r')
            
            # Display response
            if self.output_formatter:
                formatted = self.output_formatter.markdown(response.content)
                print(formatted)
            else:
                print(response.content)
            
            # Process any self-modification commands
            if self.mod_bridge:
                mod_results = self.mod_bridge.process_response(response.content)
                
                if mod_results['commands_found'] > 0:
                    print()
                    print(Colors.CYAN + "═══ Self-Modification Results ═══" + Colors.RESET)
                    
                    for exec_result in mod_results['executions']:
                        cmd = exec_result['command']
                        target = exec_result['target']
                        
                        if exec_result['success']:
                            print(Colors.GREEN + f"✓ {cmd}: {target}" + Colors.RESET)
                            if exec_result.get('backup_id'):
                                print(f"  Backup: {exec_result['backup_id']}")
                        else:
                            print(Colors.RED + f"✗ {cmd}: {target}" + Colors.RESET)
                            if exec_result.get('error'):
                                print(f"  Error: {exec_result['error']}")
                    
                    if mod_results['modified_files']:
                        print()
                        print(Colors.YELLOW + "Modified files:" + Colors.RESET)
                        for f in mod_results['modified_files']:
                            print(f"  • {f}")
            
            # Store in chat history
            if self.chat_storage:
                self.chat_storage.add_message(
                    conversation_id=context_id,
                    role='user',
                    content=command
                )
                self.chat_storage.add_message(
                    conversation_id=context_id,
                    role='assistant',
                    content=response.content
                )
        else:
            print(Colors.RED + f"AI Error: {response.error or 'Unknown error'}" + Colors.RESET)
            
    except Exception as e:
        print(Colors.RED + f"AI error: {e}" + Colors.RESET)
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 6: TESTING STRATEGY
## ═══════════════════════════════════════════════════════════════════════════════

### Test Cases

```python
TEST_CASES = [
    # Test 1: Read file
    {
        'name': 'read_file',
        'input': 'read the main.py file',
        'expected_command': 'READ',
        'expected_target': 'main.py',
    },
    
    # Test 2: Modify file
    {
        'name': 'modify_file',
        'input': 'add a debug command to main.py',
        'expected_command': 'MODIFY',
        'expected_target': 'main.py',
    },
    
    # Test 3: Create file
    {
        'name': 'create_file',
        'input': 'create a new file utils.py with helper functions',
        'expected_command': 'CREATE',
    },
    
    # Test 4: Analyze code
    {
        'name': 'analyze_code',
        'input': 'analyze the code_analyzer.py for issues',
        'expected_command': 'ANALYZE',
    },
    
    # Test 5: Protected file
    {
        'name': 'protected_file',
        'input': 'read the .env file',
        'expected_success': False,
        'expected_error': 'protected',
    },
    
    # Test 6: Syntax error
    {
        'name': 'syntax_error',
        'code': 'def broken(:\n    pass',
        'expected_success': False,
        'expected_error': 'Syntax error',
    },
]
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 7: ERROR HANDLING MATRIX
## ═══════════════════════════════════════════════════════════════════════════════

| Error Type | Scenario | Recovery Action |
|------------|----------|-----------------|
| FileNotFound | READ non-existent file | Return error, suggest LIST |
| PermissionDenied | Modify protected file | Return error, explain why |
| SyntaxError | Invalid Python code | Reject modification, show error |
| BackupFailed | Backup creation error | Abort modification, log error |
| WriteFailed | Cannot write file | Attempt rollback, log error |
| ParseError | Cannot parse AI response | Continue without action |
| NetworkError | AI API timeout | Retry with fallback model |
| MemoryError | Large file operation | Abort, suggest chunking |
| RollbackFailed | Cannot restore backup | Log critical, manual intervention |

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 8: ROLLBACK PROCEDURES
## ═══════════════════════════════════════════════════════════ backups
    
    echo "3. Restore specific file:"
    echo "   python -c \"from core.self_mod.bridge import *; ...\""
    ;;
esac
```

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 9: IMPLEMENTATION TIMELINE
## ═══════════════════════════════════════════════════════════════════════════════

| Phase | Task | Estimated Time | Priority |
|-------|------|----------------|----------|
| 1 | Create bridge.py | 2 hours | CRITICAL |
| 2 | Modify main.py | 1 hour | CRITICAL |
| 3 | Test basic commands | 1 hour | HIGH |
| 4 | Add safety checks | 2 hours | HIGH |
| 5 | Integration testing | 2 hours | HIGH |
| 6 | Error handling | 1 hour | MEDIUM |
| 7 | Learning engine integration | 1 hour | MEDIUM |
| 8 | Documentation | 1 hour | LOW |
| **TOTAL** | | **~11 hours** | |

## ═══════════════════════════════════════════════════════════════════════════════
## PHASE 10: SUCCESS CRITERIA
## ═══════════════════════════════════════════════════════════════════════════════

### Must Have (MVP)
- [ ] AI can READ files and return content
- [ ] AI can MODIFY files with automatic backup
- [ ] AI can CREATE new files
- [ ] AI can DELETE files with backup
- [ ] Protected files cannot be modified
- [ ] All modifications create backups
- [ ] Syntax validation before write
- [ ] Error messages are clear

### Should Have
- [ ] AI can ANALYZE code for issues
- [ ] AI can SEARCH codebase
- [ ] Learning engine records outcomes
- [ ] Automatic rollback on syntax error
- [ ] Modification history tracking

### Nice to Have
- [ ] AI suggests improvements proactively
- [ ] Code diff visualization
- [ ] Multi-file modifications
- [ ] Test execution in sandbox
- [ ] Performance metrics tracking

## ═══════════════════════════════════════════════════════════════════════════════
## FINAL NOTES
## ═══════════════════════════════════════════════════════════════════════════════

This plan provides a complete roadmap to make JARVIS truly self-modifying. The key insight is that all the pieces already exist - they just need to be connected through a bridge layer.

The bridge.py file is the MISSING LINK that transforms JARVIS from a chatbot into a self-modifying AI.

**Without Bridge:** AI talks, nothing happens
**With Bridge:** AI talks, files get modified, backups created, learning happens

Implementation should start with bridge.py creation, then main.py modification, followed by thorough testing.

---

*End of Strategic Plan*
