#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Modification Bridge
==============================================

THE CRITICAL MISSING PIECE that connects:
- AI responses → File operations
- AI commands → Self-mod engine  
- User requests → Actual modifications

WITHOUT THIS FILE: JARVIS is just a chatbot
WITH THIS FILE: JARVIS can modify itself!

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux
"""

import os
import re
import ast
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    
    # Protected files that cannot be modified (CRITICAL SYSTEM FILES)
    # CRITICAL FIX: Expanded from forensic analysis recommendations
    PROTECTED_FILES = {
        # Environment and secrets
        '.env',
        '.env.local',
        '.env.production',
        'credentials.json',
        'secrets.json',
        '.gitignore',
        
        # SSH keys
        'id_rsa',
        'id_rsa.pub',
        'id_ed25519',
        'id_ed25519.pub',
        
        # Git
        '.git',
        
        # CRITICAL: Self-modification system files (prevent corruption)
        'core/self_mod/safe_modifier.py',  # Can disable its own safety
        'core/self_mod/backup_manager.py', # Can corrupt backup system
        
        # CRITICAL: Security system files
        'security/sandbox.py',  # Can disable sandboxing
        
        # CRITICAL: Entry point
        'main.py',  # Can modify entry point
        
        # System config
        '.bashrc',
        '.zshrc',
        'authorized_keys',
        'known_hosts',
    }
    
    # Protected patterns (regex)
    PROTECTED_PATTERNS = [
        r'\.env(\..*)?$',
        r'credentials?\..*$',
        r'secrets?\..*$',
        r'\.git/.*$',
        r'private.*\.pem$',
        r'private.*\.key$',
        r'\.ssh/.*$',
        r'.*\.pub$',
        # CRITICAL: Prevent modification of self-mod system
        r'core/self_mod/safe_modifier\.py$',
        r'core/self_mod/backup_manager\.py$',
        r'security/sandbox\.py$',
    ]
    
    # Maximum modification depth to prevent infinite loops
    MAX_MODIFICATION_DEPTH = 5
    # Cooldown between modifications of same file (seconds)
    MODIFICATION_COOLDOWN = 30
    
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
        
        # Initialize project context cache
        self._project_context = None
        self._last_context_update = 0
        self._context_cache_ttl = 300  # 5 minutes
        
        # CRITICAL FIX: Modification tracking (prevent infinite loops)
        self._modification_depth = 0
        self._recently_modified = {}  # {file_path: timestamp}
        self._modification_history = []  # List of recent modifications
        
        logger.info("SelfModificationBridge initialized with circular modification protection")
    
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
# This REPLACES the entire file content
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
3. NEVER modify: .env, credentials, .git directory, private keys
4. Test modifications when possible
5. Explain what you're changing and WHY
6. One modification at a time for safety
7. Validate syntax before writing code

## 📁 YOUR PROJECT STRUCTURE

''' + project_context + '''

## 💡 BEST PRACTICES

1. When asked to modify code:
   - First [READ:file] to see current content
   - Analyze what needs to change
   - [MODIFY:file] with complete new content
   - Explain the changes you made

2. When asked to add a feature:
   - Identify the best file to modify or create
   - Read existing code first
   - Make minimal, focused changes
   - Add proper error handling

3. When asked to debug:
   - Read the relevant files
   - [ANALYZE:file] to find issues
   - Propose specific fixes
   - Apply fixes with MODIFY

REMEMBER: You are not just suggesting - you are actually DOING the modifications. Use the tools above to make real changes to files. Execute the commands in your response.
'''
    
    def _get_project_context(self) -> str:
        """Build context about project structure"""
        # Check cache
        now = time.time()
        if self._project_context and (now - self._last_context_update) < self._context_cache_ttl:
            return self._project_context
        
        context_lines = ["Project root: " + str(self.project_root), ""]
        
        try:
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
        except Exception as e:
            logger.warning(f"Error building project context: {e}")
        
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
                    if cmd_name != 'EXECUTE':
                        cmd_pattern = re.escape(f'[{cmd_name}:{command["target"]}]')
                    else:
                        cmd_pattern = r'\[EXECUTE\]'
                    
                    code_match = re.search(
                        rf'{cmd_pattern}\s*```(?:python)?\s*\n(.*?)\n```',
                        response,
                        re.DOTALL
                    )
                    if code_match:
                        command['code'] = code_match.group(1)
                    else:
                        command['error'] = 'No code block found after command'
                
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
                error=f"File is protected and cannot be modified: {file_path}"
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
                error=f"Unknown command type: {cmd_type}"
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
                error=f"Error reading file: {e}"
            )
    
    def _execute_modify(self, file_path: Path, new_code: str) -> ModificationResult:
        """Execute MODIFY command with backup and circular modification protection"""
        import time
        
        # CRITICAL FIX: Circular modification detection
        file_key = str(file_path)
        current_time = time.time()
        
        # Check modification depth
        if self._modification_depth >= self.MAX_MODIFICATION_DEPTH:
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"Maximum modification depth ({self.MAX_MODIFICATION_DEPTH}) exceeded - possible circular modification detected"
            )
        
        # Check cooldown for recently modified files
        if file_key in self._recently_modified:
            time_since_last = current_time - self._recently_modified[file_key]
            if time_since_last < self.MODIFICATION_COOLDOWN:
                return ModificationResult(
                    success=False,
                    operation='MODIFY',
                    target=str(file_path),
                    error=f"File was modified {time_since_last:.1f}s ago. Cooldown period: {self.MODIFICATION_COOLDOWN}s"
                )
        
        # Check if file exists
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
            ast.parse(new_code)
        except SyntaxError as e:
            return ModificationResult(
                success=False,
                operation='MODIFY',
                target=str(file_path),
                error=f"Syntax error in new code: {e}"
            )
        
        # Increment modification depth
        self._modification_depth += 1
        
        try:
            # Create backup
            backup_id = None
            if self.backup_manager:
                try:
                    backup_id = self.backup_manager.create_backup(
                        str(file_path),
                        description="Before AI modification"
                    )
                    logger.info(f"Created backup {backup_id} before modifying {file_path}")
                except Exception as e:
                    self._modification_depth -= 1
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
                self._modification_depth -= 1
                return ModificationResult(
                    success=False,
                    operation='MODIFY',
                    target=str(file_path),
                    error=f"Cannot read file: {e}",
                    backup_id=backup_id
                )
            
            # Write new content
            file_path.write_text(new_code, encoding='utf-8')
            
            # Track this modification
            self._recently_modified[file_key] = current_time
            self._modification_history.append({
                'file': file_key,
                'time': current_time,
                'backup_id': backup_id
            })
            
            # Log to improvement engine
            if self.improvement_engine:
                try:
                    from core.self_mod.improvement_engine import ModificationOutcome, OutcomeType
                    outcome = ModificationOutcome(
                        modification_id=backup_id or f"mod_{int(time.time())}",
                        outcome_type=OutcomeType.SUCCESS,
                        timestamp=time.time()
                    )
                    self.improvement_engine.record_outcome(outcome)
                except Exception as e:
                    logger.warning(f"Could not log to improvement engine: {e}")
            
            logger.info(f"Successfully modified {file_path}")
            
            # Decrement depth after successful modification
            self._modification_depth -= 1
            
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
            # Decrement depth on failure
            self._modification_depth -= 1
            
            # Attempt rollback
            if backup_id and self.backup_manager:
                try:
                    self.backup_manager.rollback(backup_id)
                    logger.warning(f"Rolled back {file_path} after failed modification")
                except Exception as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")
            
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
        
        # Check if file already exists
        if file_path.exists():
            return ModificationResult(
                success=False,
                operation='CREATE',
                target=str(file_path),
                error=f"File already exists: {file_path}. Use MODIFY to update existing files."
            )
        
        # Validate syntax for Python files
        if file_path.suffix == '.py':
            try:
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
                error=f"Cannot create parent directory: {e}"
            )
        
        # Write file
        try:
            file_path.write_text(code, encoding='utf-8')
            logger.info(f"Created new file: {file_path}")
            
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
                error=f"Failed to write file: {e}"
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
                logger.info(f"Created backup {backup_id} before deleting {file_path}")
            except Exception as e:
                logger.warning(f"Backup before deletion failed: {e}")
        
        # Delete file
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            
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
                error=f"Failed to delete: {e}",
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
                logger.warning(f"Code analyzer failed: {e}")
        
        # Basic analysis without CodeAnalyzer
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            
            # Count Python structures
            func_count = len([l for l in lines if l.strip().startswith('def ')])
            class_count = len([l for l in lines if l.strip().startswith('class ')])
            import_count = len([l for l in lines if 'import ' in l])
            
            return ModificationResult(
                success=True,
                operation='ANALYZE',
                target=str(file_path),
                details={
                    'lines': len(lines),
                    'characters': len(content),
                    'size_bytes': file_path.stat().st_size,
                    'functions': func_count,
                    'classes': class_count,
                    'imports': import_count,
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
                description="Manual backup requested"
            )
            logger.info(f"Created manual backup {backup_id} for {file_path}")
            
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
            logger.info(f"Rollback to {backup_id}: {'success' if result.success else 'failed'}")
            
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
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
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
        
        This is the MAIN ENTRY POINT for processing AI responses.
        
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
                'details': result.details if result.success else None,
            })
            
            if result.success:
                results['successes'] += 1
                if cmd.get('command') in ('MODIFY', 'CREATE', 'DELETE'):
                    results['modified_files'].append(str(result.target))
            else:
                results['failures'] += 1
        
        logger.info(f"Processed {len(commands)} commands: {results['successes']} success, {results['failures']} failed")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_bridge(jarvis_instance) -> SelfModificationBridge:
    """Create a bridge instance"""
    return SelfModificationBridge(jarvis_instance)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for bridge"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    # Test 1: Command parsing
    bridge = SelfModificationBridge(type('MockJarvis', (), {})())
    
    test_response = """
Let me read the main.py file:
[READ:main.py]

Now I'll modify it:
[MODIFY:main.py]
```python
def hello():
    print("Hello")
```
"""
    
    commands = bridge.parse_commands(test_response)
    if len(commands) == 2:
        results['passed'].append('parse_commands')
    else:
        results['failed'].append(f'parse_commands: found {len(commands)} commands')
    
    # Test 2: Protected file detection
    if bridge._is_protected(Path('.env')):
        results['passed'].append('protected_file_detection')
    else:
        results['failed'].append('protected_file_detection')
    
    # Test 3: System prompt generation
    prompt = bridge.get_system_prompt()
    if 'AVAILABLE TOOLS' in prompt and '[READ:' in prompt:
        results['passed'].append('system_prompt')
    else:
        results['failed'].append('system_prompt')
    
    # Test 4: Project context
    context = bridge._get_project_context()
    if 'Project root' in context:
        results['passed'].append('project_context')
    else:
        results['failed'].append('project_context')
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Self-Modification Bridge - Self Test")
    print("=" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed Tests:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed Tests:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    print("\n" + "=" * 70)
