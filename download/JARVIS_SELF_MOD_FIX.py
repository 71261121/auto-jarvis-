#!/usr/bin/env python3
"""
JARVIS Self-Modification Integration Fix
=========================================

Is file ko main.py ke _handle_ai_command() function mein
integrate karna hai.

PROBLEM: AI sirf text respond karta hai, files modify nahi karta.
SOLUTION: AI ko self-modification tools access dena.
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# SELF-MODIFICATION BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class SelfModificationBridge:
    """
    Bridge between AI responses and actual file modifications.
    
    This class:
    1. Parses AI responses for modification commands
    2. Reads project files
    3. Executes modifications through ModificationEngine
    4. Creates backups before changes
    """
    
    # Commands that AI can use
    COMMANDS = {
        'READ_FILE': r'\[READ:(.+?)\]',
        'MODIFY_FILE': r'\[MODIFY:(.+?)\]',
        'CREATE_FILE': r'\[CREATE:(.+?)\]',
        'DELETE_FILE': r'\[DELETE:(.+?)\]',
        'LIST_DIR': r'\[LIST:(.+?)\]',
        'ANALYZE': r'\[ANALYZE:(.+?)\]',
    }
    
    def __init__(self, jarvis_instance):
        """
        Initialize bridge with JARVIS instance.
        
        Args:
            jarvis_instance: Main JARVIS class instance
        """
        self.jarvis = jarvis_instance
        self.project_root = Path(__file__).parent.parent
        
        # Get components from JARVIS
        self.code_analyzer = getattr(jarvis_instance, 'code_analyzer', None)
        self.backup_manager = getattr(jarvis_instance, 'backup_manager', None)
        self.safe_modifier = getattr(jarvis_instance, 'safe_modifier', None)
        self.ai_client = getattr(jarvis_instance, 'ai_client', None)
        
        # Project context for AI
        self.project_context = self._build_project_context()
    
    def _build_project_context(self) -> str:
        """Build context about project structure"""
        context = []
        context.append("PROJECT STRUCTURE:")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            level = root.replace(str(self.project_root), '').count(os.sep)
            indent = ' ' * 2 * level
            context.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Limit files per directory
                if file.endswith('.py'):
                    context.append(f"{subindent}{file}")
        
        return '\n'.join(context)
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt with self-modification capabilities.
        
        This prompt tells the AI it CAN modify files.
        """
        return """You are JARVIS, a SELF-MODIFYING AI assistant.

CRITICAL: You have the ability to READ, MODIFY, CREATE, and DELETE files in your own codebase!

## AVAILABLE COMMANDS:

1. **Read a file:**
   [READ:path/to/file.py]
   Returns the file content for you to analyze.

2. **Modify a file:**
   [MODIFY:path/to/file.py]
   ```python
   # New code here
   ```
   Creates backup automatically, then applies changes.

3. **Create a new file:**
   [CREATE:path/to/new_file.py]
   ```python
   # New file content
   ```

4. **Delete a file:**
   [DELETE:path/to/file.py]
   Creates backup before deletion.

5. **List directory:**
   [LIST:path/to/directory]
   Shows files and subdirectories.

6. **Analyze code:**
   [ANALYZE:path/to/file.py]
   Runs code analysis (complexity, issues, etc.)

## SAFETY RULES:
- ALWAYS read a file before modifying
- Create backups are automatic
- Explain what you're changing and why
- Test after modifications if possible

## YOUR PROJECT:
""" + self.project_context + """

When asked to modify code, USE THE COMMANDS above!
Do NOT just suggest - actually DO the modification.
"""

    def process_ai_response(self, response: str) -> Dict[str, Any]:
        """
        Process AI response and execute any commands.
        
        Args:
            response: AI's response text
            
        Returns:
            Dict with executed commands and results
        """
        results = {
            'commands_found': [],
            'commands_executed': [],
            'results': [],
            'modified_files': [],
            'errors': [],
        }
        
        # Find all commands in response
        for cmd_name, pattern in self.COMMANDS.items():
            matches = re.findall(pattern, response)
            for match in matches:
                results['commands_found'].append({
                    'command': cmd_name,
                    'target': match
                })
        
        # Execute each command
        for cmd in results['commands_found']:
            try:
                result = self._execute_command(
                    cmd['command'], 
                    cmd['target'], 
                    response
                )
                results['commands_executed'].append(cmd)
                results['results'].append(result)
                
                if result.get('modified'):
                    results['modified_files'].append(cmd['target'])
                    
            except Exception as e:
                results['errors'].append({
                    'command': cmd['command'],
                    'target': cmd['target'],
                    'error': str(e)
                })
        
        return results
    
    def _execute_command(
        self, 
        command: str, 
        target: str, 
        full_response: str
    ) -> Dict[str, Any]:
        """Execute a single command"""
        
        result = {
            'command': command,
            'target': target,
            'success': False,
            'modified': False,
            'output': '',
        }
        
        # Resolve path
        if target.startswith('/'):
            file_path = Path(target)
        else:
            file_path = self.project_root / target
        
        # Execute based on command type
        if command == 'READ_FILE':
            result.update(self._cmd_read_file(file_path))
            
        elif command == 'MODIFY_FILE':
            result.update(self._cmd_modify_file(file_path, full_response))
            
        elif command == 'CREATE_FILE':
            result.update(self._cmd_create_file(file_path, full_response))
            
        elif command == 'DELETE_FILE':
            result.update(self._cmd_delete_file(file_path))
            
        elif command == 'LIST_DIR':
            result.update(self._cmd_list_dir(file_path))
            
        elif command == 'ANALYZE':
            result.update(self._cmd_analyze(file_path))
        
        return result
    
    def _cmd_read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read a file"""
        if not file_path.exists():
            return {
                'success': False,
                'output': f'File not found: {file_path}'
            }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return {
                'success': True,
                'output': content,
                'lines': len(content.splitlines()),
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Error reading file: {e}'
            }
    
    def _cmd_modify_file(
        self, 
        file_path: Path, 
        full_response: str
    ) -> Dict[str, Any]:
        """Modify a file with backup"""
        
        if not file_path.exists():
            return {
                'success': False,
                'output': f'File not found: {file_path}'
            }
        
        # Extract new code from response
        code_match = re.search(
            r'\[MODIFY:.+?\]\s*```(?:python)?\s*\n(.*?)\n```',
            full_response,
            re.DOTALL
        )
        
        if not code_match:
            return {
                'success': False,
                'output': 'No code block found after MODIFY command'
            }
        
        new_code = code_match.group(1)
        
        # Create backup
        backup_id = None
        if self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(
                    str(file_path),
                    description=f"Before AI modification"
                )
            except Exception as e:
                return {
                    'success': False,
                    'output': f'Backup failed: {e}'
                }
        
        # Read original content
        try:
            original = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return {
                'success': False,
                'output': f'Cannot read file: {e}'
            }
        
        # Write new content
        try:
            file_path.write_text(new_code, encoding='utf-8')
            return {
                'success': True,
                'modified': True,
                'output': f'File modified successfully. Backup: {backup_id}',
                'backup_id': backup_id,
                'original_lines': len(original.splitlines()),
                'new_lines': len(new_code.splitlines()),
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Write failed: {e}',
                'backup_id': backup_id,
            }
    
    def _cmd_create_file(
        self, 
        file_path: Path, 
        full_response: str
    ) -> Dict[str, Any]:
        """Create a new file"""
        
        # Extract code from response
        code_match = re.search(
            r'\[CREATE:.+?\]\s*```(?:python)?\s*\n(.*?)\n```',
            full_response,
            re.DOTALL
        )
        
        if not code_match:
            return {
                'success': False,
                'output': 'No code block found after CREATE command'
            }
        
        new_code = code_match.group(1)
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        try:
            file_path.write_text(new_code, encoding='utf-8')
            return {
                'success': True,
                'modified': True,
                'output': f'File created: {file_path}',
                'lines': len(new_code.splitlines()),
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Create failed: {e}'
            }
    
    def _cmd_delete_file(self, file_path: Path) -> Dict[str, Any]:
        """Delete a file with backup"""
        
        if not file_path.exists():
            return {
                'success': False,
                'output': f'File not found: {file_path}'
            }
        
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
            return {
                'success': True,
                'modified': True,
                'output': f'File deleted. Backup: {backup_id}',
                'backup_id': backup_id,
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Delete failed: {e}'
            }
    
    def _cmd_list_dir(self, dir_path: Path) -> Dict[str, Any]:
        """List directory contents"""
        
        if not dir_path.exists():
            dir_path = self.project_root / dir_path
        
        if not dir_path.exists():
            return {
                'success': False,
                'output': f'Directory not found: {dir_path}'
            }
        
        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            return {
                'success': True,
                'output': '\n'.join(items),
                'count': len(items),
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'List failed: {e}'
            }
    
    def _cmd_analyze(self, file_path: Path) -> Dict[str, Any]:
        """Analyze code file"""
        
        if not file_path.exists():
            return {
                'success': False,
                'output': f'File not found: {file_path}'
            }
        
        if not self.code_analyzer:
            return {
                'success': False,
                'output': 'Code analyzer not available'
            }
        
        try:
            analysis = self.code_analyzer.analyze_file(str(file_path))
            
            output = []
            output.append(f"📊 Analysis of {file_path.name}")
            output.append(f"")
            output.append(f"Functions: {len(analysis.functions)}")
            output.append(f"Classes: {len(analysis.classes)}")
            output.append(f"Imports: {len(analysis.imports)}")
            
            if analysis.total_complexity:
                output.append(f"Complexity: {analysis.total_complexity.cyclomatic}")
            
            if analysis.issues:
                output.append(f"\n⚠️ Issues ({len(analysis.issues)}):")
                for issue in analysis.issues[:10]:
                    output.append(f"  - {issue.message}")
            
            return {
                'success': True,
                'output': '\n'.join(output),
                'analysis': analysis,
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Analysis failed: {e}'
            }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION CODE FOR main.py
# ═══════════════════════════════════════════════════════════════════════════════

"""
## main.py mein yeh changes karne honge:

### 1. Import add karo (top of file):
```python
from core.self_mod.bridge import SelfModificationBridge
```

### 2. __init__ mein bridge initialize karo:
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Self-modification bridge
    self.mod_bridge = SelfModificationBridge(self)
```

### 3. _handle_ai_command() ko update karo:
```python
def _handle_ai_command(self, command: str):
    if not self.ai_client:
        print("AI client not available")
        return
    
    try:
        # Get system prompt with modification capabilities
        system_prompt = self.mod_bridge.get_system_prompt()
        
        # Send to AI
        response = self.ai_client.chat(
            message=command,
            system=system_prompt,
            context_id="default"
        )
        
        if response.success:
            # Display AI response
            print(response.content)
            
            # Process any modification commands
            mod_result = self.mod_bridge.process_ai_response(response.content)
            
            if mod_result['modified_files']:
                print("\n✅ Modified files:")
                for f in mod_result['modified_files']:
                    print(f"   - {f}")
            
            if mod_result['errors']:
                print("\n❌ Errors:")
                for e in mod_result['errors']:
                    print(f"   - {e}")
        else:
            print(f"AI Error: {response.error}")
            
    except Exception as e:
        print(f"AI error: {e}")
```
"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

EXAMPLE_PROMPTS = """
## Test Prompts for Self-Modifying JARVIS:

### 1. Read a file:
"Read the main.py file and tell me what it does."

### 2. Modify a file:
"Modify main.py to add a new command called 'debug' that enables debug mode."

### 3. Create a new file:
"Create a new file called core/utils.py with helper functions for string manipulation."

### 4. Analyze code:
"Analyze the code_analyzer.py file and tell me about its complexity."

### 5. Self-improvement:
"Look at your own main.py file and suggest improvements to the _handle_command function."

### 6. Add a new feature:
"Add a new feature to export chat history to a JSON file. Create the necessary files."

### 7. Fix a bug:
"There's a bug in the chat storage. Find it and fix it."

### 8. List project structure:
"List all files in the core directory and tell me what each one does."
"""

if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Self-Modification Integration Fix")
    print("=" * 70)
    print()
    print("This file provides the bridge between AI responses and")
    print("actual file modifications.")
    print()
    print("To enable self-modification:")
    print("1. Save this file as: core/self_mod/bridge.py")
    print("2. Apply the changes shown above to main.py")
    print("3. Restart JARVIS")
    print()
    print(EXAMPLE_PROMPTS)
