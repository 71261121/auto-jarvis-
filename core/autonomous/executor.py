#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Autonomous Executor
==========================================

This module EXECUTES operations directly. No more passive waiting.
When user says "read main.py", we READ IT. Immediately.

Key Features:
- Direct file operations (read, write, create, delete)
- Terminal command execution
- Directory listing and search
- AI-assisted modifications (AI generates, we execute)

Device: Realme 2 Pro Lite | RAM: 4GB | Platform: Termux
"""

import os
import re
import ast
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
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
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.operation}: {self.target or self.output[:50]}"


class AutonomousExecutor:
    """
    The EXECUTOR - Actually performs operations.
    
    This is what makes JARVIS autonomous. It doesn't wait for AI
    to generate commands - it directly executes based on detected intent.
    
    Usage:
        executor = AutonomousExecutor(jarvis_instance)
        result = executor.read_file("main.py")
        result = executor.list_directory("core/")
        result = executor.execute_command("python test.py")
    """
    
    # Maximum output size to prevent memory issues on 4GB device
    MAX_OUTPUT_SIZE = 50000  # characters
    MAX_FILE_SIZE = 1000000  # 1MB max file to read
    
    def __init__(self, jarvis_instance=None, project_root: str = None):
        """
        Initialize Executor.
        
        Args:
            jarvis_instance: Main JARVIS instance (for AI calls, backup manager)
            project_root: Root directory of the project
        """
        self.jarvis = jarvis_instance
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Get components from JARVIS if available
        if jarvis_instance:
            self.ai_client = getattr(jarvis_instance, 'ai_client', None)
            self.backup_manager = getattr(jarvis_instance, 'backup_manager', None)
            self.code_analyzer = getattr(jarvis_instance, 'code_analyzer', None)
        else:
            self.ai_client = None
            self.backup_manager = None
            self.code_analyzer = None
        
        logger.info(f"AutonomousExecutor initialized (root: {self.project_root})")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FILE OPERATIONS - Direct execution, no AI needed
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def read_file(self, file_path: str) -> ExecutionResult:
        """
        Read a file and return its content.
        
        DIRECT OPERATION - No AI needed.
        
        Args:
            file_path: Path to file (relative or absolute)
            
        Returns:
            ExecutionResult with file content
        """
        start_time = time.time()
        
        # Resolve path
        path = self._resolve_path(file_path)
        
        # Check if exists
        if not path.exists():
            return ExecutionResult(
                success=False,
                operation="READ",
                target=str(path),
                error=f"File not found: {path}"
            )
        
        # Check if it's a file
        if not path.is_file():
            return ExecutionResult(
                success=False,
                operation="READ",
                target=str(path),
                error=f"Not a file: {path}"
            )
        
        # Check size
        try:
            size = path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                return ExecutionResult(
                    success=False,
                    operation="READ",
                    target=str(path),
                    error=f"File too large ({size} bytes). Max: {self.MAX_FILE_SIZE}"
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="READ",
                target=str(path),
                error=f"Cannot stat file: {e}"
            )
        
        # Read file
        try:
            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.splitlines()
            
            # Build output
            output = self._format_file_content(path.name, content, lines)
            
            elapsed = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                operation="READ",
                target=str(path),
                output=output,
                data={
                    'content': content,
                    'lines': len(lines),
                    'size': size,
                    'path': str(path),
                },
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="READ",
                target=str(path),
                error=f"Cannot read file: {e}"
            )
    
    def list_directory(self, dir_path: str = ".") -> ExecutionResult:
        """
        List contents of a directory.
        
        DIRECT OPERATION - No AI needed.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            ExecutionResult with directory listing
        """
        start_time = time.time()
        
        # Resolve path
        path = self._resolve_path(dir_path)
        
        # Check if exists
        if not path.exists():
            return ExecutionResult(
                success=False,
                operation="LIST",
                target=str(path),
                error=f"Directory not found: {path}"
            )
        
        # Check if directory
        if not path.is_dir():
            return ExecutionResult(
                success=False,
                operation="LIST",
                target=str(path),
                error=f"Not a directory: {path}"
            )
        
        # List contents
        try:
            items = []
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for item in sorted(path.iterdir()):
                # Skip hidden files
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
                        size_str = self._format_size(size)
                        items.append(f"📄 {item.name} ({size_str})")
                except PermissionError:
                    items.append(f"🔒 {item.name} (no access)")
            
            # Build output
            output_lines = [
                f"📂 Directory: {path}",
                f"   {file_count} files, {dir_count} directories",
                f"   Total size: {self._format_size(total_size)}",
                "",
            ]
            output_lines.extend(items)
            output = "\n".join(output_lines)
            
            elapsed = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                operation="LIST",
                target=str(path),
                output=output,
                data={
                    'items': [i.name for i in sorted(path.iterdir()) if not i.name.startswith('.')],
                    'file_count': file_count,
                    'dir_count': dir_count,
                    'total_size': total_size,
                },
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="LIST",
                target=str(path),
                error=f"Cannot list directory: {e}"
            )
    
    def search_files(self, pattern: str, directory: str = ".") -> ExecutionResult:
        """
        Search for files containing a pattern.
        
        DIRECT OPERATION - No AI needed.
        
        Args:
            pattern: Search pattern
            directory: Directory to search in
            
        Returns:
            ExecutionResult with search results
        """
        start_time = time.time()
        
        # Resolve directory
        search_dir = self._resolve_path(directory)
        
        if not search_dir.exists():
            return ExecutionResult(
                success=False,
                operation="SEARCH",
                target=pattern,
                error=f"Directory not found: {search_dir}"
            )
        
        results = []
        pattern_lower = pattern.lower()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Only search text files
                    if not any(file.endswith(ext) for ext in ['.py', '.txt', '.md', '.json', '.yaml', '.yml', '.sh', '.js', '.ts', '.html', '.css']):
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if pattern_lower in content.lower():
                            rel_path = file_path.relative_to(self.project_root)
                            results.append(str(rel_path))
                            
                            if len(results) >= 20:  # Limit results
                                break
                    except Exception:
                        pass
                
                if len(results) >= 20:
                    break
            
            # Build output
            if results:
                output_lines = [f"🔍 Found '{pattern}' in {len(results)} files:"]
                for r in results:
                    output_lines.append(f"   📄 {r}")
                output = "\n".join(output_lines)
            else:
                output = f"🔍 No files found containing '{pattern}'"
            
            elapsed = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                operation="SEARCH",
                target=pattern,
                output=output,
                data={'results': results, 'count': len(results)},
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="SEARCH",
                target=pattern,
                error=f"Search failed: {e}"
            )
    
    def delete_file(self, file_path: str, create_backup: bool = True) -> ExecutionResult:
        """
        Delete a file with optional backup.
        
        Args:
            file_path: Path to file
            create_backup: Whether to create backup before deletion
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # Resolve path
        path = self._resolve_path(file_path)
        
        # Check if exists
        if not path.exists():
            return ExecutionResult(
                success=False,
                operation="DELETE",
                target=str(path),
                error=f"File not found: {path}"
            )
        
        # Create backup if requested
        backup_id = ""
        if create_backup and self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(
                    str(path),
                    description="Before deletion"
                )
            except Exception as e:
                logger.warning(f"Backup before deletion failed: {e}")
        
        # Delete file
        try:
            path.unlink()
            
            elapsed = (time.time() - start_time) * 1000
            
            output = f"🗑️ Deleted: {path}"
            if backup_id:
                output += f"\n   Backup ID: {backup_id}"
            
            return ExecutionResult(
                success=True,
                operation="DELETE",
                target=str(path),
                output=output,
                backup_id=backup_id,
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="DELETE",
                target=str(path),
                error=f"Cannot delete file: {e}"
            )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TERMINAL OPERATIONS - Direct execution with safety checks
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def execute_command(self, command: str, timeout: int = 60) -> ExecutionResult:
        """
        Execute a terminal command.
        
        DIRECT OPERATION - No AI needed.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            ExecutionResult with command output
        """
        start_time = time.time()
        
        # Safety check - block dangerous commands
        dangerous_patterns = [
            r'rm\s+-rf\s+/', r'dd\s+if=', r'mkfs', r'format',
            r'shutdown', r'reboot', r'init\s+0', r'halt',
            r'>\s*/dev/sd', r':(){ :|:& };:',  # Fork bomb
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return ExecutionResult(
                    success=False,
                    operation="EXECUTE",
                    target=command,
                    error=f"Blocked dangerous command pattern: {pattern}"
                )
        
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root)
            )
            
            # Get output
            stdout = result.stdout[:self.MAX_OUTPUT_SIZE]
            stderr = result.stderr[:self.MAX_OUTPUT_SIZE]
            
            elapsed = (time.time() - start_time) * 1000
            
            # Build output
            output_lines = [f"⚡ Executed: {command}", ""]
            if stdout:
                output_lines.append("Output:")
                output_lines.append(stdout)
            if stderr:
                output_lines.append("Errors:")
                output_lines.append(stderr)
            if not stdout and not stderr:
                output_lines.append("(no output)")
            
            output = "\n".join(output_lines)
            
            return ExecutionResult(
                success=result.returncode == 0,
                operation="EXECUTE",
                target=command,
                output=output,
                error=stderr if result.returncode != 0 else "",
                data={
                    'return_code': result.returncode,
                    'stdout': stdout,
                    'stderr': stderr,
                },
                execution_time_ms=elapsed
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                operation="EXECUTE",
                target=command,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="EXECUTE",
                target=command,
                error=f"Execution failed: {e}"
            )
    
    def install_package(self, package: str, package_manager: str = "auto") -> ExecutionResult:
        """
        Install a package using pip or pkg.
        
        Args:
            package: Package name
            package_manager: 'pip', 'pkg', or 'auto'
            
        Returns:
            ExecutionResult
        """
        # Determine package manager
        if package_manager == "auto":
            # Check if in Termux
            if os.environ.get('TERMUX_VERSION'):
                package_manager = "pkg"
            else:
                package_manager = "pip"
        
        if package_manager == "pkg":
            command = f"pkg install -y {package}"
        else:
            command = f"pip install {package}"
        
        return self.execute_command(command, timeout=120)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # AI-ASSISTED OPERATIONS - AI generates, we execute
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def modify_file(self, file_path: str, modification_request: str) -> ExecutionResult:
        """
        Modify a file using AI assistance.
        
        AI-ASSISTED OPERATION:
        1. Read current file content
        2. Ask AI to generate modification
        3. Validate syntax
        4. Create backup
        5. Write new content
        
        Args:
            file_path: Path to file
            modification_request: What modification to make
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # Resolve path
        path = self._resolve_path(file_path)
        
        # Read current content
        if not path.exists():
            return ExecutionResult(
                success=False,
                operation="MODIFY",
                target=str(path),
                error=f"File not found: {path}"
            )
        
        try:
            current_content = path.read_text(encoding='utf-8')
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="MODIFY",
                target=str(path),
                error=f"Cannot read file: {e}"
            )
        
        # Get AI assistance
        if not self.ai_client:
            return ExecutionResult(
                success=False,
                operation="MODIFY",
                target=str(path),
                error="AI client not available for modification"
            )
        
        # Build prompt for AI
        system_prompt = """You are a code modification expert. You will be given:
1. A file's current content
2. A modification request

You must return the COMPLETE MODIFIED file content. Do not explain, just return the code.
Only output the code, nothing else. The code should be a complete, working replacement."""

        user_prompt = f"""File: {path.name}

Current content:
```python
{current_content}
```

Modification request: {modification_request}

Return the complete modified file content:"""

        try:
            # Get AI response
            response = self.ai_client.chat(
                message=user_prompt,
                system=system_prompt,
                temperature=0.3
            )
            
            if not response.success:
                return ExecutionResult(
                    success=False,
                    operation="MODIFY",
                    target=str(path),
                    error=f"AI failed: {response.error}"
                )
            
            # Extract code from response
            new_content = self._extract_code(response.content)
            
            if not new_content:
                return ExecutionResult(
                    success=False,
                    operation="MODIFY",
                    target=str(path),
                    error="Could not extract code from AI response"
                )
            
            # Validate syntax for Python files
            if path.suffix == '.py':
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    return ExecutionResult(
                        success=False,
                        operation="MODIFY",
                        target=str(path),
                        error=f"Generated code has syntax error: {e}"
                    )
            
            # Create backup
            backup_id = ""
            if self.backup_manager:
                try:
                    backup_id = self.backup_manager.create_backup(
                        str(path),
                        description=f"Before: {modification_request[:50]}"
                    )
                except Exception as e:
                    logger.warning(f"Backup failed: {e}")
            
            # Write new content
            path.write_text(new_content, encoding='utf-8')
            
            elapsed = (time.time() - start_time) * 1000
            
            output = f"✏️ Modified: {path}"
            output += f"\n   Change: {modification_request[:100]}"
            if backup_id:
                output += f"\n   Backup ID: {backup_id}"
            
            return ExecutionResult(
                success=True,
                operation="MODIFY",
                target=str(path),
                output=output,
                backup_id=backup_id,
                data={
                    'original_lines': len(current_content.splitlines()),
                    'new_lines': len(new_content.splitlines()),
                },
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="MODIFY",
                target=str(path),
                error=f"Modification failed: {e}"
            )
    
    def create_file(self, file_path: str, content_description: str) -> ExecutionResult:
        """
        Create a new file using AI assistance.
        
        AI-ASSISTED OPERATION:
        1. Ask AI to generate content
        2. Validate syntax
        3. Create file
        
        Args:
            file_path: Path for new file
            content_description: What the file should contain
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # Resolve path
        path = self._resolve_path(file_path)
        
        # Check if already exists
        if path.exists():
            return ExecutionResult(
                success=False,
                operation="CREATE",
                target=str(path),
                error=f"File already exists: {path}"
            )
        
        # Get AI assistance
        if not self.ai_client:
            return ExecutionResult(
                success=False,
                operation="CREATE",
                target=str(path),
                error="AI client not available for file creation"
            )
        
        # Build prompt
        system_prompt = """You are a code generation expert. Generate the requested file content.
Only output the code, nothing else. Include proper imports, error handling, and documentation."""

        user_prompt = f"""Create a file named '{path.name}' with the following:
{content_description}

Return only the complete file content:"""

        try:
            # Get AI response
            response = self.ai_client.chat(
                message=user_prompt,
                system=system_prompt,
                temperature=0.3
            )
            
            if not response.success:
                return ExecutionResult(
                    success=False,
                    operation="CREATE",
                    target=str(path),
                    error=f"AI failed: {response.error}"
                )
            
            # Extract code
            content = self._extract_code(response.content)
            
            if not content:
                content = response.content  # Use raw response if no code block
            
            # Validate syntax for Python files
            if path.suffix == '.py' and content.strip():
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    return ExecutionResult(
                        success=False,
                        operation="CREATE",
                        target=str(path),
                        error=f"Generated code has syntax error: {e}"
                    )
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(content, encoding='utf-8')
            
            elapsed = (time.time() - start_time) * 1000
            
            output = f"📝 Created: {path}"
            output += f"\n   Lines: {len(content.splitlines())}"
            
            return ExecutionResult(
                success=True,
                operation="CREATE",
                target=str(path),
                output=output,
                data={
                    'lines': len(content.splitlines()),
                    'size': len(content),
                },
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                operation="CREATE",
                target=str(path),
                error=f"Creation failed: {e}"
            )
    
    def analyze_code(self, file_path: str, focus: str = None) -> ExecutionResult:
        """
        Analyze code for issues and suggestions.
        
        AI-ASSISTED OPERATION
        
        Args:
            file_path: Path to file
            focus: What to focus analysis on
            
        Returns:
            ExecutionResult with analysis
        """
        start_time = time.time()
        
        # First read the file
        read_result = self.read_file(file_path)
        
        if not read_result.success:
            return read_result
        
        content = read_result.data.get('content', '')
        path = Path(read_result.target)
        
        # Basic analysis (local)
        analysis = self._basic_code_analysis(content)
        
        # AI deep analysis
        if self.ai_client:
            system_prompt = """You are a code analysis expert. Analyze the code and provide:
1. Overall assessment
2. Potential issues/bugs
3. Security concerns
4. Performance issues
5. Suggestions for improvement

Be specific and actionable."""

            user_prompt = f"""Analyze this code from '{path.name}':
```python
{content[:5000]}  # Limit for API
```
"""
            if focus:
                user_prompt += f"\nFocus on: {focus}"
            
            try:
                response = self.ai_client.chat(
                    message=user_prompt,
                    system=system_prompt
                )
                
                if response.success:
                    analysis['ai_analysis'] = response.content
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
        
        elapsed = (time.time() - start_time) * 1000
        
        # Build output
        output_lines = [
            f"📊 Analysis of: {path.name}",
            "",
            f"Lines: {analysis.get('lines', 0)}",
            f"Functions: {analysis.get('functions', 0)}",
            f"Classes: {analysis.get('classes', 0)}",
            f"Imports: {analysis.get('imports', 0)}",
        ]
        
        if analysis.get('ai_analysis'):
            output_lines.append("")
            output_lines.append("AI Analysis:")
            output_lines.append(analysis['ai_analysis'])
        
        return ExecutionResult(
            success=True,
            operation="ANALYZE",
            target=str(path),
            output="\n".join(output_lines),
            data=analysis,
            execution_time_ms=elapsed
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _resolve_path(self, target: str) -> Path:
        """Resolve a target string to an absolute path"""
        if not target or target == ".":
            return self.project_root
        
        target_path = Path(target)
        
        if target_path.is_absolute():
            return target_path
        
        # Check relative to project root
        full_path = self.project_root / target
        if full_path.exists():
            return full_path
        
        return full_path
    
    def _format_file_content(self, filename: str, content: str, lines: list) -> str:
        """Format file content for display"""
        output = [f"📄 {filename} ({len(lines)} lines)", ""]
        
        # Add line numbers
        for i, line in enumerate(lines[:100], 1):  # Limit to 100 lines for display
            output.append(f"{i:4d} | {line}")
        
        if len(lines) > 100:
            output.append(f"... ({len(lines) - 100} more lines)")
        
        return "\n".join(output)
    
    def _format_size(self, size: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text (from markdown code blocks)"""
        # Try to find code block
        patterns = [
            r'```(?:python)?\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'```python\n(.*?)\n```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # No code block found, check if whole text is code
        if text.strip().startswith(('import ', 'def ', 'class ', 'from ', '#!')):
            return text.strip()
        
        return ""
    
    def _basic_code_analysis(self, content: str) -> Dict[str, Any]:
        """Perform basic code analysis"""
        lines = content.splitlines()
        
        return {
            'lines': len(lines),
            'characters': len(content),
            'functions': len([l for l in lines if l.strip().startswith('def ')]),
            'classes': len([l for l in lines if l.strip().startswith('class ')]),
            'imports': len([l for l in lines if 'import ' in l]),
            'comments': len([l for l in lines if l.strip().startswith('#')]),
            'empty_lines': len([l for l in lines if not l.strip()]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for AutonomousExecutor"""
    results = {
        'passed': [],
        'failed': [],
    }
    
    executor = AutonomousExecutor(project_root=os.getcwd())
    
    # Test 1: List directory
    result = executor.list_directory(".")
    if result.success:
        results['passed'].append('list_directory')
    else:
        results['failed'].append(f'list_directory: {result.error}')
    
    # Test 2: Read file (main.py should exist)
    result = executor.read_file("main.py")
    if result.success:
        results['passed'].append('read_file')
    else:
        results['failed'].append(f'read_file: {result.error}')
    
    # Test 3: Search files
    result = executor.search_files("import")
    if result.success:
        results['passed'].append('search_files')
    else:
        results['failed'].append(f'search_files: {result.error}')
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Autonomous Executor - Self Test")
    print("=" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    print(f"\nTotal: {len(test_results['passed'])}/{len(test_results['passed']) + len(test_results['failed'])} passed")
