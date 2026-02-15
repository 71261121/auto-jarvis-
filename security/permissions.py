#!/usr/bin/env python3
"""
JARVIS Permission Manager
Ultra-Advanced Permission and Access Control System

Features:
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)
- Hierarchical permissions
- Dynamic permission evaluation
- Permission inheritance
- Resource-level permissions
- Action-level permissions
- Context-aware authorization
- Permission caching
- Audit trail for permission checks

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from abc import ABC, abstractmethod
from functools import wraps
import re


# Constants
DEFAULT_CACHE_TTL = 300  # 5 minutes
MAX_CACHE_SIZE = 10000


class PermissionAction(Enum):
    """Permission actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    ALL = "*"


class ResourceScope(Enum):
    """Resource scope levels"""
    GLOBAL = "global"
    SYSTEM = "system"
    MODULE = "module"
    RESOURCE = "resource"
    FIELD = "field"
    OWN = "own"


@dataclass
class Permission:
    """Permission definition"""
    name: str
    resource: str
    actions: Set[PermissionAction]
    scope: ResourceScope = ResourceScope.RESOURCE
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __hash__(self):
        return hash((self.name, self.resource))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'resource': self.resource,
            'actions': [a.value for a in self.actions],
            'scope': self.scope.value,
            'conditions': self.conditions,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            resource=data['resource'],
            actions={PermissionAction(a) for a in data['actions']},
            scope=ResourceScope(data.get('scope', 'resource')),
            conditions=data.get('conditions', {}),
            description=data.get('description', '')
        )

    def matches_action(self, action: PermissionAction) -> bool:
        """Check if permission includes action"""
        return PermissionAction.ALL in self.actions or action in self.actions


@dataclass
class Role:
    """Role definition"""
    role_id: str
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits: Set[str] = field(default_factory=set)  # Role IDs to inherit from
    priority: int = 0
    description: str = ""
    is_system: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'role_id': self.role_id,
            'name': self.name,
            'permissions': [p.to_dict() for p in self.permissions],
            'inherits': list(self.inherits),
            'priority': self.priority,
            'description': self.description,
            'is_system': self.is_system,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create from dictionary"""
        return cls(
            role_id=data['role_id'],
            name=data['name'],
            permissions={Permission.from_dict(p) for p in data.get('permissions', [])},
            inherits=set(data.get('inherits', [])),
            priority=data.get('priority', 0),
            description=data.get('description', ''),
            is_system=data.get('is_system', False),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()
        )


@dataclass
class PermissionCheckResult:
    """Result of permission check"""
    granted: bool
    permission: Optional[Permission] = None
    role: Optional[Role] = None
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'granted': self.granted,
            'permission': self.permission.to_dict() if self.permission else None,
            'role': self.role.name if self.role else None,
            'reason': self.reason,
            'context': self.context
        }


class PermissionCondition(ABC):
    """Abstract permission condition"""

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context"""
        pass


class TimeCondition(PermissionCondition):
    """Time-based permission condition"""

    def __init__(self, allowed_hours: Tuple[int, int] = None,
                 allowed_days: List[int] = None):
        self.allowed_hours = allowed_hours  # (start_hour, end_hour)
        self.allowed_days = allowed_days  # 0=Monday, 6=Sunday

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if current time is allowed"""
        now = datetime.now()

        if self.allowed_hours:
            if not (self.allowed_hours[0] <= now.hour < self.allowed_hours[1]):
                return False

        if self.allowed_days:
            if now.weekday() not in self.allowed_days:
                return False

        return True


class IPCondition(PermissionCondition):
    """IP-based permission condition"""

    def __init__(self, allowed_ips: List[str] = None,
                 blocked_ips: List[str] = None,
                 allowed_ranges: List[str] = None):
        self.allowed_ips = set(allowed_ips or [])
        self.blocked_ips = set(blocked_ips or [])
        self.allowed_ranges = allowed_ranges or []

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if IP is allowed"""
        ip = context.get('ip_address')

        if not ip:
            return True  # No IP context, allow

        if ip in self.blocked_ips:
            return False

        if self.allowed_ips and ip not in self.allowed_ips:
            # Check ranges
            if self.allowed_ranges:
                import ipaddress
                try:
                    ip_obj = ipaddress.ip_address(ip)
                    in_range = False
                    for range_str in self.allowed_ranges:
                        network = ipaddress.ip_network(range_str, strict=False)
                        if ip_obj in network:
                            in_range = True
                            break
                    if not in_range:
                        return False
                except ValueError:
                    return False
            else:
                return False

        return True


class AttributeCondition(PermissionCondition):
    """Attribute-based permission condition"""

    def __init__(self, attributes: Dict[str, Any]):
        self.attributes = attributes

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if context matches required attributes"""
        for key, value in self.attributes.items():
            if context.get(key) != value:
                return False
        return True


class PermissionCache:
    """Permission check cache"""

    def __init__(self, ttl: int = DEFAULT_CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[bool, float]] = {}
        self._lock = threading.Lock()

    def _make_key(self, user_id: str, resource: str, action: str,
                  context: Dict = None) -> str:
        """Generate cache key"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        return f"{user_id}:{resource}:{action}:{hash(context_str)}"

    def get(self, user_id: str, resource: str, action: str,
            context: Dict = None) -> Optional[bool]:
        """Get cached result"""
        key = self._make_key(user_id, resource, action, context)

        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return result
                else:
                    del self._cache[key]

        return None

    def set(self, user_id: str, resource: str, action: str,
            result: bool, context: Dict = None) -> None:
        """Cache result"""
        key = self._make_key(user_id, resource, action, context)

        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entries
                sorted_keys = sorted(self._cache.keys(),
                                    key=lambda k: self._cache[k][1])
                for old_key in sorted_keys[:self.max_size // 4]:
                    del self._cache[old_key]

            self._cache[key] = (result, time.time())

    def invalidate(self, user_id: str = None) -> None:
        """Invalidate cache entries"""
        with self._lock:
            if user_id:
                keys_to_remove = [k for k in self._cache.keys()
                                 if k.startswith(f"{user_id}:")]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                self._cache.clear()


class RoleStore:
    """Storage for roles"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.jarvis/roles.json")
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._init_default_roles()
        self._load()

    def _init_default_roles(self) -> None:
        """Initialize default system roles"""
        # Guest role
        guest = Role(
            role_id="guest",
            name="Guest",
            permissions={
                Permission("read_public", "public", {PermissionAction.READ}),
            },
            priority=0,
            description="Guest user with minimal access",
            is_system=True
        )

        # User role
        user = Role(
            role_id="user",
            name="User",
            permissions={
                Permission("read_own", "data", {PermissionAction.READ}, ResourceScope.OWN),
                Permission("write_own", "data", {PermissionAction.CREATE, PermissionAction.UPDATE}, ResourceScope.OWN),
                Permission("delete_own", "data", {PermissionAction.DELETE}, ResourceScope.OWN),
                Permission("execute_commands", "commands", {PermissionAction.EXECUTE}),
            },
            inherits={"guest"},
            priority=10,
            description="Standard user",
            is_system=True
        )

        # Developer role
        developer = Role(
            role_id="developer",
            name="Developer",
            permissions={
                Permission("read_all", "data", {PermissionAction.READ}, ResourceScope.GLOBAL),
                Permission("modify_code", "code", {PermissionAction.READ, PermissionAction.UPDATE}),
                Permission("execute_sandbox", "sandbox", {PermissionAction.EXECUTE}),
                Permission("view_logs", "logs", {PermissionAction.READ}),
            },
            inherits={"user"},
            priority=30,
            description="Developer with code access",
            is_system=True
        )

        # Admin role
        admin = Role(
            role_id="admin",
            name="Administrator",
            permissions={
                Permission("admin_all", "*", {PermissionAction.ALL}, ResourceScope.GLOBAL),
                Permission("manage_users", "users", {PermissionAction.ALL}),
                Permission("manage_roles", "roles", {PermissionAction.ALL}),
                Permission("view_audit", "audit", {PermissionAction.READ}),
                Permission("manage_settings", "settings", {PermissionAction.ALL}),
            },
            inherits={"developer"},
            priority=50,
            description="Administrator with full access",
            is_system=True
        )

        # Super Admin role
        super_admin = Role(
            role_id="super_admin",
            name="Super Administrator",
            permissions={
                Permission("super_all", "*", {PermissionAction.ALL}, ResourceScope.GLOBAL),
                Permission("system_admin", "system", {PermissionAction.ALL}),
            },
            inherits={"admin"},
            priority=100,
            description="Super administrator with unlimited access",
            is_system=True
        )

        # System role
        system = Role(
            role_id="system",
            name="System",
            permissions={
                Permission("system_all", "*", {PermissionAction.ALL}, ResourceScope.GLOBAL),
            },
            priority=200,
            description="System role for internal operations",
            is_system=True
        )

        self._roles = {
            "guest": guest,
            "user": user,
            "developer": developer,
            "admin": admin,
            "super_admin": super_admin,
            "system": system,
        }

    def _load(self) -> None:
        """Load roles from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for role_data in data.get('roles', []):
                    role = Role.from_dict(role_data)
                    if not role.is_system:  # Don't override system roles
                        self._roles[role.role_id] = role

                for user_id, role_ids in data.get('user_roles', {}).items():
                    self._user_roles[user_id] = set(role_ids)

        except Exception as e:
            print(f"Warning: Could not load roles: {e}")

    def _save(self) -> None:
        """Save roles to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            data = {
                'roles': [r.to_dict() for r in self._roles.values() if not r.is_system],
                'user_roles': {uid: list(roles) for uid, roles in self._user_roles.items()},
                'version': 1
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save roles: {e}")

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return self._roles.get(role_id)

    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name"""
        for role in self._roles.values():
            if role.name.lower() == name.lower():
                return role
        return None

    def create_role(self, role: Role) -> bool:
        """Create a new role"""
        with self._lock:
            if role.role_id in self._roles:
                return False

            self._roles[role.role_id] = role
            self._save()
            return True

    def update_role(self, role: Role) -> bool:
        """Update an existing role"""
        with self._lock:
            if role.role_id not in self._roles:
                return False

            existing = self._roles[role.role_id]
            if existing.is_system:
                return False  # Can't modify system roles

            role.updated_at = datetime.now()
            self._roles[role.role_id] = role
            self._save()
            return True

    def delete_role(self, role_id: str) -> bool:
        """Delete a role"""
        with self._lock:
            role = self._roles.get(role_id)
            if not role or role.is_system:
                return False

            del self._roles[role_id]

            # Remove from user assignments
            for user_roles in self._user_roles.values():
                user_roles.discard(role_id)

            self._save()
            return True

    def list_roles(self) -> List[Role]:
        """List all roles"""
        return list(self._roles.values())

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user"""
        with self._lock:
            if role_id not in self._roles:
                return False

            self._user_roles[user_id].add(role_id)
            self._save()
            return True

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user"""
        with self._lock:
            if role_id in self._user_roles[user_id]:
                self._user_roles[user_id].discard(role_id)
                self._save()
                return True
            return False

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for a user"""
        role_ids = self._user_roles.get(user_id, set())
        roles = []

        for role_id in role_ids:
            role = self._roles.get(role_id)
            if role:
                roles.append(role)
                # Get inherited roles
                roles.extend(self._get_inherited_roles(role))

        # Remove duplicates, sort by priority
        seen = set()
        unique_roles = []
        for role in sorted(roles, key=lambda r: r.priority, reverse=True):
            if role.role_id not in seen:
                seen.add(role.role_id)
                unique_roles.append(role)

        return unique_roles

    def _get_inherited_roles(self, role: Role) -> List[Role]:
        """Get all inherited roles recursively"""
        inherited = []
        for parent_id in role.inherits:
            parent = self._roles.get(parent_id)
            if parent:
                inherited.append(parent)
                inherited.extend(self._get_inherited_roles(parent))
        return inherited


class PermissionManager:
    """Main permission management system"""

    def __init__(self, role_store: RoleStore = None,
                 cache: PermissionCache = None):
        self.role_store = role_store or RoleStore()
        self.cache = cache or PermissionCache()
        self._conditions: Dict[str, PermissionCondition] = {}
        self._permission_handlers: List[Callable] = []
        self._lock = threading.Lock()

    def check_permission(self, user_id: str, resource: str,
                         action: PermissionAction,
                         context: Dict[str, Any] = None,
                         use_cache: bool = True) -> PermissionCheckResult:
        """Check if user has permission for action on resource"""

        # Check cache
        if use_cache:
            cached = self.cache.get(user_id, resource, action.value, context)
            if cached is not None:
                return PermissionCheckResult(
                    granted=cached,
                    reason="Cached result"
                )

        # Get user roles
        roles = self.role_store.get_user_roles(user_id)

        # Check each role's permissions
        for role in roles:
            result = self._check_role_permission(role, resource, action, context)
            if result.granted:
                # Cache result
                if use_cache:
                    self.cache.set(user_id, resource, action.value, True, context)
                return result

        # No permission found
        result = PermissionCheckResult(
            granted=False,
            reason=f"No permission for {action.value} on {resource}"
        )

        if use_cache:
            self.cache.set(user_id, resource, action.value, False, context)

        return result

    def _check_role_permission(self, role: Role, resource: str,
                               action: PermissionAction,
                               context: Dict[str, Any]) -> PermissionCheckResult:
        """Check if role grants permission"""

        for permission in role.permissions:
            # Check if permission matches resource
            if self._resource_matches(permission.resource, resource):
                # Check if permission includes action
                if permission.matches_action(action):
                    # Check conditions
                    if self._check_conditions(permission, context or {}):
                        return PermissionCheckResult(
                            granted=True,
                            permission=permission,
                            role=role,
                            reason=f"Granted by role {role.name}"
                        )

        return PermissionCheckResult(granted=False)

    def _resource_matches(self, pattern: str, resource: str) -> bool:
        """Check if resource matches permission pattern"""
        if pattern == "*":
            return True

        if pattern == resource:
            return True

        # Support wildcard patterns
        if "*" in pattern:
            regex = pattern.replace("*", ".*")
            return bool(re.match(f"^{regex}$", resource))

        # Support resource hierarchy
        if resource.startswith(pattern + "."):
            return True

        return False

    def _check_conditions(self, permission: Permission,
                          context: Dict[str, Any]) -> bool:
        """Check permission conditions"""

        if not permission.conditions:
            return True

        # Check time conditions
        if 'time' in permission.conditions:
            time_cond = permission.conditions['time']
            condition = TimeCondition(
                allowed_hours=tuple(time_cond.get('hours', ())),
                allowed_days=time_cond.get('days')
            )
            if not condition.evaluate(context):
                return False

        # Check IP conditions
        if 'ip' in permission.conditions:
            ip_cond = permission.conditions['ip']
            condition = IPCondition(
                allowed_ips=ip_cond.get('allowed'),
                blocked_ips=ip_cond.get('blocked'),
                allowed_ranges=ip_cond.get('ranges')
            )
            if not condition.evaluate(context):
                return False

        # Check attribute conditions
        if 'attributes' in permission.conditions:
            condition = AttributeCondition(permission.conditions['attributes'])
            if not condition.evaluate(context):
                return False

        return True

    def grant_permission(self, user_id: str, resource: str,
                         action: PermissionAction,
                         role_id: str = None) -> bool:
        """Grant permission to user (via role)"""

        if role_id:
            return self.role_store.assign_role(user_id, role_id)

        # Create temporary role for user-specific permission
        temp_role_id = f"user_{user_id}_temp"
        temp_role = Role(
            role_id=temp_role_id,
            name=f"User {user_id} Permissions",
            permissions={
                Permission(f"perm_{resource}_{action.value}", resource, {action})
            },
            priority=5
        )

        self.role_store.create_role(temp_role)
        return self.role_store.assign_role(user_id, temp_role_id)

    def revoke_permission(self, user_id: str, resource: str,
                          action: PermissionAction) -> bool:
        """Revoke permission from user"""
        # Invalidate cache
        self.cache.invalidate(user_id)

        # Remove from user's temporary roles
        roles = self.role_store.get_user_roles(user_id)
        for role in roles:
            for perm in role.permissions:
                if perm.resource == resource and action in perm.actions:
                    if role.role_id.startswith("user_"):
                        self.role_store.revoke_role(user_id, role.role_id)

        return True

    def create_role(self, name: str, permissions: List[Dict],
                    inherits: List[str] = None,
                    priority: int = 10,
                    description: str = "") -> Optional[Role]:
        """Create a new role"""
        import uuid

        role_id = str(uuid.uuid4())[:8]

        perms = set()
        for p in permissions:
            perm = Permission(
                name=p.get('name', f"perm_{p['resource']}"),
                resource=p['resource'],
                actions={PermissionAction(a) for a in p.get('actions', ['read'])},
                scope=ResourceScope(p.get('scope', 'resource')),
                conditions=p.get('conditions', {}),
                description=p.get('description', '')
            )
            perms.add(perm)

        role = Role(
            role_id=role_id,
            name=name,
            permissions=perms,
            inherits=set(inherits or []),
            priority=priority,
            description=description
        )

        if self.role_store.create_role(role):
            return role
        return None

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all effective permissions for user"""
        roles = self.role_store.get_user_roles(user_id)
        permissions = set()

        for role in roles:
            permissions.update(role.permissions)

        return list(permissions)

    def add_permission_handler(self, handler: Callable) -> None:
        """Add custom permission handler"""
        self._permission_handlers.append(handler)

    def clear_cache(self) -> None:
        """Clear permission cache"""
        self.cache.invalidate()


def require_permission(resource: str, action: PermissionAction):
    """Decorator for permission checking"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get user_id from args or kwargs
            user_id = kwargs.get('user_id') or (args[0] if args else None)

            if not user_id:
                raise PermissionError("No user context provided")

            # Get permission manager
            manager = get_permission_manager()

            # Check permission
            result = manager.check_permission(user_id, resource, action)

            if not result.granted:
                raise PermissionError(f"Permission denied: {result.reason}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role_name: str):
    """Decorator for role checking"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id') or (args[0] if args else None)

            if not user_id:
                raise PermissionError("No user context provided")

            manager = get_permission_manager()
            roles = manager.role_store.get_user_roles(user_id)

            if not any(r.name.lower() == role_name.lower() for r in roles):
                raise PermissionError(f"Role '{role_name}' required")

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global permission manager instance
_permission_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """Get or create global permission manager"""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


# Export classes
__all__ = [
    'PermissionAction',
    'ResourceScope',
    'Permission',
    'Role',
    'PermissionCheckResult',
    'PermissionCondition',
    'TimeCondition',
    'IPCondition',
    'AttributeCondition',
    'PermissionCache',
    'RoleStore',
    'PermissionManager',
    'require_permission',
    'require_role',
    'get_permission_manager'
]


if __name__ == "__main__":
    print("JARVIS Permission Manager")
    print("=" * 50)

    manager = PermissionManager()

    # Test role assignment
    manager.role_store.assign_role("user123", "developer")

    # Test permission check
    result = manager.check_permission(
        user_id="user123",
        resource="code",
        action=PermissionAction.READ
    )

    print(f"Permission check result: {result.granted}")
    print(f"Reason: {result.reason}")

    if result.role:
        print(f"Granted by role: {result.role.name}")

    # List user permissions
    permissions = manager.get_user_permissions("user123")
    print(f"\nUser permissions: {len(permissions)}")
    for perm in permissions[:5]:
        print(f"  - {perm.name}: {perm.resource} ({[a.value for a in perm.actions]})")

    print("\nPermission system ready!")
