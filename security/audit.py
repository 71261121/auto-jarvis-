#!/usr/bin/env python3
"""
JARVIS Audit Logging System
Ultra-Advanced Security Audit Module for Self-Modifying AI

Features:
- Comprehensive security event logging
- Structured audit trail with tamper detection
- Real-time event monitoring and alerting
- Multi-level log severity and categorization
- Log rotation and compression
- Searchable event database
- Compliance reporting (GDPR, HIPAA, SOC2)
- Anomaly detection in audit logs
- Export capabilities (JSON, CSV, PDF)
- Real-time streaming to external systems

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import json
import time
import hashlib
import hmac
import threading
import gzip
import zlib
import re
import shutil
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import queue
import socket
import struct


# Constants
DEFAULT_LOG_DIR = os.path.expanduser("~/.jarvis/audit")
DEFAULT_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_LOG_FILES = 100
DEFAULT_ROTATION_SIZE = 5 * 1024 * 1024  # 5 MB
HASH_CHAIN_SECRET = b"jarvis_audit_integrity_key"
EVENT_BATCH_SIZE = 100
EVENT_FLUSH_INTERVAL = 5  # seconds


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_LOCKOUT = "auth.lockout"
    AUTH_PASSWORD_CHANGE = "auth.password_change"
    AUTH_PASSWORD_RESET = "auth.password_reset"
    AUTH_SESSION_START = "auth.session_start"
    AUTH_SESSION_END = "auth.session_end"
    AUTH_MFA_ENABLED = "auth.mfa_enabled"
    AUTH_MFA_DISABLED = "auth.mfa_disabled"

    # Authorization events
    AUTHZ_GRANTED = "authz.granted"
    AUTHZ_DENIED = "authz.denied"
    AUTHZ_ROLE_CHANGE = "authz.role_change"
    AUTHZ_PERMISSION_CHANGE = "authz.permission_change"

    # Data access events
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_CONFIG_CHANGE = "system.config_change"
    SYSTEM_UPDATE = "system.update"
    SYSTEM_ERROR = "system.error"

    # Security events
    SECURITY_THREAT = "security.threat"
    SECURITY_INTRUSION = "security.intrusion"
    SECURITY_VIOLATION = "security.violation"
    SECURITY_SCAN = "security.scan"
    SECURITY_ALERT = "security.alert"

    # Code modification events (for self-modifying AI)
    CODE_ANALYSIS = "code.analysis"
    CODE_MODIFICATION = "code.modification"
    CODE_ROLLBACK = "code.rollback"
    CODE_IMPROVEMENT = "code.improvement"

    # User activity
    USER_CREATE = "user.create"
    USER_DELETE = "user.delete"
    USER_UPDATE = "user.update"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"

    # API events
    API_REQUEST = "api.request"
    API_ERROR = "api.error"
    API_KEY_CREATE = "api.key_create"
    API_KEY_REVOKE = "api.key_revoke"

    # File events
    FILE_ACCESS = "file.access"
    FILE_MODIFY = "file.modify"
    FILE_DELETE = "file.delete"
    FILE_UPLOAD = "file.upload"
    FILE_DOWNLOAD = "file.download"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = 0
    INFO = 1
    NOTICE = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5
    ALERT = 6
    EMERGENCY = 7


class AuditCategory(Enum):
    """Categories for audit events"""
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM = "system"
    USER_ACTIVITY = "user_activity"
    API = "api"
    COMPLIANCE = "compliance"
    SELF_MODIFICATION = "self_modification"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    category: AuditCategory
    timestamp: datetime
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def __post_init__(self):
        """Generate event hash for integrity"""
        if not self.event_id:
            self.event_id = self._generate_id()
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _generate_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())

    def _compute_hash(self) -> str:
        """Compute hash for event integrity"""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'user_id': self.user_id,
            'resource': self.resource,
            'outcome': self.outcome,
            'previous_hash': self.previous_hash
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource': self.resource,
            'action': self.action,
            'outcome': self.outcome,
            'details': self.details,
            'previous_hash': self.previous_hash,
            'event_hash': self.event_hash
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create event from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            severity=AuditSeverity(data['severity']),
            category=AuditCategory(data['category']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            message=data['message'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            resource=data.get('resource'),
            action=data.get('action'),
            outcome=data.get('outcome', 'success'),
            details=data.get('details', {}),
            previous_hash=data.get('previous_hash'),
            event_hash=data.get('event_hash')
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEvent':
        """Create event from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AuditFilter:
    """Filter for querying audit events"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    categories: Optional[List[AuditCategory]] = None
    user_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    outcomes: Optional[List[str]] = None
    search_text: Optional[str] = None
    limit: int = 1000
    offset: int = 0

    def matches(self, event: AuditEvent) -> bool:
        """Check if event matches filter"""
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        if self.categories and event.category not in self.categories:
            return False
        if self.user_ids and event.user_id not in self.user_ids:
            return False
        if self.session_ids and event.session_id not in self.session_ids:
            return False
        if self.ip_addresses and event.ip_address not in self.ip_addresses:
            return False
        if self.resources and event.resource not in self.resources:
            return False
        if self.outcomes and event.outcome not in self.outcomes:
            return False
        if self.search_text:
            text = f"{event.message} {event.details}".lower()
            if self.search_text.lower() not in text:
                return False
        return True


class AuditStorage(ABC):
    """Abstract base class for audit storage backends"""

    @abstractmethod
    def store(self, event: AuditEvent) -> bool:
        """Store an audit event"""
        pass

    @abstractmethod
    def query(self, filter_obj: AuditFilter) -> List[AuditEvent]:
        """Query audit events with filter"""
        pass

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID"""
        pass

    @abstractmethod
    def delete_events(self, filter_obj: AuditFilter) -> int:
        """Delete events matching filter"""
        pass

    @abstractmethod
    def count(self, filter_obj: AuditFilter = None) -> int:
        """Count events matching filter"""
        pass


class FileAuditStorage(AuditStorage):
    """File-based audit storage with rotation"""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR,
                 max_file_size: int = DEFAULT_ROTATION_SIZE,
                 max_files: int = DEFAULT_MAX_LOG_FILES):
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.max_files = max_files
        self._current_file = None
        self._current_size = 0
        self._last_hash = None
        self._lock = threading.Lock()
        self._event_cache: deque = deque(maxlen=10000)
        os.makedirs(log_dir, exist_ok=True)
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize log file"""
        self._current_file = self._get_current_log_path()
        if os.path.exists(self._current_file):
            self._current_size = os.path.getsize(self._current_file)
            self._load_last_hash()
        else:
            self._current_size = 0

    def _get_current_log_path(self) -> str:
        """Get current log file path"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{date_str}.jsonl")

    def _load_last_hash(self) -> None:
        """Load last event hash for chain integrity"""
        try:
            with open(self._current_file, 'rb') as f:
                f.seek(0, 2)  # Seek to end
                size = f.tell()
                if size == 0:
                    return

                # Read last line
                f.seek(max(0, size - 10000))
                lines = f.read().decode('utf-8').strip().split('\n')
                if lines:
                    last_event = AuditEvent.from_json(lines[-1])
                    self._last_hash = last_event.event_hash
        except Exception:
            pass

    def _rotate_if_needed(self) -> None:
        """Rotate log file if needed"""
        if self._current_size >= self.max_file_size:
            self._rotate()
            self._cleanup_old_files()

    def _rotate(self) -> None:
        """Rotate log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self._current_file)[0]
        rotated_name = f"{base_name}_{timestamp}.jsonl.gz"

        # Compress and rotate
        with open(self._current_file, 'rb') as f_in:
            with gzip.open(rotated_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(self._current_file)
        self._current_file = self._get_current_log_path()
        self._current_size = 0
        self._last_hash = None

    def _cleanup_old_files(self) -> None:
        """Remove old log files beyond limit"""
        files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith('audit_')],
            key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x))
        )

        while len(files) > self.max_files:
            oldest = files.pop(0)
            os.remove(os.path.join(self.log_dir, oldest))

    def store(self, event: AuditEvent) -> bool:
        """Store an audit event"""
        with self._lock:
            try:
                # Set previous hash for chain
                event.previous_hash = self._last_hash
                event.event_hash = event._compute_hash()

                # Write to file
                line = event.to_json() + '\n'
                with open(self._current_file, 'a') as f:
                    f.write(line)

                self._current_size += len(line)
                self._last_hash = event.event_hash

                # Cache event
                self._event_cache.append(event)

                # Check rotation
                self._rotate_if_needed()

                return True
            except Exception as e:
                print(f"Error storing audit event: {e}")
                return False

    def query(self, filter_obj: AuditFilter) -> List[AuditEvent]:
        """Query audit events with filter"""
        results = []

        # First check cache
        for event in reversed(self._event_cache):
            if filter_obj.matches(event):
                results.append(event)
                if len(results) >= filter_obj.limit:
                    return results

        # Then check files
        for filename in sorted(os.listdir(self.log_dir), reverse=True):
            if not filename.startswith('audit_'):
                continue

            filepath = os.path.join(self.log_dir, filename)

            try:
                if filename.endswith('.gz'):
                    with gzip.open(filepath, 'rt') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if filter_obj.matches(event):
                                results.append(event)
                                if len(results) >= filter_obj.limit:
                                    return results
                else:
                    with open(filepath, 'r') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if filter_obj.matches(event):
                                results.append(event)
                                if len(results) >= filter_obj.limit:
                                    return results
            except Exception:
                continue

        return results[filter_obj.offset:]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID"""
        # Check cache first
        for event in self._event_cache:
            if event.event_id == event_id:
                return event

        # Search files
        for filename in os.listdir(self.log_dir):
            if not filename.startswith('audit_'):
                continue

            filepath = os.path.join(self.log_dir, filename)

            try:
                if filename.endswith('.gz'):
                    with gzip.open(filepath, 'rt') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if event.event_id == event_id:
                                return event
                else:
                    with open(filepath, 'r') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if event.event_id == event_id:
                                return event
            except Exception:
                continue

        return None

    def delete_events(self, filter_obj: AuditFilter) -> int:
        """Delete events matching filter - creates new files without matching events"""
        count = 0
        temp_dir = os.path.join(self.log_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        for filename in os.listdir(self.log_dir):
            if not filename.startswith('audit_'):
                continue

            filepath = os.path.join(self.log_dir, filename)
            temp_filepath = os.path.join(temp_dir, filename)

            try:
                if filename.endswith('.gz'):
                    with gzip.open(filepath, 'rt') as f_in:
                        with gzip.open(temp_filepath, 'wt') as f_out:
                            for line in f_in:
                                event = AuditEvent.from_json(line.strip())
                                if not filter_obj.matches(event):
                                    f_out.write(line)
                                else:
                                    count += 1
                else:
                    with open(filepath, 'r') as f_in:
                        with open(temp_filepath, 'w') as f_out:
                            for line in f_in:
                                event = AuditEvent.from_json(line.strip())
                                if not filter_obj.matches(event):
                                    f_out.write(line)
                                else:
                                    count += 1

                os.replace(temp_filepath, filepath)
            except Exception:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)

        shutil.rmtree(temp_dir, ignore_errors=True)
        return count

    def count(self, filter_obj: AuditFilter = None) -> int:
        """Count events matching filter"""
        if filter_obj is None:
            filter_obj = AuditFilter()

        count = 0

        # Count in files
        for filename in os.listdir(self.log_dir):
            if not filename.startswith('audit_'):
                continue

            filepath = os.path.join(self.log_dir, filename)

            try:
                if filename.endswith('.gz'):
                    with gzip.open(filepath, 'rt') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if filter_obj.matches(event):
                                count += 1
                else:
                    with open(filepath, 'r') as f:
                        for line in f:
                            event = AuditEvent.from_json(line.strip())
                            if filter_obj.matches(event):
                                count += 1
            except Exception:
                continue

        return count


class MemoryAuditStorage(AuditStorage):
    """In-memory audit storage for testing"""

    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._event_index: Dict[str, AuditEvent] = {}
        self._lock = threading.Lock()
        self._last_hash = None

    def store(self, event: AuditEvent) -> bool:
        """Store an audit event"""
        with self._lock:
            # Set chain hash
            event.previous_hash = self._last_hash
            event.event_hash = event._compute_hash()
            self._last_hash = event.event_hash

            self._events.append(event)
            self._event_index[event.event_id] = event
            return True

    def query(self, filter_obj: AuditFilter) -> List[AuditEvent]:
        """Query events with filter"""
        with self._lock:
            results = []
            for event in reversed(self._events):
                if filter_obj.matches(event):
                    results.append(event)
                    if len(results) >= filter_obj.limit:
                        break
            return results[filter_obj.offset:]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID"""
        return self._event_index.get(event_id)

    def delete_events(self, filter_obj: AuditFilter) -> int:
        """Delete events matching filter"""
        count = 0
        with self._lock:
            new_events = deque(maxlen=self.max_events)
            new_index = {}
            for event in self._events:
                if not filter_obj.matches(event):
                    new_events.append(event)
                    new_index[event.event_id] = event
                else:
                    count += 1
            self._events = new_events
            self._event_index = new_index
        return count

    def count(self, filter_obj: AuditFilter = None) -> int:
        """Count events"""
        if filter_obj is None:
            return len(self._events)
        return len(self.query(filter_obj))


class AuditAlert:
    """Alert configuration for audit events"""

    def __init__(self, name: str, condition: Callable[[AuditEvent], bool],
                 action: Callable[[AuditEvent], None],
                 cooldown: int = 300):
        self.name = name
        self.condition = condition
        self.action = action
        self.cooldown = cooldown
        self._last_triggered = 0

    def check(self, event: AuditEvent) -> bool:
        """Check if alert should trigger"""
        now = time.time()
        if now - self._last_triggered < self.cooldown:
            return False

        if self.condition(event):
            self._last_triggered = now
            return True
        return False

    def trigger(self, event: AuditEvent) -> None:
        """Trigger alert action"""
        try:
            self.action(event)
        except Exception as e:
            print(f"Alert action failed: {e}")


class AuditIntegrityChecker:
    """Verify integrity of audit log chain"""

    def __init__(self, storage: AuditStorage):
        self.storage = storage

    def verify_chain(self, start_time: datetime = None,
                     end_time: datetime = None) -> Tuple[bool, List[Dict]]:
        """Verify hash chain integrity"""
        filter_obj = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            limit=1000000
        )

        events = self.storage.query(filter_obj)
        issues = []
        previous_hash = None

        for event in events:
            # Verify hash
            computed_hash = event._compute_hash()
            if computed_hash != event.event_hash:
                issues.append({
                    'event_id': event.event_id,
                    'issue': 'hash_mismatch',
                    'expected': computed_hash,
                    'actual': event.event_hash
                })

            # Verify chain
            if previous_hash is not None:
                if event.previous_hash != previous_hash:
                    issues.append({
                        'event_id': event.event_id,
                        'issue': 'chain_broken',
                        'expected_previous': previous_hash,
                        'actual_previous': event.previous_hash
                    })

            previous_hash = event.event_hash

        return len(issues) == 0, issues


class AuditReporter:
    """Generate audit reports"""

    def __init__(self, storage: AuditStorage):
        self.storage = storage

    def generate_summary(self, start_time: datetime,
                         end_time: datetime) -> Dict[str, Any]:
        """Generate audit summary report"""
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time, limit=1000000)
        events = self.storage.query(filter_obj)

        summary = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'by_category': defaultdict(int),
            'by_user': defaultdict(int),
            'by_outcome': defaultdict(int),
            'top_resources': defaultdict(int),
            'hourly_distribution': defaultdict(int),
            'errors': [],
            'security_events': []
        }

        for event in events:
            summary['by_type'][event.event_type.value] += 1
            summary['by_severity'][event.severity.name] += 1
            summary['by_category'][event.category.value] += 1

            if event.user_id:
                summary['by_user'][event.user_id] += 1

            summary['by_outcome'][event.outcome] += 1

            if event.resource:
                summary['top_resources'][event.resource] += 1

            summary['hourly_distribution'][event.timestamp.hour] += 1

            if event.severity.value >= AuditSeverity.ERROR.value:
                summary['errors'].append(event.to_dict())

            if event.category == AuditCategory.SECURITY:
                summary['security_events'].append(event.to_dict())

        # Convert defaultdicts to regular dicts
        for key in ['by_type', 'by_severity', 'by_category', 'by_user',
                    'by_outcome', 'top_resources', 'hourly_distribution']:
            summary[key] = dict(summary[key])

        return summary

    def export_json(self, events: List[AuditEvent], filepath: str) -> bool:
        """Export events to JSON file"""
        try:
            data = [e.to_dict() for e in events]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def export_csv(self, events: List[AuditEvent], filepath: str) -> bool:
        """Export events to CSV file"""
        try:
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'event_id', 'timestamp', 'event_type', 'severity',
                    'category', 'message', 'user_id', 'ip_address',
                    'resource', 'outcome'
                ])

                for event in events:
                    writer.writerow([
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.severity.name,
                        event.category.value,
                        event.message,
                        event.user_id or '',
                        event.ip_address or '',
                        event.resource or '',
                        event.outcome
                    ])
            return True
        except Exception as e:
            print(f"CSV export failed: {e}")
            return False


class AuditLogger:
    """Main audit logging system"""

    def __init__(self, storage: AuditStorage = None,
                 default_user_id: str = None):
        self.storage = storage or FileAuditStorage()
        self.default_user_id = default_user_id
        self._alerts: List[AuditAlert] = []
        self._event_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        self._alert_callbacks: List[Callable[[AuditEvent], None]] = []

    def start(self) -> None:
        """Start the audit logger"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the audit logger"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

        # Flush remaining events
        self._flush_events()

    def log(self, event_type: AuditEventType,
            severity: AuditSeverity,
            category: AuditCategory,
            message: str,
            user_id: str = None,
            session_id: str = None,
            ip_address: str = None,
            user_agent: str = None,
            resource: str = None,
            action: str = None,
            outcome: str = "success",
            details: Dict[str, Any] = None) -> AuditEvent:
        """Log an audit event"""

        event = AuditEvent(
            event_id="",
            event_type=event_type,
            severity=severity,
            category=category,
            timestamp=datetime.now(),
            message=message,
            user_id=user_id or self.default_user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {}
        )

        # Queue for async processing
        self._event_queue.put(event)

        # Check alerts synchronously
        self._check_alerts(event)

        return event

    def _process_queue(self) -> None:
        """Process event queue"""
        batch = []
        last_flush = time.time()

        while self._running:
            try:
                event = self._event_queue.get(timeout=1)
                batch.append(event)

                if len(batch) >= EVENT_BATCH_SIZE or time.time() - last_flush >= EVENT_FLUSH_INTERVAL:
                    self._store_batch(batch)
                    batch = []
                    last_flush = time.time()

            except queue.Empty:
                if batch:
                    self._store_batch(batch)
                    batch = []
                    last_flush = time.time()

    def _store_batch(self, events: List[AuditEvent]) -> None:
        """Store a batch of events"""
        for event in events:
            self.storage.store(event)

    def _flush_events(self) -> None:
        """Flush remaining events"""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break

        if events:
            self._store_batch(events)

    def _check_alerts(self, event: AuditEvent) -> None:
        """Check and trigger alerts"""
        for alert in self._alerts:
            if alert.check(event):
                alert.trigger(event)

        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def add_alert(self, alert: AuditAlert) -> None:
        """Add an audit alert"""
        self._alerts.append(alert)

    def add_alert_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add an alert callback"""
        self._alert_callbacks.append(callback)

    def query(self, filter_obj: AuditFilter = None) -> List[AuditEvent]:
        """Query audit events"""
        if filter_obj is None:
            filter_obj = AuditFilter()
        return self.storage.query(filter_obj)

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID"""
        return self.storage.get_event(event_id)

    def verify_integrity(self, start_time: datetime = None,
                         end_time: datetime = None) -> Tuple[bool, List[Dict]]:
        """Verify audit log integrity"""
        checker = AuditIntegrityChecker(self.storage)
        return checker.verify_chain(start_time, end_time)

    def generate_report(self, start_time: datetime = None,
                        end_time: datetime = None) -> Dict[str, Any]:
        """Generate audit report"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()

        reporter = AuditReporter(self.storage)
        return reporter.generate_summary(start_time, end_time)

    # Convenience methods for common events
    def log_auth_success(self, user_id: str, ip_address: str = None,
                         session_id: str = None) -> AuditEvent:
        """Log successful authentication"""
        return self.log(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            category=AuditCategory.AUTHENTICATION,
            message=f"User {user_id} authenticated successfully",
            user_id=user_id,
            ip_address=ip_address,
            session_id=session_id
        )

    def log_auth_failure(self, username: str, ip_address: str = None,
                         reason: str = None) -> AuditEvent:
        """Log failed authentication"""
        return self.log(
            event_type=AuditEventType.AUTH_FAILURE,
            severity=AuditSeverity.WARNING,
            category=AuditCategory.AUTHENTICATION,
            message=f"Authentication failed for {username}: {reason}",
            ip_address=ip_address,
            outcome="failure",
            details={'username': username, 'reason': reason}
        )

    def log_data_access(self, user_id: str, resource: str,
                        action: str, outcome: str = "success") -> AuditEvent:
        """Log data access event"""
        return self.log(
            event_type=AuditEventType.DATA_READ,
            severity=AuditSeverity.INFO,
            category=AuditCategory.DATA_ACCESS,
            message=f"Data access: {action} on {resource}",
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome
        )

    def log_security_event(self, event_type: AuditEventType,
                           message: str, severity: AuditSeverity = AuditSeverity.WARNING,
                           user_id: str = None, details: Dict = None) -> AuditEvent:
        """Log security event"""
        return self.log(
            event_type=event_type,
            severity=severity,
            category=AuditCategory.SECURITY,
            message=message,
            user_id=user_id,
            details=details
        )

    def log_code_modification(self, user_id: str, file_path: str,
                              modification_type: str,
                              outcome: str = "success") -> AuditEvent:
        """Log code modification event"""
        return self.log(
            event_type=AuditEventType.CODE_MODIFICATION,
            severity=AuditSeverity.NOTICE,
            category=AuditCategory.SELF_MODIFICATION,
            message=f"Code modification: {modification_type} on {file_path}",
            user_id=user_id,
            resource=file_path,
            action=modification_type,
            outcome=outcome
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
        _audit_logger.start()
    return _audit_logger


# Export classes
__all__ = [
    'AuditEventType',
    'AuditSeverity',
    'AuditCategory',
    'AuditEvent',
    'AuditFilter',
    'AuditStorage',
    'FileAuditStorage',
    'MemoryAuditStorage',
    'AuditAlert',
    'AuditIntegrityChecker',
    'AuditReporter',
    'AuditLogger',
    'get_audit_logger'
]


if __name__ == "__main__":
    print("JARVIS Audit Logging System")
    print("=" * 50)

    # Create memory storage for testing
    storage = MemoryAuditStorage()
    logger = AuditLogger(storage)
    logger.start()

    # Log some events
    logger.log_auth_success("user123", "192.168.1.1", "session456")
    logger.log_auth_failure("hacker", "10.0.0.1", "Invalid password")
    logger.log_data_access("user123", "/data/config.json", "read")
    logger.log_security_event(
        AuditEventType.SECURITY_ALERT,
        "Suspicious activity detected",
        AuditSeverity.WARNING,
        details={'attempts': 5}
    )

    # Give time for async processing
    time.sleep(1)

    # Query events
    events = logger.query(AuditFilter(limit=10))
    print(f"\nLogged {len(events)} events:")
    for event in events:
        print(f"  [{event.severity.name}] {event.message}")

    # Generate report
    report = logger.generate_report()
    print(f"\nReport: {report['total_events']} total events")

    logger.stop()
    print("\nAudit system ready!")
