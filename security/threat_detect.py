#!/usr/bin/env python3
"""
JARVIS Threat Detection System
Ultra-Advanced Security Threat Detection for Self-Modifying AI

Features:
- Real-time threat detection and analysis
- Pattern-based threat identification
- Anomaly detection algorithms
- Behavioral analysis
- Attack signature matching
- Intrusion detection
- Malicious code detection
- Rate-based attack detection
- IP reputation checking
- Threat scoring and classification
- Automated response triggers
- Threat intelligence integration

Author: JARVIS Self-Modifying AI Project
Version: 1.0.0
"""

import os
import sys
import re
import json
import time
import hashlib
import threading
import ipaddress
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Callable, Union, Set, Pattern
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import statistics


# Constants
THREAT_SCORE_LOW = 25
THREAT_SCORE_MEDIUM = 50
THREAT_SCORE_HIGH = 75
THREAT_SCORE_CRITICAL = 90
DEFAULT_ANALYSIS_WINDOW = 300  # 5 minutes
MAX_EVENTS_IN_MEMORY = 10000


class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    DENIAL_OF_SERVICE = "dos"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_CODE = "malicious_code"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    IP_REPUTATION = "ip_reputation"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    SESSION_HIJACKING = "session_hijacking"
    MAN_IN_MIDDLE = "mitm"
    REPLAY_ATTACK = "replay_attack"
    ZERO_DAY = "zero_day"


class ThreatSeverity(Enum):
    """Severity levels for threats"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ThreatStatus(Enum):
    """Status of detected threats"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


@dataclass
class ThreatIndicator:
    """Individual threat indicator"""
    indicator_type: str
    value: str
    confidence: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Threat:
    """Detected threat"""
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    status: ThreatStatus
    score: int  # 0-100
    confidence: float  # 0.0 to 1.0
    detected_at: datetime
    source_ip: Optional[str] = None
    target_resource: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    description: str = ""
    indicators: List[ThreatIndicator] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type.value,
            'severity': self.severity.name,
            'status': self.status.value,
            'score': self.score,
            'confidence': self.confidence,
            'detected_at': self.detected_at.isoformat(),
            'source_ip': self.source_ip,
            'target_resource': self.target_resource,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'description': self.description,
            'indicators': [
                {
                    'type': i.indicator_type,
                    'value': i.value,
                    'confidence': i.confidence
                } for i in self.indicators
            ],
            'affected_resources': self.affected_resources,
            'recommended_actions': self.recommended_actions,
            'timeline': self.timeline,
            'metadata': self.metadata
        }


@dataclass
class ThreatPattern:
    """Pattern definition for threat detection"""
    name: str
    threat_type: ThreatType
    severity: ThreatSeverity
    patterns: List[str]  # Regex patterns
    fields: List[str]  # Fields to check
    min_occurrences: int = 1
    time_window: int = DEFAULT_ANALYSIS_WINDOW
    score_weight: float = 1.0
    description: str = ""


class AttackSignatureDatabase:
    """Database of known attack signatures"""

    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"('|\")(;|--|\)|union|select|insert|delete|update|drop|exec)",
        r"(union\s+(all\s+)?select)",
        r"(select\s+.+\s+from)",
        r"(insert\s+into\s+.+\s+values)",
        r"(delete\s+from)",
        r"(drop\s+(table|database|column))",
        r"(update\s+.+\s+set)",
        r"(exec(\s*\(|\s+))",
        r"(or\s+1\s*=\s*1)",
        r"(or\s+'[^']*'\s*=\s*'[^']*')",
        r"(;\s*--)",
        r"(/\*.*\*/)",
        r"(waitfor\s+delay)",
        r"(benchmark\s*\()",
        r"(sleep\s*\()",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"on(error|load|click|mouse|focus|blur)\s*=",
        r"<img[^>]+onerror\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<svg[^>]*onload\s*=",
        r"document\.(cookie|location|write)",
        r"eval\s*\(",
        r"expression\s*\(",
        r"vbscript\s*:",
    ]

    # Command Injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\([^)]+\)",
        r"`[^`]+`",
        r"\|\s*\w+",
        r";\s*\w+",
        r"&&\s*\w+",
        r"\|\|\s*\w+",
        r">\s*/",
        r"<\s*/",
        r"2>&1",
        r"/bin/(bash|sh|zsh|ksh)",
        r"(nc|netcat|ncat)\s+",
        r"(wget|curl)\s+",
        r"(chmod|chown|chgrp)\s+",
    ]

    # Path Traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e[/\\]",
        r"%252e%252e",
        r"\.\.%2f",
        r"\.\.%5c",
        r"/etc/passwd",
        r"/etc/shadow",
        r"/proc/self",
        r"c:\\windows",
        r"\\\\",
    ]

    # Malicious code patterns
    MALICIOUS_CODE_PATTERNS = [
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"__import__\s*\(",
        r"globals\s*\(\s*\)",
        r"locals\s*\(\s*\)",
        r"vars\s*\(\s*\)",
        r"dir\s*\(\s*\)",
        r"getattr\s*\(",
        r"setattr\s*\(",
        r"delattr\s*\(",
        r"hasattr\s*\(",
        r"__class__",
        r"__bases__",
        r"__subclasses__",
        r"__mro__",
        r"__code__",
        r"__globals__",
        r"os\.system",
        r"subprocess\.",
        r"pickle\.loads",
        r"marshal\.loads",
    ]

    def __init__(self):
        self._compiled_patterns: Dict[str, List[Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns"""
        pattern_groups = {
            'sql_injection': self.SQL_INJECTION_PATTERNS,
            'xss': self.XSS_PATTERNS,
            'command_injection': self.COMMAND_INJECTION_PATTERNS,
            'path_traversal': self.PATH_TRAVERSAL_PATTERNS,
            'malicious_code': self.MALICIOUS_CODE_PATTERNS,
        }

        for group_name, patterns in pattern_groups.items():
            self._compiled_patterns[group_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check_sql_injection(self, data: str) -> List[Tuple[str, str]]:
        """Check for SQL injection patterns"""
        matches = []
        for pattern in self._compiled_patterns.get('sql_injection', []):
            found = pattern.findall(data)
            if found:
                matches.append((pattern.pattern, str(found)))
        return matches

    def check_xss(self, data: str) -> List[Tuple[str, str]]:
        """Check for XSS patterns"""
        matches = []
        for pattern in self._compiled_patterns.get('xss', []):
            found = pattern.findall(data)
            if found:
                matches.append((pattern.pattern, str(found)))
        return matches

    def check_command_injection(self, data: str) -> List[Tuple[str, str]]:
        """Check for command injection patterns"""
        matches = []
        for pattern in self._compiled_patterns.get('command_injection', []):
            found = pattern.findall(data)
            if found:
                matches.append((pattern.pattern, str(found)))
        return matches

    def check_path_traversal(self, data: str) -> List[Tuple[str, str]]:
        """Check for path traversal patterns"""
        matches = []
        for pattern in self._compiled_patterns.get('path_traversal', []):
            found = pattern.findall(data)
            if found:
                matches.append((pattern.pattern, str(found)))
        return matches

    def check_malicious_code(self, data: str) -> List[Tuple[str, str]]:
        """Check for malicious code patterns"""
        matches = []
        for pattern in self._compiled_patterns.get('malicious_code', []):
            found = pattern.findall(data)
            if found:
                matches.append((pattern.pattern, str(found)))
        return matches

    def analyze(self, data: str) -> Dict[str, List[Tuple[str, str]]]:
        """Analyze data for all attack types"""
        return {
            'sql_injection': self.check_sql_injection(data),
            'xss': self.check_xss(data),
            'command_injection': self.check_command_injection(data),
            'path_traversal': self.check_path_traversal(data),
            'malicious_code': self.check_malicious_code(data),
        }


class IPReputationChecker:
    """Check IP address reputation"""

    # Known malicious IP ranges (example - should use threat intelligence feeds)
    KNOWN_MALICIOUS_RANGES = [
        # Tor exit nodes (example)
        "185.220.101.0/24",
        "198.96.155.0/24",
        # Known VPN/proxy ranges (example)
        "45.33.0.0/16",
        # Private/loopback (internal testing)
        "10.0.0.0/8",
        "127.0.0.0/8",
    ]

    # Suspicious countries (configurable)
    HIGH_RISK_COUNTRIES = set()

    def __init__(self):
        self._ip_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def check_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation"""
        with self._lock:
            if ip_address in self._ip_cache:
                cached = self._ip_cache[ip_address]
                if datetime.now() - cached['checked_at'] < timedelta(hours=24):
                    return cached

        result = {
            'ip': ip_address,
            'checked_at': datetime.now(),
            'is_malicious': False,
            'is_suspicious': False,
            'is_proxy': False,
            'is_tor': False,
            'is_vpn': False,
            'is_private': False,
            'risk_score': 0,
            'threats': [],
            'country': None,
            'asn': None,
        }

        try:
            ip_obj = ipaddress.ip_address(ip_address)

            # Check if private
            if ip_obj.is_private:
                result['is_private'] = True
                result['risk_score'] = 5

            # Check against known malicious ranges
            for range_str in self.KNOWN_MALICIOUS_RANGES:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip_obj in network:
                    result['is_malicious'] = True
                    result['risk_score'] = 80
                    result['threats'].append(f"Known malicious range: {range_str}")

            # Check for Tor/VPN patterns
            if self._is_tor_exit(ip_address):
                result['is_tor'] = True
                result['is_proxy'] = True
                result['risk_score'] = max(result['risk_score'], 40)
                result['threats'].append("Tor exit node")

        except ValueError:
            result['risk_score'] = 50
            result['threats'].append("Invalid IP format")

        with self._lock:
            self._ip_cache[ip_address] = result

        return result

    def _is_tor_exit(self, ip: str) -> bool:
        """Check if IP is a Tor exit node (simplified)"""
        # In production, would check against Tor exit node list
        tor_indicators = ['185.220.101', '198.96.155']
        return any(ip.startswith(ind) for ind in tor_indicators)

    def get_cached_reputation(self, ip: str) -> Optional[Dict]:
        """Get cached reputation"""
        with self._lock:
            return self._ip_cache.get(ip)


class AnomalyDetector:
    """Detect anomalous behavior"""

    def __init__(self, analysis_window: int = DEFAULT_ANALYSIS_WINDOW):
        self.analysis_window = analysis_window
        self._event_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_EVENTS_IN_MEMORY))
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def record_event(self, identifier: str, event_type: str,
                     timestamp: datetime = None, value: float = 1.0) -> None:
        """Record an event for analysis"""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self._event_history[identifier].append({
                'type': event_type,
                'timestamp': timestamp,
                'value': value
            })

    def calculate_baseline(self, identifier: str) -> Dict[str, float]:
        """Calculate baseline statistics for identifier"""
        events = list(self._event_history.get(identifier, []))

        if not events:
            return {'mean': 0, 'std': 0, 'count': 0}

        values = [e['value'] for e in events]
        timestamps = [e['timestamp'] for e in events]

        # Calculate rate (events per minute)
        if len(timestamps) > 1:
            time_diff = (timestamps[-1] - timestamps[0]).total_seconds() / 60
            rate = len(events) / max(time_diff, 1)
        else:
            rate = len(events)

        mean = statistics.mean(values) if values else 0
        std = statistics.stdev(values) if len(values) > 1 else 0

        baseline = {
            'mean': mean,
            'std': std,
            'count': len(events),
            'rate_per_minute': rate
        }

        with self._lock:
            self._baselines[identifier] = baseline

        return baseline

    def detect_anomaly(self, identifier: str, value: float,
                       threshold_std: float = 3.0) -> Tuple[bool, float, Dict]:
        """Detect if value is anomalous"""
        baseline = self._baselines.get(identifier)

        if baseline is None or baseline['count'] < 10:
            baseline = self.calculate_baseline(identifier)
            if baseline['count'] < 10:
                return False, 0.0, baseline

        mean = baseline['mean']
        std = baseline['std'] if baseline['std'] > 0 else 1.0

        z_score = abs(value - mean) / std
        is_anomaly = z_score > threshold_std

        return is_anomaly, z_score, baseline

    def detect_rate_anomaly(self, identifier: str,
                            window_seconds: int = 60,
                            threshold: float = 10.0) -> Tuple[bool, float]:
        """Detect rate-based anomaly"""
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)

        events = self._event_history.get(identifier, [])
        recent_count = sum(1 for e in events if e['timestamp'] >= window_start)

        rate = recent_count / (window_seconds / 60)  # Events per minute

        baseline = self._baselines.get(identifier, {})
        baseline_rate = baseline.get('rate_per_minute', 5)

        is_anomaly = rate > baseline_rate * threshold

        return is_anomaly, rate


class BruteForceDetector:
    """Detect brute force attacks"""

    def __init__(self, threshold: int = 5, window_seconds: int = 300):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._attempts: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.Lock()

    def record_attempt(self, identifier: str, success: bool,
                       timestamp: datetime = None) -> None:
        """Record an authentication attempt"""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self._attempts[identifier].append({
                'timestamp': timestamp,
                'success': success
            })

            # Clean old attempts
            cutoff = timestamp - timedelta(seconds=self.window_seconds)
            while self._attempts[identifier] and self._attempts[identifier][0]['timestamp'] < cutoff:
                self._attempts[identifier].popleft()

    def is_brute_force(self, identifier: str) -> Tuple[bool, int, int]:
        """Check if brute force attack detected"""
        with self._lock:
            attempts = list(self._attempts.get(identifier, []))

        if not attempts:
            return False, 0, 0

        failed_count = sum(1 for a in attempts if not a['success'])
        total_count = len(attempts)

        is_attack = failed_count >= self.threshold

        return is_attack, failed_count, total_count

    def get_failed_rate(self, identifier: str) -> float:
        """Get failed attempt rate"""
        with self._lock:
            attempts = list(self._attempts.get(identifier, []))

        if not attempts:
            return 0.0

        failed = sum(1 for a in attempts if not a['success'])
        return failed / len(attempts)


class ThreatScorer:
    """Calculate threat scores"""

    @staticmethod
    def calculate_score(threat_type: ThreatType, confidence: float,
                        severity: ThreatSeverity, indicators: List[ThreatIndicator]) -> int:
        """Calculate overall threat score"""
        base_scores = {
            ThreatType.BRUTE_FORCE: 30,
            ThreatType.CREDENTIAL_STUFFING: 35,
            ThreatType.SQL_INJECTION: 70,
            ThreatType.XSS: 50,
            ThreatType.COMMAND_INJECTION: 80,
            ThreatType.PATH_TRAVERSAL: 60,
            ThreatType.DENIAL_OF_SERVICE: 75,
            ThreatType.DATA_EXFILTRATION: 85,
            ThreatType.UNAUTHORIZED_ACCESS: 65,
            ThreatType.PRIVILEGE_ESCALATION: 80,
            ThreatType.MALICIOUS_CODE: 85,
            ThreatType.ANOMALOUS_BEHAVIOR: 40,
            ThreatType.SUSPICIOUS_PATTERN: 35,
            ThreatType.IP_REPUTATION: 25,
            ThreatType.RATE_LIMIT_ABUSE: 45,
            ThreatType.SESSION_HIJACKING: 70,
            ThreatType.MAN_IN_MIDDLE: 75,
            ThreatType.REPLAY_ATTACK: 55,
            ThreatType.ZERO_DAY: 90,
        }

        severity_multipliers = {
            ThreatSeverity.INFO: 0.5,
            ThreatSeverity.LOW: 0.7,
            ThreatSeverity.MEDIUM: 1.0,
            ThreatSeverity.HIGH: 1.3,
            ThreatSeverity.CRITICAL: 1.6,
            ThreatSeverity.EMERGENCY: 2.0,
        }

        base_score = base_scores.get(threat_type, 50)
        severity_mult = severity_multipliers.get(severity, 1.0)

        # Factor in confidence
        confidence_factor = confidence

        # Factor in indicators
        indicator_count = len(indicators)
        indicator_bonus = min(indicator_count * 2, 10)

        # Calculate final score
        score = int(base_score * severity_mult * confidence_factor) + indicator_bonus

        return min(100, max(0, score))

    @staticmethod
    def classify_severity(score: int) -> ThreatSeverity:
        """Classify threat severity based on score"""
        if score >= 90:
            return ThreatSeverity.EMERGENCY
        elif score >= 75:
            return ThreatSeverity.CRITICAL
        elif score >= 60:
            return ThreatSeverity.HIGH
        elif score >= 40:
            return ThreatSeverity.MEDIUM
        elif score >= 20:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO


class ThreatDetector:
    """Main threat detection system"""

    def __init__(self):
        self.signature_db = AttackSignatureDatabase()
        self.ip_checker = IPReputationChecker()
        self.anomaly_detector = AnomalyDetector()
        self.brute_force_detector = BruteForceDetector()
        self._threats: Dict[str, Threat] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[Threat], None]] = []

    def analyze_request(self, data: Dict[str, Any]) -> List[Threat]:
        """Analyze a request for threats"""
        threats = []

        # Get request data
        ip = data.get('ip_address', '')
        user_id = data.get('user_id', '')
        endpoint = data.get('endpoint', '')
        method = data.get('method', 'GET')
        params = data.get('params', {})
        headers = data.get('headers', {})
        body = data.get('body', '')

        # Check IP reputation
        ip_reputation = self.ip_checker.check_reputation(ip)
        if ip_reputation['risk_score'] > 50:
            threat = self._create_threat(
                threat_type=ThreatType.IP_REPUTATION,
                source_ip=ip,
                description=f"Suspicious IP detected: {ip}",
                indicators=[
                    ThreatIndicator(
                        indicator_type='ip',
                        value=ip,
                        confidence=ip_reputation['risk_score'] / 100,
                        source='ip_reputation',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    )
                ]
            )
            threats.append(threat)

        # Check for attack patterns in all input
        all_input = ' '.join([
            str(params),
            str(headers),
            str(body),
            endpoint
        ])

        attack_results = self.signature_db.analyze(all_input)

        if attack_results['sql_injection']:
            threat = self._create_threat(
                threat_type=ThreatType.SQL_INJECTION,
                source_ip=ip,
                user_id=user_id,
                target_resource=endpoint,
                description="SQL injection attempt detected",
                indicators=[
                    ThreatIndicator(
                        indicator_type='pattern',
                        value=str(p),
                        confidence=0.9,
                        source='signature_db',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    ) for p, _ in attack_results['sql_injection'][:3]
                ]
            )
            threats.append(threat)

        if attack_results['xss']:
            threat = self._create_threat(
                threat_type=ThreatType.XSS,
                source_ip=ip,
                user_id=user_id,
                target_resource=endpoint,
                description="XSS attempt detected",
                indicators=[
                    ThreatIndicator(
                        indicator_type='pattern',
                        value=str(p),
                        confidence=0.85,
                        source='signature_db',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    ) for p, _ in attack_results['xss'][:3]
                ]
            )
            threats.append(threat)

        if attack_results['command_injection']:
            threat = self._create_threat(
                threat_type=ThreatType.COMMAND_INJECTION,
                source_ip=ip,
                user_id=user_id,
                target_resource=endpoint,
                description="Command injection attempt detected",
                indicators=[
                    ThreatIndicator(
                        indicator_type='pattern',
                        value=str(p),
                        confidence=0.95,
                        source='signature_db',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    ) for p, _ in attack_results['command_injection'][:3]
                ]
            )
            threats.append(threat)

        if attack_results['path_traversal']:
            threat = self._create_threat(
                threat_type=ThreatType.PATH_TRAVERSAL,
                source_ip=ip,
                user_id=user_id,
                target_resource=endpoint,
                description="Path traversal attempt detected",
                indicators=[
                    ThreatIndicator(
                        indicator_type='pattern',
                        value=str(p),
                        confidence=0.9,
                        source='signature_db',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    ) for p, _ in attack_results['path_traversal'][:3]
                ]
            )
            threats.append(threat)

        # Record event for anomaly detection
        self.anomaly_detector.record_event(ip or 'unknown', 'request')

        # Check for rate anomaly
        is_rate_anomaly, rate = self.anomaly_detector.detect_rate_anomaly(ip or 'unknown')
        if is_rate_anomaly:
            threat = self._create_threat(
                threat_type=ThreatType.DENIAL_OF_SERVICE,
                source_ip=ip,
                description=f"High request rate detected: {rate:.2f} req/min",
                indicators=[
                    ThreatIndicator(
                        indicator_type='rate',
                        value=str(rate),
                        confidence=0.8,
                        source='anomaly_detector',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    )
                ]
            )
            threats.append(threat)

        # Store and trigger callbacks
        for threat in threats:
            with self._lock:
                self._threats[threat.threat_id] = threat
            self._trigger_callbacks(threat)

        return threats

    def record_auth_attempt(self, identifier: str, success: bool,
                            ip: str = None) -> Optional[Threat]:
        """Record authentication attempt and check for brute force"""
        self.brute_force_detector.record_attempt(identifier, success)

        if not success:
            is_attack, failed_count, total = self.brute_force_detector.is_brute_force(identifier)

            if is_attack:
                threat = self._create_threat(
                    threat_type=ThreatType.BRUTE_FORCE,
                    source_ip=ip,
                    target_resource='authentication',
                    description=f"Brute force attack detected: {failed_count} failed attempts",
                    indicators=[
                        ThreatIndicator(
                            indicator_type='failed_attempts',
                            value=str(failed_count),
                            confidence=0.9,
                            source='brute_force_detector',
                            first_seen=datetime.now(),
                            last_seen=datetime.now()
                        )
                    ]
                )

                with self._lock:
                    self._threats[threat.threat_id] = threat

                self._trigger_callbacks(threat)
                return threat

        return None

    def analyze_code(self, code: str) -> List[Threat]:
        """Analyze code for malicious patterns"""
        threats = []

        results = self.signature_db.check_malicious_code(code)

        if results:
            threat = self._create_threat(
                threat_type=ThreatType.MALICIOUS_CODE,
                description="Malicious code patterns detected",
                indicators=[
                    ThreatIndicator(
                        indicator_type='pattern',
                        value=str(p),
                        confidence=0.85,
                        source='signature_db',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    ) for p, _ in results[:5]
                ],
                recommended_actions=[
                    "Review code manually",
                    "Run additional security scans",
                    "Check code source/author"
                ]
            )
            threats.append(threat)

        for threat in threats:
            with self._lock:
                self._threats[threat.threat_id] = threat
            self._trigger_callbacks(threat)

        return threats

    def detect_behavior_anomaly(self, user_id: str, action: str,
                                resource: str = None) -> Optional[Threat]:
        """Detect behavioral anomaly"""
        identifier = f"{user_id}:{action}"

        self.anomaly_detector.record_event(identifier, action)
        is_anomaly, z_score, baseline = self.anomaly_detector.detect_anomaly(identifier, 1.0)

        if is_anomaly:
            threat = self._create_threat(
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                user_id=user_id,
                target_resource=resource,
                description=f"Anomalous behavior detected (z-score: {z_score:.2f})",
                indicators=[
                    ThreatIndicator(
                        indicator_type='z_score',
                        value=str(z_score),
                        confidence=min(z_score / 5.0, 1.0),
                        source='anomaly_detector',
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    )
                ],
                metadata={'baseline': baseline}
            )

            with self._lock:
                self._threats[threat.threat_id] = threat

            self._trigger_callbacks(threat)
            return threat

        return None

    def _create_threat(self, threat_type: ThreatType,
                       severity: ThreatSeverity = None,
                       source_ip: str = None,
                       user_id: str = None,
                       target_resource: str = None,
                       description: str = "",
                       indicators: List[ThreatIndicator] = None,
                       recommended_actions: List[str] = None,
                       metadata: Dict = None) -> Threat:
        """Create a threat object"""
        import uuid

        indicators = indicators or []

        # Calculate confidence from indicators
        confidence = 0.5
        if indicators:
            confidence = sum(i.confidence for i in indicators) / len(indicators)

        # Calculate score
        score = ThreatScorer.calculate_score(threat_type, confidence, severity, indicators)

        # Classify severity if not provided
        if severity is None:
            severity = ThreatScorer.classify_severity(score)

        threat = Threat(
            threat_id=str(uuid.uuid4()),
            threat_type=threat_type,
            severity=severity,
            status=ThreatStatus.DETECTED,
            score=score,
            confidence=confidence,
            detected_at=datetime.now(),
            source_ip=source_ip,
            user_id=user_id,
            target_resource=target_resource,
            description=description,
            indicators=indicators,
            recommended_actions=recommended_actions or [],
            metadata=metadata or {}
        )

        return threat

    def _trigger_callbacks(self, threat: Threat) -> None:
        """Trigger registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(threat)
            except Exception as e:
                print(f"Threat callback error: {e}")

    def register_callback(self, callback: Callable[[Threat], None]) -> None:
        """Register a threat callback"""
        self._callbacks.append(callback)

    def get_threat(self, threat_id: str) -> Optional[Threat]:
        """Get threat by ID"""
        return self._threats.get(threat_id)

    def get_active_threats(self, min_severity: ThreatSeverity = None) -> List[Threat]:
        """Get all active threats"""
        threats = list(self._threats.values())

        if min_severity:
            threats = [t for t in threats if t.severity.value >= min_severity.value]

        return sorted(threats, key=lambda t: t.score, reverse=True)

    def update_threat_status(self, threat_id: str, status: ThreatStatus) -> bool:
        """Update threat status"""
        threat = self._threats.get(threat_id)
        if threat:
            threat.status = status
            threat.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'action': f'Status changed to {status.value}'
            })
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        threats = list(self._threats.values())

        stats = {
            'total_threats': len(threats),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'by_status': defaultdict(int),
            'active_threats': 0,
            'average_score': 0,
        }

        if threats:
            stats['average_score'] = sum(t.score for t in threats) / len(threats)

            for threat in threats:
                stats['by_type'][threat.threat_type.value] += 1
                stats['by_severity'][threat.severity.name] += 1
                stats['by_status'][threat.status.value] += 1

                if threat.status in (ThreatStatus.DETECTED, ThreatStatus.INVESTIGATING, ThreatStatus.CONFIRMED):
                    stats['active_threats'] += 1

        stats['by_type'] = dict(stats['by_type'])
        stats['by_severity'] = dict(stats['by_severity'])
        stats['by_status'] = dict(stats['by_status'])

        return stats


# Export classes
__all__ = [
    'ThreatType',
    'ThreatSeverity',
    'ThreatStatus',
    'ThreatIndicator',
    'Threat',
    'ThreatPattern',
    'AttackSignatureDatabase',
    'IPReputationChecker',
    'AnomalyDetector',
    'BruteForceDetector',
    'ThreatScorer',
    'ThreatDetector'
]


if __name__ == "__main__":
    print("JARVIS Threat Detection System")
    print("=" * 50)

    detector = ThreatDetector()

    # Test SQL injection detection
    request = {
        'ip_address': '192.168.1.100',
        'endpoint': '/api/users',
        'method': 'GET',
        'params': {'id': "1' OR '1'='1"},
        'headers': {},
        'body': ''
    }

    threats = detector.analyze_request(request)
    print(f"\nDetected {len(threats)} threats:")
    for threat in threats:
        print(f"  [{threat.severity.name}] {threat.threat_type.value}: {threat.description}")
        print(f"    Score: {threat.score}, Confidence: {threat.confidence:.2f}")

    # Test brute force detection
    for i in range(6):
        threat = detector.record_auth_attempt('user123', False, '10.0.0.1')
        if threat:
            print(f"\n  Brute force detected: {threat.description}")

    # Get statistics
    stats = detector.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")

    print("\nThreat detection system ready!")
