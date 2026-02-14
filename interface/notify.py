#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Notification System
==========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- Termux notification integration
- Sound alerts
- Priority levels
- Quiet mode support

Features:
- Termux notifications
- Sound alerts
- Priority-based handling
- Quiet mode
- Notification history
- Custom handlers

Memory Impact: < 2MB for notifications
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class NotificationType(Enum):
    """Types of notifications"""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    SYSTEM = auto()
    AI = auto()
    TASK = auto()


@dataclass
class NotificationConfig:
    """Notification configuration"""
    enabled: bool = True
    quiet_mode: bool = False
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 7     # 7 AM
    sound_enabled: bool = True
    termux_notifications: bool = True
    max_history: int = 100
    default_priority: NotificationPriority = NotificationPriority.NORMAL


@dataclass
class Notification:
    """
    Notification data container.
    
    Represents a single notification with
    all its metadata and content.
    """
    id: str
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    read: bool = False
    action: Optional[str] = None  # URL or command
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Termux-specific
    sound: bool = True
    vibrate: bool = False
    led_color: str = ""  # Hex color
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'type': self.notification_type.name,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'read': self.read,
            'action': self.action,
        }
    
    @property
    def age_seconds(self) -> float:
        """Get notification age in seconds"""
        return time.time() - self.timestamp


# ═══════════════════════════════════════════════════════════════════════════════
# TERMUX NOTIFICATION HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class TermuxNotifier:
    """
    Handle Termux-specific notifications.
    
    Uses termux-notification command for
    Android notifications.
    """
    
    def __init__(self):
        """Initialize Termux notifier"""
        self._available: Optional[bool] = None
    
    @property
    def available(self) -> bool:
        """Check if Termux notifications are available"""
        if self._available is None:
            self._available = self._check_availability()
        return self._available
    
    def _check_availability(self) -> bool:
        """Check Termux availability"""
        # Check for Termux environment
        if 'TERMUX_VERSION' not in os.environ:
            return False
        
        # Check for termux-notification command
        try:
            result = subprocess.run(
                ['which', 'termux-notification'],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False
    
    def notify(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        sound: bool = True,
        vibrate: bool = False,
        led_color: str = "",
        action: str = "",
    ) -> bool:
        """
        Send Termux notification.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level
            sound: Play sound
            vibrate: Vibrate device
            led_color: LED color (hex)
            action: Action URL or command
            
        Returns:
            True if successful
        """
        if not self.available:
            return False
        
        try:
            cmd = [
                'termux-notification',
                '--title', title,
                '--content', message,
            ]
            
            # Priority
            if priority.value >= NotificationPriority.HIGH.value:
                cmd.extend(['--priority', 'high'])
            
            # Sound
            if sound:
                cmd.append('--sound')
            
            # Vibrate
            if vibrate:
                cmd.append('--vibrate')
            
            # LED
            if led_color:
                cmd.extend(['--led-color', led_color])
            
            # Action
            if action:
                cmd.extend(['--action', action])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Termux notification error: {e}")
            return False
    
    def cancel(self, notification_id: str) -> bool:
        """Cancel a notification"""
        if not self.available:
            return False
        
        try:
            result = subprocess.run(
                ['termux-notification', '--cancel', notification_id],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# SOUND ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

class SoundAlerts:
    """Handle sound alerts for notifications"""
    
    # Sound types
    SOUNDS = {
        'info': '\a',      # Bell
        'success': '\a',   # Bell
        'warning': '\a\a', # Double bell
        'error': '\a\a\a', # Triple bell
    }
    
    def __init__(self, enabled: bool = True):
        """Initialize sound alerts"""
        self._enabled = enabled
    
    def play(self, notification_type: NotificationType):
        """Play sound for notification type"""
        if not self._enabled:
            return
        
        sound_type = notification_type.name.lower()
        sound = self.SOUNDS.get(sound_type, '\a')
        
        try:
            # Print bell character
            sys.stdout.write(sound)
            sys.stdout.flush()
        except:
            pass
    
    def beep(self, count: int = 1):
        """Play beep sound"""
        for _ in range(count):
            sys.stdout.write('\a')
            sys.stdout.flush()
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION MANAGER (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class Notifier:
    """
    Ultra-Advanced Notification System for JARVIS.
    
    Features:
    - Termux notifications
    - Sound alerts
    - Priority-based handling
    - Quiet mode
    - Notification history
    - Custom handlers
    
    Memory Budget: < 2MB
    
    Usage:
        notifier = Notifier()
        
        # Send notification
        notifier.info("Task completed")
        notifier.success("Operation successful!")
        notifier.warning("Low disk space")
        notifier.error("Connection failed")
    """
    
    def __init__(self, config: NotificationConfig = None):
        """
        Initialize Notifier.
        
        Args:
            config: Notification configuration
        """
        self._config = config or NotificationConfig()
        
        # Components
        self._termux = TermuxNotifier()
        self._sound = SoundAlerts(self._config.sound_enabled)
        
        # History
        self._history: List[Notification] = []
        self._lock = threading.Lock()
        
        # Custom handlers
        self._handlers: List[Callable] = []
        
        logger.info(f"Notifier initialized (Termux: {self._termux.available})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core Notification Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def notify(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = None,
        sound: bool = None,
        vibrate: bool = False,
        action: str = "",
        metadata: Dict = None,
    ) -> Notification:
        """
        Send a notification.
        
        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification
            priority: Priority level
            sound: Play sound
            vibrate: Vibrate device
            action: Action URL/command
            metadata: Additional metadata
            
        Returns:
            Created Notification
        """
        # Check if notifications are enabled
        if not self._config.enabled:
            return self._create_notification(
                title, message, notification_type,
                priority, action, metadata
            )
        
        # Check quiet mode
        if self._is_quiet_time() and priority and priority.value < NotificationPriority.HIGH.value:
            logger.debug(f"Quiet time - skipping notification: {title}")
            return self._create_notification(
                title, message, notification_type,
                priority, action, metadata
            )
        
        # Set defaults
        priority = priority or self._config.default_priority
        sound = sound if sound is not None else self._config.sound_enabled
        
        # Create notification
        notification = self._create_notification(
            title, message, notification_type,
            priority, action, metadata
        )
        
        # Send via Termux
        if self._config.termux_notifications and self._termux.available:
            self._termux.notify(
                title=title,
                message=message,
                priority=priority,
                sound=sound,
                vibrate=vibrate,
                action=action,
            )
        
        # Play sound
        if sound and not self._is_quiet_time():
            self._sound.play(notification_type)
        
        # Run handlers
        for handler in self._handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
        
        return notification
    
    def _create_notification(
        self,
        title: str,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority,
        action: str,
        metadata: Dict,
    ) -> Notification:
        """Create notification object"""
        import uuid
        
        notification = Notification(
            id=uuid.uuid4().hex[:8],
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority or self._config.default_priority,
            action=action,
            metadata=metadata or {},
        )
        
        # Add to history
        with self._lock:
            self._history.append(notification)
            
            # Trim history
            if len(self._history) > self._config.max_history:
                self._history = self._history[-self._config.max_history:]
        
        return notification
    
    def _is_quiet_time(self) -> bool:
        """Check if currently in quiet hours"""
        if not self._config.quiet_mode:
            return False
        
        hour = datetime.now().hour
        
        if self._config.quiet_hours_start > self._config.quiet_hours_end:
            # Spans midnight
            return hour >= self._config.quiet_hours_start or hour < self._config.quiet_hours_end
        else:
            return self._config.quiet_hours_start <= hour < self._config.quiet_hours_end
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def info(self, title: str, message: str = "", **kwargs):
        """Send info notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.INFO,
            **kwargs
        )
    
    def success(self, title: str, message: str = "", **kwargs):
        """Send success notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.SUCCESS,
            priority=NotificationPriority.NORMAL,
            **kwargs
        )
    
    def warning(self, title: str, message: str = "", **kwargs):
        """Send warning notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.WARNING,
            priority=NotificationPriority.HIGH,
            **kwargs
        )
    
    def error(self, title: str, message: str = "", **kwargs):
        """Send error notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.ERROR,
            priority=NotificationPriority.HIGH,
            **kwargs
        )
    
    def urgent(self, title: str, message: str = "", **kwargs):
        """Send urgent notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.SYSTEM,
            priority=NotificationPriority.URGENT,
            sound=True,
            vibrate=True,
            **kwargs
        )
    
    def task(self, title: str, message: str = "", **kwargs):
        """Send task notification"""
        return self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.TASK,
            **kwargs
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # History Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_history(self, limit: int = None) -> List[Notification]:
        """Get notification history"""
        with self._lock:
            history = list(self._history)
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def mark_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        with self._lock:
            for notification in self._history:
                if notification.id == notification_id:
                    notification.read = True
                    return True
        return False
    
    def clear_history(self):
        """Clear notification history"""
        with self._lock:
            self._history.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_quiet_mode(self, enabled: bool):
        """Set quiet mode"""
        self._config.quiet_mode = enabled
    
    def set_sound_enabled(self, enabled: bool):
        """Set sound enabled"""
        self._config.sound_enabled = enabled
        self._sound._enabled = enabled
    
    def add_handler(self, handler: Callable):
        """Add custom notification handler"""
        self._handlers.append(handler)
    
    @property
    def termux_available(self) -> bool:
        """Check if Termux notifications are available"""
        return self._termux.available
    
    @property
    def history_count(self) -> int:
        """Get history count"""
        return len(self._history)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo notification system"""
    notifier = Notifier()
    
    print("JARVIS Notification System Demo")
    print("=" * 40)
    
    print(f"\nTermux available: {notifier.termux_available}")
    
    # Send various notifications
    notifier.info("Info", "This is an information message")
    time.sleep(0.5)
    
    notifier.success("Success", "Operation completed successfully!")
    time.sleep(0.5)
    
    notifier.warning("Warning", "Low disk space detected")
    time.sleep(0.5)
    
    notifier.error("Error", "Connection failed")
    
    # Show history
    print(f"\nNotification history: {notifier.history_count}")
    for n in notifier.get_history(5):
        print(f"  [{n.notification_type.name}] {n.title}: {n.message}")


if __name__ == '__main__':
    main()
