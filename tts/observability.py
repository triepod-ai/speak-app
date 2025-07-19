#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
TTS Observability module for intelligent voice filtering and event management.
Provides priority-based queuing, rate limiting, and pattern recognition for audio alerts.
"""

import json
import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from dotenv import load_dotenv

# Load environment variables
env_path = Path.home() / "brainpods" / ".env"
if env_path.exists():
    load_dotenv(env_path)

class EventPriority(Enum):
    """Event priority levels for TTS queue management."""
    CRITICAL = 1    # Errors, security blocks
    HIGH = 2        # Slow ops, large files, permissions
    MEDIUM = 3      # Notable successes, completions
    LOW = 4         # Regular operations
    MINIMAL = 5     # Rarely spoken events

class EventCategory(Enum):
    """Categories of events for pattern recognition."""
    ERROR = "error"
    SECURITY = "security"
    PERMISSION = "permission"
    PERFORMANCE = "performance"
    FILE_OPERATION = "file_operation"
    COMMAND_EXECUTION = "command_execution"
    COMPLETION = "completion"
    GENERAL = "general"

@dataclass
class TTSEvent:
    """Represents a TTS event with metadata."""
    message: str
    priority: EventPriority
    category: EventCategory
    timestamp: datetime = field(default_factory=datetime.now)
    hook_type: str = ""
    tool_name: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """Get age of event in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()

class TTSObservability:
    """Manages TTS event filtering, queuing, and observability."""
    
    def __init__(self):
        """Initialize the observability system."""
        self.event_queue: deque[TTSEvent] = deque(maxlen=100)
        self.spoken_events: deque[TTSEvent] = deque(maxlen=50)
        self.event_counts: Dict[EventCategory, int] = defaultdict(int)
        self.recent_messages: deque[str] = deque(maxlen=20)
        self.rate_limits: Dict[EventCategory, float] = {
            EventCategory.ERROR: 0,           # No rate limit
            EventCategory.SECURITY: 0,        # No rate limit
            EventCategory.PERMISSION: 2,      # 2 seconds between
            EventCategory.PERFORMANCE: 5,     # 5 seconds between
            EventCategory.FILE_OPERATION: 3,  # 3 seconds between
            EventCategory.COMMAND_EXECUTION: 2,
            EventCategory.COMPLETION: 10,     # 10 seconds between
            EventCategory.GENERAL: 15,        # 15 seconds between
        }
        self.last_spoken: Dict[EventCategory, datetime] = {}
        self.pattern_cache: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.max_queue_age = 30  # seconds
        self.similarity_threshold = 0.7
        self.burst_threshold = 5  # events in 10 seconds
        self.quiet_hours = self._load_quiet_hours()
    
    def _load_quiet_hours(self) -> Optional[Tuple[int, int]]:
        """Load quiet hours from environment."""
        start = os.getenv("TTS_QUIET_HOURS_START")
        end = os.getenv("TTS_QUIET_HOURS_END")
        
        if start and end:
            try:
                return (int(start), int(end))
            except ValueError:
                pass
        return None
    
    def should_speak(self, event: TTSEvent) -> bool:
        """Determine if an event should be spoken based on filtering rules."""
        # Check quiet hours
        if self._in_quiet_hours() and event.priority.value > 2:
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(event):
            return False
        
        # Check for similar recent messages
        if self._is_duplicate(event.message):
            return False
        
        # Check burst detection
        if self._detect_burst(event.category) and event.priority.value > 2:
            return False
        
        # Priority-based probability
        speak_probability = {
            EventPriority.CRITICAL: 1.0,
            EventPriority.HIGH: 0.9,
            EventPriority.MEDIUM: 0.5,
            EventPriority.LOW: 0.2,
            EventPriority.MINIMAL: 0.05,
        }
        
        import random
        return random.random() < speak_probability.get(event.priority, 0.1)
    
    def _in_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        if not self.quiet_hours:
            return False
        
        current_hour = datetime.now().hour
        start, end = self.quiet_hours
        
        if start <= end:
            return start <= current_hour < end
        else:  # Crosses midnight
            return current_hour >= start or current_hour < end
    
    def _check_rate_limit(self, event: TTSEvent) -> bool:
        """Check if event passes rate limiting."""
        limit = self.rate_limits.get(event.category, 10)
        if limit == 0:
            return True
        
        last_time = self.last_spoken.get(event.category)
        if not last_time:
            return True
        
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= limit
    
    def _is_duplicate(self, message: str) -> bool:
        """Check if message is too similar to recent messages."""
        # Simple check - in production, use fuzzy matching
        message_lower = message.lower()
        for recent in self.recent_messages:
            if self._calculate_similarity(message_lower, recent.lower()) > self.similarity_threshold:
                return True
        return False
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings."""
        # Basic word overlap similarity
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _detect_burst(self, category: EventCategory) -> bool:
        """Detect if we're in a burst of similar events."""
        recent_count = sum(
            1 for event in list(self.event_queue)[-10:]
            if event.category == category and event.age_seconds() < 10
        )
        return recent_count >= self.burst_threshold
    
    def add_event(self, event: TTSEvent) -> bool:
        """Add event to the system and determine if it should be spoken."""
        # Add to queue
        self.event_queue.append(event)
        self.event_counts[event.category] += 1
        
        # Clean old events
        self._clean_old_events()
        
        # Check if should speak
        if self.should_speak(event):
            self.spoken_events.append(event)
            self.recent_messages.append(event.message)
            self.last_spoken[event.category] = datetime.now()
            return True
        
        return False
    
    def _clean_old_events(self):
        """Remove events older than max age from queue."""
        while self.event_queue and self.event_queue[0].age_seconds() > self.max_queue_age:
            self.event_queue.popleft()
    
    def get_statistics(self) -> Dict:
        """Get current observability statistics."""
        return {
            "total_events": sum(self.event_counts.values()),
            "events_by_category": dict(self.event_counts),
            "queue_size": len(self.event_queue),
            "spoken_count": len(self.spoken_events),
            "in_quiet_hours": self._in_quiet_hours(),
            "recent_categories": [
                event.category.value 
                for event in list(self.event_queue)[-10:]
            ],
        }
    
    def get_event_pattern(self) -> Dict[str, List[str]]:
        """Analyze and return event patterns."""
        patterns = defaultdict(list)
        
        # Group by tool and category
        for event in self.event_queue:
            key = f"{event.tool_name}:{event.category.value}"
            patterns[key].append(event.message)
        
        # Return top patterns
        return dict(sorted(
            patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10])

# Global instance
_observability = None

def get_observability() -> TTSObservability:
    """Get or create the global observability instance."""
    global _observability
    if _observability is None:
        _observability = TTSObservability()
    return _observability

def create_event(
    message: str,
    priority: int,
    category: str,
    hook_type: str = "",
    tool_name: str = "",
    metadata: Optional[Dict] = None
) -> TTSEvent:
    """Create a TTS event with proper typing."""
    return TTSEvent(
        message=message,
        priority=EventPriority(priority),
        category=EventCategory(category),
        hook_type=hook_type,
        tool_name=tool_name,
        metadata=metadata or {}
    )

def should_speak_event(
    message: str,
    priority: int,
    category: str,
    hook_type: str = "",
    tool_name: str = "",
    metadata: Optional[Dict] = None
) -> bool:
    """Check if an event should be spoken and add it to the system."""
    obs = get_observability()
    event = create_event(
        message=message,
        priority=priority,
        category=category,
        hook_type=hook_type,
        tool_name=tool_name,
        metadata=metadata
    )
    return obs.add_event(event)

if __name__ == "__main__":
    # Test the observability system
    obs = get_observability()
    
    # Test events
    test_events = [
        ("Error: File not found", 1, "error", "post_tool_use", "Read"),
        ("Permission needed for Bash", 2, "permission", "notification", "Bash"),
        ("File write completed", 3, "file_operation", "post_tool_use", "Write"),
        ("Command executed", 4, "command_execution", "post_tool_use", "Bash"),
        ("Task completed", 3, "completion", "stop", ""),
    ]
    
    for message, priority, category, hook, tool in test_events:
        spoken = should_speak_event(message, priority, category, hook, tool)
        print(f"Event: {message[:30]}... Spoken: {spoken}")
    
    print("\nStatistics:")
    print(json.dumps(obs.get_statistics(), indent=2))