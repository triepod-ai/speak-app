#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
Test suite for TTS Observability system.
Tests event filtering, rate limiting, burst detection, and pattern recognition.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import time

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.observability import (
    TTSObservability, 
    EventPriority, 
    EventCategory,
    TTSEvent,
    get_observability,
    create_event,
    should_speak_event
)

class TestEventPriority:
    """Test EventPriority enum."""
    
    def test_priority_values(self):
        """Test priority values are correct."""
        assert EventPriority.CRITICAL.value == 1
        assert EventPriority.HIGH.value == 2
        assert EventPriority.MEDIUM.value == 3
        assert EventPriority.LOW.value == 4
        assert EventPriority.MINIMAL.value == 5
    
    def test_priority_ordering(self):
        """Test priorities can be compared."""
        assert EventPriority.CRITICAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.MEDIUM.value

class TestEventCategory:
    """Test EventCategory enum."""
    
    def test_category_values(self):
        """Test category string values."""
        assert EventCategory.ERROR.value == "error"
        assert EventCategory.SECURITY.value == "security"
        assert EventCategory.PERMISSION.value == "permission"
        assert EventCategory.PERFORMANCE.value == "performance"
        assert EventCategory.FILE_OPERATION.value == "file_operation"
        assert EventCategory.COMMAND_EXECUTION.value == "command_execution"
        assert EventCategory.COMPLETION.value == "completion"
        assert EventCategory.GENERAL.value == "general"

class TestTTSEvent:
    """Test TTSEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating a TTS event."""
        event = TTSEvent(
            message="Test message",
            priority=EventPriority.HIGH,
            category=EventCategory.ERROR,
            hook_type="test_hook",
            tool_name="TestTool"
        )
        
        assert event.message == "Test message"
        assert event.priority == EventPriority.HIGH
        assert event.category == EventCategory.ERROR
        assert event.hook_type == "test_hook"
        assert event.tool_name == "TestTool"
        assert isinstance(event.timestamp, datetime)
    
    def test_event_age(self):
        """Test event age calculation."""
        event = TTSEvent(
            message="Test",
            priority=EventPriority.LOW,
            category=EventCategory.GENERAL
        )
        
        # Should be very close to 0 initially
        assert event.age_seconds() < 0.1
        
        # Mock timestamp to be 5 seconds ago
        event.timestamp = datetime.now() - timedelta(seconds=5)
        assert 4.9 < event.age_seconds() < 5.1

class TestTTSObservability:
    """Test TTSObservability class."""
    
    @pytest.fixture
    def observability(self):
        """Create a fresh observability instance."""
        return TTSObservability()
    
    def test_initialization(self, observability):
        """Test observability system initialization."""
        assert len(observability.event_queue) == 0
        assert len(observability.spoken_events) == 0
        assert len(observability.recent_messages) == 0
        assert observability.max_queue_age == 30
        assert observability.similarity_threshold == 0.7
        assert observability.burst_threshold == 5
    
    def test_rate_limits(self, observability):
        """Test rate limit configuration."""
        assert observability.rate_limits[EventCategory.ERROR] == 0  # No limit
        assert observability.rate_limits[EventCategory.SECURITY] == 0
        assert observability.rate_limits[EventCategory.PERMISSION] == 2
        assert observability.rate_limits[EventCategory.PERFORMANCE] == 5
        assert observability.rate_limits[EventCategory.FILE_OPERATION] == 3
        assert observability.rate_limits[EventCategory.COMMAND_EXECUTION] == 2
        assert observability.rate_limits[EventCategory.COMPLETION] == 10
        assert observability.rate_limits[EventCategory.GENERAL] == 15
    
    @patch.dict(os.environ, {"TTS_QUIET_HOURS_START": "22", "TTS_QUIET_HOURS_END": "8"})
    def test_quiet_hours_loading(self):
        """Test loading quiet hours from environment."""
        obs = TTSObservability()
        assert obs.quiet_hours == (22, 8)
    
    @patch.dict(os.environ, {"TTS_QUIET_HOURS_START": "invalid", "TTS_QUIET_HOURS_END": "8"})
    def test_quiet_hours_invalid(self):
        """Test handling invalid quiet hours."""
        obs = TTSObservability()
        assert obs.quiet_hours is None
    
    def test_in_quiet_hours(self, observability):
        """Test quiet hours detection."""
        # No quiet hours set
        assert not observability._in_quiet_hours()
        
        # Set quiet hours
        observability.quiet_hours = (22, 8)
        
        # Mock current time
        with patch('tts.observability.datetime') as mock_datetime:
            # During quiet hours (23:00)
            mock_datetime.now.return_value.hour = 23
            assert observability._in_quiet_hours()
            
            # During quiet hours (3:00)
            mock_datetime.now.return_value.hour = 3
            assert observability._in_quiet_hours()
            
            # Outside quiet hours (12:00)
            mock_datetime.now.return_value.hour = 12
            assert not observability._in_quiet_hours()
    
    def test_check_rate_limit(self, observability):
        """Test rate limiting functionality."""
        event = TTSEvent(
            message="Test",
            priority=EventPriority.HIGH,
            category=EventCategory.PERMISSION
        )
        
        # First event should pass
        assert observability._check_rate_limit(event)
        
        # Mark as spoken
        observability.last_spoken[EventCategory.PERMISSION] = datetime.now()
        
        # Immediate second event should fail
        assert not observability._check_rate_limit(event)
        
        # After rate limit period, should pass
        observability.last_spoken[EventCategory.PERMISSION] = datetime.now() - timedelta(seconds=3)
        assert observability._check_rate_limit(event)
    
    def test_calculate_similarity(self, observability):
        """Test string similarity calculation."""
        # Identical strings
        assert observability._calculate_similarity("hello world", "hello world") == 1.0
        
        # Completely different
        assert observability._calculate_similarity("hello world", "foo bar") == 0.0
        
        # Partial overlap
        sim = observability._calculate_similarity("hello world test", "hello world")
        assert 0.5 < sim < 0.8
        
        # Empty strings
        assert observability._calculate_similarity("", "") == 0.0
        assert observability._calculate_similarity("hello", "") == 0.0
    
    def test_is_duplicate(self, observability):
        """Test duplicate detection."""
        # No recent messages
        assert not observability._is_duplicate("Test message")
        
        # Add a message
        observability.recent_messages.append("Test message")
        
        # Same message is duplicate
        assert observability._is_duplicate("Test message")
        
        # Similar message is duplicate
        assert observability._is_duplicate("Test MESSAGE")
        
        # Different message is not duplicate
        assert not observability._is_duplicate("Completely different")
    
    def test_detect_burst(self, observability):
        """Test burst detection."""
        # No events, no burst
        assert not observability._detect_burst(EventCategory.ERROR)
        
        # Add multiple events quickly
        for i in range(6):
            event = TTSEvent(
                message=f"Error {i}",
                priority=EventPriority.HIGH,
                category=EventCategory.ERROR
            )
            observability.event_queue.append(event)
        
        # Should detect burst
        assert observability._detect_burst(EventCategory.ERROR)
        
        # Different category should not be in burst
        assert not observability._detect_burst(EventCategory.GENERAL)
    
    @patch('random.random')
    def test_should_speak_priority_probability(self, mock_random, observability):
        """Test priority-based probability."""
        # Critical always speaks (probability 1.0)
        mock_random.return_value = 0.99
        event = TTSEvent(
            message="Critical",
            priority=EventPriority.CRITICAL,
            category=EventCategory.ERROR
        )
        assert observability.should_speak(event)
        
        # Minimal rarely speaks (probability 0.05)
        mock_random.return_value = 0.06
        event = TTSEvent(
            message="Minimal",
            priority=EventPriority.MINIMAL,
            category=EventCategory.GENERAL
        )
        assert not observability.should_speak(event)
        
        mock_random.return_value = 0.04
        assert observability.should_speak(event)
    
    def test_add_event(self, observability):
        """Test adding events to the system."""
        event = TTSEvent(
            message="Test event",
            priority=EventPriority.CRITICAL,
            category=EventCategory.ERROR
        )
        
        # Critical events should be spoken
        result = observability.add_event(event)
        assert result is True
        assert len(observability.event_queue) == 1
        assert len(observability.spoken_events) == 1
        assert observability.event_counts[EventCategory.ERROR] == 1
    
    def test_clean_old_events(self, observability):
        """Test cleaning old events from queue."""
        # Add old event
        old_event = TTSEvent(
            message="Old",
            priority=EventPriority.LOW,
            category=EventCategory.GENERAL
        )
        old_event.timestamp = datetime.now() - timedelta(seconds=40)
        observability.event_queue.append(old_event)
        
        # Add recent event
        new_event = TTSEvent(
            message="New",
            priority=EventPriority.LOW,
            category=EventCategory.GENERAL
        )
        observability.event_queue.append(new_event)
        
        # Clean old events
        observability._clean_old_events()
        
        # Old event should be removed
        assert len(observability.event_queue) == 1
        assert observability.event_queue[0].message == "New"
    
    def test_get_statistics(self, observability):
        """Test getting observability statistics."""
        # Add some events
        for i in range(3):
            event = TTSEvent(
                message=f"Event {i}",
                priority=EventPriority.HIGH,
                category=EventCategory.ERROR
            )
            observability.add_event(event)
        
        stats = observability.get_statistics()
        
        assert stats["total_events"] == 3
        assert stats["events_by_category"]["error"] == 3
        assert stats["queue_size"] == 3
        assert stats["spoken_count"] == 3
        assert isinstance(stats["in_quiet_hours"], bool)
        assert len(stats["recent_categories"]) == 3
    
    def test_get_event_pattern(self, observability):
        """Test event pattern analysis."""
        # Add events with patterns
        tools = ["Read", "Write", "Read", "Read"]
        categories = [EventCategory.FILE_OPERATION, EventCategory.FILE_OPERATION,
                     EventCategory.FILE_OPERATION, EventCategory.ERROR]
        
        for i, (tool, cat) in enumerate(zip(tools, categories)):
            event = TTSEvent(
                message=f"Operation {i}",
                priority=EventPriority.MEDIUM,
                category=cat,
                tool_name=tool
            )
            observability.event_queue.append(event)
        
        patterns = observability.get_event_pattern()
        
        # Should have pattern for Read:file_operation
        assert "Read:file_operation" in patterns
        assert len(patterns["Read:file_operation"]) == 2

class TestGlobalFunctions:
    """Test global helper functions."""
    
    def test_get_observability_singleton(self):
        """Test get_observability returns singleton."""
        obs1 = get_observability()
        obs2 = get_observability()
        assert obs1 is obs2
    
    def test_create_event(self):
        """Test create_event helper."""
        event = create_event(
            message="Test",
            priority=1,
            category="error",
            hook_type="test",
            tool_name="Tool"
        )
        
        assert isinstance(event, TTSEvent)
        assert event.message == "Test"
        assert event.priority == EventPriority.CRITICAL
        assert event.category == EventCategory.ERROR
        assert event.hook_type == "test"
        assert event.tool_name == "Tool"
    
    @patch('tts.observability.get_observability')
    def test_should_speak_event(self, mock_get_obs):
        """Test should_speak_event helper."""
        mock_obs = Mock()
        mock_obs.add_event.return_value = True
        mock_get_obs.return_value = mock_obs
        
        result = should_speak_event(
            message="Test",
            priority=1,
            category="error",
            hook_type="test",
            tool_name="Tool"
        )
        
        assert result is True
        assert mock_obs.add_event.called

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def observability(self):
        """Create a fresh observability instance."""
        return TTSObservability()
    
    def test_rapid_error_scenario(self, observability):
        """Test handling rapid error events."""
        # Errors should have no rate limit
        for i in range(10):
            event = TTSEvent(
                message=f"Error {i}",
                priority=EventPriority.CRITICAL,
                category=EventCategory.ERROR
            )
            # All critical errors should be spoken
            assert observability.add_event(event) is True
    
    def test_file_operation_rate_limiting(self, observability):
        """Test file operation rate limiting."""
        # First file operation should be spoken
        event1 = TTSEvent(
            message="File saved",
            priority=EventPriority.MEDIUM,
            category=EventCategory.FILE_OPERATION
        )
        
        with patch('random.random', return_value=0.4):  # Below 0.5 threshold
            assert observability.add_event(event1) is True
        
        # Immediate second should be rate limited
        event2 = TTSEvent(
            message="Another file saved",
            priority=EventPriority.MEDIUM,
            category=EventCategory.FILE_OPERATION
        )
        assert observability.add_event(event2) is False
    
    def test_quiet_hours_filtering(self, observability):
        """Test quiet hours filtering."""
        observability.quiet_hours = (22, 8)
        
        with patch('tts.observability.datetime') as mock_datetime:
            # During quiet hours
            mock_datetime.now.return_value.hour = 23
            
            # Low priority should be filtered
            low_event = TTSEvent(
                message="Low priority",
                priority=EventPriority.LOW,
                category=EventCategory.GENERAL
            )
            assert observability.should_speak(low_event) is False
            
            # Critical should still speak
            critical_event = TTSEvent(
                message="Critical error",
                priority=EventPriority.CRITICAL,
                category=EventCategory.ERROR
            )
            assert observability.should_speak(critical_event) is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])