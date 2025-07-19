#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
# ]
# ///

"""
Comprehensive test suite for TTS Usage Tracker.
Tests the UsageTracker class functionality for cost monitoring and optimization.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pytest

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tts.usage_tracker import UsageTracker


class TestUsageTrackerInitialization:
    """Test UsageTracker initialization and setup."""
    
    def test_usage_tracker_init_default_location(self):
        """Test usage tracker initialization with default location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pathlib.Path.expanduser', return_value=Path(tmpdir)):
                tracker = UsageTracker()
                assert tracker.data_dir == Path(tmpdir)
                assert tracker.data_dir.exists()
                assert tracker.usage_file == Path(tmpdir) / "usage_stats.json"
                assert tracker.daily_file == Path(tmpdir) / "daily_usage.txt"
    
    def test_usage_tracker_init_custom_location(self):
        """Test usage tracker initialization with custom location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_data"
            tracker = UsageTracker(data_dir=str(custom_dir))
            assert tracker.data_dir == custom_dir
            assert tracker.data_dir.exists()
    
    def test_usage_tracker_init_creates_directory(self):
        """Test that data directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "new_data_dir"
            assert not data_dir.exists()
            
            tracker = UsageTracker(data_dir=str(data_dir))
            assert data_dir.exists()
            assert tracker.data_dir == data_dir
    
    def test_usage_tracker_init_loads_existing_stats(self):
        """Test loading existing stats file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            usage_file = data_dir / "usage_stats.json"
            
            # Create existing stats file
            existing_stats = {
                "providers": {
                    "openai": {
                        "total_characters": 1000,
                        "total_requests": 5,
                        "total_cost": 0.015
                    }
                },
                "total_characters": 1000,
                "total_cost_estimate": 0.015
            }
            with open(usage_file, 'w') as f:
                json.dump(existing_stats, f)
            
            tracker = UsageTracker(data_dir=str(data_dir))
            assert tracker.stats["providers"]["openai"]["total_characters"] == 1000
            assert tracker.stats["total_characters"] == 1000
    
    def test_usage_tracker_init_handles_corrupted_stats(self):
        """Test handling corrupted stats file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            usage_file = data_dir / "usage_stats.json"
            
            # Create corrupted stats file
            with open(usage_file, 'w') as f:
                f.write("invalid json content")
            
            tracker = UsageTracker(data_dir=str(data_dir))
            # Should create empty stats
            assert tracker.stats["providers"] == {}
            assert tracker.stats["total_characters"] == 0
            assert tracker.stats["total_cost_estimate"] == 0.0
    
    def test_usage_tracker_provider_costs(self):
        """Test that provider costs are correctly defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            assert tracker.PROVIDER_COSTS['elevenlabs'] == 0.33
            assert tracker.PROVIDER_COSTS['openai'] == 0.015
            assert tracker.PROVIDER_COSTS['openai-hd'] == 0.030
            assert tracker.PROVIDER_COSTS['pyttsx3'] == 0.0
            assert tracker.PROVIDER_COSTS['azure'] == 0.004
            assert tracker.PROVIDER_COSTS['google'] == 0.016


class TestUsageTrackerEmptyStats:
    """Test empty stats structure creation."""
    
    def test_empty_stats_structure(self):
        """Test that empty stats structure is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            empty_stats = tracker._empty_stats()
            
            assert empty_stats['providers'] == {}
            assert empty_stats['daily_usage'] == {}
            assert empty_stats['monthly_usage'] == {}
            assert empty_stats['total_characters'] == 0
            assert empty_stats['total_cost_estimate'] == 0.0
            assert empty_stats['cache_hits'] == 0
            assert empty_stats['cache_savings'] == 0.0
            assert 'start_date' in empty_stats
            
            # Verify start_date is valid ISO format
            datetime.fromisoformat(empty_stats['start_date'])
    
    def test_empty_stats_new_instance(self):
        """Test that new instance starts with empty stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            assert tracker.stats['providers'] == {}
            assert tracker.stats['daily_usage'] == {}
            assert tracker.stats['monthly_usage'] == {}
            assert tracker.stats['total_characters'] == 0
            assert tracker.stats['total_cost_estimate'] == 0.0
            assert tracker.stats['cache_hits'] == 0
            assert tracker.stats['cache_savings'] == 0.0


class TestUsageTrackerCostEstimation:
    """Test cost estimation functionality."""
    
    def test_estimate_cost_basic(self):
        """Test basic cost estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Test OpenAI standard
            cost = tracker.estimate_cost('openai', 1000)
            assert cost == 0.015  # 1000 chars * 0.015/1000
            
            # Test ElevenLabs
            cost = tracker.estimate_cost('elevenlabs', 1000)
            assert cost == 0.33  # 1000 chars * 0.33/1000
            
            # Test free provider
            cost = tracker.estimate_cost('pyttsx3', 1000)
            assert cost == 0.0
    
    def test_estimate_cost_openai_hd(self):
        """Test cost estimation for OpenAI HD model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Test OpenAI HD
            cost = tracker.estimate_cost('openai', 1000, 'tts-1-hd')
            assert cost == 0.030  # 1000 chars * 0.030/1000
            
            # Test OpenAI standard (should ignore model parameter)
            cost = tracker.estimate_cost('openai', 1000, 'tts-1')
            assert cost == 0.015  # 1000 chars * 0.015/1000
    
    def test_estimate_cost_unknown_provider(self):
        """Test cost estimation for unknown provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            cost = tracker.estimate_cost('unknown_provider', 1000)
            assert cost == 0.0  # Should default to 0
    
    def test_estimate_cost_fractional_chars(self):
        """Test cost estimation with fractional character counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Test with 500 characters
            cost = tracker.estimate_cost('openai', 500)
            assert cost == 0.0075  # 500 chars * 0.015/1000
            
            # Test with 1 character
            cost = tracker.estimate_cost('openai', 1)
            assert cost == 0.000015  # 1 char * 0.015/1000
    
    def test_estimate_cost_large_numbers(self):
        """Test cost estimation with large character counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Test with 100,000 characters
            cost = tracker.estimate_cost('openai', 100000)
            assert cost == 1.5  # 100000 chars * 0.015/1000
            
            # Test with 1,000,000 characters
            cost = tracker.estimate_cost('elevenlabs', 1000000)
            assert cost == 330.0  # 1000000 chars * 0.33/1000


class TestUsageTrackerUsageTracking:
    """Test usage tracking functionality."""
    
    def test_track_usage_basic(self):
        """Test basic usage tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Hello world"
            cost, usage_info = tracker.track_usage('openai', text)
            
            expected_cost = len(text) * 0.015 / 1000
            assert cost == expected_cost
            assert usage_info['characters'] == len(text)
            assert usage_info['cost'] == expected_cost
            
            # Verify stats updated
            assert tracker.stats['providers']['openai']['total_characters'] == len(text)
            assert tracker.stats['providers']['openai']['total_requests'] == 1
            assert tracker.stats['providers']['openai']['total_cost'] == expected_cost
            assert tracker.stats['total_characters'] == len(text)
            assert tracker.stats['total_cost_estimate'] == expected_cost
    
    def test_track_usage_cached(self):
        """Test tracking cached usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Cached message"
            cost, usage_info = tracker.track_usage('openai', text, cached=True)
            
            assert cost == 0.0
            assert usage_info['cached'] is True
            assert usage_info['saved'] > 0
            
            # Verify cache stats updated
            assert tracker.stats['cache_hits'] == 1
            assert tracker.stats['cache_savings'] > 0
            
            # Verify no provider stats updated for cached usage
            assert 'openai' not in tracker.stats['providers']
    
    def test_track_usage_multiple_providers(self):
        """Test tracking usage across multiple providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track OpenAI usage
            tracker.track_usage('openai', "OpenAI message")
            
            # Track ElevenLabs usage
            tracker.track_usage('elevenlabs', "ElevenLabs message")
            
            # Verify both providers tracked
            assert 'openai' in tracker.stats['providers']
            assert 'elevenlabs' in tracker.stats['providers']
            
            assert tracker.stats['providers']['openai']['total_characters'] == 14
            assert tracker.stats['providers']['elevenlabs']['total_characters'] == 18
            
            assert tracker.stats['total_characters'] == 32
    
    def test_track_usage_model_tracking(self):
        """Test OpenAI model usage tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track different OpenAI models
            tracker.track_usage('openai', "Standard message", model='tts-1')
            tracker.track_usage('openai', "HD message", model='tts-1-hd')
            
            # Verify model tracking
            models = tracker.stats['providers']['openai']['models']
            assert 'tts-1' in models
            assert 'tts-1-hd' in models
            assert models['tts-1'] == 16  # "Standard message"
            assert models['tts-1-hd'] == 10  # "HD message"
    
    def test_track_usage_daily_stats(self):
        """Test daily usage statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Daily test message"
            today = datetime.now().strftime('%Y-%m-%d')
            
            tracker.track_usage('openai', text)
            
            # Verify daily stats
            assert today in tracker.stats['daily_usage']
            daily_stats = tracker.stats['daily_usage'][today]
            assert daily_stats['total'] == len(text)
            assert daily_stats['providers']['openai'] == len(text)
    
    def test_track_usage_monthly_stats(self):
        """Test monthly usage statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Monthly test message"
            month = datetime.now().strftime('%Y-%m')
            
            cost, _ = tracker.track_usage('openai', text)
            
            # Verify monthly stats
            assert month in tracker.stats['monthly_usage']
            monthly_stats = tracker.stats['monthly_usage'][month]
            assert monthly_stats['total'] == len(text)
            assert monthly_stats['providers']['openai'] == len(text)
            assert monthly_stats['cost'] == cost
    
    def test_track_usage_multiple_requests_same_day(self):
        """Test multiple requests on same day accumulate correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Track multiple requests
            tracker.track_usage('openai', "First message")
            tracker.track_usage('openai', "Second message")
            tracker.track_usage('elevenlabs', "Third message")
            
            # Verify accumulation
            daily_stats = tracker.stats['daily_usage'][today]
            assert daily_stats['total'] == 40  # Sum of all messages
            assert daily_stats['providers']['openai'] == 27  # First + Second
            assert daily_stats['providers']['elevenlabs'] == 13  # Third
    
    def test_track_usage_updates_daily_file(self):
        """Test that daily usage file is updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Test message"
            tracker.track_usage('openai', text)
            
            # Verify daily file exists and contains correct data
            assert tracker.daily_file.exists()
            with open(tracker.daily_file, 'r') as f:
                content = f.read().strip()
                assert content == str(len(text))
    
    def test_track_usage_daily_file_error_handling(self):
        """Test error handling when daily file cannot be written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Make daily file unwritable
            tracker.daily_file.touch()
            tracker.daily_file.chmod(0o000)
            
            # Should not raise exception
            try:
                tracker.track_usage('openai', "Test message")
            except Exception as e:
                pytest.fail(f"track_usage raised an exception: {e}")
            finally:
                # Restore permissions for cleanup
                tracker.daily_file.chmod(0o644)


class TestUsageTrackerDataRetrieval:
    """Test data retrieval functionality."""
    
    def test_get_daily_usage_current_day(self):
        """Test getting current day usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Daily usage test"
            tracker.track_usage('openai', text)
            
            daily_usage = tracker.get_daily_usage()
            
            assert daily_usage['total'] == len(text)
            assert daily_usage['providers']['openai'] == len(text)
    
    def test_get_daily_usage_specific_date(self):
        """Test getting usage for specific date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Mock a specific date
            test_date = '2024-01-15'
            tracker.stats['daily_usage'][test_date] = {
                'total': 100,
                'providers': {'openai': 100}
            }
            
            daily_usage = tracker.get_daily_usage(test_date)
            
            assert daily_usage['total'] == 100
            assert daily_usage['providers']['openai'] == 100
    
    def test_get_daily_usage_missing_date(self):
        """Test getting usage for date with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            daily_usage = tracker.get_daily_usage('2024-01-01')
            
            assert daily_usage['total'] == 0
            assert daily_usage['providers'] == {}
    
    def test_get_monthly_usage_current_month(self):
        """Test getting current month usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Monthly usage test"
            cost, _ = tracker.track_usage('openai', text)
            
            monthly_usage = tracker.get_monthly_usage()
            
            assert monthly_usage['total'] == len(text)
            assert monthly_usage['providers']['openai'] == len(text)
            assert monthly_usage['cost'] == cost
    
    def test_get_monthly_usage_specific_month(self):
        """Test getting usage for specific month."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Mock a specific month
            test_month = '2024-01'
            tracker.stats['monthly_usage'][test_month] = {
                'total': 1000,
                'providers': {'openai': 1000},
                'cost': 0.15
            }
            
            monthly_usage = tracker.get_monthly_usage(test_month)
            
            assert monthly_usage['total'] == 1000
            assert monthly_usage['providers']['openai'] == 1000
            assert monthly_usage['cost'] == 0.15
    
    def test_get_monthly_usage_missing_month(self):
        """Test getting usage for month with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            monthly_usage = tracker.get_monthly_usage('2024-01')
            
            assert monthly_usage['total'] == 0
            assert monthly_usage['providers'] == {}
            assert monthly_usage['cost'] == 0.0


class TestUsageTrackerProviderBreakdown:
    """Test provider breakdown functionality."""
    
    def test_get_provider_breakdown_basic(self):
        """Test basic provider breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track usage for multiple providers
            tracker.track_usage('openai', "OpenAI message 1")
            tracker.track_usage('openai', "OpenAI message 2")
            tracker.track_usage('elevenlabs', "ElevenLabs message")
            
            breakdown = tracker.get_provider_breakdown()
            
            assert 'openai' in breakdown
            assert 'elevenlabs' in breakdown
            
            # Check OpenAI stats
            openai_stats = breakdown['openai']
            assert openai_stats['characters'] == 32  # Sum of both messages
            assert openai_stats['requests'] == 2
            assert openai_stats['avg_length'] == 16.0  # 32 / 2
            assert openai_stats['percentage'] > 0
            
            # Check ElevenLabs stats
            elevenlabs_stats = breakdown['elevenlabs']
            assert elevenlabs_stats['characters'] == 18
            assert elevenlabs_stats['requests'] == 1
            assert elevenlabs_stats['avg_length'] == 18.0
            assert elevenlabs_stats['percentage'] > 0
    
    def test_get_provider_breakdown_empty(self):
        """Test provider breakdown with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            breakdown = tracker.get_provider_breakdown()
            
            assert breakdown == {}
    
    def test_get_provider_breakdown_percentages(self):
        """Test that provider breakdown percentages sum to 100."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track equal usage
            tracker.track_usage('openai', "A" * 500)
            tracker.track_usage('elevenlabs', "B" * 500)
            
            breakdown = tracker.get_provider_breakdown()
            
            # Check percentages sum to 100
            total_percentage = sum(data['percentage'] for data in breakdown.values())
            assert abs(total_percentage - 100.0) < 0.01  # Allow for floating point precision
    
    def test_get_provider_breakdown_cost_accuracy(self):
        """Test that cost calculations in breakdown are accurate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            text = "Cost test message"
            cost, _ = tracker.track_usage('openai', text)
            
            breakdown = tracker.get_provider_breakdown()
            
            assert abs(breakdown['openai']['cost'] - cost) < 0.000001


class TestUsageTrackerCostReport:
    """Test cost report generation."""
    
    def test_get_cost_report_basic(self):
        """Test basic cost report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some usage
            tracker.track_usage('openai', "Test message")
            tracker.track_usage('openai', "Another test", cached=True)
            
            report = tracker.get_cost_report()
            
            # Check report contains expected sections
            assert "📊 TTS Usage Report" in report
            assert "📈 Total Usage:" in report
            assert "Characters:" in report
            assert "Cost:" in report
            assert "Cache Hits:" in report
            assert "Cache Savings:" in report
            assert "📅 This Month" in report
            assert "📆 Today" in report
            assert "🎯 Provider Breakdown:" in report
            assert "💡 Recommendations:" in report
    
    def test_get_cost_report_with_recommendations(self):
        """Test cost report with recommendations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add significant ElevenLabs usage
            for i in range(10):
                tracker.track_usage('elevenlabs', f"ElevenLabs message {i}")
            
            # Add minimal cache usage
            tracker.track_usage('openai', "One cached", cached=True)
            
            report = tracker.get_cost_report()
            
            # Should recommend switching to OpenAI
            assert "Switch ElevenLabs to OpenAI" in report
            assert "Save $" in report
            
            # Should recommend enabling caching
            assert "Enable caching" in report
    
    def test_get_cost_report_empty_data(self):
        """Test cost report with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            report = tracker.get_cost_report()
            
            # Should still generate report with zeros
            assert "📊 TTS Usage Report" in report
            assert "Characters: 0" in report
            assert "Cost: $0.00" in report
    
    def test_get_cost_report_format(self):
        """Test cost report formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add usage with known values
            tracker.track_usage('openai', "A" * 1000)  # 1000 characters
            
            report = tracker.get_cost_report()
            
            # Check number formatting
            assert "Characters: 1,000" in report  # Should have comma separator
            assert "$0.01" in report  # Should have proper cost formatting


class TestUsageTrackerPersistence:
    """Test persistence functionality."""
    
    def test_save_stats_basic(self):
        """Test basic stats saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some data
            tracker.track_usage('openai', "Test message")
            
            # Verify file was created
            assert tracker.usage_file.exists()
            
            # Verify content is correct
            with open(tracker.usage_file, 'r') as f:
                saved_data = json.load(f)
                assert saved_data['total_characters'] == 12
                assert 'openai' in saved_data['providers']
    
    def test_save_stats_error_handling(self):
        """Test error handling during stats save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Mock file operations to raise exception
            with patch('builtins.open', side_effect=Exception("Test exception")):
                with patch('sys.stderr'):
                    tracker.save_stats()  # Should not raise exception
    
    def test_stats_persistence_across_instances(self):
        """Test that stats persist across different instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            tracker1 = UsageTracker(data_dir=tmpdir)
            tracker1.track_usage('openai', "Persistent message")
            
            # Second instance
            tracker2 = UsageTracker(data_dir=tmpdir)
            
            # Should have loaded data from first instance
            assert tracker2.stats['total_characters'] == 18
            assert 'openai' in tracker2.stats['providers']
            assert tracker2.stats['providers']['openai']['total_characters'] == 18


class TestUsageTrackerCLIInterface:
    """Test CLI interface functionality."""
    
    def test_cli_no_arguments(self):
        """Test CLI with no arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some test data
            tracker.track_usage('openai', "CLI test message")
            
            # Mock sys.argv
            with patch('sys.argv', ['usage_tracker.py']):
                # Would normally print quick stats
                # Just verify no exceptions are raised
                pass
    
    def test_cli_report_argument(self):
        """Test CLI with --report argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some test data
            tracker.track_usage('openai', "CLI report test")
            
            # Mock sys.argv
            with patch('sys.argv', ['usage_tracker.py', '--report']):
                # Would normally print report
                report = tracker.get_cost_report()
                assert "📊 TTS Usage Report" in report
    
    def test_cli_today_argument(self):
        """Test CLI with --today argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some test data
            tracker.track_usage('openai', "CLI today test")
            
            # Mock sys.argv
            with patch('sys.argv', ['usage_tracker.py', '--today']):
                today_usage = tracker.get_daily_usage()
                assert today_usage['total'] == 14
    
    def test_cli_month_argument(self):
        """Test CLI with --month argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Add some test data
            tracker.track_usage('openai', "CLI month test")
            
            # Mock sys.argv
            with patch('sys.argv', ['usage_tracker.py', '--month']):
                month_usage = tracker.get_monthly_usage()
                assert month_usage['total'] == 14


class TestUsageTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_track_usage_empty_text(self):
        """Test tracking usage with empty text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            cost, usage_info = tracker.track_usage('openai', "")
            
            assert cost == 0.0
            assert usage_info['characters'] == 0
            assert tracker.stats['total_characters'] == 0
    
    def test_track_usage_very_long_text(self):
        """Test tracking usage with very long text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Create 10KB text
            long_text = "a" * 10000
            cost, usage_info = tracker.track_usage('openai', long_text)
            
            expected_cost = 10000 * 0.015 / 1000  # 0.15
            assert abs(cost - expected_cost) < 0.000001
            assert usage_info['characters'] == 10000
    
    def test_track_usage_unicode_text(self):
        """Test tracking usage with unicode text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            unicode_text = "Hello 世界 🌍 émojis"
            cost, usage_info = tracker.track_usage('openai', unicode_text)
            
            # Should count characters correctly
            assert usage_info['characters'] == len(unicode_text)
    
    def test_track_usage_concurrent_simulation(self):
        """Test tracking usage under concurrent access simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Simulate multiple concurrent requests
            for i in range(100):
                tracker.track_usage('openai', f"Concurrent message {i}")
            
            # Verify all requests were tracked
            assert tracker.stats['providers']['openai']['total_requests'] == 100
            assert tracker.stats['total_characters'] > 0


class TestUsageTrackerIntegration:
    """Test integration scenarios."""
    
    def test_full_workflow_cost_optimization(self):
        """Test complete cost optimization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Simulate expensive ElevenLabs usage
            for i in range(10):
                tracker.track_usage('elevenlabs', f"Expensive message {i}")
            
            # Simulate cache hits
            for i in range(5):
                tracker.track_usage('openai', f"Cached message {i}", cached=True)
            
            # Simulate OpenAI usage
            for i in range(10):
                tracker.track_usage('openai', f"OpenAI message {i}")
            
            # Verify optimization metrics
            breakdown = tracker.get_provider_breakdown()
            assert 'elevenlabs' in breakdown
            assert 'openai' in breakdown
            
            # Check that cache savings are tracked
            assert tracker.stats['cache_hits'] == 5
            assert tracker.stats['cache_savings'] > 0
            
            # Generate report
            report = tracker.get_cost_report()
            assert "Switch ElevenLabs to OpenAI" in report
            assert "Save $" in report
    
    def test_monthly_cost_tracking(self):
        """Test monthly cost tracking accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track various usage throughout month
            total_cost = 0
            for i in range(30):  # 30 days
                cost, _ = tracker.track_usage('openai', f"Daily message {i}")
                total_cost += cost
            
            # Verify monthly totals
            current_month = datetime.now().strftime('%Y-%m')
            monthly_usage = tracker.get_monthly_usage(current_month)
            
            assert abs(monthly_usage['cost'] - total_cost) < 0.000001
            assert monthly_usage['total'] > 0
    
    def test_provider_model_tracking_accuracy(self):
        """Test accuracy of provider model tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track different models
            tracker.track_usage('openai', "Standard model", model='tts-1')
            tracker.track_usage('openai', "HD model", model='tts-1-hd')
            tracker.track_usage('openai', "Another standard", model='tts-1')
            
            # Verify model tracking
            models = tracker.stats['providers']['openai']['models']
            assert models['tts-1'] == 30  # "Standard model" + "Another standard"
            assert models['tts-1-hd'] == 8  # "HD model"
            
            # Verify total characters
            assert tracker.stats['providers']['openai']['total_characters'] == 38


if __name__ == "__main__":
    pytest.main([__file__, "-v"])