#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
# ]
# ///

"""
Comprehensive test suite for TTS Cache Manager.
Tests the TTSCache class functionality for cost optimization through caching.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta
import pytest

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tts.cache_manager import TTSCache


class TestTTSCacheInitialization:
    """Test TTSCache initialization and setup."""
    
    def test_cache_init_default_location(self):
        """Test cache initialization with default location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('pathlib.Path.expanduser', return_value=Path(tmpdir)):
                cache = TTSCache()
                assert cache.cache_dir == Path(tmpdir)
                assert cache.cache_dir.exists()
                assert cache.cache_file == Path(tmpdir) / "phrase_cache.json"
    
    def test_cache_init_custom_location(self):
        """Test cache initialization with custom location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_cache"
            cache = TTSCache(cache_dir=str(custom_dir))
            assert cache.cache_dir == custom_dir
            assert cache.cache_dir.exists()
    
    def test_cache_init_creates_directory(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "new_cache_dir"
            assert not cache_dir.exists()
            
            cache = TTSCache(cache_dir=str(cache_dir))
            assert cache_dir.exists()
            assert cache.cache_dir == cache_dir

    def test_cache_init_loads_existing_cache(self):
        """Test loading existing cache file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "phrase_cache.json"
            
            # Create existing cache file
            existing_cache = {
                "test_key": {
                    "text": "Test message",
                    "provider": "openai",
                    "created": "2024-01-01T00:00:00"
                }
            }
            with open(cache_file, 'w') as f:
                json.dump(existing_cache, f)
            
            cache = TTSCache(cache_dir=str(cache_dir))
            assert cache.cache == existing_cache
    
    def test_cache_init_handles_corrupted_cache(self):
        """Test handling corrupted cache file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "phrase_cache.json"
            
            # Create corrupted cache file
            with open(cache_file, 'w') as f:
                f.write("invalid json content")
            
            with patch('sys.stderr'):
                cache = TTSCache(cache_dir=str(cache_dir))
                assert cache.cache == {}


class TestTTSCacheKeyGeneration:
    """Test cache key generation logic."""
    
    def test_get_cache_key_basic(self):
        """Test basic cache key generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key = cache.get_cache_key("Hello world", "openai")
            assert isinstance(key, str)
            assert len(key) == 32  # MD5 hash length
    
    def test_get_cache_key_with_voice(self):
        """Test cache key generation with voice parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key1 = cache.get_cache_key("Hello world", "openai", "nova")
            key2 = cache.get_cache_key("Hello world", "openai", "onyx")
            
            assert key1 != key2
            assert len(key1) == 32
            assert len(key2) == 32
    
    def test_get_cache_key_case_insensitive(self):
        """Test that cache keys are case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key1 = cache.get_cache_key("Hello World", "openai")
            key2 = cache.get_cache_key("hello world", "openai")
            
            assert key1 == key2
    
    def test_get_cache_key_whitespace_normalization(self):
        """Test that whitespace is normalized in cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key1 = cache.get_cache_key("  hello world  ", "openai")
            key2 = cache.get_cache_key("hello world", "openai")
            
            assert key1 == key2
    
    def test_get_cache_key_different_providers(self):
        """Test that different providers generate different keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key1 = cache.get_cache_key("Hello world", "openai")
            key2 = cache.get_cache_key("Hello world", "elevenlabs")
            
            assert key1 != key2
    
    def test_get_cache_key_special_characters(self):
        """Test cache key generation with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            special_text = "Hello! @#$%^&*()_+{}|:<>?[]\\;'\",./"
            key = cache.get_cache_key(special_text, "openai")
            
            assert isinstance(key, str)
            assert len(key) == 32
    
    def test_get_cache_key_unicode(self):
        """Test cache key generation with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            unicode_text = "Hello 世界 🌍 émojis"
            key = cache.get_cache_key(unicode_text, "openai")
            
            assert isinstance(key, str)
            assert len(key) == 32


class TestTTSCacheAudioHandling:
    """Test audio caching and retrieval functionality."""
    
    def test_get_audio_path_cache_miss(self):
        """Test getting audio path when not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            result = cache.get_audio_path("Not cached", "openai")
            assert result is None
    
    def test_save_audio_basic(self):
        """Test basic audio saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            audio_data = b"fake audio data"
            audio_path = cache.save_audio("Test message", "openai", audio_data)
            
            assert Path(audio_path).exists()
            assert Path(audio_path).suffix == ".mp3"
            
            # Verify file content
            with open(audio_path, 'rb') as f:
                assert f.read() == audio_data
    
    def test_save_audio_with_voice(self):
        """Test audio saving with voice parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            audio_data = b"fake audio data"
            audio_path = cache.save_audio("Test message", "openai", audio_data, "nova")
            
            assert Path(audio_path).exists()
            
            # Verify cache metadata includes voice
            key = cache.get_cache_key("Test message", "openai", "nova")
            assert key in cache.cache
            assert cache.cache[key]['voice'] == "nova"
    
    def test_save_audio_metadata(self):
        """Test that audio saving creates correct metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            audio_data = b"fake audio data"
            text = "Test message for metadata"
            
            cache.save_audio(text, "openai", audio_data)
            
            key = cache.get_cache_key(text, "openai")
            metadata = cache.cache[key]
            
            assert metadata['text'] == text
            assert metadata['text_length'] == len(text)
            assert metadata['provider'] == "openai"
            assert metadata['file_size'] == len(audio_data)
            assert metadata['hit_count'] == 0
            assert 'created' in metadata
            assert 'last_accessed' in metadata
    
    def test_save_audio_long_text_truncation(self):
        """Test that long text is truncated in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create text longer than 100 characters
            long_text = "This is a very long text that should be truncated in the metadata " * 3
            audio_data = b"fake audio data"
            
            cache.save_audio(long_text, "openai", audio_data)
            
            key = cache.get_cache_key(long_text, "openai")
            metadata = cache.cache[key]
            
            assert len(metadata['text']) == 100
            assert metadata['text_length'] == len(long_text)
    
    def test_get_audio_path_cache_hit(self):
        """Test getting audio path when cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Save audio first
            audio_data = b"fake audio data"
            text = "Cached message"
            saved_path = cache.save_audio(text, "openai", audio_data)
            
            # Reset hit count
            key = cache.get_cache_key(text, "openai")
            cache.cache[key]['hit_count'] = 0
            
            # Get cached audio
            retrieved_path = cache.get_audio_path(text, "openai")
            
            assert retrieved_path == saved_path
            assert Path(retrieved_path).exists()
            assert cache.cache[key]['hit_count'] == 1
    
    def test_get_audio_path_updates_stats(self):
        """Test that getting audio path updates statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Save audio first
            audio_data = b"fake audio data"
            text = "Stats test message"
            cache.save_audio(text, "openai", audio_data)
            
            key = cache.get_cache_key(text, "openai")
            original_last_accessed = cache.cache[key]['last_accessed']
            
            # Small delay to ensure timestamp difference
            import time
            time.sleep(0.01)
            
            cache.get_audio_path(text, "openai")
            
            # Check that stats were updated
            assert cache.cache[key]['hit_count'] == 1
            assert cache.cache[key]['last_accessed'] != original_last_accessed
    
    def test_get_audio_path_missing_file_cleanup(self):
        """Test cleanup when cache entry exists but file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create cache entry without file
            key = cache.get_cache_key("Missing file", "openai")
            cache.cache[key] = {
                'text': "Missing file",
                'provider': "openai",
                'created': datetime.now().isoformat()
            }
            
            # Try to get audio path
            result = cache.get_audio_path("Missing file", "openai")
            
            assert result is None
            assert key not in cache.cache


class TestTTSCacheAudioPlayback:
    """Test audio playback functionality."""
    
    @patch('subprocess.run')
    def test_play_cached_audio_mpv_available(self, mock_run):
        """Test playing cached audio when mpv is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Mock mpv being available
            mock_run.side_effect = [
                Mock(returncode=0),  # which mpv
                Mock(returncode=0)   # mpv playback
            ]
            
            audio_path = "/fake/path/audio.mp3"
            result = cache.play_cached_audio(audio_path)
            
            assert result is True
            assert mock_run.call_count == 2
            mock_run.assert_any_call(['which', 'mpv'], capture_output=True)
            mock_run.assert_any_call(['mpv', audio_path], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
    
    @patch('subprocess.run')
    def test_play_cached_audio_fallback_players(self, mock_run):
        """Test fallback to other players when mpv is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Mock mpv not available, but mplayer is
            mock_run.side_effect = [
                Mock(returncode=1),  # which mpv - not found
                Mock(returncode=0),  # which mplayer - found
                Mock(returncode=0)   # mplayer playback
            ]
            
            audio_path = "/fake/path/audio.mp3"
            result = cache.play_cached_audio(audio_path)
            
            assert result is True
            assert mock_run.call_count == 3
    
    @patch('subprocess.run')
    def test_play_cached_audio_no_players(self, mock_run):
        """Test behavior when no audio players are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Mock all players not available
            mock_run.return_value = Mock(returncode=1)
            
            audio_path = "/fake/path/audio.mp3"
            with patch('sys.stderr'):
                result = cache.play_cached_audio(audio_path)
            
            assert result is False
    
    @patch('subprocess.run')
    def test_play_cached_audio_exception_handling(self, mock_run):
        """Test exception handling during audio playback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Mock subprocess raising exception
            mock_run.side_effect = Exception("Test exception")
            
            audio_path = "/fake/path/audio.mp3"
            with patch('sys.stderr'):
                result = cache.play_cached_audio(audio_path)
            
            assert result is False


class TestTTSCachePersistence:
    """Test cache save/load functionality."""
    
    def test_save_cache_basic(self):
        """Test basic cache saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Add some data to cache
            cache.cache["test_key"] = {
                "text": "Test message",
                "provider": "openai"
            }
            
            cache.save_cache()
            
            # Verify file was created and contains correct data
            assert cache.cache_file.exists()
            with open(cache.cache_file, 'r') as f:
                loaded_data = json.load(f)
                assert loaded_data == cache.cache
    
    def test_save_cache_exception_handling(self):
        """Test exception handling during cache save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Mock file operations to raise exception
            with patch('builtins.open', side_effect=Exception("Test exception")):
                with patch('sys.stderr'):
                    cache.save_cache()  # Should not raise exception
    
    def test_load_cache_basic(self):
        """Test basic cache loading functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "phrase_cache.json"
            
            # Create cache file
            test_data = {
                "test_key": {
                    "text": "Test message",
                    "provider": "openai"
                }
            }
            with open(cache_file, 'w') as f:
                json.dump(test_data, f)
            
            cache = TTSCache(cache_dir=str(cache_dir))
            assert cache.cache == test_data
    
    def test_load_cache_missing_file(self):
        """Test loading cache when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            assert cache.cache == {}
    
    def test_load_cache_exception_handling(self):
        """Test exception handling during cache load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "phrase_cache.json"
            
            # Create invalid JSON file
            with open(cache_file, 'w') as f:
                f.write("invalid json")
            
            with patch('sys.stderr'):
                cache = TTSCache(cache_dir=str(cache_dir))
                assert cache.cache == {}


class TestTTSCacheStatistics:
    """Test cache statistics functionality."""
    
    def test_get_cache_stats_empty_cache(self):
        """Test getting statistics from empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            stats = cache.get_cache_stats()
            
            assert stats['total_entries'] == 0
            assert stats['total_size_bytes'] == 0
            assert stats['total_size_mb'] == 0
            assert stats['total_hits'] == 0
            assert stats['providers'] == {}
            assert stats['cache_directory'] == str(cache.cache_dir)
    
    def test_get_cache_stats_with_data(self):
        """Test getting statistics with cache data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Add test data
            cache.cache["key1"] = {
                "provider": "openai",
                "file_size": 1024,
                "hit_count": 5
            }
            cache.cache["key2"] = {
                "provider": "openai",
                "file_size": 2048,
                "hit_count": 3
            }
            cache.cache["key3"] = {
                "provider": "elevenlabs",
                "file_size": 512,
                "hit_count": 2
            }
            
            stats = cache.get_cache_stats()
            
            assert stats['total_entries'] == 3
            assert stats['total_size_bytes'] == 3584
            assert stats['total_size_mb'] == 0.0  # Too small for MB
            assert stats['total_hits'] == 10
            
            # Check provider breakdown
            assert 'openai' in stats['providers']
            assert 'elevenlabs' in stats['providers']
            
            openai_stats = stats['providers']['openai']
            assert openai_stats['count'] == 2
            assert openai_stats['size'] == 3072
            assert openai_stats['hits'] == 8
            
            elevenlabs_stats = stats['providers']['elevenlabs']
            assert elevenlabs_stats['count'] == 1
            assert elevenlabs_stats['size'] == 512
            assert elevenlabs_stats['hits'] == 2
    
    def test_get_cache_stats_missing_fields(self):
        """Test getting statistics with missing fields in cache data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Add test data with missing fields
            cache.cache["key1"] = {
                "provider": "openai"
                # missing file_size and hit_count
            }
            
            stats = cache.get_cache_stats()
            
            assert stats['total_entries'] == 1
            assert stats['total_size_bytes'] == 0
            assert stats['total_hits'] == 0
            assert stats['providers']['openai']['size'] == 0
            assert stats['providers']['openai']['hits'] == 0


class TestTTSCacheCleanup:
    """Test cache cleanup functionality."""
    
    def test_cleanup_old_cache_no_old_files(self):
        """Test cleanup when no old files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Add recent cache entry
            cache.cache["recent_key"] = {
                "last_accessed": datetime.now().isoformat()
            }
            
            removed = cache.cleanup_old_cache(days=30)
            assert removed == 0
            assert "recent_key" in cache.cache
    
    def test_cleanup_old_cache_removes_old_files(self):
        """Test cleanup removes old files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create old cache entry with actual file
            old_date = datetime.now() - timedelta(days=35)
            key = "old_key"
            cache.cache[key] = {
                "last_accessed": old_date.isoformat()
            }
            
            # Create corresponding audio file
            audio_file = cache.cache_dir / f"{key}.mp3"
            audio_file.write_bytes(b"old audio data")
            
            # Add recent cache entry
            cache.cache["recent_key"] = {
                "last_accessed": datetime.now().isoformat()
            }
            
            removed = cache.cleanup_old_cache(days=30)
            
            assert removed == 1
            assert key not in cache.cache
            assert "recent_key" in cache.cache
            assert not audio_file.exists()
    
    def test_cleanup_old_cache_missing_files(self):
        """Test cleanup handles missing audio files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create old cache entry without audio file
            old_date = datetime.now() - timedelta(days=35)
            key = "old_key_no_file"
            cache.cache[key] = {
                "last_accessed": old_date.isoformat()
            }
            
            removed = cache.cleanup_old_cache(days=30)
            
            assert removed == 1
            assert key not in cache.cache
    
    def test_cleanup_old_cache_custom_days(self):
        """Test cleanup with custom days parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create cache entry that's 10 days old
            old_date = datetime.now() - timedelta(days=10)
            key = "ten_day_old"
            cache.cache[key] = {
                "last_accessed": old_date.isoformat()
            }
            
            # Cleanup with 5 days - should remove
            removed = cache.cleanup_old_cache(days=5)
            assert removed == 1
            assert key not in cache.cache
    
    def test_cleanup_old_cache_saves_cache(self):
        """Test that cleanup saves cache after removing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create old cache entry
            old_date = datetime.now() - timedelta(days=35)
            key = "old_key"
            cache.cache[key] = {
                "last_accessed": old_date.isoformat()
            }
            
            # Mock save_cache to verify it's called
            with patch.object(cache, 'save_cache') as mock_save:
                cache.cleanup_old_cache(days=30)
                mock_save.assert_called_once()


class TestTTSCacheCommonPhrases:
    """Test common phrases functionality."""
    
    def test_precache_common_phrases(self):
        """Test getting list of common phrases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            phrases = cache.precache_common_phrases()
            
            assert isinstance(phrases, list)
            assert len(phrases) > 0
            assert "Build complete" in phrases
            assert "Tests passed" in phrases
            assert "Deployment successful" in phrases


class TestTTSCacheEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_cache_with_empty_text(self):
        """Test cache handling with empty text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key = cache.get_cache_key("", "openai")
            assert isinstance(key, str)
            assert len(key) == 32
    
    def test_cache_with_none_voice(self):
        """Test cache handling with None voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            key1 = cache.get_cache_key("test", "openai", None)
            key2 = cache.get_cache_key("test", "openai")
            
            assert key1 == key2
    
    def test_cache_with_very_long_text(self):
        """Test cache handling with very long text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create very long text (10KB)
            long_text = "a" * 10000
            audio_data = b"audio data"
            
            cache.save_audio(long_text, "openai", audio_data)
            
            # Verify it works
            key = cache.get_cache_key(long_text, "openai")
            assert key in cache.cache
            assert cache.cache[key]['text_length'] == 10000
            assert len(cache.cache[key]['text']) == 100  # Truncated to 100 chars
    
    def test_cache_with_binary_audio_data(self):
        """Test cache handling with various binary audio data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Test with empty binary data
            empty_data = b""
            path = cache.save_audio("empty test", "openai", empty_data)
            assert Path(path).exists()
            assert Path(path).stat().st_size == 0
            
            # Test with large binary data
            large_data = b"x" * 1000000  # 1MB
            path = cache.save_audio("large test", "openai", large_data)
            assert Path(path).exists()
            assert Path(path).stat().st_size == 1000000
    
    def test_cache_concurrent_access(self):
        """Test cache behavior under concurrent access simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Simulate multiple saves
            for i in range(10):
                text = f"concurrent message {i}"
                audio_data = f"audio data {i}".encode()
                cache.save_audio(text, "openai", audio_data)
            
            # Verify all saved
            assert len(cache.cache) == 10
            
            # Simulate multiple gets
            for i in range(10):
                text = f"concurrent message {i}"
                result = cache.get_audio_path(text, "openai")
                assert result is not None
                assert Path(result).exists()


class TestTTSCacheIntegration:
    """Test cache integration scenarios."""
    
    def test_full_workflow_cache_miss_then_hit(self):
        """Test complete workflow from cache miss to hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            text = "Integration test message"
            provider = "openai"
            voice = "nova"
            audio_data = b"fake audio data for integration test"
            
            # Initial cache miss
            result = cache.get_audio_path(text, provider, voice)
            assert result is None
            
            # Save audio
            saved_path = cache.save_audio(text, provider, audio_data, voice)
            assert Path(saved_path).exists()
            
            # Cache hit
            hit_result = cache.get_audio_path(text, provider, voice)
            assert hit_result == saved_path
            
            # Verify statistics
            stats = cache.get_cache_stats()
            assert stats['total_entries'] == 1
            assert stats['total_hits'] == 1
            assert stats['providers'][provider]['count'] == 1
            assert stats['providers'][provider]['hits'] == 1
    
    def test_cache_persistence_across_instances(self):
        """Test that cache persists across different instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            cache1 = TTSCache(cache_dir=tmpdir)
            text = "Persistence test"
            audio_data = b"persistent audio data"
            
            saved_path = cache1.save_audio(text, "openai", audio_data)
            
            # Second instance
            cache2 = TTSCache(cache_dir=tmpdir)
            result = cache2.get_audio_path(text, "openai")
            
            assert result == saved_path
            assert Path(result).exists()
            assert len(cache2.cache) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])