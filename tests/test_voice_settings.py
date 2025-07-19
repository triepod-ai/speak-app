#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "requests>=2.31.0",
#   "pygame>=2.5.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
Voice settings customization tests for ElevenLabs TTS provider.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to the path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from elevenlabs_tts import (
    validate_voice_settings,
    speak_with_elevenlabs
)

class TestVoiceSettings:
    """Test voice settings validation and customization."""
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_settings_valid(self, voice_settings_test_cases):
        """Test voice settings validation with valid settings."""
        for test_case in voice_settings_test_cases:
            if test_case["expected_success"]:
                settings = {
                    "stability": test_case["stability"],
                    "similarity_boost": test_case["similarity_boost"]
                }
                assert validate_voice_settings(settings) is True
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_settings_invalid(self, voice_settings_test_cases):
        """Test voice settings validation with invalid settings."""
        for test_case in voice_settings_test_cases:
            if not test_case["expected_success"]:
                settings = {
                    "stability": test_case["stability"],
                    "similarity_boost": test_case["similarity_boost"]
                }
                assert validate_voice_settings(settings) is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_settings_boundary_values(self):
        """Test voice settings validation with boundary values."""
        # Test minimum valid values
        assert validate_voice_settings({"stability": 0.0, "similarity_boost": 0.0}) is True
        
        # Test maximum valid values
        assert validate_voice_settings({"stability": 1.0, "similarity_boost": 1.0}) is True
        
        # Test just below minimum
        assert validate_voice_settings({"stability": -0.001, "similarity_boost": 0.5}) is False
        assert validate_voice_settings({"stability": 0.5, "similarity_boost": -0.001}) is False
        
        # Test just above maximum
        assert validate_voice_settings({"stability": 1.001, "similarity_boost": 0.5}) is False
        assert validate_voice_settings({"stability": 0.5, "similarity_boost": 1.001}) is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_settings_partial_settings(self):
        """Test voice settings validation with partial settings."""
        # Test with only stability
        assert validate_voice_settings({"stability": 0.5}) is True
        
        # Test with only similarity_boost
        assert validate_voice_settings({"similarity_boost": 0.5}) is True
        
        # Test with empty settings
        assert validate_voice_settings({}) is True
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_settings_invalid_types(self):
        """Test voice settings validation with invalid types."""
        # Test with non-dict
        assert validate_voice_settings("not a dict") is False
        assert validate_voice_settings(None) is False
        assert validate_voice_settings(123) is False
        assert validate_voice_settings([]) is False
        
        # Test with non-numeric values
        assert validate_voice_settings({"stability": "0.5", "similarity_boost": 0.5}) is False
        assert validate_voice_settings({"stability": 0.5, "similarity_boost": "0.5"}) is False
        assert validate_voice_settings({"stability": None, "similarity_boost": 0.5}) is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_custom_stability(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with custom stability settings."""
        stability_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        
        for stability in stability_values:
            voice_settings = {
                "stability": stability,
                "similarity_boost": 0.5
            }
            
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Test message", voice_settings=voice_settings)
                
                assert result is True
                # Verify the stability value was passed correctly
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert kwargs['json']['voice_settings']['stability'] == stability
                mock_post.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_custom_similarity_boost(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with custom similarity boost settings."""
        similarity_boost_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for similarity_boost in similarity_boost_values:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": similarity_boost
            }
            
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Test message", voice_settings=voice_settings)
                
                assert result is True
                # Verify the similarity boost value was passed correctly
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert kwargs['json']['voice_settings']['similarity_boost'] == similarity_boost
                mock_post.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_extreme_settings(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with extreme but valid settings."""
        # Test very low settings (robotic, monotone)
        low_settings = {
            "stability": 0.0,
            "similarity_boost": 0.0
        }
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Test message", voice_settings=low_settings)
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs['json']['voice_settings'] == low_settings
        
        # Test very high settings (emotional, expressive)
        high_settings = {
            "stability": 1.0,
            "similarity_boost": 1.0
        }
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Test message", voice_settings=high_settings)
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs['json']['voice_settings'] == high_settings
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_partial_settings(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with partial voice settings."""
        # Test with only stability
        stability_only = {"stability": 0.7}
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Test message", voice_settings=stability_only)
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            # Should include the provided stability and default similarity_boost
            voice_settings = kwargs['json']['voice_settings']
            assert voice_settings['stability'] == 0.7
            assert 'similarity_boost' in voice_settings
        
        # Test with only similarity_boost
        similarity_only = {"similarity_boost": 0.9}
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Test message", voice_settings=similarity_only)
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            # Should include the provided similarity_boost and default stability
            voice_settings = kwargs['json']['voice_settings']
            assert voice_settings['similarity_boost'] == 0.9
            assert 'stability' in voice_settings
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_invalid_settings_rejected(self, mock_env):
        """Test that invalid voice settings are rejected."""
        invalid_settings = [
            {"stability": -0.1, "similarity_boost": 0.5},
            {"stability": 0.5, "similarity_boost": -0.1},
            {"stability": 1.1, "similarity_boost": 0.5},
            {"stability": 0.5, "similarity_boost": 1.1},
            {"stability": "0.5", "similarity_boost": 0.5},
            {"stability": 0.5, "similarity_boost": "0.5"},
            {"stability": None, "similarity_boost": 0.5},
            {"stability": 0.5, "similarity_boost": None},
        ]
        
        for settings in invalid_settings:
            result = speak_with_elevenlabs("Test message", voice_settings=settings)
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_voice_settings_with_voice_selection(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test voice settings combined with voice selection."""
        voice_id = "AZnzlk1XvdvUeBnXmlld"  # Domi voice
        voice_settings = {
            "stability": 0.3,   # More variable/emotional
            "similarity_boost": 0.9  # High similarity to original
        }
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Test message", voice_id=voice_id, voice_settings=voice_settings)
            
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            
            # Verify both voice ID and settings were used
            assert voice_id in str(args[0])
            assert kwargs['json']['voice_settings'] == voice_settings
    
    @pytest.mark.integration
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_voice_settings_priority_order(self, mock_pygame, mock_tempfile, mock_os_unlink, mock_requests_success):
        """Test that explicit settings override environment settings."""
        # Set environment voice settings
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'ELEVENLABS_STABILITY': '0.2',
            'ELEVENLABS_SIMILARITY_BOOST': '0.3'
        }):
            # Override with explicit settings
            explicit_settings = {
                "stability": 0.8,
                "similarity_boost": 0.9
            }
            
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Test message", voice_settings=explicit_settings)
                
                assert result is True
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                
                # Verify explicit settings were used, not environment settings
                assert kwargs['json']['voice_settings'] == explicit_settings