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
Basic voice synthesis tests for ElevenLabs TTS provider.
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
    speak_with_elevenlabs,
    get_available_voices,
    validate_voice_settings,
    find_voice_by_name,
    validate_voice_id
)

class TestBasicVoiceSynthesis:
    """Test basic voice synthesis functionality."""
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_default_voice(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with default voice (Rachel)."""
        with patch('requests.post', return_value=mock_requests_success):
            result = speak_with_elevenlabs("Hello, world!")
            
            assert result is True
            mock_pygame.init.assert_called_once()
            mock_pygame.music.load.assert_called_once()
            mock_pygame.music.play.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_custom_voice_id(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with custom voice ID."""
        custom_voice_id = "AZnzlk1XvdvUeBnXmlld"  # Domi voice
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Hello, world!", voice_id=custom_voice_id)
            
            assert result is True
            # Verify the correct voice ID was used in the API call
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert custom_voice_id in str(args[0])
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_custom_voice_settings(self, mock_env, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with custom voice settings."""
        voice_settings = {
            "stability": 0.8,
            "similarity_boost": 0.9
        }
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            result = speak_with_elevenlabs("Hello, world!", voice_settings=voice_settings)
            
            assert result is True
            # Verify the voice settings were passed to the API
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs['json']['voice_settings'] == voice_settings
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_invalid_voice_settings(self, mock_env):
        """Test speaking with invalid voice settings."""
        invalid_settings = {
            "stability": 1.5,  # Invalid: > 1.0
            "similarity_boost": -0.1  # Invalid: < 0.0
        }
        
        result = speak_with_elevenlabs("Hello, world!", voice_settings=invalid_settings)
        assert result is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_api_error(self, mock_env, mock_requests_error):
        """Test speaking with API error response."""
        with patch('requests.post', return_value=mock_requests_error):
            result = speak_with_elevenlabs("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_without_api_key(self):
        """Test speaking without API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = speak_with_elevenlabs("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_network_error(self, mock_env):
        """Test speaking with network error."""
        with patch('requests.post', side_effect=Exception("Network error")):
            result = speak_with_elevenlabs("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_pygame_error(self, mock_env, mock_requests_success, mock_tempfile, mock_os_unlink):
        """Test speaking with pygame error."""
        with patch('requests.post', return_value=mock_requests_success):
            with patch('pygame.mixer.init', side_effect=Exception("Pygame error")):
                result = speak_with_elevenlabs("Hello, world!")
                assert result is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_environment_voice_id(self, mock_pygame, mock_tempfile, mock_os_unlink, mock_requests_success):
        """Test speaking with voice ID from environment variable."""
        custom_voice_id = "EXAVITQu4vr4xnSDxMaL"  # Bella voice
        
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'ELEVENLABS_VOICE_ID': custom_voice_id
        }):
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Hello, world!")
                
                assert result is True
                # Verify the environment voice ID was used
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert custom_voice_id in str(args[0])
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_environment_voice_settings(self, mock_pygame, mock_tempfile, mock_os_unlink, mock_requests_success):
        """Test speaking with voice settings from environment variables."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'ELEVENLABS_STABILITY': '0.7',
            'ELEVENLABS_SIMILARITY_BOOST': '0.8'
        }):
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Hello, world!")
                
                assert result is True
                # Verify the environment voice settings were used
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                expected_settings = {
                    "stability": 0.7,
                    "similarity_boost": 0.8
                }
                assert kwargs['json']['voice_settings'] == expected_settings
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_speak_with_environment_model_id(self, mock_pygame, mock_tempfile, mock_os_unlink, mock_requests_success):
        """Test speaking with model ID from environment variable."""
        custom_model_id = "eleven_multilingual_v2"
        
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'ELEVENLABS_MODEL_ID': custom_model_id
        }):
            with patch('requests.post', return_value=mock_requests_success) as mock_post:
                result = speak_with_elevenlabs("Hello, world!")
                
                assert result is True
                # Verify the environment model ID was used
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert kwargs['json']['model_id'] == custom_model_id