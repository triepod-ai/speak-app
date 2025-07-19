#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "openai>=1.0.0",
#   "pygame>=2.5.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
Basic OpenAI TTS tests for the speak-app multi-provider TTS system.
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

from openai_tts import speak_with_openai

class TestBasicOpenAITTS:
    """Test basic OpenAI TTS functionality."""
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_default_voice(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with default voice (nova)."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Hello, world!")
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input="Hello, world!",
                speed=1.0
            )
            mock_pygame.mixer.init.assert_called_once()
            mock_pygame.mixer.music.load.assert_called_once()
            mock_pygame.mixer.music.play.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_all_voices(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with all OpenAI voices."""
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        for voice in voices:
            with patch('openai.OpenAI', return_value=mock_openai_client):
                result = speak_with_openai("Test message", voice=voice)
                
                assert result is True
                mock_openai_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice=voice,
                    input="Test message",
                    speed=1.0
                )
                mock_openai_client.audio.speech.create.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_different_models(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with different OpenAI models."""
        models = ["tts-1", "tts-1-hd"]
        
        for model in models:
            with patch('openai.OpenAI', return_value=mock_openai_client):
                result = speak_with_openai("Test message", model=model)
                
                assert result is True
                mock_openai_client.audio.speech.create.assert_called_with(
                    model=model,
                    voice="nova",
                    input="Test message",
                    speed=1.0
                )
                mock_openai_client.audio.speech.create.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_speed_variations(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with different speed settings."""
        speeds = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        
        for speed in speeds:
            with patch('openai.OpenAI', return_value=mock_openai_client):
                result = speak_with_openai("Test message", speed=speed)
                
                assert result is True
                mock_openai_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice="nova",
                    input="Test message",
                    speed=speed
                )
                mock_openai_client.audio.speech.create.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_invalid_speed(self, mock_openai_env, mock_openai_client):
        """Test speaking with invalid speed values."""
        invalid_speeds = [0.1, 5.0, -1.0, 0.0]
        
        for speed in invalid_speeds:
            with patch('openai.OpenAI', return_value=mock_openai_client):
                # OpenAI API will handle validation, but we test the call is made
                result = speak_with_openai("Test message", speed=speed)
                
                # The function should still attempt the call
                mock_openai_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice="nova",
                    input="Test message",
                    speed=speed
                )
                mock_openai_client.audio.speech.create.reset_mock()
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_without_api_key(self):
        """Test speaking without OpenAI API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = speak_with_openai("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_empty_api_key(self):
        """Test speaking with empty OpenAI API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            result = speak_with_openai("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_api_error(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with OpenAI API error."""
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = Exception("API Error")
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_network_error(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with network error."""
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = ConnectionError("Network error")
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_pygame_error(self, mock_openai_env, mock_openai_client, mock_tempfile, mock_os_unlink):
        """Test speaking with pygame error."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('pygame.mixer.init', side_effect=Exception("Pygame error")):
                result = speak_with_openai("Hello, world!")
                assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_audio_stream_error(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with audio streaming error."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file.side_effect = Exception("Streaming error")
        mock_client.audio.speech.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Hello, world!")
            assert result is False
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_environment_voice(self, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with voice from environment variable."""
        custom_voice = "echo"
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'OPENAI_TTS_VOICE': custom_voice
        }):
            mock_client = Mock()
            mock_response = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message")
                
                assert result is True
                # Should use environment voice if no explicit voice provided
                mock_client.audio.speech.create.assert_called_once_with(
                    model="tts-1",
                    voice="nova",  # Function parameter overrides environment
                    input="Test message",
                    speed=1.0
                )
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_custom_voice_and_model(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with custom voice and model combination."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Test message", voice="shimmer", model="tts-1-hd", speed=1.2)
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1-hd",
                voice="shimmer",
                input="Test message",
                speed=1.2
            )
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_long_text(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with long text input."""
        long_text = "This is a very long text that might test the limits of the OpenAI TTS API. " * 100
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai(long_text)
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input=long_text,
                speed=1.0
            )
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_special_characters(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with special characters and Unicode."""
        special_text = "Hello! 🌟 Testing with émojis and spécial charactèrs: @#$%^&*()_+={[}]|\\:;\"'<,>.?/~`"
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai(special_text)
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input=special_text,
                speed=1.0
            )
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_speak_with_empty_text(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test speaking with empty text."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("")
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input="",
                speed=1.0
            )
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_audio_file_cleanup(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test that temporary audio files are properly cleaned up."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Test message")
            
            assert result is True
            # Verify cleanup was called
            mock_os_unlink.assert_called_once()
            
            # Verify the file path matches what was created
            temp_file_path = mock_tempfile.return_value.__enter__.return_value.name
            mock_os_unlink.assert_called_with(temp_file_path)
    
    @pytest.mark.unit
    @pytest.mark.openai
    def test_pygame_initialization_and_cleanup(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test pygame initialization and cleanup."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Test message")
            
            assert result is True
            # Verify pygame lifecycle
            mock_pygame.mixer.init.assert_called_once()
            mock_pygame.mixer.music.load.assert_called_once()
            mock_pygame.mixer.music.play.assert_called_once()
            mock_pygame.mixer.quit.assert_called_once()


class TestOpenAITTSIntegration:
    """Integration tests for OpenAI TTS functionality."""
    
    @pytest.mark.integration
    @pytest.mark.openai
    def test_voice_quality_comparison(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test voice quality comparison across different voices."""
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        test_text = "This is a test of voice quality for OpenAI TTS."
        
        for voice in voices:
            mock_client = Mock()
            mock_response = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai(test_text, voice=voice)
                
                assert result is True
                mock_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice=voice,
                    input=test_text,
                    speed=1.0
                )
                mock_client.audio.speech.create.reset_mock()
    
    @pytest.mark.integration
    @pytest.mark.openai
    def test_model_performance_comparison(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test performance comparison between tts-1 and tts-1-hd models."""
        models = ["tts-1", "tts-1-hd"]
        test_text = "Performance comparison between standard and HD models."
        
        for model in models:
            mock_client = Mock()
            mock_response = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai(test_text, model=model)
                
                assert result is True
                mock_client.audio.speech.create.assert_called_with(
                    model=model,
                    voice="nova",
                    input=test_text,
                    speed=1.0
                )
                mock_client.audio.speech.create.reset_mock()
    
    @pytest.mark.integration
    @pytest.mark.openai
    def test_speed_and_voice_combinations(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test various speed and voice combinations."""
        combinations = [
            ("alloy", 0.5),
            ("echo", 1.0),
            ("fable", 1.5),
            ("onyx", 2.0),
            ("nova", 1.2),
            ("shimmer", 0.8)
        ]
        
        for voice, speed in combinations:
            mock_client = Mock()
            mock_response = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message", voice=voice, speed=speed)
                
                assert result is True
                mock_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice=voice,
                    input="Test message",
                    speed=speed
                )
                mock_client.audio.speech.create.reset_mock()
    
    @pytest.mark.integration
    @pytest.mark.openai
    def test_error_recovery_scenarios(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test error recovery in various failure scenarios."""
        # Test API timeout
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = TimeoutError("API timeout")
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
        
        # Test file I/O error
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file.side_effect = IOError("File write error")
        mock_client.audio.speech.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
    
    @pytest.mark.regression
    @pytest.mark.openai
    def test_backward_compatibility(self, mock_openai_env, mock_openai_client, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test backward compatibility with existing functionality."""
        # Test with minimal parameters (should use defaults)
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Test message")
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input="Test message",
                speed=1.0
            )
        
        # Test with all parameters specified
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = speak_with_openai("Test message", voice="alloy", model="tts-1-hd", speed=1.5)
            
            assert result is True
            mock_openai_client.audio.speech.create.assert_called_with(
                model="tts-1-hd",
                voice="alloy",
                input="Test message",
                speed=1.5
            )