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
OpenAI API integration tests for the speak-app TTS system.
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

class TestOpenAIAPIIntegration:
    """Test OpenAI API integration and error handling."""
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_key_validation(self):
        """Test API key validation scenarios."""
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            result = speak_with_openai("Test message")
            assert result is False
        
        # Test with empty API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            result = speak_with_openai("Test message")
            assert result is False
        
        # Test with valid API key format
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stream_to_file = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client):
                with patch('pygame.mixer'):
                    with patch('tempfile.NamedTemporaryFile'):
                        with patch('os.unlink'):
                            result = speak_with_openai("Test message")
                            assert result is True
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_openai_api_error_responses(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test various OpenAI API error response scenarios."""
        from openai import OpenAIError, APIError, RateLimitError, APITimeoutError
        
        error_scenarios = [
            (APIError("Bad Request", response=Mock(status_code=400), body=None), "API Error"),
            (RateLimitError("Rate limit exceeded", response=Mock(status_code=429), body=None), "Rate Limit"),
            (APITimeoutError("Request timeout"), "Timeout"),
            (OpenAIError("General OpenAI error"), "General Error"),
            (Exception("Generic error"), "Generic Error"),
        ]
        
        for error, description in error_scenarios:
            mock_client = Mock()
            mock_client.audio.speech.create.side_effect = error
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message")
                assert result is False, f"Failed for {description}"
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_network_error_handling(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test network error handling scenarios."""
        network_errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable"),
            Exception("Network error"),
        ]
        
        for error in network_errors:
            mock_client = Mock()
            mock_client.audio.speech.create.side_effect = error
            
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message")
                assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_audio_streaming_scenarios(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test audio streaming scenarios."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # Test successful streaming
        mock_response.stream_to_file = Mock()
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is True
            mock_response.stream_to_file.assert_called_once()
        
        # Test streaming error
        mock_response.stream_to_file.side_effect = Exception("Streaming error")
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_request_parameters(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test that API requests contain correct parameters."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        voice = "alloy"
        model = "tts-1-hd"
        speed = 1.5
        text = "Test message for API"
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai(text, voice=voice, model=model, speed=speed)
            
            assert result is True
            mock_client.audio.speech.create.assert_called_once_with(
                model=model,
                voice=voice,
                input=text,
                speed=speed
            )
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_rate_limiting_behavior(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test behavior when API rate limits are hit."""
        from openai import RateLimitError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_timeout_handling(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test API timeout handling."""
        from openai import APITimeoutError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APITimeoutError("Request timeout")
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_invalid_voice_parameter(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling of invalid voice parameters."""
        from openai import APIError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APIError(
            "Invalid voice parameter",
            response=Mock(status_code=400),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message", voice="invalid_voice")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_invalid_model_parameter(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling of invalid model parameters."""
        from openai import APIError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APIError(
            "Invalid model parameter",
            response=Mock(status_code=400),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message", model="invalid_model")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_invalid_speed_parameter(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling of invalid speed parameters."""
        from openai import APIError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APIError(
            "Invalid speed parameter",
            response=Mock(status_code=400),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message", speed=10.0)  # Invalid speed
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_large_text_handling(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling of large text inputs."""
        large_text = "A" * 10000  # Very large text
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai(large_text)
            
            assert result is True
            mock_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input=large_text,
                speed=1.0
            )
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_text_length_limit_exceeded(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling when text length limit is exceeded."""
        from openai import APIError
        
        # OpenAI has a 4096 character limit
        very_large_text = "A" * 5000
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APIError(
            "Text too long",
            response=Mock(status_code=400),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai(very_large_text)
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_audio_file_corruption_handling(self, mock_openai_env, mock_tempfile, mock_os_unlink):
        """Test handling of corrupted audio files."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # Mock pygame failing to load corrupted audio
        with patch('openai.OpenAI', return_value=mock_client):
            with patch('pygame.mixer.init'):
                with patch('pygame.mixer.music.load', side_effect=Exception("Corrupted audio")):
                    result = speak_with_openai("Test message")
                    assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_disk_space_error_handling(self, mock_openai_env, mock_pygame, mock_os_unlink):
        """Test handling of disk space errors."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # Mock disk space error during file creation
        with patch('openai.OpenAI', return_value=mock_client):
            with patch('tempfile.NamedTemporaryFile', side_effect=OSError("No space left on device")):
                result = speak_with_openai("Test message")
                assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_audio_playback_interruption(self, mock_openai_env, mock_openai_client, mock_tempfile, mock_os_unlink):
        """Test handling of audio playback interruption."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('pygame.mixer.init'):
                with patch('pygame.mixer.music.load'):
                    with patch('pygame.mixer.music.play'):
                        with patch('pygame.mixer.music.get_busy', side_effect=Exception("Playback interrupted")):
                            result = speak_with_openai("Test message")
                            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_concurrent_requests_handling(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test handling of concurrent API requests."""
        from openai import APIError
        
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = APIError(
            "Too many concurrent requests",
            response=Mock(status_code=429),
            body=None
        )
        
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_version_compatibility(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test compatibility with different OpenAI API versions."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # Test with different client versions
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            assert result is True
            
            # Verify the API call structure matches expected format
            mock_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input="Test message",
                speed=1.0
            )
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_environment_variable_loading(self, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test loading environment variables from different sources."""
        # Test with environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env_key'}):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stream_to_file = Mock()
            mock_client.audio.speech.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_client) as mock_openai:
                result = speak_with_openai("Test message")
                assert result is True
                
                # Verify client was created with the API key
                mock_openai.assert_called_once_with(api_key='env_key')
        
        # Test with .env file loading
        with patch.dict(os.environ, {}, clear=True):
            with patch('dotenv.load_dotenv'):
                with patch('os.getenv', return_value='dotenv_key'):
                    mock_client = Mock()
                    mock_response = Mock()
                    mock_response.stream_to_file = Mock()
                    mock_client.audio.speech.create.return_value = mock_response
                    
                    with patch('openai.OpenAI', return_value=mock_client) as mock_openai:
                        result = speak_with_openai("Test message")
                        assert result is True
                        
                        # Verify client was created with the API key
                        mock_openai.assert_called_once_with(api_key='dotenv_key')
    
    @pytest.mark.regression
    @pytest.mark.api
    @pytest.mark.openai
    def test_api_backward_compatibility(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test backward compatibility with older OpenAI API usage patterns."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # Test with minimal parameters (should use defaults)
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai("Test message")
            
            assert result is True
            mock_client.audio.speech.create.assert_called_once_with(
                model="tts-1",
                voice="nova",
                input="Test message",
                speed=1.0
            )
        
        # Test with all parameters specified
        with patch('openai.OpenAI', return_value=mock_client):
            result = speak_with_openai(
                "Test message",
                voice="shimmer",
                model="tts-1-hd",
                speed=1.2
            )
            
            assert result is True
            mock_client.audio.speech.create.assert_called_with(
                model="tts-1-hd",
                voice="shimmer",
                input="Test message",
                speed=1.2
            )
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_audio_quality_parameters(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test audio quality parameters and their effects."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        quality_tests = [
            ("tts-1", "Standard quality model"),
            ("tts-1-hd", "High quality model"),
        ]
        
        for model, description in quality_tests:
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message", model=model)
                
                assert result is True, f"Failed for {description}"
                mock_client.audio.speech.create.assert_called_with(
                    model=model,
                    voice="nova",
                    input="Test message",
                    speed=1.0
                )
                mock_client.audio.speech.create.reset_mock()
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.openai
    def test_voice_characteristic_parameters(self, mock_openai_env, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test voice characteristic parameters."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        voice_tests = [
            ("alloy", "Neutral voice"),
            ("echo", "Male voice"),
            ("fable", "British accent"),
            ("onyx", "Deep voice"),
            ("nova", "Female voice"),
            ("shimmer", "Soft voice"),
        ]
        
        for voice, description in voice_tests:
            with patch('openai.OpenAI', return_value=mock_client):
                result = speak_with_openai("Test message", voice=voice)
                
                assert result is True, f"Failed for {description}"
                mock_client.audio.speech.create.assert_called_with(
                    model="tts-1",
                    voice=voice,
                    input="Test message",
                    speed=1.0
                )
                mock_client.audio.speech.create.reset_mock()