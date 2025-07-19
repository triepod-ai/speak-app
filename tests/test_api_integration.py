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
API integration and error handling tests for ElevenLabs TTS provider.
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
    validate_voice_id,
    find_voice_by_name
)

class TestAPIIntegration:
    """Test API integration and error handling."""
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_key_validation(self):
        """Test API key validation scenarios."""
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            result = speak_with_elevenlabs("Test message")
            assert result is False
            
            voices = get_available_voices()
            assert voices is None
        
        # Test with empty API key
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': ''}):
            result = speak_with_elevenlabs("Test message")
            assert result is False
            
            voices = get_available_voices()
            assert voices is None
        
        # Test with valid API key format
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'valid_key_123'}):
            with patch('requests.post', return_value=Mock(status_code=200, content=b'audio_data')):
                with patch('pygame.mixer'):
                    with patch('tempfile.NamedTemporaryFile'):
                        with patch('os.unlink'):
                            result = speak_with_elevenlabs("Test message")
                            assert result is True
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_error_responses(self, mock_env):
        """Test various API error response scenarios."""
        error_scenarios = [
            (400, "Bad Request: Invalid voice_id"),
            (401, "Unauthorized: Invalid API key"),
            (403, "Forbidden: Access denied"),
            (404, "Not Found: Voice not found"),
            (429, "Too Many Requests: Rate limit exceeded"),
            (500, "Internal Server Error"),
            (503, "Service Unavailable"),
        ]
        
        for status_code, error_message in error_scenarios:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.text = error_message
            
            with patch('requests.post', return_value=mock_response):
                result = speak_with_elevenlabs("Test message")
                assert result is False
            
            # Test same scenarios for voice listing
            with patch('requests.get', return_value=mock_response):
                voices = get_available_voices()
                assert voices is None
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_network_error_handling(self, mock_env):
        """Test network error handling scenarios."""
        network_errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
            Exception("Network unreachable"),
            OSError("Network error"),
        ]
        
        for error in network_errors:
            # Test speech synthesis
            with patch('requests.post', side_effect=error):
                result = speak_with_elevenlabs("Test message")
                assert result is False
            
            # Test voice listing
            with patch('requests.get', side_effect=error):
                voices = get_available_voices()
                assert voices is None
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_malformed_api_responses(self, mock_env):
        """Test handling of malformed API responses."""
        # Test invalid JSON response for voice listing
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with patch('requests.get', return_value=mock_response):
            voices = get_available_voices()
            assert voices is None
        
        # Test empty response body
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b''
        
        with patch('requests.post', return_value=mock_response):
            with patch('pygame.mixer'):
                with patch('tempfile.NamedTemporaryFile'):
                    with patch('os.unlink'):
                        result = speak_with_elevenlabs("Test message")
                        assert result is True  # Should still work with empty audio
        
        # Test corrupted audio response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'corrupted_audio_data'
        
        with patch('requests.post', return_value=mock_response):
            with patch('pygame.mixer.music.load', side_effect=Exception("Corrupted audio")):
                with patch('tempfile.NamedTemporaryFile'):
                    with patch('os.unlink'):
                        result = speak_with_elevenlabs("Test message")
                        assert result is False
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_request_parameters(self, mock_env, mock_requests_success):
        """Test that API requests contain correct parameters."""
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        voice_settings = {
            "stability": 0.7,
            "similarity_boost": 0.8
        }
        model_id = "eleven_turbo_v2_5"
        text = "Test message for API"
        
        with patch('requests.post', return_value=mock_requests_success) as mock_post:
            with patch('pygame.mixer'):
                with patch('tempfile.NamedTemporaryFile'):
                    with patch('os.unlink'):
                        result = speak_with_elevenlabs(text, voice_id, voice_settings)
                        
                        assert result is True
                        mock_post.assert_called_once()
                        
                        # Verify request URL contains voice ID
                        args, kwargs = mock_post.call_args
                        assert voice_id in str(args[0])
                        
                        # Verify request headers
                        headers = kwargs['headers']
                        assert headers['Accept'] == 'audio/mpeg'
                        assert headers['Content-Type'] == 'application/json'
                        assert 'xi-api-key' in headers
                        
                        # Verify request body
                        json_data = kwargs['json']
                        assert json_data['text'] == text
                        assert json_data['model_id'] == model_id
                        assert json_data['voice_settings'] == voice_settings
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_voice_listing_api_parameters(self, mock_env, mock_requests_voices_success):
        """Test that voice listing API requests contain correct parameters."""
        with patch('requests.get', return_value=mock_requests_voices_success) as mock_get:
            voices = get_available_voices()
            
            assert voices is not None
            mock_get.assert_called_once()
            
            # Verify request URL and headers
            args, kwargs = mock_get.call_args
            assert "https://api.elevenlabs.io/v1/voices" in str(args[0])
            
            headers = kwargs['headers']
            assert headers['Accept'] == 'application/json'
            assert 'xi-api-key' in headers
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_rate_limiting_behavior(self, mock_env):
        """Test behavior when API rate limits are hit."""
        # Simulate rate limiting response
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.text = "Rate limit exceeded"
        
        with patch('requests.post', return_value=rate_limit_response):
            result = speak_with_elevenlabs("Test message")
            assert result is False
        
        # Test rate limiting for voice listing
        with patch('requests.get', return_value=rate_limit_response):
            voices = get_available_voices()
            assert voices is None
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_timeout_handling(self, mock_env):
        """Test API timeout handling."""
        from requests.exceptions import Timeout
        
        # Test timeout during speech synthesis
        with patch('requests.post', side_effect=Timeout("Request timeout")):
            result = speak_with_elevenlabs("Test message")
            assert result is False
        
        # Test timeout during voice listing
        with patch('requests.get', side_effect=Timeout("Request timeout")):
            voices = get_available_voices()
            assert voices is None
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_voice_not_found_handling(self, mock_env, mock_requests_voices_success):
        """Test handling of voice not found scenarios."""
        # Test with invalid voice ID in synthesis
        invalid_voice_response = Mock()
        invalid_voice_response.status_code = 404
        invalid_voice_response.text = "Voice not found"
        
        with patch('requests.post', return_value=invalid_voice_response):
            result = speak_with_elevenlabs("Test message", voice_id="invalid_voice_id")
            assert result is False
        
        # Test voice validation with invalid ID
        with patch('requests.get', return_value=mock_requests_voices_success):
            assert validate_voice_id("invalid_voice_id") is False
            assert find_voice_by_name("NonExistentVoice") is None
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_partial_api_failures(self, mock_env, mock_requests_voices_success):
        """Test scenarios where some API calls succeed and others fail."""
        # Voice listing succeeds but synthesis fails
        synthesis_error = Mock()
        synthesis_error.status_code = 500
        synthesis_error.text = "Internal server error"
        
        with patch('requests.get', return_value=mock_requests_voices_success):
            with patch('requests.post', return_value=synthesis_error):
                # Voice listing should work
                voices = get_available_voices()
                assert voices is not None
                assert len(voices) > 0
                
                # But synthesis should fail
                result = speak_with_elevenlabs("Test message")
                assert result is False
        
        # Voice listing fails but we can still attempt synthesis with known voice
        voices_error = Mock()
        voices_error.status_code = 503
        voices_error.text = "Service unavailable"
        
        with patch('requests.get', return_value=voices_error):
            with patch('requests.post', return_value=mock_requests_voices_success):
                # Voice listing should fail
                voices = get_available_voices()
                assert voices is None
                
                # But synthesis with known voice ID might still work
                # (This depends on implementation - in our case it would still work)
    
    @pytest.mark.regression
    @pytest.mark.api
    @pytest.mark.elevenlabs
    def test_api_backward_compatibility(self, mock_env, mock_requests_success):
        """Test backward compatibility with older API versions."""
        # Test with older model ID
        old_model_response = Mock()
        old_model_response.status_code = 200
        old_model_response.content = b'audio_data'
        
        with patch.dict(os.environ, {'ELEVENLABS_MODEL_ID': 'eleven_monolingual_v1'}):
            with patch('requests.post', return_value=old_model_response) as mock_post:
                with patch('pygame.mixer'):
                    with patch('tempfile.NamedTemporaryFile'):
                        with patch('os.unlink'):
                            result = speak_with_elevenlabs("Test message")
                            
                            assert result is True
                            mock_post.assert_called_once()
                            args, kwargs = mock_post.call_args
                            assert kwargs['json']['model_id'] == 'eleven_monolingual_v1'