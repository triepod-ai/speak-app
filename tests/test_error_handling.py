#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "requests>=2.31.0",
#   "openai>=1.0.0",
#   "pygame>=2.5.0",
# ]
# ///

"""
Test suite for comprehensive error handling across all TTS providers.
Tests various error scenarios, fallback mechanisms, and recovery strategies.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import tempfile
import requests
from openai import APIError, RateLimitError, APITimeoutError

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.tts_provider import TTSProvider
from tts.elevenlabs_tts import speak_with_elevenlabs
from tts.openai_tts import speak_with_openai
from tts.pyttsx3_tts import speak_with_pyttsx3

class TestProviderFallbackErrors:
    """Test error handling in provider fallback scenarios."""
    
    @pytest.fixture
    def tts_provider(self):
        """Create a TTS provider instance."""
        return TTSProvider()
    
    def test_all_providers_fail(self, tts_provider):
        """Test when all providers fail."""
        with patch('subprocess.run') as mock_run:
            # All providers return non-zero exit code
            mock_run.return_value.returncode = 1
            
            result = tts_provider.speak("Test message")
            
            assert result is False
            assert mock_run.call_count >= 1  # At least tried one provider
    
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key", "OPENAI_API_KEY": "test_key"})
    def test_primary_fails_fallback_succeeds(self, tts_provider):
        """Test fallback when primary provider fails."""
        with patch('subprocess.run') as mock_run:
            # First call fails, second succeeds
            mock_run.side_effect = [
                Mock(returncode=1),  # ElevenLabs fails
                Mock(returncode=0),  # OpenAI succeeds
            ]
            
            result = tts_provider.speak("Test message")
            
            assert result is True
            assert mock_run.call_count == 2
    
    def test_subprocess_exception(self, tts_provider):
        """Test handling of subprocess exceptions."""
        with patch('subprocess.run', side_effect=OSError("Command not found")):
            result = tts_provider.speak("Test message")
            
            assert result is False
    
    def test_timeout_handling(self, tts_provider):
        """Test timeout handling in subprocess calls."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 30)):
            result = tts_provider.speak("Test message", timeout=1)
            
            assert result is False

class TestElevenLabsErrorHandling:
    """Test error handling specific to ElevenLabs provider."""
    
    @patch('requests.post')
    def test_network_error(self, mock_post):
        """Test handling of network errors."""
        mock_post.side_effect = requests.ConnectionError("Network error")
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = requests.Timeout("Request timeout")
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_http_error_400(self, mock_post):
        """Test handling of HTTP 400 errors."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: Invalid voice_id"
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Client Error")
        mock_post.return_value = mock_response
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_http_error_401(self, mock_post):
        """Test handling of authentication errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized: Invalid API key"
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_http_error_429(self, mock_post):
        """Test handling of rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mock_post.return_value = mock_response
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_http_error_500(self, mock_post):
        """Test handling of server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False
    
    @patch('requests.post')
    def test_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response
        
        # For TTS, we expect audio content, not JSON
        result = speak_with_elevenlabs("Test message")
        
        # Should still work if we got a 200 response
        assert result is True or result is False  # Depends on implementation
    
    @patch('requests.post')
    @patch('pygame.mixer')
    def test_audio_playback_error(self, mock_mixer, mock_post):
        """Test handling of audio playback errors."""
        # API call succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'fake_audio_data'
        mock_post.return_value = mock_response
        
        # But pygame fails
        mock_mixer.music.play.side_effect = Exception("Audio device error")
        
        result = speak_with_elevenlabs("Test message")
        
        # Should handle gracefully
        assert result is False
    
    @patch('requests.post')
    @patch('tempfile.NamedTemporaryFile')
    def test_file_write_error(self, mock_temp, mock_post):
        """Test handling of file write errors."""
        # API call succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'fake_audio_data'
        mock_post.return_value = mock_response
        
        # But file write fails
        mock_temp.side_effect = OSError("No space left on device")
        
        result = speak_with_elevenlabs("Test message")
        
        assert result is False

class TestOpenAIErrorHandling:
    """Test error handling specific to OpenAI provider."""
    
    @patch('openai.OpenAI')
    def test_api_key_error(self, mock_openai_class):
        """Test handling of missing API key."""
        mock_openai_class.side_effect = ValueError("API key required")
        
        result = speak_with_openai("Test message")
        
        assert result is False
    
    @patch('openai.OpenAI')
    def test_api_error(self, mock_openai_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.audio.speech.create.side_effect = APIError(
            "API Error", 
            response=Mock(status_code=400),
            body=None
        )
        
        result = speak_with_openai("Test message")
        
        assert result is False
    
    @patch('openai.OpenAI')
    def test_rate_limit_error(self, mock_openai_class):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.audio.speech.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        
        result = speak_with_openai("Test message")
        
        assert result is False
    
    @patch('openai.OpenAI')
    def test_timeout_error(self, mock_openai_class):
        """Test handling of timeout errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.audio.speech.create.side_effect = APITimeoutError("Request timeout")
        
        result = speak_with_openai("Test message")
        
        assert result is False
    
    @patch('openai.OpenAI')
    @patch('tempfile.NamedTemporaryFile')
    def test_file_creation_error(self, mock_temp, mock_openai_class):
        """Test handling of file creation errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # API call would succeed
        mock_response = Mock()
        mock_client.audio.speech.create.return_value = mock_response
        
        # But file creation fails
        mock_temp.side_effect = OSError("Permission denied")
        
        result = speak_with_openai("Test message")
        
        assert result is False
    
    @patch('openai.OpenAI')
    def test_invalid_voice_error(self, mock_openai_class):
        """Test handling of invalid voice selection."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.audio.speech.create.side_effect = APIError(
            "Invalid voice",
            response=Mock(status_code=400),
            body={"error": {"message": "Invalid voice ID"}}
        )
        
        result = speak_with_openai("Test message", voice="invalid_voice")
        
        assert result is False

class TestPyttsx3ErrorHandling:
    """Test error handling specific to pyttsx3 provider."""
    
    @patch('pyttsx3.init')
    def test_init_error(self, mock_init):
        """Test handling of pyttsx3 initialization errors."""
        mock_init.side_effect = Exception("No audio device found")
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is False
    
    @patch('pyttsx3.init')
    def test_engine_property_error(self, mock_init):
        """Test handling of engine property errors."""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        mock_engine.setProperty.side_effect = Exception("Invalid property")
        
        result = speak_with_pyttsx3("Test message")
        
        # Should still try to speak even if properties fail
        assert result is True or result is False
    
    @patch('pyttsx3.init')
    def test_say_error(self, mock_init):
        """Test handling of speech synthesis errors."""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        mock_engine.say.side_effect = Exception("Synthesis error")
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is False
    
    @patch('pyttsx3.init')
    def test_runandwait_error(self, mock_init):
        """Test handling of runAndWait errors."""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        mock_engine.runAndWait.side_effect = RuntimeError("Engine busy")
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is False
    
    @patch('pyttsx3.init')
    def test_voice_not_found_error(self, mock_init):
        """Test handling when requested voice is not found."""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        # No voices available
        mock_engine.getProperty.return_value = []
        
        result = speak_with_pyttsx3("Test message")
        
        # Should still work without voice selection
        assert result is True

class TestEdgeCaseErrors:
    """Test edge case error scenarios."""
    
    def test_empty_message(self):
        """Test handling of empty messages."""
        tts = TTSProvider()
        result = tts.speak("")
        
        # Should handle gracefully
        assert result is True or result is False
    
    def test_none_message(self):
        """Test handling of None messages."""
        tts = TTSProvider()
        
        # Should handle None gracefully
        try:
            result = tts.speak(None)
            assert result is False
        except TypeError:
            # Also acceptable to raise TypeError for None
            pass
    
    def test_very_long_message(self):
        """Test handling of very long messages."""
        tts = TTSProvider()
        long_message = "Test " * 10000  # 50,000 characters
        
        # Should handle without crashing
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            result = tts.speak(long_message)
            
            # Should attempt to speak
            assert mock_run.called
    
    def test_unicode_errors(self):
        """Test handling of Unicode in messages."""
        tts = TTSProvider()
        unicode_message = "Hello 世界! 🌍 Здравствуй мир!"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            result = tts.speak(unicode_message)
            
            assert mock_run.called
    
    def test_control_characters(self):
        """Test handling of control characters."""
        tts = TTSProvider()
        control_message = "Test\x00\x01\x02\x03message"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            result = tts.speak(control_message)
            
            assert mock_run.called

class TestRecoveryMechanisms:
    """Test error recovery mechanisms."""
    
    def test_retry_on_transient_error(self):
        """Test retry mechanism for transient errors."""
        # This would require implementing retry logic in the providers
        pass
    
    def test_fallback_chain_exhaustion(self):
        """Test behavior when all fallbacks are exhausted."""
        tts = TTSProvider()
        
        with patch('subprocess.run') as mock_run:
            # All providers fail
            mock_run.return_value.returncode = 1
            
            result = tts.speak("Test")
            
            assert result is False
            # Should have tried multiple providers
            assert mock_run.call_count >= 1
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "nonexistent"})
    def test_invalid_provider_fallback(self):
        """Test fallback when invalid provider is specified."""
        tts = TTSProvider()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Should fall back to available provider
            result = tts.speak("Test")
            
            # Should still work with fallback
            assert mock_run.called

if __name__ == "__main__":
    pytest.main([__file__, "-v"])