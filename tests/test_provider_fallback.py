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
Provider fallback tests for the speak-app multi-provider TTS system.
"""

import pytest
import sys
import os
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

# Add the project root to the path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts_provider import TTSProvider

class TestProviderFallback:
    """Test provider fallback functionality."""
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_fallback_chain_all_providers_available(self):
        """Test fallback chain when all providers are available."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            available = tts.get_available_providers()
            
            # Should have all three providers
            assert 'elevenlabs' in available
            assert 'openai' in available
            assert 'pyttsx3' in available
            
            # Should select ElevenLabs as primary
            assert tts.select_provider() == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_fallback_chain_elevenlabs_unavailable(self):
        """Test fallback chain when ElevenLabs is unavailable."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            
            # Should have OpenAI and pyttsx3
            assert 'elevenlabs' not in available
            assert 'openai' in available
            assert 'pyttsx3' in available
            
            # Should select OpenAI as primary
            assert tts.select_provider() == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_fallback_chain_only_pyttsx3_available(self):
        """Test fallback chain when only pyttsx3 is available."""
        with patch.dict(os.environ, {
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            
            # Should only have pyttsx3
            assert 'elevenlabs' not in available
            assert 'openai' not in available
            assert 'pyttsx3' in available
            
            # Should select pyttsx3 as fallback
            assert tts.select_provider() == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_fallback_chain_tts_disabled(self):
        """Test fallback chain when TTS is disabled."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'false'
        }):
            tts = TTSProvider()
            
            # Should return None when TTS is disabled
            assert tts.select_provider() is None
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_fallback_on_elevenlabs_failure(self, mock_subprocess_run):
        """Test fallback to OpenAI when ElevenLabs fails."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock ElevenLabs failure, OpenAI success
            mock_subprocess_run.side_effect = [
                # ElevenLabs fails
                Mock(returncode=1, stderr="ElevenLabs API error"),
                # OpenAI succeeds
                Mock(returncode=0, stderr="")
            ]
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is True
            assert mock_subprocess_run.call_count == 2
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_fallback_on_openai_failure(self, mock_subprocess_run):
        """Test fallback to pyttsx3 when OpenAI fails."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            
            # Mock OpenAI failure, pyttsx3 success
            mock_subprocess_run.side_effect = [
                # OpenAI fails
                Mock(returncode=1, stderr="OpenAI API error"),
                # pyttsx3 succeeds
                Mock(returncode=0, stderr="")
            ]
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is True
            assert mock_subprocess_run.call_count == 2
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_fallback_all_providers_fail(self, mock_subprocess_run):
        """Test fallback when all providers fail."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock all providers failing
            mock_subprocess_run.return_value = Mock(returncode=1, stderr="Provider error")
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is False
            assert mock_subprocess_run.call_count == 3  # All three providers tried
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_provider_preference_override(self):
        """Test provider preference override."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_PROVIDER': 'openai',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Should respect provider preference
            assert tts.select_provider() == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_invalid_provider_preference_fallback(self):
        """Test fallback when invalid provider is preferred."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_PROVIDER': 'invalid_provider',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Should fallback to auto selection (ElevenLabs)
            assert tts.select_provider() == 'elevenlabs'
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_voice_parameters_only_passed_to_supporting_providers(self, mock_subprocess_run):
        """Test that voice parameters are only passed to supporting providers."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            voice_id = "test_voice_id"
            voice_settings = {"stability": 0.5, "similarity_boost": 0.8}
            
            # Mock ElevenLabs failure, OpenAI success
            mock_subprocess_run.side_effect = [
                # ElevenLabs fails
                Mock(returncode=1, stderr="ElevenLabs API error"),
                # OpenAI succeeds  
                Mock(returncode=0, stderr="")
            ]
            
            result = tts.speak_with_fallback("Test message", voice_id, voice_settings)
            
            assert result is True
            
            # Verify ElevenLabs was called with voice parameters
            elevenlabs_call = mock_subprocess_run.call_args_list[0]
            assert '--voice-id' in elevenlabs_call[0][0]
            assert voice_id in elevenlabs_call[0][0]
            assert '--stability' in elevenlabs_call[0][0]
            assert '--similarity-boost' in elevenlabs_call[0][0]
            
            # Verify OpenAI was called without voice parameters
            openai_call = mock_subprocess_run.call_args_list[1]
            assert '--voice-id' not in openai_call[0][0]
            assert '--stability' not in openai_call[0][0]
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_timeout_handling_in_fallback(self, mock_subprocess_run):
        """Test timeout handling during fallback."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock timeout then success
            mock_subprocess_run.side_effect = [
                # ElevenLabs times out
                subprocess.TimeoutExpired("cmd", 30),
                # OpenAI succeeds
                Mock(returncode=0, stderr="")
            ]
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is True
            assert mock_subprocess_run.call_count == 2
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_subprocess_exception_handling(self, mock_subprocess_run):
        """Test subprocess exception handling during fallback."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock subprocess exception then success
            mock_subprocess_run.side_effect = [
                # ElevenLabs subprocess error
                Exception("Subprocess error"),
                # OpenAI succeeds
                Mock(returncode=0, stderr="")
            ]
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is True
            assert mock_subprocess_run.call_count == 2


class TestProviderScriptExecution:
    """Test provider script execution and parameter passing."""
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_elevenlabs_script_execution(self, mock_subprocess_run):
        """Test ElevenLabs script execution with parameters."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.return_value = Mock(returncode=0, stderr="")
            
            voice_id = "test_voice_id"
            voice_settings = {"stability": 0.7, "similarity_boost": 0.9}
            
            result = tts.speak("Test message", "elevenlabs", voice_id, voice_settings)
            
            assert result is True
            mock_subprocess_run.assert_called_once()
            
            # Verify command structure
            call_args = mock_subprocess_run.call_args[0][0]
            assert 'elevenlabs_tts.py' in call_args[1]
            assert 'Test message' in call_args
            assert '--voice-id' in call_args
            assert voice_id in call_args
            assert '--stability' in call_args
            assert '0.7' in call_args
            assert '--similarity-boost' in call_args
            assert '0.9' in call_args
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_openai_script_execution(self, mock_subprocess_run):
        """Test OpenAI script execution without voice parameters."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.return_value = Mock(returncode=0, stderr="")
            
            result = tts.speak("Test message", "openai")
            
            assert result is True
            mock_subprocess_run.assert_called_once()
            
            # Verify command structure
            call_args = mock_subprocess_run.call_args[0][0]
            assert 'openai_tts.py' in call_args[1]
            assert 'Test message' in call_args
            # OpenAI doesn't support voice parameters via CLI
            assert '--voice-id' not in call_args
            assert '--stability' not in call_args
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_pyttsx3_script_execution(self, mock_subprocess_run):
        """Test pyttsx3 script execution."""
        with patch.dict(os.environ, {'TTS_ENABLED': 'true'}):
            tts = TTSProvider()
            mock_subprocess_run.return_value = Mock(returncode=0, stderr="")
            
            result = tts.speak("Test message", "pyttsx3")
            
            assert result is True
            mock_subprocess_run.assert_called_once()
            
            # Verify command structure
            call_args = mock_subprocess_run.call_args[0][0]
            assert 'pyttsx3_tts.py' in call_args[1]
            assert 'Test message' in call_args
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_nonexistent_provider_script(self, mock_subprocess_run):
        """Test handling of nonexistent provider script."""
        with patch.dict(os.environ, {'TTS_ENABLED': 'true'}):
            tts = TTSProvider()
            
            # Mock script doesn't exist
            with patch('pathlib.Path.exists', return_value=False):
                result = tts.speak("Test message", "elevenlabs")
                
                assert result is False
                mock_subprocess_run.assert_not_called()
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_script_execution_timeout(self, mock_subprocess_run):
        """Test script execution timeout handling."""
        with patch.dict(os.environ, {'TTS_ENABLED': 'true'}):
            tts = TTSProvider()
            mock_subprocess_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            
            result = tts.speak("Test message", "pyttsx3")
            
            assert result is False
            mock_subprocess_run.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_script_execution_exception(self, mock_subprocess_run):
        """Test script execution exception handling."""
        with patch.dict(os.environ, {'TTS_ENABLED': 'true'}):
            tts = TTSProvider()
            mock_subprocess_run.side_effect = Exception("Subprocess error")
            
            result = tts.speak("Test message", "pyttsx3")
            
            assert result is False
            mock_subprocess_run.assert_called_once()


class TestProviderSelection:
    """Test provider selection logic."""
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_auto_provider_selection_priority(self):
        """Test automatic provider selection priority."""
        # All providers available
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            assert tts.select_provider() == 'elevenlabs'
        
        # Only OpenAI and pyttsx3 available
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            assert tts.select_provider() == 'openai'
        
        # Only pyttsx3 available
        with patch.dict(os.environ, {
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            assert tts.select_provider() == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_provider_preference_validation(self):
        """Test provider preference validation."""
        valid_providers = ['elevenlabs', 'openai', 'pyttsx3']
        
        for provider in valid_providers:
            with patch.dict(os.environ, {
                'ELEVENLABS_API_KEY': 'test_key',
                'OPENAI_API_KEY': 'test_key',
                'TTS_PROVIDER': provider,
                'TTS_ENABLED': 'true'
            }):
                tts = TTSProvider()
                assert tts.select_provider() == provider
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_provider_preference_unavailable(self):
        """Test provider preference when preferred provider is unavailable."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'TTS_PROVIDER': 'elevenlabs',  # Preferred but not available
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            # Should fallback to available provider
            assert tts.select_provider() == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_provider_availability_detection(self):
        """Test provider availability detection based on API keys."""
        # Test with no API keys
        with patch.dict(os.environ, {}, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert available == ['pyttsx3']
        
        # Test with ElevenLabs key only
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'pyttsx3' in available
            assert 'openai' not in available
        
        # Test with OpenAI key only
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'openai' in available
            assert 'pyttsx3' in available
            assert 'elevenlabs' not in available
        
        # Test with both keys
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'openai' in available
            assert 'pyttsx3' in available


class TestFallbackConfiguration:
    """Test fallback configuration scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_tts_disabled_globally(self):
        """Test behavior when TTS is disabled globally."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key',
            'TTS_ENABLED': 'false'
        }):
            tts = TTSProvider()
            
            assert tts.select_provider() is None
            assert tts.speak("Test message") is False
            assert tts.speak_with_fallback("Test message") is False
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_empty_api_keys_ignored(self):
        """Test that empty API keys are ignored."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': '',
            'OPENAI_API_KEY': '',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            available = tts.get_available_providers()
            
            # Should only have pyttsx3 available
            assert available == ['pyttsx3']
            assert tts.select_provider() == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.fallback
    def test_provider_preference_case_sensitivity(self):
        """Test provider preference case sensitivity."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key',
            'TTS_PROVIDER': 'OpenAI',  # Mixed case
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            # Should fallback to auto selection due to case mismatch
            assert tts.select_provider() == 'elevenlabs'
    
    @pytest.mark.integration
    @pytest.mark.fallback
    def test_fallback_performance_monitoring(self, mock_subprocess_run):
        """Test fallback performance monitoring."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock first provider slow, second provider fast
            mock_subprocess_run.side_effect = [
                # ElevenLabs succeeds but slow
                Mock(returncode=0, stderr=""),
                # This shouldn't be called since first succeeded
            ]
            
            result = tts.speak_with_fallback("Test message")
            
            assert result is True
            assert mock_subprocess_run.call_count == 1  # Only first provider called