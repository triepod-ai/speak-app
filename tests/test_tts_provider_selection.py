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
TTS provider selection tests for the speak-app multi-provider TTS system.
"""

import pytest
import sys
import os
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to the path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts_provider import TTSProvider

class TestTTSProviderSelection:
    """Test TTS provider selection logic."""
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_selection_all_available(self):
        """Test provider selection when all providers are available."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'auto'
        }):
            tts = TTSProvider()
            
            # Should have all providers available
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'openai' in available
            assert 'pyttsx3' in available
            assert len(available) == 3
            
            # Should select ElevenLabs as priority
            selected = tts.select_provider()
            assert selected == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_selection_elevenlabs_only(self):
        """Test provider selection when only ElevenLabs is available."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'openai' not in available
            assert 'pyttsx3' in available
            assert len(available) == 2
            
            selected = tts.select_provider()
            assert selected == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_selection_openai_only(self):
        """Test provider selection when only OpenAI is available."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            
            available = tts.get_available_providers()
            assert 'elevenlabs' not in available
            assert 'openai' in available
            assert 'pyttsx3' in available
            assert len(available) == 2
            
            selected = tts.select_provider()
            assert selected == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_selection_pyttsx3_only(self):
        """Test provider selection when only pyttsx3 is available."""
        with patch.dict(os.environ, {
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            
            available = tts.get_available_providers()
            assert 'elevenlabs' not in available
            assert 'openai' not in available
            assert 'pyttsx3' in available
            assert len(available) == 1
            
            selected = tts.select_provider()
            assert selected == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_selection_tts_disabled(self):
        """Test provider selection when TTS is disabled."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'false'
        }):
            tts = TTSProvider()
            
            # Should still detect available providers
            available = tts.get_available_providers()
            assert len(available) == 3
            
            # But should not select any provider
            selected = tts.select_provider()
            assert selected is None
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_elevenlabs(self):
        """Test explicit ElevenLabs provider preference."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'elevenlabs'
        }):
            tts = TTSProvider()
            selected = tts.select_provider()
            assert selected == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_openai(self):
        """Test explicit OpenAI provider preference."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'openai'
        }):
            tts = TTSProvider()
            selected = tts.select_provider()
            assert selected == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_pyttsx3(self):
        """Test explicit pyttsx3 provider preference."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'pyttsx3'
        }):
            tts = TTSProvider()
            selected = tts.select_provider()
            assert selected == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_unavailable(self):
        """Test provider preference when preferred provider is unavailable."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'elevenlabs'  # Preferred but API key not available
        }, clear=True):
            tts = TTSProvider()
            
            # Should fallback to auto selection
            selected = tts.select_provider()
            assert selected == 'openai'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_invalid(self):
        """Test provider preference with invalid provider name."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'invalid_provider'
        }):
            tts = TTSProvider()
            
            # Should fallback to auto selection
            selected = tts.select_provider()
            assert selected == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_api_key_validation(self):
        """Test API key validation for provider availability."""
        # Test with valid keys
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'sk-test123',
            'OPENAI_API_KEY': 'sk-test456'
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'openai' in available
        
        # Test with empty keys
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': '',
            'OPENAI_API_KEY': ''
        }, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'elevenlabs' not in available
            assert 'openai' not in available
            assert 'pyttsx3' in available
        
        # Test with missing keys
        with patch.dict(os.environ, {}, clear=True):
            tts = TTSProvider()
            available = tts.get_available_providers()
            assert 'elevenlabs' not in available
            assert 'openai' not in available
            assert 'pyttsx3' in available
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_priority_ordering(self):
        """Test provider priority ordering in auto selection."""
        # Test priority: elevenlabs > openai > pyttsx3
        
        # All available - should select elevenlabs
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'key1',
            'OPENAI_API_KEY': 'key2',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            assert tts.select_provider() == 'elevenlabs'
        
        # Only openai and pyttsx3 - should select openai
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'key2',
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            assert tts.select_provider() == 'openai'
        
        # Only pyttsx3 - should select pyttsx3
        with patch.dict(os.environ, {
            'TTS_ENABLED': 'true'
        }, clear=True):
            tts = TTSProvider()
            assert tts.select_provider() == 'pyttsx3'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_tts_enabled_flag_variations(self):
        """Test various TTS_ENABLED flag values."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', False),  # Only 'true' (case insensitive) enables TTS
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('', False),
            ('invalid', False),
        ]
        
        for value, expected in test_cases:
            with patch.dict(os.environ, {
                'ELEVENLABS_API_KEY': 'test_key',
                'TTS_ENABLED': value
            }):
                tts = TTSProvider()
                if expected:
                    assert tts.select_provider() == 'elevenlabs'
                else:
                    assert tts.select_provider() is None
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_script_path_resolution(self):
        """Test provider script path resolution."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Test script path mapping
            script_paths = {
                'elevenlabs': 'elevenlabs_tts.py',
                'openai': 'openai_tts.py',
                'pyttsx3': 'pyttsx3_tts.py'
            }
            
            for provider, script_name in script_paths.items():
                # Mock the script path exists
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('subprocess.run', return_value=Mock(returncode=0, stderr="")):
                        result = tts.speak("Test", provider)
                        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_script_not_found(self):
        """Test handling when provider script is not found."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            
            # Mock script doesn't exist
            with patch('pathlib.Path.exists', return_value=False):
                result = tts.speak("Test", "elevenlabs")
                assert result is False
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_provider_execution_with_parameters(self, mock_subprocess_run):
        """Test provider execution with various parameters."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.return_value = Mock(returncode=0, stderr="")
            
            # Test with voice parameters
            voice_id = "test_voice"
            voice_settings = {"stability": 0.5, "similarity_boost": 0.8}
            
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
            assert '0.5' in call_args
            assert '--similarity-boost' in call_args
            assert '0.8' in call_args
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_provider_execution_timeout(self, mock_subprocess_run):
        """Test provider execution timeout handling."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            
            result = tts.speak("Test message", "elevenlabs")
            
            assert result is False
            mock_subprocess_run.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_provider_execution_error(self, mock_subprocess_run):
        """Test provider execution error handling."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.return_value = Mock(returncode=1, stderr="API Error")
            
            result = tts.speak("Test message", "elevenlabs")
            
            assert result is False
            mock_subprocess_run.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_provider_execution_exception(self, mock_subprocess_run):
        """Test provider execution exception handling."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            mock_subprocess_run.side_effect = Exception("Subprocess error")
            
            result = tts.speak("Test message", "elevenlabs")
            
            assert result is False
            mock_subprocess_run.assert_called_once()


class TestTTSProviderConfiguration:
    """Test TTS provider configuration scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_environment_variable_loading(self):
        """Test loading environment variables from different sources."""
        # Test direct environment variables
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'env_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            assert tts.tts_enabled is True
            assert tts.provider_preference == 'auto'
        
        # Test with custom provider preference
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'env_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'elevenlabs'
        }):
            tts = TTSProvider()
            assert tts.provider_preference == 'elevenlabs'
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_dotenv_file_loading(self):
        """Test loading configuration from .env file."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('dotenv.load_dotenv') as mock_load_dotenv:
                with patch.dict(os.environ, {
                    'ELEVENLABS_API_KEY': 'dotenv_key',
                    'TTS_ENABLED': 'true'
                }):
                    tts = TTSProvider()
                    
                    # Verify .env file was loaded
                    mock_load_dotenv.assert_called_once()
                    
                    # Verify configuration was loaded
                    assert tts.tts_enabled is True
                    available = tts.get_available_providers()
                    assert 'elevenlabs' in available
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_utils_directory_resolution(self):
        """Test utils directory resolution."""
        tts = TTSProvider()
        
        # Verify utils directory is set correctly
        assert tts.utils_dir == Path(__file__).parent.parent / "tts"
        
        # Verify it's a valid directory path
        assert isinstance(tts.utils_dir, Path)
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_script_mapping(self):
        """Test provider script mapping accuracy."""
        with patch.dict(os.environ, {'TTS_ENABLED': 'true'}):
            tts = TTSProvider()
            
            expected_scripts = {
                'elevenlabs': 'elevenlabs_tts.py',
                'openai': 'openai_tts.py',
                'pyttsx3': 'pyttsx3_tts.py'
            }
            
            for provider, script_name in expected_scripts.items():
                expected_path = tts.utils_dir / script_name
                
                # Mock the script exists
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('subprocess.run', return_value=Mock(returncode=0, stderr="")):
                        result = tts.speak("Test", provider)
                        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_configuration_validation(self):
        """Test configuration validation scenarios."""
        # Test invalid TTS_ENABLED values
        invalid_values = ['yes', 'no', '1', '0', 'on', 'off', 'enabled', 'disabled']
        
        for value in invalid_values:
            with patch.dict(os.environ, {
                'ELEVENLABS_API_KEY': 'test_key',
                'TTS_ENABLED': value
            }):
                tts = TTSProvider()
                # Should be disabled for non-'true' values
                assert tts.tts_enabled is False
    
    @pytest.mark.unit
    @pytest.mark.provider
    def test_provider_preference_case_sensitivity(self):
        """Test provider preference case sensitivity."""
        # Test various case combinations
        test_cases = [
            ('elevenlabs', 'elevenlabs'),
            ('ElevenLabs', 'elevenlabs'),  # Should fallback to auto
            ('ELEVENLABS', 'elevenlabs'),  # Should fallback to auto
            ('openai', 'openai'),
            ('OpenAI', 'elevenlabs'),      # Should fallback to auto
            ('OPENAI', 'elevenlabs'),      # Should fallback to auto
            ('pyttsx3', 'pyttsx3'),
            ('PyTTSx3', 'elevenlabs'),     # Should fallback to auto
        ]
        
        for preference, expected in test_cases:
            with patch.dict(os.environ, {
                'ELEVENLABS_API_KEY': 'test_key',
                'OPENAI_API_KEY': 'test_key',
                'TTS_ENABLED': 'true',
                'TTS_PROVIDER': preference
            }):
                tts = TTSProvider()
                selected = tts.select_provider()
                assert selected == expected
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_provider_configuration_persistence(self):
        """Test that provider configuration persists across calls."""
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true',
            'TTS_PROVIDER': 'elevenlabs'
        }):
            tts = TTSProvider()
            
            # Configuration should be consistent across multiple calls
            assert tts.select_provider() == 'elevenlabs'
            assert tts.select_provider() == 'elevenlabs'
            assert tts.select_provider() == 'elevenlabs'
            
            # Available providers should be consistent
            available1 = tts.get_available_providers()
            available2 = tts.get_available_providers()
            assert available1 == available2
    
    @pytest.mark.integration
    @pytest.mark.provider
    def test_dynamic_configuration_changes(self):
        """Test handling of dynamic configuration changes."""
        # Start with one configuration
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'test_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            assert tts.select_provider() == 'elevenlabs'
            
            # Change environment (simulating runtime changes)
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test_key',
                'TTS_ENABLED': 'true'
            }, clear=True):
                # Create new provider instance (simulating reinitialization)
                tts2 = TTSProvider()
                assert tts2.select_provider() == 'openai'
    
    @pytest.mark.regression
    @pytest.mark.provider
    def test_provider_selection_regression(self):
        """Test provider selection regression scenarios."""
        # Test historical behavior preservation
        
        # Legacy configuration should still work
        with patch.dict(os.environ, {
            'ELEVENLABS_API_KEY': 'legacy_key',
            'TTS_ENABLED': 'true'
        }):
            tts = TTSProvider()
            assert tts.select_provider() == 'elevenlabs'
            available = tts.get_available_providers()
            assert 'elevenlabs' in available
            assert 'pyttsx3' in available
        
        # Default behavior should be consistent
        with patch.dict(os.environ, {}, clear=True):
            tts = TTSProvider()
            # Should default to pyttsx3 when no API keys available
            assert tts.select_provider() == 'pyttsx3'
            available = tts.get_available_providers()
            assert available == ['pyttsx3']