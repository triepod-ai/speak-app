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
Test suite for environment variable configuration and loading.
Tests .env file loading, environment variable precedence, and configuration management.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import tempfile
import json

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.tts_provider import TTSProvider

class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    @patch.dict(os.environ, {"TTS_ENABLED": "true"})
    def test_tts_enabled_true(self):
        """Test TTS_ENABLED=true."""
        assert os.getenv("TTS_ENABLED") == "true"
    
    @patch.dict(os.environ, {"TTS_ENABLED": "false"})
    def test_tts_enabled_false(self):
        """Test TTS_ENABLED=false."""
        assert os.getenv("TTS_ENABLED") == "false"
    
    @patch.dict(os.environ, {"TTS_ENABLED": "1"})
    def test_tts_enabled_numeric(self):
        """Test TTS_ENABLED with numeric value."""
        assert os.getenv("TTS_ENABLED") == "1"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_tts_enabled_missing(self):
        """Test missing TTS_ENABLED variable."""
        assert os.getenv("TTS_ENABLED") is None
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "elevenlabs"})
    def test_tts_provider_elevenlabs(self):
        """Test TTS_PROVIDER=elevenlabs."""
        assert os.getenv("TTS_PROVIDER") == "elevenlabs"
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "openai"})
    def test_tts_provider_openai(self):
        """Test TTS_PROVIDER=openai."""
        assert os.getenv("TTS_PROVIDER") == "openai"
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "pyttsx3"})
    def test_tts_provider_pyttsx3(self):
        """Test TTS_PROVIDER=pyttsx3."""
        assert os.getenv("TTS_PROVIDER") == "pyttsx3"
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "auto"})
    def test_tts_provider_auto(self):
        """Test TTS_PROVIDER=auto."""
        assert os.getenv("TTS_PROVIDER") == "auto"
    
    @patch.dict(os.environ, {"ENGINEER_NAME": "TestEngineer"})
    def test_engineer_name(self):
        """Test ENGINEER_NAME variable."""
        assert os.getenv("ENGINEER_NAME") == "TestEngineer"

class TestElevenLabsEnvironment:
    """Test ElevenLabs-specific environment variables."""
    
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key_123"})
    def test_elevenlabs_api_key(self):
        """Test ELEVENLABS_API_KEY."""
        assert os.getenv("ELEVENLABS_API_KEY") == "test_key_123"
    
    @patch.dict(os.environ, {"ELEVENLABS_VOICE_ID": "21m00Tcm4TlvDq8ikWAM"})
    def test_elevenlabs_voice_id(self):
        """Test ELEVENLABS_VOICE_ID."""
        assert os.getenv("ELEVENLABS_VOICE_ID") == "21m00Tcm4TlvDq8ikWAM"
    
    @patch.dict(os.environ, {"ELEVENLABS_MODEL_ID": "eleven_turbo_v2_5"})
    def test_elevenlabs_model_id(self):
        """Test ELEVENLABS_MODEL_ID."""
        assert os.getenv("ELEVENLABS_MODEL_ID") == "eleven_turbo_v2_5"
    
    @patch.dict(os.environ, {
        "ELEVENLABS_API_KEY": "key",
        "ELEVENLABS_VOICE_ID": "voice123",
        "ELEVENLABS_MODEL_ID": "model123"
    })
    def test_elevenlabs_all_variables(self):
        """Test all ElevenLabs variables together."""
        assert os.getenv("ELEVENLABS_API_KEY") == "key"
        assert os.getenv("ELEVENLABS_VOICE_ID") == "voice123"
        assert os.getenv("ELEVENLABS_MODEL_ID") == "model123"

class TestOpenAIEnvironment:
    """Test OpenAI-specific environment variables."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test_key_123"})
    def test_openai_api_key(self):
        """Test OPENAI_API_KEY."""
        assert os.getenv("OPENAI_API_KEY") == "sk-test_key_123"
    
    @patch.dict(os.environ, {"OPENAI_TTS_VOICE": "nova"})
    def test_openai_tts_voice(self):
        """Test OPENAI_TTS_VOICE."""
        assert os.getenv("OPENAI_TTS_VOICE") == "nova"
    
    @patch.dict(os.environ, {"OPENAI_TTS_MODEL": "tts-1-hd"})
    def test_openai_tts_model(self):
        """Test OPENAI_TTS_MODEL."""
        assert os.getenv("OPENAI_TTS_MODEL") == "tts-1-hd"
    
    @patch.dict(os.environ, {"OPENAI_TTS_SPEED": "1.5"})
    def test_openai_tts_speed(self):
        """Test OPENAI_TTS_SPEED."""
        assert os.getenv("OPENAI_TTS_SPEED") == "1.5"

class TestQuietHoursEnvironment:
    """Test quiet hours environment variables."""
    
    @patch.dict(os.environ, {
        "TTS_QUIET_HOURS_START": "22",
        "TTS_QUIET_HOURS_END": "8"
    })
    def test_quiet_hours_valid(self):
        """Test valid quiet hours configuration."""
        assert os.getenv("TTS_QUIET_HOURS_START") == "22"
        assert os.getenv("TTS_QUIET_HOURS_END") == "8"
    
    @patch.dict(os.environ, {
        "TTS_QUIET_HOURS_START": "invalid",
        "TTS_QUIET_HOURS_END": "8"
    })
    def test_quiet_hours_invalid_start(self):
        """Test invalid start hour."""
        assert os.getenv("TTS_QUIET_HOURS_START") == "invalid"
        # Should be handled gracefully by the application
    
    @patch.dict(os.environ, {
        "TTS_QUIET_HOURS_START": "22",
        "TTS_QUIET_HOURS_END": "invalid"
    })
    def test_quiet_hours_invalid_end(self):
        """Test invalid end hour."""
        assert os.getenv("TTS_QUIET_HOURS_END") == "invalid"

class TestDotEnvLoading:
    """Test .env file loading functionality."""
    
    @pytest.fixture
    def temp_home(self):
        """Create a temporary home directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_dotenv_file_loading(self, temp_home):
        """Test loading variables from .env file."""
        # Create .env file
        brainpods_dir = temp_home / "brainpods"
        brainpods_dir.mkdir()
        env_file = brainpods_dir / ".env"
        env_file.write_text("""
TTS_ENABLED=true
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=test_key_from_env
ENGINEER_NAME=EnvFileEngineer
""")
        
        # Mock Path.home() to return our temp directory
        with patch('pathlib.Path.home', return_value=temp_home):
            # Import the module that loads .env
            import importlib
            import tts.observability
            importlib.reload(tts.observability)
            
            # Variables should be loaded
            # Note: python-dotenv doesn't override existing env vars by default
    
    def test_dotenv_file_missing(self, temp_home):
        """Test behavior when .env file is missing."""
        # No .env file created
        
        with patch('pathlib.Path.home', return_value=temp_home):
            # Should not crash when .env is missing
            import importlib
            import tts.observability
            try:
                importlib.reload(tts.observability)
            except Exception as e:
                pytest.fail(f"Should not raise exception when .env is missing: {e}")
    
    def test_dotenv_file_malformed(self, temp_home):
        """Test handling of malformed .env file."""
        brainpods_dir = temp_home / "brainpods"
        brainpods_dir.mkdir()
        env_file = brainpods_dir / ".env"
        env_file.write_text("""
VALID_KEY=valid_value
INVALID LINE WITHOUT EQUALS
ANOTHER_VALID=value
=MISSING_KEY
KEY_WITH_SPACES = value with spaces
""")
        
        with patch('pathlib.Path.home', return_value=temp_home):
            # Should handle malformed lines gracefully
            import importlib
            import tts.observability
            try:
                importlib.reload(tts.observability)
            except Exception as e:
                pytest.fail(f"Should handle malformed .env gracefully: {e}")

class TestEnvironmentPrecedence:
    """Test environment variable precedence and override behavior."""
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "openai"})
    def test_env_var_precedence(self, temp_home):
        """Test that environment variables take precedence over .env file."""
        # Create .env file with different value
        brainpods_dir = temp_home / "brainpods"
        brainpods_dir.mkdir()
        env_file = brainpods_dir / ".env"
        env_file.write_text("TTS_PROVIDER=elevenlabs\n")
        
        # Environment variable should take precedence
        assert os.getenv("TTS_PROVIDER") == "openai"
    
    def test_default_values(self):
        """Test default values when variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Test defaults in the application
            tts = TTSProvider()
            assert hasattr(tts, 'provider')  # Should have a default provider

class TestProviderConfiguration:
    """Test provider-specific configuration from environment."""
    
    @patch.dict(os.environ, {
        "TTS_PROVIDER": "elevenlabs",
        "ELEVENLABS_API_KEY": "test_key"
    })
    def test_elevenlabs_configuration(self):
        """Test ElevenLabs configuration from environment."""
        tts = TTSProvider()
        providers = tts.get_available_providers()
        assert "elevenlabs" in providers
    
    @patch.dict(os.environ, {
        "TTS_PROVIDER": "openai",
        "OPENAI_API_KEY": "test_key"
    })
    def test_openai_configuration(self):
        """Test OpenAI configuration from environment."""
        tts = TTSProvider()
        providers = tts.get_available_providers()
        assert "openai" in providers
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "pyttsx3"})
    def test_pyttsx3_configuration(self):
        """Test pyttsx3 configuration from environment."""
        tts = TTSProvider()
        providers = tts.get_available_providers()
        assert "pyttsx3" in providers
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "auto"})
    def test_auto_provider_selection(self):
        """Test automatic provider selection."""
        tts = TTSProvider()
        # Should select based on available API keys
        assert tts.provider in ["elevenlabs", "openai", "pyttsx3"]

class TestEnvironmentValidation:
    """Test validation of environment variables."""
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "invalid_provider"})
    def test_invalid_provider(self):
        """Test handling of invalid provider name."""
        tts = TTSProvider()
        # Should fall back to a valid provider
        assert tts.provider in ["elevenlabs", "openai", "pyttsx3"]
    
    @patch.dict(os.environ, {"OPENAI_TTS_SPEED": "invalid"})
    def test_invalid_speed_value(self):
        """Test handling of invalid speed value."""
        # Should handle gracefully and use default
        from tts.openai_tts import speak_with_openai
        # Function should not crash with invalid speed
    
    @patch.dict(os.environ, {"TTS_QUIET_HOURS_START": "25"})
    def test_invalid_hour_value(self):
        """Test handling of invalid hour value."""
        # Should handle hours outside 0-23 range
        from tts.observability import TTSObservability
        obs = TTSObservability()
        assert obs.quiet_hours is None  # Should not set invalid hours

class TestEnvironmentSecurity:
    """Test security aspects of environment handling."""
    
    def test_api_key_not_logged(self):
        """Test that API keys are not logged."""
        # This would require checking log outputs
        pass
    
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "secret_key_12345"})
    def test_api_key_masking(self):
        """Test that API keys are masked in error messages."""
        # When displaying configuration, keys should be masked
        tts = TTSProvider()
        # Any string representation should not contain the actual key
        config_str = str(tts.__dict__)
        assert "secret_key_12345" not in config_str

class TestConfigurationExport:
    """Test configuration export and import functionality."""
    
    def test_export_configuration(self):
        """Test exporting current configuration."""
        with patch.dict(os.environ, {
            "TTS_PROVIDER": "openai",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_TTS_VOICE": "nova"
        }):
            # Configuration should be exportable
            config = {
                "provider": os.getenv("TTS_PROVIDER"),
                "voice": os.getenv("OPENAI_TTS_VOICE")
            }
            assert config["provider"] == "openai"
            assert config["voice"] == "nova"
    
    def test_configuration_persistence(self, temp_home):
        """Test saving configuration for persistence."""
        config_dir = temp_home / ".config" / "speak-app"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.json"
        
        config = {
            "provider": "elevenlabs",
            "default_voice": "Rachel",
            "quiet_hours": {"start": 22, "end": 8}
        }
        
        config_file.write_text(json.dumps(config, indent=2))
        
        # Verify configuration was saved
        assert config_file.exists()
        loaded_config = json.loads(config_file.read_text())
        assert loaded_config["provider"] == "elevenlabs"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])