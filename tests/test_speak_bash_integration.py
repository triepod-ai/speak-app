#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
# ]
# ///

"""
Test suite for speak bash script integration.
Tests the main speak command entry point and bash script functionality.
"""

import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import tempfile
import shutil

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestSpeakBashScript:
    """Test the speak bash script functionality."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    @pytest.fixture
    def temp_env(self):
        """Create a temporary environment for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy necessary files
            temp_bin = Path(tmpdir) / "bin"
            temp_bin.mkdir()
            
            # Copy speak script
            speak_src = PROJECT_ROOT / "speak"
            speak_dst = temp_bin / "speak"
            shutil.copy2(speak_src, speak_dst)
            speak_dst.chmod(0o755)
            
            # Copy TTS directory
            tts_src = PROJECT_ROOT / "tts"
            tts_dst = temp_bin / "tts"
            shutil.copytree(tts_src, tts_dst)
            
            yield temp_bin
    
    def test_speak_script_exists(self, speak_script_path):
        """Test that the speak script exists."""
        assert speak_script_path.exists()
        assert speak_script_path.is_file()
        # Check it's executable
        assert os.access(speak_script_path, os.X_OK)
    
    def test_speak_help(self, speak_script_path):
        """Test speak --help command."""
        result = subprocess.run(
            [str(speak_script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "speak - Text-to-speech command" in result.stdout
        assert "USAGE:" in result.stdout
        assert "OPTIONS:" in result.stdout
        assert "EXAMPLES:" in result.stdout
        assert "CONFIGURATION:" in result.stdout
    
    def test_speak_help_short(self, speak_script_path):
        """Test speak -h command."""
        result = subprocess.run(
            [str(speak_script_path), "-h"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "speak - Text-to-speech command" in result.stdout
    
    @patch.dict(os.environ, {"TTS_ENABLED": "false"})
    def test_speak_disabled(self, speak_script_path):
        """Test speak when TTS is disabled."""
        result = subprocess.run(
            [str(speak_script_path), "Test message"],
            capture_output=True,
            text=True
        )
        
        # Should exit silently when disabled
        assert result.returncode == 0
        assert "TTS is disabled" in result.stdout or result.stdout == ""
    
    def test_speak_off_flag(self, speak_script_path):
        """Test speak --off flag."""
        result = subprocess.run(
            [str(speak_script_path), "--off", "Test message"],
            capture_output=True,
            text=True
        )
        
        # Should skip TTS
        assert result.returncode == 0
    
    def test_speak_status(self, speak_script_path):
        """Test speak --status command."""
        result = subprocess.run(
            [str(speak_script_path), "--status"],
            capture_output=True,
            text=True,
            env={**os.environ, "TTS_ENABLED": "true"}
        )
        
        assert result.returncode == 0
        assert "TTS Status" in result.stdout or "Configuration" in result.stdout
    
    def test_speak_list(self, speak_script_path):
        """Test speak --list command."""
        result = subprocess.run(
            [str(speak_script_path), "--list"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "providers" in result.stdout.lower() or "available" in result.stdout.lower()
    
    def test_speak_enable(self, temp_env):
        """Test speak --enable command."""
        speak_path = temp_env / "speak"
        
        # Create a mock config file
        config_file = temp_env / ".tts_config"
        
        result = subprocess.run(
            [str(speak_path), "--enable"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_env.parent)}
        )
        
        # Should indicate TTS is enabled
        assert "enabled" in result.stdout.lower() or result.returncode == 0
    
    def test_speak_disable(self, temp_env):
        """Test speak --disable command."""
        speak_path = temp_env / "speak"
        
        result = subprocess.run(
            [str(speak_path), "--disable"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_env.parent)}
        )
        
        # Should indicate TTS is disabled
        assert "disabled" in result.stdout.lower() or result.returncode == 0
    
    def test_speak_with_provider(self, speak_script_path):
        """Test speak with specific provider."""
        result = subprocess.run(
            [str(speak_script_path), "--provider", "pyttsx3", "--off", "Test"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_speak_with_invalid_provider(self, speak_script_path):
        """Test speak with invalid provider."""
        result = subprocess.run(
            [str(speak_script_path), "--provider", "invalid_provider", "Test"],
            capture_output=True,
            text=True
        )
        
        # Should fail or show error
        assert result.returncode != 0 or "error" in result.stderr.lower()
    
    def test_speak_pipe_input(self, speak_script_path):
        """Test piping input to speak."""
        result = subprocess.run(
            f"echo 'Piped message' | {speak_script_path} --off",
            shell=True,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_speak_test_flag(self, speak_script_path):
        """Test speak --test command."""
        result = subprocess.run(
            [str(speak_script_path), "--test", "--off"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0

class TestSpeakProviderIntegration:
    """Test integration between speak script and providers."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    @patch.dict(os.environ, {"TTS_PROVIDER": "pyttsx3", "TTS_ENABLED": "true"})
    def test_pyttsx3_integration(self, speak_script_path):
        """Test integration with pyttsx3 provider."""
        # Use --off to avoid actual TTS
        result = subprocess.run(
            [str(speak_script_path), "--provider", "pyttsx3", "--off", "Test pyttsx3"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "TTS_ENABLED": "true"})
    def test_openai_integration(self, speak_script_path):
        """Test integration with OpenAI provider."""
        result = subprocess.run(
            [str(speak_script_path), "--provider", "openai", "--off", "Test OpenAI"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key", "TTS_ENABLED": "true"})
    def test_elevenlabs_integration(self, speak_script_path):
        """Test integration with ElevenLabs provider."""
        result = subprocess.run(
            [str(speak_script_path), "--provider", "elevenlabs", "--off", "Test ElevenLabs"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_auto_provider_selection(self, speak_script_path):
        """Test automatic provider selection."""
        result = subprocess.run(
            [str(speak_script_path), "--provider", "auto", "--off", "Test auto"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0

class TestSpeakVoiceOptions:
    """Test voice-related options in speak script."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"})
    def test_list_voices(self, speak_script_path):
        """Test --list-voices option."""
        result = subprocess.run(
            [str(speak_script_path), "--list-voices"],
            capture_output=True,
            text=True
        )
        
        # Should either succeed or indicate no API key
        assert result.returncode == 0 or "API" in result.stderr
    
    def test_voice_option(self, speak_script_path):
        """Test --voice option."""
        result = subprocess.run(
            [str(speak_script_path), "--voice", "Rachel", "--off", "Test voice"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_voice_settings(self, speak_script_path):
        """Test voice settings options."""
        result = subprocess.run(
            [str(speak_script_path), 
             "--stability", "0.8",
             "--similarity-boost", "0.9",
             "--off", "Test settings"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_test_voice(self, speak_script_path):
        """Test --test-voice option."""
        result = subprocess.run(
            [str(speak_script_path), "--test-voice", "Rachel", "--off"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0

class TestSpeakErrorHandling:
    """Test error handling in speak script."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    def test_missing_tts_provider_script(self, temp_env):
        """Test handling of missing TTS provider script."""
        speak_path = temp_env / "speak"
        
        # Remove the TTS directory
        tts_dir = temp_env / "tts"
        if tts_dir.exists():
            shutil.rmtree(tts_dir)
        
        result = subprocess.run(
            [str(speak_path), "Test"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "Error" in result.stderr or "not found" in result.stderr
    
    def test_invalid_arguments(self, speak_script_path):
        """Test handling of invalid arguments."""
        result = subprocess.run(
            [str(speak_script_path), "--invalid-option"],
            capture_output=True,
            text=True
        )
        
        # Should show help or error
        assert result.returncode != 0 or "Invalid option" in result.stderr
    
    def test_conflicting_options(self, speak_script_path):
        """Test handling of conflicting options."""
        result = subprocess.run(
            [str(speak_script_path), "--enable", "--disable"],
            capture_output=True,
            text=True
        )
        
        # Should handle gracefully
        assert result.returncode == 0 or "conflict" in result.stderr.lower()

class TestSpeakEnvironmentIntegration:
    """Test environment variable integration."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    @patch.dict(os.environ, {
        "TTS_ENABLED": "true",
        "TTS_PROVIDER": "pyttsx3",
        "ENGINEER_NAME": "TestEngineer"
    })
    def test_environment_variables(self, speak_script_path):
        """Test that environment variables are respected."""
        result = subprocess.run(
            [str(speak_script_path), "--status"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        # Status should reflect environment settings
        output = result.stdout.lower()
        assert "enabled" in output or "true" in output
    
    def test_dotenv_loading(self, temp_env):
        """Test loading from .env file."""
        speak_path = temp_env / "speak"
        
        # Create a .env file
        env_dir = temp_env.parent / "brainpods"
        env_dir.mkdir(exist_ok=True)
        env_file = env_dir / ".env"
        env_file.write_text("TTS_ENABLED=true\nTTS_PROVIDER=pyttsx3\n")
        
        result = subprocess.run(
            [str(speak_path), "--status"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_env.parent.parent)}
        )
        
        assert result.returncode == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])