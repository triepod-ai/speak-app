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
Voice selection tests for ElevenLabs TTS provider.
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
    get_available_voices,
    find_voice_by_name,
    validate_voice_id
)

class TestVoiceSelection:
    """Test voice selection and validation functionality."""
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_get_available_voices_success(self, mock_env, mock_requests_voices_success):
        """Test getting available voices with successful API response."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            voices = get_available_voices()
            
            assert voices is not None
            assert len(voices) == 4  # Based on our mock data
            assert any(voice['name'] == 'Rachel' for voice in voices)
            assert any(voice['name'] == 'Domi' for voice in voices)
            assert any(voice['name'] == 'Bella' for voice in voices)
            assert any(voice['name'] == 'CustomVoice' for voice in voices)
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_get_available_voices_api_error(self, mock_env, mock_requests_error):
        """Test getting available voices with API error."""
        with patch('requests.get', return_value=mock_requests_error):
            voices = get_available_voices()
            assert voices is None
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_get_available_voices_no_api_key(self):
        """Test getting available voices without API key."""
        with patch.dict(os.environ, {}, clear=True):
            voices = get_available_voices()
            assert voices is None
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_get_available_voices_network_error(self, mock_env):
        """Test getting available voices with network error."""
        with patch('requests.get', side_effect=Exception("Network error")):
            voices = get_available_voices()
            assert voices is None
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_find_voice_by_name_success(self, mock_env, mock_requests_voices_success):
        """Test finding voice by name with successful match."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            # Test finding by exact name
            voice_id = find_voice_by_name("Rachel")
            assert voice_id == "21m00Tcm4TlvDq8ikWAM"
            
            # Test finding by exact name (different case)
            voice_id = find_voice_by_name("rachel")
            assert voice_id == "21m00Tcm4TlvDq8ikWAM"
            
            # Test finding by exact name (mixed case)
            voice_id = find_voice_by_name("DOMI")
            assert voice_id == "AZnzlk1XvdvUeBnXmlld"
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_find_voice_by_name_not_found(self, mock_env, mock_requests_voices_success):
        """Test finding voice by name with no match."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            voice_id = find_voice_by_name("NonExistentVoice")
            assert voice_id is None
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_find_voice_by_name_api_error(self, mock_env, mock_requests_error):
        """Test finding voice by name with API error."""
        with patch('requests.get', return_value=mock_requests_error):
            voice_id = find_voice_by_name("Rachel")
            assert voice_id is None
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_id_success(self, mock_env, mock_requests_voices_success):
        """Test validating voice ID with successful match."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            # Test valid voice IDs
            assert validate_voice_id("21m00Tcm4TlvDq8ikWAM") is True
            assert validate_voice_id("AZnzlk1XvdvUeBnXmlld") is True
            assert validate_voice_id("EXAVITQu4vr4xnSDxMaL") is True
            assert validate_voice_id("custom_voice_123") is True
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_id_not_found(self, mock_env, mock_requests_voices_success):
        """Test validating voice ID with no match."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            assert validate_voice_id("invalid_voice_id") is False
            assert validate_voice_id("") is False
            assert validate_voice_id("nonexistent123") is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_validate_voice_id_api_error(self, mock_env, mock_requests_error):
        """Test validating voice ID with API error."""
        with patch('requests.get', return_value=mock_requests_error):
            assert validate_voice_id("21m00Tcm4TlvDq8ikWAM") is False
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_voice_categories(self, mock_env, mock_requests_voices_success):
        """Test voice categorization."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            voices = get_available_voices()
            
            # Test premade voices
            premade_voices = [v for v in voices if v['category'] == 'premade']
            assert len(premade_voices) == 3
            assert any(v['name'] == 'Rachel' for v in premade_voices)
            assert any(v['name'] == 'Domi' for v in premade_voices)
            assert any(v['name'] == 'Bella' for v in premade_voices)
            
            # Test generated voices
            generated_voices = [v for v in voices if v['category'] == 'generated']
            assert len(generated_voices) == 1
            assert generated_voices[0]['name'] == 'CustomVoice'
    
    @pytest.mark.unit
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_voice_properties(self, mock_env, mock_requests_voices_success):
        """Test voice properties and metadata."""
        with patch('requests.get', return_value=mock_requests_voices_success):
            voices = get_available_voices()
            
            # Test Rachel voice properties
            rachel = next(v for v in voices if v['name'] == 'Rachel')
            assert rachel['voice_id'] == '21m00Tcm4TlvDq8ikWAM'
            assert rachel['category'] == 'premade'
            assert rachel['labels']['accent'] == 'american'
            assert rachel['labels']['gender'] == 'female'
            assert rachel['labels']['use_case'] == 'narration'
            assert rachel['settings']['stability'] == 0.5
            assert rachel['settings']['similarity_boost'] == 0.75
            
            # Test Domi voice properties
            domi = next(v for v in voices if v['name'] == 'Domi')
            assert domi['voice_id'] == 'AZnzlk1XvdvUeBnXmlld'
            assert domi['labels']['description'] == 'strong'
            assert domi['settings']['stability'] == 0.6
            assert domi['settings']['similarity_boost'] == 0.8
            
            # Test CustomVoice properties
            custom = next(v for v in voices if v['name'] == 'CustomVoice')
            assert custom['voice_id'] == 'custom_voice_123'
            assert custom['category'] == 'generated'
            assert custom['labels']['accent'] == 'british'
            assert custom['labels']['gender'] == 'male'
    
    @pytest.mark.integration
    @pytest.mark.voice
    @pytest.mark.elevenlabs
    def test_voice_selection_workflow(self, mock_env, mock_requests_voices_success, mock_requests_success, mock_pygame, mock_tempfile, mock_os_unlink):
        """Test complete voice selection workflow."""
        from elevenlabs_tts import speak_with_elevenlabs
        
        with patch('requests.get', return_value=mock_requests_voices_success):
            with patch('requests.post', return_value=mock_requests_success):
                # Step 1: Get available voices
                voices = get_available_voices()
                assert voices is not None
                assert len(voices) > 0
                
                # Step 2: Find voice by name
                voice_id = find_voice_by_name("Domi")
                assert voice_id == "AZnzlk1XvdvUeBnXmlld"
                
                # Step 3: Validate voice ID
                assert validate_voice_id(voice_id) is True
                
                # Step 4: Use voice for synthesis
                result = speak_with_elevenlabs("Test message", voice_id=voice_id)
                assert result is True