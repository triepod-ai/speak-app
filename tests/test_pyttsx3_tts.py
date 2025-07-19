#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "pyttsx3>=2.90",
# ]
# ///

"""
Test suite for pyttsx3 TTS provider.
Tests offline text-to-speech functionality using pyttsx3.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.pyttsx3_tts import speak_with_pyttsx3, main

class TestPyttsx3TTS:
    """Test pyttsx3 TTS provider functionality."""
    
    @pytest.fixture
    def mock_pyttsx3_engine(self):
        """Mock pyttsx3 engine."""
        with patch('pyttsx3.init') as mock_init:
            mock_engine = MagicMock()
            mock_init.return_value = mock_engine
            
            # Mock voice properties
            mock_voice1 = Mock()
            mock_voice1.id = 'voice1'
            mock_voice2 = Mock()
            mock_voice2.id = 'voice2'
            
            mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]
            
            yield mock_engine
    
    def test_speak_with_pyttsx3_success(self, mock_pyttsx3_engine):
        """Test successful text-to-speech conversion."""
        result = speak_with_pyttsx3("Hello, world!")
        
        assert result is True
        
        # Verify engine initialization
        import pyttsx3
        pyttsx3.init.assert_called_once()
        
        # Verify property settings
        mock_pyttsx3_engine.setProperty.assert_any_call('rate', 180)
        mock_pyttsx3_engine.setProperty.assert_any_call('volume', 0.8)
        
        # Verify voice selection (should use second voice)
        mock_pyttsx3_engine.getProperty.assert_called_with('voices')
        mock_pyttsx3_engine.setProperty.assert_any_call('voice', 'voice2')
        
        # Verify speech
        mock_pyttsx3_engine.say.assert_called_once_with("Hello, world!")
        mock_pyttsx3_engine.runAndWait.assert_called_once()
    
    def test_speak_with_pyttsx3_single_voice(self, mock_pyttsx3_engine):
        """Test with only one voice available."""
        # Mock only one voice
        mock_voice = Mock()
        mock_voice.id = 'single_voice'
        mock_pyttsx3_engine.getProperty.return_value = [mock_voice]
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is True
        
        # Should not try to set voice when only one available
        voice_calls = [call for call in mock_pyttsx3_engine.setProperty.call_args_list 
                      if call[0][0] == 'voice']
        assert len(voice_calls) == 0
    
    def test_speak_with_pyttsx3_no_voices(self, mock_pyttsx3_engine):
        """Test with no voices available."""
        mock_pyttsx3_engine.getProperty.return_value = []
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is True
        
        # Should still work without voice selection
        mock_pyttsx3_engine.say.assert_called_once_with("Test message")
        mock_pyttsx3_engine.runAndWait.assert_called_once()
    
    def test_speak_with_pyttsx3_import_error(self):
        """Test handling when pyttsx3 is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'pyttsx3'")):
            result = speak_with_pyttsx3("Test message")
            
            assert result is False
    
    def test_speak_with_pyttsx3_runtime_error(self, mock_pyttsx3_engine):
        """Test handling runtime errors."""
        mock_pyttsx3_engine.say.side_effect = RuntimeError("Engine error")
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is False
    
    def test_speak_with_pyttsx3_init_error(self):
        """Test handling initialization errors."""
        with patch('pyttsx3.init', side_effect=Exception("Init failed")):
            result = speak_with_pyttsx3("Test message")
            
            assert result is False
    
    @patch('sys.stderr')
    def test_speak_with_pyttsx3_error_logging(self, mock_stderr, mock_pyttsx3_engine):
        """Test error logging to stderr."""
        mock_pyttsx3_engine.runAndWait.side_effect = Exception("Test error")
        
        result = speak_with_pyttsx3("Test message")
        
        assert result is False
        # Check that error was printed to stderr
        mock_stderr.write.assert_called()
    
    def test_speak_with_pyttsx3_empty_text(self, mock_pyttsx3_engine):
        """Test with empty text."""
        result = speak_with_pyttsx3("")
        
        assert result is True
        mock_pyttsx3_engine.say.assert_called_once_with("")
    
    def test_speak_with_pyttsx3_unicode_text(self, mock_pyttsx3_engine):
        """Test with Unicode text."""
        result = speak_with_pyttsx3("Hello 世界! 🌍")
        
        assert result is True
        mock_pyttsx3_engine.say.assert_called_once_with("Hello 世界! 🌍")
    
    def test_speak_with_pyttsx3_long_text(self, mock_pyttsx3_engine):
        """Test with long text."""
        long_text = "This is a very long message. " * 100
        result = speak_with_pyttsx3(long_text)
        
        assert result is True
        mock_pyttsx3_engine.say.assert_called_once_with(long_text)

class TestPyttsx3Main:
    """Test the main entry point."""
    
    @patch('tts.pyttsx3_tts.speak_with_pyttsx3')
    @patch('sys.argv', ['pyttsx3_tts.py', 'Hello', 'world'])
    def test_main_with_arguments(self, mock_speak):
        """Test main with command line arguments."""
        mock_speak.return_value = True
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_speak.assert_called_once_with("Hello world")
    
    @patch('tts.pyttsx3_tts.speak_with_pyttsx3')
    @patch('random.choice')
    @patch('sys.argv', ['pyttsx3_tts.py'])
    def test_main_no_arguments(self, mock_choice, mock_speak):
        """Test main with no arguments (uses random message)."""
        mock_choice.return_value = "Offline TTS is working perfectly!"
        mock_speak.return_value = True
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_speak.assert_called_once_with("Offline TTS is working perfectly!")
        
        # Verify random choice was called with the default messages
        args = mock_choice.call_args[0][0]
        assert "Offline TTS is working perfectly!" in args
        assert "No internet? No problem! Local speech synthesis active." in args
        assert "Claude Code offline voice ready." in args
        assert "Privacy-first TTS enabled." in args
    
    @patch('tts.pyttsx3_tts.speak_with_pyttsx3')
    @patch('sys.argv', ['pyttsx3_tts.py', 'Test'])
    def test_main_failure(self, mock_speak):
        """Test main with TTS failure."""
        mock_speak.return_value = False
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    @patch('tts.pyttsx3_tts.speak_with_pyttsx3')
    @patch('sys.argv', ['pyttsx3_tts.py', 'Special', 'chars:', '!@#$%'])
    def test_main_special_characters(self, mock_speak):
        """Test main with special characters."""
        mock_speak.return_value = True
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
        mock_speak.assert_called_once_with("Special chars: !@#$%")

class TestPyttsx3Integration:
    """Test pyttsx3 integration scenarios."""
    
    @pytest.fixture
    def mock_full_engine(self):
        """Mock complete pyttsx3 engine with all methods."""
        with patch('pyttsx3.init') as mock_init:
            mock_engine = MagicMock()
            mock_init.return_value = mock_engine
            
            # Mock multiple voices with different properties
            mock_voice1 = Mock()
            mock_voice1.id = 'com.apple.speech.synthesis.voice.Alex'
            mock_voice1.name = 'Alex'
            mock_voice1.languages = ['en_US']
            mock_voice1.gender = 'male'
            mock_voice1.age = 'adult'
            
            mock_voice2 = Mock()
            mock_voice2.id = 'com.apple.speech.synthesis.voice.Victoria'
            mock_voice2.name = 'Victoria'
            mock_voice2.languages = ['en_US']
            mock_voice2.gender = 'female'
            mock_voice2.age = 'adult'
            
            mock_voice3 = Mock()
            mock_voice3.id = 'com.apple.speech.synthesis.voice.Daniel'
            mock_voice3.name = 'Daniel'
            mock_voice3.languages = ['en_GB']
            mock_voice3.gender = 'male'
            mock_voice3.age = 'adult'
            
            mock_engine.getProperty.side_effect = lambda prop: {
                'voices': [mock_voice1, mock_voice2, mock_voice3],
                'rate': 200,
                'volume': 0.9,
                'voice': mock_voice1.id
            }.get(prop)
            
            yield mock_engine
    
    def test_voice_selection_logic(self, mock_full_engine):
        """Test voice selection with multiple voices."""
        result = speak_with_pyttsx3("Testing voice selection")
        
        assert result is True
        
        # Should select second voice (Victoria)
        voice_calls = [call for call in mock_full_engine.setProperty.call_args_list 
                      if call[0][0] == 'voice']
        assert len(voice_calls) == 1
        assert voice_calls[0][0][1] == 'com.apple.speech.synthesis.voice.Victoria'
    
    def test_property_configuration(self, mock_full_engine):
        """Test all property configurations."""
        result = speak_with_pyttsx3("Testing properties")
        
        assert result is True
        
        # Check all properties were set
        property_calls = mock_full_engine.setProperty.call_args_list
        
        # Rate property
        rate_calls = [call for call in property_calls if call[0][0] == 'rate']
        assert len(rate_calls) == 1
        assert rate_calls[0][0][1] == 180
        
        # Volume property
        volume_calls = [call for call in property_calls if call[0][0] == 'volume']
        assert len(volume_calls) == 1
        assert volume_calls[0][0][1] == 0.8
    
    def test_engine_lifecycle(self, mock_full_engine):
        """Test proper engine lifecycle management."""
        result = speak_with_pyttsx3("Testing lifecycle")
        
        assert result is True
        
        # Verify proper call order
        import pyttsx3
        assert pyttsx3.init.called
        assert mock_full_engine.setProperty.called
        assert mock_full_engine.say.called
        assert mock_full_engine.runAndWait.called
        
        # Verify runAndWait is called after say
        say_order = None
        wait_order = None
        
        for i, call in enumerate(mock_full_engine.method_calls):
            if call[0] == 'say':
                say_order = i
            elif call[0] == 'runAndWait':
                wait_order = i
        
        assert say_order is not None
        assert wait_order is not None
        assert say_order < wait_order

if __name__ == "__main__":
    pytest.main([__file__, "-v"])