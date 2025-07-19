#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "requests>=2.31.0",
#   "pygame>=2.5.0",
#   "python-dotenv>=1.0.0",
#   "openai>=1.0.0",
#   "pyttsx3>=2.90",
# ]
# ///

"""
Manual audio test suite for speak-app.
These tests produce actual audio output for human verification.
Run with: pytest tests/test_manual_audio.py -v -s -m audio_output
"""

import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.tts_provider import TTSProvider
from tts.elevenlabs_tts import speak_with_elevenlabs
from tts.openai_tts import speak_with_openai
from tts.pyttsx3_tts import speak_with_pyttsx3

# Custom marker for audio output tests
pytestmark = pytest.mark.audio_output

class TestManualAudio:
    """Manual audio tests that produce actual sound output."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        print("\n" + "="*60)
        print("🔊 AUDIO TEST - Please listen to the output")
        print("="*60)
        yield
        print("\n⏸️  Pausing before next test...")
        time.sleep(2)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_basic_tts_provider(self):
        """Test basic TTS provider functionality with audio."""
        print("\n🎯 Testing basic TTS provider (auto-selection)...")
        
        tts = TTSProvider()
        text = "Hello! This is a test of the automatic TTS provider selection."
        
        result = tts.speak(text)
        assert result is True
        
        print(f"✅ Successfully used provider: {tts.provider}")
        time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.elevenlabs
    def test_elevenlabs_basic_audio(self):
        """Test ElevenLabs with actual audio output."""
        print("\n🎤 Testing ElevenLabs TTS...")
        
        # Check if API key is available
        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available")
        
        text = "Testing ElevenLabs text-to-speech. This is Rachel speaking with a calm and composed voice."
        result = speak_with_elevenlabs(text)
        
        assert result is True
        print("✅ ElevenLabs audio played successfully")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.openai
    def test_openai_basic_audio(self):
        """Test OpenAI with actual audio output."""
        print("\n🎤 Testing OpenAI TTS...")
        
        # Check if API key is available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        text = "Testing OpenAI text-to-speech. This is the Nova voice, optimized for natural conversation."
        result = speak_with_openai(text)
        
        assert result is True
        print("✅ OpenAI audio played successfully")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.pyttsx3
    def test_pyttsx3_basic_audio(self):
        """Test pyttsx3 with actual audio output."""
        print("\n🎤 Testing pyttsx3 (offline) TTS...")
        
        text = "Testing offline text-to-speech with pyttsx3. This works without any API keys."
        result = speak_with_pyttsx3(text)
        
        assert result is True
        print("✅ pyttsx3 audio played successfully")
        time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_all_providers_sequence(self):
        """Test all providers in sequence."""
        print("\n🔄 Testing all providers in sequence...")
        
        providers_tests = [
            ("pyttsx3", speak_with_pyttsx3, "Testing offline TTS with pyttsx3."),
            ("openai", speak_with_openai, "Testing OpenAI TTS with Nova voice."),
            ("elevenlabs", speak_with_elevenlabs, "Testing ElevenLabs TTS with Rachel voice.")
        ]
        
        for provider_name, speak_func, text in providers_tests:
            print(f"\n🎯 Testing {provider_name}...")
            
            # Skip if no API key for cloud providers
            if provider_name == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
                print(f"⚠️  Skipping {provider_name} - No API key")
                continue
            if provider_name == "openai" and not os.getenv("OPENAI_API_KEY"):
                print(f"⚠️  Skipping {provider_name} - No API key")
                continue
            
            try:
                result = speak_func(text)
                if result:
                    print(f"✅ {provider_name} successful")
                else:
                    print(f"❌ {provider_name} failed")
                time.sleep(3)
            except Exception as e:
                print(f"❌ {provider_name} error: {e}")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_technical_content(self):
        """Test TTS with technical content."""
        print("\n💻 Testing technical content pronunciation...")
        
        tts = TTSProvider()
        text = """
        The Python function uses async/await syntax. 
        The API endpoint is /api/v1/users. 
        The error code is HTTP 404. 
        The regex pattern is: ^[a-zA-Z0-9]+$
        """
        
        result = tts.speak(text)
        assert result is True
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_numbers_and_symbols(self):
        """Test TTS with numbers and symbols."""
        print("\n🔢 Testing numbers and symbols...")
        
        tts = TTSProvider()
        text = """
        The temperature is 72.5 degrees Fahrenheit.
        The price is $19.99 plus tax.
        The meeting is at 3:30 PM.
        My email is user@example.com.
        The calculation is: 2 + 2 = 4.
        """
        
        result = tts.speak(text)
        assert result is True
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_different_speeds(self):
        """Test TTS at different speeds (if supported)."""
        print("\n⚡ Testing different speech speeds...")
        
        # Only OpenAI supports speed adjustment
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Speed test requires OpenAI API key")
        
        text = "This message is played at different speeds."
        
        speeds = [0.75, 1.0, 1.25]
        for speed in speeds:
            print(f"\n🎚️ Testing speed: {speed}x")
            result = speak_with_openai(text, speed=speed)
            assert result is True
            time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_emotional_content(self):
        """Test TTS with different emotional contexts."""
        print("\n😊 Testing emotional content...")
        
        tts = TTSProvider()
        
        emotions = [
            ("Excitement", "Wow! This is absolutely amazing! I can't believe it worked!"),
            ("Calm", "Everything is proceeding smoothly. The system is functioning as expected."),
            ("Question", "Have you considered the implications? What do you think about this approach?"),
            ("Professional", "The quarterly report indicates a 15% increase in productivity.")
        ]
        
        for emotion, text in emotions:
            print(f"\n🎭 Testing {emotion}...")
            result = tts.speak(text)
            assert result is True
            time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.interactive
    def test_interactive_mode(self):
        """Interactive test mode for custom input."""
        print("\n🎮 Interactive test mode")
        print("Press Enter to skip, or type custom text to test:")
        
        tts = TTSProvider()
        
        while True:
            user_input = input("\n📝 Enter text (or press Enter to finish): ")
            if not user_input:
                break
            
            print("🔊 Playing...")
            result = tts.speak(user_input)
            if result:
                print("✅ Success")
            else:
                print("❌ Failed")
            time.sleep(1)

class TestAudioQuality:
    """Test audio quality and characteristics."""
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_volume_consistency(self):
        """Test volume consistency across providers."""
        print("\n🔊 Testing volume consistency...")
        
        test_phrase = "Testing volume level consistency."
        
        # Test each available provider
        if os.getenv("ELEVENLABS_API_KEY"):
            print("\n📊 ElevenLabs volume test...")
            speak_with_elevenlabs(test_phrase)
            time.sleep(2)
        
        if os.getenv("OPENAI_API_KEY"):
            print("\n📊 OpenAI volume test...")
            speak_with_openai(test_phrase)
            time.sleep(2)
        
        print("\n📊 pyttsx3 volume test...")
        speak_with_pyttsx3(test_phrase)
        time.sleep(2)
        
        print("\n❓ Were the volume levels consistent? (This is a manual check)")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.slow
    def test_long_content(self):
        """Test TTS with longer content."""
        print("\n📚 Testing long content...")
        
        tts = TTSProvider()
        text = """
        This is a test of longer content to verify that the text-to-speech system 
        can handle extended passages without issues. The system should maintain 
        consistent voice quality, pacing, and clarity throughout the entire passage. 
        
        We're testing multiple sentences, various punctuation marks, and different 
        paragraph structures. This helps ensure that the TTS system properly handles 
        natural pauses, emphasis, and intonation patterns that occur in longer texts.
        
        The final part of this test verifies that the audio playback completes 
        successfully without cutting off prematurely or experiencing buffer underruns.
        """
        
        result = tts.speak(text)
        assert result is True
        print("\n✅ Long content test completed")

if __name__ == "__main__":
    # Run manual audio tests
    pytest.main([__file__, "-v", "-s", "-m", "audio_output"])