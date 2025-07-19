#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "openai>=1.0.0",
#   "requests>=2.31.0",
#   "pygame>=2.5.0",
#   "python-dotenv>=1.0.0",
#   "pyttsx3>=2.90",
# ]
# ///

"""
Simple test to verify all voices work and produce sound.
Run with: pytest tests/test_all_voices_simple.py -v -s -m voice_test
"""

import os
import sys
import time
import pytest
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))

from tts.openai_tts import speak_with_openai
from tts.elevenlabs_tts import speak_with_elevenlabs, get_available_voices
from tts.pyttsx3_tts import speak_with_pyttsx3

# Mark all tests as voice tests
pytestmark = pytest.mark.voice_test


class TestAllVoicesSimple:
    """Simple tests to verify each voice works."""
    
    @pytest.mark.voice_test
    def test_pyttsx3_voice(self):
        """Test offline pyttsx3 voice works."""
        print("\n🔊 Testing pyttsx3 (offline)...")
        
        result = speak_with_pyttsx3("Testing offline voice. This should work without API keys.")
        
        assert result is True, "pyttsx3 voice should always work"
        print("✅ pyttsx3 voice works!")
        time.sleep(2)
    
    @pytest.mark.voice_test
    @pytest.mark.openai
    def test_all_openai_voices(self):
        """Test all OpenAI voices work."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        # All OpenAI voices
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        print("\n🔊 Testing all OpenAI voices...")
        
        for voice in voices:
            print(f"\n  Testing {voice}...")
            result = speak_with_openai(
                f"Hello, this is {voice} speaking. Testing OpenAI text to speech.",
                voice=voice
            )
            
            if result:
                print(f"  ✅ {voice} works!")
            else:
                print(f"  ❌ {voice} failed!")
                
            assert result is True, f"OpenAI voice {voice} should work"
            time.sleep(2)
        
        print("\n✅ All OpenAI voices tested successfully!")
    
    @pytest.mark.voice_test
    @pytest.mark.elevenlabs
    def test_elevenlabs_voices(self):
        """Test ElevenLabs voices work."""
        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available")
        
        print("\n🔊 Testing ElevenLabs voices...")
        
        # Get available voices
        voices = get_available_voices()
        if not voices:
            pytest.fail("Could not get ElevenLabs voices")
        
        # Test first 3 voices to save API credits
        test_voices = voices[:3]
        
        for voice in test_voices:
            voice_name = voice.get('name', 'Unknown')
            voice_id = voice.get('voice_id')
            
            print(f"\n  Testing {voice_name}...")
            result = speak_with_elevenlabs(
                f"Hello, this is {voice_name} speaking. Testing ElevenLabs.",
                voice_id=voice_id
            )
            
            if result:
                print(f"  ✅ {voice_name} works!")
            else:
                print(f"  ❌ {voice_name} failed!")
                
            assert result is True, f"ElevenLabs voice {voice_name} should work"
            time.sleep(2)
        
        print(f"\n✅ Tested {len(test_voices)} ElevenLabs voices successfully!")
    
    @pytest.mark.voice_test
    def test_voice_with_special_text(self):
        """Test voices with numbers and special characters."""
        print("\n🔊 Testing voices with special content...")
        
        test_texts = [
            "The price is $19.99",
            "The meeting is at 3:30 PM",
            "Error code: HTTP 404",
            "Email: test@example.com"
        ]
        
        # Use pyttsx3 for this test (always available)
        for text in test_texts:
            print(f"\n  Testing: {text}")
            result = speak_with_pyttsx3(text)
            assert result is True
            time.sleep(1)
        
        print("\n✅ Special text handling works!")
    
    @pytest.mark.voice_test
    @pytest.mark.quick
    def test_quick_voice_check(self):
        """Quick test with just one voice from each provider."""
        print("\n🚀 Quick voice check...")
        
        # Test pyttsx3
        print("\n  pyttsx3...")
        assert speak_with_pyttsx3("Quick test") is True
        time.sleep(1)
        
        # Test OpenAI if available
        if os.getenv("OPENAI_API_KEY"):
            print("  OpenAI...")
            assert speak_with_openai("Quick test", voice="nova") is True
            time.sleep(1)
        
        # Test ElevenLabs if available
        if os.getenv("ELEVENLABS_API_KEY"):
            print("  ElevenLabs...")
            assert speak_with_elevenlabs("Quick test") is True
            time.sleep(1)
        
        print("\n✅ Quick voice check complete!")


class TestVoiceErrors:
    """Test voice error handling."""
    
    @pytest.mark.voice_test
    def test_empty_text(self):
        """Test voices handle empty text gracefully."""
        print("\n🔊 Testing empty text handling...")
        
        # Should handle empty text without crashing
        result = speak_with_pyttsx3("")
        print(f"Empty text result: {result}")
        
        # Most TTS systems will fail with empty text, that's OK
        assert result in [True, False], "Should return boolean"
    
    @pytest.mark.voice_test
    def test_very_long_text(self):
        """Test voices handle long text."""
        print("\n🔊 Testing long text handling...")
        
        long_text = "This is a long text. " * 50  # About 250 words
        
        result = speak_with_pyttsx3(long_text[:500])  # Limit to 500 chars
        assert result is True, "Should handle long text"
        
        print("✅ Long text handling works!")


if __name__ == "__main__":
    # Run the voice tests
    print("🎤 Running simple voice tests...")
    print("This will produce actual audio output!")
    print("-" * 50)
    
    pytest.main([__file__, "-v", "-s", "-m", "voice_test"])