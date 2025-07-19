#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "openai>=1.0.0",
#   "python-dotenv>=1.0.0",
#   "pygame>=2.5.0",
# ]
# ///

"""
Manual test for all OpenAI voices.
Run with: pytest tests/test_manual_openai_voices.py -v -s -m audio_output
Or directly: python tests/test_manual_openai_voices.py
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

# Custom marker for audio output tests
pytestmark = [pytest.mark.audio_output, pytest.mark.openai]


class TestOpenAIVoices:
    """Test all OpenAI voices with actual audio output."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        # Check if API key is available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        print("\n" + "="*60)
        print("🔊 OPENAI VOICE TEST - Please listen to the output")
        print("="*60)
        yield
        print("\n⏸️  Pausing before next test...")
        time.sleep(2)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_alloy_voice(self):
        """Test Alloy voice - neutral and balanced."""
        print("\n🎤 Testing Alloy voice...")
        
        text = """
        Hello, this is Alloy speaking. I have a neutral and balanced voice,
        suitable for general-purpose applications. My tone is clear and 
        professional, making me ideal for various content types.
        """
        
        result = speak_with_openai(text, voice="alloy")
        assert result is True
        print("✅ Alloy voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_echo_voice(self):
        """Test Echo voice - warm and engaging."""
        print("\n🎤 Testing Echo voice...")
        
        text = """
        Hi there! This is Echo. I have a warm and engaging voice that's
        perfect for conversational content. My friendly tone helps create
        a connection with listeners.
        """
        
        result = speak_with_openai(text, voice="echo")
        assert result is True
        print("✅ Echo voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_fable_voice(self):
        """Test Fable voice - expressive and dynamic."""
        print("\n🎤 Testing Fable voice...")
        
        text = """
        Greetings! I'm Fable. My voice is expressive and dynamic,
        perfect for storytelling and narrative content. I bring
        characters and stories to life with varied intonation.
        """
        
        result = speak_with_openai(text, voice="fable")
        assert result is True
        print("✅ Fable voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_onyx_voice(self):
        """Test Onyx voice - deep and authoritative."""
        print("\n🎤 Testing Onyx voice...")
        
        text = """
        Good day. This is Onyx speaking. I have a deep, authoritative voice
        that conveys confidence and expertise. My tone is perfect for
        professional presentations and educational content.
        """
        
        result = speak_with_openai(text, voice="onyx")
        assert result is True
        print("✅ Onyx voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_nova_voice(self):
        """Test Nova voice - natural and conversational."""
        print("\n🎤 Testing Nova voice (default)...")
        
        text = """
        Hello! I'm Nova, the default OpenAI voice. I sound natural and
        conversational, making me perfect for everyday use. My balanced
        tone works well for most applications.
        """
        
        result = speak_with_openai(text, voice="nova")
        assert result is True
        print("✅ Nova voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_shimmer_voice(self):
        """Test Shimmer voice - gentle and soothing."""
        print("\n🎤 Testing Shimmer voice...")
        
        text = """
        Hello, this is Shimmer. My voice is gentle and soothing,
        ideal for meditation, relaxation content, or bedtime stories.
        I speak with a calm and peaceful tone.
        """
        
        result = speak_with_openai(text, voice="shimmer")
        assert result is True
        print("✅ Shimmer voice test completed")
        time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voice_comparison_same_text(self):
        """Compare all voices with the same text."""
        print("\n🎯 Voice comparison - same text, different voices...")
        
        test_text = "Testing OpenAI text-to-speech. Each voice has unique characteristics."
        
        voices = [
            ("alloy", "neutral and balanced"),
            ("echo", "warm and engaging"),
            ("fable", "expressive and dynamic"),
            ("onyx", "deep and authoritative"),
            ("nova", "natural and conversational"),
            ("shimmer", "gentle and soothing")
        ]
        
        for voice, description in voices:
            print(f"\n🎭 {voice.capitalize()} ({description})...")
            result = speak_with_openai(test_text, voice=voice)
            assert result is True
            time.sleep(3)
        
        print("\n✅ Voice comparison completed")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voices_with_different_speeds(self):
        """Test voices at different speeds."""
        print("\n⚡ Testing different speech speeds...")
        
        text = "Testing speech speed variations."
        speeds = [0.75, 1.0, 1.25]
        
        # Test with Nova voice
        for speed in speeds:
            print(f"\n🎚️ Nova at {speed}x speed...")
            result = speak_with_openai(text, voice="nova", speed=speed)
            assert result is True
            time.sleep(2)
        
        print("\n✅ Speed variation test completed")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voices_with_models(self):
        """Test voices with different models (tts-1 vs tts-1-hd)."""
        print("\n🎨 Testing different models...")
        
        text = "Comparing standard and HD audio quality."
        
        # Test Nova with both models
        print("\n📱 Nova with tts-1 (standard)...")
        result = speak_with_openai(text, voice="nova", model="tts-1")
        assert result is True
        time.sleep(3)
        
        print("\n🎧 Nova with tts-1-hd (high definition)...")
        result = speak_with_openai(text, voice="nova", model="tts-1-hd")
        assert result is True
        time.sleep(3)
        
        print("\n✅ Model comparison completed")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.interactive
    def test_interactive_voice_selection(self):
        """Interactive test to try any voice with custom text."""
        print("\n🎮 Interactive OpenAI voice testing")
        
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        print("\nAvailable voices:")
        for i, voice in enumerate(voices, 1):
            print(f"{i}. {voice}")
        
        while True:
            print("\n" + "-"*40)
            voice_input = input("Enter voice number (1-6) or voice name (or Enter to quit): ")
            if not voice_input:
                break
            
            # Determine voice
            if voice_input.isdigit():
                idx = int(voice_input) - 1
                if 0 <= idx < len(voices):
                    selected_voice = voices[idx]
                else:
                    print("Invalid number")
                    continue
            elif voice_input.lower() in voices:
                selected_voice = voice_input.lower()
            else:
                print("Invalid voice")
                continue
            
            text_input = input(f"Enter text for {selected_voice} (or Enter for default): ")
            if not text_input:
                text_input = f"Hello, this is {selected_voice} speaking. How do you like my voice?"
            
            speed_input = input("Enter speed (0.25-4.0, or Enter for 1.0): ")
            speed = 1.0
            if speed_input:
                try:
                    speed = float(speed_input)
                    speed = max(0.25, min(4.0, speed))
                except ValueError:
                    print("Invalid speed, using 1.0")
            
            print(f"\n🔊 Playing {selected_voice} at {speed}x speed...")
            result = speak_with_openai(text_input, voice=selected_voice, speed=speed)
            if result:
                print("✅ Success")
            else:
                print("❌ Failed")


def main():
    """Run voice tests directly."""
    print("🎤 OpenAI Voice Test Suite")
    print("This will test all 6 OpenAI voices")
    print("-" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found!")
        print("Set OPENAI_API_KEY environment variable")
        return
    
    # Create test instance
    tester = TestOpenAIVoices()
    
    # Test each voice
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    for voice in voices:
        print(f"\n🎤 Testing {voice}...")
        try:
            result = speak_with_openai(
                f"Hello, this is {voice}. Testing OpenAI text-to-speech.",
                voice=voice
            )
            if result:
                print(f"✅ {voice} works!")
            else:
                print(f"❌ {voice} failed!")
        except Exception as e:
            print(f"❌ {voice} error: {e}")
        
        time.sleep(2)
    
    print("\n✅ All voices tested!")


if __name__ == "__main__":
    # Option 1: Run with pytest
    # pytest.main([__file__, "-v", "-s", "-m", "audio_output"])
    
    # Option 2: Run directly
    main()