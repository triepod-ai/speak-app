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
Manual audio test suite for ElevenLabs voices.
Tests 4 different ElevenLabs voices with actual audio output.
Run with: pytest tests/test_manual_elevenlabs_voices.py -v -s -m audio_output
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

from tts.elevenlabs_tts import (
    speak_with_elevenlabs, 
    get_available_voices,
    find_voice_by_name,
    validate_voice_id
)

# Custom marker for audio output tests
pytestmark = [pytest.mark.audio_output, pytest.mark.elevenlabs]

class TestElevenLabsVoices:
    """Test different ElevenLabs voices with actual audio output."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        # Check if API key is available
        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available")
        
        print("\n" + "="*60)
        print("🔊 ELEVENLABS VOICE TEST - Please listen to the output")
        print("="*60)
        yield
        print("\n⏸️  Pausing before next test...")
        time.sleep(2)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_rachel_voice(self):
        """Test Rachel voice - calm and composed narration."""
        print("\n🎤 Testing Rachel voice (calm narration)...")
        
        text = """
        Hello, this is Rachel speaking. I have a calm and composed voice, 
        perfect for narration and long-form content. Notice how my tone 
        remains steady and professional throughout this message.
        """
        
        result = speak_with_elevenlabs(
            text, 
            voice_id="21m00Tcm4TlvDq8ikWAM"
        )
        
        assert result is True
        print("✅ Rachel voice test completed")
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_domi_voice(self):
        """Test Domi voice - strong and confident."""
        print("\n🎤 Testing Domi voice (strong and confident)...")
        
        text = """
        Hi there! This is Domi. I have a strong and confident voice that's 
        perfect for commercials and presentations. My delivery is more 
        energetic and assertive compared to other voices.
        """
        
        result = speak_with_elevenlabs(
            text,
            voice_id="AZnzlk1XvdvUeBnXmlld"
        )
        
        assert result is True
        print("✅ Domi voice test completed")
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_bella_voice(self):
        """Test Bella voice - soft and gentle."""
        print("\n🎤 Testing Bella voice (soft and gentle)...")
        
        text = """
        Hello, I'm Bella. My voice is soft and gentle, making it ideal for 
        audiobooks and meditation content. Listen to how soothing and 
        peaceful my tone is throughout this passage.
        """
        
        result = speak_with_elevenlabs(
            text,
            voice_id="EXAVITQu4vr4xnSDxMaL"
        )
        
        assert result is True
        print("✅ Bella voice test completed")
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_adam_voice(self):
        """Test Adam voice - deep and authoritative male voice."""
        print("\n🎤 Testing Adam voice (deep male voice)...")
        
        text = """
        Good day, this is Adam speaking. I have a deep, authoritative voice 
        that works well for professional content and announcements. My tone 
        conveys trust and reliability.
        """
        
        # Adam's voice ID
        result = speak_with_elevenlabs(
            text,
            voice_id="pNInz6obpgDQGcFmaJgB"
        )
        
        assert result is True
        print("✅ Adam voice test completed")
        time.sleep(5)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voice_comparison(self):
        """Compare all 4 voices with the same text."""
        print("\n🎯 Voice comparison test - same text, different voices...")
        
        test_text = "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet."
        
        voices = [
            ("Rachel", "21m00Tcm4TlvDq8ikWAM", "calm narration"),
            ("Domi", "AZnzlk1XvdvUeBnXmlld", "strong and confident"),
            ("Bella", "EXAVITQu4vr4xnSDxMaL", "soft and gentle"),
            ("Adam", "pNInz6obpgDQGcFmaJgB", "deep male voice")
        ]
        
        for name, voice_id, description in voices:
            print(f"\n🎭 {name} ({description})...")
            result = speak_with_elevenlabs(test_text, voice_id=voice_id)
            assert result is True
            time.sleep(3)
        
        print("\n✅ Voice comparison completed")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voice_settings_variations(self):
        """Test Rachel voice with different stability and similarity settings."""
        print("\n⚙️ Testing voice settings variations with Rachel...")
        
        base_text = "Testing different voice settings. Notice how the stability and similarity affect my speech patterns."
        
        settings_tests = [
            ("Low stability (more variation)", {"stability": 0.3, "similarity_boost": 0.5}),
            ("Default settings", {"stability": 0.5, "similarity_boost": 0.75}),
            ("High stability (more consistent)", {"stability": 0.8, "similarity_boost": 0.9}),
            ("Maximum stability", {"stability": 1.0, "similarity_boost": 1.0})
        ]
        
        for description, settings in settings_tests:
            print(f"\n🎚️ {description}: {settings}")
            result = speak_with_elevenlabs(
                base_text,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
                voice_settings=settings
            )
            assert result is True
            time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_emotional_delivery(self):
        """Test how different voices handle emotional content."""
        print("\n😊 Testing emotional delivery across voices...")
        
        emotional_texts = [
            ("Excitement", "Wow! This is absolutely incredible! I can't wait to share this news!"),
            ("Sadness", "I'm feeling quite melancholy today. The rain matches my mood perfectly."),
            ("Professional", "Our quarterly earnings exceeded expectations by fifteen percent."),
            ("Storytelling", "Once upon a time, in a land far, far away, there lived a wise old wizard.")
        ]
        
        voices = [
            ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
            ("Bella", "EXAVITQu4vr4xnSDxMaL")
        ]
        
        for emotion, text in emotional_texts:
            print(f"\n🎭 {emotion}:")
            for voice_name, voice_id in voices:
                print(f"  → {voice_name} speaking...")
                result = speak_with_elevenlabs(text, voice_id=voice_id)
                assert result is True
                time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_technical_content_voices(self):
        """Test how different voices handle technical content."""
        print("\n💻 Testing technical content delivery...")
        
        technical_text = """
        The API endpoint returns a JSON response with status code 200. 
        The function uses async/await syntax with a try-catch block. 
        Memory usage is optimized using a hash map with O(1) lookup time.
        """
        
        voices = [
            ("Rachel", "21m00Tcm4TlvDq8ikWAM", "professional narration"),
            ("Adam", "pNInz6obpgDQGcFmaJgB", "authoritative delivery")
        ]
        
        for name, voice_id, style in voices:
            print(f"\n🔧 {name} - {style}...")
            result = speak_with_elevenlabs(technical_text, voice_id=voice_id)
            assert result is True
            time.sleep(4)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_list_available_voices(self):
        """List all available voices from API."""
        print("\n📋 Fetching available voices from ElevenLabs API...")
        
        voices = get_available_voices()
        if voices:
            print(f"\n✅ Found {len(voices)} voices:")
            for voice in voices[:10]:  # Show first 10
                print(f"  • {voice['name']} ({voice['voice_id']})")
                print(f"    Category: {voice.get('category', 'N/A')}")
                if 'labels' in voice:
                    labels = voice['labels']
                    print(f"    Labels: {labels.get('gender', 'N/A')} {labels.get('age', 'N/A')} {labels.get('accent', 'N/A')}")
        else:
            print("❌ Could not fetch voices")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voice_by_name_lookup(self):
        """Test finding voices by name."""
        print("\n🔍 Testing voice lookup by name...")
        
        test_names = ["Rachel", "Domi", "Bella", "Adam"]
        
        for name in test_names:
            voice_id = find_voice_by_name(name)
            if voice_id:
                print(f"✅ Found {name}: {voice_id}")
                # Test with a short phrase
                result = speak_with_elevenlabs(
                    f"This is {name} speaking.",
                    voice_id=voice_id
                )
                assert result is True
                time.sleep(2)
            else:
                print(f"❌ Could not find {name}")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.slow
    def test_voice_endurance(self):
        """Test voices with longer content to check consistency."""
        print("\n📚 Testing voice endurance with longer content...")
        
        long_text = """
        This is a comprehensive test of voice consistency over extended passages. 
        We're evaluating how well the voice maintains its character and quality 
        when processing longer texts that might be used in real-world applications.
        
        In practical use cases, such as audiobook narration or educational content, 
        the text-to-speech system needs to handle multiple paragraphs while keeping 
        the voice natural and engaging. This includes proper pacing, appropriate 
        pauses at punctuation marks, and maintaining energy throughout.
        
        The test concludes by verifying that the voice remains clear and pleasant 
        to listen to, even after speaking for an extended period. Thank you for 
        listening to this voice endurance test.
        """
        
        voices = [
            ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
            ("Bella", "EXAVITQu4vr4xnSDxMaL")
        ]
        
        for name, voice_id in voices:
            print(f"\n📖 {name} reading long content...")
            result = speak_with_elevenlabs(long_text, voice_id=voice_id)
            assert result is True
            time.sleep(2)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.interactive
    def test_interactive_voice_selection(self):
        """Interactive mode to test any voice with custom text."""
        print("\n🎮 Interactive voice testing mode")
        
        # Get available voices
        voices = get_available_voices()
        if not voices:
            print("❌ Could not fetch available voices")
            return
        
        # Show first 10 voices
        print("\nAvailable voices:")
        for i, voice in enumerate(voices[:10], 1):
            print(f"{i}. {voice['name']} - {voice.get('labels', {}).get('description', 'No description')}")
        
        while True:
            print("\n" + "-"*40)
            voice_input = input("Enter voice number (1-10) or voice name (or Enter to quit): ")
            if not voice_input:
                break
            
            # Find voice
            voice_id = None
            voice_name = None
            
            if voice_input.isdigit():
                idx = int(voice_input) - 1
                if 0 <= idx < len(voices):
                    voice_id = voices[idx]['voice_id']
                    voice_name = voices[idx]['name']
            else:
                voice_id = find_voice_by_name(voice_input)
                voice_name = voice_input
            
            if not voice_id:
                print(f"❌ Voice not found: {voice_input}")
                continue
            
            text_input = input(f"Enter text for {voice_name} (or Enter for default): ")
            if not text_input:
                text_input = f"Hello, this is {voice_name} speaking. How do you like my voice?"
            
            print(f"\n🔊 Playing {voice_name}...")
            result = speak_with_elevenlabs(text_input, voice_id=voice_id)
            if result:
                print("✅ Success")
            else:
                print("❌ Failed")

if __name__ == "__main__":
    # Run ElevenLabs voice tests
    pytest.main([__file__, "-v", "-s", "-m", "audio_output"])