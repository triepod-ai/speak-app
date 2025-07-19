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
Manual audio test suite for comparing TTS providers.
Tests all providers side-by-side with actual audio output.
Run with: pytest tests/test_manual_provider_comparison.py -v -s -m audio_output
"""

import os
import sys
import time
import pytest
from pathlib import Path
from datetime import datetime

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

class TestProviderComparison:
    """Compare TTS providers with actual audio output."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        print("\n" + "="*60)
        print("🔊 PROVIDER COMPARISON TEST - Please listen carefully")
        print("="*60)
        yield
        print("\n⏸️  Test completed")
        time.sleep(2)
    
    def _test_provider(self, provider_name, speak_func, text, **kwargs):
        """Helper to test a provider and measure performance."""
        print(f"\n🎤 Testing {provider_name}...")
        
        # Check API key for cloud providers
        if provider_name == "ElevenLabs" and not os.getenv("ELEVENLABS_API_KEY"):
            print(f"⚠️  Skipping {provider_name} - No API key")
            return None
        if provider_name == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
            print(f"⚠️  Skipping {provider_name} - No API key")
            return None
        
        start_time = time.time()
        try:
            result = speak_func(text, **kwargs)
            end_time = time.time()
            
            if result:
                latency = end_time - start_time
                print(f"✅ {provider_name}: Success (Latency: {latency:.2f}s)")
                return latency
            else:
                print(f"❌ {provider_name}: Failed")
                return None
        except Exception as e:
            print(f"❌ {provider_name}: Error - {e}")
            return None
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_basic_comparison(self):
        """Basic comparison of all providers with same text."""
        print("\n🎯 Basic provider comparison...")
        
        test_text = "Hello! This is a test of the text-to-speech system. Each provider has unique characteristics."
        
        # Test each provider
        results = []
        
        # pyttsx3 (offline)
        latency = self._test_provider("pyttsx3", speak_with_pyttsx3, test_text)
        if latency:
            results.append(("pyttsx3", latency))
        time.sleep(2)
        
        # OpenAI
        latency = self._test_provider("OpenAI", speak_with_openai, test_text)
        if latency:
            results.append(("OpenAI", latency))
        time.sleep(2)
        
        # ElevenLabs
        latency = self._test_provider("ElevenLabs", speak_with_elevenlabs, test_text)
        if latency:
            results.append(("ElevenLabs", latency))
        time.sleep(2)
        
        # Summary
        if results:
            print("\n📊 Performance Summary:")
            for provider, latency in sorted(results, key=lambda x: x[1]):
                print(f"  {provider}: {latency:.2f}s")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_quality_comparison(self):
        """Compare voice quality across providers."""
        print("\n🎨 Voice quality comparison...")
        
        # Test with content that highlights voice quality
        quality_text = """
        The rain in Spain stays mainly in the plain. 
        She sells seashells by the seashore. 
        Peter Piper picked a peck of pickled peppers.
        """
        
        print("\n📝 Testing pronunciation and clarity with tongue twisters...")
        
        self._test_provider("pyttsx3", speak_with_pyttsx3, quality_text)
        time.sleep(3)
        
        self._test_provider("OpenAI", speak_with_openai, quality_text)
        time.sleep(3)
        
        self._test_provider("ElevenLabs", speak_with_elevenlabs, quality_text)
        time.sleep(3)
        
        print("\n❓ Which provider had the clearest pronunciation?")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_naturalness_comparison(self):
        """Compare how natural each provider sounds."""
        print("\n🌿 Naturalness comparison...")
        
        natural_text = """
        I was thinking about going to the store later, but I'm not sure if I have time. 
        Maybe I'll just order online instead. What do you think?
        """
        
        print("\n💬 Testing conversational naturalness...")
        
        self._test_provider("pyttsx3", speak_with_pyttsx3, natural_text)
        time.sleep(4)
        
        self._test_provider("OpenAI", speak_with_openai, natural_text)
        time.sleep(4)
        
        self._test_provider("ElevenLabs", speak_with_elevenlabs, natural_text)
        time.sleep(4)
        
        print("\n❓ Which provider sounded most natural and conversational?")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_emotion_comparison(self):
        """Compare emotional expression across providers."""
        print("\n😊 Emotional expression comparison...")
        
        emotions = [
            ("Excitement", "Oh wow! This is absolutely incredible! I can't believe it!"),
            ("Sadness", "I'm feeling quite down today. Everything seems gray and gloomy."),
            ("Professional", "The quarterly report indicates substantial growth in all sectors."),
            ("Friendly", "Hey there! It's so nice to meet you. How's your day going?")
        ]
        
        for emotion, text in emotions:
            print(f"\n🎭 Testing {emotion}...")
            
            self._test_provider("pyttsx3", speak_with_pyttsx3, text)
            time.sleep(3)
            
            self._test_provider("OpenAI", speak_with_openai, text)
            time.sleep(3)
            
            self._test_provider("ElevenLabs", speak_with_elevenlabs, text)
            time.sleep(3)
            
            print("-" * 40)
        
        print("\n❓ Which provider best conveyed emotions?")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_technical_content_comparison(self):
        """Compare handling of technical content."""
        print("\n💻 Technical content comparison...")
        
        tech_text = """
        The API uses REST architecture with JSON responses. 
        Status code 404 indicates 'not found'. 
        The regex pattern is: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
        Memory complexity is O(n log n).
        """
        
        print("\n🔧 Testing technical terminology and symbols...")
        
        self._test_provider("pyttsx3", speak_with_pyttsx3, tech_text)
        time.sleep(5)
        
        self._test_provider("OpenAI", speak_with_openai, tech_text)
        time.sleep(5)
        
        self._test_provider("ElevenLabs", speak_with_elevenlabs, tech_text)
        time.sleep(5)
        
        print("\n❓ Which provider handled technical content best?")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_speed_comparison(self):
        """Compare different speech speeds (where supported)."""
        print("\n⚡ Speech speed comparison...")
        
        speed_text = "This text is being read at different speeds to compare pacing and clarity."
        
        print("\n🐌 Slow speed (0.75x)...")
        if os.getenv("OPENAI_API_KEY"):
            self._test_provider("OpenAI", speak_with_openai, speed_text, speed=0.75)
            time.sleep(3)
        
        print("\n🚶 Normal speed (1.0x)...")
        self._test_provider("pyttsx3", speak_with_pyttsx3, speed_text)
        time.sleep(2)
        self._test_provider("OpenAI", speak_with_openai, speed_text, speed=1.0)
        time.sleep(2)
        self._test_provider("ElevenLabs", speak_with_elevenlabs, speed_text)
        time.sleep(2)
        
        print("\n🏃 Fast speed (1.25x)...")
        if os.getenv("OPENAI_API_KEY"):
            self._test_provider("OpenAI", speak_with_openai, speed_text, speed=1.25)
            time.sleep(3)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_voice_variety_comparison(self):
        """Compare voice variety options across providers."""
        print("\n🎭 Voice variety comparison...")
        
        test_text = "Testing different voice options available in each provider."
        
        print("\n🔵 pyttsx3 voices:")
        # pyttsx3 typically has system voices
        self._test_provider("pyttsx3", speak_with_pyttsx3, test_text)
        time.sleep(3)
        
        print("\n🟢 OpenAI voices:")
        if os.getenv("OPENAI_API_KEY"):
            voices = ["nova", "alloy", "echo", "shimmer"]
            for voice in voices:
                print(f"  Testing {voice} voice...")
                self._test_provider("OpenAI", speak_with_openai, test_text, voice=voice)
                time.sleep(2)
        
        print("\n🟣 ElevenLabs voices:")
        if os.getenv("ELEVENLABS_API_KEY"):
            voices = [
                ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
                ("Domi", "AZnzlk1XvdvUeBnXmlld")
            ]
            for name, voice_id in voices:
                print(f"  Testing {name} voice...")
                self._test_provider("ElevenLabs", speak_with_elevenlabs, test_text, voice_id=voice_id)
                time.sleep(2)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.slow
    def test_long_content_comparison(self):
        """Compare providers with longer content."""
        print("\n📚 Long content comparison...")
        
        long_text = """
        This is a test of how each text-to-speech provider handles longer content. 
        When processing extended passages, it's important to maintain consistent 
        voice quality, appropriate pacing, and clear pronunciation throughout.
        
        Different providers may have varying approaches to handling paragraph breaks, 
        sentence boundaries, and natural pauses. Some may sound more robotic over 
        time, while others maintain a natural flow.
        
        This test helps identify which provider is best suited for applications 
        requiring longer audio content, such as audiobooks, educational materials, 
        or extended notifications.
        """
        
        print("\n📖 Testing extended content handling...")
        
        start = time.time()
        self._test_provider("pyttsx3", speak_with_pyttsx3, long_text)
        pyttsx3_duration = time.time() - start
        time.sleep(2)
        
        start = time.time()
        self._test_provider("OpenAI", speak_with_openai, long_text)
        openai_duration = time.time() - start
        time.sleep(2)
        
        start = time.time()
        self._test_provider("ElevenLabs", speak_with_elevenlabs, long_text)
        elevenlabs_duration = time.time() - start
        
        print("\n⏱️ Duration comparison:")
        print(f"  pyttsx3: {pyttsx3_duration:.2f}s")
        if os.getenv("OPENAI_API_KEY"):
            print(f"  OpenAI: {openai_duration:.2f}s")
        if os.getenv("ELEVENLABS_API_KEY"):
            print(f"  ElevenLabs: {elevenlabs_duration:.2f}s")
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    def test_cost_benefit_analysis(self):
        """Provide cost-benefit analysis of providers."""
        print("\n💰 Cost-Benefit Analysis...")
        
        print("\n📊 Provider Characteristics:")
        print("\n1. pyttsx3 (Offline)")
        print("   ✅ Free and instant")
        print("   ✅ No internet required")
        print("   ✅ Privacy (no data sent)")
        print("   ❌ Lower quality")
        print("   ❌ Limited voices")
        
        print("\n2. OpenAI")
        print("   ✅ Good quality")
        print("   ✅ Multiple voices")
        print("   ✅ Speed control")
        print("   ❌ Requires API key")
        print("   ❌ Costs per character")
        
        print("\n3. ElevenLabs")
        print("   ✅ Best quality")
        print("   ✅ Most natural")
        print("   ✅ Many voices")
        print("   ❌ Most expensive")
        print("   ❌ Requires API key")
        
        # Quick demo of each
        demo_text = "This is a quick demonstration of each provider."
        
        print("\n🎵 Quick quality demo...")
        self._test_provider("pyttsx3", speak_with_pyttsx3, demo_text)
        time.sleep(2)
        self._test_provider("OpenAI", speak_with_openai, demo_text)
        time.sleep(2)
        self._test_provider("ElevenLabs", speak_with_elevenlabs, demo_text)
    
    @pytest.mark.manual
    @pytest.mark.audio_output
    @pytest.mark.interactive
    def test_interactive_comparison(self):
        """Interactive mode to compare providers with custom text."""
        print("\n🎮 Interactive provider comparison mode")
        print("Enter text to hear it spoken by all available providers")
        
        while True:
            print("\n" + "-"*40)
            user_text = input("📝 Enter text (or press Enter to quit): ")
            if not user_text:
                break
            
            print("\n🔊 Playing through all providers...")
            
            # Test each provider
            self._test_provider("pyttsx3", speak_with_pyttsx3, user_text)
            time.sleep(2)
            
            self._test_provider("OpenAI", speak_with_openai, user_text)
            time.sleep(2)
            
            self._test_provider("ElevenLabs", speak_with_elevenlabs, user_text)
            time.sleep(2)
            
            print("\n✅ Comparison complete")

if __name__ == "__main__":
    # Run provider comparison tests
    pytest.main([__file__, "-v", "-s", "-m", "audio_output"])