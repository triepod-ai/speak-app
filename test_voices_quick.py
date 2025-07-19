#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pygame>=2.5.0",
#   "requests>=2.31.0",
#   "python-dotenv>=1.0.0",
#   "openai>=1.0.0",
#   "pyttsx3>=2.90",
# ]
# ///
"""
Quick script to test if TTS voices are working.
Run directly: python test_voices_quick.py
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tts"))


def test_pyttsx3():
    """Test offline voice."""
    print("\n1️⃣ Testing pyttsx3 (offline)...")
    try:
        from tts.pyttsx3_tts import speak_with_pyttsx3
        result = speak_with_pyttsx3("Testing offline voice")
        if result:
            print("   ✅ pyttsx3 works!")
        else:
            print("   ❌ pyttsx3 failed")
        return result
    except Exception as e:
        print(f"   ❌ pyttsx3 error: {e}")
        return False


def test_openai():
    """Test OpenAI voices."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n2️⃣ OpenAI: ⚠️  No API key (set OPENAI_API_KEY)")
        return None
    
    print("\n2️⃣ Testing OpenAI...")
    try:
        from tts.openai_tts import speak_with_openai
        
        # Test default voice (nova)
        result = speak_with_openai("Testing OpenAI voice")
        if result:
            print("   ✅ OpenAI works!")
            
            # Quick test of other voices
            print("   Testing other voices...")
            voices = ["alloy", "echo", "onyx"]
            for voice in voices:
                print(f"   - {voice}...", end=" ")
                if speak_with_openai(f"Testing {voice}", voice=voice):
                    print("✓")
                else:
                    print("✗")
                time.sleep(1)
        else:
            print("   ❌ OpenAI failed")
        return result
    except Exception as e:
        print(f"   ❌ OpenAI error: {e}")
        return False


def test_elevenlabs():
    """Test ElevenLabs voices."""
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("\n3️⃣ ElevenLabs: ⚠️  No API key (set ELEVENLABS_API_KEY)")
        return None
    
    print("\n3️⃣ Testing ElevenLabs...")
    try:
        from tts.elevenlabs_tts import speak_with_elevenlabs
        result = speak_with_elevenlabs("Testing ElevenLabs voice")
        if result:
            print("   ✅ ElevenLabs works!")
        else:
            print("   ❌ ElevenLabs failed")
        return result
    except Exception as e:
        print(f"   ❌ ElevenLabs error: {e}")
        return False


def test_speak_command():
    """Test the main speak command."""
    print("\n4️⃣ Testing speak command...")
    try:
        import subprocess
        speak_path = Path(__file__).parent / "speak"
        
        if not speak_path.exists():
            print("   ❌ speak script not found")
            return False
        
        # Test with --off flag (no audio)
        result = subprocess.run(
            [str(speak_path), "--off", "Test message"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ speak command works!")
            
            # Test --status
            result = subprocess.run(
                [str(speak_path), "--status"],
                capture_output=True,
                text=True
            )
            if "Provider:" in result.stdout:
                print(f"   Current provider: {result.stdout.strip()}")
            
            return True
        else:
            print("   ❌ speak command failed")
            return False
    except Exception as e:
        print(f"   ❌ speak command error: {e}")
        return False


def main():
    """Run all voice tests."""
    print("🎤 Quick Voice Test")
    print("=" * 50)
    print("This will play audio to test each TTS provider")
    print("=" * 50)
    
    results = {
        "pyttsx3": test_pyttsx3(),
        "openai": test_openai(),
        "elevenlabs": test_elevenlabs(),
        "speak": test_speak_command()
    }
    
    # Summary
    print("\n📊 Summary:")
    print("-" * 30)
    
    working = [k for k, v in results.items() if v is True]
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    
    if working:
        print(f"✅ Working: {', '.join(working)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    if skipped:
        print(f"⚠️  Skipped (no API key): {', '.join(skipped)}")
    
    # Overall result
    if failed:
        print("\n❌ Some voices are not working!")
        return 1
    elif working:
        print("\n✅ All configured voices are working!")
        return 0
    else:
        print("\n⚠️  No voices tested successfully")
        return 1


if __name__ == "__main__":
    exit(main())