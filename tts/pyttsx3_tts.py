#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyttsx3>=2.90",
# ]
# ///

"""
Offline TTS provider using pyttsx3 for Claude Code hooks.
No API required - works completely offline.
Falls back to espeak directly on Linux if pyttsx3 has issues.
"""

import os
import sys
import random
import subprocess
import platform

def speak_with_espeak_direct(text: str) -> bool:
    """
    Fallback: Use espeak directly via subprocess on Linux.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Configure espeak parameters
        rate = os.getenv('ESPEAK_RATE', '180')  # Words per minute
        pitch = os.getenv('ESPEAK_PITCH', '50')  # 0-99
        
        # Build espeak command
        cmd = ['espeak', '-s', rate, '-p', pitch]
        
        # Add voice if specified
        voice = os.getenv('ESPEAK_VOICE', 'en')
        cmd.extend(['-v', voice])
        
        # Add the text
        cmd.append(text)
        
        # Execute espeak
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"espeak error: {result.stderr}", file=sys.stderr)
            return False
            
    except FileNotFoundError:
        print("espeak not found. Install with: sudo apt-get install espeak", file=sys.stderr)
        return False
    except Exception as e:
        print(f"espeak fallback error: {e}", file=sys.stderr)
        return False

def speak_with_pyttsx3(text: str) -> bool:
    """
    Convert text to speech using pyttsx3 (offline).
    
    Args:
        text: Text to convert to speech
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import pyttsx3
        
        # Initialize the TTS engine
        # Note: On some Linux systems, pyttsx3 may try to set an invalid voice during init
        # We'll try to work around this by catching the init error and trying with espeak driver
        engine = None
        init_errors = []
        
        # Try default initialization first
        try:
            engine = pyttsx3.init()
        except Exception as e:
            init_errors.append(f"Default init: {e}")
            
            # Try with espeak driver explicitly
            try:
                engine = pyttsx3.init('espeak')
            except Exception as e2:
                init_errors.append(f"Espeak init: {e2}")
        
        if engine is None:
            # If both failed, try espeak fallback on Linux
            if platform.system() == 'Linux':
                print("pyttsx3 initialization failed, trying espeak fallback...", file=sys.stderr)
                return speak_with_espeak_direct(text)
            else:
                # Report errors on non-Linux systems
                for err in init_errors:
                    print(f"pyttsx3 init error: {err}", file=sys.stderr)
                return False
        
        # Configure voice properties (these are safe to set)
        try:
            engine.setProperty('rate', 180)  # Speed of speech
        except Exception:
            pass  # Ignore if rate setting fails
            
        try:
            engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        except Exception:
            pass  # Ignore if volume setting fails
        
        # Optional: Try to use a specific voice if configured
        # Note: Voice selection can fail on some systems, so we handle it gracefully
        voice_id = os.getenv('PYTTSX3_VOICE_ID')
        if voice_id:
            try:
                engine.setProperty('voice', voice_id)
            except Exception as e:
                # If setting voice fails, just use default
                print(f"Warning: Could not set voice '{voice_id}': {e}. Using default voice.", file=sys.stderr)
        
        # Debug: List available voices if requested
        if os.getenv('PYTTSX3_LIST_VOICES', '').lower() == 'true':
            try:
                voices = engine.getProperty('voices')
                if voices:
                    print(f"Available pyttsx3 voices ({len(voices)}):", file=sys.stderr)
                    for voice in voices:
                        print(f"  - ID: {voice.id}, Name: {getattr(voice, 'name', 'Unknown')}", file=sys.stderr)
            except Exception:
                pass
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        return True
        
    except ImportError as e:
        print(f"pyttsx3 module not available: {e}", file=sys.stderr)
        print("Install with: pip install pyttsx3 or use uv run", file=sys.stderr)
        return False
    except RuntimeError as e:
        # Common on Linux when audio system is not available
        print(f"pyttsx3 runtime error: {e}", file=sys.stderr)
        print("Ensure audio system is available and espeak is installed", file=sys.stderr)
        return False
    except Exception as e:
        print(f"pyttsx3 TTS error: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for standalone usage."""
    
    # Get text from command line or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        # Fun default messages
        messages = [
            "Offline TTS is working perfectly!",
            "No internet? No problem! Local speech synthesis active.",
            "Claude Code offline voice ready.",
            "Privacy-first TTS enabled.",
        ]
        text = random.choice(messages)
    
    # Speak the text
    success = speak_with_pyttsx3(text)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()