#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=1.0.0",
#   "python-dotenv>=1.0.0",
#   "pygame>=2.5.0",
# ]
# ///

"""
OpenAI TTS provider for Claude Code hooks.
Uses OpenAI's text-to-speech API with multiple voice options.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv

# Voice options
VoiceType = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

def speak_with_openai(
    text: str, 
    voice: VoiceType = "nova",
    model: str = "tts-1",
    speed: float = 1.0
) -> bool:
    """
    Convert text to speech using OpenAI's TTS API.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model: TTS model to use (tts-1 or tts-1-hd)
        speed: Speech speed (0.25 to 4.0)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from openai import OpenAI
        import pygame
        
        # Load environment variables
        env_path = Path.home() / "brainpods" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not found in environment", file=sys.stderr)
            return False
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create speech
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed
        )
        
        # Save to temporary file and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            # Write response content to file (avoiding deprecated stream_to_file)
            for chunk in response.iter_bytes():
                tmp_file.write(chunk)
        
        # Play the audio using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Cleanup
        pygame.mixer.quit()
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"OpenAI TTS error: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for standalone usage."""
    import random
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OpenAI TTS provider for Claude Code')
    parser.add_argument('text', nargs='*', help='Text to speak')
    parser.add_argument('--voice', choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"], 
                        help='Voice to use (default: nova)')
    parser.add_argument('--model', choices=["tts-1", "tts-1-hd"], 
                        help='Model to use (default: tts-1)')
    parser.add_argument('--speed', type=float, 
                        help='Speech speed (0.25-4.0, default: 1.0)')
    parser.add_argument('--test', action='store_true', 
                        help='Test OpenAI TTS functionality')
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices')
    parser.add_argument('--test-voice', 
                        help='Test a specific voice')
    
    args = parser.parse_args()
    
    # Handle list voices
    if args.list_voices:
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        print("Available OpenAI voices:")
        for voice in voices:
            print(f"  - {voice}")
        sys.exit(0)
    
    # Handle test voice
    if args.test_voice:
        if args.test_voice not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            print(f"Invalid voice: {args.test_voice}", file=sys.stderr)
            sys.exit(1)
        
        test_text = f"Testing OpenAI voice: {args.test_voice}"
        success = speak_with_openai(test_text, voice=args.test_voice)
        sys.exit(0 if success else 1)
    
    # Get text from command line or use default
    if args.text:
        text = " ".join(args.text)
    elif args.test:
        # Fun test messages
        messages = [
            "OpenAI text-to-speech is operational!",
            "High-quality voice synthesis ready.",
            "Claude Code audio feedback enabled.",
            "Neural voice synthesis active.",
        ]
        text = random.choice(messages)
    else:
        # Default message
        text = "OpenAI TTS ready."
    
    # Get parameters from args or environment
    voice = args.voice or os.getenv("OPENAI_TTS_VOICE", "nova")
    model = args.model or os.getenv("OPENAI_TTS_MODEL", "tts-1")
    speed = args.speed or float(os.getenv("OPENAI_TTS_SPEED", "1.0"))
    
    # Validate parameters
    if voice not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
        print(f"Invalid voice: {voice}", file=sys.stderr)
        sys.exit(1)
    
    if model not in ["tts-1", "tts-1-hd"]:
        print(f"Invalid model: {model}", file=sys.stderr)
        sys.exit(1)
    
    if not (0.25 <= speed <= 4.0):
        print(f"Invalid speed: {speed} (must be between 0.25 and 4.0)", file=sys.stderr)
        sys.exit(1)
    
    # Speak the text
    success = speak_with_openai(text, voice=voice, model=model, speed=speed)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()