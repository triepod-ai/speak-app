# TTS Provider Documentation

This document provides detailed information about each Text-to-Speech provider supported by the speak command.

## Table of Contents

- [Provider Overview](#provider-overview)
- [ElevenLabs](#elevenlabs)
- [OpenAI TTS](#openai-tts)
- [pyttsx3](#pyttsx3)
- [Provider Selection Logic](#provider-selection-logic)
- [Adding Custom Providers](#adding-custom-providers)

## Provider Overview

The speak command supports multiple TTS providers to ensure maximum flexibility and reliability:

| Provider | Type | Requirements | Best For |
|----------|------|--------------|----------|
| ElevenLabs | Cloud | API Key | High-quality production use |
| OpenAI | Cloud | API Key | Alternative cloud option |
| pyttsx3 | Local | None | Offline/development use |

## ElevenLabs

### Overview
ElevenLabs provides state-of-the-art AI voice synthesis with natural-sounding voices and excellent prosody.

### Features
- Ultra-realistic AI voices
- Multiple voice options
- Emotion and style control
- Low latency streaming
- High-quality audio output

### Setup

1. Create an account at [elevenlabs.io](https://elevenlabs.io)
2. Generate an API key from your profile
3. Set the environment variable:
   ```bash
   export ELEVENLABS_API_KEY="your_api_key_here"
   ```

### Configuration

```bash
# Use specific voice (default: Rachel)
export ELEVENLABS_VOICE_ID="21m00Tcm4TlvDq8ikWAM"

# Available voices:
# - Rachel: 21m00Tcm4TlvDq8ikWAM
# - Domi: AZnzlk1XvdvUeBnXmlld
# - Bella: EXAVITQu4vr4xnSDxMaL
# - Antoni: ErXwobaYiN019PkySvjV
# - Elli: MF3mGyEYCl7XYWbV9V6O
# - Josh: TxGEqnHWrfWFTfGW9XjX
# - Arnold: VR6AewLTigWG4xSOukaG
# - Adam: pNInz6obpgDQGcFmaJgB
# - Sam: yoZ06aMxZJJ28mfd3POQ
```

### API Limits
- Free tier: 10,000 characters/month
- Paid tiers: Starting at $5/month for 30,000 characters
- Rate limiting: 2 requests/second

### Error Handling
The provider handles common errors:
- Invalid API key
- Rate limit exceeded
- Network timeouts
- Voice ID not found

### Code Example
```python
# Direct usage
from tts.elevenlabs_tts import speak_with_elevenlabs
speak_with_elevenlabs("Hello, world!", voice_id="21m00Tcm4TlvDq8ikWAM")
```

## OpenAI TTS

### Overview
OpenAI's Text-to-Speech API provides high-quality voice synthesis with multiple voice options.

### Features
- High-quality neural voices
- Multiple voice personalities
- Fast response times
- Reliable uptime
- Standard and HD quality options

### Setup

1. Create an OpenAI account
2. Generate an API key from the API settings
3. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

### Configuration

```bash
# Voice options: alloy, echo, fable, onyx, nova, shimmer
export OPENAI_TTS_VOICE="alloy"

# Model options: tts-1, tts-1-hd
export OPENAI_TTS_MODEL="tts-1"

# Speed: 0.25 to 4.0 (default: 1.0)
export OPENAI_TTS_SPEED="1.0"
```

### API Limits
- Rate limits based on your OpenAI plan
- Pricing: $0.015 per 1,000 characters (tts-1)
- HD pricing: $0.030 per 1,000 characters (tts-1-hd)

### Voice Descriptions
- **alloy**: Neutral and balanced
- **echo**: Warm and conversational  
- **fable**: Expressive and dynamic
- **onyx**: Deep and authoritative
- **nova**: Friendly and upbeat
- **shimmer**: Soft and pleasant

## pyttsx3

### Overview
pyttsx3 is a text-to-speech conversion library that works offline using system TTS engines.

### Features
- Works completely offline
- No API keys required
- Cross-platform support
- Customizable voice properties
- Zero latency

### System Dependencies

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get install espeak ffmpeg libespeak1

# Fedora
sudo dnf install espeak

# Arch
sudo pacman -S espeak
```

#### macOS
- Uses built-in `NSSpeechSynthesizer`
- No additional installation required

#### Windows
- Uses built-in SAPI5
- No additional installation required

### Configuration

```bash
# Voice selection (system-dependent)
export PYTTSX3_VOICE="english"

# Speech rate (words per minute)
export PYTTSX3_RATE="175"

# Volume (0.0 to 1.0)
export PYTTSX3_VOLUME="1.0"
```

### Available Voices

List system voices:
```python
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"ID: {voice.id}")
    print(f"Name: {voice.name}")
```

### Limitations
- Voice quality varies by system
- Limited voice customization
- No emotion or style control
- Platform-dependent behavior

## Provider Selection Logic

The speak command uses intelligent provider selection:

### Automatic Selection Priority
1. **Check API Keys**: Verify which cloud providers are configured
2. **Provider Preference**: Respect `TTS_PROVIDER` environment variable
3. **Fallback Order**:
   - ElevenLabs (if API key present)
   - OpenAI (if API key present)
   - pyttsx3 (always available)

### Selection Algorithm
```python
def select_provider():
    available = get_available_providers()
    
    if provider_preference != "auto":
        if provider_preference in available:
            return provider_preference
    
    # Priority order
    for provider in ["elevenlabs", "openai", "pyttsx3"]:
        if provider in available:
            return provider
    
    return None
```

### Manual Override
```bash
# Force specific provider
speak --provider pyttsx3 "Force offline TTS"

# Set default provider
export TTS_PROVIDER=elevenlabs
```

## Adding Custom Providers

### Provider Interface

Every provider must implement:

```python
def speak_with_provider(text: str, **kwargs) -> bool:
    """
    Convert text to speech.
    
    Args:
        text: Text to speak
        **kwargs: Provider-specific options
        
    Returns:
        True if successful, False otherwise
    """
```

### Integration Steps

1. Create provider script in `tts/` directory
2. Update `tts_provider.py` to recognize new provider
3. Add configuration documentation
4. Test thoroughly
5. Submit PR with examples

### Provider Requirements

- Handle errors gracefully
- Support timeout/cancellation
- Log errors to stderr
- Return proper exit codes
- Document all environment variables
- Provide fallback behavior

## Performance Comparison

| Provider | Latency | Quality | CPU Usage | Memory |
|----------|---------|---------|-----------|---------|
| ElevenLabs | 300-500ms | Excellent | Low | Low |
| OpenAI | 200-400ms | Very Good | Low | Low |
| pyttsx3 | <10ms | Good | Medium | Medium |

## Troubleshooting

### ElevenLabs Issues
- **401 Error**: Check API key validity
- **429 Error**: Rate limit exceeded, wait or upgrade plan
- **No audio**: Check audio device settings
- **Voice not found**: Verify voice ID

### OpenAI Issues
- **Invalid API key**: Ensure key starts with "sk-"
- **Model not found**: Check model name (tts-1 or tts-1-hd)
- **Timeout**: Increase timeout or check network

### pyttsx3 Issues
- **No audio on Linux**: Install espeak: `sudo apt-get install espeak`
- **Import error**: Reinstall with system packages
- **Voice not found**: Use system default
- **Segmentation fault**: Update system TTS engine

## Best Practices

1. **API Key Security**
   - Never commit API keys
   - Use environment variables
   - Rotate keys regularly

2. **Provider Selection**
   - Use ElevenLabs for production
   - Use pyttsx3 for development
   - Have fallback strategies

3. **Error Handling**
   - Always check return values
   - Provide user feedback
   - Log errors appropriately

4. **Performance**
   - Cache provider selection
   - Reuse connections
   - Handle timeouts gracefully