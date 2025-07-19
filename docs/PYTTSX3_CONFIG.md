# pyttsx3 Configuration Guide

This guide explains how to configure the offline TTS provider (pyttsx3) in the speak-app.

## Overview

The pyttsx3 provider offers completely offline text-to-speech functionality. On Linux systems, it uses espeak as the underlying engine. If pyttsx3 initialization fails, the system automatically falls back to using espeak directly.

## Environment Variables

### pyttsx3 Configuration

- `PYTTSX3_VOICE_ID`: Specify a particular voice ID to use
  ```bash
  export PYTTSX3_VOICE_ID="english"  # Use specific voice
  ```

- `PYTTSX3_LIST_VOICES`: List available voices when set to "true"
  ```bash
  export PYTTSX3_LIST_VOICES=true
  speak --provider pyttsx3 "test"  # Will list voices to stderr
  ```

### espeak Fallback Configuration

When pyttsx3 fails to initialize (common on some Linux systems), the system falls back to espeak directly. You can configure espeak with these variables:

- `ESPEAK_RATE`: Speech rate in words per minute (default: 180)
  ```bash
  export ESPEAK_RATE=200  # Faster speech
  export ESPEAK_RATE=150  # Slower speech
  ```

- `ESPEAK_PITCH`: Voice pitch from 0-99 (default: 50)
  ```bash
  export ESPEAK_PITCH=30  # Lower pitch
  export ESPEAK_PITCH=70  # Higher pitch
  ```

- `ESPEAK_VOICE`: Voice/language to use (default: en)
  ```bash
  export ESPEAK_VOICE=en-us  # US English
  export ESPEAK_VOICE=en-gb  # British English
  export ESPEAK_VOICE=de     # German
  ```

## Usage Examples

### Basic Usage
```bash
# Use pyttsx3 provider explicitly
speak --provider pyttsx3 "Hello world"

# Set as default provider
export TTS_PROVIDER=pyttsx3
speak "This uses offline TTS"
```

### Voice Configuration
```bash
# List available voices
PYTTSX3_LIST_VOICES=true speak --provider pyttsx3 "test" 2>&1 | grep "ID:"

# Use specific voice
export PYTTSX3_VOICE_ID="english-us"
speak --provider pyttsx3 "Testing voice selection"
```

### espeak Fallback Customization
```bash
# Customize espeak fallback parameters
export ESPEAK_RATE=160
export ESPEAK_PITCH=40
export ESPEAK_VOICE=en-gb
speak --provider pyttsx3 "Testing with British accent"
```

## Troubleshooting

### Common Issues

1. **"SetVoiceByName failed" error**
   - This is automatically handled by falling back to espeak
   - You'll see: "pyttsx3 initialization failed, trying espeak fallback..."
   - This is normal and the TTS will still work

2. **No audio output**
   - Ensure espeak is installed: `sudo apt-get install espeak`
   - Check audio system is working: `espeak "test"`
   - Verify volume is not muted

3. **Voice not found**
   - List available voices with `espeak --voices`
   - Use a voice from the list with `ESPEAK_VOICE`

### Debug Mode
```bash
# See what's happening under the hood
PYTTSX3_LIST_VOICES=true speak --provider pyttsx3 --status
```

## Performance Tips

1. **Offline Usage**: pyttsx3 is perfect for development and offline environments
2. **Speed**: Instant response time (no network latency)
3. **Privacy**: All processing happens locally
4. **Cost**: Completely free

## Integration with Development Workflows

### Set as Development Default
```bash
# In ~/.bash_aliases
alias speak-dev='TTS_PROVIDER=pyttsx3 speak'

# Use in scripts
speak-dev "Build complete"
```

### CI/CD Integration
```bash
# Force offline TTS in CI environments
export TTS_PROVIDER=pyttsx3
export TTS_ENABLED=true

# Your build script
make build && speak "Build successful" || speak "Build failed"
```