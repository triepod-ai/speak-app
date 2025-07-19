# Batch TTS Processing Guide

## Overview

The `speak-batch` command processes multiple text lines efficiently using OpenAI's TTS API. This is perfect for:
- Pre-generating common notifications
- Processing dialogue scripts
- Creating audio assets for applications
- Batch converting text files to audio

## Quick Start

### 1. Generate Common Developer Notifications

```bash
# This creates and processes 30+ common messages
speak-batch --common

# Output:
# ✅ 001_build_complete.mp3
# ✅ 002_build_failed.mp3
# ✅ 003_compilation_successful.mp3
# ... etc
```

### 2. Process Your Own Text File

Create a file `messages.txt`:
```
Welcome to the application
Your download is complete
New message received
System update available
```

Then process it:
```bash
speak-batch messages.txt
```

### 3. Use Different Voices

```bash
# Deep, authoritative voice (recommended)
speak-batch messages.txt --voice onyx

# British storyteller voice
speak-batch messages.txt --voice fable

# Friendly female voice
speak-batch messages.txt --voice nova
```

## Features

### 🚀 Cost Optimization
- **Automatic Caching**: Reuses previously generated audio
- **Batch Processing**: More efficient than individual calls
- **Usage Tracking**: Shows cost per item and total

### 📊 Output Organization
- Descriptive filenames: `001_build_complete.mp3`
- Manifest file with metadata
- Organized output directory

### 💾 Cache Integration
- Checks cache before generating
- Saves new audio to cache
- Shows cache savings in summary

## Example Session

```bash
$ speak-batch --common

🎯 Processing 30 lines from common_notifications.txt
📂 Output directory: tts_output
🎤 Voice: onyx, Model: tts-1
--------------------------------------------------

[1/30] Processing: Build complete...
  ✅ Using cached audio: tts_output/001_build_complete.mp3

[2/30] Processing: Build failed...
  ✅ Generated: tts_output/002_build_failed.mp3 ($0.0002)

[3/30] Processing: Compilation successful...
  ✅ Using cached audio: tts_output/003_compilation_successful.mp3

...

==================================================
📊 Batch Processing Summary
==================================================
Total processed: 30
  ✅ Generated: 12
  💾 From cache: 18
  ❌ Failed: 0

Total characters: 542
Generation cost: $0.0081
Cache savings: $0.0122
Net cost: -$0.0041

Average cost per item: $0.0007
```

## Cost Analysis

| Scenario | Items | Avg Length | Total Cost |
|----------|-------|------------|------------|
| Common notifications | 30 | 18 chars | $0.0081 |
| Daily status messages | 100 | 25 chars | $0.0375 |
| Game dialogue | 500 | 50 chars | $0.3750 |
| With 80% cache hit | 500 | 50 chars | $0.0750 |

## Advanced Usage

### Custom Output Directory
```bash
speak-batch messages.txt --output /path/to/audio/assets
```

### HD Quality (2x cost)
```bash
speak-batch messages.txt --model tts-1-hd
```

### Skip Manifest File
```bash
speak-batch messages.txt --no-manifest
```

### Process from Pipeline
```bash
cat error_messages.txt | speak-batch -
```

## Integration Examples

### In Build Scripts
```bash
#!/bin/bash
# Pre-generate all notification sounds
speak-batch notifications.txt --output assets/audio

# Use in application
play_notification() {
    aplay "assets/audio/$1.mp3"
}
```

### With CI/CD
```yaml
# GitHub Actions example
- name: Generate TTS Assets
  run: |
    speak-batch dialogue.txt --output public/audio
    cp tts_output/manifest.json public/audio/
```

### Python Integration
```python
import subprocess
import json

# Generate audio files
subprocess.run(['speak-batch', 'messages.txt'])

# Load manifest
with open('tts_output/manifest.json') as f:
    manifest = json.load(f)

# Use in application
for item in manifest['items']:
    audio_file = f"tts_output/{item['file']}"
    # Play audio_file when needed
```

## Tips for Maximum Savings

1. **Pre-generate Common Phrases**: Use `--common` to cache frequent messages
2. **Batch Similar Content**: Process related messages together
3. **Use Standard Model**: `tts-1` is sufficient for most notifications
4. **Enable Caching**: Automatic with our integration
5. **Monitor Usage**: Check `speak-costs` regularly

## Comparison with ElevenLabs

| Feature | ElevenLabs | OpenAI Batch |
|---------|------------|--------------|
| 100 notifications | $3.30 | $0.15 |
| With caching (80%) | $0.66 | $0.03 |
| Quality | Excellent | Very Good |
| Voices | Many | 6 options |
| Batch efficiency | No | Yes |

## Next Steps

1. Generate common notifications: `speak-batch --common`
2. Create your custom message list
3. Set up automated generation in your build process
4. Enjoy 95%+ cost savings!