# Speak App - Complete Command Reference

## 📋 Quick Reference

| Command | Purpose | Cost | Speed |
|---------|---------|------|-------|
| `speak` | Real-time TTS | Variable | Fast |
| `speak-batch` | Bulk TTS processing | Optimized | Efficient |
| `speak-dev` | Development mode | Free | Instant |
| `speak-costs` | Cost analysis | Free | Instant |
| `speak-with-tracking` | TTS with cost display | Variable | Fast |
| `set_openai_default.py` | Setup optimization | Free | Instant |

## 🎙️ Core Commands

### `speak` - Real-time Text-to-Speech

**Purpose**: Convert text to speech using intelligent provider selection

**Syntax**: `speak [OPTIONS] [TEXT]`

**Options**:
- `--provider PROVIDER` / `-p PROVIDER`: Use specific provider (openai, elevenlabs, pyttsx3)
- `--voice VOICE`: Select voice (provider-specific)
- `--list` / `-l`: List available providers
- `--test` / `-t`: Test TTS with sample message
- `--status` / `-s`: Show current configuration
- `--enable` / `-e`: Enable TTS for session
- `--disable` / `-d`: Disable TTS for session
- `--off` / `-o`: Skip TTS for this invocation
- `--help` / `-h`: Show help message

**Examples**:
```bash
# Basic usage
speak "Hello world"
speak --provider openai "Test message"

# From pipe
echo "Build complete" | speak
cat file.txt | speak

# Configuration
speak --status                 # Show current config
speak --list                   # List providers
speak --test                   # Test functionality
```

**Cost**: $0.000015 per character (OpenAI), $0.00033 per character (ElevenLabs), Free (pyttsx3)

---

### `speak-batch` - Batch TTS Processing

**Purpose**: Process multiple texts efficiently with automatic caching

**Syntax**: `speak-batch [OPTIONS] [INPUT_FILE]`

**Options**:
- `--voice VOICE`: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
- `--model MODEL`: Model to use (tts-1, tts-1-hd)
- `--output OUTPUT`: Output directory (default: tts_output)
- `--common`: Generate common developer notifications
- `--no-manifest`: Skip creating manifest.json

**Examples**:
```bash
# Generate common notifications
speak-batch --common

# Process custom file
speak-batch messages.txt --voice onyx

# Advanced options
speak-batch dialogue.txt --voice fable --model tts-1-hd --output assets/audio

# Different voices
speak-batch script.txt --voice echo
speak-batch script.txt --voice shimmer
```

**Input Format**: Text file with one message per line
```
Build complete
Tests passed
Deployment successful
Error detected
```

**Output**:
- Audio files: `001_build_complete.mp3`, `002_tests_passed.mp3`, etc.
- Manifest file: `manifest.json` with metadata
- Progress display with cost tracking

**Cost**: Batch-optimized, automatic caching, ~95% savings vs individual calls

---

### `speak-dev` - Development Mode

**Purpose**: Force offline TTS to prevent accidental API usage

**Syntax**: `speak-dev [TEXT]`

**Features**:
- Always uses pyttsx3 (offline)
- Prevents API key usage
- Shows development mode indicator
- Perfect for build scripts and testing

**Examples**:
```bash
# Development testing
speak-dev "Development message"

# Build script integration
make build && speak-dev "Build complete"

# Alias for safety
alias speak=speak-dev  # Force dev mode
```

**Cost**: Free (uses offline pyttsx3)

---

## 💰 Cost Management Commands

### `speak-costs` - Cost Analysis

**Purpose**: Analyze TTS costs and provide optimization recommendations

**Syntax**: `speak-costs [OPTIONS]`

**Features**:
- Real-time cost comparison
- Monthly projections
- Optimization recommendations
- Provider cost breakdown

**Example Output**:
```
🎙️  TTS Cost Analysis
==================================================

📊 Provider Costs (per 1000 characters):
  elevenlabs   $0.330
  openai       $0.015
  pyttsx3      $0.000

💰 Potential Savings with Optimization:
  Current (ElevenLabs): $99.00/month
  With OpenAI:          $4.50/month
  With Caching (80%):   $0.90/month
  Total Savings:        $98.10/month (99%!)
```

**Cost**: Free

---

### `speak-with-tracking` - Cost Tracking

**Purpose**: Real-time cost display after each TTS use

**Syntax**: `speak-with-tracking [TEXT]`

**Features**:
- Shows character count and cost
- Displays provider used
- Useful for cost awareness

**Example**:
```bash
speak-with-tracking "Hello world"
# Output: [TTS: 11 chars, ~$0.000165 via openai]
```

**Cost**: Same as underlying provider + tracking display

---

## 🔧 Setup and Configuration Commands

### `set_openai_default.py` - Quick Setup

**Purpose**: Interactive setup for OpenAI cost optimization

**Syntax**: `./set_openai_default.py`

**Features**:
- Adds OpenAI configuration to ~/.bash_aliases
- Shows cost comparison
- Provides setup instructions
- Checks for API key

**Interactive Flow**:
1. Shows cost benefits
2. Asks for confirmation
3. Updates configuration
4. Provides next steps

**Cost**: Free (setup only)

---

## 📊 Monitoring and Analysis Commands

### Usage Tracker

**Purpose**: Monitor TTS usage patterns and costs

**Syntax**: `python3 tts/usage_tracker.py [OPTIONS]`

**Options**:
- `--report`: Detailed usage report
- `--today`: Today's usage summary
- `--month`: Monthly usage summary

**Example Reports**:
```bash
# Detailed report
python3 tts/usage_tracker.py --report

# Quick stats
python3 tts/usage_tracker.py --today
# Output: Today: 1,234 characters

python3 tts/usage_tracker.py --month
# Output: This month: 45,678 characters ($0.68)
```

**Cost**: Free (analysis only)

---

### Cache Manager

**Purpose**: Manage TTS cache and statistics

**Syntax**: `python3 tts/cache_manager.py [OPTIONS]`

**Options**:
- `--stats`: Show cache statistics
- `--cleanup`: Remove old cache files
- `--clear`: Clear entire cache

**Example Output**:
```bash
python3 tts/cache_manager.py --stats

# Output:
{
  "total_entries": 150,
  "total_size_mb": 12.5,
  "total_hits": 450,
  "providers": {
    "openai": {
      "count": 120,
      "size": 8500000,
      "hits": 360
    }
  }
}
```

**Cost**: Free (management only)

---

## 🎯 Advanced Usage Patterns

### Voice Selection

**OpenAI Voices**:
```bash
speak-batch messages.txt --voice alloy     # Neutral, clear
speak-batch messages.txt --voice echo      # Male, warm
speak-batch messages.txt --voice fable     # British, storyteller
speak-batch messages.txt --voice onyx      # Deep, authoritative
speak-batch messages.txt --voice nova      # Female, friendly
speak-batch messages.txt --voice shimmer   # Female, energetic
```

**ElevenLabs Voices**:
```bash
speak --provider elevenlabs --voice Rachel "Premium voice"
speak --provider elevenlabs --voice Adam "Male voice"
speak --provider elevenlabs --voice Bella "Alternative voice"
```

### Model Selection

**OpenAI Models**:
```bash
speak-batch messages.txt --model tts-1      # Standard (cheaper)
speak-batch messages.txt --model tts-1-hd   # HD (2x cost)
```

### Integration Examples

**Build Scripts**:
```bash
#!/bin/bash
# Cost-optimized build notifications
npm run build && speak-dev "Build successful" || speak-dev "Build failed"

# Production notifications (with cost tracking)
npm run build && speak-with-tracking "Build complete"
```

**CI/CD Integration**:
```yaml
# GitHub Actions
- name: Generate Audio Assets
  run: |
    speak-batch notifications.txt --output public/audio
    speak-batch errors.txt --voice onyx --output public/audio/errors
```

**Python Integration**:
```python
import subprocess
import json

# Generate batch audio
result = subprocess.run(['speak-batch', 'messages.txt'], capture_output=True)

# Load manifest
with open('tts_output/manifest.json') as f:
    manifest = json.load(f)

# Process results
for item in manifest['items']:
    if item['cached']:
        print(f"Using cached: {item['file']}")
    else:
        print(f"Generated: {item['file']} (${item['cost']:.4f})")
```

## 🔍 Troubleshooting Commands

### Diagnostics

**Check Configuration**:
```bash
speak --status                 # Show current config
speak --list                   # List available providers
env | grep -E "(TTS|OPENAI|ELEVENLABS)"  # Check environment
```

**Test Providers**:
```bash
speak --provider openai "Test OpenAI"
speak --provider elevenlabs "Test ElevenLabs"
speak --provider pyttsx3 "Test offline"
```

**Cost Analysis**:
```bash
speak-costs                    # Show cost breakdown
python3 tts/usage_tracker.py --report  # Detailed usage
```

### Common Issues

**"API key not found"**:
```bash
# Check environment
echo $OPENAI_API_KEY
echo $ELEVENLABS_API_KEY

# Set keys
export OPENAI_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"
```

**"No audio output"**:
```bash
# Test audio system
speak --provider pyttsx3 "Test audio"

# Check audio devices
pactl list short sinks  # Linux
```

**"High costs"**:
```bash
# Switch to OpenAI
export TTS_PROVIDER=openai

# Use development mode
speak-dev "Cost-free testing"

# Check cache efficiency
python3 tts/cache_manager.py --stats
```

## 📈 Performance Tips

### Optimization Strategies

1. **Use OpenAI by default** - 95% cost reduction
2. **Enable caching** - Automatic with batch processing
3. **Use development mode** - Free testing with `speak-dev`
4. **Batch similar operations** - More efficient than individual calls
5. **Monitor usage** - Regular cost analysis

### Best Practices

1. **Start with `speak-batch --common`** - Pre-generate frequent phrases
2. **Use descriptive filenames** - Automatic with batch processing
3. **Monitor cache hit rates** - Aim for 70-80% hits
4. **Set appropriate voices** - `onyx` for notifications, `nova` for general use
5. **Use manifest files** - Easy integration with applications

### Cost Optimization Workflow

1. **Setup**: Run `./set_openai_default.py`
2. **Pre-generate**: Use `speak-batch --common`
3. **Monitor**: Check `speak-costs` weekly
4. **Optimize**: Use cache statistics to identify patterns
5. **Scale**: Batch process large operations

## 🚀 Integration Examples

### Shell Aliases

```bash
# Add to ~/.bash_aliases
alias speak-safe='speak-dev'                    # Force dev mode
alias speak-cost='speak-with-tracking'          # Always show costs
alias speak-openai='speak --provider openai'    # Force OpenAI
alias speak-fast='speak --provider pyttsx3'     # Force offline
```

### Environment Setup

```bash
# Optimal configuration in ~/.bash_aliases
export TTS_PROVIDER=openai
export OPENAI_API_KEY="your-key"
export OPENAI_TTS_VOICE=onyx
export OPENAI_TTS_MODEL=tts-1
export TTS_ENABLED=true
export ENGINEER_NAME="Your Name"
```

### Script Integration

```bash
# Build script with notifications
#!/bin/bash
set -e

echo "Starting build..."
npm run build

if [ $? -eq 0 ]; then
    speak-dev "Build successful"
    exit 0
else
    speak-dev "Build failed"
    exit 1
fi
```

This comprehensive command reference provides all the tools needed to effectively use the Speak app's cost-optimized TTS system.