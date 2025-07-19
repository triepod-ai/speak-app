# Speak - Universal Text-to-Speech Command

<div align="center">

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/speak)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL-lightgrey.svg)]()

**A cost-optimized, multi-provider text-to-speech command-line tool with intelligent caching and batch processing**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## 🎯 Overview

`speak` is a universal text-to-speech (TTS) command that provides seamless voice synthesis across multiple providers. It intelligently selects the best available TTS engine based on your configuration and automatically falls back to alternatives when needed.

### Key Features

- 🎙️ **Multi-Provider Support**: OpenAI (default), ElevenLabs, and pyttsx3 (offline)
- 💰 **Cost Optimization**: 95% cost reduction with OpenAI as default, intelligent caching system
- 🔄 **Intelligent Fallback**: Automatic provider selection (OpenAI → ElevenLabs → pyttsx3)
- 📦 **Batch Processing**: Process multiple texts efficiently with `speak-batch`
- 💾 **Smart Caching**: Automatic caching of frequently used phrases
- 📊 **Usage Tracking**: Monitor costs and usage patterns
- 📝 **Flexible Input**: Direct text, piped input, or file reading
- ⚡ **Zero Latency**: Non-blocking execution for script integration
- 🔧 **Highly Configurable**: Environment variables and command-line options
- 🌐 **Global Access**: Available from any directory via PATH
- 🔌 **Script-Friendly**: Silent mode and conditional execution options
- 🤖 **Claude Code Integration**: Voice notifications for AI coding operations

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (for Python script execution)
- API keys for cloud providers (optional):
  - ElevenLabs API key for high-quality AI voices
  - OpenAI API key for alternative AI voices

### Verify Installation

The `speak` command is pre-installed at `/home/bryan/bin/speak-app/` with a symlink in PATH:

```bash
# Check if speak is available
which speak
# Output: /home/bryan/bin/speak

# Verify installation
speak --status
```

### Directory Structure

```
/home/bryan/bin/speak-app/
├── speak                # Main executable
├── tts/                 # TTS provider infrastructure
│   ├── tts_provider.py  # Provider selection logic
│   ├── elevenlabs_tts.py
│   ├── openai_tts.py
│   └── pyttsx3_tts.py
├── docs/                # Detailed documentation
├── examples/            # Usage examples
└── README.md           # This file
```

## 🚀 Quick Start

### Basic Usage

```bash
# Speak text directly
speak "Hello, world!"

# Pipe text to speak
echo "Build complete" | speak

# Read file aloud
cat README.md | speak

# Test TTS system
speak --test
```

### Configuration

Set up your environment variables:

```bash
# Enable/disable TTS
export TTS_ENABLED=true

# Set preferred provider
export TTS_PROVIDER=auto  # auto, elevenlabs, openai, pyttsx3

# API keys (for cloud providers)
export ELEVENLABS_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Personalization
export ENGINEER_NAME="Your Name"
```

## 💰 Cost Optimization

### Quick Setup for Maximum Savings

```bash
# Switch to OpenAI for 95% cost reduction
./set_openai_default.py

# Or manually set in ~/.bash_aliases
export TTS_PROVIDER=openai
export OPENAI_API_KEY="your-openai-key"
```

### Cost Comparison

| Provider | Cost per 10K chars | Monthly (300K chars) | Quality |
|----------|-------------------|---------------------|---------|
| **ElevenLabs** | $3.30 | $99.00 | ⭐⭐⭐⭐⭐ |
| **OpenAI** | $0.15 | $4.50 | ⭐⭐⭐⭐ |
| **pyttsx3** | $0.00 | $0.00 | ⭐⭐⭐ |

### Batch Processing

Generate multiple TTS files efficiently:

```bash
# Generate common developer notifications
speak-batch --common

# Process your own text file
speak-batch messages.txt --voice onyx

# Custom output directory
speak-batch messages.txt --output assets/audio
```

### Usage Monitoring

Track your TTS usage and costs:

```bash
# Cost analysis and recommendations
speak-costs

# Usage tracking report
python3 tts/usage_tracker.py --report

# Development mode (free offline TTS)
speak-dev "This saves money"
```

## 🛠️ New Tools

### `speak-batch` - Batch TTS Processing

Process multiple texts efficiently with automatic caching:

```bash
# Generate 30+ common notifications
speak-batch --common

# Process custom file
speak-batch messages.txt --voice echo --model tts-1-hd

# Show all options
speak-batch --help
```

### `speak-dev` - Development Mode

Force offline TTS to prevent accidental API usage:

```bash
# Always uses pyttsx3 (free)
speak-dev "Development message"

# Perfect for build scripts
make build && speak-dev "Build complete"
```

### `speak-costs` - Cost Analysis

Monitor and optimize TTS costs:

```bash
# Show cost breakdown and recommendations
speak-costs

# See potential savings
speak-costs --projection
```

### `speak-with-tracking` - Cost Tracking

Real-time cost display after each use:

```bash
# Shows cost per message
speak-with-tracking "Hello world"
# Output: [TTS: 11 chars, ~$0.00017 via openai]
```

## 📖 Documentation

### Command Options

| Option | Short | Description |
|--------|-------|-------------|
| `--provider PROVIDER` | `-p` | Use specific TTS provider |
| `--list` | `-l` | List available TTS providers |
| `--test` | `-t` | Test TTS with sample message |
| `--status` | `-s` | Show current TTS configuration |
| `--enable` | `-e` | Enable TTS for session |
| `--disable` | `-d` | Disable TTS for session |
| `--off` | `-o` | Skip TTS for this invocation |
| `--help` | `-h` | Show help message |

### Integration Guides

- [Shell Script Integration](docs/INTEGRATION.md) - Integrate with bash scripts and workflows
- [Claude Code Integration](docs/CLAUDE_CODE_INTEGRATION.md) - Voice notifications for Claude Code operations
- [API Documentation](docs/API.md) - Programmatic usage and Python integration
- [Provider Configuration](docs/PROVIDERS.md) - Setting up TTS providers

### Testing Documentation

- [Voice Testing Guide](docs/VOICE_TESTING_GUIDE.md) - Comprehensive guide to voice quality testing
- [Testing Priorities](docs/TESTING_PRIORITIES.md) - What to test first and why
- [Quick Test Summary](tests/VOICE_TEST_SUMMARY.md) - Quick reference for running tests

### Provider Comparison

| Provider | Requirements | Quality | Latency | Cost | Best For |
|----------|-------------|---------|---------|------|----------|
| **OpenAI** | API Key | ⭐⭐⭐⭐ | ~300ms | $0.015/1K chars | **Default** - Best value, 95% cheaper |
| **ElevenLabs** | API Key | ⭐⭐⭐⭐⭐ | ~500ms | $0.33/1K chars | Premium voices when quality matters most |
| **pyttsx3** | None | ⭐⭐⭐ | Instant | Free | Development/offline, Linux espeak fallback |

### Environment Variables

- `TTS_ENABLED`: Enable/disable TTS globally (true/false)
- `TTS_PROVIDER`: Default provider selection (auto/elevenlabs/openai/pyttsx3)
- `ENGINEER_NAME`: Name used in personalized messages
- `ELEVENLABS_API_KEY`: ElevenLabs API authentication
- `OPENAI_API_KEY`: OpenAI API authentication
- `PYTTSX3_VOICE_ID`: Specific voice for pyttsx3 (optional)
- `ESPEAK_RATE`: Speech rate for espeak fallback (default: 180)
- `ESPEAK_VOICE`: Voice for espeak fallback (default: en)

Configuration is automatically loaded from `~/brainpods/.env` if available.

See [pyttsx3 Configuration Guide](docs/PYTTSX3_CONFIG.md) for offline TTS options.

## 💡 Examples

### Development Workflow

```bash
# Build notifications
make build && speak "Build successful" || speak "Build failed"

# Test completion
npm test && speak "All tests passed"

# Git hooks
speak "Running pre-commit hooks"
```

### Claude Code Integration

The speak command integrates with Claude Code's Dynamic Hook System to provide voice notifications:

```bash
# Automatic notifications for:
- File operations: "Reading configuration file"
- Data storage: "Storing five kilobytes of data"
- Analysis: "Starting code analysis"
- Errors: "Connection failed"

# Control voice notifications
export VOICE_HOOKS=false  # Disable Claude Code voice
export VOICE_MIN_GAP=3.0  # Slower pacing
```

#### Get-Up-To-Speed Integration

The speak command is now integrated into all get-up-to-speed slash commands, providing meaningful audio feedback:

```bash
# Notifications you'll hear:
- "Loading project context for [project] from Redis cache"
- "Found [N] Redis keys for [project]"
- "Sub-agent completed: AI analysis for [project]"
- "Memory operation confirmed: [N] bytes saved to key [key]"
- "Context loading complete for [project]. Ready to work."

# Priority-based messages:
- Normal: "Developer, Loading project context"
- Error: "Developer, Error: Redis unavailable"
- Sub-agent: "Developer, Sub-agent completed: AI analysis"
- Memory: "Developer, Memory operation confirmed: 2KB saved"
```

See [Claude Code Integration Guide](docs/CLAUDE_CODE_INTEGRATION.md) for details.

### Script Integration

```bash
#!/bin/bash
# Long-running script with notifications

speak "Starting data processing"

for file in *.csv; do
    process_data "$file"
    speak --off "Processed $file"  # Silent during loop
done

speak "Processing complete. All files processed."
```

### Conditional Notifications

```bash
# Only speak on errors
if ! ./deploy.sh; then
    speak "Deployment failed! Check logs for details."
    exit 1
fi

# Different providers for different contexts
speak --provider elevenlabs "Production deployment complete"
speak --provider pyttsx3 "Debug: Variable value is $DEBUG_VAR"
```

### System Monitoring

```bash
# CPU usage alert
CPU_USAGE=$(top -b -n1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    speak "Warning: High CPU usage detected"
fi
```

## 🔧 Advanced Usage

### Programmatic Control

```bash
# Save and restore TTS state
OLD_TTS=$TTS_ENABLED
export TTS_ENABLED=false

# ... silent operations ...

export TTS_ENABLED=$OLD_TTS
speak "Operations complete"
```

### Error Handling

```bash
# Check if speak succeeded
if speak "Important message"; then
    echo "Message delivered"
else
    echo "TTS failed, using alternative notification"
    notify-send "Important message"
fi
```

### Custom Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias build-notify='make build && speak "Build complete" || speak "Build failed"'
alias deploy-notify='./deploy.sh && speak "Deployment successful" || speak "Deployment failed"'
```

## 🐛 Troubleshooting

### Common Issues

**No audio output:**
- Check TTS is enabled: `speak --status`
- Verify providers: `speak --list`
- Test offline provider: `speak --provider pyttsx3 "Test"`

**API key issues:**
- Verify keys are set: `env | grep -E "(ELEVENLABS|OPENAI)_API_KEY"`
- Check key validity with provider's API
- Use offline fallback: `export TTS_PROVIDER=pyttsx3`

**Performance issues:**
- Use `--off` flag in loops
- Consider offline provider for frequent calls
- Check network connectivity for cloud providers

### Debug Commands

```bash
# Test TTS system
python3 /home/bryan/bin/speak-app/tts/tts_provider.py "test"

# Check provider availability
speak --list

# Force specific provider
TTS_PROVIDER=pyttsx3 speak "Testing offline TTS"
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Adding new TTS providers
- Extending functionality
- Code style and testing
- Submitting pull requests

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for a detailed version history.

## 🚧 Roadmap

- [ ] Voice selection per provider
- [ ] Speech rate and volume control  
- [ ] Language support
- [ ] Audio file export
- [ ] Queue system for multiple messages
- [ ] Web API interface

---

<div align="center">

**Created by Bryan Thomas** • [Report Bug](https://github.com/yourusername/speak/issues) • [Request Feature](https://github.com/yourusername/speak/issues)

</div>