# Configuration Guide

This guide covers all configuration options for the speak command, including environment variables, configuration files, and runtime options.

## Table of Contents

- [Configuration Hierarchy](#configuration-hierarchy)
- [Environment Variables](#environment-variables)
- [Configuration File](#configuration-file)
- [Command-Line Options](#command-line-options)
- [Provider-Specific Configuration](#provider-specific-configuration)
- [Advanced Configuration](#advanced-configuration)
- [Configuration Examples](#configuration-examples)

## Configuration Hierarchy

Configuration is loaded in the following order (later sources override earlier ones):

1. Default values (built into the application)
2. System-wide configuration (future feature)
3. User configuration file (`~/.speakrc` - future feature)
4. Environment variables from `~/brainpods/.env`
5. Environment variables from shell
6. Command-line options

## Environment Variables

### Core Settings

| Variable | Description | Default | Values |
|----------|-------------|---------|---------|
| `TTS_ENABLED` | Enable/disable TTS globally | `true` | `true`, `false` |
| `TTS_PROVIDER` | Default TTS provider | `auto` | `auto`, `elevenlabs`, `openai`, `pyttsx3` |
| `ENGINEER_NAME` | Name for personalized messages | System user | Any string |
| `TTS_TIMEOUT` | Timeout for TTS operations (seconds) | `30` | 1-300 |

### Provider API Keys

| Variable | Description | Required For |
|----------|-------------|--------------|
| `ELEVENLABS_API_KEY` | ElevenLabs API authentication | ElevenLabs provider |
| `OPENAI_API_KEY` | OpenAI API authentication | OpenAI provider |

### Setting Environment Variables

#### Temporary (Current Session)
```bash
export TTS_ENABLED=true
export TTS_PROVIDER=elevenlabs
export ELEVENLABS_API_KEY="your_key_here"
```

#### Permanent (Shell Configuration)
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export TTS_ENABLED=true' >> ~/.bashrc
echo 'export TTS_PROVIDER=elevenlabs' >> ~/.bashrc
echo 'export ELEVENLABS_API_KEY="your_key_here"' >> ~/.bashrc

# Reload configuration
source ~/.bashrc
```

#### Using .env File
Create or edit `~/brainpods/.env`:
```bash
TTS_ENABLED=true
TTS_PROVIDER=auto
ENGINEER_NAME="Bryan"
ELEVENLABS_API_KEY=your_elevenlabs_key_here
OPENAI_API_KEY=sk-your_openai_key_here
```

## Configuration File

**Note**: Configuration file support is planned for a future release.

Future `~/.speakrc` format:
```ini
[general]
enabled = true
provider = auto
timeout = 30

[personalization]
engineer_name = Bryan
default_voice = rachel

[providers]
elevenlabs_api_key = your_key_here
openai_api_key = sk-your_key_here

[voices]
elevenlabs_voice_id = 21m00Tcm4TlvDq8ikWAM
openai_voice = alloy
pyttsx3_rate = 175
```

## Command-Line Options

Command-line options override all other configuration:

### Provider Selection
```bash
# Use specific provider
speak --provider elevenlabs "Hello"
speak -p pyttsx3 "Offline TTS"

# Auto-selection
speak --provider auto "Best available provider"
```

### Control Options
```bash
# Temporarily disable TTS
speak --off "This won't speak"

# Enable for session
speak --enable

# Disable for session  
speak --disable
```

### Information Commands
```bash
# Show current configuration
speak --status

# List available providers
speak --list

# Test configuration
speak --test
```

## Provider-Specific Configuration

### ElevenLabs Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ELEVENLABS_API_KEY` | API authentication key | Required |
| `ELEVENLABS_VOICE_ID` | Default voice ID | Rachel (21m00Tcm4TlvDq8ikWAM) |
| `ELEVENLABS_MODEL` | Model version | eleven_monolingual_v1 |
| `ELEVENLABS_STABILITY` | Voice stability (0-1) | 0.5 |
| `ELEVENLABS_SIMILARITY` | Voice similarity (0-1) | 0.75 |

Example:
```bash
export ELEVENLABS_API_KEY="your_key"
export ELEVENLABS_VOICE_ID="EXAVITQu4vr4xnSDxMaL"  # Bella
export ELEVENLABS_STABILITY="0.7"
```

### OpenAI Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API authentication key | Required |
| `OPENAI_TTS_VOICE` | Voice selection | alloy |
| `OPENAI_TTS_MODEL` | Model selection | tts-1 |
| `OPENAI_TTS_SPEED` | Speech speed (0.25-4.0) | 1.0 |
| `OPENAI_TTS_FORMAT` | Audio format | mp3 |

Example:
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_TTS_VOICE="nova"
export OPENAI_TTS_MODEL="tts-1-hd"
export OPENAI_TTS_SPEED="1.2"
```

### pyttsx3 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTTSX3_VOICE` | System voice selection | System default |
| `PYTTSX3_RATE` | Speech rate (WPM) | 175 |
| `PYTTSX3_VOLUME` | Volume (0.0-1.0) | 1.0 |
| `PYTTSX3_ENGINE` | TTS engine | Auto-detect |

Example:
```bash
export PYTTSX3_RATE="200"
export PYTTSX3_VOLUME="0.8"
export PYTTSX3_VOICE="english-us"
```

## Advanced Configuration

### Logging and Debugging

| Variable | Description | Default |
|----------|-------------|---------|
| `SPEAK_DEBUG` | Enable debug output | false |
| `SPEAK_LOG_LEVEL` | Logging verbosity | info |
| `SPEAK_LOG_FILE` | Log file path | None |

```bash
# Enable debugging
export SPEAK_DEBUG=true
export SPEAK_LOG_LEVEL=debug
export SPEAK_LOG_FILE="$HOME/speak.log"
```

### Performance Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `SPEAK_CACHE_ENABLED` | Cache provider selection | true |
| `SPEAK_CACHE_TTL` | Cache duration (seconds) | 300 |
| `SPEAK_MAX_RETRIES` | Max retry attempts | 3 |
| `SPEAK_RETRY_DELAY` | Delay between retries (ms) | 1000 |

### Network Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HTTP_PROXY` | HTTP proxy server | None |
| `HTTPS_PROXY` | HTTPS proxy server | None |
| `NO_PROXY` | Bypass proxy for hosts | localhost |

## Configuration Examples

### Development Setup
```bash
# ~/brainpods/.env
TTS_ENABLED=true
TTS_PROVIDER=pyttsx3  # Use offline TTS
ENGINEER_NAME="Developer"
SPEAK_DEBUG=true
```

### Production Setup
```bash
# ~/brainpods/.env
TTS_ENABLED=true
TTS_PROVIDER=elevenlabs
ENGINEER_NAME="Bryan"
ELEVENLABS_API_KEY=your_production_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
SPEAK_CACHE_ENABLED=true
SPEAK_LOG_FILE="/var/log/speak.log"
```

### CI/CD Setup
```bash
# Disable TTS in CI
export TTS_ENABLED=false

# Or use offline TTS only
export TTS_ENABLED=true
export TTS_PROVIDER=pyttsx3
```

### Multi-Provider Setup
```bash
# ~/brainpods/.env
TTS_ENABLED=true
TTS_PROVIDER=auto

# Configure all providers
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=sk-your_openai_key

# Provider preferences
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
OPENAI_TTS_VOICE=nova
PYTTSX3_RATE=200
```

## Troubleshooting Configuration

### Verify Configuration
```bash
# Check all TTS-related variables
env | grep -E "(TTS|SPEAK|ELEVENLABS|OPENAI|PYTTSX3)"

# Test configuration
speak --status

# Test specific provider
speak --provider elevenlabs --test
```

### Common Issues

**TTS Not Working**
```bash
# Check if enabled
echo $TTS_ENABLED

# Should output "true"
# If not, enable it:
export TTS_ENABLED=true
```

**Provider Not Available**
```bash
# Check API key
echo $ELEVENLABS_API_KEY

# List available providers
speak --list
```

**Wrong Voice/Speed**
```bash
# Check provider-specific settings
env | grep ELEVENLABS
env | grep OPENAI
env | grep PYTTSX3
```

## Security Best Practices

1. **API Key Management**
   - Never commit API keys to version control
   - Use environment variables or secure key management
   - Rotate keys regularly
   - Use read-only keys when possible

2. **File Permissions**
   ```bash
   # Secure .env file
   chmod 600 ~/brainpods/.env
   ```

3. **Audit Configuration**
   ```bash
   # Check for exposed keys
   speak --status | grep -E "(API_KEY|configured)"
   ```

## Configuration Validation

The speak command validates configuration on startup:

1. **Environment Check**: Verifies required variables
2. **API Key Validation**: Checks key format (not validity)
3. **Provider Availability**: Confirms providers can be loaded
4. **Permission Check**: Ensures file access permissions

Invalid configuration results in:
- Warning messages for non-critical issues
- Fallback to default values
- Error messages for critical failures
- Graceful degradation to available providers