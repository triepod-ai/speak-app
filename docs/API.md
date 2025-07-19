# API Documentation

This document provides technical API documentation for developers who want to integrate with or extend the speak TTS system.

## Table of Contents

- [Python API](#python-api)
- [Shell Script API](#shell-script-api)
- [Provider Interface](#provider-interface)
- [Event System](#event-system)
- [Plugin Development](#plugin-development)
- [Testing API](#testing-api)

## Python API

### TTSProvider Class

The main class for TTS operations is located at `/home/bryan/bin/speak-app/tts/tts_provider.py`.

```python
from tts_provider import TTSProvider

# Initialize provider
tts = TTSProvider()
```

#### Methods

##### `__init__(self)`
Initialize the TTS provider with environment configuration.

```python
tts = TTSProvider()
```

##### `get_available_providers(self) -> List[str]`
Get list of available TTS providers based on API keys and system configuration.

```python
providers = tts.get_available_providers()
# Returns: ['elevenlabs', 'openai', 'pyttsx3']
```

##### `select_provider(self) -> Optional[str]`
Select the best available TTS provider based on configuration and availability.

```python
provider = tts.select_provider()
# Returns: 'elevenlabs' (or None if no providers available)
```

##### `speak(self, text: str, provider: Optional[str] = None) -> bool`
Speak text using the selected or specified provider.

```python
# Auto-select provider
success = tts.speak("Hello, world!")

# Use specific provider
success = tts.speak("Hello", provider="pyttsx3")
```

**Parameters:**
- `text` (str): Text to speak
- `provider` (str, optional): Specific provider to use

**Returns:**
- `bool`: True if successful, False otherwise

##### `speak_with_fallback(self, text: str) -> bool`
Attempt to speak text with automatic fallback to other providers.

```python
success = tts.speak_with_fallback("Important message")
```

### Direct Provider Usage

#### ElevenLabs

```python
from elevenlabs_tts import speak_with_elevenlabs

# Basic usage
success = speak_with_elevenlabs("Hello, world!")

# With voice selection
success = speak_with_elevenlabs(
    "Hello", 
    voice_id="21m00Tcm4TlvDq8ikWAM"  # Rachel
)
```

#### OpenAI

```python
from openai_tts import speak_with_openai

# Basic usage
success = speak_with_openai("Hello, world!")

# With voice selection
success = speak_with_openai(
    "Hello",
    voice="nova",
    model="tts-1-hd",
    speed=1.2
)
```

#### pyttsx3

```python
from pyttsx3_tts import speak_with_pyttsx3

# Basic usage
success = speak_with_pyttsx3("Hello, world!")

# With customization
success = speak_with_pyttsx3(
    "Hello",
    rate=200,
    volume=0.9,
    voice_index=0
)
```

### Integration Example

```python
#!/usr/bin/env python3
"""Example integration with speak TTS system."""

import sys
import os
from pathlib import Path

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent / "tts"))

from tts_provider import TTSProvider

class NotificationManager:
    """Manage notifications with TTS support."""
    
    def __init__(self):
        self.tts = TTSProvider()
        self.enabled = os.getenv("NOTIFICATIONS_ENABLED", "true").lower() == "true"
    
    def notify(self, message: str, priority: str = "normal") -> bool:
        """Send notification with appropriate priority."""
        if not self.enabled:
            return False
        
        # High priority uses best provider
        if priority == "high":
            return self.tts.speak(message, provider="elevenlabs") or \
                   self.tts.speak_with_fallback(message)
        
        # Normal priority uses auto-selection
        return self.tts.speak_with_fallback(message)
    
    def notify_async(self, message: str) -> None:
        """Send notification asynchronously."""
        import threading
        thread = threading.Thread(
            target=self.notify,
            args=(message,),
            daemon=True
        )
        thread.start()

# Usage
notifier = NotificationManager()
notifier.notify("Process started")
notifier.notify("Critical error!", priority="high")
notifier.notify_async("Background task complete")
```

## Shell Script API

### Command Synopsis

```bash
speak [OPTIONS] [TEXT]
echo "text" | speak [OPTIONS]
```

### Options

| Option | Short | Description | Type |
|--------|-------|-------------|------|
| `--provider` | `-p` | TTS provider | String |
| `--list` | `-l` | List providers | Flag |
| `--test` | `-t` | Test TTS | Flag |
| `--status` | `-s` | Show status | Flag |
| `--enable` | `-e` | Enable TTS | Flag |
| `--disable` | `-d` | Disable TTS | Flag |
| `--off` | `-o` | Skip this call | Flag |
| `--help` | `-h` | Show help | Flag |

### Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Provider not available |
| 4 | TTS disabled |
| 5 | Timeout |

### Environment Variables

```bash
# Core variables
TTS_ENABLED=true|false
TTS_PROVIDER=auto|elevenlabs|openai|pyttsx3
ENGINEER_NAME="Name"

# Provider keys
ELEVENLABS_API_KEY="..."
OPENAI_API_KEY="sk-..."

# Advanced
TTS_TIMEOUT=30
SPEAK_DEBUG=true|false
```

### Shell Functions

```bash
# Source this file for shell functions
source /home/bryan/bin/speak-app/speak_functions.sh

# Check if speak is available
if speak_available; then
    speak "TTS is available"
fi

# Speak with fallback
speak_or_echo "Message"

# Rate-limited speak
speak_rate_limited "Message" 5  # Max once per 5 seconds

# Priority-based speak
speak_priority "Critical error" "high"
speak_priority "Info message" "low"
```

## Provider Interface

### Creating a New Provider

All providers must implement this interface:

```python
#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "your-dependencies>=1.0.0",
# ]
# ///

"""
YourProvider TTS provider for speak command.
"""

import os
import sys
from typing import Optional

def speak_with_yourprovider(
    text: str,
    **kwargs
) -> bool:
    """
    Convert text to speech using YourProvider.
    
    Args:
        text: Text to convert to speech
        **kwargs: Provider-specific options
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # 1. Validate configuration
        api_key = os.getenv("YOURPROVIDER_API_KEY")
        if not api_key:
            print("YourProvider API key not set", file=sys.stderr)
            return False
        
        # 2. Prepare request
        # Your implementation here
        
        # 3. Generate audio
        audio_data = generate_audio(text, api_key, **kwargs)
        
        # 4. Play audio
        play_audio(audio_data)
        
        return True
        
    except Exception as e:
        print(f"YourProvider error: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: yourprovider_tts.py <text>", file=sys.stderr)
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    success = speak_with_yourprovider(text)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### Provider Requirements

1. **Error Handling**: Must not raise unhandled exceptions
2. **Timeout Support**: Respect TTS_TIMEOUT environment variable
3. **Logging**: Log errors to stderr, not stdout
4. **Exit Codes**: Return appropriate exit codes
5. **Dependencies**: Declare all dependencies in script header

### Provider Registration

Update `tts_provider.py`:

```python
# In get_available_providers()
if os.getenv("YOURPROVIDER_API_KEY"):
    providers.append("yourprovider")

# In script_map
"yourprovider": self.utils_dir / "yourprovider_tts.py",
```

## Event System

### Hooks (Future Feature)

```python
# Future hook system
class TTSHooks:
    """Hook system for TTS events."""
    
    def before_speak(self, text: str, provider: str) -> Optional[str]:
        """Called before speaking. Return modified text or None to cancel."""
        pass
    
    def after_speak(self, text: str, provider: str, success: bool) -> None:
        """Called after speaking attempt."""
        pass
    
    def on_error(self, text: str, provider: str, error: Exception) -> bool:
        """Called on error. Return True to retry with next provider."""
        pass
```

### Event Types

| Event | Description | Parameters |
|-------|-------------|------------|
| `before_speak` | Before TTS attempt | text, provider |
| `after_speak` | After TTS attempt | text, provider, success |
| `on_error` | On TTS error | text, provider, error |
| `provider_selected` | Provider chosen | provider, available |
| `fallback_activated` | Fallback triggered | failed_provider, next_provider |

## Plugin Development

### Plugin Structure

```
yourplugin/
├── __init__.py
├── provider.py      # Provider implementation
├── config.py        # Configuration handling
├── requirements.txt # Dependencies
└── README.md       # Documentation
```

### Plugin Interface

```python
# provider.py
from typing import Dict, Any, Optional

class YourPlugin:
    """Your TTS provider plugin."""
    
    # Required attributes
    name: str = "yourplugin"
    requires_api_key: bool = True
    supports_voices: bool = True
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self.config.get("api_key"))
    
    def get_voices(self) -> List[Dict[str, str]]:
        """Return available voices."""
        return [
            {"id": "voice1", "name": "Voice 1"},
            {"id": "voice2", "name": "Voice 2"},
        ]
    
    def speak(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Convert text to speech."""
        try:
            # Implementation
            return True
        except Exception as e:
            return False
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return provider metadata."""
        return {
            "version": "1.0.0",
            "author": "Your Name",
            "description": "Your provider description",
            "documentation": "https://...",
        }
```

## Testing API

### Unit Testing

```python
# test_tts_provider.py
import unittest
from unittest.mock import patch, MagicMock
from tts_provider import TTSProvider

class TestTTSProvider(unittest.TestCase):
    """Test TTS provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tts = TTSProvider()
    
    @patch.dict('os.environ', {'ELEVENLABS_API_KEY': 'test_key'})
    def test_provider_availability(self):
        """Test provider availability detection."""
        providers = self.tts.get_available_providers()
        self.assertIn('elevenlabs', providers)
        self.assertIn('pyttsx3', providers)
    
    @patch('subprocess.run')
    def test_speak_success(self, mock_run):
        """Test successful speech."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = self.tts.speak("Test", provider="pyttsx3")
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    def test_fallback_behavior(self):
        """Test provider fallback."""
        with patch.object(self.tts, 'speak') as mock_speak:
            mock_speak.side_effect = [False, True]  # First fails, second succeeds
            
            result = self.tts.speak_with_fallback("Test")
            self.assertTrue(result)
            self.assertEqual(mock_speak.call_count, 2)
```

### Integration Testing

```bash
#!/bin/bash
# test_integration.sh

# Test basic functionality
echo "Testing basic speech..."
if speak "Test message"; then
    echo "✓ Basic speech works"
else
    echo "✗ Basic speech failed"
    exit 1
fi

# Test provider selection
echo "Testing provider selection..."
for provider in elevenlabs openai pyttsx3; do
    if speak --provider "$provider" "Testing $provider" 2>/dev/null; then
        echo "✓ Provider $provider works"
    else
        echo "✗ Provider $provider failed"
    fi
done

# Test pipe input
echo "Testing pipe input..."
if echo "Pipe test" | speak; then
    echo "✓ Pipe input works"
else
    echo "✗ Pipe input failed"
fi

# Test error handling
echo "Testing error handling..."
if speak --provider nonexistent "Test" 2>/dev/null; then
    echo "✗ Should have failed with invalid provider"
else
    echo "✓ Error handling works"
fi
```

### Mocking for Tests

```python
# mock_tts.py
"""Mock TTS for testing."""

class MockTTSProvider:
    """Mock TTS provider for testing."""
    
    def __init__(self, always_succeed=True):
        self.always_succeed = always_succeed
        self.calls = []
    
    def speak(self, text, provider=None):
        """Mock speak method."""
        self.calls.append({
            'text': text,
            'provider': provider,
            'timestamp': time.time()
        })
        return self.always_succeed
    
    def get_call_count(self):
        """Get number of speak calls."""
        return len(self.calls)
    
    def reset(self):
        """Reset mock state."""
        self.calls = []
```

### Performance Testing

```python
# perf_test.py
import time
import statistics
from tts_provider import TTSProvider

def benchmark_provider(provider_name, iterations=10):
    """Benchmark a specific provider."""
    tts = TTSProvider()
    times = []
    
    for i in range(iterations):
        start = time.time()
        success = tts.speak(f"Test {i}", provider=provider_name)
        end = time.time()
        
        if success:
            times.append(end - start)
    
    if times:
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'success_rate': len(times) / iterations
        }
    return None

# Run benchmarks
for provider in ['elevenlabs', 'openai', 'pyttsx3']:
    print(f"\nBenchmarking {provider}:")
    results = benchmark_provider(provider)
    if results:
        print(f"  Mean: {results['mean']:.3f}s")
        print(f"  Median: {results['median']:.3f}s")
        print(f"  Success rate: {results['success_rate']:.1%}")
```