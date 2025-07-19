# Audio Test Guide for Speak App

This guide explains how to use the manual audio test suite to verify TTS functionality with actual audio output.

## Overview

The audio test suite allows you to:
- **Hear actual TTS output** from all providers
- **Compare voice quality** between providers
- **Test 4 different ElevenLabs voices** (Rachel, Domi, Bella, Adam)
- **Verify provider functionality** with real audio
- **Test various content types** (technical, emotional, conversational)

## Quick Start

### Basic Usage

```bash
# Run all audio tests
./run_audio_tests.sh

# Run specific test types
./run_audio_tests.sh -t elevenlabs    # Test ElevenLabs voices
./run_audio_tests.sh -t basic         # Basic functionality
./run_audio_tests.sh -t comparison    # Compare providers

# Run interactive tests
./run_audio_tests.sh -i

# Run specific test
./run_audio_tests.sh -s test_rachel_voice
```

### Using the Interactive Menu

Simply run `./run_audio_tests.sh` without arguments to see the menu:

```
=== Speak App Audio Test Menu ===

1. Run all audio tests
2. Test basic TTS functionality
3. Test ElevenLabs voices (Rachel, Domi, Bella, Adam)
4. Compare all providers side-by-side
5. Run interactive tests
6. Run specific test by name
7. Show available tests
8. Exit
```

## Test Categories

### 1. Basic Audio Tests (`test_manual_audio.py`)

Tests basic TTS functionality:
- **Provider Auto-Selection**: Tests automatic provider selection
- **Individual Providers**: Tests each provider separately
- **Content Types**: Technical content, numbers, symbols, emotions
- **Audio Quality**: Volume consistency, long content handling
- **Interactive Mode**: Custom text input

**Key Tests:**
- `test_basic_tts_provider` - Auto provider selection
- `test_all_providers_sequence` - All providers in sequence
- `test_technical_content` - Code and technical terms
- `test_emotional_content` - Different emotional contexts
- `test_interactive_mode` - Custom input testing

### 2. ElevenLabs Voice Tests (`test_manual_elevenlabs_voices.py`)

Tests 4 different ElevenLabs voices:

**Voices Tested:**
1. **Rachel** (21m00Tcm4TlvDq8ikWAM) - Calm narration voice
2. **Domi** (AZnzlk1XvdvUeBnXmlld) - Strong and confident
3. **Bella** (EXAVITQu4vr4xnSDxMaL) - Soft and gentle
4. **Adam** (pNInz6obpgDQGcFmaJgB) - Deep male voice

**Key Tests:**
- `test_rachel_voice` - Rachel's calm narration
- `test_domi_voice` - Domi's confident delivery
- `test_bella_voice` - Bella's gentle tone
- `test_adam_voice` - Adam's authoritative voice
- `test_voice_comparison` - All 4 voices with same text
- `test_voice_settings_variations` - Different stability/similarity settings
- `test_interactive_voice_selection` - Try any available voice

### 3. Provider Comparison Tests (`test_manual_provider_comparison.py`)

Side-by-side provider comparisons:
- **Quality Comparison**: Pronunciation and clarity
- **Naturalness Test**: Conversational flow
- **Emotion Handling**: How providers express emotions
- **Technical Content**: Handling of technical terms
- **Speed Variations**: Different speech rates
- **Cost-Benefit Analysis**: Provider characteristics

**Key Tests:**
- `test_basic_comparison` - Same text, all providers
- `test_quality_comparison` - Tongue twisters for clarity
- `test_naturalness_comparison` - Conversational speech
- `test_emotion_comparison` - Emotional expression
- `test_voice_variety_comparison` - Available voices per provider

## Running Specific Tests

### By Test Name

```bash
# Run a specific test function
./run_audio_tests.sh -s test_rachel_voice
./run_audio_tests.sh -s test_voice_comparison
./run_audio_tests.sh -s test_emotion_comparison
```

### By Category

```bash
# ElevenLabs voices only
./run_audio_tests.sh -t elevenlabs

# Provider comparisons only
./run_audio_tests.sh -t comparison

# Basic tests only
./run_audio_tests.sh -t basic
```

### Interactive Tests

```bash
# Run all interactive tests
./run_audio_tests.sh -i

# These tests allow you to:
# - Enter custom text
# - Choose specific voices
# - Test repeatedly with different inputs
```

## API Key Requirements

### Required Keys

- **ElevenLabs Tests**: Requires `ELEVENLABS_API_KEY`
- **OpenAI Tests**: Requires `OPENAI_API_KEY`
- **pyttsx3 Tests**: No API key needed (offline)

### Setting API Keys

```bash
# Set temporarily
export ELEVENLABS_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Or add to ~/.bashrc or ~/.bash_aliases
echo 'export ELEVENLABS_API_KEY="your_key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your_key"' >> ~/.bashrc
```

## Understanding Test Output

### Visual Indicators

- 🔊 **Audio Test** - Audio is about to play
- 🎤 **Testing Provider** - Specific provider test
- ✅ **Success** - Test passed
- ❌ **Failed** - Test failed
- ⚠️ **Warning** - Non-critical issue (e.g., missing API key)
- 🎭 **Voice/Emotion** - Testing specific voice or emotion
- 📊 **Performance** - Performance metrics
- ⏸️ **Pause** - Pause between tests

### Performance Metrics

Tests show latency measurements:
```
✅ ElevenLabs: Success (Latency: 0.52s)
✅ OpenAI: Success (Latency: 0.31s)
✅ pyttsx3: Success (Latency: 0.01s)
```

## Troubleshooting

### No Audio Output

1. Check speaker/headphone connection
2. Verify system volume is not muted
3. Check TTS is enabled: `speak --status`
4. Try offline provider: `speak --provider pyttsx3 "Test"`

### API Key Issues

```bash
# Check if keys are set
env | grep -E "(ELEVENLABS|OPENAI)_API_KEY"

# Test with offline provider
./run_audio_tests.sh -t basic -s test_pyttsx3_basic_audio
```

### Test Failures

- Provider tests skip if API key missing (this is normal)
- Network errors may cause cloud provider tests to fail
- Audio device errors require checking system audio settings

## Advanced Usage

### Custom Test Combinations

```bash
# Test only manual, non-interactive tests
pytest tests/test_manual_*.py -v -s -m "audio_output and manual and not interactive"

# Test only ElevenLabs with specific marker
pytest tests/test_manual_elevenlabs_voices.py -v -s -m "audio_output and elevenlabs"

# Run with coverage (though less relevant for audio tests)
pytest tests/test_manual_*.py -v -s -m audio_output --cov=tts
```

### Continuous Testing

```bash
# Run tests in a loop (useful for testing stability)
while true; do
    ./run_audio_tests.sh -t basic -s test_basic_tts_provider
    sleep 5
done
```

## Best Practices

1. **Volume Check**: Start with low volume, adjust as needed
2. **Quiet Environment**: Test in quiet space to hear quality differences
3. **Sequential Testing**: Let each test complete before starting next
4. **API Limits**: Be mindful of API usage limits for cloud providers
5. **Documentation**: Note any issues or quality observations

## Test Markers

Audio tests use these pytest markers:
- `@pytest.mark.audio_output` - All audio tests
- `@pytest.mark.manual` - Requires human verification
- `@pytest.mark.interactive` - Accepts user input
- `@pytest.mark.elevenlabs` - ElevenLabs-specific
- `@pytest.mark.openai` - OpenAI-specific
- `@pytest.mark.pyttsx3` - pyttsx3-specific
- `@pytest.mark.slow` - Longer running tests

## Contributing

When adding new audio tests:
1. Add `@pytest.mark.audio_output` marker
2. Include clear print statements about what's being tested
3. Add pauses between audio outputs
4. Provide context in test docstrings
5. Handle missing API keys gracefully