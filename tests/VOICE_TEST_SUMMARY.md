# Voice Test Summary

Quick reference for the speak-app voice testing framework.

## Test Files Overview

| Test File | Purpose | Providers Tested | Key Features |
|-----------|---------|-----------------|--------------|
| `test_manual_audio.py` | Core functionality | All | Basic TTS, content types, quality checks |
| `test_manual_elevenlabs_voices.py` | ElevenLabs voices | ElevenLabs only | 4 voices, settings, emotions |
| `test_manual_openai_voices.py` | OpenAI voices | OpenAI only | 6 voices, speed control, models |
| `test_manual_provider_comparison.py` | Provider comparison | All | Side-by-side quality, performance |

## Quick Test Commands

### Test Everything
```bash
# All audio tests
pytest tests/test_manual_*.py -v -s -m audio_output
```

### Test Specific Provider
```bash
# pyttsx3 only
pytest -k pyttsx3 -v -s -m audio_output

# OpenAI only
pytest -k openai -v -s -m audio_output

# ElevenLabs only
pytest -k elevenlabs -v -s -m audio_output
```

### Interactive Testing
```bash
# Try any text with any provider
pytest tests/test_manual_audio.py::TestManualAudio::test_interactive_mode -v -s

# Try ElevenLabs voices
pytest tests/test_manual_elevenlabs_voices.py::TestElevenLabsVoices::test_interactive_voice_selection -v -s

# Compare all providers
pytest tests/test_manual_provider_comparison.py::TestProviderComparison::test_interactive_comparison -v -s
```

## Test Categories

### 1. Functionality Tests
- Basic TTS operation
- Provider selection
- Error handling
- Fallback mechanisms

### 2. Content Tests
- Technical terminology (API, code, regex)
- Numbers and symbols ($, @, mathematical)
- Emotional expression
- Long-form content

### 3. Quality Tests
- Volume consistency
- Pronunciation clarity
- Speech naturalness
- Emotional delivery

### 4. Performance Tests
- Latency measurement
- Long content handling
- Provider comparison
- Cost-benefit analysis

## Voice Reference

### ElevenLabs Voices
| Voice | ID | Character |
|-------|-----|-----------|
| Rachel | 21m00Tcm4TlvDq8ikWAM | Calm narration |
| Domi | AZnzlk1XvdvUeBnXmlld | Strong & confident |
| Bella | EXAVITQu4vr4xnSDxMaL | Soft & gentle |
| Adam | pNInz6obpgDQGcFmaJgB | Deep male voice |

### OpenAI Voices
| Voice | Character |
|-------|-----------|
| alloy | Neutral and balanced |
| echo | Warm and engaging |
| fable | Expressive and dynamic |
| onyx | Deep and authoritative |
| nova | Natural and conversational |
| shimmer | Gentle and soothing |

## Key Test Scenarios

### Basic Smoke Test
```python
# Does TTS work at all?
test_basic_tts_provider()
```

### Quality Comparison
```python
# Which provider sounds best?
test_basic_comparison()
test_quality_comparison()
test_naturalness_comparison()
```

### Voice Selection
```python
# Test specific voices
test_rachel_voice()  # ElevenLabs
test_nova_voice()    # OpenAI
```

### Edge Cases
```python
# Technical content
test_technical_content()
test_numbers_and_symbols()

# Long content
test_long_content()
test_voice_endurance()
```

## Expected Outcomes

### pyttsx3
- ✅ Always works (offline)
- ⚠️ Robotic sound
- ⚡ Instant response

### OpenAI
- ✅ Good quality
- 💰 $0.015/1K chars
- 🎭 6 voice options

### ElevenLabs
- ✅ Best quality
- 💰 $0.33/1K chars
- 🎭 Many voices

## Common Test Patterns

### 1. Setup API Keys
```bash
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="..."
```

### 2. Run Quick Check
```bash
speak --test --provider pyttsx3
speak --test --provider openai
speak --test --provider elevenlabs
```

### 3. Run Full Tests
```bash
pytest tests/test_manual_audio.py -v -s
```

### 4. Listen and Evaluate
- Clarity
- Naturalness
- Consistency
- Appropriate emotion

## Debugging Tips

### No Audio?
1. Check volume: `amixer sset Master 75%`
2. Test system: `espeak "test"`
3. Check logs: Add `-v` to pytest

### API Errors?
1. Verify keys: `echo $OPENAI_API_KEY`
2. Check quota: API dashboard
3. Test manually: `speak --provider openai "test"`

### Quality Issues?
1. Try different voices
2. Adjust settings (stability, speed)
3. Compare providers side-by-side