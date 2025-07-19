# Voice Testing Guide

This guide documents the comprehensive voice testing framework for the speak-app, which is critical for ensuring audio quality and functionality across all TTS providers.

## Overview

The speak-app includes extensive manual audio tests that verify actual sound output, voice quality, and provider performance. These tests are designed to be listened to by humans to ensure the TTS system meets quality standards.

## Test Categories

### 1. Core Audio Tests (`test_manual_audio.py`)

The main test suite covering fundamental TTS functionality:

#### Basic Provider Tests
- **Auto-selection Test**: Verifies automatic provider selection works correctly
- **Individual Provider Tests**: Tests ElevenLabs, OpenAI, and pyttsx3 separately
- **Provider Sequence Test**: Tests all providers in sequence for comparison

#### Content-Specific Tests
- **Technical Content**: Tests pronunciation of code, APIs, regex patterns
- **Numbers and Symbols**: Verifies handling of prices, times, emails, math
- **Emotional Content**: Tests expressiveness with excitement, calm, questions
- **Long Content**: Verifies extended passages play without issues

#### Quality Tests
- **Volume Consistency**: Ensures consistent volume across providers
- **Speed Variations**: Tests different speech speeds (OpenAI only)
- **Interactive Mode**: Allows custom text input for testing

### 2. ElevenLabs Voice Tests (`test_manual_elevenlabs_voices.py`)

Comprehensive testing of ElevenLabs voice varieties:

#### Individual Voice Tests
- **Rachel** (21m00Tcm4TlvDq8ikWAM): Calm, composed narration voice
- **Domi** (AZnzlk1XvdvUeBnXmlld): Strong, confident commercial voice
- **Bella** (EXAVITQu4vr4xnSDxMaL): Soft, gentle audiobook voice
- **Adam** (pNInz6obpgDQGcFmaJgB): Deep, authoritative male voice

#### Voice Comparison Tests
- **Same Text Comparison**: All voices speak identical text
- **Voice Settings Variations**: Tests stability and similarity parameters
- **Emotional Delivery**: How different voices handle emotions
- **Technical Content**: Voice performance with technical terminology

#### Advanced Features
- **Voice Endurance**: Long content consistency testing
- **Voice Lookup**: Finding voices by name
- **Interactive Selection**: Try any available voice with custom text

### 3. OpenAI Voice Tests (`test_manual_openai_voices.py`)

Complete testing of all 6 OpenAI voices:

#### Voice Characteristics
- **Alloy**: Neutral and balanced
- **Echo**: Warm and engaging
- **Fable**: Expressive and dynamic
- **Onyx**: Deep and authoritative
- **Nova**: Natural and conversational (default)
- **Shimmer**: Gentle and soothing

#### Feature Tests
- **Voice Comparison**: All voices with same text
- **Speed Variations**: 0.75x, 1.0x, 1.25x speeds
- **Model Comparison**: tts-1 vs tts-1-hd quality
- **Interactive Mode**: Custom voice and text selection

### 4. Provider Comparison Tests (`test_manual_provider_comparison.py`)

Side-by-side comparison of all TTS providers:

#### Comparison Categories
- **Basic Comparison**: Same text across all providers
- **Quality Comparison**: Tongue twisters for pronunciation clarity
- **Naturalness**: Conversational speech evaluation
- **Emotion Expression**: How providers convey different emotions
- **Technical Content**: Handling of specialized terminology

#### Performance Metrics
- **Latency Measurement**: Time to generate and play audio
- **Voice Variety**: Available voice options per provider
- **Long Content**: Performance with extended passages
- **Cost-Benefit Analysis**: Quality vs. price comparison

## Running Voice Tests

### Prerequisites

1. **API Keys** (for cloud providers):
   ```bash
   export ELEVENLABS_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   ```

2. **Audio Output**: Ensure speakers/headphones are connected

3. **Dependencies**: Tests use uv for dependency management

### Test Execution

#### Run All Audio Tests
```bash
# Run all manual audio tests
pytest tests/test_manual_audio.py -v -s -m audio_output

# Run specific test
pytest tests/test_manual_audio.py::TestManualAudio::test_basic_tts_provider -v -s
```

#### Run Provider-Specific Tests
```bash
# ElevenLabs voices
pytest tests/test_manual_elevenlabs_voices.py -v -s -m audio_output

# OpenAI voices
pytest tests/test_manual_openai_voices.py -v -s -m audio_output

# Provider comparison
pytest tests/test_manual_provider_comparison.py -v -s -m audio_output
```

#### Run Interactive Tests
```bash
# Interactive mode for custom testing
pytest tests/test_manual_audio.py::TestManualAudio::test_interactive_mode -v -s
```

### Test Markers

- `@pytest.mark.audio_output`: Tests that produce actual audio
- `@pytest.mark.manual`: Tests requiring human verification
- `@pytest.mark.interactive`: Tests with user input
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.elevenlabs`: ElevenLabs-specific tests
- `@pytest.mark.openai`: OpenAI-specific tests
- `@pytest.mark.pyttsx3`: pyttsx3-specific tests

## Quality Criteria

### Audio Quality Checklist
- [ ] **Clarity**: Words are clearly pronounced
- [ ] **Naturalness**: Speech sounds human-like
- [ ] **Consistency**: Voice quality remains stable
- [ ] **Volume**: Audio level is appropriate
- [ ] **Pacing**: Speech speed is comfortable
- [ ] **Emotion**: Appropriate emotional expression

### Technical Requirements
- [ ] **Latency**: Response time is acceptable
- [ ] **Reliability**: No crashes or failures
- [ ] **Completeness**: Full text is spoken
- [ ] **Accuracy**: Technical terms pronounced correctly

## Test Output Interpretation

### Success Indicators
- ✅ Green checkmarks indicate successful tests
- 🔊 Speaker emoji indicates audio is playing
- Latency measurements show performance metrics

### Common Issues
- **"No API key"**: Set required environment variables
- **"No module named 'pyttsx3'"**: Install system dependencies
- **"SetVoiceByName failed"**: Normal on Linux, uses espeak fallback
- **Audio not playing**: Check system audio settings

## Provider Performance Summary

### pyttsx3 (Offline)
- **Pros**: Instant, free, privacy-focused
- **Cons**: Robotic voice, limited customization
- **Best for**: Development, CI/CD, offline use

### OpenAI
- **Pros**: Good quality, 6 voices, speed control
- **Cons**: Requires API key, costs money
- **Best for**: Production use, balanced quality/cost

### ElevenLabs
- **Pros**: Best quality, most natural, many voices
- **Cons**: Most expensive, requires API key
- **Best for**: Premium applications, audiobooks

## Testing Best Practices

1. **Listen Actively**: Pay attention to quality differences
2. **Test Regularly**: Run tests after any TTS changes
3. **Document Issues**: Note any quality problems
4. **Compare Providers**: Use comparison tests for decisions
5. **Test Edge Cases**: Try unusual text and symbols
6. **Verify Fallbacks**: Ensure offline fallback works

## Continuous Testing

### Automated Tests
While audio quality requires human verification, these automated tests ensure functionality:
- Unit tests for each provider
- Integration tests for provider selection
- Error handling tests
- Fallback mechanism tests

### Manual Verification Schedule
- **Weekly**: Basic provider tests
- **Monthly**: Full voice comparison
- **Release**: Complete test suite

## Troubleshooting Audio Tests

### No Audio Output
1. Check system volume
2. Verify audio device selection
3. Test with system TTS first: `espeak "test"`
4. Check provider logs for errors

### Poor Audio Quality
1. Verify API keys are valid
2. Check network connection
3. Try different voices
4. Adjust voice settings (stability, similarity)

### Test Failures
1. Check test markers are correct
2. Ensure dependencies installed
3. Verify provider scripts are executable
4. Check error messages in output

## Contributing to Voice Tests

When adding new voice tests:
1. Use appropriate test markers
2. Include setup/teardown pauses
3. Provide clear test descriptions
4. Add human verification prompts
5. Document expected outcomes
6. Consider provider-specific features

## Future Testing Plans

- Automated audio quality metrics
- A/B testing framework
- User preference studies
- Accessibility compliance tests
- Multi-language voice tests
- Performance benchmarking suite