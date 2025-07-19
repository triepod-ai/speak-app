# Testing Priorities for speak-app

This document outlines the testing priorities for the speak-app, emphasizing that **voice functionality is the top priority**.

## Testing Priority Levels

### 🔴 Priority 1: Core Voice Functionality (CRITICAL)

These tests verify the fundamental purpose of the application - producing audio output.

#### Must-Test Scenarios
1. **Basic TTS Output** - Can each provider produce audio?
   - `test_basic_tts_provider()` 
   - `test_pyttsx3_basic_audio()`
   - `test_openai_basic_audio()`
   - `test_elevenlabs_basic_audio()`

2. **Provider Fallback** - Does fallback work when primary fails?
   - `test_all_providers_sequence()`
   - pyttsx3 → espeak fallback on Linux

3. **Audio Quality** - Is the audio clear and understandable?
   - `test_quality_comparison()`
   - `test_volume_consistency()`

4. **Content Handling** - Can it speak different content types?
   - `test_technical_content()` - Code, APIs, technical terms
   - `test_numbers_and_symbols()` - Prices, emails, math

#### Test Command
```bash
# Priority 1 - Core voice tests only
pytest tests/test_manual_audio.py::TestManualAudio -v -s -k "basic or technical or numbers"
```

### 🟡 Priority 2: Voice Variety & Quality (IMPORTANT)

These tests ensure users can select appropriate voices for their use cases.

#### Voice Selection Tests
1. **Multiple Voices** - Do different voices work?
   - All ElevenLabs voices (Rachel, Domi, Bella, Adam)
   - All OpenAI voices (alloy, echo, fable, onyx, nova, shimmer)

2. **Voice Comparison** - Which voice is best for specific content?
   - `test_voice_comparison()` - Same text, different voices
   - `test_emotional_delivery()` - Emotion expression

3. **Voice Settings** - Can voices be customized?
   - `test_voice_settings_variations()` - Stability, similarity
   - `test_different_speeds()` - Speech rate control

#### Test Command
```bash
# Priority 2 - Voice variety tests
pytest tests/test_manual_elevenlabs_voices.py -v -s -m audio_output
pytest tests/test_manual_openai_voices.py -v -s -m audio_output
```

### 🟢 Priority 3: Performance & Comparison (VALUABLE)

These tests help users make informed decisions about provider selection.

#### Performance Tests
1. **Latency Comparison** - How fast is each provider?
   - `test_basic_comparison()` - Measures response times

2. **Cost-Benefit Analysis** - Which provider offers best value?
   - `test_cost_benefit_analysis()`
   - Quality vs. price comparison

3. **Long Content** - Performance with extended text
   - `test_long_content_comparison()`
   - `test_voice_endurance()`

#### Test Command
```bash
# Priority 3 - Performance comparison
pytest tests/test_manual_provider_comparison.py -v -s -m audio_output
```

### 🔵 Priority 4: Edge Cases & Special Features (NICE-TO-HAVE)

These tests cover less common but still important scenarios.

#### Special Cases
1. **Emotional Expression** - Natural speech patterns
   - `test_emotional_content()`
   - `test_emotion_comparison()`

2. **Interactive Testing** - Custom content verification
   - `test_interactive_mode()`
   - `test_interactive_voice_selection()`

3. **Advanced Features** - Speed control, HD models
   - `test_voices_with_different_speeds()`
   - `test_voices_with_models()`

## Quick Testing Guide

### Minimal Test Set (5 minutes)
```bash
# Test that each provider works
speak --test --provider pyttsx3
speak --test --provider openai  
speak --test --provider elevenlabs

# Test core functionality
pytest tests/test_manual_audio.py::TestManualAudio::test_basic_tts_provider -v -s
```

### Standard Test Set (15 minutes)
```bash
# Test all providers with various content
pytest tests/test_manual_audio.py -v -s -m audio_output

# Quick voice comparison
pytest tests/test_manual_provider_comparison.py::TestProviderComparison::test_basic_comparison -v -s
```

### Comprehensive Test Set (45 minutes)
```bash
# All voice tests
pytest tests/test_manual_*.py -v -s -m audio_output
```

## Testing Decision Tree

```
Is audio being produced?
├── NO → Fix Priority 1 issues first
│   └── Check: API keys, dependencies, audio system
└── YES → Is quality acceptable?
    ├── NO → Test Priority 2 (try different voices)
    └── YES → Is performance adequate?
        ├── NO → Test Priority 3 (compare providers)
        └── YES → Test Priority 4 (optimize for specific use cases)
```

## Key Testing Principles

1. **Audio First**: If it doesn't speak, nothing else matters
2. **Human Verification**: Audio quality requires human ears
3. **Real-World Content**: Test with actual use cases
4. **Provider Comparison**: Help users choose the right provider
5. **Fallback Verification**: Ensure system always works

## Test Frequency Recommendations

| Test Type | Frequency | When to Run |
|-----------|-----------|-------------|
| Basic TTS | Every change | Any code modification |
| Voice Quality | Weekly | Provider updates |
| Full Comparison | Monthly | Cost/quality reviews |
| Interactive | As needed | New use cases |

## Red Flags That Require Immediate Testing

- 🚨 No audio output from any provider
- 🚨 Consistent failures with specific content
- 🚨 API authentication errors
- 🚨 Fallback mechanism not working
- 🚨 Audio cutting off mid-sentence
- 🚨 Extreme latency (>5 seconds)

## Success Criteria

The speak-app is considered working when:
1. ✅ At least one provider produces audio
2. ✅ Audio is clear and understandable
3. ✅ Technical content is pronounced correctly
4. ✅ Fallback to offline TTS works
5. ✅ Response time is under 3 seconds