# OpenAI TTS Setup Guide

## Quick Setup (Recommended)

### 1. Set OpenAI as Default Provider

Add to your `~/.bash_aliases` or `~/.bashrc`:

```bash
# Use OpenAI TTS as default (22x cheaper than ElevenLabs)
export TTS_PROVIDER=openai
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Set preferred voice (alloy, echo, fable, onyx, nova, shimmer)
export OPENAI_TTS_VOICE=onyx  # Default is nova

# For best quality/cost balance
export OPENAI_TTS_MODEL=tts-1  # Use tts-1-hd only when needed
```

### 2. Cost Comparison

| Use Case | Characters | ElevenLabs | OpenAI (tts-1) | Savings |
|----------|------------|------------|----------------|---------|
| "Build complete" | 14 | $0.0046 | $0.00021 | 95% |
| 100 notifications/day | ~5,000 | $1.65/day | $0.075/day | 95% |
| Your 10K chars/day | 10,000 | $3.30/day | $0.15/day | 95% |
| Monthly (300K chars) | 300,000 | $99/month | $4.50/month | 95% |

### 3. Voice Options

OpenAI offers 6 voices with different characteristics:

- **alloy**: Neutral, clear
- **echo**: Male, warm
- **fable**: British, storyteller
- **onyx**: Deep, authoritative (recommended for notifications)
- **nova**: Female, friendly (default)
- **shimmer**: Female, energetic

Test them:
```bash
speak --provider openai --voice alloy "Testing alloy voice"
speak --provider openai --voice onyx "Testing onyx voice"
```

### 4. Instant Cost Savings

```bash
# See your potential savings
speak-costs

# Use OpenAI for all speak commands
alias speak='TTS_PROVIDER=openai speak'

# Keep ElevenLabs only for special cases
alias speak-premium='TTS_PROVIDER=elevenlabs speak'
```

## Advanced Configuration

### Enable Caching (Save 70-80% more)

The cache manager can store frequently used phrases:

```bash
# Pre-cache common developer phrases
speak --cache "Build complete"
speak --cache "Tests passed"
speak --cache "Deployment successful"
speak --cache "Error detected"
```

### Usage Tracking

Track your usage to stay within budget:

```bash
# Check daily usage
speak --usage

# Set daily limits
export TTS_DAILY_LIMIT=10000  # Characters per day
```

### Smart Provider Selection

Enable automatic provider selection based on context:

```bash
# In ~/.bash_aliases
export TTS_SMART_MODE=true

# This will:
# - Use cache first (free)
# - Use pyttsx3 for errors/warnings (free)
# - Use OpenAI for important messages (cheap)
# - Reserve ElevenLabs for demos only (expensive)
```

## Migration Checklist

- [ ] Add OpenAI API key to ~/.bash_aliases
- [ ] Set TTS_PROVIDER=openai
- [ ] Test with: `speak "Hello from OpenAI"`
- [ ] Update any scripts using speak command
- [ ] Consider pre-caching common phrases
- [ ] Monitor usage for first week

## Cost Monitoring

```bash
# Check your estimated monthly costs
speak-costs

# View cache statistics
python3 tts/cache_manager.py --stats

# See provider usage breakdown
speak --stats
```

## Emergency Fallback

If OpenAI is down or you hit limits:

```bash
# Immediate fallback to free offline TTS
export TTS_PROVIDER=pyttsx3

# Or use development mode
speak-dev "This uses offline TTS"
```

## FAQ

**Q: Is OpenAI quality good enough?**
A: Yes! OpenAI's neural TTS is very high quality, especially with voices like "onyx" or "nova". It's nearly indistinguishable from ElevenLabs for short notifications.

**Q: What about rate limits?**
A: OpenAI TTS has generous rate limits. At 10K chars/day, you're using 0.3M chars/month - well within limits.

**Q: Can I still use ElevenLabs?**
A: Yes! Just use `speak --provider elevenlabs` or set up an alias for special cases.

**Q: What if I need emotion/style control?**
A: For advanced voice control, you can still use ElevenLabs selectively. For most notifications, OpenAI's voice variety is sufficient.

## Next Steps

1. **Today**: Switch to OpenAI (save 95% immediately)
2. **This Week**: Implement caching (save another 70-80%)
3. **This Month**: Add usage tracking and smart selection
4. **Long Term**: Consider Azure TTS for even lower costs at scale