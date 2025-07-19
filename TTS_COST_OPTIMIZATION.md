# TTS Cost Optimization Strategy

## Current Cost Analysis

### ElevenLabs Pricing
- **Free Tier**: 10,000 characters/month
- **Starter**: $5/month for 30,000 characters
- **Creator**: $22/month for 100,000 characters
- **Cost**: ~$0.00033 per character at higher tiers

**Your Usage**: 10,000 credits burned = entire free tier exhausted!

### OpenAI TTS Pricing
- **Standard (tts-1)**: $15 per 1M characters ($0.000015/char)
- **HD (tts-1-hd)**: $30 per 1M characters ($0.000030/char)
- **22x cheaper than ElevenLabs!**

### Cost Comparison
For 10,000 characters:
- ElevenLabs: ~$3.30 (or entire free tier)
- OpenAI: ~$0.15 (standard) or $0.30 (HD)
- pyttsx3: FREE (offline)

## Immediate Solutions

### 1. Switch to OpenAI as Primary Provider
```bash
# In ~/.bash_aliases
export TTS_PROVIDER=openai
export OPENAI_API_KEY="your-openai-key"
```

### 2. Install and Use pyttsx3 for Development
```bash
# For Ubuntu/Debian (if system allows)
sudo apt-get install python3-espeak espeak
pip install --user pyttsx3

# Or use in virtual environment
python3 -m venv ~/speak-venv
source ~/speak-venv/bin/activate
pip install pyttsx3
```

### 3. Implement Smart Caching System

Create `tts/cache_manager.py`:
```python
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class TTSCache:
    def __init__(self, cache_dir="~/.cache/speak-app"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "phrase_cache.json"
        self.load_cache()
    
    def load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
    
    def get_cache_key(self, text, provider, voice=None):
        """Generate unique key for text+provider+voice combo"""
        key_parts = [text.lower().strip(), provider]
        if voice:
            key_parts.append(voice)
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def get_audio_path(self, text, provider, voice=None):
        """Check if we have cached audio for this text"""
        key = self.get_cache_key(text, provider, voice)
        if key in self.cache:
            audio_file = self.cache_dir / f"{key}.mp3"
            if audio_file.exists():
                # Update last accessed time
                self.cache[key]['last_accessed'] = datetime.now().isoformat()
                self.save_cache()
                return str(audio_file)
        return None
    
    def save_audio(self, text, provider, audio_data, voice=None):
        """Cache audio data for future use"""
        key = self.get_cache_key(text, provider, voice)
        audio_file = self.cache_dir / f"{key}.mp3"
        
        # Save audio file
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        
        # Update cache metadata
        self.cache[key] = {
            'text': text,
            'provider': provider,
            'voice': voice,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'file_size': len(audio_data)
        }
        self.save_cache()
        return str(audio_file)
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def cleanup_old_cache(self, days=30):
        """Remove cached files older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []
        
        for key, data in self.cache.items():
            last_accessed = datetime.fromisoformat(data['last_accessed'])
            if last_accessed < cutoff:
                audio_file = self.cache_dir / f"{key}.mp3"
                if audio_file.exists():
                    audio_file.unlink()
                to_remove.append(key)
        
        for key in to_remove:
            del self.cache[key]
        
        if to_remove:
            self.save_cache()
```

### 4. Implement Usage Tracking

Create `tts/usage_tracker.py`:
```python
import json
from datetime import datetime
from pathlib import Path

class UsageTracker:
    def __init__(self, data_dir="~/.local/share/speak-app"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.data_dir / "usage_stats.json"
        self.load_stats()
    
    def load_stats(self):
        if self.usage_file.exists():
            with open(self.usage_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                'providers': {},
                'daily_usage': {},
                'total_characters': 0,
                'total_cost_estimate': 0.0
            }
    
    def track_usage(self, provider, text, voice=None):
        """Track TTS usage for cost monitoring"""
        char_count = len(text)
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Update provider stats
        if provider not in self.stats['providers']:
            self.stats['providers'][provider] = {
                'total_characters': 0,
                'total_requests': 0,
                'voices': {}
            }
        
        self.stats['providers'][provider]['total_characters'] += char_count
        self.stats['providers'][provider]['total_requests'] += 1
        
        if voice:
            if voice not in self.stats['providers'][provider]['voices']:
                self.stats['providers'][provider]['voices'][voice] = 0
            self.stats['providers'][provider]['voices'][voice] += 1
        
        # Update daily usage
        if today not in self.stats['daily_usage']:
            self.stats['daily_usage'][today] = {}
        if provider not in self.stats['daily_usage'][today]:
            self.stats['daily_usage'][today][provider] = 0
        self.stats['daily_usage'][today][provider] += char_count
        
        # Update totals
        self.stats['total_characters'] += char_count
        
        # Estimate costs
        cost = self.estimate_cost(provider, char_count)
        self.stats['total_cost_estimate'] += cost
        
        self.save_stats()
        return cost
    
    def estimate_cost(self, provider, char_count):
        """Estimate cost based on provider pricing"""
        costs = {
            'elevenlabs': 0.00033,  # per character
            'openai': 0.000015,     # per character (standard)
            'pyttsx3': 0.0          # free
        }
        return char_count * costs.get(provider, 0.0)
    
    def get_monthly_report(self):
        """Generate monthly usage report"""
        # Implementation for monthly reporting
        pass
    
    def save_stats(self):
        with open(self.usage_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
```

### 5. Smart Provider Selection

Create `tts/smart_selector.py`:
```python
class SmartProviderSelector:
    def __init__(self, usage_tracker, cache_manager):
        self.usage_tracker = usage_tracker
        self.cache_manager = cache_manager
        self.thresholds = {
            'elevenlabs_daily_limit': 300,  # characters per day
            'prefer_cache': True,
            'development_mode': False
        }
    
    def select_provider(self, text, requested_provider=None):
        """Intelligently select TTS provider based on context"""
        
        # 1. Check cache first
        if self.thresholds['prefer_cache']:
            for provider in ['elevenlabs', 'openai', 'pyttsx3']:
                cached = self.cache_manager.get_audio_path(text, provider)
                if cached:
                    return provider, cached
        
        # 2. Development mode - always use free provider
        if self.thresholds['development_mode']:
            return 'pyttsx3', None
        
        # 3. Check daily limits
        today_usage = self.usage_tracker.stats['daily_usage'].get(
            datetime.now().strftime('%Y-%m-%d'), {}
        )
        
        # 4. Smart selection based on content
        if requested_provider:
            if requested_provider == 'elevenlabs':
                # Check if we're within daily limit
                used_today = today_usage.get('elevenlabs', 0)
                if used_today + len(text) > self.thresholds['elevenlabs_daily_limit']:
                    print(f"⚠️  ElevenLabs daily limit reached ({used_today} chars). Using OpenAI instead.")
                    return 'openai', None
            return requested_provider, None
        
        # 5. Content-based selection
        if len(text) < 50 and any(keyword in text.lower() for keyword in ['error', 'failed', 'warning']):
            # Short error messages - use cheap provider
            return 'pyttsx3', None
        elif len(text) > 500:
            # Long content - use OpenAI for cost efficiency
            return 'openai', None
        else:
            # Check ElevenLabs limit first
            used_today = today_usage.get('elevenlabs', 0)
            if used_today < self.thresholds['elevenlabs_daily_limit']:
                return 'elevenlabs', None
            else:
                return 'openai', None
```

## Implementation Plan

### Phase 1: Immediate Relief (Today)
1. Set `export TTS_PROVIDER=pyttsx3` for all development work
2. Install pyttsx3 locally for free offline TTS
3. Reserve ElevenLabs only for production/demos

### Phase 2: Smart Caching (This Week)
1. Implement cache manager to store frequently used phrases
2. Common developer messages: "Build complete", "Tests passed", etc.
3. Estimated savings: 70-80% reduction in API calls

### Phase 3: Usage Monitoring (This Week)
1. Track daily/monthly usage per provider
2. Set up alerts for approaching limits
3. Generate cost reports

### Phase 4: Intelligent Selection (Next Week)
1. Context-aware provider selection
2. Automatic fallback when limits approached
3. Development vs. production modes

## Quick Wins

### 1. Common Phrases Cache
Create a pre-cached set of common developer phrases:
```bash
# Pre-cache common messages
speak --cache-only "Build complete"
speak --cache-only "Build failed" 
speak --cache-only "Tests passed"
speak --cache-only "Tests failed"
speak --cache-only "Deployment successful"
speak --cache-only "Error detected"
```

### 2. Development Mode
Add to ~/.bash_aliases:
```bash
# Development mode - always use free TTS
alias speak-dev='TTS_PROVIDER=pyttsx3 speak'

# Production mode - use best quality
alias speak-prod='TTS_PROVIDER=elevenlabs speak'
```

### 3. Cost-Aware Usage
```bash
# Add daily limit checking
speak() {
    # Check daily usage before calling
    daily_used=$(cat ~/.local/share/speak-app/daily_usage.txt 2>/dev/null || echo 0)
    if [ $daily_used -gt 300 ]; then
        TTS_PROVIDER=openai command speak "$@"
    else
        command speak "$@"
    fi
}
```

## Estimated Savings

With this optimization strategy:
- **Current**: $3.30/day (10K chars on ElevenLabs)
- **Optimized**: ~$0.50/day
  - 300 chars ElevenLabs (premium voices): $0.10
  - 5,000 chars OpenAI (common messages): $0.08
  - 4,700 chars cached/pyttsx3: $0.00
- **Monthly savings**: ~$84 (from $99 to $15)

## Next Steps

1. Install pyttsx3 for immediate relief
2. Implement caching system
3. Add usage tracking
4. Deploy smart provider selection
5. Monitor and adjust thresholds based on actual usage