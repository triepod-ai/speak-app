# Speak Installation Checklist

Quick reference for installing and configuring the Speak TTS system.

## Pre-Installation Checklist

- [ ] Python 3.11+ installed (`python3 --version`)
- [ ] `uv` package manager installed (`uv --version`)
- [ ] OpenAI API key obtained from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- [ ] (Optional) ElevenLabs API key from [elevenlabs.io](https://elevenlabs.io)

## Installation Steps

### 1. Clone Repository
```bash
cd ~
git clone https://github.com/triepod-ai/speak-app.git
cd speak-app
```
- [ ] Repository cloned to `~/speak-app`
- [ ] Verified files present with `ls -la speak*`

### 2. Create ~/bin Directory
```bash
mkdir -p ~/bin
```
- [ ] Directory created/verified with `ls -ld ~/bin`

### 3. Create Symlinks
```bash
ln -sf ~/speak-app/speak ~/bin/speak
ln -sf ~/speak-app/speak-batch ~/bin/speak-batch
ln -sf ~/speak-app/speak-costs ~/bin/speak-costs
ln -sf ~/speak-app/speak-dev ~/bin/speak-dev
ln -sf ~/speak-app/speak-with-tracking ~/bin/speak-with-tracking
```
- [ ] Symlinks created
- [ ] Verified with `ls -la ~/bin/speak*`

### 4. Add ~/bin to PATH
```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bash_aliases
```
- [ ] PATH updated in `~/.bash_aliases`
- [ ] Verified with `grep 'bin:.*PATH' ~/.bash_aliases`

### 5. Configure Environment Variables

Add to `~/.bash_aliases`:
```bash
# Speak App TTS Configuration - Cost Optimized
export TTS_ENABLED=true
export TTS_PROVIDER=openai  # 22x cheaper than ElevenLabs
export OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"
export OPENAI_TTS_VOICE=onyx  # Best voice for notifications
export OPENAI_TTS_MODEL=tts-1  # Use tts-1-hd only when needed
export ELEVENLABS_API_KEY="sk_YOUR-KEY-HERE"  # Backup provider
export ENGINEER_NAME="Your Name"
```
- [ ] Configuration added to `~/.bash_aliases`
- [ ] API keys replaced with actual values
- [ ] Engineer name personalized

### 6. Reload Configuration
```bash
source ~/.bash_aliases
```
- [ ] Configuration reloaded
- [ ] New terminal opened (alternative)

## Verification Steps

### Check PATH
```bash
echo $PATH | grep "$HOME/bin"
which speak
```
- [ ] `~/bin` appears in PATH
- [ ] `which speak` shows `/home/username/bin/speak`

### Check Environment Variables
```bash
env | grep -E "^(TTS_|OPENAI_|ELEVENLABS_|ENGINEER_)"
```
- [ ] `TTS_ENABLED=true`
- [ ] `TTS_PROVIDER=openai`
- [ ] `OPENAI_API_KEY` is set
- [ ] `OPENAI_TTS_VOICE=onyx`
- [ ] `ENGINEER_NAME` is set

### Test Commands
```bash
speak --status
speak "Installation complete"
speak --test
speak-costs
```
- [ ] `speak --status` shows configuration
- [ ] Voice output heard on `speak "Installation complete"`
- [ ] `speak --test` plays test message
- [ ] `speak-costs` displays cost analysis

### Test Providers
```bash
speak --provider openai "Testing OpenAI"
speak --provider pyttsx3 "Testing offline"
```
- [ ] OpenAI provider works
- [ ] Offline provider (pyttsx3) works

## Troubleshooting Checklist

### If "command not found"
- [ ] Symlinks exist: `ls -la ~/bin/speak`
- [ ] PATH includes ~/bin: `echo $PATH`
- [ ] Configuration reloaded: `source ~/.bash_aliases`
- [ ] New terminal opened

### If "API key not found"
- [ ] Environment variable set: `echo $OPENAI_API_KEY`
- [ ] Key format correct (starts with `sk-proj-` or `sk-`)
- [ ] Configuration reloaded
- [ ] Using correct file (`~/.bash_aliases` not `~/.bashrc`)

### If no audio output
- [ ] Audio device working: `speaker-test -t wav -c 2`
- [ ] Provider status checked: `speak --status`
- [ ] Offline fallback tested: `speak-dev "test"`
- [ ] Audio dependencies installed (Linux): `sudo apt-get install espeak ffmpeg`

## Post-Installation

### Configuration Options
- [ ] Read [SETUP_OPENAI.md](SETUP_OPENAI.md) for voice options
- [ ] Read [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for advanced settings
- [ ] Review cost analysis: `speak-costs`

### Integration
- [ ] Set up Claude Code hooks (see [docs/CLAUDE_CODE_INTEGRATION.md](docs/CLAUDE_CODE_INTEGRATION.md))
- [ ] Configure script integration (see [docs/INTEGRATION.md](docs/INTEGRATION.md))
- [ ] Review usage examples (see [examples/](examples/))

### Testing
- [ ] Run voice tests: `pytest tests/test_manual_audio.py -v -s`
- [ ] Test batch processing: `speak-batch --common`
- [ ] Monitor usage: `speak-with-tracking "test message"`

## Quick Reference

### Essential Commands
| Command | Purpose |
|---------|---------|
| `speak "text"` | Speak text using default provider |
| `speak --status` | Show configuration and available providers |
| `speak --test` | Run built-in test |
| `speak-dev "text"` | Use offline TTS (free) |
| `speak-costs` | Show cost analysis |
| `speak-batch file.txt` | Process file in batch |

### Environment Variables (Essential)
| Variable | Value | Purpose |
|----------|-------|---------|
| `TTS_ENABLED` | `true` | Enable TTS globally |
| `TTS_PROVIDER` | `openai` | Default provider (cheapest) |
| `OPENAI_API_KEY` | `sk-proj-...` | OpenAI authentication |
| `OPENAI_TTS_VOICE` | `onyx` | Voice selection |
| `ENGINEER_NAME` | Your name | Personalization |

### File Locations
| File | Purpose |
|------|---------|
| `~/speak-app/` | Source repository |
| `~/bin/speak*` | Symlinked executables |
| `~/.bash_aliases` | Configuration and PATH |
| `~/brainpods/.env` | Alternative config location |

## Success Criteria

Installation is complete when:
- [x] `which speak` shows the command path
- [x] `speak --status` shows correct configuration
- [x] `speak "test"` produces audio output
- [x] All environment variables are set correctly
- [x] Cost analysis runs: `speak-costs`

## Next Steps

1. **Test voice options**: Try different OpenAI voices (alloy, echo, fable, onyx, nova, shimmer)
2. **Batch process common phrases**: Run `speak-batch --common` to cache frequently used messages
3. **Monitor costs**: Use `speak-with-tracking` for a week to understand usage patterns
4. **Set up integrations**: Configure Claude Code hooks or script integrations
5. **Read documentation**: Review [docs/](docs/) for advanced features

## Support

- **Full Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Configuration Guide**: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- **Cost Optimization**: [SETUP_OPENAI.md](SETUP_OPENAI.md)
- **Project Documentation**: [README.md](README.md)

---

**Installation Date**: _______________

**Installed By**: _______________

**Notes**:
```
[Space for installation notes, custom configurations, or issues encountered]
```
