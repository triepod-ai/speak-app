# TTS Integration System Documentation

**Version**: 1.0  
**Date**: 2025-07-17  
**Status**: ✅ Production Ready  

## Overview

The TTS (Text-to-Speech) integration system provides intelligent audio notifications for Claude Code hooks with robust fallback mechanisms and graceful degradation. This system enhances user experience by providing audio feedback for important operations like conversation compacting, session exports, and workflow automation.

## Quick Start

### 1. Installation
```bash
# Install system dependencies
sudo apt update && sudo apt install -y python3-requests python3-pygame python3-dotenv python3-pyttsx3

# Verify installation
python3 -c "import requests, pygame, dotenv; print('✅ Dependencies installed')"
```

### 2. Configuration
```bash
# Edit the TTS configuration file
nano /home/bryan/brainpods/.env

# Add your API key (optional - system will fallback to offline TTS)
ELEVENLABS_API_KEY=your_api_key_here
TTS_ENABLED=true
ENGINEER_NAME=YourName
```

### 3. Test TTS
```bash
# Test the TTS system
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Hello! TTS is working!"
```

### 4. Enable Hook Integration
```bash
# The before-compact hook is already configured
# Test it with:
/compact
```

### 5. View Logs
```bash
# Check TTS operation logs
tail -f /tmp/claude-before-compact-hook.log
```

That's it! Your TTS integration is ready to use.

## Architecture

### Core Components

1. **TTS Provider System** (`/home/bryan/brainpods/.claude/hooks/utils/tts/`)
   - `tts_provider.py` - Main provider selection and orchestration
   - `elevenlabs_tts.py` - ElevenLabs API integration
   - `pyttsx3_tts.py` - Offline TTS fallback
   - `openai_tts.py` - OpenAI TTS provider (optional)

2. **Hook Integration** (`/home/bryan/.claude/hooks/`)
   - `before-compact-export.sh` - Before-compact hook with TTS notifications
   - `settings.json` - Hook configuration and TTS integration

3. **Configuration Management** (`/home/bryan/brainpods/.env`)
   - API keys and provider preferences
   - TTS behavior settings
   - Personalization options

## Provider Selection Logic

### Priority Chain
1. **ElevenLabs** (Primary) - High-quality AI voice synthesis
2. **pyttsx3** (Secondary) - Offline fallback requiring espeak
3. **System TTS** (Tertiary) - espeak, say, spd-say
4. **Graceful Skip** (Final) - Continue without audio

### Selection Algorithm
```python
def select_provider(self) -> Optional[str]:
    """Select best available TTS provider based on API keys and system capabilities"""
    
    # Check API key availability
    if os.getenv("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    
    # Fallback to offline TTS
    return "pyttsx3"
```

## Dependency Management

### System Requirements
```bash
# Required system packages
sudo apt update && sudo apt install -y \
    python3-requests \
    python3-pygame \
    python3-dotenv \
    python3-pyttsx3 \
    espeak-ng
```

### Verification Commands
```bash
# Test system packages
python3 -c "import requests, pygame, dotenv; print('✅ System packages available')"

# Test TTS providers
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Test message"
```

## Configuration

### Environment Variables (`/home/bryan/brainpods/.env`)
```bash
# ElevenLabs Configuration
ELEVENLABS_API_KEY=sk_your_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice (default)

# TTS Configuration
TTS_ENABLED=true
TTS_PROVIDER=auto  # auto, elevenlabs, openai, pyttsx3
ENGINEER_NAME=Bryan  # Personalization

# Optional: OpenAI Configuration
OPENAI_API_KEY=your_openai_key_here
```

### Hook Configuration (`/home/bryan/.claude/settings.json`)
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task(.*compact.*)",
        "hooks": [
          {
            "type": "command",
            "command": "/home/bryan/.claude/hooks/before-compact-export.sh"
          }
        ]
      }
    ]
  }
}
```

## Implementation Patterns

### Non-Blocking Execution
```bash
# ✅ Correct: Non-blocking with timeout
if timeout 15 python3 "$TTS_PROVIDER_SCRIPT" "$TTS_MESSAGE" >/dev/null 2>&1; then
    log_message "TTS notification sent: $TTS_MESSAGE"
else
    log_message "TTS failed, continuing without audio"
fi
```

### Subprocess Execution
```python
# ✅ Reliable: System python executable
result = subprocess.run(
    [sys.executable, str(script_path), text],
    capture_output=True,
    text=True,
    timeout=30
)

# ❌ Problematic: uv run in subprocess
result = subprocess.run(
    ["uv", "run", str(script_path), text],
    capture_output=True,
    text=True,
    timeout=30
)
```

### Error Handling
```python
def speak_with_fallback(self, text: str) -> bool:
    """Attempt TTS with automatic fallback through provider chain"""
    
    for provider in self.get_available_providers():
        try:
            if self.speak(text, provider):
                return True
        except Exception as e:
            logger.warning(f"TTS provider {provider} failed: {e}")
            continue
    
    return False  # All providers failed
```

## Message Templates

### Hook-Specific Messages
```python
HOOK_MESSAGES = {
    "before_compact": "Before compact hook completed for project {project}. Conversation exported and ready for storage.",
    "post_edit": "File modification completed in {project}. Changes have been saved.",
    "session_end": "Session completed for {project}. Context has been automatically exported.",
    "mcp_operation": "MCP operation completed in {project}. Data has been processed."
}
```

### Personalized Notifications
```python
def create_personalized_message(hook_type: str, project: str, engineer: str) -> str:
    """Create personalized TTS message with engineer name"""
    
    base_message = HOOK_MESSAGES.get(hook_type, "Operation completed")
    return f"{engineer}, {base_message.format(project=project)}"
```

## Observability & Logging

### Log Structure
```bash
# TTS operation logs
[2025-07-17 11:37:52] TTS notification sent: Before compact hook completed for project tts
[2025-07-17 11:37:53] TTS provider elevenlabs succeeded in 1.2s
[2025-07-17 11:37:54] TTS fallback to pyttsx3 due to API rate limit

# Error logs
[2025-07-17 11:38:01] TTS provider elevenlabs failed: API rate limit exceeded
[2025-07-17 11:38:02] TTS provider pyttsx3 failed: espeak not installed
[2025-07-17 11:38:03] TTS notification skipped: no providers available
```

### JSON Logging (Optional)
```python
def log_tts_operation(provider: str, success: bool, duration: float, message: str):
    """Log TTS operation with structured data"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "success": success,
        "duration_seconds": duration,
        "message_length": len(message),
        "project": os.path.basename(os.getcwd())
    }
    
    logger.info(json.dumps(log_entry))
```

## Usage Examples

### Basic Usage

**1. Direct TTS Provider Usage**
```bash
# Test ElevenLabs provider directly
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/elevenlabs_tts.py "Hello from ElevenLabs!"

# Test pyttsx3 offline provider
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/pyttsx3_tts.py "Hello from offline TTS!"

# Use main provider with automatic selection
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "This will use the best available provider"
```

**2. Hook Integration Usage**
```bash
# Trigger the before-compact hook (automatic)
/compact

# Test the hook manually
/home/bryan/.claude/hooks/test-hook.sh

# Check TTS logs
tail -f /tmp/claude-before-compact-hook.log
```

**3. Environment Configuration**
```bash
# Enable TTS
export TTS_ENABLED=true

# Set specific provider
export TTS_PROVIDER=elevenlabs

# Set engineer name for personalization
export ENGINEER_NAME="Bryan"

# Test with new settings
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Configuration test"
```

### Advanced Usage Examples

**1. Custom Hook with TTS**
```bash
#!/bin/bash
# /home/bryan/.claude/hooks/custom-notification-hook.sh

PROJECT_NAME=$(basename "$PWD")
TTS_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"

# Custom operation
perform_git_commit() {
    git add .
    git commit -m "Automated commit via hook"
    
    # TTS notification
    if [ -f "$TTS_SCRIPT" ]; then
        timeout 10 python3 "$TTS_SCRIPT" "Git commit completed for $PROJECT_NAME" &
    fi
}

perform_git_commit
```

**2. Conditional TTS Based on Operation Type**
```bash
#!/bin/bash
# Hook with conditional TTS

OPERATION_TYPE="$1"
PROJECT_NAME=$(basename "$PWD")

case "$OPERATION_TYPE" in
    "critical")
        MESSAGE="Critical operation completed in $PROJECT_NAME. Please review immediately."
        ;;
    "warning")
        MESSAGE="Warning: Operation completed in $PROJECT_NAME with potential issues."
        ;;
    "success")
        MESSAGE="Operation successfully completed in $PROJECT_NAME."
        ;;
    *)
        MESSAGE="Operation completed in $PROJECT_NAME."
        ;;
esac

# Send TTS notification
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "$MESSAGE"
```

**3. Personalized Notifications**
```bash
#!/bin/bash
# Personalized TTS notifications

ENGINEER_NAME=${ENGINEER_NAME:-"Developer"}
PROJECT_NAME=$(basename "$PWD")
TIMESTAMP=$(date '+%H:%M')

# Create personalized message
create_message() {
    local operation="$1"
    echo "$ENGINEER_NAME, $operation completed in $PROJECT_NAME at $TIMESTAMP."
}

# Usage examples
notify_file_saved() {
    MESSAGE=$(create_message "file save")
    python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "$MESSAGE"
}

notify_build_complete() {
    MESSAGE=$(create_message "build process")
    python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "$MESSAGE"
}

notify_tests_passed() {
    MESSAGE=$(create_message "test suite")
    python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "$MESSAGE"
}
```

### Real-World Integration Examples

**1. CI/CD Pipeline Integration**
```yaml
# .github/workflows/tts-notifications.yml
name: TTS Notifications
on: [push, pull_request]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Setup TTS
        run: |
          sudo apt install -y python3-requests python3-pygame python3-dotenv
          
      - name: Notify Build Start
        run: |
          python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Build started for ${{ github.repository }}"
          
      - name: Run Tests
        run: |
          # Your test commands here
          
      - name: Notify Build Complete
        run: |
          python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Build completed for ${{ github.repository }}"
```

**2. Development Workflow Integration**
```bash
#!/bin/bash
# dev-workflow.sh - Development workflow with TTS notifications

TTS_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"

notify() {
    local message="$1"
    if [ -f "$TTS_SCRIPT" ]; then
        timeout 10 python3 "$TTS_SCRIPT" "$message" >/dev/null 2>&1 &
    fi
}

# Development workflow
echo "🚀 Starting development workflow..."
notify "Development workflow started"

# Run tests
echo "Running tests..."
if npm test; then
    notify "Tests passed successfully"
else
    notify "Warning: Tests failed, please review"
fi

# Build project
echo "Building project..."
if npm run build; then
    notify "Build completed successfully"
else
    notify "Error: Build failed, please check logs"
fi

# Deploy
echo "Deploying..."
if npm run deploy; then
    notify "Deployment completed successfully"
else
    notify "Error: Deployment failed, please investigate"
fi

notify "Development workflow completed"
```

**3. Multi-Project Monitoring**
```bash
#!/bin/bash
# multi-project-monitor.sh

PROJECTS=("project1" "project2" "project3")
TTS_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"

monitor_project() {
    local project_path="$1"
    local project_name=$(basename "$project_path")
    
    cd "$project_path"
    
    # Check git status
    if git status --porcelain | grep -q .; then
        python3 "$TTS_SCRIPT" "Uncommitted changes detected in $project_name"
    fi
    
    # Check for running processes
    if pgrep -f "$project_name" >/dev/null; then
        python3 "$TTS_SCRIPT" "$project_name is currently running"
    fi
}

# Monitor all projects
for project in "${PROJECTS[@]}"; do
    if [ -d "$project" ]; then
        monitor_project "$project"
    fi
done
```

### Claude Code Hook Examples

**1. Enhanced Before-Compact Hook**
```bash
#!/bin/bash
# Enhanced before-compact-export.sh with advanced TTS

# ... existing export logic ...

# Advanced TTS notification
TTS_PROVIDER_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"
PROJECT_NAME=$(basename "$PWD")
ENGINEER_NAME=${ENGINEER_NAME:-"Developer"}

# Create detailed message
create_detailed_message() {
    local file_count=$(find . -name "*.py" -o -name "*.js" -o -name "*.md" | wc -l)
    local git_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    
    echo "$ENGINEER_NAME, before compact hook completed for $PROJECT_NAME. Processed $file_count files on branch $git_branch. Conversation exported and ready for storage."
}

# Send TTS notification
if [ -f "$TTS_PROVIDER_SCRIPT" ]; then
    DETAILED_MESSAGE=$(create_detailed_message)
    
    if timeout 15 python3 "$TTS_PROVIDER_SCRIPT" "$DETAILED_MESSAGE" >/dev/null 2>&1; then
        log_message "TTS notification sent: $DETAILED_MESSAGE"
    else
        # Fallback to simple message
        SIMPLE_MESSAGE="$ENGINEER_NAME, compact hook completed for $PROJECT_NAME"
        for cmd in espeak say spd-say; do
            if command -v $cmd >/dev/null 2>&1; then
                $cmd "$SIMPLE_MESSAGE" &
                log_message "TTS fallback used: $cmd"
                break
            fi
        done
    fi
fi
```

**2. PostToolUse Hook with TTS**
```bash
#!/bin/bash
# post-tool-use-hook.sh

TOOL_NAME="$1"
PROJECT_NAME=$(basename "$PWD")
TTS_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"

# Tool-specific messages
case "$TOOL_NAME" in
    "Edit"|"MultiEdit")
        MESSAGE="File editing completed in $PROJECT_NAME"
        ;;
    "Write")
        MESSAGE="New file created in $PROJECT_NAME"
        ;;
    "Bash")
        MESSAGE="Command execution completed in $PROJECT_NAME"
        ;;
    "mcp__qdrant__*")
        MESSAGE="Vector database operation completed in $PROJECT_NAME"
        ;;
    "mcp__chroma__*")
        MESSAGE="Process tracking operation completed in $PROJECT_NAME"
        ;;
    *)
        MESSAGE="Tool operation completed in $PROJECT_NAME"
        ;;
esac

# Send notification
if [ -f "$TTS_SCRIPT" ]; then
    timeout 10 python3 "$TTS_SCRIPT" "$MESSAGE" >/dev/null 2>&1 &
fi
```

**3. Session End Hook with Summary**
```bash
#!/bin/bash
# session-end-hook.sh

PROJECT_NAME=$(basename "$PWD")
TTS_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"
ENGINEER_NAME=${ENGINEER_NAME:-"Developer"}

# Generate session summary
generate_session_summary() {
    local start_time=$(redis-cli GET "session:start:$PROJECT_NAME" 2>/dev/null || echo "unknown")
    local operations=$(redis-cli HGET "claude:project_stats:$PROJECT_NAME:$(date +%Y%m%d)" "total_operations" 2>/dev/null || echo "0")
    local files_modified=$(redis-cli HGET "claude:project_stats:$PROJECT_NAME:$(date +%Y%m%d)" "files_modified" 2>/dev/null || echo "0")
    
    echo "$ENGINEER_NAME, session completed for $PROJECT_NAME. $operations operations performed, $files_modified files modified. Session data exported to Redis for continuity."
}

# Send summary notification
if [ -f "$TTS_SCRIPT" ]; then
    SUMMARY=$(generate_session_summary)
    timeout 15 python3 "$TTS_SCRIPT" "$SUMMARY" >/dev/null 2>&1 &
fi
```

### Integration Examples
```bash
#!/bin/bash
# before-compact-export.sh with TTS integration

# ... existing export logic ...

# TTS notification at completion
TTS_PROVIDER_SCRIPT="/home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py"
TTS_MESSAGE="Before compact hook completed for project $PROJECT_NAME. Conversation exported and ready for storage."

if [ -f "$TTS_PROVIDER_SCRIPT" ]; then
    if timeout 15 python3 "$TTS_PROVIDER_SCRIPT" "$TTS_MESSAGE" >/dev/null 2>&1; then
        log_message "TTS notification sent: $TTS_MESSAGE"
    else
        # System TTS fallback
        for cmd in espeak say spd-say; do
            if command -v $cmd >/dev/null 2>&1; then
                $cmd "$TTS_MESSAGE" &
                log_message "TTS fallback used: $cmd"
                break
            fi
        done
    fi
fi
```

### Custom Hook with TTS
```bash
#!/bin/bash
# custom-hook.sh with TTS integration

source /home/bryan/brainpods/.claude/hooks/utils/tts/tts_common.sh

# Your hook logic here
perform_custom_operation() {
    # ... custom logic ...
    
    # TTS notification
    notify_tts "Custom operation completed for project $(basename "$PWD")"
}

# Execute with TTS
perform_custom_operation
```

## Testing & Validation

### Unit Tests
```python
# test_tts_integration.py
import unittest
from unittest.mock import patch, MagicMock
from tts_provider import TTSProvider

class TestTTSIntegration(unittest.TestCase):
    
    def setUp(self):
        self.tts = TTSProvider()
    
    def test_provider_selection(self):
        """Test provider selection logic"""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}):
            self.assertEqual(self.tts.select_provider(), "elevenlabs")
    
    def test_fallback_chain(self):
        """Test fallback through provider chain"""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.tts.select_provider(), "pyttsx3")
    
    def test_graceful_degradation(self):
        """Test graceful degradation when all providers fail"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(self.tts.speak("test message"))
```

### Integration Tests
```bash
#!/bin/bash
# test_tts_integration.sh

echo "🧪 Testing TTS Integration..."

# Test 1: System packages
echo "Testing system packages..."
python3 -c "import requests, pygame, dotenv" && echo "✅ System packages OK" || echo "❌ System packages missing"

# Test 2: TTS provider
echo "Testing TTS provider..."
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py "Integration test message" && echo "✅ TTS provider OK" || echo "❌ TTS provider failed"

# Test 3: Hook integration
echo "Testing hook integration..."
/home/bryan/.claude/hooks/test-hook.sh && echo "✅ Hook integration OK" || echo "❌ Hook integration failed"

echo "🏁 Integration tests completed"
```

## Performance Considerations

### Timeout Management
- **TTS Operations**: 15-second timeout for hook integration
- **API Calls**: 30-second timeout for individual provider calls
- **Fallback Chain**: 5-second timeout per provider in chain

### Resource Usage
- **Memory**: ~50MB for pygame audio playback
- **CPU**: Minimal impact with background execution
- **Network**: ~1KB per ElevenLabs API call

### Optimization Strategies
```python
# Connection pooling for API providers
class ElevenLabsProvider:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"xi-api-key": self.api_key})
    
    def __del__(self):
        self.session.close()
```

## Troubleshooting Guide

### Common Issues

**1. "No module named 'requests'" Error**
```bash
# Solution: Install system packages
sudo apt install python3-requests python3-pygame python3-dotenv
```

**2. ElevenLabs API Rate Limiting**
```bash
# Solution: Check API usage and implement caching
# Error: "Unusual activity detected"
# Fix: Use different API key or switch to pyttsx3
```

**3. "espeak not installed" Error**
```bash
# Solution: Install espeak for pyttsx3 fallback
sudo apt install espeak-ng
```

**4. TTS Blocking Hook Execution**
```bash
# Solution: Ensure background execution with timeout
timeout 15 python3 "$TTS_SCRIPT" "$MESSAGE" >/dev/null 2>&1 &
```

### Debug Commands
```bash
# Test individual providers
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/elevenlabs_tts.py "test"
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/pyttsx3_tts.py "test"

# Check provider availability
python3 /home/bryan/brainpods/.claude/hooks/utils/tts/tts_provider.py --debug

# Verify configuration
cat /home/bryan/brainpods/.env | grep -E "(TTS|ELEVENLABS)"
```

## Future Enhancements

### Planned Features
1. **Voice Cloning**: Custom voice models for personalization
2. **Adaptive Notifications**: Context-aware message selection
3. **Multi-Language Support**: Localized TTS messages
4. **Emotion Detection**: Tone-appropriate voice synthesis
5. **Performance Metrics**: TTS operation analytics

### Extension Points
```python
# Custom provider interface
class CustomTTSProvider:
    def speak(self, text: str) -> bool:
        """Implement custom TTS logic"""
        pass
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
```

## Security Considerations

### API Key Protection
- Store API keys in environment variables only
- Never commit API keys to version control
- Use restricted API keys with minimal permissions
- Implement key rotation procedures

### Audio Content Safety
- Sanitize text input to prevent injection attacks
- Limit message length to prevent abuse
- Log all TTS operations for audit trails
- Implement rate limiting for API calls

## Conclusion

The TTS integration system provides a robust, production-ready solution for audio notifications in Claude Code hooks. With proper dependency management, comprehensive fallback mechanisms, and graceful degradation, it enhances user experience while maintaining system reliability.

The system follows best practices for error handling, resource management, and security, making it suitable for production environments and easy to extend with additional providers or features.

---

**Maintenance**: Review and update this documentation when adding new providers or modifying integration patterns.  
**Contact**: Bryan Thomas - System Administrator  
**Repository**: setup-mcp-server.sh.APP/docs/