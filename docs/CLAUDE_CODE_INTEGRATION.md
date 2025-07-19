# Claude Code Integration with Speak App

This document describes the integration between Claude Code's Dynamic Hook System and the Speak App for voice-friendly notifications.

## Overview

The Claude Code Dynamic Hook System has been enhanced to provide voice-friendly notifications for all tool operations, data storage/retrieval, and analysis tasks. This integration uses the speak command to deliver natural, human-friendly audio feedback.

## Architecture

### Components

1. **Voice Formatter** (`.claude-logs/hooks/voice_formatter.py`)
   - Converts technical data to natural speech
   - Handles numbers, paths, durations, and commands
   - Removes technical IDs and long hashes

2. **Voice Queue** (`.claude-logs/hooks/voice_queue.py`)
   - Manages message pacing and prioritization
   - Prevents audio overload with intelligent batching
   - Integrates with speak command for TTS output

3. **Enhanced Templates** (`.claude-logs/hooks/dynamic_message_templates.py`)
   - Voice-friendly message templates
   - Category-based message generation
   - Support for data and analysis operations

4. **Hook Logger** (`.claude-logs/hooks/dynamic_hook_logger.py`)
   - Captures all Claude Code operations
   - Generates voice messages for tool usage
   - Manages priority and timing

## Configuration

### Environment Variables

```bash
# Enable/disable voice hooks in Claude Code
export VOICE_HOOKS=true

# Voice queue configuration
export VOICE_MIN_GAP=2.0      # Seconds between messages
export VOICE_MAX_QUEUE=5      # Maximum queued messages
export VOICE_BATCH_TIMEOUT=0.5 # Batch collection timeout

# TTS configuration (used by speak command)
export TTS_ENABLED=true
export TTS_PROVIDER=auto      # auto, elevenlabs, openai, pyttsx3
```

### Claude Code Settings

In `.claude/settings.local.json`:

```json
{
  "hooks": {
    // Standard tool hooks (Read, Write, Bash, etc.)
    "DataOperation": [
      // Hooks for data storage/retrieval operations
    ],
    "AnalysisOperation": [
      // Hooks for AI analysis operations
    ]
  },
  "voiceSettings": {
    "enabled": true,
    "minGapSeconds": 2.0,
    "maxQueueSize": 5,
    "priorityThresholds": {
      "critical": ["error", "failure"],
      "high": ["complete", "success"],
      "normal": ["start", "processing"],
      "low": ["progress", "update"]
    }
  }
}
```

## Voice Message Examples

### Technical to Natural Conversion

| Technical Data | Voice Output |
|----------------|--------------|
| 12345678 bytes | "twelve megabytes" |
| 3456ms | "in three seconds" |
| 95.7% | "nearly complete" |
| /home/user/project/src/main.py | "main file in project" |
| git commit -m 'Fix bug' | "run git command" |
| Error code 404 | "not found error" |

### Operation Messages

| Operation | Voice Message |
|-----------|---------------|
| Reading file | "Now reading config file" |
| Storing data | "Storing twelve kilobytes of data" |
| Analysis complete | "Analysis complete in two seconds" |
| Error occurred | "Failed to connect to database" |

## Priority System

Messages are prioritized to ensure important notifications are heard:

1. **CRITICAL** (Priority 0)
   - Errors and failures
   - Spoken immediately
   - Never batched or skipped

2. **HIGH** (Priority 1)
   - Important completions
   - Success notifications
   - Normal pacing

3. **NORMAL** (Priority 2)
   - Operation starts
   - Regular processing
   - Can be batched

4. **LOW** (Priority 3)
   - Progress updates
   - May be skipped if queue is full
   - Often batched together

## Intelligent Batching

The system groups related operations to reduce voice spam:

```
Instead of:
- "Reading file one"
- "Reading file two"
- "Reading file three"

You hear:
- "Processing three files"
```

## Usage Patterns

### Basic Operations

When you use Claude Code tools, you'll hear:

```bash
# File operations
"Now reading project configuration"
"Writing to output file"

# Data operations
"Storing five kilobytes of data"
"Retrieved user preferences"

# Analysis operations
"Starting code analysis"
"Analysis complete"

# Errors
"Error: File not found"
"Connection failed"
```

### Controlling Voice Output

```bash
# Temporarily disable voice
export VOICE_HOOKS=false

# Adjust speaking speed
export VOICE_MIN_GAP=1.0  # Faster
export VOICE_MIN_GAP=3.0  # Slower

# Skip voice for specific operations
# (The hook system respects TTS_ENABLED)
export TTS_ENABLED=false
```

### Integration with Development Workflow

The voice system enhances your development workflow by:

1. **Non-intrusive Notifications**: Operations speak their status without blocking
2. **Error Awareness**: Immediate voice alerts for failures
3. **Progress Tracking**: Know when long operations complete
4. **Context Switching**: Audio cues help maintain awareness during multitasking

## Technical Integration Details

### Message Flow

1. Claude Code executes a tool (Read, Write, MCP operation, etc.)
2. Hook system captures the operation via settings.local.json
3. Dynamic hook logger processes the event
4. Voice formatter converts technical data to natural speech
5. Voice queue manages pacing and priority
6. Speak command delivers the audio using best available TTS provider

### Performance Considerations

- Very fast operations (<500ms) are given LOW priority
- Messages are batched within 0.5 second windows
- Queue holds maximum 5 messages before summarizing
- Non-blocking execution ensures no performance impact

### Error Handling

The system gracefully handles:
- Missing speak command (falls back to logging)
- Disabled TTS (respects TTS_ENABLED)
- Queue overflow (summarizes pending messages)
- Provider failures (speak command handles fallback)

## Troubleshooting

### No Voice Output

1. Check voice hooks are enabled:
   ```bash
   echo $VOICE_HOOKS  # Should be "true"
   ```

2. Verify speak command works:
   ```bash
   speak "Test message"
   ```

3. Check TTS is enabled:
   ```bash
   echo $TTS_ENABLED  # Should be "true"
   ```

### Too Much Voice Output

1. Increase message gap:
   ```bash
   export VOICE_MIN_GAP=3.0
   ```

2. Reduce queue size:
   ```bash
   export VOICE_MAX_QUEUE=3
   ```

3. Temporarily disable:
   ```bash
   export VOICE_HOOKS=false
   ```

### Voice Timing Issues

1. Adjust batch timeout:
   ```bash
   export VOICE_BATCH_TIMEOUT=1.0  # Longer batching window
   ```

2. Check system load affecting queue processing

## Future Enhancements

- [ ] Per-tool voice settings
- [ ] Custom voice profiles for different operation types
- [ ] Sound effects for errors vs. success
- [ ] Voice commands to control Claude Code
- [ ] Multi-language support for international developers

## API Reference

### Voice Queue API (Python)

```python
from voice_queue import get_voice_queue, Priority

# Get queue instance
queue = get_voice_queue()

# Add a message
queue.add_message(
    text="Operation complete",
    priority=Priority.HIGH,
    category="completion",
    metadata={"duration": 1234}
)

# Control queue
queue.pause()
queue.resume()
queue.set_speed("slow")  # slow, normal, fast
```

### Voice Formatter API (Python)

```python
from voice_formatter import VoiceFormatter

formatter = VoiceFormatter()

# Format various data types
formatter.format_number(12345678, "size")  # "12 megabytes"
formatter.format_path("/long/path/to/file")  # "file in path"
formatter.format_command("git status")  # "run git command"
formatter.format_timestamp("2024-01-01T12:00:00Z")  # "3 hours ago"
```

## Contributing

To enhance the Claude Code voice integration:

1. Voice templates are in `dynamic_message_templates.py`
2. Formatting rules are in `voice_formatter.py`
3. Queue behavior is in `voice_queue.py`
4. Hook integration is in `dynamic_hook_logger.py`

Test changes with:
```bash
cd .claude-logs/hooks
python3 test_voice_system.py
python3 demo_voice_hooks.py
```