# notify_tts() Function - Speak Command Integration

## Overview

The `notify_tts()` function is a unified TTS notification system integrated into the get-up-to-speed command suite. It provides intelligent, context-aware voice notifications using the speak command infrastructure.

## Function Signature

```bash
notify_tts() {
    local message="$1"
    local priority="${2:-normal}"  # Priority level for message formatting
}
```

## Parameters

### message (required)
- **Type**: String
- **Description**: The message to be spoken
- **Example**: `"Loading project context for setup-mcp-server"`

### priority (optional)
- **Type**: String
- **Default**: `"normal"`
- **Valid Values**:
  - `normal` - Standard notification
  - `important` - High-priority information
  - `error` - Error notifications
  - `subagent_complete` - AI sub-agent completion status
  - `memory_confirmed` - Successful memory operation
  - `memory_failed` - Failed memory operation

## Implementation Details

### 1. Environment Variables

```bash
ENGINEER_NAME=${ENGINEER_NAME:-"Developer"}  # Personalization
TTS_ENABLED=${TTS_ENABLED:-"true"}          # Global enable/disable
```

### 2. Priority-Based Message Formatting

The function formats messages based on priority levels to provide context-aware notifications:

```bash
case "$priority" in
    "subagent_complete")
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, Sub-agent completed: $message"
        ;;
    "memory_confirmed")
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, Memory operation confirmed: $message"
        ;;
    "memory_failed")
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, Memory operation failed: $message"
        ;;
    "error")
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, Error: $message"
        ;;
    "important")
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, Important: $message"
        ;;
    *)
        PERSONALIZED_MESSAGE="$ENGINEER_NAME, $message"
        ;;
esac
```

### 3. Non-Blocking Execution

The function uses background execution (`&`) to prevent blocking script flow:

```bash
speak "$PERSONALIZED_MESSAGE" &
```

## Usage Examples

### Basic Notification
```bash
notify_tts "Loading project context"
# Output: "Developer, Loading project context"
```

### Error Notification
```bash
notify_tts "Redis connection failed" "error"
# Output: "Developer, Error: Redis connection failed"
```

### Memory Operation Feedback
```bash
notify_tts "2048 bytes saved to Redis cache" "memory_confirmed"
# Output: "Developer, Memory operation confirmed: 2048 bytes saved to Redis cache"
```

### AI Sub-agent Status
```bash
notify_tts "AI analysis for project completed in 3s" "subagent_complete"
# Output: "Developer, Sub-agent completed: AI analysis for project completed in 3s"
```

### With Custom Engineer Name
```bash
export ENGINEER_NAME="Bryan"
notify_tts "Context loading complete"
# Output: "Bryan, Context loading complete"
```

## Integration Points

### 1. Get-Up-To-Speed Scripts
The function is integrated into all get-up-to-speed variants:
- `/get-up-to-speed` - Context loading notifications
- `/get-up-to-speed-export` - Session export status
- `/get-up-to-speed-dump-session-to-storage` - Storage operation feedback
- `/get-up-to-speed-analyze` - Analysis progress updates

### 2. Common Notification Patterns

#### Context Loading
```bash
notify_tts "Loading project context for $PROJECT_NAME from Redis cache"
notify_tts "Found $key_count Redis keys for $PROJECT_NAME"
notify_tts "Context loading complete for $PROJECT_NAME. Ready to work."
```

#### Data Discovery
```bash
notify_tts "Found recent Redis updates for $PROJECT_NAME"
notify_tts "No Redis data found for $PROJECT_NAME" "error"
notify_tts "Searching for structured export data across all projects"
```

#### AI Operations
```bash
notify_tts "Sub-agent started: AI analysis for $PROJECT_NAME" "subagent_complete"
notify_tts "AI analysis for $PROJECT_NAME completed in ${duration}s" "subagent_complete"
notify_tts "AI analysis for $PROJECT_NAME failed" "subagent_complete"
```

#### Memory Operations
```bash
notify_tts "Redis storage: ${value_size} bytes saved to key $key" "memory_confirmed"
notify_tts "Chroma storage: failed to save to $collection_name" "memory_failed"
notify_tts "Qdrant storage: ${info_size} bytes saved in ${duration}s" "memory_confirmed"
```

## Configuration

### Enable/Disable TTS
```bash
# Disable TTS globally
export TTS_ENABLED=false

# Enable TTS (default)
export TTS_ENABLED=true
```

### Personalization
```bash
# Set custom engineer name
export ENGINEER_NAME="Your Name"
```

### Provider Selection
The speak command handles provider selection automatically, but you can override:
```bash
# Force specific provider
export TTS_PROVIDER=openai  # or elevenlabs, pyttsx3
```

## Benefits Over Previous Implementation

### 1. Cost Efficiency
- 95% cost reduction with OpenAI as default provider
- Intelligent caching system reduces API calls

### 2. Reliability
- Multi-provider fallback chain ensures TTS always works
- Automatic offline fallback with pyttsx3

### 3. Performance
- Built-in observability filtering prevents audio spam
- Non-blocking execution maintains script performance

### 4. Simplicity
- Single `speak` command replaces complex Python script
- No additional dependencies or configuration needed

### 5. CI/CD Safety
- Automatically disables in non-interactive environments
- Respects environment variables for flexible control

## Troubleshooting

### No Audio Output
1. Check if TTS is enabled: `echo $TTS_ENABLED`
2. Verify speak command is available: `which speak`
3. Test speak directly: `speak "test message"`
4. Check provider status: `speak --status`

### Wrong Voice/Provider
1. Check current provider: `speak --list`
2. Force specific provider: `export TTS_PROVIDER=openai`
3. Verify API keys are set for cloud providers

### Too Many Notifications
1. Use priority levels to filter important messages
2. Implement rate limiting in loops
3. Consider disabling TTS for specific operations

## Best Practices

1. **Use Appropriate Priority Levels**: Match priority to message importance
2. **Avoid Loops**: Don't call notify_tts in tight loops without rate limiting
3. **Provide Context**: Include relevant details in messages (sizes, durations, counts)
4. **Test Silent Mode**: Always test scripts with `TTS_ENABLED=false`
5. **Personalize Responsibly**: Use ENGINEER_NAME for better user experience

## Migration from Old System

If migrating from the old `tts_provider.py` system:

1. Replace the entire `notify_tts()` function with the new version
2. Remove references to `TTS_SCRIPT` variable
3. Remove `timeout 15 python3` wrapper
4. Ensure `speak` command is in PATH
5. Test all priority levels work correctly

## Future Enhancements

Potential improvements for the notify_tts system:
- Rate limiting for high-frequency notifications
- Queue system for sequential messages
- Custom voice selection per priority level
- Integration with system notification APIs
- Logging of spoken messages for debugging