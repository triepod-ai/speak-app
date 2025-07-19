# notify_tts() Quick Reference

## Function Call Syntax
```bash
notify_tts "message" ["priority"]
```

## Priority Levels
| Priority | Prefix Added | Use Case |
|----------|-------------|----------|
| `normal` (default) | `$ENGINEER_NAME,` | General notifications |
| `important` | `$ENGINEER_NAME, Important:` | High-priority info |
| `error` | `$ENGINEER_NAME, Error:` | Error conditions |
| `subagent_complete` | `$ENGINEER_NAME, Sub-agent completed:` | AI operation status |
| `memory_confirmed` | `$ENGINEER_NAME, Memory operation confirmed:` | Successful storage |
| `memory_failed` | `$ENGINEER_NAME, Memory operation failed:` | Failed storage |

## Common Usage Patterns

### Basic Notifications
```bash
notify_tts "Starting operation"
notify_tts "Operation complete"
notify_tts "Loading data from cache"
```

### Error Handling
```bash
notify_tts "Connection failed" "error"
notify_tts "File not found" "error"
notify_tts "Invalid configuration" "error"
```

### AI Operations
```bash
notify_tts "AI analysis for project started" "subagent_complete"
notify_tts "AI analysis completed in 5 seconds" "subagent_complete"
notify_tts "AI analysis failed after timeout" "subagent_complete"
```

### Storage Operations
```bash
notify_tts "1024 bytes saved to Redis" "memory_confirmed"
notify_tts "Document added to Chroma" "memory_confirmed"
notify_tts "Failed to connect to Qdrant" "memory_failed"
```

### Important Updates
```bash
notify_tts "Critical update available" "important"
notify_tts "System resources low" "important"
notify_tts "Manual intervention required" "important"
```

## Environment Variables
```bash
export TTS_ENABLED=true          # Enable/disable TTS
export ENGINEER_NAME="Bryan"     # Personalize messages
export TTS_PROVIDER=openai       # Force specific provider
```

## Integration Checklist
- [ ] Replace old `notify_tts()` function with new version
- [ ] Remove `TTS_SCRIPT` variable references
- [ ] Ensure `speak` command is in PATH
- [ ] Test with `TTS_ENABLED=false`
- [ ] Verify all priority levels work

## Testing Commands
```bash
# Test basic notification
notify_tts "Test message"

# Test all priorities
for priority in normal important error subagent_complete memory_confirmed memory_failed; do
    notify_tts "Testing $priority priority" "$priority"
    sleep 1
done

# Test with disabled TTS
TTS_ENABLED=false notify_tts "This should be silent"

# Test with custom name
ENGINEER_NAME="Test User" notify_tts "Personalized message"
```