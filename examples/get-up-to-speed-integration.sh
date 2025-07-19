#!/bin/bash

# Get-Up-To-Speed Integration Example
# Shows how the speak command is integrated into Claude Code's get-up-to-speed commands

# Example 1: Basic notify_tts function implementation
notify_tts() {
    local message="$1"
    local priority="${2:-normal}"
    
    ENGINEER_NAME=${ENGINEER_NAME:-"Developer"}
    
    # Skip TTS if disabled
    if [ "${TTS_ENABLED:-true}" != "true" ]; then
        return 0
    fi
    
    # Format message based on priority
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
    
    # Use speak command (non-blocking)
    speak "$PERSONALIZED_MESSAGE" &
}

# Example 2: Context loading notifications
echo "=== Example: Context Loading Workflow ==="
notify_tts "Loading project context for setup-mcp-server from Redis cache"
sleep 1

notify_tts "Found 5 Redis keys for setup-mcp-server"
sleep 1

notify_tts "AI analysis for setup-mcp-server project context" "subagent_complete"
sleep 2

notify_tts "Context loading complete for setup-mcp-server. Ready to work." "important"

# Example 3: Memory operation notifications
echo ""
echo "=== Example: Memory Operations ==="
notify_tts "2048 bytes saved to key project:status:setup-mcp-server" "memory_confirmed"
sleep 1

notify_tts "Failed to save to Chroma collection" "memory_failed"

# Example 4: Error handling
echo ""
echo "=== Example: Error Conditions ==="
notify_tts "Redis unavailable, using fallback" "error"

# Example 5: Disabling TTS
echo ""
echo "=== Example: Silent Mode ==="
export TTS_ENABLED=false
notify_tts "This message will not be spoken"
export TTS_ENABLED=true

# Example 6: Custom engineer name
echo ""
echo "=== Example: Personalization ==="
export ENGINEER_NAME="Bryan"
notify_tts "Welcome back! Loading your project context." "important"

echo ""
echo "✅ Integration examples complete!"
echo ""
echo "These examples show how get-up-to-speed commands use the speak command"
echo "for meaningful audio feedback throughout the context loading process."