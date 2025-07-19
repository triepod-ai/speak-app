#!/bin/bash
# advanced-features.sh - Advanced features and patterns for speak command

set -euo pipefail

echo "=== Advanced Speak Command Features ==="
echo

# Example 1: Provider fallback chain
echo "1. Provider fallback chain:"
cat << 'EOF'
#!/bin/bash
speak_with_fallback() {
    local message="$1"
    
    # Try providers in order of preference
    speak --provider elevenlabs "$message" 2>/dev/null || \
    speak --provider openai "$message" 2>/dev/null || \
    speak --provider pyttsx3 "$message" 2>/dev/null || \
    echo "ALERT: $message" >&2
}

speak_with_fallback "Critical system alert"
EOF
echo

# Example 2: Rate-limited notifications
echo -e "\n2. Rate-limited notifications to prevent spam:"
cat << 'EOF'
#!/bin/bash
LAST_SPEAK_TIME=0
MIN_INTERVAL=5  # seconds

rate_limited_speak() {
    local message="$1"
    local current_time=$(date +%s)
    
    if (( current_time - LAST_SPEAK_TIME >= MIN_INTERVAL )); then
        speak "$message"
        LAST_SPEAK_TIME=$current_time
        return 0
    else
        echo "[QUEUED] $message" >> /tmp/speak_queue.log
        return 1
    fi
}

# In a monitoring loop
while true; do
    if check_condition; then
        rate_limited_speak "Condition detected"
    fi
    sleep 1
done
EOF
echo

# Example 3: Async notifications
echo -e "\n3. Asynchronous notifications (non-blocking):"
cat << 'EOF'
#!/bin/bash
speak_async() {
    local message="$1"
    local timeout="${2:-10}"
    
    # Run in background with timeout
    (
        timeout "${timeout}s" speak "$message" 2>/dev/null || \
        echo "$(date): TTS timeout: $message" >> speak_errors.log
    ) &
    
    local pid=$!
    echo "Speaking in background (PID: $pid)"
    return 0
}

# Non-blocking notifications
speak_async "Starting long operation"
perform_long_operation
speak_async "Operation complete"

# Wait for all background speech to finish
wait
EOF
echo

# Example 4: Priority-based provider selection
echo -e "\n4. Priority-based provider selection:"
cat << 'EOF'
#!/bin/bash
speak_priority() {
    local message="$1"
    local priority="${2:-normal}"
    
    case "$priority" in
        critical)
            # Use best quality for critical messages
            speak --provider elevenlabs "$message" || \
            speak --provider openai "$message" || \
            speak "$message"
            
            # Also log and notify via other means
            logger -p user.crit "CRITICAL: $message"
            notify-send "Critical Alert" "$message" 2>/dev/null || true
            ;;
            
        high)
            # Use any cloud provider
            speak --provider elevenlabs "$message" || \
            speak --provider openai "$message" || \
            speak --provider pyttsx3 "$message"
            ;;
            
        normal|low)
            # Use fastest available
            speak --provider pyttsx3 "$message" || \
            speak "$message"
            ;;
    esac
}

speak_priority "System failure detected" "critical"
speak_priority "Task completed" "low"
EOF
echo

# Example 5: Multilingual support (future feature simulation)
echo -e "\n5. Multilingual notification pattern:"
cat << 'EOF'
#!/bin/bash
speak_multilingual() {
    local message="$1"
    local lang="${2:-en}"
    
    case "$lang" in
        es)
            # Spanish
            speak "Mensaje en español: $message"
            ;;
        fr)
            # French
            speak "Message en français: $message"
            ;;
        *)
            # Default to English
            speak "$message"
            ;;
    esac
}

speak_multilingual "Task complete" "en"
speak_multilingual "Tarea completa" "es"
EOF
echo

# Example 6: Notification queuing system
echo -e "\n6. Notification queue implementation:"
cat << 'EOF'
#!/bin/bash
QUEUE_FILE="/tmp/speak_queue.txt"

enqueue_message() {
    echo "$(date +%s):$1" >> "$QUEUE_FILE"
}

process_queue() {
    if [[ ! -f "$QUEUE_FILE" ]]; then
        return
    fi
    
    while IFS=: read -r timestamp message; do
        speak "$message"
        sleep 1  # Prevent overwhelming
    done < "$QUEUE_FILE"
    
    > "$QUEUE_FILE"  # Clear queue
}

# Usage
enqueue_message "First message"
enqueue_message "Second message"
enqueue_message "Third message"

# Process all queued messages
process_queue
EOF
echo

# Example 7: Context-aware notifications
echo -e "\n7. Context-aware notification system:"
cat << 'EOF'
#!/bin/bash
get_notification_context() {
    local context=""
    
    # Time-based context
    hour=$(date +%H)
    if (( hour >= 22 || hour <= 6 )); then
        context="quiet"  # Late night/early morning
    elif (( hour >= 9 && hour <= 17 )); then
        context="work"   # Work hours
    else
        context="normal"
    fi
    
    # System load context
    load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}')
    if (( $(echo "$load > 2" | bc -l) )); then
        context="${context}_highload"
    fi
    
    echo "$context"
}

context_aware_speak() {
    local message="$1"
    local context=$(get_notification_context)
    
    case "$context" in
        quiet*)
            # Use offline TTS at night
            speak --provider pyttsx3 "$message"
            ;;
        work*)
            # Normal operation during work
            speak "$message"
            ;;
        *highload)
            # Skip non-critical messages under high load
            [[ "$2" == "critical" ]] && speak "$message"
            ;;
    esac
}
EOF
echo

# Example 8: Speak with retry logic
echo -e "\n8. Robust speak with retry logic:"
cat << 'EOF'
#!/bin/bash
speak_with_retry() {
    local message="$1"
    local max_attempts="${2:-3}"
    local delay="${3:-2}"
    
    for ((i=1; i<=max_attempts; i++)); do
        if speak "$message"; then
            return 0
        fi
        
        if (( i < max_attempts )); then
            echo "Speak attempt $i failed, retrying in ${delay}s..." >&2
            sleep "$delay"
        fi
    done
    
    echo "Failed to speak after $max_attempts attempts" >&2
    return 1
}

speak_with_retry "Important message" 3 2
EOF
echo

# Example 9: Logging and debugging
echo -e "\n9. Debug mode for troubleshooting:"
cat << 'EOF'
#!/bin/bash
DEBUG_SPEAK=${DEBUG_SPEAK:-false}

debug_speak() {
    local message="$1"
    
    if [[ "$DEBUG_SPEAK" == "true" ]]; then
        echo "[DEBUG] Speak called with: $message" >&2
        echo "[DEBUG] Provider: ${TTS_PROVIDER:-auto}" >&2
        echo "[DEBUG] Enabled: ${TTS_ENABLED:-true}" >&2
    fi
    
    # Time the operation
    local start=$(date +%s.%N)
    speak "$message"
    local result=$?
    local end=$(date +%s.%N)
    
    if [[ "$DEBUG_SPEAK" == "true" ]]; then
        local duration=$(echo "$end - $start" | bc)
        echo "[DEBUG] Speak took ${duration}s, result: $result" >&2
    fi
    
    return $result
}

# Enable debugging
export DEBUG_SPEAK=true
debug_speak "Test message with debugging"
EOF
echo

# Example 10: Integration with notification systems
echo -e "\n10. Multi-channel notification system:"
cat << 'EOF'
#!/bin/bash
notify_all_channels() {
    local message="$1"
    local severity="${2:-info}"
    
    # Speak notification
    speak_priority "$message" "$severity"
    
    # Desktop notification
    command -v notify-send &>/dev/null && \
        notify-send "Script Alert" "$message"
    
    # System log
    logger -t "script_alert" -p "user.$severity" "$message"
    
    # Email for critical
    if [[ "$severity" == "critical" ]]; then
        echo "$message" | mail -s "Critical Alert" admin@example.com 2>/dev/null || true
    fi
    
    # Slack/Discord webhook (if configured)
    if [[ -n "$WEBHOOK_URL" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$message\"}" \
            2>/dev/null || true
    fi
}

# Usage
notify_all_channels "Deployment complete" "info"
notify_all_channels "Database connection lost" "critical"
EOF

echo -e "\n=== Advanced features demonstration complete ===

These examples show advanced patterns for:
- Fallback chains and retry logic
- Rate limiting and async operations
- Priority-based routing
- Context awareness
- Multi-channel notifications
- Debugging and logging

For more examples, see the documentation in ../docs/"

# Make all example scripts executable
chmod +x /home/bryan/bin/speak-app/examples/*.sh 2>/dev/null || true