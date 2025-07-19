#!/bin/bash
# script-integration.sh - Examples of integrating speak into scripts

set -euo pipefail

echo "=== Script Integration Examples ==="
echo

# Example 1: Build script with notifications
echo "1. Build script with notifications:"
cat << 'EOF'
#!/bin/bash
speak "Starting build process"

if make clean && make build; then
    speak "Build completed successfully"
else
    speak "Build failed. Check error logs"
    exit 1
fi
EOF
echo

# Example 2: Backup script with progress
echo "2. Backup script with progress updates:"
cat << 'EOF'
#!/bin/bash
speak "Starting backup"

for dir in Documents Pictures Videos; do
    if [ -d "$HOME/$dir" ]; then
        tar -czf "backup_${dir}.tar.gz" "$HOME/$dir" 2>/dev/null
        speak "Backed up $dir"
    fi
done

speak "Backup complete"
EOF
echo

# Example 3: Monitoring script
echo "3. System monitoring with alerts:"
cat << 'EOF'
#!/bin/bash
# Monitor CPU usage
CPU_USAGE=$(top -b -n1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( ${CPU_USAGE%.*} > 80 )); then
    speak "Warning: High CPU usage detected at ${CPU_USAGE}%"
fi

# Monitor disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if (( DISK_USAGE > 90 )); then
    speak "Critical: Disk space low. Only $((100-DISK_USAGE))% free"
fi
EOF
echo

# Example 4: Error handling with notifications
echo "4. Error handling pattern:"
cat << 'EOF'
#!/bin/bash
notify_and_exit() {
    local message="$1"
    local exit_code="${2:-1}"
    
    speak "$message"
    exit "$exit_code"
}

# Usage
command_that_might_fail || notify_and_exit "Command failed!"
EOF
echo

# Example 5: Long-running task notifications
echo "5. Long-running task with completion notification:"
cat << 'EOF'
#!/bin/bash
# Run task in background with notification
(
    sleep 30  # Simulate long task
    speak "Background task completed"
) &

echo "Task running in background (PID: $!)"
EOF
echo

# Example 6: Interactive script
echo "6. Interactive script with voice feedback:"
cat << 'EOF'
#!/bin/bash
speak "Please make a selection"
echo "1) Option One"
echo "2) Option Two"
echo "3) Exit"

read -p "Choice: " choice

case $choice in
    1) speak "You selected option one";;
    2) speak "You selected option two";;
    3) speak "Goodbye"; exit 0;;
    *) speak "Invalid selection";;
esac
EOF
echo

# Example 7: Git hook integration
echo "7. Git pre-commit hook:"
cat << 'EOF'
#!/bin/bash
# .git/hooks/pre-commit
speak --provider pyttsx3 "Running pre-commit checks"

if ! npm test &>/dev/null; then
    speak "Tests failed. Commit aborted"
    exit 1
fi

speak --off "Pre-commit checks passed"  # Silent success
EOF
echo

# Example 8: Cron job with conditional speech
echo "8. Cron job that only speaks when interactive:"
cat << 'EOF'
#!/bin/bash
# Detect if running interactively
if [[ -t 0 ]] && [[ "$TTS_ENABLED" == "true" ]]; then
    SPEAK="speak"
else
    SPEAK="echo"  # Fallback to echo in cron
fi

$SPEAK "Running scheduled maintenance"
# ... maintenance tasks ...
$SPEAK "Maintenance complete"
EOF
echo

# Example 9: Function wrapper for consistent notifications
echo "9. Notification function wrapper:"
cat << 'EOF'
#!/bin/bash
notify() {
    local level="$1"
    local message="$2"
    
    case "$level" in
        info)
            speak --provider pyttsx3 "$message"
            ;;
        warning)
            speak "$message"
            ;;
        error)
            speak --provider elevenlabs "$message" || speak "$message"
            logger -p user.err "$message"
            ;;
    esac
}

# Usage
notify info "Process started"
notify warning "Resource usage high"
notify error "Critical failure detected"
EOF
echo

# Example 10: Performance-aware integration
echo "10. Performance-aware batch processing:"
cat << 'EOF'
#!/bin/bash
# Disable TTS during batch operations
ORIGINAL_TTS=$TTS_ENABLED
export TTS_ENABLED=false

speak "Processing 1000 files"

for file in *.dat; do
    process_file "$file"
done

# Re-enable and notify completion
export TTS_ENABLED=$ORIGINAL_TTS
speak "Batch processing complete"
EOF

echo -e "\n=== Script integration examples complete ===

These examples demonstrate various patterns for integrating
the speak command into your scripts effectively."