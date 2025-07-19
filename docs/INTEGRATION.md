# Integration Guide

This guide demonstrates how to integrate the speak command into various workflows, scripts, and applications.

## Table of Contents

- [Shell Scripts](#shell-scripts)
- [Git Hooks](#git-hooks)
- [CI/CD Pipelines](#cicd-pipelines)
- [Cron Jobs](#cron-jobs)
- [System Monitoring](#system-monitoring)
- [Development Workflows](#development-workflows)
- [Python Integration](#python-integration)
- [Claude Code Integration](#claude-code-integration)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)

## Shell Scripts

### Basic Integration

```bash
#!/bin/bash
# build.sh - Build script with voice notifications

set -euo pipefail

speak "Starting build process"

# Build steps
if make clean && make build; then
    speak "Build completed successfully"
    exit 0
else
    speak "Build failed. Check the error logs"
    exit 1
fi
```

### Progress Notifications

```bash
#!/bin/bash
# data_processor.sh - Long-running script with progress updates

speak "Starting data processing"

total_files=$(ls data/*.csv | wc -l)
processed=0

for file in data/*.csv; do
    # Process file (silent during loop)
    process_data "$file" || {
        speak "Error processing $file"
        exit 1
    }
    
    processed=$((processed + 1))
    
    # Speak progress every 10 files
    if (( processed % 10 == 0 )); then
        speak "Processed $processed of $total_files files"
    fi
done

speak "Data processing complete. Processed $total_files files"
```

### Conditional Notifications

```bash
#!/bin/bash
# deploy.sh - Deployment script with conditional alerts

# Only speak in production
if [[ "$ENVIRONMENT" == "production" ]]; then
    SPEAK_CMD="speak"
else
    SPEAK_CMD="speak --off"  # Silent in dev/staging
fi

$SPEAK_CMD "Starting deployment to $ENVIRONMENT"

# Deployment logic...
deploy_application || {
    $SPEAK_CMD "Deployment failed! Rolling back..."
    rollback_deployment
    exit 1
}

$SPEAK_CMD "Deployment successful"
```

## Git Hooks

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Announce hook execution
speak --provider pyttsx3 "Running pre-commit checks"

# Run tests silently
if ! npm test > /dev/null 2>&1; then
    speak "Tests failed. Commit aborted"
    exit 1
fi

# Run linter silently
if ! npm run lint > /dev/null 2>&1; then
    speak "Linting failed. Please fix errors"
    exit 1
fi

speak --off "Pre-commit checks passed"  # Silent success
exit 0
```

### Post-merge Hook

```bash
#!/bin/bash
# .git/hooks/post-merge

# Check if package.json changed
if git diff-tree -r --name-only --no-commit-id ORIG_HEAD HEAD | grep -q "package.json"; then
    speak "Package.json changed. Running npm install"
    npm install
fi

# Check if database migrations exist
if ls db/migrations/*.sql 2>/dev/null | grep -q .; then
    speak "New database migrations detected. Remember to run migrations"
fi
```

## CI/CD Pipelines

### GitHub Actions

```yaml
# .github/workflows/notify.yml
name: Build with Notifications

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup speak command
        run: |
          # Install speak (if not in container)
          # For CI, use offline TTS only
          echo 'export TTS_PROVIDER=pyttsx3' >> $GITHUB_ENV
          
      - name: Build
        run: |
          speak --off "Building project"  # Silent in CI
          make build
          
      - name: Test
        run: |
          speak --off "Running tests"
          make test
          
      - name: Notify on failure
        if: failure()
        run: |
          # Could send actual notification here
          echo "Build failed - would send notification"
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        TTS_ENABLED = 'false'  // Disable in CI
    }
    
    stages {
        stage('Build') {
            steps {
                sh 'speak --off "Building application"'
                sh 'make build'
            }
        }
        
        stage('Test') {
            steps {
                sh 'speak --off "Running tests"'
                sh 'make test'
            }
        }
    }
    
    post {
        success {
            script {
                // Send notification on success
                sh 'echo "Build successful"'
            }
        }
        failure {
            script {
                // Send notification on failure
                sh 'echo "Build failed"'
            }
        }
    }
}
```

## Cron Jobs

### Backup Script

```bash
#!/bin/bash
# /etc/cron.daily/backup

# Cron environment is minimal, set PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/bryan

# Load TTS configuration
source $HOME/brainpods/.env

# Only speak if running interactively (not from cron)
if [[ -t 0 ]]; then
    SPEAK="speak"
else
    SPEAK="speak --off"
fi

$SPEAK "Starting daily backup"

# Backup logic
backup_status=0
tar -czf /backup/daily-$(date +%Y%m%d).tar.gz /important/data || backup_status=$?

if [[ $backup_status -eq 0 ]]; then
    $SPEAK "Backup completed successfully"
else
    # In cron, this would go to email
    echo "Backup failed with status $backup_status" | mail -s "Backup Failed" admin@example.com
fi
```

### Monitoring Script

```bash
#!/bin/bash
# /usr/local/bin/monitor.sh - Run every 5 minutes via cron

# Check if running interactively
if [[ -t 0 ]] && [[ "$TTS_ENABLED" == "true" ]]; then
    NOTIFY="speak"
else
    NOTIFY="logger -t monitor"
fi

# Check disk usage
disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if (( disk_usage > 90 )); then
    $NOTIFY "Critical: Disk usage at ${disk_usage}%"
fi

# Check memory
mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
if (( mem_usage > 85 )); then
    $NOTIFY "Warning: Memory usage at ${mem_usage}%"
fi
```

## System Monitoring

### Service Monitor

```bash
#!/bin/bash
# service_monitor.sh - Monitor critical services

services=("nginx" "postgresql" "redis" "elasticsearch")

for service in "${services[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        speak "Alert: $service is not running"
        
        # Attempt restart
        if sudo systemctl restart "$service"; then
            speak "$service restarted successfully"
        else
            speak "Failed to restart $service. Manual intervention required"
        fi
    fi
done
```

### Log Monitor

```bash
#!/bin/bash
# error_monitor.sh - Monitor logs for errors

# Monitor application logs
tail -F /var/log/app/error.log | while read line; do
    if echo "$line" | grep -q "CRITICAL"; then
        speak "Critical error in application log"
        echo "$line" | mail -s "Critical Error" admin@example.com
    fi
done &

# Monitor system logs
journalctl -f | grep -i "error\|fail" | while read line; do
    # Rate limit notifications
    if [[ $(date +%s) -gt $((last_notify + 300)) ]]; then
        speak --provider pyttsx3 "System error detected"
        last_notify=$(date +%s)
    fi
done
```

## Development Workflows

### Test Runner Integration

```bash
#!/bin/bash
# test_notify.sh - Test runner with notifications

# Function to run tests with notification
run_tests() {
    local test_type="$1"
    
    echo "Running $test_type tests..."
    
    if npm run "test:$test_type" > test.log 2>&1; then
        speak --off "echo '$test_type tests passed'"  # Silent success
        return 0
    else
        speak "$test_type tests failed"
        tail -n 20 test.log
        return 1
    fi
}

# Run all test suites
speak "Starting test suite"

run_tests "unit" && \
run_tests "integration" && \
run_tests "e2e" && \
speak "All tests passed" || \
speak "Test suite failed"
```

### File Watcher

```bash
#!/bin/bash
# watch_and_build.sh - Auto-build on file changes

speak "Starting file watcher"

# Use inotifywait or fswatch
fswatch -o src/ | while read event; do
    echo "File changed, rebuilding..."
    
    if make build > build.log 2>&1; then
        speak --provider pyttsx3 "Build successful"
    else
        speak "Build failed"
        tail -n 10 build.log
    fi
done
```

## Python Integration

### Direct Library Usage

```python
#!/usr/bin/env python3
# python_integration.py

import sys
import subprocess
from pathlib import Path

# Add speak TTS to path
sys.path.insert(0, '/home/bryan/bin/speak-app/tts')

from tts_provider import TTSProvider

def notify(message):
    """Send TTS notification."""
    tts = TTSProvider()
    return tts.speak_with_fallback(message)

def main():
    notify("Starting Python script")
    
    try:
        # Your code here
        result = process_data()
        notify(f"Processing complete. Processed {result} items")
    except Exception as e:
        notify(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

### Subprocess Integration

```python
#!/usr/bin/env python3
# subprocess_integration.py

import subprocess
import os

def speak(text, provider=None):
    """Call speak command via subprocess."""
    cmd = ["speak"]
    
    if provider:
        cmd.extend(["--provider", provider])
    
    # Add --off flag if TTS is disabled
    if os.getenv("TTS_ENABLED", "true").lower() == "false":
        cmd.append("--off")
    
    cmd.append(text)
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Usage
speak("Starting data analysis")

for i in range(100):
    process_item(i)
    
    if i % 25 == 0:
        speak(f"Processed {i} items")

speak("Analysis complete")
```

## Claude Code Integration

The speak command is integrated with Claude Code's Dynamic Hook System to provide voice-friendly notifications for all AI coding operations.

### Overview

When using Claude Code, the Dynamic Hook System automatically generates voice notifications for:
- Tool operations (Read, Write, Edit, Bash commands)
- Data operations (storing, retrieving, caching)
- Analysis operations (AI thinking, code analysis)
- Errors and completions

### Configuration

```bash
# Enable/disable voice notifications in Claude Code
export VOICE_HOOKS=true

# Configure voice pacing
export VOICE_MIN_GAP=2.0      # Seconds between messages
export VOICE_MAX_QUEUE=5      # Maximum queued messages

# The speak command uses these standard settings
export TTS_ENABLED=true
export TTS_PROVIDER=auto
```

### Features

1. **Voice-Friendly Formatting**
   - Numbers converted to natural speech (1234567 → "1.2 million")
   - Paths abbreviated intelligently
   - Technical data simplified for voice

2. **Intelligent Queue Management**
   - Prioritized messages (errors speak immediately)
   - Batching of related operations
   - Configurable pacing to prevent audio overload

3. **Universal Coverage**
   - All Claude Code operations covered
   - Automatic categorization and prioritization
   - Context-aware message generation

### Example Integration

When Claude Code performs operations, you'll hear natural voice notifications:

```bash
# File operation
Claude Code: Read("/path/to/config.json")
Voice: "Now reading config file"

# Data storage
Claude Code: mcp__redis__store_memory(key="data", value="...")
Voice: "Storing five kilobytes of data"

# Error handling
Claude Code: Error - Connection failed
Voice: "Error: Connection failed" (spoken immediately)

# Completion
Claude Code: Operation complete in 3456ms
Voice: "Completed in three seconds"
```

### Hook System Details

The integration is implemented in `.claude-logs/hooks/`:
- `voice_formatter.py` - Converts technical data to natural speech
- `voice_queue.py` - Manages message pacing and priority
- `dynamic_message_templates.py` - Voice-friendly templates
- `dynamic_hook_logger.py` - Captures Claude Code operations

See the [Claude Code Integration Guide](CLAUDE_CODE_INTEGRATION.md) for complete details.

## Hook System Integration

### Direct Hook Integration

You can also integrate the speak command directly into other hook systems:

```bash
#!/bin/bash
# git pre-commit hook with voice

speak "Running pre-commit checks"

if ! npm test; then
    speak "Tests failed. Commit aborted."
    exit 1
fi

if ! npm run lint; then
    speak "Linting failed. Please fix errors."
    exit 1
fi

speak "Pre-commit checks passed"
```

### Webhook Handler

```python
#!/usr/bin/env python3
# webhook_handler.py - Handle webhooks with voice notifications

from flask import Flask, request
import subprocess

app = Flask(__name__)

def speak(message):
    subprocess.run(["speak", message])

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    data = request.json
    
    if data.get('event') == 'deployment':
        if data.get('status') == 'success':
            speak(f"Deployment to {data.get('environment')} successful")
        else:
            speak(f"Deployment failed: {data.get('error')}")
    
    return {'status': 'ok'}

if __name__ == '__main__':
    app.run(port=5000)
```

## Error Handling

### Robust Notification Function

```bash
#!/bin/bash
# robust_speak.sh - Speak with fallback options

notify() {
    local message="$1"
    local priority="${2:-normal}"  # normal, high, critical
    
    # Try speak command
    if command -v speak >/dev/null 2>&1; then
        case "$priority" in
            high)
                speak --provider elevenlabs "$message" || \
                speak --provider pyttsx3 "$message" || \
                echo "NOTIFICATION: $message"
                ;;
            critical)
                # Try all methods for critical messages
                speak "$message"
                notify-send "Critical Alert" "$message" 2>/dev/null || true
                echo "CRITICAL: $message" | tee -a /var/log/alerts.log
                ;;
            *)
                speak "$message" || echo "NOTIFICATION: $message"
                ;;
        esac
    else
        # Fallback if speak not available
        echo "NOTIFICATION: $message"
        command -v notify-send >/dev/null 2>&1 && \
            notify-send "Script Notification" "$message"
    fi
}

# Usage
notify "Starting backup" normal
notify "Disk space low" high
notify "System failure" critical
```

## Performance Considerations

### Batch Operations

```bash
#!/bin/bash
# batch_processor.sh - Efficient notification for batch ops

# Disable TTS during batch operations
export TTS_ENABLED_SAVE="$TTS_ENABLED"
export TTS_ENABLED=false

speak "Starting batch processing of 1000 items"

# Process items silently
for i in {1..1000}; do
    process_item "$i"
done

# Re-enable and notify
export TTS_ENABLED="$TTS_ENABLED_SAVE"
speak "Batch processing complete"
```

### Rate Limiting

```bash
#!/bin/bash
# rate_limited_speak.sh - Prevent notification spam

last_speak=0
min_interval=5  # Minimum seconds between notifications

rate_limited_speak() {
    local message="$1"
    local current_time=$(date +%s)
    
    if (( current_time - last_speak >= min_interval )); then
        speak "$message"
        last_speak=$current_time
    else
        # Queue or log instead
        echo "$(date): $message" >> speak_queue.log
    fi
}

# Usage in monitoring loop
while true; do
    if check_condition; then
        rate_limited_speak "Condition detected"
    fi
    sleep 1
done
```

### Async Notifications

```bash
#!/bin/bash
# async_speak.sh - Non-blocking notifications

speak_async() {
    local message="$1"
    
    # Run in background with timeout
    (
        timeout 5s speak "$message" 2>/dev/null || \
        echo "$(date): TTS timeout: $message" >> speak_errors.log
    ) &
}

# Usage
speak_async "Starting long operation"

# Do work without waiting for TTS
perform_long_operation

# Wait for any pending notifications before exit
wait
```

## Best Practices

1. **Silent Success**: Only notify on errors or important events
2. **Rate Limiting**: Prevent notification spam in loops
3. **Priority Levels**: Use different providers for different priorities
4. **Fallback Methods**: Always have alternative notification methods
5. **Context Awareness**: Detect if running interactively vs cron/CI
6. **Error Handling**: Don't let TTS failures break your scripts
7. **Performance**: Use `--off` flag in performance-critical sections
8. **Testing**: Test with TTS disabled to ensure scripts work without it