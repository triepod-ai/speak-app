#!/usr/bin/env python3
"""
Simple TTS queue system to prevent simultaneous speak commands.
Uses file-based locking for cross-process coordination.
"""

import os
import time
import fcntl
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Queue configuration
QUEUE_DIR = Path("/tmp/tts-queue")
LOCK_FILE = QUEUE_DIR / "tts.lock"
LOG_FILE = QUEUE_DIR / "tts-queue.log"
MAX_QUEUE_AGE = 30  # seconds

def init_queue():
    """Initialize queue directory and log file."""
    QUEUE_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("")

def log_message(message: str, level: str = "INFO"):
    """Log a message to the queue log file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
    except Exception:
        pass  # Silent failure for logging

def should_skip_message(text: str) -> tuple[bool, str]:
    """
    Determine if a message should be skipped.

    Returns:
        (should_skip, reason)
    """
    text_lower = text.lower()

    # Skip if contains training data mentions
    training_phrases = [
        "trained on data",
        "training data",
        "knowledge cutoff",
        "as of my last update",
        "i don't have",
        "i cannot",
        "as an ai"
    ]

    for phrase in training_phrases:
        if phrase in text_lower:
            return True, f"Contains training data mention: '{phrase}'"

    # Skip if just says "unknown"
    if text.strip().lower() in ["unknown", "unknown.", "unknown executed"]:
        return True, "Generic 'unknown' message"

    # Skip if contains "unknown executed in"
    if "unknown executed in" in text_lower:
        return True, "Generic 'unknown executed' message"

    # Skip if too short (less than 3 words)
    if len(text.split()) < 3:
        return True, "Too short (less than 3 words)"

    # Skip if empty or whitespace
    if not text or not text.strip():
        return True, "Empty or whitespace only"

    return False, ""

def enqueue_tts(text: str, priority: str = "normal", context: dict = None) -> bool:
    """
    Enqueue a TTS message with queue management.

    Args:
        text: Message text to speak
        priority: Priority level (low, normal, high, urgent)
        context: Optional context dict for logging

    Returns:
        True if queued/spoken, False if skipped
    """
    init_queue()

    # Check if message should be skipped
    should_skip, skip_reason = should_skip_message(text)
    if should_skip:
        log_message(f"SKIPPED: {text[:50]}... | Reason: {skip_reason}", "SKIP")
        return False

    # Log the enqueue attempt
    context_str = json.dumps(context) if context else "{}"
    log_message(f"ENQUEUE [{priority}]: {text} | Context: {context_str}", "ENQUEUE")

    # Acquire lock and speak
    try:
        with open(LOCK_FILE, "w") as lock:
            # Try to acquire lock with timeout
            start_time = time.time()
            timeout = 10  # seconds

            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        log_message(f"TIMEOUT waiting for lock: {text[:50]}...", "ERROR")
                        return False
                    time.sleep(0.1)

            # Execute speak command
            import subprocess

            log_message(f"SPEAKING [{priority}]: {text}", "SPEAK")

            result = subprocess.run(
                ["speak", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                text=True
            )

            if result.returncode == 0:
                log_message(f"SUCCESS: {text[:50]}...", "SUCCESS")
                return True
            else:
                error = result.stderr[:200] if result.stderr else "Unknown error"
                log_message(f"FAILED: {text[:50]}... | Error: {error}", "ERROR")
                return False

    except Exception as e:
        log_message(f"EXCEPTION: {text[:50]}... | Error: {str(e)[:200]}", "ERROR")
        return False

def get_queue_stats() -> dict:
    """Get queue statistics from log file."""
    try:
        if not LOG_FILE.exists():
            return {"total": 0, "spoken": 0, "skipped": 0, "errors": 0}

        lines = LOG_FILE.read_text().strip().split("\n")

        stats = {
            "total": len(lines),
            "spoken": sum(1 for line in lines if "[SPEAK]" in line),
            "skipped": sum(1 for line in lines if "[SKIP]" in line),
            "errors": sum(1 for line in lines if "[ERROR]" in line),
            "recent": lines[-10:] if len(lines) > 10 else lines
        }

        return stats
    except Exception:
        return {"error": "Failed to read stats"}

def clear_queue_log():
    """Clear the queue log file."""
    try:
        if LOG_FILE.exists():
            LOG_FILE.write_text("")
        log_message("Queue log cleared", "INFO")
        return True
    except Exception as e:
        log_message(f"Failed to clear log: {e}", "ERROR")
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            stats = get_queue_stats()
            print(json.dumps(stats, indent=2))
        elif sys.argv[1] == "--clear":
            clear_queue_log()
            print("Queue log cleared")
        elif sys.argv[1] == "--log":
            if LOG_FILE.exists():
                print(LOG_FILE.read_text())
            else:
                print("No log file found")
        else:
            # Enqueue the text
            text = " ".join(sys.argv[1:])
            success = enqueue_tts(text)
            sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  tts_queue.py <text>     - Enqueue and speak text")
        print("  tts_queue.py --stats    - Show queue statistics")
        print("  tts_queue.py --log      - Show queue log")
        print("  tts_queue.py --clear    - Clear queue log")
