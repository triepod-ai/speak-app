#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=1.0.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
Smart TTS message processor using AI to enhance notifications.
Takes raw hook messages and transforms them into concise, human-friendly TTS output.
"""

import os
import sys
import argparse
from pathlib import Path
try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

# Load environment
env_path = Path.home() / "brainpods" / ".env"
if env_path.exists():
    load_dotenv(env_path)

CONTEXT_PROMPTS = {
    "tool_notification": """Convert this tool notification into a brief, natural spoken message (max 12 words).
Rules: NO training data mentions, NO meta-commentary, NO apologies. Just state what happened clearly.
If the action is unclear or generic, say 'Operation completed' instead.""",

    "session_start": """Convert this to a brief greeting (max 8 words).
Rules: NO training data mentions, NO meta-commentary. Just a friendly hello.""",

    "session_end": """Convert this to a brief farewell (max 8 words).
Rules: NO training data mentions, NO meta-commentary. Just a friendly goodbye.""",

    "error": """Convert this error into a brief alert (max 10 words).
Rules: NO training data mentions, NO apologies. State what failed, nothing else.""",

    "success": """Convert this to a brief positive message (max 8 words).
Rules: NO training data mentions, NO meta-commentary. Just confirm success.""",

    "warning": """Convert this warning to a brief alert (max 10 words).
Rules: NO training data mentions. State the concern clearly.""",

    "progress": """Convert this to a brief status (max 8 words).
Rules: NO training data mentions. State current progress only.""",

    "agent": """Convert this agent notification to a brief update (max 10 words).
Rules: NO training data mentions. Include agent name if clear, otherwise say 'Agent completed task'.""",

    "default": """Convert this to a brief spoken message (max 10 words).
Rules: NO training data mentions, NO meta-commentary, NO apologies. Be direct and clear.
If unclear what happened, say 'Operation completed'."""
}

def enhance_message(text: str, context: str = "default") -> str:
    """
    Use OpenAI to enhance a message for TTS output with logging.

    Args:
        text: Raw message text
        context: Context type for prompting

    Returns:
        Enhanced message suitable for TTS (or None if should be skipped)
    """
    # Check for generic/unknown messages
    text_lower = text.lower()
    skip_phrases = [
        "unknown executed",
        "unknown tool",
        "unknown operation"
    ]

    for phrase in skip_phrases:
        if phrase in text_lower and len(text.split()) < 6:
            # Too generic, skip it
            return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: return truncated original if it's meaningful
        if "unknown" in text_lower and len(text.split()) < 5:
            return None
        return text[:100]

    try:
        client = OpenAI(api_key=api_key)
        prompt = CONTEXT_PROMPTS.get(context, CONTEXT_PROMPTS["default"])

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.3
        )

        enhanced = response.choices[0].message.content.strip()

        # Remove quotes if AI added them
        if enhanced.startswith('"') and enhanced.endswith('"'):
            enhanced = enhanced[1:-1]
        if enhanced.startswith("'") and enhanced.endswith("'"):
            enhanced = enhanced[1:-1]

        # Filter out bad responses
        enhanced_lower = enhanced.lower()
        bad_phrases = [
            "trained on data",
            "training data",
            "knowledge cutoff",
            "as of my last",
            "i don't have",
            "i cannot",
            "as an ai"
        ]

        for phrase in bad_phrases:
            if phrase in enhanced_lower:
                # AI misbehaved, use fallback
                return text[:100] if "unknown" not in text_lower else None

        return enhanced

    except Exception as e:
        # Fallback: return truncated original if meaningful
        print(f"Error enhancing message: {e}", file=sys.stderr)
        if "unknown" in text.lower() and len(text.split()) < 5:
            return None
        return text[:100]

def main():
    parser = argparse.ArgumentParser(description="Smart TTS message processor")
    parser.add_argument("--context", default="default",
                       help="Context type for message enhancement")
    parser.add_argument("--log", action="store_true",
                       help="Enable logging to file")
    parser.add_argument("text", nargs="+", help="Text to process")

    args = parser.parse_args()
    text = " ".join(args.text)

    if not text:
        return

    # Log input if requested
    if args.log:
        log_dir = Path("/tmp/tts-queue")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "smart-processor.log"

        with open(log_file, "a") as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] INPUT [{args.context}]: {text}\n")

    enhanced = enhance_message(text, args.context)

    if enhanced is None:
        # Message was filtered out
        if args.log:
            with open(log_dir / "smart-processor.log", "a") as f:
                f.write(f"[{timestamp}] FILTERED: Message skipped\n")
        sys.exit(1)  # Signal that message was filtered

    # Log output if requested
    if args.log:
        with open(log_dir / "smart-processor.log", "a") as f:
            f.write(f"[{timestamp}] OUTPUT: {enhanced}\n")

    print(enhanced)

if __name__ == "__main__":
    main()
