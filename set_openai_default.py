#!/usr/bin/env python3
"""
Set OpenAI as the default TTS provider to save costs
"""

import os
import sys
from pathlib import Path

def update_bashrc():
    """Add OpenAI configuration to .bashrc or .bash_aliases"""
    
    # Check which file to update
    bashrc = Path.home() / ".bashrc"
    bash_aliases = Path.home() / ".bash_aliases"
    
    # Prefer .bash_aliases if it exists
    config_file = bash_aliases if bash_aliases.exists() else bashrc
    
    # Configuration to add
    config_lines = [
        "\n# Speak app cost optimization - Use OpenAI by default",
        "export TTS_PROVIDER=openai  # 22x cheaper than ElevenLabs",
        "export OPENAI_TTS_VOICE=onyx  # Best voice for notifications",
        "export OPENAI_TTS_MODEL=tts-1  # Use tts-1-hd only when needed",
        "\n"
    ]
    
    # Check if already configured
    if config_file.exists():
        content = config_file.read_text()
        if "TTS_PROVIDER=openai" in content:
            print("✅ OpenAI TTS is already configured as default")
            return True
    
    # Ask for confirmation
    print("🎙️  Speak App Cost Optimization")
    print("=" * 50)
    print("\nThis will set OpenAI as your default TTS provider.")
    print("\nBenefits:")
    print("  • 95% cost reduction vs ElevenLabs")
    print("  • High-quality neural voices")
    print("  • $0.15/day instead of $3.30/day")
    print(f"\nWill add configuration to: {config_file}")
    
    response = input("\nProceed? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return False
    
    # Add configuration
    try:
        with open(config_file, 'a') as f:
            f.writelines(line + '\n' for line in config_lines)
        
        print("\n✅ Configuration added successfully!")
        print("\nTo activate immediately, run:")
        print(f"  source {config_file}")
        print("\nOr open a new terminal.")
        
        # Show test command
        print("\nTest with:")
        print('  speak "Hello from OpenAI"')
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error updating {config_file}: {e}")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("\nTo use OpenAI TTS, add to your ~/.bash_aliases:")
        print('  export OPENAI_API_KEY="your-key-here"')
        print("\nGet your API key from: https://platform.openai.com/api-keys")
        return False
    return True

def show_cost_comparison():
    """Show cost comparison"""
    print("\n💰 Cost Comparison (10,000 chars/day):")
    print("  ElevenLabs: $3.30/day ($99/month)")
    print("  OpenAI:     $0.15/day ($4.50/month)")
    print("  Savings:    $3.15/day ($94.50/month) - 95%!")

if __name__ == "__main__":
    print("🚀 OpenAI TTS Setup Tool\n")
    
    # Update configuration
    if update_bashrc():
        check_api_key()
        show_cost_comparison()
        
        print("\n📌 Next steps:")
        print("  1. Set your OpenAI API key (if not already set)")
        print("  2. Source your shell config or open new terminal")
        print("  3. Test with: speak 'Cost optimized!'")
        print("\n✨ Enjoy 95% cost savings on TTS!")
    else:
        print("\nSetup cancelled.")
        sys.exit(1)