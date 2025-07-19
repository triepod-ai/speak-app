#!/usr/bin/env python3
"""
Demo of the batch TTS processor showing the cost savings potential
"""

import os
from pathlib import Path

def demo_batch_processing():
    """Demonstrate the batch processing capabilities and cost savings."""
    
    print("🎙️  Batch TTS Demo - Cost Savings Analysis")
    print("=" * 60)
    
    # Read the common notifications file
    common_file = Path("common_notifications.txt")
    if common_file.exists():
        with open(common_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        lines = ["Build complete", "Tests passed", "Deploy successful"]
    
    print(f"\n📋 Processing {len(lines)} common notifications:")
    print("-" * 40)
    
    total_chars = 0
    for i, line in enumerate(lines[:10], 1):  # Show first 10
        chars = len(line)
        total_chars += chars
        print(f"{i:2d}. {line:<25} ({chars:2d} chars)")
    
    if len(lines) > 10:
        remaining = len(lines) - 10
        remaining_chars = sum(len(line) for line in lines[10:])
        total_chars += remaining_chars
        print(f"... and {remaining} more ({remaining_chars} chars)")
    
    print("-" * 40)
    print(f"Total characters: {total_chars:,}")
    
    # Cost calculations
    print(f"\n💰 Cost Analysis:")
    print(f"{'Provider':<12} {'Cost':<10} {'Monthly*':<12}")
    print("-" * 35)
    
    # Cost per 1000 chars
    costs = {
        'ElevenLabs': 0.33,
        'OpenAI': 0.015,
        'pyttsx3': 0.0
    }
    
    for provider, cost_per_1k in costs.items():
        daily_cost = (total_chars / 1000) * cost_per_1k
        monthly_cost = daily_cost * 30
        print(f"{provider:<12} ${daily_cost:<9.4f} ${monthly_cost:<11.2f}")
    
    print("\n*If generated daily for 30 days")
    
    # Show savings
    elevenlabs_cost = (total_chars / 1000) * 0.33
    openai_cost = (total_chars / 1000) * 0.015
    savings = elevenlabs_cost - openai_cost
    savings_pct = (savings / elevenlabs_cost) * 100 if elevenlabs_cost > 0 else 0
    
    print(f"\n🚀 Savings by switching to OpenAI:")
    print(f"   Per batch: ${savings:.4f} ({savings_pct:.1f}% savings)")
    print(f"   Monthly: ${savings * 30:.2f}")
    print(f"   Annual: ${savings * 365:.2f}")
    
    # Cache benefits
    print(f"\n💾 With 80% cache hit rate:")
    cached_cost = openai_cost * 0.2  # Only 20% needs generation
    cache_savings = openai_cost - cached_cost
    print(f"   Generation needed: {total_chars * 0.2:.0f} chars")
    print(f"   Cost with cache: ${cached_cost:.4f}")
    print(f"   Cache savings: ${cache_savings:.4f}")
    print(f"   Monthly with cache: ${cached_cost * 30:.2f}")
    
    # Show file structure
    print(f"\n📂 Generated file structure:")
    print(f"   tts_output/")
    for i, line in enumerate(lines[:3], 1):
        safe_name = line.lower().replace(' ', '_')[:20]
        print(f"   ├── {i:03d}_{safe_name}.mp3")
    print(f"   ├── ... ({len(lines) - 3} more files)")
    print(f"   └── manifest.json")
    
    print(f"\n✨ Benefits of batch processing:")
    print(f"   • Pre-generate common notifications once")
    print(f"   • Reuse cached audio files (zero cost)")
    print(f"   • Organized output with manifest")
    print(f"   • 95%+ cost savings vs ElevenLabs")
    print(f"   • Integration-ready for applications")
    
    print(f"\n🔧 To use with valid API key:")
    print(f"   export OPENAI_API_KEY='your-key-here'")
    print(f"   ./speak-batch --common")
    
    return total_chars, len(lines)

if __name__ == "__main__":
    demo_batch_processing()