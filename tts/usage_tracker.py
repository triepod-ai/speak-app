#!/usr/bin/env python3
"""
Usage tracking for TTS to monitor costs and usage patterns
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class UsageTracker:
    """Track TTS usage for cost monitoring and optimization."""
    
    # Cost per 1000 characters
    PROVIDER_COSTS = {
        'elevenlabs': 0.33,
        'openai': 0.015,  # tts-1 standard
        'openai-hd': 0.030,  # tts-1-hd
        'pyttsx3': 0.0,
        'azure': 0.004,  # For reference
        'google': 0.016,  # For reference
    }
    
    def __init__(self, data_dir: str = "~/.local/share/speak-app"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.data_dir / "usage_stats.json"
        self.daily_file = self.data_dir / "daily_usage.txt"
        self.load_stats()
    
    def load_stats(self):
        """Load usage statistics from disk."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = self._empty_stats()
        else:
            self.stats = self._empty_stats()
    
    def _empty_stats(self) -> Dict:
        """Create empty stats structure."""
        return {
            'providers': {},
            'daily_usage': {},
            'monthly_usage': {},
            'total_characters': 0,
            'total_cost_estimate': 0.0,
            'start_date': datetime.now().isoformat(),
            'cache_hits': 0,
            'cache_savings': 0.0
        }
    
    def track_usage(self, provider: str, text: str, 
                   cached: bool = False, model: Optional[str] = None) -> Tuple[float, Dict]:
        """
        Track TTS usage for cost monitoring.
        
        Returns:
            Tuple of (cost, usage_info)
        """
        char_count = len(text)
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        
        # If cached, just track the cache hit
        if cached:
            self.stats['cache_hits'] += 1
            # Calculate savings based on what would have been used
            saved_cost = self.estimate_cost(provider, char_count, model)
            self.stats['cache_savings'] += saved_cost
            self.save_stats()
            return 0.0, {'cached': True, 'saved': saved_cost}
        
        # Update provider stats
        if provider not in self.stats['providers']:
            self.stats['providers'][provider] = {
                'total_characters': 0,
                'total_requests': 0,
                'total_cost': 0.0,
                'models': {}
            }
        
        self.stats['providers'][provider]['total_characters'] += char_count
        self.stats['providers'][provider]['total_requests'] += 1
        
        # Track model usage for OpenAI
        if model and provider == 'openai':
            if model not in self.stats['providers'][provider]['models']:
                self.stats['providers'][provider]['models'][model] = 0
            self.stats['providers'][provider]['models'][model] += char_count
        
        # Update daily usage
        if today not in self.stats['daily_usage']:
            self.stats['daily_usage'][today] = {
                'total': 0,
                'providers': {}
            }
        
        self.stats['daily_usage'][today]['total'] += char_count
        
        if provider not in self.stats['daily_usage'][today]['providers']:
            self.stats['daily_usage'][today]['providers'][provider] = 0
        self.stats['daily_usage'][today]['providers'][provider] += char_count
        
        # Update monthly usage
        if month not in self.stats['monthly_usage']:
            self.stats['monthly_usage'][month] = {
                'total': 0,
                'providers': {},
                'cost': 0.0
            }
        
        self.stats['monthly_usage'][month]['total'] += char_count
        if provider not in self.stats['monthly_usage'][month]['providers']:
            self.stats['monthly_usage'][month]['providers'][provider] = 0
        self.stats['monthly_usage'][month]['providers'][provider] += char_count
        
        # Calculate and track costs
        cost = self.estimate_cost(provider, char_count, model)
        self.stats['providers'][provider]['total_cost'] += cost
        self.stats['total_cost_estimate'] += cost
        self.stats['monthly_usage'][month]['cost'] += cost
        
        # Update totals
        self.stats['total_characters'] += char_count
        
        # Update daily usage file for quick checks
        self._update_daily_file(char_count)
        
        self.save_stats()
        
        usage_info = {
            'characters': char_count,
            'cost': cost,
            'daily_total': self.stats['daily_usage'][today]['total'],
            'monthly_total': self.stats['monthly_usage'][month]['total'],
            'monthly_cost': self.stats['monthly_usage'][month]['cost']
        }
        
        return cost, usage_info
    
    def estimate_cost(self, provider: str, char_count: int, 
                     model: Optional[str] = None) -> float:
        """Estimate cost based on provider pricing."""
        # Handle OpenAI model variants
        if provider == 'openai' and model == 'tts-1-hd':
            cost_per_1k = self.PROVIDER_COSTS.get('openai-hd', 0.03)
        else:
            cost_per_1k = self.PROVIDER_COSTS.get(provider, 0.0)
        
        return (char_count / 1000.0) * cost_per_1k
    
    def _update_daily_file(self, chars: int):
        """Update simple daily usage file for quick shell checks."""
        today = datetime.now().strftime('%Y-%m-%d')
        current_total = self.stats['daily_usage'][today]['total']
        
        try:
            with open(self.daily_file, 'w') as f:
                f.write(str(current_total))
        except Exception:
            pass
    
    def get_daily_usage(self, date: Optional[str] = None) -> Dict:
        """Get usage for a specific day."""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.stats['daily_usage'].get(date, {
            'total': 0,
            'providers': {}
        })
    
    def get_monthly_usage(self, month: Optional[str] = None) -> Dict:
        """Get usage for a specific month."""
        if not month:
            month = datetime.now().strftime('%Y-%m')
        
        return self.stats['monthly_usage'].get(month, {
            'total': 0,
            'providers': {},
            'cost': 0.0
        })
    
    def get_provider_breakdown(self) -> Dict:
        """Get breakdown by provider."""
        breakdown = {}
        
        for provider, data in self.stats['providers'].items():
            breakdown[provider] = {
                'characters': data['total_characters'],
                'requests': data['total_requests'],
                'cost': data['total_cost'],
                'avg_length': data['total_characters'] / max(data['total_requests'], 1),
                'percentage': (data['total_characters'] / max(self.stats['total_characters'], 1)) * 100
            }
        
        return breakdown
    
    def get_cost_report(self) -> str:
        """Generate a cost report."""
        report = []
        report.append("📊 TTS Usage Report")
        report.append("=" * 50)
        
        # Overall stats
        report.append(f"\n📈 Total Usage:")
        report.append(f"  Characters: {self.stats['total_characters']:,}")
        report.append(f"  Cost: ${self.stats['total_cost_estimate']:.2f}")
        report.append(f"  Cache Hits: {self.stats['cache_hits']:,}")
        report.append(f"  Cache Savings: ${self.stats['cache_savings']:.2f}")
        
        # This month
        current_month = datetime.now().strftime('%Y-%m')
        month_data = self.get_monthly_usage(current_month)
        report.append(f"\n📅 This Month ({current_month}):")
        report.append(f"  Characters: {month_data['total']:,}")
        report.append(f"  Cost: ${month_data['cost']:.2f}")
        
        # Today
        today = datetime.now().strftime('%Y-%m-%d')
        today_data = self.get_daily_usage(today)
        report.append(f"\n📆 Today ({today}):")
        report.append(f"  Characters: {today_data['total']:,}")
        
        # Provider breakdown
        report.append("\n🎯 Provider Breakdown:")
        breakdown = self.get_provider_breakdown()
        for provider, data in sorted(breakdown.items(), 
                                   key=lambda x: x[1]['characters'], 
                                   reverse=True):
            report.append(f"\n  {provider}:")
            report.append(f"    Characters: {data['characters']:,} ({data['percentage']:.1f}%)")
            report.append(f"    Requests: {data['requests']:,}")
            report.append(f"    Avg Length: {data['avg_length']:.0f} chars")
            report.append(f"    Cost: ${data['cost']:.2f}")
        
        # Recommendations
        report.append("\n💡 Recommendations:")
        if 'elevenlabs' in breakdown and breakdown['elevenlabs']['percentage'] > 10:
            savings = breakdown['elevenlabs']['cost'] * 0.95
            report.append(f"  • Switch ElevenLabs to OpenAI: Save ${savings:.2f}")
        
        if self.stats['cache_hits'] < self.stats['total_characters'] * 0.2:
            report.append("  • Enable caching for common phrases")
        
        return '\n'.join(report)
    
    def save_stats(self):
        """Save statistics to disk."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.stats, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Warning: Could not save usage stats: {e}", file=sys.stderr)


# CLI interface
if __name__ == "__main__":
    import sys
    
    tracker = UsageTracker()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--report":
            print(tracker.get_cost_report())
        elif sys.argv[1] == "--today":
            today = tracker.get_daily_usage()
            print(f"Today: {today['total']:,} characters")
        elif sys.argv[1] == "--month":
            month = tracker.get_monthly_usage()
            print(f"This month: {month['total']:,} characters (${month['cost']:.2f})")
    else:
        # Quick stats
        print(f"Total: {tracker.stats['total_characters']:,} chars")
        print(f"Cost: ${tracker.stats['total_cost_estimate']:.2f}")
        print(f"Saved: ${tracker.stats['cache_savings']:.2f}")