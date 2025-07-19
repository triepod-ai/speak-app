#!/usr/bin/env python3
"""
TTS Cache Manager - Reduce API costs by caching frequently used phrases
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


class TTSCache:
    """Manages cached TTS audio files to reduce API calls and costs."""
    
    def __init__(self, cache_dir: str = "~/.cache/speak-app"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "phrase_cache.json"
        self.cache: Dict[str, Any] = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cache metadata from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}", file=sys.stderr)
                self.cache = {}
    
    def get_cache_key(self, text: str, provider: str, voice: Optional[str] = None) -> str:
        """Generate unique key for text+provider+voice combination."""
        # Normalize text for consistent caching
        normalized_text = text.lower().strip()
        
        # Build key components
        key_parts = [normalized_text, provider]
        if voice:
            key_parts.append(voice)
        
        # Create MD5 hash for filename safety
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_audio_path(self, text: str, provider: str, voice: Optional[str] = None) -> Optional[str]:
        """Check if we have cached audio for this text."""
        key = self.get_cache_key(text, provider, voice)
        
        if key in self.cache:
            audio_file = self.cache_dir / f"{key}.mp3"
            if audio_file.exists():
                # Update last accessed time
                self.cache[key]['last_accessed'] = datetime.now().isoformat()
                self.cache[key]['hit_count'] = self.cache[key].get('hit_count', 0) + 1
                self.save_cache()
                return str(audio_file)
            else:
                # Cache entry exists but file is missing - clean up
                del self.cache[key]
                self.save_cache()
        
        return None
    
    def save_audio(self, text: str, provider: str, audio_data: bytes, 
                   voice: Optional[str] = None) -> str:
        """Cache audio data for future use."""
        key = self.get_cache_key(text, provider, voice)
        audio_file = self.cache_dir / f"{key}.mp3"
        
        # Save audio file
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        
        # Update cache metadata
        self.cache[key] = {
            'text': text[:100],  # Store first 100 chars for reference
            'text_length': len(text),
            'provider': provider,
            'voice': voice,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'file_size': len(audio_data),
            'hit_count': 0
        }
        self.save_cache()
        
        return str(audio_file)
    
    def play_cached_audio(self, audio_path: str) -> bool:
        """Play cached audio file."""
        try:
            # Use mpv if available, fallback to other players
            players = ['mpv', 'mplayer', 'afplay', 'ffplay']
            
            for player in players:
                if subprocess.run(['which', player], capture_output=True).returncode == 0:
                    subprocess.run([player, audio_path], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
                    return True
            
            print(f"No audio player found. Cached at: {audio_path}", file=sys.stderr)
            return False
            
        except Exception as e:
            print(f"Error playing cached audio: {e}", file=sys.stderr)
            return False
    
    def save_cache(self):
        """Save cache metadata to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}", file=sys.stderr)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        total_size = 0
        total_hits = 0
        provider_stats = {}
        
        for key, data in self.cache.items():
            provider = data['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'count': 0,
                    'size': 0,
                    'hits': 0
                }
            
            provider_stats[provider]['count'] += 1
            provider_stats[provider]['size'] += data.get('file_size', 0)
            provider_stats[provider]['hits'] += data.get('hit_count', 0)
            
            total_size += data.get('file_size', 0)
            total_hits += data.get('hit_count', 0)
        
        return {
            'total_entries': len(self.cache),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'total_hits': total_hits,
            'providers': provider_stats,
            'cache_directory': str(self.cache_dir)
        }
    
    def cleanup_old_cache(self, days: int = 30) -> int:
        """Remove cached files older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []
        removed_size = 0
        
        for key, data in self.cache.items():
            last_accessed = datetime.fromisoformat(data['last_accessed'])
            if last_accessed < cutoff:
                audio_file = self.cache_dir / f"{key}.mp3"
                if audio_file.exists():
                    removed_size += audio_file.stat().st_size
                    audio_file.unlink()
                to_remove.append(key)
        
        for key in to_remove:
            del self.cache[key]
        
        if to_remove:
            self.save_cache()
        
        return len(to_remove)
    
    def precache_common_phrases(self):
        """Precache common developer phrases to save API calls."""
        common_phrases = [
            "Build complete",
            "Build failed",
            "Tests passed",
            "Tests failed",
            "Deployment successful",
            "Deployment failed",
            "Error detected",
            "Warning",
            "Process complete",
            "Task finished",
            "Ready",
            "Done",
            "Success",
            "Failed"
        ]
        
        # This would need to be integrated with the TTS providers
        # For now, just return the list
        return common_phrases


# Simple test
if __name__ == "__main__":
    import sys
    cache = TTSCache()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        stats = cache.get_cache_stats()
        print(json.dumps(stats, indent=2))
    else:
        print("TTS Cache Manager")
        print(f"Cache directory: {cache.cache_dir}")
        print(f"Cache entries: {len(cache.cache)}")