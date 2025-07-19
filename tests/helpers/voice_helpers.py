"""
Helper functions for voice testing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

def load_mock_voices() -> Dict:
    """Load mock voices from fixtures."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    voices_file = fixtures_dir / "voices.json"
    
    with open(voices_file, 'r') as f:
        return json.load(f)

def get_voice_by_id(voice_id: str) -> Optional[Dict]:
    """Get voice data by ID from mock voices."""
    voices_data = load_mock_voices()
    for voice in voices_data["voices"]:
        if voice["voice_id"] == voice_id:
            return voice
    return None

def get_voice_by_name(name: str) -> Optional[Dict]:
    """Get voice data by name from mock voices."""
    voices_data = load_mock_voices()
    for voice in voices_data["voices"]:
        if voice["name"] == name:
            return voice
    return None

def get_voices_by_category(category: str) -> List[Dict]:
    """Get voices by category from mock voices."""
    voices_data = load_mock_voices()
    return [voice for voice in voices_data["voices"] if voice["category"] == category]

def get_premade_voices() -> List[Dict]:
    """Get premade voices from mock voices."""
    return get_voices_by_category("premade")

def get_generated_voices() -> List[Dict]:
    """Get generated voices from mock voices."""
    return get_voices_by_category("generated")

def validate_voice_settings(stability: float, similarity_boost: float) -> bool:
    """Validate voice settings are within acceptable ranges."""
    return (0.0 <= stability <= 1.0) and (0.0 <= similarity_boost <= 1.0)

def create_test_voice_data(voice_id: str, name: str, category: str = "premade") -> Dict:
    """Create test voice data for testing."""
    return {
        "voice_id": voice_id,
        "name": name,
        "category": category,
        "fine_tuning": {
            "model_id": "eleven_turbo_v2_5",
            "is_allowed_to_fine_tune": True,
            "finetuning_state": "not_finetuned"
        },
        "labels": {
            "accent": "american",
            "description": "test",
            "age": "young",
            "gender": "neutral",
            "use_case": "testing"
        },
        "description": f"Test voice {name}",
        "preview_url": f"https://example.com/voices/{voice_id}/preview.mp3",
        "available_for_tiers": [],
        "settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

def extract_voice_ids(voices_data: Dict) -> List[str]:
    """Extract voice IDs from voices data."""
    return [voice["voice_id"] for voice in voices_data["voices"]]

def extract_voice_names(voices_data: Dict) -> List[str]:
    """Extract voice names from voices data."""
    return [voice["name"] for voice in voices_data["voices"]]

def is_valid_voice_id(voice_id: str) -> bool:
    """Check if a voice ID is valid (non-empty and reasonable length)."""
    return isinstance(voice_id, str) and len(voice_id) > 0 and len(voice_id) < 100

def is_valid_voice_name(name: str) -> bool:
    """Check if a voice name is valid."""
    return isinstance(name, str) and len(name) > 0 and len(name) < 50