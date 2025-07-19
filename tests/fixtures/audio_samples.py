"""
Mock audio samples for testing purposes.
"""

# Minimal MP3 header for testing
MP3_HEADER = b'\xff\xfb\x90\x00'

# Mock audio content for different tests
MOCK_AUDIO_SAMPLES = {
    "rachel_sample": MP3_HEADER + b'\x00' * 1000,
    "domi_sample": MP3_HEADER + b'\x01' * 1000,
    "bella_sample": MP3_HEADER + b'\x02' * 1000,
    "custom_sample": MP3_HEADER + b'\x03' * 1000,
    "long_sample": MP3_HEADER + b'\x04' * 5000,
    "short_sample": MP3_HEADER + b'\x05' * 100,
}

def get_mock_audio_sample(voice_name: str = "default") -> bytes:
    """Get a mock audio sample for testing."""
    return MOCK_AUDIO_SAMPLES.get(f"{voice_name}_sample", MOCK_AUDIO_SAMPLES["rachel_sample"])