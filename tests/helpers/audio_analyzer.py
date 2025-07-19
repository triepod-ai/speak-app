#!/usr/bin/env python3
"""
Audio analysis utilities for voice testing.
Provides tools to validate and analyze audio files programmatically.
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import subprocess
import wave
import struct


class AudioAnalyzer:
    """Analyze audio files for testing purposes."""
    
    def __init__(self):
        """Initialize audio analyzer."""
        self.supported_formats = ['.mp3', '.wav', '.ogg', '.flac']
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an audio file and extract basic properties.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results and properties
        """
        result = {
            'valid': False,
            'exists': False,
            'format': None,
            'size_bytes': 0,
            'error': None
        }
        
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            result['error'] = 'File does not exist'
            return result
        
        result['exists'] = True
        
        # Check file size
        result['size_bytes'] = path.stat().st_size
        if result['size_bytes'] == 0:
            result['error'] = 'File is empty'
            return result
        
        # Check format
        suffix = path.suffix.lower()
        if suffix not in self.supported_formats:
            result['error'] = f'Unsupported format: {suffix}'
            return result
        
        result['format'] = suffix
        result['valid'] = True
        
        return result
    
    def get_audio_duration(self, file_path: str) -> Optional[float]:
        """
        Get audio duration in seconds using ffprobe if available.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds or None if unable to determine
        """
        try:
            # Check if ffprobe is available
            result = subprocess.run(['which', 'ffprobe'], capture_output=True)
            if result.returncode != 0:
                return None
            
            # Use ffprobe to get duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
            
        except Exception:
            pass
        
        return None
    
    def detect_silence(self, file_path: str, threshold_db: float = -40.0) -> Dict[str, Any]:
        """
        Detect if audio file contains mostly silence.
        
        Args:
            file_path: Path to audio file
            threshold_db: Silence threshold in dB
            
        Returns:
            Dictionary with silence detection results
        """
        result = {
            'has_content': True,
            'silence_ratio': 0.0,
            'error': None
        }
        
        try:
            # Check if ffmpeg is available
            ffmpeg_check = subprocess.run(['which', 'ffmpeg'], capture_output=True)
            if ffmpeg_check.returncode != 0:
                result['error'] = 'ffmpeg not available'
                return result
            
            # Use ffmpeg to detect silence
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-af', f'silencedetect=noise={threshold_db}dB:d=0.5',
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            output = process.stdout
            
            # Parse silence detection output
            silence_count = output.count('silence_start')
            if silence_count > 0:
                # Rough estimate of silence ratio
                duration = self.get_audio_duration(file_path)
                if duration and duration > 0:
                    # Count silence duration mentions
                    import re
                    silence_durations = re.findall(r'silence_duration: ([\d.]+)', output)
                    total_silence = sum(float(d) for d in silence_durations)
                    result['silence_ratio'] = min(total_silence / duration, 1.0)
                    result['has_content'] = result['silence_ratio'] < 0.9
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def estimate_text_to_duration_ratio(self, text: str, duration: float) -> float:
        """
        Estimate if audio duration is reasonable for given text.
        
        Args:
            text: Source text
            duration: Audio duration in seconds
            
        Returns:
            Ratio of actual to expected duration (1.0 is perfect)
        """
        # Rough estimate: 150 words per minute average speech
        words = len(text.split())
        expected_duration = (words / 150) * 60  # Convert to seconds
        
        if expected_duration == 0:
            return 0.0
        
        return duration / expected_duration
    
    def calculate_audio_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate hash of audio file for consistency checks.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            MD5 hash or None if error
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    def compare_audio_files(self, file1: str, file2: str) -> Dict[str, Any]:
        """
        Compare two audio files for testing.
        
        Args:
            file1: Path to first audio file
            file2: Path to second audio file
            
        Returns:
            Comparison results
        """
        result = {
            'identical': False,
            'same_size': False,
            'same_duration': False,
            'size_diff_bytes': 0,
            'duration_diff_seconds': 0.0
        }
        
        # Validate both files
        val1 = self.validate_audio_file(file1)
        val2 = self.validate_audio_file(file2)
        
        if not (val1['valid'] and val2['valid']):
            result['error'] = 'One or both files are invalid'
            return result
        
        # Compare sizes
        result['same_size'] = val1['size_bytes'] == val2['size_bytes']
        result['size_diff_bytes'] = abs(val1['size_bytes'] - val2['size_bytes'])
        
        # Compare hashes
        hash1 = self.calculate_audio_hash(file1)
        hash2 = self.calculate_audio_hash(file2)
        result['identical'] = hash1 == hash2 and hash1 is not None
        
        # Compare durations
        dur1 = self.get_audio_duration(file1)
        dur2 = self.get_audio_duration(file2)
        
        if dur1 is not None and dur2 is not None:
            result['same_duration'] = abs(dur1 - dur2) < 0.1  # Within 100ms
            result['duration_diff_seconds'] = abs(dur1 - dur2)
        
        return result
    
    def analyze_audio_characteristics(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze audio characteristics using available tools.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio characteristics
        """
        chars = {
            'duration': None,
            'size_bytes': 0,
            'format': None,
            'bitrate': None,
            'sample_rate': None,
            'channels': None,
            'has_content': True
        }
        
        # Basic validation
        validation = self.validate_audio_file(file_path)
        if not validation['valid']:
            chars['error'] = validation['error']
            return chars
        
        chars['size_bytes'] = validation['size_bytes']
        chars['format'] = validation['format']
        
        # Get duration
        chars['duration'] = self.get_audio_duration(file_path)
        
        # Detect silence
        silence = self.detect_silence(file_path)
        chars['has_content'] = silence['has_content']
        chars['silence_ratio'] = silence.get('silence_ratio', 0.0)
        
        # Try to get detailed info with ffprobe
        try:
            if subprocess.run(['which', 'ffprobe'], capture_output=True).returncode == 0:
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    file_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    
                    # Extract format info
                    if 'format' in data:
                        fmt = data['format']
                        chars['bitrate'] = int(fmt.get('bit_rate', 0))
                        
                    # Extract stream info
                    if 'streams' in data:
                        for stream in data['streams']:
                            if stream.get('codec_type') == 'audio':
                                chars['sample_rate'] = int(stream.get('sample_rate', 0))
                                chars['channels'] = int(stream.get('channels', 0))
                                break
        except Exception:
            pass
        
        return chars
    
    def create_test_audio_file(self, duration: float = 1.0, frequency: int = 440) -> str:
        """
        Create a simple test audio file (WAV format).
        
        Args:
            duration: Duration in seconds
            frequency: Frequency in Hz (440 = A4 note)
            
        Returns:
            Path to created file
        """
        import math
        
        # Audio parameters
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        # Generate sine wave
        samples = []
        for i in range(num_samples):
            sample = 32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate)
            samples.append(int(sample))
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        # Write WAV file
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Pack samples as binary data
            packed_samples = struct.pack('<%dh' % num_samples, *samples)
            wav_file.writeframes(packed_samples)
        
        return temp_path


# Helper functions for common test scenarios
def validate_tts_output(audio_path: str, source_text: str, 
                       min_duration: float = 0.5,
                       max_duration: float = 60.0) -> Tuple[bool, str]:
    """
    Validate TTS output for common issues.
    
    Args:
        audio_path: Path to generated audio
        source_text: Original text that was synthesized
        min_duration: Minimum expected duration
        max_duration: Maximum expected duration
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    analyzer = AudioAnalyzer()
    
    # Validate file
    validation = analyzer.validate_audio_file(audio_path)
    if not validation['valid']:
        return False, f"Invalid audio file: {validation['error']}"
    
    # Check file size (should be reasonable)
    if validation['size_bytes'] < 1000:  # Less than 1KB is suspicious
        return False, "Audio file too small"
    
    if validation['size_bytes'] > 10 * 1024 * 1024:  # More than 10MB is suspicious
        return False, "Audio file too large"
    
    # Check duration
    duration = analyzer.get_audio_duration(audio_path)
    if duration is not None:
        if duration < min_duration:
            return False, f"Audio too short: {duration:.1f}s"
        if duration > max_duration:
            return False, f"Audio too long: {duration:.1f}s"
        
        # Check text-to-duration ratio
        ratio = analyzer.estimate_text_to_duration_ratio(source_text, duration)
        if ratio < 0.5 or ratio > 2.0:
            return False, f"Unusual text-to-duration ratio: {ratio:.2f}"
    
    # Check for silence
    silence = analyzer.detect_silence(audio_path)
    if not silence['has_content']:
        return False, "Audio appears to be mostly silence"
    
    return True, "Valid"


def compare_voice_outputs(audio_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Compare multiple voice outputs.
    
    Args:
        audio_files: Dictionary mapping voice names to audio file paths
        
    Returns:
        Comparison results
    """
    analyzer = AudioAnalyzer()
    results = {
        'voices': {},
        'comparisons': {},
        'summary': {}
    }
    
    # Analyze each voice
    for voice_name, audio_path in audio_files.items():
        chars = analyzer.analyze_audio_characteristics(audio_path)
        results['voices'][voice_name] = chars
    
    # Compare pairs
    voice_names = list(audio_files.keys())
    for i in range(len(voice_names)):
        for j in range(i + 1, len(voice_names)):
            voice1, voice2 = voice_names[i], voice_names[j]
            comparison = analyzer.compare_audio_files(
                audio_files[voice1],
                audio_files[voice2]
            )
            results['comparisons'][f"{voice1}_vs_{voice2}"] = comparison
    
    # Summary statistics
    if results['voices']:
        durations = [v['duration'] for v in results['voices'].values() if v.get('duration')]
        sizes = [v['size_bytes'] for v in results['voices'].values()]
        
        if durations:
            results['summary']['avg_duration'] = sum(durations) / len(durations)
            results['summary']['duration_variance'] = max(durations) - min(durations)
        
        if sizes:
            results['summary']['avg_size'] = sum(sizes) / len(sizes)
            results['summary']['size_variance'] = max(sizes) - min(sizes)
    
    return results


if __name__ == "__main__":
    # Simple test
    analyzer = AudioAnalyzer()
    
    # Create a test audio file
    test_file = analyzer.create_test_audio_file(duration=2.0)
    print(f"Created test audio: {test_file}")
    
    # Analyze it
    chars = analyzer.analyze_audio_characteristics(test_file)
    print(f"Characteristics: {json.dumps(chars, indent=2)}")
    
    # Validate it
    is_valid, msg = validate_tts_output(test_file, "Test audio content")
    print(f"Validation: {is_valid} - {msg}")
    
    # Cleanup
    os.unlink(test_file)