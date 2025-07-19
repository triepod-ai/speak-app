#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=7.0.0",
#   "pytest-mock>=3.10.0",
#   "psutil>=5.9.0",
# ]
# ///

"""
Performance stress test suite for the speak app.
Tests high-frequency usage, resource utilization, and performance under load.
"""

import os
import sys
import time
import subprocess
import threading
import queue
import tempfile
import psutil
import statistics
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from tts.cache_manager import TTSCache
from tts.usage_tracker import UsageTracker


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu = self.process.cpu_percent()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitor loop for collecting performance data."""
        while self.monitoring:
            try:
                memory = self.process.memory_info().rss
                cpu = self.process.cpu_percent()
                self.memory_samples.append(memory)
                self.cpu_samples.append(cpu)
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def get_stats(self):
        """Get performance statistics."""
        if not self.memory_samples:
            return {}
        
        stats = {
            'memory_initial': self.initial_memory,
            'memory_peak': max(self.memory_samples),
            'memory_final': self.memory_samples[-1],
            'memory_mean': statistics.mean(self.memory_samples),
            'memory_increase': self.memory_samples[-1] - self.initial_memory,
            'samples': len(self.memory_samples),
            'cpu_samples': self.cpu_samples
        }
        
        if self.cpu_samples:
            stats['cpu_peak'] = max(self.cpu_samples)
            stats['cpu_mean'] = statistics.mean(self.cpu_samples)
        
        return stats


class TestHighFrequencyOperations:
    """Test high-frequency operations and rapid usage patterns."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        yield monitor
        monitor.stop_monitoring()
    
    def test_rapid_sequential_operations(self, speak_script_path, performance_monitor):
        """Test rapid sequential speak operations."""
        num_operations = 100
        start_time = time.time()
        
        for i in range(num_operations):
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Rapid test {i}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            assert result.returncode == 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 30  # Should complete within 30 seconds
        ops_per_second = num_operations / duration
        assert ops_per_second > 3  # At least 3 operations per second
        
        # Check memory usage
        stats = performance_monitor.get_stats()
        if stats:
            memory_increase_mb = stats['memory_increase'] / (1024 * 1024)
            assert memory_increase_mb < 50  # Less than 50MB increase
    
    def test_burst_operations(self, speak_script_path, performance_monitor):
        """Test burst operations with pauses."""
        burst_size = 20
        num_bursts = 5
        
        for burst in range(num_bursts):
            start_time = time.time()
            
            # Rapid burst
            for i in range(burst_size):
                result = subprocess.run(
                    [str(speak_script_path), "--off", f"Burst {burst}-{i}"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                assert result.returncode == 0
            
            burst_duration = time.time() - start_time
            assert burst_duration < 10  # Each burst should complete quickly
            
            # Brief pause between bursts
            time.sleep(0.5)
        
        # Check system stability
        stats = performance_monitor.get_stats()
        if stats:
            assert stats['memory_increase'] < 100 * 1024 * 1024  # Less than 100MB
    
    def test_concurrent_high_frequency(self, speak_script_path, performance_monitor):
        """Test concurrent high-frequency operations."""
        num_threads = 5
        ops_per_thread = 20
        results = queue.Queue()
        
        def worker(thread_id):
            thread_results = []
            for i in range(ops_per_thread):
                start_time = time.time()
                result = subprocess.run(
                    [str(speak_script_path), "--off", f"Thread {thread_id} op {i}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                duration = time.time() - start_time
                thread_results.append({
                    'returncode': result.returncode,
                    'duration': duration,
                    'thread_id': thread_id,
                    'op_id': i
                })
            results.put(thread_results)
        
        # Start all threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_duration = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results.empty():
            all_results.extend(results.get())
        
        # Verify all operations succeeded
        assert len(all_results) == num_threads * ops_per_thread
        success_count = sum(1 for r in all_results if r['returncode'] == 0)
        assert success_count == len(all_results)
        
        # Performance assertions
        assert total_duration < 60  # Should complete within 60 seconds
        durations = [r['duration'] for r in all_results]
        avg_duration = statistics.mean(durations)
        assert avg_duration < 2  # Average operation should be under 2 seconds
        
        # Check system stability
        stats = performance_monitor.get_stats()
        if stats:
            memory_increase_mb = stats['memory_increase'] / (1024 * 1024)
            assert memory_increase_mb < 100  # Less than 100MB increase
    
    def test_sustained_load(self, speak_script_path, performance_monitor):
        """Test sustained load over time."""
        duration_seconds = 30
        target_ops_per_second = 2
        
        operations = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            op_start = time.time()
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Sustained {len(operations)}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            op_duration = time.time() - op_start
            
            operations.append({
                'returncode': result.returncode,
                'duration': op_duration,
                'timestamp': time.time()
            })
            
            # Rate limiting
            if op_duration < 1.0 / target_ops_per_second:
                time.sleep(1.0 / target_ops_per_second - op_duration)
        
        # Verify operations
        success_count = sum(1 for op in operations if op['returncode'] == 0)
        assert success_count >= len(operations) * 0.95  # 95% success rate
        
        # Check rate
        actual_rate = len(operations) / duration_seconds
        assert actual_rate >= target_ops_per_second * 0.8  # Within 80% of target
        
        # Check system stability
        stats = performance_monitor.get_stats()
        if stats:
            memory_increase_mb = stats['memory_increase'] / (1024 * 1024)
            assert memory_increase_mb < 75  # Less than 75MB increase
    
    def test_memory_leak_detection(self, speak_script_path):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        memory_samples = []
        num_operations = 50
        
        for i in range(num_operations):
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Memory test {i}"],
                capture_output=True,
                text=True,
                timeout=3
            )
            assert result.returncode == 0
            
            # Sample memory every 10 operations
            if i % 10 == 0:
                memory_samples.append(process.memory_info().rss)
        
        # Check for memory growth
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            memory_growth_mb = memory_growth / (1024 * 1024)
            
            # Should not grow more than 25MB during test
            assert memory_growth_mb < 25
    
    def test_cpu_utilization_limits(self, speak_script_path, performance_monitor):
        """Test that CPU utilization stays within reasonable bounds."""
        num_operations = 30
        
        for i in range(num_operations):
            result = subprocess.run(
                [str(speak_script_path), "--off", f"CPU test {i}"],
                capture_output=True,
                text=True,
                timeout=3
            )
            assert result.returncode == 0
            
            # Small delay to allow CPU monitoring
            time.sleep(0.1)
        
        # Check CPU usage
        stats = performance_monitor.get_stats()
        if stats and len(stats.get('cpu_samples', [])) > 0:
            # Average CPU should be reasonable
            assert stats['cpu_mean'] < 80  # Less than 80% average
            assert stats['cpu_peak'] < 100  # Should not peg CPU


class TestCachePerformance:
    """Test cache performance under high load."""
    
    def test_cache_hit_performance(self):
        """Test cache hit performance under load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Pre-populate cache
            test_texts = [f"Cache test {i}" for i in range(100)]
            provider = "openai"
            voice = "nova"
            
            for text in test_texts:
                cache.save_audio(text, provider, b"fake audio data", voice)
            
            # Test cache hit performance
            start_time = time.time()
            
            for _ in range(5):  # Multiple rounds
                for text in test_texts:
                    result = cache.get_audio_path(text, provider, voice)
                    assert result is not None
            
            duration = time.time() - start_time
            
            # Should be very fast (500 cache hits)
            assert duration < 2  # Less than 2 seconds
            hits_per_second = (500 / duration) if duration > 0 else float('inf')
            assert hits_per_second > 250  # At least 250 hits per second
    
    def test_cache_concurrent_access(self):
        """Test cache performance with concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Pre-populate cache
            test_texts = [f"Concurrent test {i}" for i in range(50)]
            provider = "openai"
            voice = "nova"
            
            for text in test_texts:
                cache.save_audio(text, provider, b"fake audio data", voice)
            
            results = queue.Queue()
            
            def worker(worker_id):
                worker_results = []
                for text in test_texts:
                    start_time = time.time()
                    result = cache.get_audio_path(text, provider, voice)
                    duration = time.time() - start_time
                    worker_results.append({
                        'worker_id': worker_id,
                        'found': result is not None,
                        'duration': duration
                    })
                results.put(worker_results)
            
            # Start multiple workers
            num_workers = 5
            threads = []
            start_time = time.time()
            
            for worker_id in range(num_workers):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_duration = time.time() - start_time
            
            # Collect results
            all_results = []
            while not results.empty():
                all_results.extend(results.get())
            
            # Verify all cache hits succeeded
            assert len(all_results) == num_workers * len(test_texts)
            hit_count = sum(1 for r in all_results if r['found'])
            assert hit_count == len(all_results)
            
            # Performance assertions
            assert total_duration < 5  # Should complete quickly
            durations = [r['duration'] for r in all_results]
            avg_duration = statistics.mean(durations)
            assert avg_duration < 0.01  # Average cache hit should be very fast
    
    def test_cache_cleanup_performance(self):
        """Test cache cleanup performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Create many cache entries
            provider = "openai"
            voice = "nova"
            
            for i in range(200):
                cache.save_audio(f"Cleanup test {i}", provider, b"fake audio data", voice)
            
            # Test cleanup performance
            start_time = time.time()
            # cleanup_old_cache takes days parameter, not max_files
            cleaned = cache.cleanup_old_cache(days=0)  # Remove all files
            duration = time.time() - start_time
            
            # Should complete quickly
            assert duration < 5
            # Note: cleanup_old_cache removes files by age, 
            # so we won't assert a specific number cleaned


class TestUsageTrackerPerformance:
    """Test usage tracker performance under load."""
    
    def test_usage_tracking_performance(self):
        """Test usage tracking performance with many operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            # Track many operations
            num_operations = 1000
            start_time = time.time()
            
            for i in range(num_operations):
                provider = "openai" if i % 2 == 0 else "elevenlabs"
                cached = i % 3 == 0
                # track_usage only increments stats if not cached
                tracker.track_usage(provider, f"Performance test {i}", cached=cached)
            
            duration = time.time() - start_time
            
            # Should complete quickly
            assert duration < 5  # Less than 5 seconds for 1000 operations
            ops_per_second = num_operations / duration
            assert ops_per_second > 200  # At least 200 operations per second
            
            # Verify data integrity
            # UsageTracker tracks requests per provider
            total_requests = 0
            for provider_data in tracker.stats['providers'].values():
                total_requests += provider_data['total_requests']
            # Every 3rd operation is cached (i % 3 == 0), so only 2/3 are actual requests
            expected_requests = sum(1 for i in range(num_operations) if i % 3 != 0)
            assert total_requests == expected_requests
            # Should have num_operations/3 cache hits
            expected_cache_hits = sum(1 for i in range(num_operations) if i % 3 == 0)
            assert tracker.stats['cache_hits'] == expected_cache_hits
    
    def test_usage_tracker_concurrent_access(self):
        """Test usage tracker with concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UsageTracker(data_dir=tmpdir)
            
            results = queue.Queue()
            
            def worker(worker_id):
                worker_results = []
                for i in range(50):
                    start_time = time.time()
                    provider = "openai" if i % 2 == 0 else "elevenlabs"
                    tracker.track_usage(provider, f"Worker {worker_id} op {i}")
                    duration = time.time() - start_time
                    worker_results.append(duration)
                results.put(worker_results)
            
            # Start multiple workers
            num_workers = 5
            threads = []
            start_time = time.time()
            
            for worker_id in range(num_workers):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_duration = time.time() - start_time
            
            # Collect results
            all_durations = []
            while not results.empty():
                all_durations.extend(results.get())
            
            # Performance assertions
            assert total_duration < 10  # Should complete within 10 seconds
            avg_duration = statistics.mean(all_durations)
            assert avg_duration < 0.1  # Average tracking should be very fast
            
            # Verify data integrity
            # UsageTracker tracks requests per provider
            total_requests = 0
            for provider_data in tracker.stats['providers'].values():
                total_requests += provider_data['total_requests']
            assert total_requests == num_workers * 50


class TestStressScenarios:
    """Test stress scenarios and edge cases."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    def test_rapid_pipe_operations_stress(self, speak_script_path):
        """Test rapid pipe operations under stress."""
        num_operations = 100
        start_time = time.time()
        
        for i in range(num_operations):
            result = subprocess.run(
                f"echo 'Pipe stress {i}' | {speak_script_path} --off",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0
        
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 60  # Less than 60 seconds
        ops_per_second = num_operations / duration
        assert ops_per_second > 1.5  # At least 1.5 operations per second
    
    def test_large_message_performance(self, speak_script_path):
        """Test performance with large messages."""
        large_message = "x" * 10000  # 10KB message
        
        start_time = time.time()
        result = subprocess.run(
            [str(speak_script_path), "--off", large_message],
            capture_output=True,
            text=True,
            timeout=10
        )
        duration = time.time() - start_time
        
        assert result.returncode == 0
        assert duration < 5  # Should complete within 5 seconds
    
    def test_unicode_performance(self, speak_script_path):
        """Test performance with unicode messages."""
        unicode_messages = [
            "Hello 世界 🌍",
            "Testing émojis 🎉",
            "Русский текст",
            "العربية",
            "日本語テスト"
        ]
        
        start_time = time.time()
        
        for message in unicode_messages * 10:  # 50 total operations
            result = subprocess.run(
                [str(speak_script_path), "--off", message],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0
        
        duration = time.time() - start_time
        assert duration < 30  # Should complete within 30 seconds
    
    def test_mixed_workload_stress(self, speak_script_path):
        """Test mixed workload with different operation types."""
        operations = [
            # Regular operations
            ([str(speak_script_path), "--off", "Regular message"], 3),
            # Pipe operations
            (f"echo 'Pipe message' | {speak_script_path} --off", 3),
            # Status checks
            ([str(speak_script_path), "--status"], 2),
            # Test operations
            ([str(speak_script_path), "--test", "--off"], 5),
        ]
        
        start_time = time.time()
        
        for cmd, timeout in operations * 10:  # 40 total operations
            if isinstance(cmd, list):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            assert result.returncode == 0
        
        duration = time.time() - start_time
        assert duration < 60  # Should complete within 60 seconds
    
    def test_error_recovery_performance(self, speak_script_path):
        """Test performance during error recovery scenarios."""
        # Mix of valid and invalid operations
        operations = []
        
        # Add valid operations
        for i in range(20):
            operations.append((
                [str(speak_script_path), "--off", f"Valid {i}"],
                True
            ))
        
        # Add invalid operations
        for i in range(5):
            operations.append((
                [str(speak_script_path), "--invalid-flag", f"Invalid {i}"],
                False
            ))
        
        start_time = time.time()
        
        for cmd, should_succeed in operations:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if should_succeed:
                assert result.returncode == 0
            else:
                assert result.returncode != 0
        
        duration = time.time() - start_time
        assert duration < 30  # Should complete within 30 seconds
    
    def test_system_resource_limits(self, speak_script_path):
        """Test behavior under system resource constraints."""
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        num_operations = 200
        
        for i in range(num_operations):
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Resource test {i}"],
                capture_output=True,
                text=True,
                timeout=3
            )
            assert result.returncode == 0
            
            # Check memory usage periodically
            if i % 50 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                memory_increase_mb = memory_increase / (1024 * 1024)
                
                # Should not consume excessive memory
                assert memory_increase_mb < 100  # Less than 100MB increase


class TestPerformanceBenchmarks:
    """Performance benchmarks for continuous monitoring."""
    
    @pytest.fixture
    def speak_script_path(self):
        """Get the path to the speak script."""
        return PROJECT_ROOT / "speak"
    
    def test_baseline_performance_benchmark(self, speak_script_path):
        """Establish baseline performance metrics."""
        num_operations = 50
        durations = []
        
        for i in range(num_operations):
            start_time = time.time()
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Benchmark {i}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            duration = time.time() - start_time
            
            assert result.returncode == 0
            durations.append(duration)
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Performance assertions (baseline targets)
        assert avg_duration < 0.5  # Average under 500ms
        assert median_duration < 0.4  # Median under 400ms
        assert max_duration < 2.0  # Max under 2 seconds
        assert min_duration < 1.0  # Min under 1 second
        
        # Print benchmark results for monitoring
        print(f"\nPerformance Benchmark Results:")
        print(f"  Average duration: {avg_duration:.3f}s")
        print(f"  Median duration:  {median_duration:.3f}s")
        print(f"  Max duration:     {max_duration:.3f}s")
        print(f"  Min duration:     {min_duration:.3f}s")
        print(f"  Operations/sec:   {num_operations / sum(durations):.2f}")
    
    def test_throughput_benchmark(self, speak_script_path):
        """Benchmark maximum throughput."""
        duration_seconds = 10
        operations = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            op_start = time.time()
            result = subprocess.run(
                [str(speak_script_path), "--off", f"Throughput {len(operations)}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            op_duration = time.time() - op_start
            
            operations.append({
                'success': result.returncode == 0,
                'duration': op_duration
            })
        
        # Calculate throughput
        total_duration = time.time() - start_time
        successful_ops = sum(1 for op in operations if op['success'])
        throughput = successful_ops / total_duration
        
        # Throughput assertions
        assert throughput > 5  # At least 5 operations per second
        assert successful_ops >= len(operations) * 0.95  # 95% success rate
        
        print(f"\nThroughput Benchmark Results:")
        print(f"  Total operations: {len(operations)}")
        print(f"  Successful ops:   {successful_ops}")
        print(f"  Duration:         {total_duration:.2f}s")
        print(f"  Throughput:       {throughput:.2f} ops/sec")
    
    def test_cache_performance_benchmark(self):
        """Benchmark cache performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TTSCache(cache_dir=tmpdir)
            
            # Test cache write performance
            num_writes = 100
            write_start = time.time()
            
            for i in range(num_writes):
                cache.save_audio(f"Cache write {i}", "openai", b"fake audio data", "nova")
            
            write_duration = time.time() - write_start
            
            # Test cache read performance
            read_start = time.time()
            
            for i in range(num_writes):
                result = cache.get_audio_path(f"Cache write {i}", "openai", "nova")
                assert result is not None
            
            read_duration = time.time() - read_start
            
            # Performance assertions
            write_throughput = num_writes / write_duration
            read_throughput = num_writes / read_duration
            
            assert write_throughput > 50  # At least 50 writes per second
            assert read_throughput > 500  # At least 500 reads per second
            
            print(f"\nCache Performance Benchmark:")
            print(f"  Write throughput: {write_throughput:.2f} ops/sec")
            print(f"  Read throughput:  {read_throughput:.2f} ops/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])