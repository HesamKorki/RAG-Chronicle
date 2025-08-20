"""Timing utilities for performance measurement."""

import time
from contextlib import contextmanager
from typing import Dict, Optional


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def get_elapsed(self) -> Optional[float]:
        """Get elapsed time without stopping the timer."""
        if self.start_time is None:
            return None
        
        if self.elapsed_time is not None:
            return self.elapsed_time
        
        return time.time() - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{name} took {elapsed:.4f} seconds")


class PerformanceTracker:
    """Track performance metrics across multiple operations."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}
    
    def print_summary(self) -> None:
        """Print a summary of all metrics."""
        print("\nPerformance Summary:")
        print("-" * 50)
        
        for name, stats in self.get_all_stats().items():
            if stats:
                print(f"{name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Min: {stats['min']:.4f}")
                print(f"  Max: {stats['max']:.4f}")
                print(f"  Total: {stats['total']:.4f}")
                print()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        with Timer(func.__name__) as timer:
            result = func(*args, **kwargs)
        print(f"{func.__name__} took {timer.elapsed_time:.4f} seconds")
        return result
    return wrapper


def format_time(seconds: float) -> str:
    """Format time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
