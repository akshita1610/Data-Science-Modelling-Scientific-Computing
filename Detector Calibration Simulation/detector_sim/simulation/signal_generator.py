"""
Signal Generator
Main class for generating and combining signals.
"""

import numpy as np
from typing import List, Union, Optional
from .signal_sources import SignalSource


class SignalGenerator:
    """Main signal generator class that combines multiple sources."""
    
    def __init__(self, width: int = 100, height: int = 100):
        """
        Initialize signal generator.
        
        Args:
            width: Detector width in pixels
            height: Detector height in pixels
        """
        self.width = width
        self.height = height
        self.sources: List[SignalSource] = []
        self.background: Optional[np.ndarray] = None
    
    def add_source(self, source: SignalSource):
        """Add a signal source to the generator."""
        self.sources.append(source)
    
    def clear_sources(self):
        """Remove all signal sources."""
        self.sources.clear()
    
    def set_background(self, background: np.ndarray):
        """Set background signal."""
        if background.shape != (self.height, self.width):
            raise ValueError(f"Background shape {background.shape} doesn't match generator shape {(self.height, self.width)}")
        self.background = background
    
    def generate_signal(self, include_background: bool = True) -> np.ndarray:
        """
        Generate combined signal from all sources.
        
        Args:
            include_background: Whether to include background signal
        
        Returns:
            Combined signal array
        """
        # Start with background if available
        if include_background and self.background is not None:
            signal = self.background.copy()
        else:
            signal = np.zeros((self.height, self.width))
        
        # Add all sources
        for source in self.sources:
            signal += source.generate_signal(self.width, self.height)
        
        return signal
    
    def generate_time_series(self, num_frames: int, include_background: bool = True) -> np.ndarray:
        """
        Generate time series of signals.
        
        Args:
            num_frames: Number of time frames to generate
            include_background: Whether to include background signal
        
        Returns:
            3D array with shape (num_frames, height, width)
        """
        time_series = np.zeros((num_frames, self.height, self.width))
        
        for frame in range(num_frames):
            time_series[frame] = self.generate_signal(include_background)
        
        return time_series
    
    def get_total_intensity(self) -> float:
        """Calculate total signal intensity."""
        signal = self.generate_signal()
        return np.sum(signal)
    
    def get_peak_intensity(self) -> float:
        """Get peak signal intensity."""
        signal = self.generate_signal()
        return np.max(signal)
    
    def get_signal_statistics(self) -> dict:
        """Get comprehensive signal statistics."""
        signal = self.generate_signal()
        
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'total': np.sum(signal),
            'nonzero_pixels': np.count_nonzero(signal),
            'coverage': np.count_nonzero(signal) / signal.size
        }
    
    def resize(self, new_width: int, new_height: int):
        """Resize the signal generator dimensions."""
        self.width = new_width
        self.height = new_height
        
        # Reset background if it doesn't match new dimensions
        if self.background is not None and self.background.shape != (new_height, new_width):
            self.background = None
