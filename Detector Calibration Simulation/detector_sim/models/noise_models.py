"""
Noise Models
Various noise models for detector simulation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add noise to signal array."""
        pass
    
    @abstractmethod
    def add_noise_scalar(self, signal: float) -> float:
        """Add noise to scalar signal."""
        pass


class GaussianNoise(NoiseModel):
    """Gaussian (normal) noise model."""
    
    def __init__(self, mean: float = 0.0, std_dev: float = 1.0):
        """
        Initialize Gaussian noise.
        
        Args:
            mean: Mean of the Gaussian distribution
            std_dev: Standard deviation of the Gaussian distribution
        """
        self.mean = mean
        self.std_dev = std_dev
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to signal array."""
        noise = np.random.normal(self.mean, self.std_dev, signal.shape)
        return signal + noise
    
    def add_noise_scalar(self, signal: float) -> float:
        """Add Gaussian noise to scalar signal."""
        noise = np.random.normal(self.mean, self.std_dev)
        return signal + noise


class PoissonNoise(NoiseModel):
    """Poisson noise model for photon counting."""
    
    def __init__(self, scale_factor: float = 1.0):
        """
        Initialize Poisson noise.
        
        Args:
            scale_factor: Scale factor for the Poisson process
        """
        self.scale_factor = scale_factor
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Poisson noise to signal array."""
        # Ensure non-negative values for Poisson
        scaled_signal = np.maximum(signal * self.scale_factor, 0)
        noisy_signal = np.random.poisson(scaled_signal) / self.scale_factor
        return noisy_signal
    
    def add_noise_scalar(self, signal: float) -> float:
        """Add Poisson noise to scalar signal."""
        scaled_signal = max(signal * self.scale_factor, 0)
        noisy_signal = np.random.poisson(scaled_signal) / self.scale_factor
        return noisy_signal


class ReadoutNoise(NoiseModel):
    """Readout noise model combining Gaussian and 1/f components."""
    
    def __init__(self, readout_std: float = 1.0, flicker_strength: float = 0.1):
        """
        Initialize readout noise.
        
        Args:
            readout_std: Standard deviation of white readout noise
            flicker_strength: Strength of 1/f (flicker) noise
        """
        self.readout_std = readout_std
        self.flicker_strength = flicker_strength
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add readout noise to signal array."""
        # White Gaussian noise
        white_noise = np.random.normal(0, self.readout_std, signal.shape)
        
        # Simple 1/f noise approximation using correlated Gaussian noise
        if signal.ndim == 2:
            # Create correlated noise for 2D arrays
            flicker_noise = self._generate_flicker_noise(signal.shape)
        else:
            flicker_noise = np.zeros_like(signal)
        
        return signal + white_noise + flicker_noise
    
    def add_noise_scalar(self, signal: float) -> float:
        """Add readout noise to scalar signal."""
        noise = np.random.normal(0, self.readout_std)
        return signal + noise
    
    def _generate_flicker_noise(self, shape: tuple) -> np.ndarray:
        """Generate simplified 1/f noise using spatial filtering."""
        noise = np.random.normal(0, 1, shape)
        
        # Apply simple low-pass filter to create spatial correlation
        from scipy import ndimage
        filtered = ndimage.gaussian_filter(noise, sigma=2.0)
        
        return filtered * self.flicker_strength


class CombinedNoise(NoiseModel):
    """Combine multiple noise models."""
    
    def __init__(self, noise_models: list):
        """
        Initialize combined noise model.
        
        Args:
            noise_models: List of noise models to combine
        """
        self.noise_models = noise_models
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add all noise models to signal array."""
        result = signal.copy()
        for noise_model in self.noise_models:
            result = noise_model.add_noise(result)
        return result
    
    def add_noise_scalar(self, signal: float) -> float:
        """Add all noise models to scalar signal."""
        result = signal
        for noise_model in self.noise_models:
            result = noise_model.add_noise_scalar(result)
        return result
