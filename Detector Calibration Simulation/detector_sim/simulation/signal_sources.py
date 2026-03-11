"""
Signal Sources
Various signal sources for detector simulation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math


class SignalSource(ABC):
    """Abstract base class for signal sources."""
    
    @abstractmethod
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate signal for given detector dimensions."""
        pass


class PointSource(SignalSource):
    """Point source signal at a specific location."""
    
    def __init__(self, x: float, y: float, intensity: float = 1.0):
        """
        Initialize point source.
        
        Args:
            x: X position of the point source (in pixel coordinates)
            y: Y position of the point source (in pixel coordinates)
            intensity: Signal intensity
        """
        self.x = x
        self.y = y
        self.intensity = intensity
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate point source signal."""
        signal = np.zeros((height, width))
        
        # Find nearest pixel
        px, py = int(self.x), int(self.y)
        
        # Check if within bounds
        if 0 <= px < width and 0 <= py < height:
            signal[py, px] = self.intensity
        
        return signal


class MultiplePointSource(SignalSource):
    """Multiple point sources with individual positions and intensities."""
    
    def __init__(self, sources: list):
        """
        Initialize multiple point sources.
        
        Args:
            sources: List of tuples (x, y, intensity) for each point source
        """
        self.sources = sources
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate combined signal from multiple point sources."""
        signal = np.zeros((height, width))
        
        for x, y, intensity in self.sources:
            px, py = int(x), int(y)
            if 0 <= px < width and 0 <= py < height:
                signal[py, px] += intensity
        
        return signal


class UniformSource(SignalSource):
    """Uniform illumination across the detector."""
    
    def __init__(self, intensity: float = 1.0):
        """
        Initialize uniform source.
        
        Args:
            intensity: Uniform signal intensity
        """
        self.intensity = intensity
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate uniform signal."""
        return np.full((height, width), self.intensity)


class GaussianSource(SignalSource):
    """Gaussian-distributed signal source."""
    
    def __init__(self, center_x: float, center_y: float, 
                 sigma_x: float = 10.0, sigma_y: float = 10.0,
                 intensity: float = 1.0, rotation: float = 0.0):
        """
        Initialize Gaussian source.
        
        Args:
            center_x: X center of Gaussian
            center_y: Y center of Gaussian
            sigma_x: Standard deviation in X direction
            sigma_y: Standard deviation in Y direction
            intensity: Peak intensity
            rotation: Rotation angle in degrees
        """
        self.center_x = center_x
        self.center_y = center_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.intensity = intensity
        self.rotation = math.radians(rotation)
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate Gaussian signal."""
        # Create coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Translate to center
        xx_shifted = xx - self.center_x
        yy_shifted = yy - self.center_y
        
        # Apply rotation
        if self.rotation != 0:
            cos_rot = math.cos(self.rotation)
            sin_rot = math.sin(self.rotation)
            xx_rot = xx_shifted * cos_rot - yy_shifted * sin_rot
            yy_rot = xx_shifted * sin_rot + yy_shifted * cos_rot
        else:
            xx_rot = xx_shifted
            yy_rot = yy_shifted
        
        # Calculate Gaussian
        gaussian = self.intensity * np.exp(
            -0.5 * ((xx_rot / self.sigma_x) ** 2 + (yy_rot / self.sigma_y) ** 2)
        )
        
        return gaussian


class GammaRaySource(SignalSource):
    """Simplified gamma ray source with energy deposition."""
    
    def __init__(self, energy: float = 511.0, intensity: float = 1.0, 
                 interaction_probability: float = 0.1):
        """
        Initialize gamma ray source.
        
        Args:
            energy: Gamma ray energy in keV
            intensity: Source intensity
            interaction_probability: Probability of interaction per pixel
        """
        self.energy = energy
        self.intensity = intensity
        self.interaction_probability = interaction_probability
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate gamma ray interaction pattern."""
        signal = np.zeros((height, width))
        
        # Simulate random interactions
        for i in range(height):
            for j in range(width):
                if np.random.random() < self.interaction_probability:
                    # Energy deposition with some variation
                    deposited_energy = self.energy * (0.8 + 0.4 * np.random.random())
                    signal[i, j] = deposited_energy * self.intensity
        
        return signal


class PatternSource(SignalSource):
    """Custom pattern source (e.g., test patterns, grids)."""
    
    def __init__(self, pattern_type: str = "grid", spacing: int = 10, 
                 intensity: float = 1.0):
        """
        Initialize pattern source.
        
        Args:
            pattern_type: Type of pattern ("grid", "checkerboard", "circles")
            spacing: Spacing between pattern elements
            intensity: Pattern intensity
        """
        self.pattern_type = pattern_type
        self.spacing = spacing
        self.intensity = intensity
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate pattern signal."""
        signal = np.zeros((height, width))
        
        if self.pattern_type == "grid":
            # Grid pattern
            signal[self.spacing::self.spacing*2, :] = self.intensity
            signal[:, self.spacing::self.spacing*2] = self.intensity
            
        elif self.pattern_type == "checkerboard":
            # Checkerboard pattern
            for i in range(0, height, self.spacing):
                for j in range(0, width, self.spacing):
                    if ((i // self.spacing) + (j // self.spacing)) % 2 == 0:
                        signal[i:i+self.spacing, j:j+self.spacing] = self.intensity
                        
        elif self.pattern_type == "circles":
            # Circle pattern
            center_x, center_y = width // 2, height // 2
            for radius in range(self.spacing, min(width, height) // 2, self.spacing):
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                mask_prev = (x - center_x) ** 2 + (y - center_y) ** 2 <= (radius - self.spacing) ** 2
                signal[mask ^ mask_prev] = self.intensity
        
        return signal


class RandomNoiseSource(SignalSource):
    """Random noise source for background simulation."""
    
    def __init__(self, mean: float = 0.0, std_dev: float = 0.1):
        """
        Initialize random noise source.
        
        Args:
            mean: Mean of the noise
            std_dev: Standard deviation of the noise
        """
        self.mean = mean
        self.std_dev = std_dev
    
    def generate_signal(self, width: int, height: int) -> np.ndarray:
        """Generate random noise signal."""
        return np.random.normal(self.mean, self.std_dev, (height, width))
