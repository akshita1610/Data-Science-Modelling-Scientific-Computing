"""
Detector Models
Base and specific detector implementations for radiation and optical detection.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Detector(ABC):
    """Abstract base class for all detector types."""
    
    def __init__(self, gain: float = 1.0, offset: float = 0.0, sensitivity: float = 1.0):
        """
        Initialize detector with basic parameters.
        
        Args:
            gain: Amplification factor for detected signals
            offset: Constant offset added to all measurements
            sensitivity: Detector sensitivity to input signals
        """
        self.gain = gain
        self.offset = offset
        self.sensitivity = sensitivity
        self.noise_model = None
    
    def set_noise_model(self, noise_model):
        """Set the noise model for this detector."""
        self.noise_model = noise_model
    
    @abstractmethod
    def detect(self, signal: np.ndarray) -> np.ndarray:
        """Process input signal and return detector response."""
        pass
    
    def apply_calibration(self, signal: np.ndarray, reference_gain: float = 1.0, 
                         reference_offset: float = 0.0) -> np.ndarray:
        """
        Apply calibration corrections to detector output.
        
        Args:
            signal: Raw detector signal
            reference_gain: Known reference gain for calibration
            reference_offset: Known reference offset for calibration
        
        Returns:
            Calibrated signal
        """
        # Remove detector-specific gain and offset
        calibrated = (signal - self.offset) / self.gain
        
        # Apply reference corrections
        calibrated = calibrated * reference_gain + reference_offset
        
        return calibrated


class PixelDetector(Detector):
    """Pixel-based detector with configurable grid size."""
    
    def __init__(self, width: int = 100, height: int = 100, 
                 gain: float = 1.0, offset: float = 0.0, 
                 sensitivity: float = 1.0, pixel_size: float = 1.0):
        """
        Initialize pixel detector.
        
        Args:
            width: Number of pixels in x-direction
            height: Number of pixels in y-direction
            gain: Amplification factor
            offset: Constant offset
            sensitivity: Detector sensitivity
            pixel_size: Physical size of each pixel (in arbitrary units)
        """
        super().__init__(gain, offset, sensitivity)
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.dark_current = np.zeros((height, width))
    
    def detect(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect input signal and add detector effects.
        
        Args:
            signal: Input signal array (should match detector dimensions)
        
        Returns:
            Detector response with noise and detector effects
        """
        if signal.shape != (self.height, self.width):
            raise ValueError(f"Signal shape {signal.shape} doesn't match detector shape {(self.height, self.width)}")
        
        # Apply sensitivity and gain
        detected = signal * self.sensitivity * self.gain
        
        # Add dark current
        detected += self.dark_current
        
        # Add offset
        detected += self.offset
        
        # Add noise if noise model is available
        if self.noise_model:
            detected = self.noise_model.add_noise(detected)
        
        return detected
    
    def set_dark_current(self, dark_current: float):
        """Set uniform dark current for all pixels."""
        self.dark_current = np.full((self.height, self.width), dark_current)
    
    def set_nonuniform_dark_current(self, dark_current_map: np.ndarray):
        """Set non-uniform dark current from a map."""
        if dark_current_map.shape != (self.height, self.width):
            raise ValueError(f"Dark current map shape {dark_current_map.shape} doesn't match detector shape")
        self.dark_current = dark_current_map
    
    def get_pixel_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get x, y coordinates for each pixel center."""
        x = np.arange(self.width) * self.pixel_size + self.pixel_size / 2
        y = np.arange(self.height) * self.pixel_size + self.pixel_size / 2
        return np.meshgrid(x, y)


class SingleChannelDetector(Detector):
    """Single-channel detector for point measurements."""
    
    def __init__(self, gain: float = 1.0, offset: float = 0.0, 
                 sensitivity: float = 1.0, integration_time: float = 1.0):
        """
        Initialize single-channel detector.
        
        Args:
            gain: Amplification factor
            offset: Constant offset
            sensitivity: Detector sensitivity
            integration_time: Time over which signal is integrated
        """
        super().__init__(gain, offset, sensitivity)
        self.integration_time = integration_time
    
    def detect(self, signal: float) -> float:
        """
        Detect single value signal.
        
        Args:
            signal: Input signal value
        
        Returns:
            Detector response
        """
        # Apply sensitivity, gain, and integration time
        detected = signal * self.sensitivity * self.gain * self.integration_time
        
        # Add offset
        detected += self.offset
        
        # Add noise if noise model is available
        if self.noise_model:
            detected = self.noise_model.add_noise_scalar(detected)
        
        return detected
