"""
Calibration Algorithms
Main calibration pipeline and specific calibration methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from scipy import optimize, stats


class CalibrationMethod(ABC):
    """Abstract base class for calibration methods."""
    
    @abstractmethod
    def calibrate(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply calibration to signal."""
        pass


class GainOffsetCalibration(CalibrationMethod):
    """Gain and offset calibration using reference measurements."""
    
    def __init__(self, reference_gain: float = 1.0, reference_offset: float = 0.0):
        """
        Initialize gain/offset calibration.
        
        Args:
            reference_gain: Known reference gain
            reference_offset: Known reference offset
        """
        self.reference_gain = reference_gain
        self.reference_offset = reference_offset
        self.estimated_gain = None
        self.estimated_offset = None
    
    def calibrate(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply gain and offset correction."""
        if self.estimated_gain is None or self.estimated_offset is None:
            # Use reference values if estimates are not available
            gain = self.reference_gain
            offset = self.reference_offset
        else:
            gain = self.estimated_gain
            offset = self.estimated_offset
        
        # Apply calibration
        calibrated = (signal - offset) / gain
        return calibrated
    
    def estimate_from_flat_field(self, flat_field: np.ndarray, 
                                 expected_value: float = 1.0) -> Tuple[float, float]:
        """
        Estimate gain and offset from flat field measurement.
        
        Args:
            flat_field: Flat field measurement
            expected_value: Expected uniform value
        
        Returns:
            Tuple of (estimated_gain, estimated_offset)
        """
        # Simple linear calibration: y = gain * x + offset
        # For flat field, we assume x = expected_value, y = mean(flat_field)
        measured_mean = np.mean(flat_field)
        
        # Estimate gain (assuming offset is small compared to signal)
        self.estimated_gain = measured_mean / expected_value
        
        # Estimate offset from dark regions (minimum values)
        self.estimated_offset = np.min(flat_field)
        
        return self.estimated_gain, self.estimated_offset
    
    def estimate_from_dark_current(self, dark_frame: np.ndarray) -> float:
        """
        Estimate offset from dark current measurement.
        
        Args:
            dark_frame: Dark frame measurement
        
        Returns:
            Estimated offset
        """
        self.estimated_offset = np.mean(dark_frame)
        return self.estimated_offset


class NonLinearCalibration(CalibrationMethod):
    """Non-linear calibration using polynomial correction."""
    
    def __init__(self, coefficients: Optional[np.ndarray] = None):
        """
        Initialize non-linear calibration.
        
        Args:
            coefficients: Polynomial coefficients for calibration
        """
        self.coefficients = coefficients
        if coefficients is None:
            self.coefficients = np.array([1.0, 0.0])  # Linear by default
    
    def calibrate(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply polynomial calibration."""
        calibrated = np.zeros_like(signal)
        
        for i, coeff in enumerate(self.coefficients):
            calibrated += coeff * (signal ** i)
        
        return calibrated
    
    def fit_calibration_curve(self, input_values: np.ndarray, 
                             output_values: np.ndarray, 
                             degree: int = 3) -> np.ndarray:
        """
        Fit calibration curve from known input/output pairs.
        
        Args:
            input_values: Known input values
            output_values: Measured output values
            degree: Polynomial degree for fitting
        
        Returns:
            Fitted coefficients
        """
        self.coefficients = np.polyfit(input_values, output_values, degree)
        return self.coefficients


class CalibrationPipeline:
    """Complete calibration pipeline combining multiple methods."""
    
    def __init__(self):
        """Initialize calibration pipeline."""
        self.methods: list[CalibrationMethod] = []
        self.calibration_history: list[Dict[str, Any]] = []
    
    def add_method(self, method: CalibrationMethod):
        """Add calibration method to pipeline."""
        self.methods.append(method)
    
    def clear_methods(self):
        """Remove all calibration methods."""
        self.methods.clear()
        self.calibration_history.clear()
    
    def calibrate(self, signal: np.ndarray, save_history: bool = True) -> np.ndarray:
        """
        Apply full calibration pipeline.
        
        Args:
            signal: Input signal to calibrate
            save_history: Whether to save calibration steps in history
        
        Returns:
            Fully calibrated signal
        """
        calibrated = signal.copy()
        
        for i, method in enumerate(self.methods):
            calibrated = method.calibrate(calibrated)
            
            if save_history:
                self.calibration_history.append({
                    'step': i,
                    'method': type(method).__name__,
                    'mean': np.mean(calibrated),
                    'std': np.std(calibrated),
                    'min': np.min(calibrated),
                    'max': np.max(calibrated)
                })
        
        return calibrated
    
    def get_calibration_history(self) -> list[Dict[str, Any]]:
        """Get history of calibration steps."""
        return self.calibration_history.copy()
    
    def auto_calibrate(self, signal: np.ndarray, dark_frame: Optional[np.ndarray] = None,
                       flat_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Automatic calibration using dark frames and flat fields.
        
        Args:
            signal: Input signal to calibrate
            dark_frame: Dark frame for offset estimation
            flat_field: Flat field for gain estimation
        
        Returns:
            Calibrated signal
        """
        calibrated = signal.copy()
        
        # Dark current correction
        if dark_frame is not None:
            calibrated = calibrated - np.mean(dark_frame)
        
        # Flat field correction
        if flat_field is not None:
            flat_mean = np.mean(flat_field)
            gain_correction = flat_mean / flat_field
            calibrated = calibrated * gain_correction
        
        return calibrated


class PixelResponseCalibration:
    """Per-pixel response calibration for non-uniform detectors."""
    
    def __init__(self):
        """Initialize pixel response calibration."""
        self.pixel_gains: Optional[np.ndarray] = None
        self.pixel_offsets: Optional[np.ndarray] = None
    
    def calibrate_pixel_response(self, flat_field: np.ndarray, 
                                 dark_frame: np.ndarray,
                                 expected_value: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate per-pixel gain and offset corrections.
        
        Args:
            flat_field: Flat field measurement
            dark_frame: Dark frame measurement
            expected_value: Expected uniform value
        
        Returns:
            Tuple of (pixel_gains, pixel_offsets)
        """
        # Calculate pixel offsets from dark frame
        self.pixel_offsets = dark_frame
        
        # Calculate pixel gains from flat field
        flat_corrected = flat_field - self.pixel_offsets
        self.pixel_gains = expected_value / flat_corrected
        
        # Handle division by zero
        self.pixel_gains = np.where(flat_corrected > 0, self.pixel_gains, 1.0)
        
        return self.pixel_gains, self.pixel_offsets
    
    def apply_pixel_calibration(self, signal: np.ndarray) -> np.ndarray:
        """Apply per-pixel calibration to signal."""
        if self.pixel_gains is None or self.pixel_offsets is None:
            raise ValueError("Pixel calibration not performed. Call calibrate_pixel_response first.")
        
        calibrated = (signal - self.pixel_offsets) * self.pixel_gains
        return calibrated


class TemperatureCalibration:
    """Temperature-dependent calibration."""
    
    def __init__(self, reference_temperature: float = 20.0):
        """
        Initialize temperature calibration.
        
        Args:
            reference_temperature: Reference temperature in Celsius
        """
        self.reference_temperature = reference_temperature
        self.temperature_coefficient = 0.01  # Typical value: 1% per degree
    
    def calibrate(self, signal: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature correction.
        
        Args:
            signal: Input signal
            temperature: Current temperature in Celsius
        
        Returns:
            Temperature-corrected signal
        """
        temp_diff = temperature - self.reference_temperature
        correction_factor = 1.0 / (1.0 + self.temperature_coefficient * temp_diff)
        
        return signal * correction_factor
