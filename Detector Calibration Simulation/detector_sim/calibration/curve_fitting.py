"""
Curve Fitting Module
Various curve fitting algorithms for calibration and analysis.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from scipy import optimize, stats
from scipy.special import erf


class CurveFitter(ABC):
    """Abstract base class for curve fitting algorithms."""
    
    @abstractmethod
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Fit curve to data."""
        pass
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate fitted curve at given points."""
        pass


class GaussianFitter(CurveFitter):
    """Gaussian curve fitting for peak analysis."""
    
    def __init__(self):
        """Initialize Gaussian fitter."""
        self.params = None
        self.covariance = None
    
    def gaussian_function(self, x: np.ndarray, amplitude: float, 
                         center: float, sigma: float, offset: float = 0.0) -> np.ndarray:
        """Gaussian function."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Fit Gaussian to data.
        
        Args:
            x_data: X coordinate data
            y_data: Y coordinate data
        
        Returns:
            Dictionary with fit parameters and statistics
        """
        # Initial parameter guess
        amplitude_guess = np.max(y_data) - np.min(y_data)
        center_guess = x_data[np.argmax(y_data)]
        sigma_guess = (np.max(x_data) - np.min(x_data)) / 4
        offset_guess = np.min(y_data)
        
        initial_params = [amplitude_guess, center_guess, sigma_guess, offset_guess]
        
        try:
            # Fit Gaussian
            self.params, self.covariance = optimize.curve_fit(
                self.gaussian_function, x_data, y_data, 
                p0=initial_params, maxfev=10000
            )
            
            # Calculate fit statistics
            y_fit = self.gaussian_function(x_data, *self.params)
            residuals = y_data - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate FWHM
            fwhm = 2 * np.sqrt(2 * np.log(2)) * self.params[2]
            
            return {
                'parameters': {
                    'amplitude': self.params[0],
                    'center': self.params[1],
                    'sigma': self.params[2],
                    'offset': self.params[3],
                    'fwhm': fwhm
                },
                'covariance': self.covariance,
                'r_squared': r_squared,
                'rmse': np.sqrt(np.mean(residuals ** 2)),
                'fitted_curve': y_fit
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'parameters': None,
                'r_squared': 0.0,
                'rmse': float('inf')
            }
    
    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate Gaussian at given points."""
        if params is None:
            if self.params is None:
                raise ValueError("No parameters available. Call fit() first.")
            params = self.params
        
        return self.gaussian_function(x, *params)


class PolynomialFitter(CurveFitter):
    """Polynomial curve fitting."""
    
    def __init__(self, degree: int = 2):
        """
        Initialize polynomial fitter.
        
        Args:
            degree: Polynomial degree
        """
        self.degree = degree
        self.coefficients = None
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Fit polynomial to data.
        
        Args:
            x_data: X coordinate data
            y_data: Y coordinate data
        
        Returns:
            Dictionary with fit parameters and statistics
        """
        try:
            # Fit polynomial
            self.coefficients = np.polyfit(x_data, y_data, self.degree)
            
            # Evaluate fit
            y_fit = self.evaluate(x_data)
            
            # Calculate statistics
            residuals = y_data - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                'coefficients': self.coefficients,
                'degree': self.degree,
                'r_squared': r_squared,
                'rmse': rmse,
                'fitted_curve': y_fit
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'coefficients': None,
                'r_squared': 0.0,
                'rmse': float('inf')
            }
    
    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate polynomial at given points."""
        if params is None:
            if self.coefficients is None:
                raise ValueError("No coefficients available. Call fit() first.")
            params = self.coefficients
        
        return np.polyval(params, x)


class ExponentialFitter(CurveFitter):
    """Exponential decay/growth fitting."""
    
    def __init__(self):
        """Initialize exponential fitter."""
        self.params = None
    
    def exponential_function(self, x: np.ndarray, amplitude: float, 
                             decay_constant: float, offset: float = 0.0) -> np.ndarray:
        """Exponential function."""
        return amplitude * np.exp(-x / decay_constant) + offset
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Fit exponential to data.
        
        Args:
            x_data: X coordinate data
            y_data: Y coordinate data
        
        Returns:
            Dictionary with fit parameters and statistics
        """
        # Initial parameter guess
        amplitude_guess = np.max(y_data) - np.min(y_data)
        decay_guess = (np.max(x_data) - np.min(x_data)) / 3
        offset_guess = np.min(y_data)
        
        initial_params = [amplitude_guess, decay_guess, offset_guess]
        
        try:
            # Fit exponential
            self.params, _ = optimize.curve_fit(
                self.exponential_function, x_data, y_data,
                p0=initial_params, maxfev=10000
            )
            
            # Evaluate fit
            y_fit = self.evaluate(x_data)
            
            # Calculate statistics
            residuals = y_data - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Calculate half-life
            half_life = self.params[1] * np.log(2)
            
            return {
                'parameters': {
                    'amplitude': self.params[0],
                    'decay_constant': self.params[1],
                    'offset': self.params[2],
                    'half_life': half_life
                },
                'r_squared': r_squared,
                'rmse': rmse,
                'fitted_curve': y_fit
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'parameters': None,
                'r_squared': 0.0,
                'rmse': float('inf')
            }
    
    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate exponential at given points."""
        if params is None:
            if self.params is None:
                raise ValueError("No parameters available. Call fit() first.")
            params = self.params
        
        return self.exponential_function(x, *params)


class SigmoidFitter(CurveFitter):
    """Sigmoid (logistic) curve fitting."""
    
    def __init__(self):
        """Initialize sigmoid fitter."""
        self.params = None
    
    def sigmoid_function(self, x: np.ndarray, amplitude: float, 
                        center: float, width: float, offset: float = 0.0) -> np.ndarray:
        """Sigmoid function."""
        return amplitude / (1 + np.exp(-(x - center) / width)) + offset
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Fit sigmoid to data.
        
        Args:
            x_data: X coordinate data
            y_data: Y coordinate data
        
        Returns:
            Dictionary with fit parameters and statistics
        """
        # Initial parameter guess
        amplitude_guess = np.max(y_data) - np.min(y_data)
        center_guess = np.median(x_data)
        width_guess = (np.max(x_data) - np.min(x_data)) / 6
        offset_guess = np.min(y_data)
        
        initial_params = [amplitude_guess, center_guess, width_guess, offset_guess]
        
        try:
            # Fit sigmoid
            self.params, _ = optimize.curve_fit(
                self.sigmoid_function, x_data, y_data,
                p0=initial_params, maxfev=10000
            )
            
            # Evaluate fit
            y_fit = self.evaluate(x_data)
            
            # Calculate statistics
            residuals = y_data - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                'parameters': {
                    'amplitude': self.params[0],
                    'center': self.params[1],
                    'width': self.params[2],
                    'offset': self.params[3]
                },
                'r_squared': r_squared,
                'rmse': rmse,
                'fitted_curve': y_fit
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'parameters': None,
                'r_squared': 0.0,
                'rmse': float('inf')
            }
    
    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate sigmoid at given points."""
        if params is None:
            if self.params is None:
                raise ValueError("No parameters available. Call fit() first.")
            params = self.params
        
        return self.sigmoid_function(x, *params)


class PeakFinder:
    """Peak finding and analysis utilities."""
    
    @staticmethod
    def find_peaks(data: np.ndarray, threshold: float = None, 
                   min_distance: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in 1D data.
        
        Args:
            data: 1D data array
            threshold: Minimum peak height
            min_distance: Minimum distance between peaks
        
        Returns:
            Tuple of (peak_indices, peak_values)
        """
        if threshold is None:
            threshold = np.mean(data) + 2 * np.std(data)
        
        # Simple peak finding algorithm
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and 
                data[i] > threshold):
                peaks.append(i)
        
        # Filter by minimum distance
        if min_distance > 1 and len(peaks) > 1:
            filtered_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
            peaks = filtered_peaks
        
        peak_indices = np.array(peaks)
        peak_values = data[peak_indices]
        
        return peak_indices, peak_values
    
    @staticmethod
    def find_2d_peaks(data: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in 2D data.
        
        Args:
            data: 2D data array
            threshold: Minimum peak height
        
        Returns:
            Tuple of (peak_coordinates, peak_values)
        """
        if threshold is None:
            threshold = np.mean(data) + 2 * np.std(data)
        
        # Find local maxima
        from scipy import ndimage
        local_max = ndimage.maximum_filter(data, size=3)
        peaks = (data == local_max) & (data > threshold)
        
        peak_coords = np.argwhere(peaks)
        peak_values = data[peaks]
        
        return peak_coords, peak_values


class CalibrationCurveFitter:
    """Specialized fitter for detector calibration curves."""
    
    def __init__(self):
        """Initialize calibration curve fitter."""
        self.fitter = PolynomialFitter(degree=1)  # Linear by default
    
    def fit_linearity_curve(self, known_values: np.ndarray, 
                           measured_values: np.ndarray) -> Dict[str, Any]:
        """
        Fit linearity curve for detector calibration.
        
        Args:
            known_values: Known input values
            measured_values: Measured detector values
        
        Returns:
            Dictionary with calibration parameters
        """
        # Fit linear relationship
        result = self.fitter.fit(known_values, measured_values)
        
        if 'error' not in result:
            # Calculate linearity error
            fitted_values = result['fitted_curve']
            linearity_error = np.abs((measured_values - fitted_values) / fitted_values)
            max_linearity_error = np.max(linearity_error)
            rms_linearity_error = np.sqrt(np.mean(linearity_error ** 2))
            
            result.update({
                'max_linearity_error': max_linearity_error,
                'rms_linearity_error': rms_linearity_error,
                'gain': result['coefficients'][0],
                'offset': result['coefficients'][1]
            })
        
        return result
    
    def fit_energy_calibration(self, channel_numbers: np.ndarray, 
                               energies: np.ndarray) -> Dict[str, Any]:
        """
        Fit energy calibration curve.
        
        Args:
            channel_numbers: Detector channel numbers
            energies: Known energies (in keV)
        
        Returns:
            Dictionary with energy calibration parameters
        """
        # Use quadratic fit for energy calibration
        energy_fitter = PolynomialFitter(degree=2)
        result = energy_fitter.fit(channel_numbers, energies)
        
        if 'error' not in result:
            # Calculate energy resolution at different points
            fitted_energies = result['fitted_curve']
            energy_error = np.abs((energies - fitted_energies) / energies)
            max_energy_error = np.max(energy_error)
            rms_energy_error = np.sqrt(np.mean(energy_error ** 2))
            
            result.update({
                'max_energy_error': max_energy_error,
                'rms_energy_error': rms_energy_error,
                'calibration_coefficients': result['coefficients']
            })
        
        return result
