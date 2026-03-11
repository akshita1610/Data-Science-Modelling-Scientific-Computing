"""
Noise Reduction Methods
Various noise reduction and filtering techniques.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from scipy import ndimage, signal
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class NoiseReducer(ABC):
    """Abstract base class for noise reduction methods."""
    
    @abstractmethod
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Reduce noise in signal."""
        pass


class GaussianFilter(NoiseReducer):
    """Gaussian filter for noise reduction."""
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Gaussian filter.
        
        Args:
            sigma: Standard deviation for Gaussian kernel
        """
        self.sigma = sigma
    
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to reduce noise."""
        return ndimage.gaussian_filter(signal, sigma=self.sigma)


class MedianFilter(NoiseReducer):
    """Median filter for noise reduction."""
    
    def __init__(self, kernel_size: int = 3):
        """
        Initialize median filter.
        
        Args:
            kernel_size: Size of the median filter kernel
        """
        self.kernel_size = kernel_size
    
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply median filter to reduce noise."""
        return ndimage.median_filter(signal, size=self.kernel_size)


class WaveletDenoiser(NoiseReducer):
    """Wavelet-based denoising."""
    
    def __init__(self, wavelet: str = 'db4', sigma: Optional[float] = None):
        """
        Initialize wavelet denoiser.
        
        Args:
            wavelet: Wavelet type to use
            sigma: Noise standard deviation (estimated if None)
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet denoising. Install with: pip install PyWavelets")
        
        self.wavelet = wavelet
        self.sigma = sigma
    
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising."""
        if self.sigma is None:
            # Estimate noise from finest scale coefficients
            coeffs = pywt.wavedec2(signal, self.wavelet, level=1)
            self.sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Perform wavelet denoising
        return pywt.threshold(signal, self.sigma, mode='soft')


class AdaptiveFilter(NoiseReducer):
    """Adaptive filter that adjusts to local noise levels."""
    
    def __init__(self, window_size: int = 5, noise_variance: Optional[float] = None):
        """
        Initialize adaptive filter.
        
        Args:
            window_size: Size of the local window
            noise_variance: Estimated noise variance (calculated if None)
        """
        self.window_size = window_size
        self.noise_variance = noise_variance
    
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply adaptive Wiener filter."""
        if self.noise_variance is None:
            # Estimate noise variance from flat regions
            self.noise_variance = self._estimate_noise_variance(signal)
        
        # Apply Wiener filter
        return signal.wiener(signal, noise=self.noise_variance)
    
    def _estimate_noise_variance(self, signal: np.ndarray) -> float:
        """Estimate noise variance from signal."""
        # Simple estimation using Laplacian
        laplacian = ndimage.laplace(signal)
        return np.var(laplacian) / 6.0


class BilateralFilter(NoiseReducer):
    """Bilateral filter for edge-preserving denoising."""
    
    def __init__(self, sigma_spatial: float = 1.0, sigma_intensity: float = 0.1):
        """
        Initialize bilateral filter.
        
        Args:
            sigma_spatial: Spatial standard deviation
            sigma_intensity: Intensity standard deviation
        """
        self.sigma_spatial = sigma_spatial
        self.sigma_intensity = sigma_intensity
    
    def reduce_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply bilateral filter."""
        # Simple implementation using scipy's gaussian_filter
        # For a more efficient implementation, consider using OpenCV
        return self._bilateral_filter(signal)
    
    def _bilateral_filter(self, signal: np.ndarray) -> np.ndarray:
        """Simple bilateral filter implementation."""
        filtered = np.zeros_like(signal)
        rows, cols = signal.shape
        
        # Create spatial kernel
        kernel_size = int(2 * self.sigma_spatial) + 1
        spatial_kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist_sq = (i - center) ** 2 + (j - center) ** 2
                spatial_kernel[i, j] = np.exp(-dist_sq / (2 * self.sigma_spatial ** 2))
        
        # Apply filter
        for i in range(rows):
            for j in range(cols):
                # Define window
                i_min = max(0, i - center)
                i_max = min(rows, i + center + 1)
                j_min = max(0, j - center)
                j_max = min(cols, j + center + 1)
                
                window = signal[i_min:i_max, j_min:j_max]
                kernel_window = spatial_kernel[
                    (i_min - i + center):(i_max - i + center),
                    (j_min - j + center):(j_max - j + center)
                ]
                
                # Intensity weights
                intensity_diff = window - signal[i, j]
                intensity_weights = np.exp(-intensity_diff ** 2 / (2 * self.sigma_intensity ** 2))
                
                # Combined weights
                combined_weights = kernel_window * intensity_weights
                combined_weights /= np.sum(combined_weights)
                
                filtered[i, j] = np.sum(window * combined_weights)
        
        return filtered


class NoiseReductionPipeline:
    """Pipeline combining multiple noise reduction methods."""
    
    def __init__(self):
        """Initialize noise reduction pipeline."""
        self.methods: list[NoiseReducer] = []
        self.processing_history: list[dict] = []
    
    def add_method(self, method: NoiseReducer):
        """Add noise reduction method to pipeline."""
        self.methods.append(method)
    
    def clear_methods(self):
        """Remove all noise reduction methods."""
        self.methods.clear()
        self.processing_history.clear()
    
    def reduce_noise(self, signal: np.ndarray, save_history: bool = True) -> np.ndarray:
        """
        Apply full noise reduction pipeline.
        
        Args:
            signal: Input signal
            save_history: Whether to save processing history
        
        Returns:
            Denoised signal
        """
        processed = signal.copy()
        
        for i, method in enumerate(self.methods):
            processed = method.reduce_noise(processed)
            
            if save_history:
                self.processing_history.append({
                    'step': i,
                    'method': type(method).__name__,
                    'mean': np.mean(processed),
                    'std': np.std(processed),
                    'min': np.min(processed),
                    'max': np.max(processed)
                })
        
        return processed
    
    def get_processing_history(self) -> list[dict]:
        """Get history of noise reduction steps."""
        return self.processing_history.copy()


class SNREstimator:
    """Signal-to-Noise Ratio estimator and optimizer."""
    
    @staticmethod
    def estimate_snr(signal: np.ndarray, noise: Optional[np.ndarray] = None) -> float:
        """
        Estimate Signal-to-Noise Ratio.
        
        Args:
            signal: Input signal
            noise: Noise estimate (calculated from signal if None)
        
        Returns:
            Estimated SNR in dB
        """
        if noise is None:
            # Estimate noise from high-frequency components
            noise = ndimage.laplace(signal)
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    @staticmethod
    def optimize_denoising_parameters(signal: np.ndarray, 
                                    denoiser_class: type,
                                    parameter_range: dict) -> dict:
        """
        Optimize denoising parameters for maximum SNR.
        
        Args:
            signal: Input signal
            denoiser_class: Class of denoiser to optimize
            parameter_range: Dictionary of parameter ranges to test
        
        Returns:
            Dictionary with optimal parameters and resulting SNR
        """
        best_snr = -float('inf')
        best_params = {}
        
        # Generate parameter combinations
        param_names = list(parameter_range.keys())
        param_values = list(parameter_range.values())
        
        # Simple grid search (could be made more sophisticated)
        for params in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names)):
            param_dict = dict(zip(param_names, params))
            
            # Create denoiser and apply
            denoiser = denoiser_class(**param_dict)
            denoised = denoiser.reduce_noise(signal)
            
            # Calculate SNR
            snr = SNREstimator.estimate_snr(denoised, signal - denoised)
            
            if snr > best_snr:
                best_snr = snr
                best_params = param_dict.copy()
        
        return {
            'optimal_parameters': best_params,
            'best_snr': best_snr,
            'parameter_range': parameter_range
        }
