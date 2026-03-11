"""
Evaluation Metrics
Comprehensive metrics for detector performance evaluation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import warnings


class EvaluationMetrics:
    """Main class for calculating various evaluation metrics."""
    
    @staticmethod
    def mse(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((reference - test) ** 2)
    
    @staticmethod
    def rmse(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(EvaluationMetrics.mse(reference, test))
    
    @staticmethod
    def mae(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(reference - test))
    
    @staticmethod
    def psnr(reference: np.ndarray, test: np.ndarray, max_value: Optional[float] = None) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            reference: Reference signal
            test: Test signal
            max_value: Maximum possible signal value (calculated if None)
        
        Returns:
            PSNR in dB
        """
        if max_value is None:
            max_value = max(np.max(reference), np.max(test))
        
        mse = EvaluationMetrics.mse(reference, test)
        
        if mse == 0:
            return float('inf')
        
        return 20 * np.log10(max_value / np.sqrt(mse))
    
    @staticmethod
    def snr(signal: np.ndarray, noise: Optional[np.ndarray] = None) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            signal: Signal array
            noise: Noise array (estimated from signal if None)
        
        Returns:
            SNR in dB
        """
        if noise is None:
            # Estimate noise from high-frequency components
            from scipy import ndimage
            noise_estimate = signal - ndimage.gaussian_filter(signal, sigma=2)
            noise = noise_estimate
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def correlation_coefficient(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        return np.corrcoef(reference.flatten(), test.flatten())[0, 1]
    
    @staticmethod
    def structural_similarity(reference: np.ndarray, test: np.ndarray, 
                           data_range: Optional[float] = None) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            reference: Reference signal
            test: Test signal
            data_range: Data range of the input image (calculated if None)
        
        Returns:
            SSIM value
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if data_range is None:
                data_range = reference.max() - reference.min()
            
            if reference.ndim == 2:
                return ssim(reference, test, data_range=data_range)
            else:
                # For multi-dimensional data, calculate average SSIM
                return ssim(reference, test, data_range=data_range, 
                           multichannel=True if reference.ndim > 2 else False)
        
        except ImportError:
            # Fallback to simple correlation-based similarity
            warnings.warn("scikit-image not available, using correlation coefficient as fallback")
            return EvaluationMetrics.correlation_coefficient(reference, test)
    
    @staticmethod
    def peak_signal_to_noise_ratio(reference: np.ndarray, test: np.ndarray) -> float:
        """Alternative PSNR calculation using peak values."""
        peak_ref = np.max(reference)
        peak_test = np.max(test)
        peak_signal = max(peak_ref, peak_test)
        
        mse = EvaluationMetrics.mse(reference, test)
        if mse == 0:
            return float('inf')
        
        return 20 * np.log10(peak_signal / np.sqrt(mse))
    
    @staticmethod
    def normalized_root_mean_square_error(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Normalized Root Mean Square Error."""
        rmse = EvaluationMetrics.rmse(reference, test)
        data_range = np.max(reference) - np.min(reference)
        
        if data_range == 0:
            return 0.0
        
        return rmse / data_range
    
    @staticmethod
    def mean_absolute_percentage_error(reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = reference != 0
        if not np.any(mask):
            return 0.0
        
        return np.mean(np.abs((reference[mask] - test[mask]) / reference[mask])) * 100
    
    @staticmethod
    def compute_all_metrics(reference: np.ndarray, test: np.ndarray, 
                          max_value: Optional[float] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            reference: Reference signal
            test: Test signal
            max_value: Maximum signal value for PSNR
        
        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'MSE': EvaluationMetrics.mse(reference, test),
            'RMSE': EvaluationMetrics.rmse(reference, test),
            'MAE': EvaluationMetrics.mae(reference, test),
            'Correlation': EvaluationMetrics.correlation_coefficient(reference, test),
            'NRMSE': EvaluationMetrics.normalized_root_mean_square_error(reference, test),
            'MAPE': EvaluationMetrics.mean_absolute_percentage_error(reference, test)
        }
        
        # Add PSNR if max_value is provided or can be calculated
        try:
            metrics['PSNR'] = EvaluationMetrics.psnr(reference, test, max_value)
        except:
            metrics['PSNR'] = float('inf')
        
        # Add SNR
        try:
            metrics['SNR'] = EvaluationMetrics.snr(reference)
        except:
            metrics['SNR'] = float('inf')
        
        # Add SSIM if available
        try:
            metrics['SSIM'] = EvaluationMetrics.structural_similarity(reference, test)
        except:
            metrics['SSIM'] = EvaluationMetrics.correlation_coefficient(reference, test)
        
        return metrics


class PerformanceMetrics:
    """Specialized metrics for detector performance evaluation."""
    
    @staticmethod
    def detective_quantum_efficiency(signal: np.ndarray, noise: np.ndarray, 
                                   incident_photons: float) -> float:
        """
        Calculate Detective Quantum Efficiency (DQE).
        
        Args:
            signal: Measured signal
            noise: Measured noise
            incident_photons: Number of incident photons
        
        Returns:
            DQE value
        """
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        # Simplified DQE calculation
        if noise_power == 0:
            return 1.0
        
        return (signal_power / noise_power) / incident_photons
    
    @staticmethod
    def modulation_transfer_function(signal: np.ndarray, 
                                  edge_position: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Modulation Transfer Function (MTF).
        
        Args:
            signal: Edge spread function or signal with edge
            edge_position: Position of edge (detected if None)
        
        Returns:
            Tuple of (spatial_frequencies, mtf_values)
        """
        # Find edge position if not provided
        if edge_position is None:
            if signal.ndim == 1:
                edge_position = np.argmax(np.abs(np.diff(signal)))
            else:
                # Use middle row for 2D data
                middle_row = signal.shape[0] // 2
                edge_position = np.argmax(np.abs(np.diff(signal[middle_row, :])))
        
        # Extract edge spread function
        if signal.ndim == 1:
            esf = signal
        else:
            middle_row = signal.shape[0] // 2
            esf = signal[middle_row, :]
        
        # Calculate line spread function (derivative of ESF)
        lsf = np.gradient(esf)
        
        # Calculate MTF (FFT of LSF)
        mtf = np.abs(np.fft.fft(lsf))
        mtf = mtf[:len(mtf)//2]  # Take positive frequencies only
        
        # Normalize MTF
        mtf = mtf / mtf[0] if mtf[0] != 0 else mtf
        
        # Spatial frequencies
        sampling_rate = 1.0  # Normalized sampling rate
        frequencies = np.fft.fftfreq(len(lsf), d=1/sampling_rate)[:len(mtf)]
        
        return frequencies, mtf
    
    @staticmethod
    def noise_power_spectrum(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Noise Power Spectrum (NPS).
        
        Args:
            signal: Input signal
        
        Returns:
            Tuple of (frequencies, nps_values)
        """
        # Remove mean to isolate noise
        noise_signal = signal - np.mean(signal)
        
        # Calculate 2D FFT
        fft_signal = np.fft.fft2(noise_signal)
        nps = np.abs(fft_signal) ** 2
        
        # Radial averaging
        center = nps.shape[0] // 2
        y, x = np.indices(nps.shape)
        r = np.sqrt((x - center)**2 + (y - center)**2)
        r = r.astype(int)
        
        # Bin the NPS radially
        max_radius = min(center, nps.shape[1] // 2)
        nps_radial = np.zeros(max_radius)
        
        for i in range(max_radius):
            mask = (r == i)
            if np.any(mask):
                nps_radial[i] = np.mean(nps[mask])
        
        # Frequency values
        frequencies = np.fft.fftfreq(nps.shape[0])[:max_radius]
        
        return frequencies, nps_radial
    
    @staticmethod
    def linearity_error(input_values: np.ndarray, output_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate linearity error metrics.
        
        Args:
            input_values: Known input values
            output_values: Measured output values
        
        Returns:
            Dictionary with linearity metrics
        """
        # Fit linear relationship
        slope, intercept, r_value, p_value, std_err = stats.linregress(input_values, output_values)
        
        # Calculate fitted values
        fitted_values = slope * input_values + intercept
        
        # Calculate percentage error
        percentage_error = np.abs((output_values - fitted_values) / fitted_values) * 100
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'standard_error': std_err,
            'max_linearity_error': np.max(percentage_error),
            'mean_linearity_error': np.mean(percentage_error),
            'rms_linearity_error': np.sqrt(np.mean(percentage_error ** 2))
        }
    
    @staticmethod
    def uniformity_metric(signal: np.ndarray) -> Dict[str, float]:
        """
        Calculate uniformity metrics for flat field images.
        
        Args:
            signal: Flat field signal
        
        Returns:
            Dictionary with uniformity metrics
        """
        # Overall statistics
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        
        # Uniformity metrics
        uniformity = (1 - std_signal / mean_signal) * 100 if mean_signal != 0 else 0
        
        # Regional uniformity (divide image into regions)
        regions = []
        if signal.ndim == 2:
            # Divide into 4 quadrants
            h, w = signal.shape
            regions = [
                signal[:h//2, :w//2],
                signal[:h//2, w//2:],
                signal[h//2:, :w//2],
                signal[h//2:, w//2:]
            ]
        
        region_means = [np.mean(region) for region in regions]
        regional_uniformity = (1 - np.std(region_means) / np.mean(region_means)) * 100 if np.mean(region_means) != 0 else 0
        
        return {
            'mean': mean_signal,
            'std': std_signal,
            'uniformity_percentage': uniformity,
            'regional_uniformity_percentage': regional_uniformity,
            'min': np.min(signal),
            'max': np.max(signal),
            'range': np.max(signal) - np.min(signal)
        }
    
    @staticmethod
    def temporal_stability(signal_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate temporal stability metrics for time series data.
        
        Args:
            signal_series: Time series of signals (shape: time, height, width)
        
        Returns:
            Dictionary with stability metrics
        """
        if signal_series.ndim < 3:
            # Convert to 3D if 2D
            signal_series = signal_series[np.newaxis, ...]
        
        # Calculate mean signal over time for each pixel
        mean_signal = np.mean(signal_series, axis=0)
        
        # Calculate temporal variance for each pixel
        temporal_variance = np.var(signal_series, axis=0)
        
        # Overall stability metrics
        overall_mean = np.mean(mean_signal)
        overall_temporal_std = np.sqrt(np.mean(temporal_variance))
        
        stability = (1 - overall_temporal_std / overall_mean) * 100 if overall_mean != 0 else 0
        
        return {
            'overall_mean': overall_mean,
            'overall_temporal_std': overall_temporal_std,
            'stability_percentage': stability,
            'max_temporal_variation': np.max(np.sqrt(temporal_variance)),
            'mean_temporal_variation': np.mean(np.sqrt(temporal_variance))
        }
