"""
Analysis Tools
Statistical analysis and quality assessment tools.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from scipy import stats
from .metrics import EvaluationMetrics, PerformanceMetrics


class StatisticalAnalyzer:
    """Statistical analysis tools for detector data."""
    
    @staticmethod
    def analyze_signal_distribution(signal: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive distribution analysis.
        
        Args:
            signal: Input signal
        
        Returns:
            Dictionary with distribution analysis results
        """
        # Basic statistics
        basic_stats = {
            'mean': np.mean(signal),
            'median': np.median(signal),
            'std': np.std(signal),
            'var': np.var(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'range': np.max(signal) - np.min(signal),
            'q25': np.percentile(signal, 25),
            'q75': np.percentile(signal, 75),
            'iqr': np.percentile(signal, 75) - np.percentile(signal, 25)
        }
        
        # Distribution fitting
        try:
            # Test normality
            _, p_value_normal = stats.normaltest(signal.flatten())
            is_normal = p_value_normal > 0.05
            
            # Fit normal distribution
            normal_params = stats.norm.fit(signal.flatten())
            
            # Fit other common distributions
            distributions = {
                'normal': normal_params,
                'exponential': stats.expon.fit(signal.flatten()),
                'gamma': stats.gamma.fit(signal.flatten())
            }
            
            # Calculate goodness of fit (AIC)
            aic_scores = {}
            for dist_name, params in distributions.items():
                if dist_name == 'normal':
                    dist = stats.norm
                elif dist_name == 'exponential':
                    dist = stats.expon
                elif dist_name == 'gamma':
                    dist = stats.gamma
                
                log_likelihood = np.sum(dist.logpdf(signal.flatten(), *params))
                k = len(params)  # Number of parameters
                aic_scores[dist_name] = 2 * k - 2 * log_likelihood
            
            best_fit = min(aic_scores, key=aic_scores.get)
            
        except Exception as e:
            is_normal = False
            distributions = {}
            aic_scores = {}
            best_fit = 'unknown'
        
        # Outlier detection
        outliers = StatisticalAnalyzer.detect_outliers(signal)
        
        return {
            'basic_statistics': basic_stats,
            'normality_test': {
                'is_normal': is_normal,
                'p_value': p_value_normal if 'p_value_normal' in locals() else None
            },
            'distribution_fits': distributions,
            'aic_scores': aic_scores,
            'best_fit_distribution': best_fit,
            'outliers': outliers
        }
    
    @staticmethod
    def detect_outliers(signal: np.ndarray, method: str = 'iqr', 
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in signal data.
        
        Args:
            signal: Input signal
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary with outlier information
        """
        flattened = signal.flatten()
        
        if method == 'iqr':
            q1 = np.percentile(flattened, 25)
            q3 = np.percentile(flattened, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (flattened < lower_bound) | (flattened > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(flattened))
            outlier_mask = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(flattened)
            mad = np.median(np.abs(flattened - median))
            modified_z_scores = 0.6745 * (flattened - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = flattened[outlier_mask]
        
        return {
            'method': method,
            'threshold': threshold,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(flattened) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_mask': outlier_mask.reshape(signal.shape)
        }
    
    @staticmethod
    def analyze_spatial_correlation(signal: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spatial correlation in 2D signal.
        
        Args:
            signal: 2D input signal
        
        Returns:
            Dictionary with spatial correlation analysis
        """
        if signal.ndim != 2:
            raise ValueError("Spatial correlation analysis requires 2D signal")
        
        # Autocorrelation function
        autocorr = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                if i == 0 and j == 0:
                    autocorr[i, j] = 1.0
                else:
                    shifted = np.roll(np.roll(signal, -i, axis=0), -j, axis=1)
                    autocorr[i, j] = np.corrcoef(signal.flatten(), shifted.flatten())[0, 1]
        
        # Correlation length (where autocorrelation drops to 1/e)
        center = signal.shape[0] // 2
        autocorr_profile = autocorr[center, center:]
        correlation_length = np.where(autocorr_profile < 1/np.e)[0]
        corr_length = correlation_length[0] if len(correlation_length) > 0 else signal.shape[1] // 2
        
        # Anisotropy analysis
        horizontal_corr = autocorr[center, center:]
        vertical_corr = autocorr[center:, center]
        anisotropy_ratio = np.mean(horizontal_corr[:10]) / np.mean(vertical_corr[:10]) if np.mean(vertical_corr[:10]) != 0 else 1.0
        
        return {
            'autocorrelation': autocorr,
            'correlation_length': corr_length,
            'anisotropy_ratio': anisotropy_ratio,
            'horizontal_correlation': horizontal_corr,
            'vertical_correlation': vertical_corr
        }
    
    @staticmethod
    def analyze_temporal_stability(signal_series: np.ndarray) -> Dict[str, Any]:
        """
        Analyze temporal stability of signal series.
        
        Args:
            signal_series: Time series of signals (shape: time, height, width)
        
        Returns:
            Dictionary with temporal stability analysis
        """
        if signal_series.ndim < 3:
            signal_series = signal_series[np.newaxis, ...]
        
        # Calculate statistics over time
        mean_signal = np.mean(signal_series, axis=0)
        std_signal = np.std(signal_series, axis=0)
        
        # Temporal coefficients of variation
        cov_signal = std_signal / mean_signal
        cov_signal[np.isnan(cov_signal)] = 0  # Handle division by zero
        
        # Drift analysis
        temporal_means = np.mean(signal_series, axis=(1, 2))
        drift_slope, _, _, _, _ = stats.linregress(np.arange(len(temporal_means)), temporal_means)
        
        # Periodicity analysis
        from scipy import signal as scipy_signal
        freqs, power = scipy_signal.periodogram(temporal_means)
        dominant_freq = freqs[np.argmax(power[1:])] + 1  # Exclude DC component
        
        return {
            'mean_signal': mean_signal,
            'std_signal': std_signal,
            'cov_signal': cov_signal,
            'temporal_means': temporal_means,
            'drift_slope': drift_slope,
            'dominant_frequency': dominant_freq,
            'overall_stability': 1 - np.mean(cov_signal),
            'drift_rate_per_frame': drift_slope
        }


class QualityAssessment:
    """Quality assessment tools for detector performance."""
    
    def __init__(self):
        """Initialize quality assessment."""
        self.quality_criteria = {
            'excellent': {'psnr': 40, 'ssim': 0.95, 'uniformity': 95},
            'good': {'psnr': 30, 'ssim': 0.85, 'uniformity': 90},
            'fair': {'psnr': 20, 'ssim': 0.75, 'uniformity': 80},
            'poor': {'psnr': 10, 'ssim': 0.6, 'uniformity': 70}
        }
    
    def assess_detector_quality(self, signal: np.ndarray, 
                               reference: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess overall detector quality.
        
        Args:
            signal: Detector signal
            reference: Reference signal (optional)
        
        Returns:
            Dictionary with quality assessment
        """
        assessment = {}
        
        # Signal quality metrics
        if reference is not None:
            metrics = EvaluationMetrics.compute_all_metrics(reference, signal)
            assessment['accuracy_metrics'] = metrics
        
        # Uniformity assessment
        uniformity_metrics = PerformanceMetrics.uniformity_metric(signal)
        assessment['uniformity'] = uniformity_metrics
        
        # Noise assessment
        noise_level = np.std(signal - np.mean(signal))
        signal_to_noise = EvaluationMetrics.snr(signal)
        assessment['noise_assessment'] = {
            'noise_level': noise_level,
            'snr_db': signal_to_noise,
            'noise_quality': self._assess_noise_quality(noise_level, signal_to_noise)
        }
        
        # Overall quality rating
        overall_rating = self._calculate_overall_quality(assessment)
        assessment['overall_quality'] = overall_rating
        
        return assessment
    
    def _assess_noise_quality(self, noise_level: float, snr: float) -> str:
        """Assess noise quality based on level and SNR."""
        if snr > 30:
            return "Excellent"
        elif snr > 20:
            return "Good"
        elif snr > 10:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_overall_quality(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality rating."""
        scores = []
        
        # Accuracy score (if reference available)
        if 'accuracy_metrics' in assessment:
            metrics = assessment['accuracy_metrics']
            if 'SSIM' in metrics and not np.isnan(metrics['SSIM']):
                scores.append(metrics['SSIM'] * 100)
            if 'PSNR' in metrics and not np.isnan(metrics['PSNR']):
                scores.append(min(metrics['PSNR'] / 40 * 100, 100))  # Normalize to 0-100
        
        # Uniformity score
        if 'uniformity' in assessment:
            uniformity = assessment['uniformity']['uniformity_percentage']
            scores.append(uniformity)
        
        # Noise score
        if 'noise_assessment' in assessment:
            snr = assessment['noise_assessment']['snr_db']
            noise_score = min(snr / 30 * 100, 100)  # Normalize to 0-100
            scores.append(noise_score)
        
        # Overall score
        overall_score = np.mean(scores) if scores else 0
        
        # Quality rating
        if overall_score >= 90:
            rating = "Excellent"
        elif overall_score >= 75:
            rating = "Good"
        elif overall_score >= 60:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            'overall_score': overall_score,
            'quality_rating': rating,
            'individual_scores': scores
        }
    
    def assess_calibration_quality(self, raw_signal: np.ndarray, 
                                 calibrated_signal: np.ndarray,
                                 reference_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess calibration quality.
        
        Args:
            raw_signal: Original raw signal
            calibrated_signal: Calibrated signal
            reference_signal: Ground truth reference
        
        Returns:
            Dictionary with calibration quality assessment
        """
        assessment = {}
        
        # Improvement metrics
        raw_metrics = EvaluationMetrics.compute_all_metrics(raw_signal, calibrated_signal)
        assessment['raw_vs_calibrated'] = raw_metrics
        
        if reference_signal is not None:
            # Compare both to reference
            raw_vs_ref = EvaluationMetrics.compute_all_metrics(reference_signal, raw_signal)
            calib_vs_ref = EvaluationMetrics.compute_all_metrics(reference_signal, calibrated_signal)
            
            assessment['raw_vs_reference'] = raw_vs_ref
            assessment['calibrated_vs_reference'] = calib_vs_ref
            
            # Calculate improvement
            improvement = {}
            for metric in ['RMSE', 'MAE']:
                if metric in raw_vs_ref and metric in calib_vs_ref:
                    improvement[metric] = (raw_vs_ref[metric] - calib_vs_ref[metric]) / raw_vs_ref[metric] * 100
            
            assessment['improvement'] = improvement
        
        # Linearity assessment
        if reference_signal is not None:
            linearity = PerformanceMetrics.linearity_error(
                reference_signal.flatten(), 
                calibrated_signal.flatten()
            )
            assessment['linearity'] = linearity
        
        # Overall calibration quality
        calibration_score = self._calculate_calibration_score(assessment)
        assessment['calibration_score'] = calibration_score
        
        return assessment
    
    def _calculate_calibration_score(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall calibration quality score."""
        scores = []
        
        # Improvement score
        if 'improvement' in assessment:
            improvements = assessment['improvement']
            if improvements:
                avg_improvement = np.mean(list(improvements.values()))
                scores.append(max(0, avg_improvement))  # Only positive improvements
        
        # Accuracy score
        if 'calibrated_vs_reference' in assessment:
            metrics = assessment['calibrated_vs_reference']
            if 'SSIM' in metrics and not np.isnan(metrics['SSIM']):
                scores.append(metrics['SSIM'] * 100)
            if 'PSNR' in metrics and not np.isnan(metrics['PSNR']):
                scores.append(min(metrics['PSNR'] / 40 * 100, 100))
        
        # Linearity score
        if 'linearity' in assessment:
            r_squared = assessment['linearity'].get('r_squared', 0)
            scores.append(r_squared * 100)
        
        overall_score = np.mean(scores) if scores else 0
        
        if overall_score >= 85:
            rating = "Excellent"
        elif overall_score >= 70:
            rating = "Good"
        elif overall_score >= 55:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            'overall_score': overall_score,
            'quality_rating': rating,
            'component_scores': scores
        }
    
    def generate_quality_report(self, assessment: Dict[str, Any]) -> str:
        """
        Generate a formatted quality report.
        
        Args:
            assessment: Quality assessment dictionary
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 50)
        report.append("QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall quality
        if 'overall_quality' in assessment:
            overall = assessment['overall_quality']
            report.append("OVERALL QUALITY:")
            report.append(f"  Score: {overall['overall_score']:.1f}/100")
            report.append(f"  Rating: {overall['quality_rating']}")
            report.append("")
        
        # Uniformity
        if 'uniformity' in assessment:
            uniformity = assessment['uniformity']
            report.append("UNIFORMITY:")
            report.append(f"  Uniformity: {uniformity['uniformity_percentage']:.1f}%")
            report.append(f"  Regional Uniformity: {uniformity['regional_uniformity_percentage']:.1f}%")
            report.append("")
        
        # Noise assessment
        if 'noise_assessment' in assessment:
            noise = assessment['noise_assessment']
            report.append("NOISE ASSESSMENT:")
            report.append(f"  Noise Level: {noise['noise_level']:.4f}")
            report.append(f"  SNR: {noise['snr_db']:.1f} dB")
            report.append(f"  Quality: {noise['noise_quality']}")
            report.append("")
        
        # Accuracy metrics
        if 'accuracy_metrics' in assessment:
            metrics = assessment['accuracy_metrics']
            report.append("ACCURACY METRICS:")
            report.append(f"  PSNR: {metrics.get('PSNR', 'N/A'):.2f} dB")
            report.append(f"  SSIM: {metrics.get('SSIM', 'N/A'):.3f}")
            report.append(f"  RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
            report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)
