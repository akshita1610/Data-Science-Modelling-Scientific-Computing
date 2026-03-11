"""
Comparison Tools
Tools for comparing signals, calibration results, and detector performance.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from .metrics import EvaluationMetrics, PerformanceMetrics


class SignalComparator:
    """Compare different signals and calculate similarity metrics."""
    
    def __init__(self):
        """Initialize signal comparator."""
        self.comparison_results = {}
    
    def compare_signals(self, reference: np.ndarray, test: np.ndarray, 
                       name: str = "comparison") -> Dict[str, Any]:
        """
        Compare two signals comprehensively.
        
        Args:
            reference: Reference signal
            test: Test signal
            name: Name for this comparison
        
        Returns:
            Dictionary with comparison results
        """
        # Basic metrics
        metrics = EvaluationMetrics.compute_all_metrics(reference, test)
        
        # Additional statistics
        reference_stats = self._compute_statistics(reference)
        test_stats = self._compute_statistics(test)
        
        # Difference statistics
        difference = reference - test
        diff_stats = self._compute_statistics(difference)
        
        # Combine results
        results = {
            'name': name,
            'metrics': metrics,
            'reference_statistics': reference_stats,
            'test_statistics': test_stats,
            'difference_statistics': diff_stats,
            'shape_comparison': {
                'reference_shape': reference.shape,
                'test_shape': test.shape,
                'shapes_match': reference.shape == test.shape
            }
        }
        
        self.comparison_results[name] = results
        return results
    
    def _compute_statistics(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive statistics for a signal."""
        return {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal)),
            'median': float(np.median(signal)),
            'q25': float(np.percentile(signal, 25)),
            'q75': float(np.percentile(signal, 75)),
            'skewness': float(self._compute_skewness(signal)),
            'kurtosis': float(self._compute_kurtosis(signal)),
            'entropy': float(self._compute_entropy(signal))
        }
    
    def _compute_skewness(self, signal: np.ndarray) -> float:
        """Compute skewness of signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 3)
    
    def _compute_kurtosis(self, signal: np.ndarray) -> float:
        """Compute kurtosis of signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, signal: np.ndarray) -> float:
        """Compute entropy of signal."""
        # Normalize signal to create histogram
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def compare_multiple_signals(self, signals: Dict[str, np.ndarray], 
                                reference_name: str) -> pd.DataFrame:
        """
        Compare multiple signals against a reference.
        
        Args:
            signals: Dictionary of signal_name -> signal_array
            reference_name: Name of reference signal
        
        Returns:
            DataFrame with comparison results
        """
        if reference_name not in signals:
            raise ValueError(f"Reference signal '{reference_name}' not found")
        
        reference = signals[reference_name]
        results = []
        
        for name, signal in signals.items():
            if name == reference_name:
                continue
            
            comparison = self.compare_signals(reference, signal, name)
            
            # Flatten metrics for DataFrame
            row = {'signal_name': name}
            row.update(comparison['metrics'])
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_comparison_summary(self, name: str) -> Dict[str, Any]:
        """Get summary of a specific comparison."""
        if name not in self.comparison_results:
            raise ValueError(f"Comparison '{name}' not found")
        
        results = self.comparison_results[name]
        
        # Create summary
        summary = {
            'name': name,
            'overall_quality': 'Good' if results['metrics']['SSIM'] > 0.8 else 'Poor',
            'key_metrics': {
                'PSNR': results['metrics']['PSNR'],
                'SSIM': results['metrics']['SSIM'],
                'RMSE': results['metrics']['RMSE']
            },
            'signal_difference': results['difference_statistics']['mean'],
            'relative_error': abs(results['difference_statistics']['mean'] / 
                                results['reference_statistics']['mean']) * 100
        }
        
        return summary


class CalibrationComparator:
    """Compare different calibration methods and results."""
    
    def __init__(self):
        """Initialize calibration comparator."""
        self.calibration_results = {}
    
    def add_calibration_result(self, name: str, raw_signal: np.ndarray,
                              calibrated_signal: np.ndarray,
                              reference_signal: Optional[np.ndarray] = None,
                              calibration_parameters: Optional[Dict[str, Any]] = None):
        """
        Add a calibration result for comparison.
        
        Args:
            name: Name of calibration method
            raw_signal: Original raw signal
            calibrated_signal: Calibrated signal
            reference_signal: Ground truth reference (optional)
            calibration_parameters: Parameters used for calibration
        """
        self.calibration_results[name] = {
            'raw_signal': raw_signal,
            'calibrated_signal': calibrated_signal,
            'reference_signal': reference_signal,
            'parameters': calibration_parameters or {}
        }
    
    def compare_calibration_methods(self) -> pd.DataFrame:
        """
        Compare all calibration methods.
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, calib_data in self.calibration_results.items():
            raw = calib_data['raw_signal']
            calibrated = calib_data['calibrated_signal']
            reference = calib_data.get('reference_signal')
            
            # Compare raw vs calibrated
            raw_vs_calib = EvaluationMetrics.compute_all_metrics(raw, calibrated)
            
            # Compare calibrated vs reference if available
            if reference is not None:
                calib_vs_ref = EvaluationMetrics.compute_all_metrics(reference, calibrated)
            else:
                calib_vs_ref = {key: np.nan for key in raw_vs_calib.keys()}
            
            # Calculate improvement metrics
            improvement = {}
            for key in ['RMSE', 'MAE']:
                if key in raw_vs_calib and key in calib_vs_ref:
                    improvement[f'{key}_improvement'] = (raw_vs_calib[key] - calib_vs_ref[key]) / raw_vs_calib[key] * 100
            
            row = {
                'method': name,
                'raw_psnr': raw_vs_calib.get('PSNR', np.nan),
                'calibrated_psnr': calib_vs_ref.get('PSNR', np.nan),
                'raw_ssim': raw_vs_calib.get('SSIM', np.nan),
                'calibrated_ssim': calib_vs_ref.get('SSIM', np.nan),
                'raw_rmse': raw_vs_calib.get('RMSE', np.nan),
                'calibrated_rmse': calib_vs_ref.get('RMSE', np.nan),
                **improvement
            }
            
            # Add calibration parameters
            for param, value in calib_data['parameters'].items():
                row[f'param_{param}'] = value
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def evaluate_calibration_quality(self, method_name: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a specific calibration method.
        
        Args:
            method_name: Name of calibration method
        
        Returns:
            Dictionary with quality evaluation
        """
        if method_name not in self.calibration_results:
            raise ValueError(f"Calibration method '{method_name}' not found")
        
        calib_data = self.calibration_results[method_name]
        raw = calib_data['raw_signal']
        calibrated = calib_data['calibrated_signal']
        reference = calib_data.get('reference_signal')
        
        evaluation = {
            'method': method_name,
            'signal_statistics': {
                'raw_mean': float(np.mean(raw)),
                'raw_std': float(np.std(raw)),
                'calibrated_mean': float(np.mean(calibrated)),
                'calibrated_std': float(np.std(calibrated))
            }
        }
        
        if reference is not None:
            # Quality metrics compared to reference
            metrics = EvaluationMetrics.compute_all_metrics(reference, calibrated)
            evaluation['quality_metrics'] = metrics
            
            # Quality score (weighted combination)
            quality_score = 0
            if 'SSIM' in metrics and not np.isnan(metrics['SSIM']):
                quality_score += metrics['SSIM'] * 0.4
            if 'PSNR' in metrics and not np.isnan(metrics['PSNR']):
                # Normalize PSNR to 0-1 range (assuming max PSNR of 50)
                quality_score += min(metrics['PSNR'] / 50, 1.0) * 0.3
            if 'Correlation' in metrics and not np.isnan(metrics['Correlation']):
                quality_score += metrics['Correlation'] * 0.3
            
            evaluation['quality_score'] = quality_score
            evaluation['quality_rating'] = self._get_quality_rating(quality_score)
        
        return evaluation
    
    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating based on score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def rank_calibration_methods(self) -> List[Tuple[str, float]]:
        """
        Rank calibration methods by quality.
        
        Returns:
            List of (method_name, quality_score) tuples sorted by score
        """
        rankings = []
        
        for method_name in self.calibration_results.keys():
            evaluation = self.evaluate_calibration_quality(method_name)
            quality_score = evaluation.get('quality_score', 0.0)
            rankings.append((method_name, quality_score))
        
        # Sort by quality score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def generate_calibration_report(self) -> str:
        """
        Generate a comprehensive calibration report.
        
        Returns:
            Formatted report string
        """
        if not self.calibration_results:
            return "No calibration results available for comparison."
        
        report = []
        report.append("=" * 60)
        report.append("CALIBRATION COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Method rankings
        rankings = self.rank_calibration_methods()
        report.append("METHOD RANKINGS:")
        for i, (method, score) in enumerate(rankings, 1):
            report.append(f"{i}. {method}: {score:.3f}")
        report.append("")
        
        # Detailed comparison table
        comparison_df = self.compare_calibration_methods()
        report.append("DETAILED COMPARISON:")
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best method details
        if rankings:
            best_method = rankings[0][0]
            best_evaluation = self.evaluate_calibration_quality(best_method)
            report.append(f"BEST METHOD DETAILS ({best_method}):")
            report.append(f"Quality Score: {best_evaluation.get('quality_score', 'N/A'):.3f}")
            report.append(f"Quality Rating: {best_evaluation.get('quality_rating', 'N/A')}")
            
            if 'quality_metrics' in best_evaluation:
                metrics = best_evaluation['quality_metrics']
                report.append(f"PSNR: {metrics.get('PSNR', 'N/A'):.2f} dB")
                report.append(f"SSIM: {metrics.get('SSIM', 'N/A'):.3f}")
                report.append(f"RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
