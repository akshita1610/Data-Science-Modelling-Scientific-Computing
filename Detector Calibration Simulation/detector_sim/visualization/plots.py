"""
Plotting and Visualization Tools
Comprehensive visualization for detector simulation and calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from typing import Optional, Tuple, List, Dict, Any
import seaborn as sns


class DetectorPlotter:
    """Main plotting class for detector visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize detector plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_detector_response(self, signal: np.ndarray, title: str = "Detector Response",
                              cmap: str = 'viridis', use_log: bool = False) -> plt.Figure:
        """
        Plot 2D detector response.
        
        Args:
            signal: 2D detector signal
            title: Plot title
            cmap: Colormap to use
            use_log: Whether to use logarithmic scale
        
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 2D heatmap
        if use_log and np.min(signal) > 0:
            im = ax1.imshow(signal, cmap=cmap, norm=LogNorm())
        else:
            im = ax1.imshow(signal, cmap=cmap)
        
        ax1.set_title(f"{title} - 2D View")
        ax1.set_xlabel("X Pixel")
        ax1.set_ylabel("Y Pixel")
        plt.colorbar(im, ax=ax1, label="Signal Intensity")
        
        # Signal profile
        # Horizontal profile (middle row)
        middle_row = signal.shape[0] // 2
        horizontal_profile = signal[middle_row, :]
        
        # Vertical profile (middle column)
        middle_col = signal.shape[1] // 2
        vertical_profile = signal[:, middle_col]
        
        ax2.plot(horizontal_profile, label='Horizontal Profile', alpha=0.7)
        ax2.plot(vertical_profile, label='Vertical Profile', alpha=0.7)
        ax2.set_title(f"{title} - Signal Profiles")
        ax2.set_xlabel("Pixel Position")
        ax2.set_ylabel("Signal Intensity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_signal_comparison(self, signals: List[np.ndarray], 
                              labels: List[str], title: str = "Signal Comparison") -> plt.Figure:
        """
        Plot multiple signals for comparison.
        
        Args:
            signals: List of 2D signal arrays
            labels: List of labels for each signal
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(2, n_signals, figsize=(4*n_signals, 8))
        
        if n_signals == 1:
            axes = axes.reshape(2, 1)
        
        for i, (signal, label) in enumerate(zip(signals, labels)):
            # 2D view
            im1 = axes[0, i].imshow(signal, cmap='viridis')
            axes[0, i].set_title(f"{label} - 2D View")
            axes[0, i].set_xlabel("X Pixel")
            axes[0, i].set_ylabel("Y Pixel")
            plt.colorbar(im1, ax=axes[0, i])
            
            # Histogram
            axes[1, i].hist(signal.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, i].set_title(f"{label} - Histogram")
            axes[1, i].set_xlabel("Signal Intensity")
            axes[1, i].set_ylabel("Pixel Count")
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_noise_analysis(self, signal: np.ndarray, 
                           title: str = "Noise Analysis") -> plt.Figure:
        """
        Plot noise analysis of detector signal.
        
        Args:
            signal: 2D detector signal
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Original signal
        im1 = axes[0, 0].imshow(signal, cmap='viridis')
        axes[0, 0].set_title("Original Signal")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Noise map (high-pass filtered)
        from scipy import ndimage
        noise_map = signal - ndimage.gaussian_filter(signal, sigma=2)
        im2 = axes[0, 1].imshow(noise_map, cmap='RdBu_r')
        axes[0, 1].set_title("Noise Map")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Noise histogram
        noise_values = noise_map.flatten()
        axes[1, 0].hist(noise_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title("Noise Distribution")
        axes[1, 0].set_xlabel("Noise Value")
        axes[1, 0].set_ylabel("Pixel Count")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add Gaussian fit
        mu, sigma = np.mean(noise_values), np.std(noise_values)
        x = np.linspace(noise_values.min(), noise_values.max(), 100)
        gaussian = (len(noise_values) * (noise_values.max() - noise_values.min()) / 50) * \
                   np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        axes[1, 0].plot(x, gaussian, 'r-', linewidth=2, 
                       label=f'Gaussian fit (μ={mu:.3f}, σ={sigma:.3f})')
        axes[1, 0].legend()
        
        # Spatial noise analysis
        row_noise = np.std(signal, axis=1)
        col_noise = np.std(signal, axis=0)
        
        axes[1, 1].plot(row_noise, label='Row-wise noise', alpha=0.7)
        axes[1, 1].plot(col_noise, label='Column-wise noise', alpha=0.7)
        axes[1, 1].set_title("Spatial Noise Analysis")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Noise (std)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_detector_statistics(self, signal: np.ndarray, 
                                title: str = "Detector Statistics") -> plt.Figure:
        """
        Plot comprehensive detector statistics.
        
        Args:
            signal: 2D detector signal
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Signal histogram
        axes[0, 0].hist(signal.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title("Signal Histogram")
        axes[0, 0].set_xlabel("Signal Intensity")
        axes[0, 0].set_ylabel("Pixel Count")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_signal = np.sort(signal.flatten())
        cumulative = np.arange(1, len(sorted_signal) + 1) / len(sorted_signal)
        axes[0, 1].plot(sorted_signal, cumulative)
        axes[0, 1].set_title("Cumulative Distribution")
        axes[0, 1].set_xlabel("Signal Intensity")
        axes[0, 1].set_ylabel("Cumulative Probability")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Signal statistics box plot
        axes[0, 2].boxplot(signal.flatten())
        axes[0, 2].set_title("Signal Statistics")
        axes[0, 2].set_ylabel("Signal Intensity")
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row-wise statistics
        row_means = np.mean(signal, axis=1)
        row_stds = np.std(signal, axis=1)
        
        ax2_twin = axes[1, 0].twinx()
        axes[1, 0].plot(row_means, 'b-', label='Row Mean', alpha=0.7)
        ax2_twin.plot(row_stds, 'r-', label='Row Std', alpha=0.7)
        axes[1, 0].set_title("Row-wise Statistics")
        axes[1, 0].set_xlabel("Row Index")
        axes[1, 0].set_ylabel("Mean", color='b')
        ax2_twin.set_ylabel("Std Dev", color='r')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Column-wise statistics
        col_means = np.mean(signal, axis=0)
        col_stds = np.std(signal, axis=0)
        
        ax3_twin = axes[1, 1].twinx()
        axes[1, 1].plot(col_means, 'b-', label='Column Mean', alpha=0.7)
        ax3_twin.plot(col_stds, 'r-', label='Column Std', alpha=0.7)
        axes[1, 1].set_title("Column-wise Statistics")
        axes[1, 1].set_xlabel("Column Index")
        axes[1, 1].set_ylabel("Mean", color='b')
        ax3_twin.set_ylabel("Std Dev", color='r')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 2D statistics heatmap
        stats_map = np.zeros_like(signal)
        window_size = 5
        for i in range(0, signal.shape[0] - window_size + 1, window_size):
            for j in range(0, signal.shape[1] - window_size + 1, window_size):
                window = signal[i:i+window_size, j:j+window_size]
                stats_map[i:i+window_size, j:j+window_size] = np.std(window)
        
        im4 = axes[1, 2].imshow(stats_map, cmap='hot')
        axes[1, 2].set_title("Local Standard Deviation")
        axes[1, 2].set_xlabel("X Pixel")
        axes[1, 2].set_ylabel("Y Pixel")
        plt.colorbar(im4, ax=axes[1, 2])
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


class CalibrationPlotter:
    """Specialized plotting for calibration analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize calibration plotter."""
        self.figsize = figsize
    
    def plot_calibration_comparison(self, raw_signal: np.ndarray, 
                                  calibrated_signal: np.ndarray,
                                  title: str = "Calibration Comparison") -> plt.Figure:
        """
        Plot comparison between raw and calibrated signals.
        
        Args:
            raw_signal: Raw detector signal
            calibrated_signal: Calibrated signal
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Raw signal
        im1 = axes[0, 0].imshow(raw_signal, cmap='viridis')
        axes[0, 0].set_title("Raw Signal")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Calibrated signal
        im2 = axes[0, 1].imshow(calibrated_signal, cmap='viridis')
        axes[0, 1].set_title("Calibrated Signal")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference
        difference = calibrated_signal - raw_signal
        im3 = axes[0, 2].imshow(difference, cmap='RdBu_r')
        axes[0, 2].set_title("Calibration Effect")
        axes[0, 2].set_xlabel("X Pixel")
        axes[0, 2].set_ylabel("Y Pixel")
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Histograms
        axes[1, 0].hist(raw_signal.flatten(), bins=50, alpha=0.7, 
                       label='Raw', edgecolor='black')
        axes[1, 0].hist(calibrated_signal.flatten(), bins=50, alpha=0.7, 
                       label='Calibrated', edgecolor='black')
        axes[1, 0].set_title("Signal Distributions")
        axes[1, 0].set_xlabel("Signal Intensity")
        axes[1, 0].set_ylabel("Pixel Count")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1, 1].scatter(raw_signal.flatten()[::100], 
                          calibrated_signal.flatten()[::100], 
                          alpha=0.5, s=1)
        axes[1, 1].plot([raw_signal.min(), raw_signal.max()], 
                        [raw_signal.min(), raw_signal.max()], 
                        'r--', label='Identity')
        axes[1, 1].set_title("Raw vs Calibrated")
        axes[1, 1].set_xlabel("Raw Signal")
        axes[1, 1].set_ylabel("Calibrated Signal")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Calibration curve (profiles)
        middle_row = raw_signal.shape[0] // 2
        raw_profile = raw_signal[middle_row, :]
        calib_profile = calibrated_signal[middle_row, :]
        
        axes[1, 2].plot(raw_profile, label='Raw Profile', alpha=0.7)
        axes[1, 2].plot(calib_profile, label='Calibrated Profile', alpha=0.7)
        axes[1, 2].set_title("Signal Profiles")
        axes[1, 2].set_xlabel("Pixel Position")
        axes[1, 2].set_ylabel("Signal Intensity")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_calibration_history(self, history: List[Dict[str, Any]], 
                                title: str = "Calibration History") -> plt.Figure:
        """
        Plot calibration step history.
        
        Args:
            history: List of calibration step dictionaries
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        if not history:
            raise ValueError("Empty calibration history")
        
        steps = [h['step'] for h in history]
        means = [h['mean'] for h in history]
        stds = [h['std'] for h in history]
        mins = [h['min'] for h in history]
        maxs = [h['max'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Mean values
        axes[0, 0].plot(steps, means, 'o-', label='Mean')
        axes[0, 0].set_title("Mean Signal vs Calibration Step")
        axes[0, 0].set_xlabel("Calibration Step")
        axes[0, 0].set_ylabel("Mean Signal")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Standard deviation
        axes[0, 1].plot(steps, stds, 'o-', label='Std Dev', color='orange')
        axes[0, 1].set_title("Noise Level vs Calibration Step")
        axes[0, 1].set_xlabel("Calibration Step")
        axes[0, 1].set_ylabel("Standard Deviation")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Min/Max values
        axes[1, 0].plot(steps, mins, 'o-', label='Min', color='green')
        axes[1, 0].plot(steps, maxs, 'o-', label='Max', color='red')
        axes[1, 0].set_title("Signal Range vs Calibration Step")
        axes[1, 0].set_xlabel("Calibration Step")
        axes[1, 0].set_ylabel("Signal Value")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Signal-to-noise ratio
        snr = [m/s if s > 0 else 0 for m, s in zip(means, stds)]
        axes[1, 1].plot(steps, snr, 'o-', label='SNR', color='purple')
        axes[1, 1].set_title("Signal-to-Noise Ratio vs Calibration Step")
        axes[1, 1].set_xlabel("Calibration Step")
        axes[1, 1].set_ylabel("SNR")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


class NoiseAnalysisPlotter:
    """Specialized plotting for noise analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize noise analysis plotter."""
        self.figsize = figsize
    
    def plot_noise_reduction_comparison(self, original: np.ndarray, 
                                       denoised: np.ndarray,
                                       title: str = "Noise Reduction Comparison") -> plt.Figure:
        """
        Plot comparison between original and denoised signals.
        
        Args:
            original: Original noisy signal
            denoised: Denoised signal
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original signal
        im1 = axes[0, 0].imshow(original, cmap='viridis')
        axes[0, 0].set_title("Original Signal")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Denoised signal
        im2 = axes[0, 1].imshow(denoised, cmap='viridis')
        axes[0, 1].set_title("Denoised Signal")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Noise removed
        noise_removed = original - denoised
        im3 = axes[0, 2].imshow(noise_removed, cmap='RdBu_r')
        axes[0, 2].set_title("Removed Noise")
        axes[0, 2].set_xlabel("X Pixel")
        axes[0, 2].set_ylabel("Y Pixel")
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Power spectral density comparison
        from scipy import signal as scipy_signal
        
        # Compute 2D FFT
        fft_original = np.fft.fft2(original)
        fft_denoised = np.fft.fft2(denoised)
        
        # Radial averaging for power spectrum
        def radial_profile(data):
            center = data.shape[0] // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center)**2 + (y - center)**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / nr
            return radialprofile
        
        psd_original = radial_profile(np.abs(fft_original)**2)
        psd_denoised = radial_profile(np.abs(fft_denoised)**2)
        
        axes[1, 0].semilogy(psd_original[:len(psd_original)//2], 
                           label='Original', alpha=0.7)
        axes[1, 0].semilogy(psd_denoised[:len(psd_denoised)//2], 
                           label='Denoised', alpha=0.7)
        axes[1, 0].set_title("Power Spectral Density")
        axes[1, 0].set_xlabel("Spatial Frequency")
        axes[1, 0].set_ylabel("Power")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Noise histograms
        original_noise = original - np.mean(original)
        denoised_noise = denoised - np.mean(denoised)
        
        axes[1, 1].hist(original_noise.flatten(), bins=50, alpha=0.7, 
                       label='Original', density=True, edgecolor='black')
        axes[1, 1].hist(denoised_noise.flatten(), bins=50, alpha=0.7, 
                       label='Denoised', density=True, edgecolor='black')
        axes[1, 1].set_title("Noise Distributions")
        axes[1, 1].set_xlabel("Noise Value")
        axes[1, 1].set_ylabel("Probability Density")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # SNR improvement
        def calculate_snr(signal):
            return np.mean(signal) / np.std(signal)
        
        original_snr = calculate_snr(original)
        denoised_snr = calculate_snr(denoised)
        
        methods = ['Original', 'Denoised']
        snr_values = [original_snr, denoised_snr]
        
        axes[1, 2].bar(methods, snr_values, alpha=0.7, color=['red', 'green'])
        axes[1, 2].set_title("Signal-to-Noise Ratio")
        axes[1, 2].set_ylabel("SNR")
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add SNR values as text
        for i, v in enumerate(snr_values):
            axes[1, 2].text(i, v + max(snr_values)*0.01, f'{v:.2f}', 
                           ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
