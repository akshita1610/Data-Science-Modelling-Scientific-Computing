"""
Interactive Visualization Tools
Interactive plots for real-time detector analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import Optional, Callable, Dict, Any, Tuple
import matplotlib.patches as patches


class InteractivePlotter:
    """Interactive plotting tools for detector simulation."""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize interactive plotter.
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.sliders = {}
        self.buttons = {}
        self.current_signal = None
        self.original_signal = None
        self.update_callback = None
    
    def create_detector_calibration_interface(self, signal: np.ndarray, 
                                            calibration_func: Callable) -> plt.Figure:
        """
        Create interactive detector calibration interface.
        
        Args:
            signal: Input signal to calibrate
            calibration_func: Function that applies calibration with parameters
        
        Returns:
            matplotlib Figure object
        """
        self.original_signal = signal.copy()
        self.current_signal = signal.copy()
        self.calibration_func = calibration_func
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=self.figsize)
        
        # Main plot area
        ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        ax_hist = plt.subplot2grid((4, 4), (0, 3), rowspan=2)
        ax_profile = plt.subplot2grid((4, 4), (2, 3), rowspan=1)
        
        # Slider axes
        ax_gain = plt.subplot2grid((4, 4), (3, 0), colspan=1)
        ax_offset = plt.subplot2grid((4, 4), (3, 1), colspan=1)
        ax_noise = plt.subplot2grid((4, 4), (3, 2), colspan=1)
        
        # Button axes
        ax_reset = plt.subplot2grid((4, 4), (3, 3))
        
        self.axes = {
            'main': ax_main,
            'hist': ax_hist,
            'profile': ax_profile
        }
        
        # Create sliders
        self.sliders['gain'] = Slider(ax_gain, 'Gain', 0.1, 3.0, valinit=1.0)
        self.sliders['offset'] = Slider(ax_offset, 'Offset', -100, 100, valinit=0.0)
        self.sliders['noise'] = Slider(ax_noise, 'Noise\nReduction', 0.0, 2.0, valinit=0.0)
        
        # Create reset button
        self.buttons['reset'] = Button(ax_reset, 'Reset')
        
        # Initial plot
        self._update_calibration_plot()
        
        # Connect slider events
        for slider in self.sliders.values():
            slider.on_changed(self._on_slider_change)
        
        self.buttons['reset'].on_clicked(self._reset_parameters)
        
        plt.tight_layout()
        return self.fig
    
    def _on_slider_change(self, val):
        """Handle slider change events."""
        self._update_calibration_plot()
        self.fig.canvas.draw_idle()
    
    def _reset_parameters(self, event):
        """Reset all parameters to default values."""
        self.sliders['gain'].set_val(1.0)
        self.sliders['offset'].set_val(0.0)
        self.sliders['noise'].set_val(0.0)
    
    def _update_calibration_plot(self):
        """Update the calibration plot based on current parameters."""
        # Get current parameters
        gain = self.sliders['gain'].val
        offset = self.sliders['offset'].val
        noise_level = self.sliders['noise'].val
        
        # Apply calibration
        calibrated = self.original_signal * gain + offset
        
        # Apply noise reduction if specified
        if noise_level > 0:
            from scipy import ndimage
            calibrated = ndimage.gaussian_filter(calibrated, sigma=noise_level)
        
        self.current_signal = calibrated
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Main plot
        im = self.axes['main'].imshow(calibrated, cmap='viridis')
        self.axes['main'].set_title(f'Calibrated Signal (Gain: {gain:.2f}, Offset: {offset:.1f})')
        self.axes['main'].set_xlabel('X Pixel')
        self.axes['main'].set_ylabel('Y Pixel')
        
        # Histogram
        self.axes['hist'].hist(calibrated.flatten(), bins=50, alpha=0.7, edgecolor='black')
        self.axes['hist'].set_title('Signal Distribution')
        self.axes['hist'].set_xlabel('Signal Intensity')
        self.axes['hist'].set_ylabel('Pixel Count')
        self.axes['hist'].grid(True, alpha=0.3)
        
        # Profile
        middle_row = calibrated.shape[0] // 2
        profile = calibrated[middle_row, :]
        self.axes['profile'].plot(profile)
        self.axes['profile'].set_title('Horizontal Profile')
        self.axes['profile'].set_xlabel('Pixel Position')
        self.axes['profile'].set_ylabel('Signal Intensity')
        self.axes['profile'].grid(True, alpha=0.3)
    
    def create_noise_analysis_interface(self, signal: np.ndarray) -> plt.Figure:
        """
        Create interactive noise analysis interface.
        
        Args:
            signal: Input signal to analyze
        
        Returns:
            matplotlib Figure object
        """
        self.original_signal = signal.copy()
        self.current_signal = signal.copy()
        
        # Create figure
        self.fig = plt.figure(figsize=self.figsize)
        
        # Main plot area
        ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        ax_noise = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        ax_psd = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=1)
        
        # Control axes
        ax_filter = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        ax_method = plt.subplot2grid((4, 4), (3, 2), colspan=1)
        ax_apply = plt.subplot2grid((4, 4), (3, 3), colspan=1)
        
        self.axes = {
            'main': ax_main,
            'noise': ax_noise,
            'psd': ax_psd
        }
        
        # Create controls
        self.sliders['filter'] = Slider(ax_filter, 'Filter\nStrength', 0.0, 5.0, valinit=1.0)
        self.buttons['apply'] = Button(ax_apply, 'Apply Filter')
        
        # Radio buttons for filter method
        self.radio = RadioButtons(ax_method, ('Gaussian', 'Median', 'Bilateral'))
        
        # Initial plot
        self._update_noise_analysis_plot()
        
        # Connect events
        self.sliders['filter'].on_changed(self._on_filter_change)
        self.buttons['apply'].on_clicked(self._apply_filter)
        self.radio.on_clicked(self._on_filter_change)
        
        plt.tight_layout()
        return self.fig
    
    def _on_filter_change(self, val):
        """Handle filter parameter change."""
        # Preview without applying
        self._update_noise_analysis_plot(preview=True)
        self.fig.canvas.draw_idle()
    
    def _apply_filter(self, event):
        """Apply the selected filter."""
        self._update_noise_analysis_plot(preview=False)
        self.fig.canvas.draw_idle()
    
    def _update_noise_analysis_plot(self, preview: bool = True):
        """Update noise analysis plot."""
        # Get filter parameters
        filter_strength = self.sliders['filter'].val
        filter_method = self.radio.value_selected
        
        # Apply filter
        if preview:
            # Use original signal for preview
            signal_to_filter = self.original_signal
        else:
            # Use current signal
            signal_to_filter = self.current_signal
        
        from scipy import ndimage
        
        if filter_method == 'Gaussian':
            filtered = ndimage.gaussian_filter(signal_to_filter, sigma=filter_strength)
        elif filter_method == 'Median':
            filtered = ndimage.median_filter(signal_to_filter, size=int(filter_strength) + 1)
        else:  # Bilateral (simplified)
            filtered = ndimage.gaussian_filter(signal_to_filter, sigma=filter_strength)
        
        if not preview:
            self.current_signal = filtered
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Main plot
        im1 = self.axes['main'].imshow(signal_to_filter, cmap='viridis')
        self.axes['main'].set_title('Original Signal' if preview else 'Filtered Signal')
        self.axes['main'].set_xlabel('X Pixel')
        self.axes['main'].set_ylabel('Y Pixel')
        
        # Noise map
        noise_map = signal_to_filter - filtered
        im2 = self.axes['noise'].imshow(noise_map, cmap='RdBu_r')
        self.axes['noise'].set_title('Noise Map')
        self.axes['noise'].set_xlabel('X Pixel')
        self.axes['noise'].set_ylabel('Y Pixel')
        
        # Power spectral density
        fft_signal = np.fft.fft2(signal_to_filter)
        fft_filtered = np.fft.fft2(filtered)
        
        # Radial averaging
        def radial_profile(data):
            center = data.shape[0] // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center)**2 + (y - center)**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / nr
            return radialprofile[:len(radialprofile)//2]
        
        psd_signal = radial_profile(np.abs(fft_signal)**2)
        psd_filtered = radial_profile(np.abs(fft_filtered)**2)
        
        self.axes['psd'].semilogy(psd_signal, label='Original', alpha=0.7)
        self.axes['psd'].semilogy(psd_filtered, label='Filtered', alpha=0.7)
        self.axes['psd'].set_title('Power Spectral Density')
        self.axes['psd'].set_xlabel('Spatial Frequency')
        self.axes['psd'].set_ylabel('Power')
        self.axes['psd'].legend()
        self.axes['psd'].grid(True, alpha=0.3)
    
    def create_roi_selector(self, signal: np.ndarray, 
                           roi_callback: Optional[Callable] = None) -> plt.Figure:
        """
        Create interactive ROI (Region of Interest) selector.
        
        Args:
            signal: Input signal
            roi_callback: Callback function when ROI is selected
        
        Returns:
            matplotlib Figure object
        """
        self.original_signal = signal.copy()
        self.roi_callback = roi_callback
        self.roi_patches = []
        self.current_roi = None
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Display signal
        self.im = self.ax.imshow(signal, cmap='viridis')
        self.ax.set_title('Click and drag to select ROI')
        self.ax.set_xlabel('X Pixel')
        self.ax.set_ylabel('Y Pixel')
        
        # Add colorbar
        plt.colorbar(self.im, ax=self.ax)
        
        # ROI selection state
        self.selecting = False
        self.start_point = None
        self.current_rect = None
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        
        return self.fig
    
    def _on_mouse_press(self, event):
        """Handle mouse press for ROI selection."""
        if event.inaxes != self.ax:
            return
        
        self.selecting = True
        self.start_point = (event.xdata, event.ydata)
        
        # Create new rectangle
        self.current_rect = patches.Rectangle(
            self.start_point, 0, 0, linewidth=2, 
            edgecolor='red', facecolor='none'
        )
        self.ax.add_patch(self.current_rect)
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion for ROI selection."""
        if not self.selecting or event.inaxes != self.ax:
            return
        
        # Update rectangle size
        width = event.xdata - self.start_point[0]
        height = event.ydata - self.start_point[1]
        
        self.current_rect.set_width(width)
        self.current_rect.set_height(height)
        
        self.fig.canvas.draw_idle()
    
    def _on_mouse_release(self, event):
        """Handle mouse release for ROI selection."""
        if not self.selecting:
            return
        
        self.selecting = False
        
        # Get ROI coordinates
        x1, y1 = self.start_point
        x2, y2 = event.xdata, event.ydata
        
        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Convert to pixel indices
        x_start, x_end = int(x_min), int(x_max)
        y_start, y_end = int(y_min), int(y_max)
        
        # Extract ROI
        roi_data = self.original_signal[y_start:y_end, x_start:x_end]
        
        # Store ROI
        self.current_roi = {
            'coordinates': (x_start, y_start, x_end, y_end),
            'data': roi_data,
            'statistics': {
                'mean': np.mean(roi_data),
                'std': np.std(roi_data),
                'min': np.min(roi_data),
                'max': np.max(roi_data),
                'sum': np.sum(roi_data)
            }
        }
        
        # Change rectangle color to indicate completion
        self.current_rect.set_edgecolor('green')
        self.roi_patches.append(self.current_rect)
        
        # Call callback if provided
        if self.roi_callback:
            self.roi_callback(self.current_roi)
        
        # Print ROI statistics
        print(f"ROI Selected: ({x_start}, {y_start}) to ({x_end}, {y_end})")
        print(f"ROI Statistics: Mean={self.current_roi['statistics']['mean']:.2f}, "
              f"Std={self.current_roi['statistics']['std']:.2f}")
        
        self.fig.canvas.draw_idle()
    
    def get_selected_rois(self) -> list:
        """Get list of selected ROIs."""
        return self.roi_patches
    
    def clear_rois(self):
        """Clear all selected ROIs."""
        for patch in self.roi_patches:
            patch.remove()
        self.roi_patches.clear()
        if self.current_rect:
            self.current_rect.remove()
            self.current_rect = None
        self.fig.canvas.draw_idle()
