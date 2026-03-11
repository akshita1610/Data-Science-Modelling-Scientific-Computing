#!/usr/bin/env python3
"""
Interactive Demo for Detector Calibration Simulation

This example demonstrates the interactive visualization features:
- Interactive calibration interface with sliders
- Real-time parameter adjustment
- ROI selection tools
- Interactive noise analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the detector_sim package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detector_sim.models.detector import PixelDetector
from detector_sim.models.noise_models import GaussianNoise
from detector_sim.simulation.signal_sources import PointSource, GaussianSource
from detector_sim.simulation.signal_generator import SignalGenerator
from detector_sim.visualization.interactive import InteractivePlotter


def create_demo_signal():
    """Create a demonstration signal with multiple features."""
    signal_gen = SignalGenerator(width=150, height=150)
    
    # Add multiple point sources
    signal_gen.add_source(PointSource(x=75, y=75, intensity=3.0))
    signal_gen.add_source(PointSource(x=50, y=50, intensity=2.0))
    signal_gen.add_source(PointSource(x=100, y=100, intensity=2.5))
    
    # Add Gaussian source
    signal_gen.add_source(GaussianSource(
        center_x=75, center_y=75,
        sigma_x=15, sigma_y=15,
        intensity=1.5
    ))
    
    # Add uniform background
    signal_gen.add_source(PointSource(x=0, y=0, intensity=0.1))  # Will be broadcasted
    
    return signal_gen.generate_signal()


def demo_interactive_calibration():
    """Demonstrate interactive calibration interface."""
    
    print("=" * 60)
    print("INTERACTIVE CALIBRATION DEMO")
    print("=" * 60)
    print("\nThis demo shows an interactive calibration interface.")
    print("Use the sliders to adjust gain, offset, and noise reduction.")
    print("Click 'Reset' to restore default values.")
    print("\nClose the window to continue to the next demo.")
    
    # Create detector with noise
    detector = PixelDetector(width=150, height=150, gain=1.5, offset=20.0)
    detector.set_noise_model(GaussianNoise(std_dev=0.15))
    
    # Generate signal
    raw_signal = create_demo_signal()
    detected_signal = detector.detect(raw_signal)
    
    # Create interactive interface
    plotter = InteractivePlotter()
    
    def calibration_func(signal, gain, offset, noise_level):
        """Apply calibration with given parameters."""
        calibrated = (signal - offset) / gain
        
        if noise_level > 0:
            from scipy import ndimage
            calibrated = ndimage.gaussian_filter(calibrated, sigma=noise_level)
        
        return calibrated
    
    fig = plotter.create_detector_calibration_interface(
        detected_signal, calibration_func
    )
    
    plt.show()


def demo_noise_analysis():
    """Demonstrate interactive noise analysis interface."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE NOISE ANALYSIS DEMO")
    print("=" * 60)
    print("\nThis demo shows an interactive noise analysis interface.")
    print("Select different filter methods and adjust filter strength.")
    print("Click 'Apply Filter' to permanently apply the selected filter.")
    print("\nClose the window to continue to the next demo.")
    
    # Create noisy signal
    detector = PixelDetector(width=150, height=150)
    detector.set_noise_model(GaussianNoise(std_dev=0.2))
    
    raw_signal = create_demo_signal()
    noisy_signal = detector.detect(raw_signal)
    
    # Create interactive noise analysis interface
    plotter = InteractivePlotter()
    fig = plotter.create_noise_analysis_interface(noisy_signal)
    
    plt.show()


def demo_roi_selection():
    """Demonstrate ROI (Region of Interest) selection."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE ROI SELECTION DEMO")
    print("=" * 60)
    print("\nThis demo shows an interactive ROI selection interface.")
    print("Click and drag on the image to select regions of interest.")
    print("The ROI statistics will be printed in the console.")
    print("\nClose the window to finish the demo.")
    
    # Create signal with interesting features
    detector = PixelDetector(width=150, height=150)
    detector.set_noise_model(GaussianNoise(std_dev=0.1))
    
    raw_signal = create_demo_signal()
    detected_signal = detector.detect(raw_signal)
    
    # Create interactive ROI selector
    plotter = InteractivePlotter()
    
    def roi_callback(roi_data):
        """Callback function when ROI is selected."""
        stats = roi_data['statistics']
        coords = roi_data['coordinates']
        
        print(f"\nROI Selected:")
        print(f"  Coordinates: ({coords[0]}, {coords[1]}) to ({coords[2]}, {coords[3]})")
        print(f"  Statistics:")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std:  {stats['std']:.3f}")
        print(f"    Min:  {stats['min']:.3f}")
        print(f"    Max:  {stats['max']:.3f}")
        print(f"    Sum:  {stats['sum']:.3f}")
    
    fig = plotter.create_roi_selector(detected_signal, roi_callback)
    plt.show()


def main():
    """Run all interactive demos."""
    
    print("DETECTOR CALIBRATION SIMULATION - INTERACTIVE DEMOS")
    print("=" * 60)
    print("\nThis demo showcases the interactive visualization features.")
    print("Each demo will open a new window - close it to continue.")
    
    try:
        # Demo 1: Interactive calibration
        demo_interactive_calibration()
        
        # Demo 2: Noise analysis
        demo_noise_analysis()
        
        # Demo 3: ROI selection
        demo_roi_selection()
        
        print("\n" + "=" * 60)
        print("ALL INTERACTIVE DEMOS COMPLETED!")
        print("=" * 60)
        print("\nThank you for trying the interactive features!")
        print("You can now explore the other examples or create your own simulations.")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        print("Make sure all required dependencies are installed:")
        print("pip install matplotlib scipy numpy")


if __name__ == "__main__":
    main()
