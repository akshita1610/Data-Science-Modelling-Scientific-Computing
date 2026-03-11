#!/usr/bin/env python3
"""
Basic Usage Example for Detector Calibration Simulation

This example demonstrates the fundamental usage of the detector simulation framework:
- Setting up a detector with noise
- Creating signal sources
- Running the simulation pipeline
- Visualizing results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the detector_sim package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detector_sim.models.detector import PixelDetector
from detector_sim.models.noise_models import GaussianNoise
from detector_sim.simulation.signal_sources import PointSource, GaussianSource, UniformSource
from detector_sim.simulation.signal_generator import SignalGenerator
from detector_sim.calibration.calibration import CalibrationPipeline, GainOffsetCalibration
from detector_sim.calibration.noise_reduction import GaussianFilter
from detector_sim.visualization.plots import DetectorPlotter, CalibrationPlotter


def main():
    """Run basic detector simulation example."""
    
    print("=" * 60)
    print("DETECTOR CALIBRATION SIMULATION - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # 1. Setup Detector
    print("\n1. Setting up detector...")
    detector = PixelDetector(
        width=100,           # 100x100 pixel detector
        height=100,
        gain=1.2,           # 20% gain
        offset=10.0,        # 10 unit offset
        sensitivity=1.0
    )
    
    # Add Gaussian noise
    noise_model = GaussianNoise(mean=0.0, std_dev=0.1)
    detector.set_noise_model(noise_model)
    print("   - Pixel detector: 100x100")
    print("   - Gain: 1.2, Offset: 10.0")
    print("   - Noise: Gaussian (sigma=0.1)")
    
    # 2. Setup Signal Sources
    print("\n2. Setting up signal sources...")
    signal_gen = SignalGenerator(width=100, height=100)
    
    # Add multiple signal sources
    signal_gen.add_source(PointSource(x=50, y=50, intensity=5.0))
    signal_gen.add_source(GaussianSource(
        center_x=30, center_y=30,
        sigma_x=8, sigma_y=8,
        intensity=2.0
    ))
    signal_gen.add_source(UniformSource(intensity=0.1))
    
    print("   - Point source at (50, 50) with intensity 5.0")
    print("   - Gaussian source at (30, 30) with intensity 2.0")
    print("   - Uniform background with intensity 0.1")
    
    # 3. Generate and Detect Signal
    print("\n3. Generating and detecting signal...")
    raw_signal = signal_gen.generate_signal()
    detected_signal = detector.detect(raw_signal)
    
    print(f"   - Raw signal: mean={np.mean(raw_signal):.3f}, std={np.std(raw_signal):.3f}")
    print(f"   - Detected signal: mean={np.mean(detected_signal):.3f}, std={np.std(detected_signal):.3f}")
    
    # 4. Apply Calibration
    print("\n4. Applying calibration...")
    calibration = CalibrationPipeline()
    
    # Add gain/offset correction
    calibration.add_method(GainOffsetCalibration(
        reference_gain=detector.gain,
        reference_offset=detector.offset
    ))
    
    # Add noise reduction
    from detector_sim.calibration.calibration import CalibrationMethod
    class NoiseReductionCalibration(CalibrationMethod):
        def __init__(self, noise_reducer):
            self.noise_reducer = noise_reducer
        
        def calibrate(self, signal):
            return self.noise_reducer.reduce_noise(signal)
    
    calibration.add_method(NoiseReductionCalibration(GaussianFilter(sigma=1.0)))
    
    calibrated_signal = calibration.calibrate(detected_signal)
    
    print(f"   - Calibrated signal: mean={np.mean(calibrated_signal):.3f}, std={np.std(calibrated_signal):.3f}")
    
    # 5. Visualize Results
    print("\n5. Generating visualizations...")
    plotter = DetectorPlotter()
    
    # Signal comparison plot
    fig1 = plotter.plot_signal_comparison(
        [raw_signal, detected_signal, calibrated_signal],
        ['Raw Signal', 'Detected Signal', 'Calibrated Signal'],
        'Signal Processing Pipeline'
    )
    
    # Detector response plot
    fig2 = plotter.plot_detector_response(
        calibrated_signal,
        'Final Calibrated Detector Response'
    )
    
    # Calibration comparison plot
    calib_plotter = CalibrationPlotter()
    fig3 = calib_plotter.plot_calibration_comparison(
        detected_signal, calibrated_signal,
        'Calibration Effect Comparison'
    )
    
    # 6. Calculate Metrics
    print("\n6. Calculating performance metrics...")
    from detector_sim.evaluation.metrics import EvaluationMetrics
    
    # Compare detected vs calibrated
    metrics = EvaluationMetrics.compute_all_metrics(detected_signal, calibrated_signal)
    
    print("   Calibration Performance:")
    print(f"   - PSNR: {metrics.get('PSNR', 'N/A'):.2f} dB")
    print(f"   - SSIM: {metrics.get('SSIM', 'N/A'):.3f}")
    print(f"   - RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
    print(f"   - Correlation: {metrics.get('Correlation', 'N/A'):.3f}")
    
    # 7. Save Results
    print("\n7. Saving results...")
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plots
    fig1.savefig(os.path.join(output_dir, 'signal_comparison.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'detector_response.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'calibration_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Save data
    from detector_sim.data.data_manager import DataManager
    data_manager = DataManager(output_dir)
    
    data_manager.save_dataset(raw_signal, 'raw_signal', {'type': 'raw', 'example': 'basic'})
    data_manager.save_dataset(detected_signal, 'detected_signal', {'type': 'detected', 'example': 'basic'})
    data_manager.save_dataset(calibrated_signal, 'calibrated_signal', {'type': 'calibrated', 'example': 'basic'})
    
    print(f"   - Results saved to: {output_dir}/")
    print("   - Plots: PNG format, 300 DPI")
    print("   - Data: NPZ format with metadata")
    
    # Show plots
    print("\n8. Displaying plots...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Check the '{output_dir}' directory for saved results.")


if __name__ == "__main__":
    main()
