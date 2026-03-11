#!/usr/bin/env python3
"""
Detector Calibration Simulation - Main Entry Point
A comprehensive simulation framework for detector calibration and analysis.

This program simulates how radiation or optical detectors respond to different
input signals, noise sources, and calibration parameters. It provides tools for
signal generation, detector modeling, calibration, and performance evaluation.

Usage:
    python main.py [--config CONFIG_FILE] [--output OUTPUT_DIR] [--verbose]

Examples:
    python main.py                                    # Use default configuration
    python main.py --config custom_config.yaml        # Use custom configuration
    python main.py --output results/ --verbose        # Custom output and verbose logging
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add the detector_sim package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_sim.models.detector import PixelDetector
from detector_sim.models.noise_models import GaussianNoise, PoissonNoise, ReadoutNoise, CombinedNoise
from detector_sim.simulation.signal_sources import PointSource, GaussianSource, UniformSource
from detector_sim.simulation.signal_generator import SignalGenerator
from detector_sim.calibration.calibration import CalibrationPipeline, GainOffsetCalibration
from detector_sim.calibration.noise_reduction import GaussianFilter, MedianFilter
from detector_sim.visualization.plots import DetectorPlotter, CalibrationPlotter
from detector_sim.data.data_manager import DataManager, DatasetGenerator
from detector_sim.evaluation.metrics import EvaluationMetrics
from detector_sim.evaluation.comparison import CalibrationComparator


class DetectorSimulation:
    """Main simulation class that orchestrates the entire pipeline."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the simulation with configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self._setup_logging()
        self._setup_directories()
        
        # Initialize components
        self.detector = None
        self.signal_generator = None
        self.calibration_pipeline = None
        self.plotter = DetectorPlotter()
        self.data_manager = DataManager(self.config['data']['output_directory'])
        
        self.logger.info("Detector Calibration Simulation initialized")
        self.logger.info(f"Configuration loaded from: {config_file}")
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _get_default_config(self) -> dict:
        """Get default configuration if config file is not found."""
        return {
            'detector': {
                'type': 'pixel',
                'width': 100,
                'height': 100,
                'gain': 1.0,
                'offset': 0.0,
                'sensitivity': 1.0,
                'dark_current': 0.1
            },
            'noise': {
                'enabled': True,
                'type': 'gaussian',
                'parameters': {
                    'mean': 0.0,
                    'std_dev': 0.1
                }
            },
            'signal_sources': {
                'enabled': ['point_source'],
                'point_source': {
                    'x': 50, 'y': 50, 'intensity': 1.0
                }
            },
            'calibration': {
                'enabled': True,
                'methods': ['gain_offset']
            },
            'visualization': {
                'enabled': True,
                'save_plots': True,
                'plot_format': 'png'
            },
            'data': {
                'output_directory': 'output',
                'save_format': 'npz',
                'auto_save': True
            },
            'evaluation': {
                'enabled': True,
                'metrics': ['mse', 'rmse', 'psnr', 'ssim']
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logger
        self.logger = logging.getLogger('detector_simulation')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file'):
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_directories(self):
        """Create necessary directories."""
        output_dir = self.config['data']['output_directory']
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config['visualization']['enabled']:
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
    
    def setup_detector(self):
        """Setup the detector based on configuration."""
        det_config = self.config['detector']
        
        if det_config['type'] == 'pixel':
            self.detector = PixelDetector(
                width=det_config['width'],
                height=det_config['height'],
                gain=det_config['gain'],
                offset=det_config['offset'],
                sensitivity=det_config['sensitivity']
            )
            
            # Set dark current
            self.detector.set_dark_current(det_config['dark_current'])
        
        # Setup noise model
        if self.config['noise']['enabled']:
            noise_config = self.config['noise']
            noise_params = noise_config['parameters']
            
            if noise_config['type'] == 'gaussian':
                noise_model = GaussianNoise(
                    mean=noise_params['mean'],
                    std_dev=noise_params['std_dev']
                )
            elif noise_config['type'] == 'poisson':
                noise_model = PoissonNoise()
            elif noise_config['type'] == 'readout':
                noise_model = ReadoutNoise(
                    readout_std=noise_params.get('readout_std', 1.0),
                    flicker_strength=noise_params.get('flicker_strength', 0.1)
                )
            else:
                noise_model = GaussianNoise(mean=0.0, std_dev=0.1)
            
            self.detector.set_noise_model(noise_model)
        
        self.logger.info(f"Detector setup complete: {det_config['type']} detector")
    
    def setup_signal_sources(self):
        """Setup signal sources based on configuration."""
        signal_config = self.config['signal_sources']
        det_config = self.config['detector']
        
        self.signal_generator = SignalGenerator(
            width=det_config['width'],
            height=det_config['height']
        )
        
        # Add enabled signal sources
        for source_name in signal_config['enabled']:
            source_params = signal_config.get(source_name, {})
            
            if source_name == 'point_source':
                source = PointSource(
                    x=source_params.get('x', 50),
                    y=source_params.get('y', 50),
                    intensity=source_params.get('intensity', 1.0)
                )
            elif source_name == 'gaussian_source':
                source = GaussianSource(
                    center_x=source_params.get('center_x', 50),
                    center_y=source_params.get('center_y', 50),
                    sigma_x=source_params.get('sigma_x', 10.0),
                    sigma_y=source_params.get('sigma_y', 10.0),
                    intensity=source_params.get('intensity', 1.0),
                    rotation=source_params.get('rotation', 0.0)
                )
            elif source_name == 'uniform_source':
                source = UniformSource(
                    intensity=source_params.get('intensity', 0.5)
                )
            else:
                continue
            
            self.signal_generator.add_source(source)
            self.logger.info(f"Added signal source: {source_name}")
    
    def setup_calibration(self):
        """Setup calibration pipeline based on configuration."""
        if not self.config['calibration']['enabled']:
            return
        
        self.calibration_pipeline = CalibrationPipeline()
        calib_config = self.config['calibration']
        
        for method_name in calib_config['methods']:
            if method_name == 'gain_offset':
                method = GainOffsetCalibration(
                    reference_gain=calib_config.get('gain_offset', {}).get('reference_gain', 1.0),
                    reference_offset=calib_config.get('gain_offset', {}).get('reference_offset', 0.0)
                )
            elif method_name == 'noise_reduction':
                noise_config = calib_config.get('noise_reduction', {})
                if noise_config.get('method', 'gaussian') == 'gaussian':
                    # Wrap noise reduction in calibration method
                    from detector_sim.calibration.calibration import CalibrationMethod
                    class NoiseReductionCalibration(CalibrationMethod):
                        def __init__(self, noise_reducer):
                            self.noise_reducer = noise_reducer
                        
                        def calibrate(self, signal):
                            return self.noise_reducer.reduce_noise(signal)
                    
                    noise_reducer = GaussianFilter(
                        sigma=noise_config.get('parameters', {}).get('sigma', 1.0)
                    )
                    method = NoiseReductionCalibration(noise_reducer)
                else:
                    noise_reducer = MedianFilter(
                        kernel_size=noise_config.get('parameters', {}).get('kernel_size', 3)
                    )
                    from detector_sim.calibration.calibration import CalibrationMethod
                    class NoiseReductionCalibration(CalibrationMethod):
                        def __init__(self, noise_reducer):
                            self.noise_reducer = noise_reducer
                        
                        def calibrate(self, signal):
                            return self.noise_reducer.reduce_noise(signal)
                    
                    method = NoiseReductionCalibration(noise_reducer)
            else:
                continue
            
            self.calibration_pipeline.add_method(method)
            self.logger.info(f"Added calibration method: {method_name}")
    
    def run_simulation(self):
        """Run the complete simulation pipeline."""
        self.logger.info("Starting simulation...")
        
        # Setup components
        self.setup_detector()
        self.setup_signal_sources()
        self.setup_calibration()
        
        # Generate signal
        self.logger.info("Generating signal...")
        raw_signal = self.signal_generator.generate_signal()
        
        # Detect signal
        self.logger.info("Detecting signal with detector...")
        detected_signal = self.detector.detect(raw_signal)
        
        # Apply calibration
        calibrated_signal = detected_signal
        if self.calibration_pipeline:
            self.logger.info("Applying calibration...")
            calibrated_signal = self.calibration_pipeline.calibrate(detected_signal)
        
        # Save data
        if self.config['data']['auto_save']:
            self.logger.info("Saving data...")
            output_dir = self.config['data']['output_directory']
            
            # Save raw signal
            self.data_manager.save_dataset(
                raw_signal, 'raw_signal',
                metadata={'type': 'raw_signal', 'simulation': 'detector_calibration'}
            )
            
            # Save detected signal
            self.data_manager.save_dataset(
                detected_signal, 'detected_signal',
                metadata={'type': 'detected_signal', 'simulation': 'detector_calibration'}
            )
            
            # Save calibrated signal
            self.data_manager.save_dataset(
                calibrated_signal, 'calibrated_signal',
                metadata={'type': 'calibrated_signal', 'simulation': 'detector_calibration'}
            )
        
        # Generate visualizations
        if self.config['visualization']['enabled']:
            self.logger.info("Generating visualizations...")
            self._generate_plots(raw_signal, detected_signal, calibrated_signal)
        
        # Perform evaluation
        if self.config['evaluation']['enabled']:
            self.logger.info("Performing evaluation...")
            self._perform_evaluation(raw_signal, detected_signal, calibrated_signal)
        
        self.logger.info("Simulation completed successfully!")
        return raw_signal, detected_signal, calibrated_signal
    
    def _generate_plots(self, raw_signal, detected_signal, calibrated_signal):
        """Generate and save visualization plots."""
        plots_dir = os.path.join(self.config['data']['output_directory'], 'plots')
        plot_format = self.config['visualization']['plot_format']
        
        # Signal comparison plot
        fig = self.plotter.plot_signal_comparison(
            [raw_signal, detected_signal, calibrated_signal],
            ['Raw Signal', 'Detected Signal', 'Calibrated Signal'],
            'Signal Processing Pipeline'
        )
        fig.savefig(os.path.join(plots_dir, f'signal_comparison.{plot_format}'))
        plt.close(fig)
        
        # Detector response plot
        fig = self.plotter.plot_detector_response(
            calibrated_signal,
            'Final Calibrated Detector Response'
        )
        fig.savefig(os.path.join(plots_dir, f'detector_response.{plot_format}'))
        plt.close(fig)
        
        # Noise analysis plot
        fig = self.plotter.plot_noise_analysis(detected_signal)
        fig.savefig(os.path.join(plots_dir, f'noise_analysis.{plot_format}'))
        plt.close(fig)
        
        # Calibration comparison plot
        if self.calibration_pipeline:
            calib_plotter = CalibrationPlotter()
            fig = calib_plotter.plot_calibration_comparison(
                detected_signal, calibrated_signal,
                'Calibration Effect Comparison'
            )
            fig.savefig(os.path.join(plots_dir, f'calibration_comparison.{plot_format}'))
            plt.close(fig)
    
    def _perform_evaluation(self, raw_signal, detected_signal, calibrated_signal):
        """Perform evaluation and generate metrics."""
        metrics = {}
        
        # Compare raw vs detected
        raw_vs_detected = EvaluationMetrics.compute_all_metrics(raw_signal, detected_signal)
        metrics['raw_vs_detected'] = raw_vs_detected
        
        # Compare detected vs calibrated
        detected_vs_calibrated = EvaluationMetrics.compute_all_metrics(
            detected_signal, calibrated_signal
        )
        metrics['detected_vs_calibrated'] = detected_vs_calibrated
        
        # Save evaluation results
        eval_data = {
            'raw_vs_detected': raw_vs_detected,
            'detected_vs_calibrated': detected_vs_calibrated
        }
        
        # Save as text file
        output_dir = self.config['data']['output_directory']
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Raw vs Detected Signal:\n")
            for metric, value in raw_vs_detected.items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nDetected vs Calibrated Signal:\n")
            for metric, value in detected_vs_calibrated.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        self.logger.info("Evaluation results saved")


def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(
        description="Detector Calibration Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default configuration
  python main.py --config custom_config.yaml        # Use custom configuration
  python main.py --output results/ --verbose        # Custom output and verbose logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory (overrides config file)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Detector Calibration Simulation 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Override config if output directory specified
    if args.output:
        # We'll handle this in the simulation class
        pass
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run simulation
        sim = DetectorSimulation(args.config)
        
        # Override output directory if specified
        if args.output:
            sim.config['data']['output_directory'] = args.output
            sim._setup_directories()
            sim.data_manager = DataManager(args.output)
        
        # Run the simulation
        raw_signal, detected_signal, calibrated_signal = sim.run_simulation()
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Output directory: {sim.config['data']['output_directory']}")
        print(f"Raw signal shape: {raw_signal.shape}")
        print(f"Detected signal shape: {detected_signal.shape}")
        print(f"Calibrated signal shape: {calibrated_signal.shape}")
        
        # Print some basic statistics
        print(f"\nSignal Statistics:")
        print(f"  Raw signal - Mean: {np.mean(raw_signal):.4f}, Std: {np.std(raw_signal):.4f}")
        print(f"  Detected signal - Mean: {np.mean(detected_signal):.4f}, Std: {np.std(detected_signal):.4f}")
        print(f"  Calibrated signal - Mean: {np.mean(calibrated_signal):.4f}, Std: {np.std(calibrated_signal):.4f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
