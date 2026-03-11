"""
Detector Calibration Simulation Package
A modular Python framework for simulating and calibrating radiation/optical detectors.
"""

__version__ = "1.0.0"
__author__ = "Detector Simulation Team"

# Import main classes for easy access
from .models.detector import PixelDetector, SingleChannelDetector
from .models.noise_models import GaussianNoise, PoissonNoise, ReadoutNoise
from .simulation.signal_sources import PointSource, UniformSource, GaussianSource
from .simulation.signal_generator import SignalGenerator
from .calibration.calibration import CalibrationPipeline, GainOffsetCalibration
from .calibration.noise_reduction import GaussianFilter, MedianFilter
from .visualization.plots import DetectorPlotter, CalibrationPlotter

__all__ = [
    # Models
    'PixelDetector', 'SingleChannelDetector',
    'GaussianNoise', 'PoissonNoise', 'ReadoutNoise',
    
    # Simulation
    'PointSource', 'UniformSource', 'GaussianSource',
    'SignalGenerator',
    
    # Calibration
    'CalibrationPipeline', 'GainOffsetCalibration',
    'GaussianFilter', 'MedianFilter',
    
    # Visualization
    'DetectorPlotter', 'CalibrationPlotter'
]
