# ⭐ Detector Calibration Simulation

A comprehensive Python framework for simulating and calibrating radiation and optical detectors. This project provides a modular pipeline inspired by real scientific simulation tools like RadCalSim, CMOS Detector Simulation Tool, and Gamma Source Fitter.

## 🎯 Project Overview

This simulation framework models how detectors respond to different input signals, noise sources, and calibration parameters. It's designed as a second-year Computer Science-level project but structured like real research tools used in scientific and engineering environments.

## 🚀 Features

### **Core Components**

- **🔬 Detector Models**
  - Pixel-based detector with configurable grid size
  - Single-channel detector for point measurements
  - Adjustable gain, offset, sensitivity, and dark current
  - Multiple noise models (Gaussian, Poisson, Readout noise)

- **📡 Signal Simulation**
  - Point sources with configurable positions and intensities
  - Gaussian sources with adjustable parameters
  - Uniform illumination sources
  - Gamma ray interaction simulation
  - Custom pattern sources (grids, checkerboards, circles)

- **⚙️ Calibration Algorithms**
  - Gain and offset correction
  - Non-linear polynomial calibration
  - Per-pixel response calibration
  - Temperature-dependent calibration
  - Multiple noise reduction methods (Gaussian, Median, Wavelet)

- **📊 Visualization Tools**
  - 2D detector response plots with signal profiles
  - Noise analysis and distribution plots
  - Calibration comparison visualizations
  - Interactive plotting with real-time parameter adjustment
  - ROI (Region of Interest) selection tools

- **💾 Data Management**
  - Multiple file format support (NPZ, NPY, CSV, HDF5, Pickle)
  - Automatic metadata storage
  - Dataset generation for testing
  - Experiment logging and tracking

- **📈 Evaluation Metrics**
  - Comprehensive error metrics (MSE, RMSE, MAE, PSNR)
  - Signal-to-noise ratio calculations
  - Structural similarity index (SSIM)
  - Linearity and uniformity analysis
  - Performance comparison tools

## 🏗️ Project Structure

```
detector_sim/
├── models/                  # Detector and noise models
│   ├── __init__.py
│   ├── detector.py         # Main detector classes
│   └── noise_models.py    # Various noise models
├── simulation/              # Signal generation
│   ├── __init__.py
│   ├── signal_sources.py   # Different signal sources
│   └── signal_generator.py # Main signal generator
├── calibration/            # Calibration algorithms
│   ├── __init__.py
│   ├── calibration.py      # Main calibration pipeline
│   ├── noise_reduction.py  # Noise reduction methods
│   └── curve_fitting.py    # Curve fitting algorithms
├── visualization/           # Plotting and visualization
│   ├── __init__.py
│   ├── plots.py           # Static plotting tools
│   └── interactive.py     # Interactive plotting
├── data/                   # Data management
│   ├── __init__.py
│   ├── data_manager.py    # Main data management
│   └── file_handlers.py   # File format handlers
└── evaluation/             # Metrics and analysis
    ├── __init__.py
    ├── metrics.py         # Evaluation metrics
    ├── comparison.py      # Comparison tools
    └── analysis.py        # Statistical analysis
```

## 📋 Requirements

### **Core Dependencies**
- Python 3.8+
- NumPy (>=1.21.0)
- SciPy (>=1.7.0)
- Matplotlib (>=3.5.0)
- Pandas (>=1.3.0)
- PyYAML (>=6.0)

### **Optional Dependencies**
- Scikit-image (advanced image processing)
- PyWavelets (wavelet denoising)
- H5py (HDF5 support)
- Plotly (interactive plotting)
- Jupyter (notebook support)

See `requirements.txt` for complete list with versions.

## 🛠️ Installation

### **Option 1: Clone and Install**
```bash
git clone <repository-url>
cd detector-calibration-simulation
pip install -r requirements.txt
```

### **Option 2: Install Dependencies Only**
```bash
pip install numpy scipy matplotlib pandas pyyaml
```

### **Option 3: Development Installation**
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## 🎮 Quick Start

### **Basic Usage**

1. **Run with Default Configuration**
```bash
python main.py
```

2. **Custom Configuration**
```bash
python main.py --config custom_config.yaml
```

3. **Specify Output Directory**
```bash
python main.py --output results/ --verbose
```

### **Simple Python Example**

```python
import numpy as np
from detector_sim.models.detector import PixelDetector
from detector_sim.models.noise_models import GaussianNoise
from detector_sim.simulation.signal_sources import PointSource, GaussianSource
from detector_sim.simulation.signal_generator import SignalGenerator
from detector_sim.calibration.calibration import CalibrationPipeline, GainOffsetCalibration
from detector_sim.visualization.plots import DetectorPlotter

# Create detector
detector = PixelDetector(width=100, height=100, gain=1.2, offset=10)
detector.set_noise_model(GaussianNoise(std_dev=0.1))

# Create signal sources
signal_gen = SignalGenerator(width=100, height=100)
signal_gen.add_source(PointSource(x=50, y=50, intensity=1.0))
signal_gen.add_source(GaussianSource(center_x=30, center_y=30, sigma_x=5, sigma_y=5))

# Generate and detect signal
raw_signal = signal_gen.generate_signal()
detected_signal = detector.detect(raw_signal)

# Apply calibration
calibration = CalibrationPipeline()
calibration.add_method(GainOffsetCalibration(reference_gain=1.0, reference_offset=0.0))
calibrated_signal = calibration.calibrate(detected_signal)

# Visualize results
plotter = DetectorPlotter()
fig = plotter.plot_signal_comparison(
    [raw_signal, detected_signal, calibrated_signal],
    ['Raw', 'Detected', 'Calibrated']
)
fig.show()
```

## ⚙️ Configuration

The simulation uses a YAML configuration file (`config.yaml`) to customize:

- **Detector parameters** (size, gain, offset, sensitivity)
- **Noise models** and their parameters
- **Signal sources** to simulate
- **Calibration methods** to apply
- **Visualization settings**
- **Data output formats**
- **Evaluation metrics**

### **Example Configuration Snippet**

```yaml
detector:
  type: "pixel"
  width: 100
  height: 100
  gain: 1.0
  offset: 0.0
  sensitivity: 1.0

noise:
  enabled: true
  type: "gaussian"
  parameters:
    mean: 0.0
    std_dev: 0.1

signal_sources:
  enabled: ["point_source", "gaussian_source"]
  point_source:
    x: 50
    y: 50
    intensity: 1.0
```

## 📊 Output and Results

The simulation generates:

1. **Data Files** (in NPZ format by default)
   - `raw_signal.npz`: Original generated signal
   - `detected_signal.npz`: Signal after detector processing
   - `calibrated_signal.npz`: Final calibrated signal

2. **Visualization Plots**
   - Signal comparison plots
   - Detector response visualizations
   - Noise analysis plots
   - Calibration comparisons

3. **Evaluation Results**
   - Comprehensive metrics (MSE, RMSE, PSNR, SSIM)
   - Performance comparisons
   - Quality assessments

## 🔬 Advanced Usage

### **Custom Noise Models**

```python
from detector_sim.models.noise_models import CombinedNoise

# Combine multiple noise sources
noise_model = CombinedNoise([
    GaussianNoise(std_dev=0.1),
    PoissonNoise(scale_factor=1.0)
])
detector.set_noise_model(noise_model)
```

### **Interactive Visualization**

```python
from detector_sim.visualization.interactive import InteractivePlotter

# Create interactive calibration interface
plotter = InteractivePlotter()
fig = plotter.create_detector_calibration_interface(
    detected_signal, 
    lambda signal, gain, offset: (signal - offset) / gain
)
fig.show()
```

### **Batch Processing**

```python
from detector_sim.data.data_manager import DatasetGenerator

# Generate calibration datasets
data_manager = DataManager("calibration_data")
generator = DatasetGenerator(data_manager)

paths = generator.generate_calibration_dataset(
    detector_width=200,
    detector_height=200,
    num_flat_fields=10,
    num_dark_frames=10
)
```

### **Performance Evaluation**

```python
from detector_sim.evaluation.comparison import CalibrationComparator

# Compare different calibration methods
comparator = CalibrationComparator()
comparator.add_calibration_result("method1", raw, calib1, reference)
comparator.add_calibration_result("method2", raw, calib2, reference)

results = comparator.compare_calibration_methods()
print(results)
```

## 🎓 Educational Value

This project demonstrates:

- **Scientific Computing**: Real-world simulation techniques
- **Signal Processing**: Noise modeling and filtering
- **Data Analysis**: Statistical analysis and metrics
- **Software Engineering**: Modular design and architecture
- **Visualization**: Scientific plotting and data presentation
- **Configuration Management**: YAML-based parameter systems

Perfect for graduate school applications in:
- Data Science
- Scientific Computing
- AI/Machine Learning
- Engineering Simulation
- Medical Imaging
- Physics Research

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Inspired by and concepts adapted from:
- RadCalSim (Radiation Detector Modeling)
- CMOS Detector Simulation Tool
- Gamma Source Fitter
- Scientific computing best practices

## 📞 Support

For questions, issues, or suggestions:

1. Check the documentation
2. Review example configurations
3. Open an issue on GitHub
4. Review the code comments for detailed explanations

---

**Happy Simulating! 🚀**
