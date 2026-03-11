# 🚀 Quick Start Guide

## Installation & Setup

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml

# Test installation
python -c "import detector_sim; print('✅ Installation successful')"
```

## Running Simulations

### 1. Basic Simulation (Default Configuration)
```bash
python main.py
```
**Output**: Creates `output/` directory with data, plots, and evaluation results.

### 2. Custom Output Directory
```bash
python main.py --output my_results/
```

### 3. Verbose Logging
```bash
python main.py --verbose
```

### 4. Custom Configuration
```bash
python main.py --config my_config.yaml
```

## Running Examples

### Basic Usage Example
```bash
python examples/basic_usage.py
```
**Demonstrates**: Complete simulation pipeline with visualizations and metrics.

### Interactive Demo
```bash
python examples/interactive_demo.py
```
**Demonstrates**: Real-time parameter adjustment, ROI selection, noise analysis.

## Configuration

Edit `config.yaml` to customize:

```yaml
detector:
  width: 100
  height: 100
  gain: 1.0
  offset: 0.0

noise:
  enabled: true
  type: "gaussian"
  parameters:
    std_dev: 0.1

signal_sources:
  enabled: ["point_source", "gaussian_source"]
```

## Output Files

Each simulation generates:

- **Data Files** (NPZ format):
  - `raw_signal_*.npz` - Original generated signal
  - `detected_signal_*.npz` - After detector processing
  - `calibrated_signal_*.npz` - Final calibrated signal

- **Visualizations** (PNG format):
  - `signal_comparison.png` - Processing pipeline comparison
  - `detector_response.png` - Final detector response
  - `noise_analysis.png` - Noise characteristics
  - `calibration_comparison.png` - Calibration effects

- **Evaluation** (Text format):
  - `evaluation_results.txt` - Performance metrics (PSNR, SSIM, RMSE, etc.)

## Python API Usage

```python
from detector_sim.models.detector import PixelDetector
from detector_sim.models.noise_models import GaussianNoise
from detector_sim.simulation.signal_sources import PointSource
from detector_sim.simulation.signal_generator import SignalGenerator
from detector_sim.calibration.calibration import CalibrationPipeline, GainOffsetCalibration

# Create detector
detector = PixelDetector(width=100, height=100, gain=1.2, offset=10.0)
detector.set_noise_model(GaussianNoise(std_dev=0.1))

# Create signal
signal_gen = SignalGenerator(width=100, height=100)
signal_gen.add_source(PointSource(x=50, y=50, intensity=5.0))
raw_signal = signal_gen.generate_signal()

# Detect and calibrate
detected_signal = detector.detect(raw_signal)
calibration = CalibrationPipeline()
calibration.add_method(GainOffsetCalibration(reference_gain=1.0, reference_offset=0.0))
calibrated_signal = calibration.calibrate(detected_signal)

print(f"Raw: mean={np.mean(raw_signal):.3f}")
print(f"Detected: mean={np.mean(detected_signal):.3f}")
print(f"Calibrated: mean={np.mean(calibrated_signal):.3f}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **NumPy Deprecation Warnings**: The code handles these automatically

3. **Display Issues**: For interactive demos, ensure matplotlib backend is configured

4. **Configuration Not Found**: Creates default config automatically

### Performance Tips

- Use smaller detector sizes for faster testing
- Disable visualization for batch processing
- Use `--output` to organize multiple runs

## Next Steps

1. **Explore Configuration**: Modify `config.yaml` for different scenarios
2. **Try Examples**: Run all example scripts to see features
3. **Custom Signals**: Create your own signal sources
4. **Advanced Calibration**: Implement custom calibration methods
5. **Batch Processing**: Use data management tools for large datasets

## Support

- Check `README.md` for detailed documentation
- Review examples for implementation patterns
- Examine configuration options in `config.yaml`

Happy Simulating! 🎯
