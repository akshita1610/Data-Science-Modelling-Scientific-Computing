# Event Reconstruction & Sequence Analysis Pipeline

A comprehensive Python pipeline for reconstructing event timelines and analyzing sequential data, inspired by scientific computing workflows like scdna-pipe, Sequana, and rpg_e2depth.

## 🎯 Overview

This modular pipeline processes time-ordered event data, extracts meaningful patterns, reconstructs missing events, and provides comprehensive analysis and visualization capabilities. It's designed as a second-year Computer Science level project but follows real-world scientific computing practices.

## 🚀 Features

### Core Capabilities
- **Multi-format Data Ingestion**: Support for CSV, JSON, and simulated event streams
- **Advanced Preprocessing**: Data cleaning, normalization, segmentation, and noise filtering
- **Rich Feature Extraction**: Time intervals, frequency patterns, transition probabilities, and sliding-window features
- **Event Reconstruction**: Rule-based and probabilistic methods for timeline reconstruction
- **Sequence Analysis**: Pattern detection, anomaly detection, and correlation analysis
- **Comprehensive Visualization**: Timeline plots, pattern networks, reconstruction comparisons
- **Thorough Evaluation**: Accuracy metrics, quality assessment, and pipeline benchmarking

### Pipeline Architecture
```
event_pipeline/
├── ingestion/          # Data loading and stream simulation
├── preprocessing/      # Data cleaning and normalization
├── features/          # Feature extraction and engineering
├── reconstruction/    # Event reconstruction algorithms
├── analysis/          # Pattern detection and analysis
├── visualization/      # Plotting and visual analysis
├── evaluation/        # Metrics and benchmarking
├── config.yaml        # Configuration file
└── main.py           # Main pipeline orchestrator
```

## 📋 Requirements

### Dependencies
```bash
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
scipy>=1.9.0
pyyaml>=6.0
tqdm>=4.64.0
```

### Python Version
- Python 3.8 or higher

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Event Reconstruction & Sequence Analysis"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python main.py --demo
```

## 🎮 Quick Start

### Run Demo with Sample Data
```bash
python main.py --demo
```

### Process Your Own Data
```bash
python main.py --input your_data.csv
```

### Generate Sample Data
```bash
python main.py --generate-sample 1000
```

### Advanced Usage
```bash
python main.py --input data.csv --config custom_config.yaml --no-viz
```

## 📊 Data Format

### Input Data Requirements
Your data should contain at least these columns:
- `timestamp`: Event timestamps (ISO format preferred)
- `event`: Event type/name

### Example CSV Format
```csv
timestamp,event,user_id,session_id
2023-01-01 10:00:00,start,1,100
2023-01-01 10:01:30,process,1,100
2023-01-01 10:05:45,complete,1,100
2023-01-01 10:06:00,start,2,101
```

### Example JSON Format
```json
{
  "events": [
    {
      "timestamp": "2023-01-01T10:00:00",
      "event": "start",
      "user_id": 1,
      "session_id": 100
    }
  ]
}
```

## ⚙️ Configuration

The pipeline is highly configurable through `config.yaml`:

### Key Configuration Sections

```yaml
# Data Ingestion
ingestion:
  supported_formats: ['csv', 'json']
  timestamp_column: 'timestamp'
  event_column: 'event'

# Preprocessing
preprocessing:
  remove_duplicates: true
  sort_by_timestamp: true
  noise_filter:
    enabled: true
    min_event_interval: 0.1

# Reconstruction
reconstruction:
  method: 'rule_based'  # or 'probabilistic', 'hybrid'
  rule_based:
    min_confidence: 0.7
    max_gap: 300

# Analysis
analysis:
  pattern_detection:
    enabled: true
    min_pattern_length: 3
  anomaly_detection:
    method: 'isolation_forest'
    contamination: 0.1

# Visualization
visualization:
  figure_size: [12, 8]
  save_plots: true
  output_format: 'png'
```

## 🔬 Pipeline Stages

### 1. Data Ingestion
- Load data from CSV/JSON files
- Validate required columns
- Handle timestamp formats
- Generate sample data for testing

### 2. Preprocessing
- Remove duplicates and corrupted entries
- Sort by timestamp
- Normalize timestamps
- Filter noise
- Segment sequences
- Add temporal features

### 3. Feature Extraction
- **Time Features**: Intervals, frequency, periodicity
- **Sequence Features**: Transitions, patterns, repetitions
- **Statistical Features**: Entropy, complexity, correlations
- **Sliding Window**: Local patterns and trends

### 4. Event Reconstruction
- **Rule-based**: Domain knowledge and heuristics
- **Probabilistic**: Markov models and statistical prediction
- **Hybrid**: Combination of multiple methods
- Confidence scoring for reconstructed events

### 5. Sequence Analysis
- **Pattern Detection**: Frequent patterns, motifs, temporal patterns
- **Anomaly Detection**: Statistical and ML-based methods
- **Correlation Analysis**: Event and feature correlations
- **Sequence Metrics**: Complexity, diversity, similarity

### 6. Visualization
- Event timelines and frequency plots
- Pattern networks and heatmaps
- Reconstruction comparisons
- Analysis dashboards
- Anomaly visualizations

### 7. Evaluation
- Reconstruction accuracy metrics
- Pattern quality assessment
- Pipeline performance benchmarking
- Comprehensive reporting

## 📈 Output Files

The pipeline generates several outputs in the `output/` directory:

### Visualizations
- `event_timeline.png` - Event sequence timeline
- `event_frequency.png` - Event type distribution
- `reconstruction_comparison.png` - Before/after reconstruction
- `analysis_dashboard.png` - Comprehensive analysis overview
- And many more...

### Data Files
- `pipeline.log` - Detailed processing log
- Evaluation reports and metrics
- Processed data files (if enabled)

## 🧪 Examples

### Basic Usage
```python
from main import EventReconstructionPipeline

# Initialize pipeline
pipeline = EventReconstructionPipeline()

# Run with sample data
results = pipeline.run_pipeline(
    input_data="your_data.csv",
    run_reconstruction=True,
    run_analysis=True,
    generate_visualizations=True
)

# Access results
print(f"Original events: {len(results['raw_data'])}")
print(f"Reconstructed events: {len(results['reconstructed_data'])}")
print(f"Pipeline score: {results['evaluation_results']['pipeline']['overall_score']:.3f}")
```

### Custom Configuration
```python
# Load custom config
pipeline = EventReconstructionPipeline("my_config.yaml")

# Generate and process sample data
sample_data = pipeline.generate_sample_data(n_events=1000)
results = pipeline.run_pipeline(sample_data)
```

### Step-by-Step Processing
```python
# Individual stage processing
raw_data = pipeline._run_data_ingestion("data.csv")
preprocessed = pipeline._run_preprocessing(raw_data)
reconstructed = pipeline._run_event_reconstruction(preprocessed)
analysis = pipeline._run_sequence_analysis(reconstructed)
```

## 📊 Evaluation Metrics

### Reconstruction Quality
- **Precision/Recall**: Against ground truth (if available)
- **Completeness**: Timeline coverage and sequence preservation
- **Temporal Consistency**: Gap analysis and time accuracy
- **Logical Consistency**: Event sequence validation

### Pattern Quality
- **Diversity**: Pattern variety and uniqueness
- **Support**: Frequency and reliability
- **Complexity**: Pattern sophistication
- **Coverage**: Event space coverage

### Pipeline Performance
- **Throughput**: Events processed per second
- **Memory Efficiency**: Resource usage
- **Robustness**: Error handling and edge cases
- **Scalability**: Performance with data size

## 🎓 Educational Value

This project demonstrates key computer science and data science concepts:

### Technical Skills
- **Software Engineering**: Modular design, configuration management
- **Data Processing**: Cleaning, normalization, feature engineering
- **Algorithms**: Pattern mining, sequence analysis, reconstruction
- **Machine Learning**: Anomaly detection, probabilistic modeling
- **Visualization**: Data storytelling and communication

### Scientific Computing
- **Pipeline Design**: Inspired by real scientific workflows
- **Data Integrity**: Validation and quality assurance
- **Reproducibility**: Configuration-driven processing
- **Evaluation**: Comprehensive metrics and benchmarking

### Graduate Scheme Readiness
- **Data Science Thinking**: Pattern recognition and insight generation
- **Problem Solving**: Complex data reconstruction challenges
- **Technical Communication**: Visualization and reporting
- **System Design**: End-to-end pipeline architecture

## 🔧 Advanced Usage

### Custom Reconstruction Methods
```python
# Extend the reconstructor
class CustomReconstructor(EventReconstructor):
    def reconstruct(self, df):
        # Your custom logic here
        return reconstructed_df

# Use in pipeline
pipeline.event_reconstructor = CustomReconstructor(pipeline.config)
```

### Custom Feature Extraction
```python
# Add custom features
def extract_custom_features(df):
    # Your feature extraction logic
    return df_with_features

pipeline.feature_extractor.extract_custom_features = extract_custom_features
```

### Batch Processing
```python
# Process multiple files
import glob

for file_path in glob.glob("data/*.csv"):
    results = pipeline.run_pipeline(file_path)
    print(f"Processed {file_path}: {results['pipeline_info']['success']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Columns**: Ensure your data has `timestamp` and `event` columns
2. **Timestamp Format**: Use ISO format or configure in `config.yaml`
3. **Memory Issues**: Reduce data size or disable memory-intensive features
4. **Visualization Errors**: Install matplotlib backend or disable visualizations

### Debug Mode
```yaml
general:
  log_level: DEBUG
```

### Performance Tuning
```yaml
preprocessing:
  segmentation:
    enabled: false  # Disable for large datasets

features:
  sliding_window:
    enabled: false  # Disable for faster processing
```

## 📚 References & Inspiration

This pipeline is inspired by established scientific computing tools:

- **scdna-pipe**: Event history reconstruction in single-cell DNA analysis
- **Sequana**: Workflow management for sequence analysis
- **rpg_e2depth**: Event-based reconstruction methods

## 🤝 Contributing

### Development Setup
```bash
git clone <repository-url>
cd "Event Reconstruction & Sequence Analysis"
pip install -r requirements.txt
pip install -e .  # Development install
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
# Format code
black *.py

# Check style
flake8 *.py
```

## 📄 License

This project is provided for educational purposes. Please see the LICENSE file for details.

## 🙏 Acknowledgments

- Scientific computing community for pipeline design patterns
- Open-source data science libraries
- Educational resources in sequence analysis

---

**Built with ❤️ for learning and demonstration purposes**
