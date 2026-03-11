"""
Data Manager
Handles data generation, storage, and management for detector simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from datetime import datetime
import pickle


class DataManager:
    """Main data management class for detector simulations."""
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize data manager.
        
        Args:
            data_directory: Directory to store data files
        """
        self.data_directory = data_directory
        self.datasets = {}
        self.metadata = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_directory, exist_ok=True)
    
    def save_dataset(self, data: np.ndarray, name: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    format: str = 'npz') -> str:
        """
        Save dataset with metadata.
        
        Args:
            data: Data array to save
            name: Dataset name
            metadata: Additional metadata
            format: File format ('npz', 'npy', 'csv', 'pickle')
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
        
        if format == 'npz':
            filepath = os.path.join(self.data_directory, f"{filename}.npz")
            np.savez_compressed(filepath, data=data, metadata=json.dumps(metadata or {}))
            
        elif format == 'npy':
            filepath = os.path.join(self.data_directory, f"{filename}.npy")
            np.save(filepath, data)
            
        elif format == 'csv':
            filepath = os.path.join(self.data_directory, f"{filename}.csv")
            if data.ndim == 1:
                pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                # Flatten 2D data for CSV
                flattened = data.reshape(-1)
                pd.DataFrame(flattened).to_csv(filepath, index=False)
                
        elif format == 'pickle':
            filepath = os.path.join(self.data_directory, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump({'data': data, 'metadata': metadata}, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Store in memory
        self.datasets[name] = {
            'data': data,
            'filepath': filepath,
            'format': format,
            'metadata': metadata or {},
            'timestamp': timestamp
        }
        
        return filepath
    
    def load_dataset(self, filepath: str, format: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to data file
            format: File format (inferred from extension if None)
        
        Returns:
            Tuple of (data, metadata)
        """
        if format is None:
            format = os.path.splitext(filepath)[1][1:]  # Remove dot
        
        if format == 'npz':
            loaded = np.load(filepath)
            data = loaded['data']
            metadata = json.loads(loaded['metadata'].item()) if 'metadata' in loaded else {}
            
        elif format == 'npy':
            data = np.load(filepath)
            metadata = {}
            
        elif format == 'csv':
            df = pd.read_csv(filepath)
            data = df.values.flatten()
            metadata = {'shape': data.shape, 'columns': list(df.columns)}
            
        elif format == 'pkl':
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
            data = loaded['data']
            metadata = loaded.get('metadata', {})
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return data, metadata
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        for filename in os.listdir(self.data_directory):
            if filename.endswith(('.npz', '.npy', '.csv', '.pkl')):
                filepath = os.path.join(self.data_directory, filename)
                stat = os.stat(filepath)
                
                datasets.append({
                    'name': filename,
                    'filepath': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'format': os.path.splitext(filename)[1][1:]
                })
        
        return sorted(datasets, key=lambda x: x['modified'], reverse=True)
    
    def delete_dataset(self, name: str) -> bool:
        """
        Delete dataset by name.
        
        Args:
            name: Dataset name to delete
        
        Returns:
            True if deleted successfully
        """
        if name in self.datasets:
            filepath = self.datasets[name]['filepath']
            if os.path.exists(filepath):
                os.remove(filepath)
            del self.datasets[name]
            return True
        return False
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset."""
        if name in self.datasets:
            dataset = self.datasets[name]
            data = dataset['data']
            
            return {
                'name': name,
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size': data.nbytes,
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'metadata': dataset['metadata'],
                'timestamp': dataset['timestamp'],
                'format': dataset['format']
            }
        return None


class DatasetGenerator:
    """Generate synthetic datasets for testing and validation."""
    
    def __init__(self, data_manager: DataManager):
        """Initialize dataset generator."""
        self.data_manager = data_manager
    
    def generate_calibration_dataset(self, detector_width: int = 100, 
                                   detector_height: int = 100,
                                   num_flat_fields: int = 5,
                                   num_dark_frames: int = 5,
                                   gain_range: Tuple[float, float] = (0.8, 1.2),
                                   noise_range: Tuple[float, float] = (0.1, 0.5)) -> Dict[str, str]:
        """
        Generate calibration dataset with flat fields and dark frames.
        
        Args:
            detector_width: Detector width in pixels
            detector_height: Detector height in pixels
            num_flat_fields: Number of flat field images
            num_dark_frames: Number of dark frame images
            gain_range: Range of gain variations
            noise_range: Range of noise levels
        
        Returns:
            Dictionary with paths to generated files
        """
        dataset_paths = {}
        
        # Generate flat fields
        flat_fields = []
        for i in range(num_flat_fields):
            # Base uniform signal
            flat_field = np.full((detector_height, detector_width), 1.0)
            
            # Add gain variations
            gain_map = np.random.uniform(gain_range[0], gain_range[1], 
                                       (detector_height, detector_width))
            flat_field *= gain_map
            
            # Add noise
            noise_level = np.random.uniform(noise_range[0], noise_range[1])
            flat_field += np.random.normal(0, noise_level, flat_field.shape)
            
            flat_fields.append(flat_field)
        
        # Save flat fields
        flat_field_stack = np.stack(flat_fields)
        flat_path = self.data_manager.save_dataset(
            flat_field_stack, 'flat_fields',
            metadata={
                'type': 'flat_fields',
                'num_frames': num_flat_fields,
                'detector_size': (detector_width, detector_height),
                'gain_range': gain_range,
                'noise_range': noise_range
            }
        )
        dataset_paths['flat_fields'] = flat_path
        
        # Generate dark frames
        dark_frames = []
        for i in range(num_dark_frames):
            # Dark current with spatial variations
            dark_frame = np.random.exponential(0.1, (detector_height, detector_width))
            
            # Add readout noise
            dark_frame += np.random.normal(0, 0.05, dark_frame.shape)
            
            dark_frames.append(dark_frame)
        
        # Save dark frames
        dark_frame_stack = np.stack(dark_frames)
        dark_path = self.data_manager.save_dataset(
            dark_frame_stack, 'dark_frames',
            metadata={
                'type': 'dark_frames',
                'num_frames': num_dark_frames,
                'detector_size': (detector_width, detector_height)
            }
        )
        dataset_paths['dark_frames'] = dark_path
        
        return dataset_paths
    
    def generate_signal_dataset(self, signal_types: List[str], 
                              detector_width: int = 100,
                              detector_height: int = 100,
                              num_samples: int = 10) -> Dict[str, str]:
        """
        Generate dataset with various signal types.
        
        Args:
            signal_types: List of signal types to generate
            detector_width: Detector width in pixels
            detector_height: Detector height in pixels
            num_samples: Number of samples per signal type
        
        Returns:
            Dictionary with paths to generated files
        """
        dataset_paths = {}
        
        for signal_type in signal_types:
            signals = []
            
            for i in range(num_samples):
                if signal_type == 'point_source':
                    # Random point source
                    x = np.random.randint(20, detector_width - 20)
                    y = np.random.randint(20, detector_height - 20)
                    intensity = np.random.uniform(0.5, 2.0)
                    
                    signal = np.zeros((detector_height, detector_width))
                    signal[y, x] = intensity
                    
                elif signal_type == 'gaussian_source':
                    # Random Gaussian source
                    x = np.random.uniform(30, detector_width - 30)
                    y = np.random.uniform(30, detector_height - 30)
                    sigma_x = np.random.uniform(5, 15)
                    sigma_y = np.random.uniform(5, 15)
                    intensity = np.random.uniform(0.5, 2.0)
                    
                    xx, yy = np.meshgrid(np.arange(detector_width), 
                                        np.arange(detector_height))
                    signal = intensity * np.exp(
                        -0.5 * ((xx - x) / sigma_x) ** 2 + 
                        ((yy - y) / sigma_y) ** 2
                    )
                    
                elif signal_type == 'uniform':
                    # Uniform illumination
                    intensity = np.random.uniform(0.1, 1.0)
                    signal = np.full((detector_height, detector_width), intensity)
                    
                elif signal_type == 'multiple_points':
                    # Multiple point sources
                    signal = np.zeros((detector_height, detector_width))
                    num_points = np.random.randint(2, 6)
                    
                    for _ in range(num_points):
                        x = np.random.randint(10, detector_width - 10)
                        y = np.random.randint(10, detector_height - 10)
                        intensity = np.random.uniform(0.2, 1.0)
                        signal[y, x] += intensity
                
                else:
                    # Default to random noise
                    signal = np.random.exponential(0.5, (detector_height, detector_width))
                
                # Add some noise
                signal += np.random.normal(0, 0.05, signal.shape)
                signals.append(signal)
            
            # Save signal dataset
            signal_stack = np.stack(signals)
            signal_path = self.data_manager.save_dataset(
                signal_stack, f'{signal_type}_signals',
                metadata={
                    'type': signal_type,
                    'num_samples': num_samples,
                    'detector_size': (detector_width, detector_height)
                }
            )
            dataset_paths[signal_type] = signal_path
        
        return dataset_paths
    
    def generate_calibration_curve_dataset(self, num_points: int = 20,
                                         noise_level: float = 0.05) -> str:
        """
        Generate dataset for calibration curve fitting.
        
        Args:
            num_points: Number of calibration points
            noise_level: Noise level to add
        
        Returns:
            Path to generated file
        """
        # Known input values
        input_values = np.linspace(0.1, 2.0, num_points)
        
        # Simulate detector response with non-linearity
        true_response = 1.0 * input_values + 0.1 + 0.05 * input_values ** 2
        
        # Add noise
        measured_values = true_response + np.random.normal(0, noise_level, input_values.shape)
        
        # Create dataset
        calibration_data = np.column_stack([input_values, measured_values])
        
        # Save dataset
        calib_path = self.data_manager.save_dataset(
            calibration_data, 'calibration_curve',
            metadata={
                'type': 'calibration_curve',
                'num_points': num_points,
                'noise_level': noise_level,
                'columns': ['input_values', 'measured_values']
            }
        )
        
        return calib_path


class ExperimentLogger:
    """Log experimental results and parameters."""
    
    def __init__(self, log_file: str = "experiment_log.json"):
        """Initialize experiment logger."""
        self.log_file = log_file
        self.experiments = []
        
        # Load existing log if it exists
        if os.path.exists(log_file):
            self.load_log()
    
    def log_experiment(self, experiment_name: str, parameters: Dict[str, Any],
                      results: Dict[str, Any], notes: str = "") -> None:
        """
        Log an experiment with parameters and results.
        
        Args:
            experiment_name: Name of the experiment
            parameters: Experiment parameters
            results: Experiment results
            notes: Additional notes
        """
        experiment_entry = {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'results': results,
            'notes': notes
        }
        
        self.experiments.append(experiment_entry)
        self.save_log()
    
    def save_log(self) -> None:
        """Save experiment log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def load_log(self) -> None:
        """Load experiment log from file."""
        with open(self.log_file, 'r') as f:
            self.experiments = json.load(f)
    
    def get_experiments(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get experiments, optionally filtered by name.
        
        Args:
            name_filter: Filter experiments by name
        
        Returns:
            List of experiment entries
        """
        if name_filter:
            return [exp for exp in self.experiments if name_filter in exp['name']]
        return self.experiments
    
    def export_to_csv(self, filename: str) -> None:
        """Export experiment log to CSV file."""
        data = []
        for exp in self.experiments:
            row = {
                'name': exp['name'],
                'timestamp': exp['timestamp'],
                'notes': exp['notes']
            }
            
            # Add parameters
            for key, value in exp['parameters'].items():
                row[f'param_{key}'] = value
            
            # Add results
            for key, value in exp['results'].items():
                row[f'result_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
