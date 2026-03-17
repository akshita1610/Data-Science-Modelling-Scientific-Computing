"""
Data Loader Module

Handles loading event data from various file formats (CSV, JSON) and performs
initial validation and basic preprocessing.
"""

import pandas as pd
import json
import numpy as np
from typing import Union, Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A versatile data loader for event logs supporting multiple formats.
    
    Attributes:
        config (Dict): Configuration dictionary for loading parameters
        supported_formats (List[str]): List of supported file formats
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing loading parameters
        """
        self.config = config.get('ingestion', {})
        self.supported_formats = self.config.get('supported_formats', ['csv', 'json'])
        self.timestamp_column = self.config.get('timestamp_column', 'timestamp')
        self.event_column = self.config.get('event_column', 'event')
        self.required_columns = self.config.get('required_columns', ['timestamp', 'event'])
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load event data from the specified file.
        
        Args:
            file_path (Union[str, Path]): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded event data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported or data is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_format = file_path.suffix.lower().lstrip('.')
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}. "
                           f"Supported formats: {self.supported_formats}")
        
        logger.info(f"Loading data from {file_path} (format: {file_format})")
        
        if file_format == 'csv':
            df = self._load_csv(file_path)
        elif file_format == 'json':
            df = self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        # Validate and clean the data
        df = self._validate_and_clean(df)
        
        logger.info(f"Successfully loaded {len(df)} events from {file_path}")
        return df
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'events' in data:
                    df = pd.DataFrame(data['events'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Invalid JSON structure")
            
            return df
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {e}")
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data.
        
        Args:
            df (pd.DataFrame): Raw loaded data
            
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=self.required_columns)
        cleaned_count = len(df)
        
        if cleaned_count < initial_count:
            logger.warning(f"Removed {initial_count - cleaned_count} rows with missing data")
        
        # Convert timestamp column to datetime
        try:
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
        except Exception as e:
            raise ValueError(f"Error converting timestamps: {e}")
        
        # Remove exact duplicates
        if self.config.get('remove_duplicates', True):
            df = df.drop_duplicates(subset=[self.timestamp_column, self.event_column])
            logger.info(f"Removed {initial_count - len(df)} duplicate events")
        
        return df
    
    def generate_sample_data(self, n_events: int = 1000, 
                           event_types: Optional[List[str]] = None,
                           output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Generate sample event data for testing purposes.
        
        Args:
            n_events (int): Number of events to generate
            event_types (Optional[List[str]]): List of event types to use
            output_path (Optional[Union[str, Path]]): Path to save the generated data
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        if event_types is None:
            event_types = ['start', 'stop', 'pause', 'resume', 'error', 'warning', 'complete']
        
        # Generate timestamps
        start_time = pd.Timestamp('2023-01-01 00:00:00')
        timestamps = [start_time + pd.Timedelta(hours=np.random.uniform(0, 24)) 
                      for _ in range(n_events)]
        timestamps.sort()
        
        # Generate events
        events = np.random.choice(event_types, n_events)
        
        # Create additional metadata
        metadata = {
            'user_id': np.random.randint(1, 10, n_events),
            'session_id': np.random.randint(100, 200, n_events),
            'duration': np.random.exponential(5, n_events)  # seconds
        }
        
        # Create DataFrame
        df = pd.DataFrame({
            self.timestamp_column: timestamps,
            self.event_column: events,
            **metadata
        })
        
        if output_path:
            self._save_sample_data(df, output_path)
        
        return df
    
    def _save_sample_data(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """Save generated sample data to file."""
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.json':
            df.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError("Output format must be .csv or .json")
        
        logger.info(f"Sample data saved to {output_path}")
