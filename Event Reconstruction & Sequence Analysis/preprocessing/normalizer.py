"""
Normalizer Module

Handles timestamp normalization and data scaling for event sequences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Data normalizer for event sequences.
    
    Handles timestamp normalization, feature scaling, and data transformation
    to prepare data for machine learning and analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Normalizer with configuration.
        
        Args:
            config (Dict): Configuration dictionary for normalization parameters
        """
        self.config = config.get('preprocessing', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Initialize scalers
        self.scalers = {}
        self.scaler_params = {}
        
    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps to a consistent format and reference point.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with normalized timestamps
        """
        df_normalized = df.copy()
        
        # Ensure timestamps are in datetime format
        df_normalized[self.timestamp_column] = pd.to_datetime(df_normalized[self.timestamp_column])
        
        # Get timestamp format from config
        timestamp_format = self.config.get('timestamp_format', '%Y-%m-%d %H:%M:%S')
        
        # Convert to specified format (string representation)
        df_normalized['timestamp_formatted'] = df_normalized[self.timestamp_column].dt.strftime(timestamp_format)
        
        # Add normalized time features
        timestamps = df_normalized[self.timestamp_column]
        
        # Normalize to seconds since first event
        first_timestamp = timestamps.min()
        df_normalized['time_seconds'] = (timestamps - first_timestamp).dt.total_seconds()
        
        # Normalize to minutes since first event
        df_normalized['time_minutes'] = df_normalized['time_seconds'] / 60
        
        # Normalize to hours since first event
        df_normalized['time_hours'] = df_normalized['time_minutes'] / 60
        
        # Add relative time features
        df_normalized['time_of_day_seconds'] = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
        df_normalized['day_of_year'] = timestamps.dt.dayofyear
        
        logger.info("Timestamps normalized successfully")
        return df_normalized
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numeric features using various scaling methods.
        
        Args:
            df (pd.DataFrame): Event data
            method (str): Scaling method ('standard', 'minmax', 'robust')
            columns (Optional[List[str]]): Columns to scale (if None, scale all numeric)
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        df_scaled = df.copy()
        
        # Identify numeric columns
        if columns is None:
            numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude timestamp and index columns
            exclude_columns = [self.timestamp_column, 'event_index', 'hour', 'day_of_week', 'month']
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        else:
            numeric_columns = [col for col in columns if col in df_scaled.columns]
        
        if not numeric_columns:
            logger.warning("No numeric columns found for scaling")
            return df_scaled
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        # Fit and transform
        scaler.fit(df_scaled[numeric_columns])
        df_scaled[numeric_columns] = scaler.transform(df_scaled[numeric_columns])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        self.scaler_params[method] = {
            'columns': numeric_columns,
            'mean': scaler.mean_ if hasattr(scaler, 'mean_') else None,
            'scale': scaler.scale_ if hasattr(scaler, 'scale_') else None
        }
        
        logger.info(f"Scaled {len(numeric_columns)} features using {method} scaling")
        return df_scaled
    
    def normalize_event_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize event sequences for analysis.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with normalized sequences
        """
        df_normalized = df.copy()
        
        # Create event sequence index
        df_normalized['sequence_index'] = range(len(df_normalized))
        
        # Calculate relative position in sequence (0 to 1)
        df_normalized['relative_position'] = df_normalized['sequence_index'] / (len(df_normalized) - 1) if len(df_normalized) > 1 else 0
        
        # Calculate cumulative event count by type
        event_counts = df_normalized.groupby(self.event_column).cumcount() + 1
        df_normalized['event_type_count'] = event_counts
        
        # Calculate event frequency (events per hour)
        if 'time_hours' in df_normalized.columns:
            df_normalized['events_per_hour'] = df_normalized['sequence_index'] / (df_normalized['time_hours'] + 1e-6)
        else:
            # Calculate time in hours from timestamps
            time_diff = (pd.to_datetime(df_normalized[self.timestamp_column]) - 
                        pd.to_datetime(df_normalized[self.timestamp_column].min()))
            df_normalized['time_hours_calc'] = time_diff.dt.total_seconds() / 3600
            df_normalized['events_per_hour'] = df_normalized['sequence_index'] / (df_normalized['time_hours_calc'] + 1e-6)
        
        logger.info("Event sequences normalized")
        return df_normalized
    
    def encode_events(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """
        Encode event types to numerical format.
        
        Args:
            df (pd.DataFrame): Event data
            method (str): Encoding method ('label', 'onehot', 'frequency')
            
        Returns:
            pd.DataFrame: Data with encoded events
        """
        df_encoded = df.copy()
        
        if method == 'label':
            # Label encoding
            unique_events = df_encoded[self.event_column].unique()
            event_to_label = {event: i for i, event in enumerate(unique_events)}
            df_encoded['event_encoded'] = df_encoded[self.event_column].map(event_to_label)
            
            # Store mapping
            self.event_mapping = event_to_label
            self.reverse_event_mapping = {i: event for event, i in event_to_label.items()}
            
        elif method == 'onehot':
            # One-hot encoding
            event_dummies = pd.get_dummies(df_encoded[self.event_column], prefix='event')
            df_encoded = pd.concat([df_encoded, event_dummies], axis=1)
            
        elif method == 'frequency':
            # Frequency encoding
            event_counts = df_encoded[self.event_column].value_counts()
            df_encoded['event_frequency'] = df_encoded[self.event_column].map(event_counts)
            
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        
        logger.info(f"Events encoded using {method} method")
        return df_encoded
    
    def create_time_windows(self, df: pd.DataFrame, 
                           window_size_minutes: int = 60,
                           overlap_minutes: int = 0) -> List[pd.DataFrame]:
        """
        Create time windows for analysis.
        
        Args:
            df (pd.DataFrame): Event data
            window_size_minutes (int): Size of each window in minutes
            overlap_minutes (int): Overlap between windows in minutes
            
        Returns:
            List[pd.DataFrame]: List of time-windowed DataFrames
        """
        timestamps = pd.to_datetime(df[self.timestamp_column])
        start_time = timestamps.min()
        end_time = timestamps.max()
        
        window_size = timedelta(minutes=window_size_minutes)
        overlap = timedelta(minutes=overlap_minutes)
        step_size = window_size - overlap
        
        windows = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + window_size
            
            # Filter events in current window
            window_mask = (timestamps >= current_start) & (timestamps < current_end)
            window_data = df[window_mask].copy()
            
            if len(window_data) > 0:
                # Add window metadata
                window_data['window_id'] = len(windows)
                window_data['window_start'] = current_start
                window_data['window_end'] = current_end
                window_data['window_position'] = (timestamps[window_mask] - current_start).dt.total_seconds()
                
                windows.append(window_data)
            
            current_start += step_size
        
        logger.info(f"Created {len(windows)} time windows")
        return windows
    
    def inverse_transform(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            df (pd.DataFrame): Scaled data
            method (str): Scaling method to inverse
            
        Returns:
            pd.DataFrame: Data with inverse transformed features
        """
        if method not in self.scalers:
            raise ValueError(f"No scaler found for method: {method}")
        
        df_inverse = df.copy()
        scaler = self.scalers[method]
        columns = self.scaler_params[method]['columns']
        
        # Check if columns exist
        available_columns = [col for col in columns if col in df_inverse.columns]
        
        if available_columns:
            df_inverse[available_columns] = scaler.inverse_transform(df_inverse[available_columns])
            logger.info(f"Inverse transformed {len(available_columns)} features")
        else:
            logger.warning("No columns found for inverse transformation")
        
        return df_inverse
    
    def get_normalization_stats(self, df_original: pd.DataFrame, 
                              df_normalized: pd.DataFrame) -> Dict:
        """
        Get statistics about the normalization process.
        
        Args:
            df_original (pd.DataFrame): Original data
            df_normalized (pd.DataFrame): Normalized data
            
        Returns:
            Dict: Normalization statistics
        """
        stats = {
            'timestamp_range': {
                'original': {
                    'start': df_original[self.timestamp_column].min(),
                    'end': df_original[self.timestamp_column].max(),
                    'duration': df_original[self.timestamp_column].max() - df_original[self.timestamp_column].min()
                },
                'normalized': {
                    'start_seconds': df_normalized['time_seconds'].min() if 'time_seconds' in df_normalized else None,
                    'end_seconds': df_normalized['time_seconds'].max() if 'time_seconds' in df_normalized else None,
                    'duration_seconds': df_normalized['time_seconds'].max() - df_normalized['time_seconds'].min() if 'time_seconds' in df_normalized else None
                }
            },
            'event_encoding': {
                'unique_events': df_original[self.event_column].nunique(),
                'encoding_method': 'label' if 'event_encoded' in df_normalized else None
            },
            'scaling_applied': list(self.scalers.keys())
        }
        
        return stats
