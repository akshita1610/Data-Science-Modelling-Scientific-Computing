"""
Preprocessor Module

Handles comprehensive data cleaning and preparation for event sequences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Comprehensive preprocessor for event data.
    
    Handles data cleaning, outlier detection, and basic transformations
    to prepare data for feature extraction and reconstruction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Preprocessor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for preprocessing parameters
        """
        self.config = config.get('preprocessing', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df (pd.DataFrame): Raw event data
            
        Returns:
            pd.DataFrame: Cleaned event data
        """
        logger.info(f"Starting data cleaning for {len(df)} events")
        
        df_clean = df.copy()
        
        # Remove exact duplicates
        if self.config.get('remove_duplicates', True):
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[self.timestamp_column, self.event_column])
            logger.info(f"Removed {initial_count - len(df_clean)} duplicate events")
        
        # Sort by timestamp
        if self.config.get('sort_by_timestamp', True):
            df_clean = df_clean.sort_values(self.timestamp_column).reset_index(drop=True)
            logger.info("Events sorted by timestamp")
        
        # Remove events with invalid timestamps
        df_clean = self._remove_invalid_timestamps(df_clean)
        
        # Handle missing values in other columns
        df_clean = self._handle_missing_values(df_clean)
        
        # Detect and handle outliers
        if self.config.get('noise_filter', {}).get('enabled', True):
            df_clean = self._filter_noise(df_clean)
        
        logger.info(f"Data cleaning completed. Final count: {len(df_clean)} events")
        return df_clean
    
    def _remove_invalid_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove events with invalid or problematic timestamps."""
        initial_count = len(df)
        
        # Remove events with timestamps in the future
        current_time = pd.Timestamp.now()
        future_mask = pd.to_datetime(df[self.timestamp_column]) > current_time
        if future_mask.any():
            logger.warning(f"Removing {future_mask.sum()} events with future timestamps")
            df = df[~future_mask]
        
        # Remove events with very old timestamps (optional) - more lenient for sample data
        cutoff_date = current_time - pd.Timedelta(days=3650)  # 10 years instead of 1 year
        old_mask = pd.to_datetime(df[self.timestamp_column]) < cutoff_date
        if old_mask.any():
            logger.warning(f"Removing {old_mask.sum()} events with very old timestamps")
            df = df[~old_mask]
        
        # Remove events with duplicate timestamps (same timestamp, different events)
        duplicate_time_mask = df.duplicated(subset=[self.timestamp_column], keep=False)
        if duplicate_time_mask.any():
            logger.warning(f"Found {duplicate_time_mask.sum()} events with duplicate timestamps")
            # Keep only the first event for each timestamp
            df = df.drop_duplicates(subset=[self.timestamp_column], keep='first')
        
        logger.info(f"Removed {initial_count - len(df)} events with invalid timestamps")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in non-critical columns."""
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].isnull().any():
                # Fill missing numeric values with median
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col not in [self.timestamp_column, self.event_column]]
        
        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                # Fill missing categorical values with mode or 'unknown'
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna('unknown')
                logger.info(f"Filled missing values in {col}")
        
        return df
    
    def _filter_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter noise events based on time intervals and event patterns."""
        noise_config = self.config.get('noise_filter', {})
        min_interval = noise_config.get('min_event_interval', 0.1)  # seconds
        
        # Calculate time differences between consecutive events
        df_sorted = df.sort_values(self.timestamp_column)
        time_diffs = df_sorted[self.timestamp_column].diff().dt.total_seconds()
        
        # Identify events that are too close together (potential noise)
        rapid_events_mask = time_diffs < min_interval
        
        if rapid_events_mask.any():
            logger.warning(f"Identified {rapid_events_mask.sum()} potentially noisy events (too close together)")
            
            # Strategy: keep the first event in rapid succession
            rapid_indices = df_sorted.index[rapid_events_mask].tolist()
            # Remove all but the first in each rapid sequence
            indices_to_remove = []
            for i, idx in enumerate(rapid_indices):
                if i > 0 and rapid_indices[i] == rapid_indices[i-1] + 1:
                    indices_to_remove.append(idx)
            
            if indices_to_remove:
                df = df.drop(indices_to_remove)
                logger.info(f"Removed {len(indices_to_remove)} noisy events")
        
        return df
    
    def standardize_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize event names and categories.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with standardized event names
        """
        df_standardized = df.copy()
        
        # Convert event names to lowercase and strip whitespace
        df_standardized[self.event_column] = df_standardized[self.event_column].str.lower().str.strip()
        
        # Group similar events (example mapping)
        event_mapping = {
            'start': 'begin',
            'begin': 'begin',
            'stop': 'end',
            'end': 'end',
            'pause': 'pause',
            'resume': 'resume',
            'error': 'error',
            'err': 'error',
            'warning': 'warning',
            'warn': 'warning',
            'complete': 'complete',
            'finished': 'complete',
            'done': 'complete'
        }
        
        # Apply mapping
        df_standardized[self.event_column] = df_standardized[self.event_column].map(event_mapping).fillna(
            df_standardized[self.event_column]
        )
        
        # Log changes
        original_events = set(df[self.event_column].unique())
        standardized_events = set(df_standardized[self.event_column].unique())
        
        if original_events != standardized_events:
            logger.info(f"Standardized events from {len(original_events)} to {len(standardized_events)} unique types")
        
        return df_standardized
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to the dataset.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with added temporal features
        """
        df_enhanced = df.copy()
        
        # Convert timestamp to datetime if not already
        timestamps = pd.to_datetime(df_enhanced[self.timestamp_column])
        
        # Add basic temporal features
        df_enhanced['hour'] = timestamps.dt.hour
        df_enhanced['day_of_week'] = timestamps.dt.dayofweek
        df_enhanced['month'] = timestamps.dt.month
        df_enhanced['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
        
        # Add time-based cyclical features
        df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
        
        # Add sequence-based features
        df_enhanced['event_index'] = range(len(df_enhanced))
        df_enhanced['time_since_previous'] = timestamps.diff().dt.total_seconds().fillna(0)
        df_enhanced['time_since_first'] = (timestamps - timestamps.min()).dt.total_seconds()
        
        logger.info("Added temporal features to dataset")
        return df_enhanced
    
    def validate_preprocessing(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict:
        """
        Validate preprocessing results and return statistics.
        
        Args:
            df_original (pd.DataFrame): Original data
            df_processed (pd.DataFrame): Processed data
            
        Returns:
            Dict: Validation statistics
        """
        stats = {
            'original_events': len(df_original),
            'processed_events': len(df_processed),
            'events_removed': len(df_original) - len(df_processed),
            'removal_percentage': (len(df_original) - len(df_processed)) / len(df_original) * 100,
            'time_range_original': {
                'start': df_original[self.timestamp_column].min(),
                'end': df_original[self.timestamp_column].max()
            },
            'time_range_processed': {
                'start': df_processed[self.timestamp_column].min(),
                'end': df_processed[self.timestamp_column].max()
            },
            'unique_events_original': df_original[self.event_column].nunique(),
            'unique_events_processed': df_processed[self.event_column].nunique()
        }
        
        logger.info(f"Preprocessing validation: {stats['events_removed']} events removed ({stats['removal_percentage']:.2f}%)")
        return stats
