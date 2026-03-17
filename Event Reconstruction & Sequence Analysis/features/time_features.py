"""
Time-based Feature Extraction Module

Extracts features related to time intervals, frequencies, and temporal patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeFeatureExtractor:
    """
    Extracts time-based features from event sequences.
    
    This class specializes in extracting features related to temporal patterns,
    intervals, frequencies, and other time-based characteristics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the TimeFeatureExtractor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for feature extraction
        """
        self.config = config.get('features', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive time-based features.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with time-based features
        """
        df_time = df.copy()
        
        # Extract basic time interval features
        df_time = self._extract_time_intervals(df_time)
        
        # Extract time-based statistical features
        df_time = self._extract_time_statistics(df_time)
        
        # Extract periodic/cyclical features
        df_time = self._extract_periodic_features(df_time)
        
        # Extract temporal context features
        df_time = self._extract_temporal_context(df_time)
        
        logger.info("Time-based features extracted successfully")
        return df_time
    
    def extract_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract frequency-based features from event sequences.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with frequency features
        """
        df_freq = df.copy()
        
        # Get frequency configuration
        freq_config = self.config.get('event_frequency', {})
        window_size = freq_config.get('window_size', 50)
        
        # Calculate event frequency over time
        df_freq = self._calculate_event_frequency(df_freq, window_size)
        
        # Calculate rolling frequency statistics
        df_freq = self._calculate_rolling_frequency(df_freq, window_size)
        
        # Calculate frequency by event type
        df_freq = self._calculate_event_type_frequency(df_freq)
        
        logger.info("Frequency features extracted successfully")
        return df_freq
    
    def _extract_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time interval features between events."""
        # Ensure timestamps are in datetime format
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        # Time since previous event
        df['time_since_previous'] = timestamps.diff().dt.total_seconds()
        df['time_since_previous'] = df['time_since_previous'].fillna(0)
        
        # Time until next event
        df['time_until_next'] = timestamps.diff(-1).dt.total_seconds().abs()
        df['time_until_next'] = df['time_until_next'].fillna(0)
        
        # Time since first event in sequence
        df['time_since_first'] = (timestamps - timestamps.min()).dt.total_seconds()
        
        # Time until last event in sequence
        df['time_until_last'] = (timestamps.max() - timestamps).dt.total_seconds()
        
        # Relative time position (0 to 1)
        total_duration = df['time_since_first'].max()
        if total_duration > 0:
            df['relative_time_position'] = df['time_since_first'] / total_duration
        else:
            df['relative_time_position'] = 0
        
        return df
    
    def _extract_time_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from time intervals."""
        if len(df) < 2:
            return df
        
        # Rolling statistics for time intervals
        window_sizes = [3, 5, 10]
        
        for window in window_sizes:
            if len(df) >= window:
                # Rolling mean of time gaps
                df[f'time_gap_mean_{window}'] = df['time_since_previous'].rolling(window=window).mean()
                
                # Rolling standard deviation of time gaps
                df[f'time_gap_std_{window}'] = df['time_since_previous'].rolling(window=window).std()
                
                # Rolling coefficient of variation
                mean_gap = df['time_since_previous'].rolling(window=window).mean()
                std_gap = df['time_since_previous'].rolling(window=window).std()
                df[f'time_gap_cv_{window}'] = std_gap / (mean_gap + 1e-6)
        
        # Global time statistics
        time_gaps = df['time_since_previous'].values
        
        # Percentile-based features
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            df[f'time_gap_percentile_{p}'] = np.percentile(time_gaps, p)
        
        # Distribution features
        df['time_gap_skewness'] = stats.skew(time_gaps)
        df['time_gap_kurtosis'] = stats.kurtosis(time_gaps)
        
        return df
    
    def _extract_periodic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract periodic and cyclical time features."""
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        # Basic time components
        df['hour'] = timestamps.dt.hour
        df['day_of_week'] = timestamps.dt.dayofweek
        df['day_of_month'] = timestamps.dt.day
        df['month'] = timestamps.dt.month
        df['quarter'] = timestamps.dt.quarter
        df['week_of_year'] = timestamps.dt.isocalendar().week
        
        # Cyclical encoding using sine and cosine
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business/weekend features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)
        
        return df
    
    def _extract_temporal_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal context features."""
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        # Time since specific reference points
        current_time = pd.Timestamp.now()
        
        # Time since start of day
        df['time_since_midnight'] = (timestamps - timestamps.dt.normalize()).dt.total_seconds()
        
        # Time since start of week
        week_start = timestamps - pd.to_timedelta(timestamps.dt.dayofweek, unit='D')
        df['time_since_week_start'] = (timestamps - week_start).dt.total_seconds()
        
        # Time since start of month
        month_start = timestamps - pd.to_timedelta(timestamps.dt.day - 1, unit='D')
        df['time_since_month_start'] = (timestamps - month_start).dt.total_seconds()
        
        # Age of event (time since it occurred)
        df['event_age'] = (current_time - timestamps).dt.total_seconds()
        
        return df
    
    def _calculate_event_frequency(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Calculate event frequency over time windows."""
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Calculate events per time window
        for window in [window_size, window_size*2, window_size*4]:
            if len(df_sorted) >= window:
                # Rolling count of events
                df_sorted[f'events_count_{window}'] = df_sorted[self.event_column].rolling(window=window).count()
                
                # Events per unit time
                time_window = df_sorted['time_since_previous'].rolling(window=window).sum()
                df_sorted[f'events_per_time_{window}'] = df_sorted[f'events_count_{window}'] / (time_window + 1e-6)
        
        return df_sorted
    
    def _calculate_rolling_frequency(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Calculate rolling frequency statistics."""
        if len(df) < window_size:
            return df
        
        # Rolling frequency by event type
        event_types = df[self.event_column].unique()
        
        for event_type in event_types:
            # Create binary series for this event type
            event_mask = (df[self.event_column] == event_type).astype(int)
            
            # Rolling sum (frequency)
            df[f'freq_{event_type}_{window_size}'] = event_mask.rolling(window=window_size).sum()
            
            # Rolling proportion
            df[f'prop_{event_type}_{window_size}'] = df[f'freq_{event_type}_{window_size}'] / window_size
        
        return df
    
    def _calculate_event_type_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate frequency features for each event type."""
        # Global frequency of each event type
        event_counts = df[self.event_column].value_counts()
        total_events = len(df)
        
        # Add frequency columns for each event type
        for event_type, count in event_counts.items():
            df[f'global_freq_{event_type}'] = count / total_events
        
        # Cumulative frequency by event type
        for event_type in event_counts.index:
            cumulative_count = (df[self.event_column] == event_type).cumsum()
            df[f'cumulative_freq_{event_type}'] = cumulative_count / (np.arange(len(df)) + 1)
        
        return df
    
    def extract_burst_features(self, df: pd.DataFrame, 
                             burst_threshold_seconds: float = 1.0) -> pd.DataFrame:
        """
        Extract features related to event bursts (rapid succession).
        
        Args:
            df (pd.DataFrame): Event data
            burst_threshold_seconds (float): Time threshold for defining bursts
            
        Returns:
            pd.DataFrame: Data with burst features
        """
        df_burst = df.copy()
        
        # Identify burst events (events closer than threshold)
        is_burst = df_burst['time_since_previous'] < burst_threshold_seconds
        df_burst['is_burst_event'] = is_burst.astype(int)
        
        # Calculate burst statistics
        if len(df_burst) > 0:
            # Burst length (consecutive burst events)
            df_burst['burst_length'] = (is_burst != is_burst.shift()).cumsum()
            burst_lengths = df_burst.groupby('burst_length')['is_burst_event'].sum()
            if len(burst_lengths) > 0:
                df_burst['current_burst_length'] = df_burst['burst_length'].map(burst_lengths)
            else:
                df_burst['current_burst_length'] = 1
            
            # Time since last burst
            last_burst_times = df_burst[is_burst][self.timestamp_column]
            if len(last_burst_times) > 0:
                df_burst['time_since_last_burst'] = (
                    pd.to_datetime(df_burst[self.timestamp_column]) - 
                    last_burst_times.iloc[-1]
                ).dt.total_seconds().fillna(df_burst['time_since_previous'])
            else:
                df_burst['time_since_last_burst'] = df_burst['time_since_previous']
        
        return df_burst
    
    def extract_rhythm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rhythmic patterns from event sequences.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with rhythm features
        """
        df_rhythm = df.copy()
        
        if len(df_rhythm) < 3:
            return df_rhythm
        
        # Calculate autocorrelation of time gaps
        time_gaps = df_rhythm['time_since_previous'].values
        
        # Autocorrelation at different lags
        lags = [1, 2, 3, 5, 10]
        for lag in lags:
            if len(time_gaps) > lag:
                autocorr = np.corrcoef(time_gaps[:-lag], time_gaps[lag:])[0, 1]
                df_rhythm[f'autocorr_lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0
            else:
                df_rhythm[f'autocorr_lag_{lag}'] = 0
        
        # Periodicity detection using FFT
        if len(time_gaps) >= 10:
            try:
                # Simple periodicity detection
                fft_vals = np.fft.fft(time_gaps - np.mean(time_gaps))
                power_spectrum = np.abs(fft_vals) ** 2
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                df_rhythm['dominant_frequency'] = dominant_freq_idx
                
                # Periodicity strength
                df_rhythm['periodicity_strength'] = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum[1:])
                
            except Exception as e:
                logger.warning(f"Could not calculate rhythm features: {e}")
                df_rhythm['dominant_frequency'] = 0
                df_rhythm['periodicity_strength'] = 0
        
        return df_rhythm
