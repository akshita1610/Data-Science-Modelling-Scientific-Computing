"""
Event Stream Module

Handles real-time event stream simulation and processing for time-ordered data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EventStream:
    """
    Simulates and processes real-time event streams.
    
    This class can generate synthetic event streams that mimic sensor data
    or process existing event data as a stream.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the EventStream with configuration.
        
        Args:
            config (Dict): Configuration dictionary for stream parameters
        """
        self.config = config.get('ingestion', {})
        self.timestamp_column = self.config.get('timestamp_column', 'timestamp')
        self.event_column = self.config.get('event_column', 'event')
        
    def simulate_stream(self, 
                       duration_minutes: int = 60,
                       events_per_minute: float = 2.0,
                       event_types: Optional[List[str]] = None,
                       noise_probability: float = 0.1) -> Iterator[Dict]:
        """
        Simulate a real-time event stream.
        
        Args:
            duration_minutes (int): Duration of simulation in minutes
            events_per_minute (float): Average events per minute
            event_types (Optional[List[str]]): Types of events to generate
            noise_probability (float): Probability of generating noise events
            
        Yields:
            Dict: Event dictionary with timestamp and event data
        """
        if event_types is None:
            event_types = ['sensor_read', 'alert', 'status_update', 'heartbeat', 'error']
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        current_time = start_time
        
        while current_time < end_time:
            # Determine if we should generate an event at this timestamp
            if np.random.random() < events_per_minute / 60:  # Convert to probability per second
                event = self._generate_event(current_time, event_types, noise_probability)
                yield event
            
            # Advance time (random interval between 0.1 and 10 seconds)
            current_time += timedelta(seconds=np.random.uniform(0.1, 10))
    
    def _generate_event(self, 
                       timestamp: datetime, 
                       event_types: List[str],
                       noise_probability: float) -> Dict:
        """Generate a single event with metadata."""
        # Determine if this is a noise event
        if np.random.random() < noise_probability:
            event_type = 'noise'
            severity = 'low'
        else:
            event_type = np.random.choice(event_types)
            severity = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05])
        
        event = {
            self.timestamp_column: timestamp,
            self.event_column: event_type,
            'severity': severity,
            'source': f'sensor_{np.random.randint(1, 5)}',
            'value': np.random.uniform(0, 100) if event_type == 'sensor_read' else None
        }
        
        return event
    
    def process_dataframe_as_stream(self, 
                                  df: pd.DataFrame,
                                  batch_size: int = 10) -> Iterator[pd.DataFrame]:
        """
        Process an existing DataFrame as a stream of batches.
        
        Args:
            df (pd.DataFrame): DataFrame to process as stream
            batch_size (int): Number of events per batch
            
        Yields:
            pd.DataFrame: Batch of events
        """
        # Sort by timestamp if not already sorted
        df_sorted = df.sort_values(self.timestamp_column)
        
        for i in range(0, len(df_sorted), batch_size):
            batch = df_sorted.iloc[i:i+batch_size]
            yield batch
    
    def stream_to_dataframe(self, 
                          stream_iterator: Iterator[Dict],
                          max_events: Optional[int] = None) -> pd.DataFrame:
        """
        Convert a stream iterator to a DataFrame.
        
        Args:
            stream_iterator (Iterator[Dict]): Stream of events
            max_events (Optional[int]): Maximum number of events to collect
            
        Returns:
            pd.DataFrame: Collected events as DataFrame
        """
        events = []
        
        for i, event in enumerate(stream_iterator):
            if max_events and i >= max_events:
                break
            events.append(event)
        
        return pd.DataFrame(events)
    
    def add_stream_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add stream-specific metadata to events.
        
        Args:
            df (pd.DataFrame): Event DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added metadata
        """
        df_copy = df.copy()
        
        # Add time-based features
        df_copy['hour_of_day'] = pd.to_datetime(df_copy[self.timestamp_column]).dt.hour
        df_copy['day_of_week'] = pd.to_datetime(df_copy[self.timestamp_column]).dt.dayofweek
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6])
        
        # Add sequence-based features
        df_copy['event_sequence'] = range(len(df_copy))
        df_copy['time_since_previous'] = df_copy[self.timestamp_column].diff()
        df_copy['time_since_previous'] = df_copy['time_since_previous'].fillna(pd.Timedelta(seconds=0))
        
        return df_copy
    
    def detect_stream_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect basic anomalies in the event stream.
        
        Args:
            df (pd.DataFrame): Event DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with anomaly flags
        """
        df_copy = df.copy()
        
        # Detect unusually long gaps between events
        time_gaps = df_copy[self.timestamp_column].diff().dt.total_seconds()
        gap_threshold = time_gaps.quantile(0.95)  # Events in top 5% of gaps
        df_copy['unusual_gap'] = (time_gaps > gap_threshold).fillna(False)
        
        # Detect rapid succession of events (possible burst)
        time_gaps_seconds = time_gaps.fillna(0)
        burst_threshold = time_gaps_seconds.quantile(0.05)  # Events in bottom 5% of gaps
        df_copy['rapid_succession'] = (time_gaps_seconds < burst_threshold).fillna(False)
        
        # Count anomalies
        anomaly_count = df_copy[['unusual_gap', 'rapid_succession']].any(axis=1).sum()
        logger.info(f"Detected {anomaly_count} anomalies in {len(df_copy)} events")
        
        return df_copy
