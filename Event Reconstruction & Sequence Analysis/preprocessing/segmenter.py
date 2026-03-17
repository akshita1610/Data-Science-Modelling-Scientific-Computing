"""
Segmenter Module

Handles sequence segmentation and windowing for event analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Segmenter:
    """
    Sequence segmenter for event data.
    
    Handles segmentation of event sequences into manageable chunks
    for analysis and reconstruction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Segmenter with configuration.
        
        Args:
            config (Dict): Configuration dictionary for segmentation parameters
        """
        self.config = config.get('preprocessing', {})
        self.segment_config = self.config.get('segmentation', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def segment_by_time(self, df: pd.DataFrame, 
                       window_size_minutes: int = 60,
                       overlap_minutes: int = 0) -> List[pd.DataFrame]:
        """
        Segment events by fixed time windows.
        
        Args:
            df (pd.DataFrame): Event data
            window_size_minutes (int): Size of each window in minutes
            overlap_minutes (int): Overlap between windows in minutes
            
        Returns:
            List[pd.DataFrame]: List of segmented DataFrames
        """
        timestamps = pd.to_datetime(df[self.timestamp_column])
        start_time = timestamps.min()
        end_time = timestamps.max()
        
        window_size = timedelta(minutes=window_size_minutes)
        overlap = timedelta(minutes=overlap_minutes)
        step_size = window_size - overlap
        
        segments = []
        current_start = start_time
        segment_id = 0
        
        while current_start < end_time:
            current_end = current_start + window_size
            
            # Filter events in current window
            window_mask = (timestamps >= current_start) & (timestamps < current_end)
            segment_data = df[window_mask].copy()
            
            if len(segment_data) > 0:
                # Add segment metadata
                segment_data['segment_id'] = segment_id
                segment_data['segment_start'] = current_start
                segment_data['segment_end'] = current_end
                segment_data['segment_duration_minutes'] = window_size_minutes
                segment_data['relative_time_in_segment'] = (timestamps[window_mask] - current_start).dt.total_seconds() / 60
                
                segments.append(segment_data)
                segment_id += 1
            
            current_start += step_size
        
        logger.info(f"Created {len(segments)} time-based segments")
        return segments
    
    def segment_by_events(self, df: pd.DataFrame, 
                         events_per_segment: int = 100,
                         overlap_events: int = 0) -> List[pd.DataFrame]:
        """
        Segment events by fixed number of events.
        
        Args:
            df (pd.DataFrame): Event data
            events_per_segment (int): Number of events per segment
            overlap_events (int): Number of overlapping events between segments
            
        Returns:
            List[pd.DataFrame]: List of segmented DataFrames
        """
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        total_events = len(df_sorted)
        
        segments = []
        step_size = events_per_segment - overlap_events
        segment_id = 0
        
        for start_idx in range(0, total_events, step_size):
            end_idx = start_idx + events_per_segment
            
            if start_idx >= total_events:
                break
            
            # Get segment data
            segment_data = df_sorted.iloc[start_idx:min(end_idx, total_events)].copy()
            
            # Add segment metadata
            segment_data['segment_id'] = segment_id
            segment_data['segment_start_event'] = start_idx
            segment_data['segment_end_event'] = min(end_idx, total_events) - 1
            segment_data['segment_event_count'] = len(segment_data)
            segment_data['relative_position_in_segment'] = range(len(segment_data))
            
            segments.append(segment_data)
            segment_id += 1
        
        logger.info(f"Created {len(segments)} event-based segments")
        return segments
    
    def segment_by_activity(self, df: pd.DataFrame, 
                          max_gap_minutes: int = 30,
                          min_events_per_segment: int = 5) -> List[pd.DataFrame]:
        """
        Segment events based on activity patterns (gaps between events).
        
        Args:
            df (pd.DataFrame): Event data
            max_gap_minutes (int): Maximum gap in minutes before starting new segment
            min_events_per_segment (int): Minimum events required for a valid segment
            
        Returns:
            List[pd.DataFrame]: List of segmented DataFrames
        """
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Calculate time gaps between consecutive events
        timestamps = pd.to_datetime(df_sorted[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
        
        segments = []
        current_segment_events = []
        segment_id = 0
        
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if i == 0:
                # First event always starts a new segment
                current_segment_events.append(row)
            else:
                # Check gap from previous event
                gap = time_gaps.iloc[i]
                
                if gap > max_gap_minutes:
                    # Gap too large, start new segment
                    if len(current_segment_events) >= min_events_per_segment:
                        segment_df = pd.DataFrame(current_segment_events)
                        segment_df['segment_id'] = segment_id
                        segment_df['segment_gap_based'] = True
                        segment_df['relative_position_in_segment'] = range(len(segment_df))
                        segments.append(segment_df)
                        segment_id += 1
                    
                    current_segment_events = [row]
                else:
                    # Continue current segment
                    current_segment_events.append(row)
        
        # Add final segment
        if len(current_segment_events) >= min_events_per_segment:
            segment_df = pd.DataFrame(current_segment_events)
            segment_df['segment_id'] = segment_id
            segment_df['segment_gap_based'] = True
            segment_df['relative_position_in_segment'] = range(len(segment_df))
            segments.append(segment_df)
        
        logger.info(f"Created {len(segments)} activity-based segments")
        return segments
    
    def segment_by_event_types(self, df: pd.DataFrame, 
                             delimiter_events: Optional[List[str]] = None) -> List[pd.DataFrame]:
        """
        Segment events based on specific delimiter event types.
        
        Args:
            df (pd.DataFrame): Event data
            delimiter_events (Optional[List[str]]): Events that mark segment boundaries
            
        Returns:
            List[pd.DataFrame]: List of segmented DataFrames
        """
        if delimiter_events is None:
            delimiter_events = ['start', 'begin', 'complete', 'end', 'stop']
        
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        segments = []
        current_segment_events = []
        segment_id = 0
        
        for idx, row in df_sorted.iterrows():
            event_type = row[self.event_column].lower()
            
            if event_type in delimiter_events and len(current_segment_events) > 0:
                # End current segment
                segment_df = pd.DataFrame(current_segment_events)
                segment_df['segment_id'] = segment_id
                segment_df['segment_delimiter_based'] = True
                segment_df['relative_position_in_segment'] = range(len(segment_df))
                segments.append(segment_df)
                segment_id += 1
                
                # Start new segment with delimiter event
                current_segment_events = [row]
            else:
                current_segment_events.append(row)
        
        # Add final segment
        if len(current_segment_events) > 0:
            segment_df = pd.DataFrame(current_segment_events)
            segment_df['segment_id'] = segment_id
            segment_df['segment_delimiter_based'] = True
            segment_df['relative_position_in_segment'] = range(len(segment_df))
            segments.append(segment_df)
        
        logger.info(f"Created {len(segments)} delimiter-based segments")
        return segments
    
    def adaptive_segmentation(self, df: pd.DataFrame, 
                            target_complexity: float = 0.5,
                            min_segment_size: int = 10,
                            max_segment_size: int = 200) -> List[pd.DataFrame]:
        """
        Adaptive segmentation based on event complexity.
        
        Args:
            df (pd.DataFrame): Event data
            target_complexity (float): Target complexity score (0-1)
            min_segment_size (int): Minimum segment size
            max_segment_size (int): Maximum segment size
            
        Returns:
            List[pd.DataFrame]: List of segmented DataFrames
        """
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        segments = []
        segment_id = 0
        current_start = 0
        
        while current_start < len(df_sorted):
            # Try different segment sizes and calculate complexity
            best_segment = None
            best_complexity_diff = float('inf')
            
            for size in range(min_segment_size, min(max_segment_size + 1, len(df_sorted) - current_start)):
                segment_data = df_sorted.iloc[current_start:current_start + size]
                complexity = self._calculate_segment_complexity(segment_data)
                complexity_diff = abs(complexity - target_complexity)
                
                if complexity_diff < best_complexity_diff:
                    best_complexity_diff = complexity_diff
                    best_segment = segment_data.copy()
            
            if best_segment is not None:
                best_segment['segment_id'] = segment_id
                best_segment['segment_adaptive'] = True
                best_segment['segment_complexity'] = self._calculate_segment_complexity(best_segment)
                best_segment['relative_position_in_segment'] = range(len(best_segment))
                segments.append(best_segment)
                
                current_start += len(best_segment)
                segment_id += 1
            else:
                break
        
        logger.info(f"Created {len(segments)} adaptive segments")
        return segments
    
    def _calculate_segment_complexity(self, segment: pd.DataFrame) -> float:
        """Calculate complexity score for a segment."""
        if len(segment) <= 1:
            return 0.0
        
        # Event type diversity
        event_diversity = segment[self.event_column].nunique() / len(segment)
        
        # Time interval variance
        timestamps = pd.to_datetime(segment[self.timestamp_column])
        time_intervals = timestamps.diff().dt.total_seconds().dropna()
        if len(time_intervals) > 0:
            interval_variance = time_intervals.var() / (time_intervals.mean() + 1e-6)
            interval_variance = min(interval_variance / 100, 1.0)  # Normalize
        else:
            interval_variance = 0.0
        
        # Combined complexity score
        complexity = (event_diversity + interval_variance) / 2
        return complexity
    
    def create_sliding_windows(self, df: pd.DataFrame, 
                             window_size: int = 50,
                             step_size: int = 25) -> List[pd.DataFrame]:
        """
        Create sliding windows over the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            window_size (int): Size of each window (number of events)
            step_size (int): Step size between windows
            
        Returns:
            List[pd.DataFrame]: List of sliding window DataFrames
        """
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        total_events = len(df_sorted)
        
        windows = []
        window_id = 0
        
        for start_idx in range(0, total_events - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = df_sorted.iloc[start_idx:end_idx].copy()
            
            # Add window metadata
            window_data['window_id'] = window_id
            window_data['window_start_idx'] = start_idx
            window_data['window_end_idx'] = end_idx - 1
            window_data['relative_position_in_window'] = range(window_size)
            
            windows.append(window_data)
            window_id += 1
        
        logger.info(f"Created {len(windows)} sliding windows")
        return windows
    
    def get_segment_statistics(self, segments: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate statistics for all segments.
        
        Args:
            segments (List[pd.DataFrame]): List of segment DataFrames
            
        Returns:
            pd.DataFrame: Statistics for each segment
        """
        stats = []
        
        for segment in segments:
            segment_stats = {
                'segment_id': segment['segment_id'].iloc[0],
                'event_count': len(segment),
                'unique_events': segment[self.event_column].nunique(),
                'duration_minutes': (pd.to_datetime(segment[self.timestamp_column].max()) - 
                                   pd.to_datetime(segment[self.timestamp_column].min())).total_seconds() / 60,
                'events_per_minute': len(segment) / max((pd.to_datetime(segment[self.timestamp_column].max()) - 
                                                        pd.to_datetime(segment[self.timestamp_column].min())).total_seconds() / 60, 1),
                'most_common_event': segment[self.event_column].mode().iloc[0] if len(segment) > 0 else None
            }
            
            # Add segment type if available
            if 'segment_gap_based' in segment.columns:
                segment_stats['segment_type'] = 'gap_based'
            elif 'segment_delimiter_based' in segment.columns:
                segment_stats['segment_type'] = 'delimiter_based'
            elif 'segment_adaptive' in segment.columns:
                segment_stats['segment_type'] = 'adaptive'
            else:
                segment_stats['segment_type'] = 'unknown'
            
            stats.append(segment_stats)
        
        return pd.DataFrame(stats)
