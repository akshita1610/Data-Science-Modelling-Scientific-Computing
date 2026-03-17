"""
Pattern Detection Module

Detects and analyzes patterns in event sequences using various algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from collections import Counter, defaultdict
from itertools import combinations, permutations

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects patterns in event sequences.
    
    This class implements various pattern detection algorithms including
    frequent pattern mining, sequential pattern mining, and motif detection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PatternDetector with configuration.
        
        Args:
            config (Dict): Configuration dictionary for pattern detection
        """
        self.config = config.get('analysis', {}).get('pattern_detection', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Pattern detection parameters
        self.min_pattern_length = self.config.get('min_pattern_length', 3)
        self.max_pattern_length = self.config.get('max_pattern_length', 10)
        self.min_support = self.config.get('min_support', 0.1)
        
        # Detected patterns storage
        self.detected_patterns = {}
        
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect all types of patterns in the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            Dict: Dictionary containing all detected patterns
        """
        logger.info("Starting pattern detection")
        
        events = df[self.event_column].values
        
        patterns = {
            'frequent_patterns': self._detect_frequent_patterns(events),
            'sequential_patterns': self._detect_sequential_patterns(events),
            'motifs': self._detect_motifs(events),
            'periodic_patterns': self._detect_periodic_patterns(df),
            'conditional_patterns': self._detect_conditional_patterns(events),
            'temporal_patterns': self._detect_temporal_patterns(df)
        }
        
        self.detected_patterns = patterns
        logger.info(f"Pattern detection completed. Found {sum(len(v) for v in patterns.values() if isinstance(v, list))} patterns")
        
        return patterns
    
    def _detect_frequent_patterns(self, events: List) -> List[Dict]:
        """Detect frequent patterns using sliding window approach."""
        patterns = []
        total_events = len(events)
        
        # Generate patterns of different lengths
        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, len(events))):
            pattern_counts = Counter()
            
            # Slide window through sequence
            for i in range(len(events) - length + 1):
                pattern = tuple(events[i:i+length])
                pattern_counts[pattern] += 1
            
            # Filter by minimum support
            min_count = int(self.min_support * (len(events) - length + 1))
            frequent_patterns = [(pattern, count) for pattern, count in pattern_counts.items() if count >= min_count]
            
            # Add to results
            for pattern, count in frequent_patterns:
                support = count / (len(events) - length + 1)
                patterns.append({
                    'pattern': pattern,
                    'length': length,
                    'count': count,
                    'support': support,
                    'type': 'frequent'
                })
        
        # Sort by support
        patterns.sort(key=lambda x: x['support'], reverse=True)
        return patterns
    
    def _detect_sequential_patterns(self, events: List) -> List[Dict]:
        """Detect sequential patterns with gaps allowed."""
        patterns = []
        
        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, len(events))):
            # Find patterns with gaps
            pattern_counts = defaultdict(int)
            
            def find_patterns_with_gaps(start_idx: int, current_pattern: List, remaining_length: int):
                if remaining_length == 0:
                    pattern_counts[tuple(current_pattern)] += 1
                    return
                
                if start_idx >= len(events):
                    return
                
                # Try each possible next position
                for next_idx in range(start_idx + 1, len(events)):
                    # Limit gap size to avoid too many patterns
                    if next_idx - start_idx > 5:
                        break
                    
                    current_pattern.append(events[next_idx])
                    find_patterns_with_gaps(next_idx, current_pattern.copy(), remaining_length - 1)
            
            # Start pattern detection from each position
            for i in range(len(events)):
                find_patterns_with_gaps(i, [events[i]], length - 1)
            
            # Filter and add patterns
            min_count = int(self.min_support * len(events))
            for pattern, count in pattern_counts.items():
                if count >= min_count:
                    patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'count': count,
                        'support': count / len(events),
                        'type': 'sequential_with_gaps'
                    })
        
        patterns.sort(key=lambda x: x['support'], reverse=True)
        return patterns
    
    def _detect_motifs(self, events: List) -> List[Dict]:
        """Detect recurring motifs (short patterns) in the sequence."""
        motifs = []
        motif_length = 3  # Short patterns for motifs
        
        if len(events) < motif_length:
            return motifs
        
        # Find all possible motifs of specified length
        motif_counts = Counter()
        
        for i in range(len(events) - motif_length + 1):
            motif = tuple(events[i:i+motif_length])
            motif_counts[motif] += 1
        
        # Filter motifs that appear multiple times
        min_occurrences = max(2, int(0.02 * len(events)))  # At least 2% or 2 occurrences
        
        for motif, count in motif_counts.items():
            if count >= min_occurrences:
                # Calculate motif positions
                positions = []
                for i in range(len(events) - motif_length + 1):
                    if tuple(events[i:i+motif_length]) == motif:
                        positions.append(i)
                
                # Calculate spacing between occurrences
                spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_spacing = np.mean(spacings) if spacings else 0
                
                motifs.append({
                    'pattern': motif,
                    'length': motif_length,
                    'count': count,
                    'positions': positions,
                    'avg_spacing': avg_spacing,
                    'spacing_std': np.std(spacings) if spacings else 0,
                    'type': 'motif'
                })
        
        # Sort by count
        motifs.sort(key=lambda x: x['count'], reverse=True)
        return motifs
    
    def _detect_periodic_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect periodic patterns in time series."""
        periodic_patterns = []
        events = df[self.event_column].values
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        # Group events by type
        event_types = set(events)
        
        for event_type in event_types:
            # Get timestamps for this event type
            event_timestamps = timestamps[events == event_type]
            
            if len(event_timestamps) < 3:
                continue
            
            # Calculate intervals between occurrences
            intervals = event_timestamps.diff().dt.total_seconds().dropna()
            
            if len(intervals) < 2:
                continue
            
            # Find dominant period using autocorrelation
            intervals_array = intervals.values
            
            # Try different periods
            for period_multiplier in range(2, min(10, len(intervals_array))):
                # Calculate autocorrelation at this lag
                if len(intervals_array) > period_multiplier:
                    correlation = np.corrcoef(
                        intervals_array[:-period_multiplier], 
                        intervals_array[period_multiplier:]
                    )[0, 1]
                    
                    if not np.isnan(correlation) and correlation > 0.7:  # High correlation threshold
                        period_seconds = intervals_array[:period_multiplier].mean()
                        
                        periodic_patterns.append({
                            'pattern': (event_type,),
                            'event_type': event_type,
                            'period_seconds': period_seconds,
                            'period_multiplier': period_multiplier,
                            'correlation': correlation,
                            'count': len(event_timestamps),
                            'type': 'periodic'
                        })
        
        # Sort by correlation
        periodic_patterns.sort(key=lambda x: x['correlation'], reverse=True)
        return periodic_patterns
    
    def _detect_conditional_patterns(self, events: List) -> List[Dict]:
        """Detect conditional patterns (if-then relationships)."""
        conditional_patterns = []
        
        # Find patterns where certain events predict others
        event_types = list(set(events))
        
        for condition_event in event_types:
            for consequence_event in event_types:
                if condition_event == consequence_event:
                    continue
                
                # Find all occurrences of condition event
                condition_positions = [i for i, event in enumerate(events) if event == condition_event]
                
                if len(condition_positions) < 2:
                    continue
                
                # Check what follows each condition
                consequence_counts = defaultdict(int)
                look_ahead = 5  # Look ahead up to 5 events
                
                for pos in condition_positions:
                    for look_pos in range(pos + 1, min(pos + look_ahead + 1, len(events))):
                        if events[look_pos] == consequence_event:
                            consequence_counts[look_pos - pos] += 1
                            break
                
                if consequence_counts:
                    # Calculate conditional probability
                    total_conditions = len(condition_positions)
                    best_distance = max(consequence_counts, key=consequence_counts.get)
                    best_count = consequence_counts[best_distance]
                    conditional_prob = best_count / total_conditions
                    
                    if conditional_prob > 0.3:  # Minimum conditional probability
                        conditional_patterns.append({
                            'condition': condition_event,
                            'consequence': consequence_event,
                            'conditional_probability': conditional_prob,
                            'best_distance': best_distance,
                            'count': best_count,
                            'total_conditions': total_conditions,
                            'type': 'conditional'
                        })
        
        # Sort by conditional probability
        conditional_patterns.sort(key=lambda x: x['conditional_probability'], reverse=True)
        return conditional_patterns
    
    def _detect_temporal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect patterns related to time of day, day of week, etc."""
        temporal_patterns = []
        timestamps = pd.to_datetime(df[self.timestamp_column])
        events = df[self.event_column].values
        
        # Extract temporal features
        hours = timestamps.dt.hour
        days_of_week = timestamps.dt.dayofweek
        months = timestamps.dt.month
        
        # Time of day patterns
        for hour in range(24):
            hour_events = events[hours == hour]
            if len(hour_events) >= 5:  # Minimum events for pattern
                event_counts = Counter(hour_events)
                most_common = event_counts.most_common(1)[0]
                
                temporal_patterns.append({
                    'pattern_type': 'hourly',
                    'time_value': hour,
                    'dominant_event': most_common[0],
                    'count': most_common[1],
                    'total_events': len(hour_events),
                    'dominance_ratio': most_common[1] / len(hour_events),
                    'type': 'temporal'
                })
        
        # Day of week patterns
        for day in range(7):
            day_events = events[days_of_week == day]
            if len(day_events) >= 5:
                event_counts = Counter(day_events)
                most_common = event_counts.most_common(1)[0]
                
                temporal_patterns.append({
                    'pattern_type': 'daily',
                    'time_value': day,
                    'dominant_event': most_common[0],
                    'count': most_common[1],
                    'total_events': len(day_events),
                    'dominance_ratio': most_common[1] / len(day_events),
                    'type': 'temporal'
                })
        
        # Sort by dominance ratio
        temporal_patterns.sort(key=lambda x: x['dominance_ratio'], reverse=True)
        return temporal_patterns
    
    def find_pattern_instances(self, df: pd.DataFrame, 
                             pattern: Tuple) -> List[Dict]:
        """
        Find all instances of a specific pattern in the data.
        
        Args:
            df (pd.DataFrame): Event data
            pattern (Tuple): Pattern to search for
            
        Returns:
            List[Dict]: List of pattern instances with metadata
        """
        events = df[self.event_column].values
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        instances = []
        pattern_length = len(pattern)
        
        for i in range(len(events) - pattern_length + 1):
            if tuple(events[i:i+pattern_length]) == pattern:
                instance = {
                    'start_index': i,
                    'end_index': i + pattern_length - 1,
                    'start_time': timestamps[i],
                    'end_time': timestamps[i + pattern_length - 1],
                    'duration': (timestamps[i + pattern_length - 1] - timestamps[i]).total_seconds()
                }
                instances.append(instance)
        
        return instances
    
    def calculate_pattern_statistics(self, patterns: List[Dict]) -> Dict:
        """
        Calculate statistics for detected patterns.
        
        Args:
            patterns (List[Dict]): List of detected patterns
            
        Returns:
            Dict: Pattern statistics
        """
        if not patterns:
            return {'total_patterns': 0}
        
        stats = {
            'total_patterns': len(patterns),
            'pattern_types': Counter(p['type'] for p in patterns),
            'avg_pattern_length': np.mean([p['length'] for p in patterns if 'length' in p]),
            'avg_support': np.mean([p['support'] for p in patterns if 'support' in p]),
            'max_support': max([p['support'] for p in patterns if 'support' in p], 0),
            'pattern_diversity': len(set(p['pattern'] for p in patterns if 'pattern' in p))
        }
        
        return stats
    
    def filter_patterns(self, patterns: List[Dict], 
                       min_support: Optional[float] = None,
                       max_length: Optional[int] = None,
                       pattern_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter patterns based on various criteria.
        
        Args:
            patterns (List[Dict]): List of patterns to filter
            min_support (Optional[float]): Minimum support threshold
            max_length (Optional[int]): Maximum pattern length
            pattern_types (Optional[List[str]]): Pattern types to include
            
        Returns:
            List[Dict]: Filtered patterns
        """
        filtered = patterns.copy()
        
        if min_support is not None:
            filtered = [p for p in filtered if p.get('support', 0) >= min_support]
        
        if max_length is not None:
            filtered = [p for p in filtered if p.get('length', 0) <= max_length]
        
        if pattern_types is not None:
            filtered = [p for p in filtered if p.get('type') in pattern_types]
        
        return filtered
    
    def visualize_patterns(self, patterns: List[Dict], 
                          top_k: int = 10) -> Dict:
        """
        Prepare pattern data for visualization.
        
        Args:
            patterns (List[Dict]): List of patterns
            top_k (int): Number of top patterns to include
            
        Returns:
            Dict: Visualization-ready data
        """
        # Sort patterns by support/count
        sorted_patterns = sorted(patterns, 
                                key=lambda x: x.get('support', x.get('count', 0)), 
                                reverse=True)[:top_k]
        
        viz_data = {
            'pattern_names': [' -> '.join(p['pattern']) if 'pattern' in p else str(p.get('event_type', 'unknown')) 
                             for p in sorted_patterns],
            'supports': [p.get('support', p.get('count', 0)) for p in sorted_patterns],
            'pattern_types': [p.get('type', 'unknown') for p in sorted_patterns],
            'pattern_lengths': [p.get('length', 1) for p in sorted_patterns]
        }
        
        return viz_data
    
    def compare_patterns(self, patterns1: List[Dict], 
                        patterns2: List[Dict]) -> Dict:
        """
        Compare two sets of patterns.
        
        Args:
            patterns1 (List[Dict]): First set of patterns
            patterns2 (List[Dict]): Second set of patterns
            
        Returns:
            Dict: Comparison results
        """
        set1_patterns = set(str(p['pattern']) for p in patterns1 if 'pattern' in p)
        set2_patterns = set(str(p['pattern']) for p in patterns2 if 'pattern' in p)
        
        comparison = {
            'unique_to_set1': list(set1_patterns - set2_patterns),
            'unique_to_set2': list(set2_patterns - set1_patterns),
            'common_patterns': list(set1_patterns & set2_patterns),
            'jaccard_similarity': len(set1_patterns & set2_patterns) / len(set1_patterns | set2_patterns) if set1_patterns | set2_patterns else 0,
            'set1_stats': self.calculate_pattern_statistics(patterns1),
            'set2_stats': self.calculate_pattern_statistics(patterns2)
        }
        
        return comparison
