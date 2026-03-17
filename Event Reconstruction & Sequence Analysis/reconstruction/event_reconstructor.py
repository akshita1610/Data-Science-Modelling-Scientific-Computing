"""
Main Event Reconstructor Module

Coordinates event reconstruction using various methods and approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .rule_based_reconstructor import RuleBasedReconstructor
from .probabilistic_reconstructor import ProbabilisticReconstructor

logger = logging.getLogger(__name__)


class EventReconstructor:
    """
    Main event reconstruction coordinator.
    
    This class serves as the main interface for event reconstruction,
    coordinating between different reconstruction methods and approaches.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the EventReconstructor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for reconstruction
        """
        self.config = config.get('reconstruction', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Initialize specialized reconstructors
        self.rule_based = RuleBasedReconstructor(config)
        self.probabilistic = ProbabilisticReconstructor(config)
        
        # Reconstruction history and statistics
        self.reconstruction_history = []
        self.reconstruction_stats = {}
        
    def reconstruct_events(self, df: pd.DataFrame, 
                          method: Optional[str] = None) -> pd.DataFrame:
        """
        Reconstruct events using the specified or default method.
        
        Args:
            df (pd.DataFrame): Original event data
            method (Optional[str]): Reconstruction method ('rule_based', 'probabilistic', 'hybrid')
            
        Returns:
            pd.DataFrame: Reconstructed event data
        """
        if method is None:
            method = self.config.get('method', 'rule_based')
        
        logger.info(f"Starting event reconstruction using {method} method")
        logger.info(f"Original event count: {len(df)}")
        
        if method == 'rule_based':
            df_reconstructed = self.rule_based.reconstruct(df)
        elif method == 'probabilistic':
            df_reconstructed = self.probabilistic.reconstruct(df)
        elif method == 'hybrid':
            df_reconstructed = self._hybrid_reconstruction(df)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        # Record reconstruction statistics
        self._record_reconstruction_stats(df, df_reconstructed, method)
        
        logger.info(f"Reconstruction completed. Final event count: {len(df_reconstructed)}")
        return df_reconstructed
    
    def _hybrid_reconstruction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform hybrid reconstruction combining multiple methods.
        
        Args:
            df (pd.DataFrame): Original event data
            
        Returns:
            pd.DataFrame: Reconstructed event data
        """
        # Start with rule-based reconstruction
        df_rule = self.rule_based.reconstruct(df)
        
        # Apply probabilistic reconstruction to fill remaining gaps
        df_hybrid = self.probabilistic.reconstruct(df_rule)
        
        # Merge and resolve conflicts
        df_merged = self._merge_reconstruction_results(df, df_rule, df_hybrid)
        
        return df_merged
    
    def _merge_reconstruction_results(self, df_original: pd.DataFrame,
                                    df_rule: pd.DataFrame,
                                    df_prob: pd.DataFrame) -> pd.DataFrame:
        """
        Merge results from different reconstruction methods.
        
        Args:
            df_original (pd.DataFrame): Original data
            df_rule (pd.DataFrame): Rule-based reconstruction
            df_prob (pd.DataFrame): Probabilistic reconstruction
            
        Returns:
            pd.DataFrame: Merged reconstruction results
        """
        # Start with original events
        df_merged = df_original.copy()
        
        # Add reconstructed events from rule-based method
        rule_events = df_rule[~df_rule.index.isin(df_original.index)]
        df_merged = pd.concat([df_merged, rule_events], ignore_index=True)
        
        # Add probabilistic events that don't conflict
        prob_events = df_prob[~df_prob.index.isin(df_merged.index)]
        df_merged = pd.concat([df_merged, prob_events], ignore_index=True)
        
        # Sort by timestamp
        df_merged = df_merged.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Add reconstruction metadata
        df_merged['reconstruction_method'] = 'original'
        df_merged.loc[df_merged.index.isin(rule_events.index), 'reconstruction_method'] = 'rule_based'
        df_merged.loc[df_merged.index.isin(prob_events.index), 'reconstruction_method'] = 'probabilistic'
        
        return df_merged
    
    def detect_missing_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential missing events in the sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: DataFrame with missing event indicators
        """
        df_missing = df.copy()
        
        # Detect large time gaps (potential missing events)
        time_gaps = df_missing[self.timestamp_column].diff().dt.total_seconds()
        gap_threshold = self.config.get('rule_based', {}).get('max_gap', 300)  # 5 minutes default
        
        df_missing['potential_missing'] = (time_gaps > gap_threshold).fillna(False)
        df_missing['gap_size'] = time_gaps
        
        # Detect sequence breaks (unexpected event transitions)
        df_missing = self._detect_sequence_breaks(df_missing)
        
        # Detect pattern violations
        df_missing = self._detect_pattern_violations(df_missing)
        
        return df_missing
    
    def _detect_sequence_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect breaks in expected event sequences."""
        events = df[self.event_column].values
        
        # Common event sequences that should be continuous
        expected_sequences = [
            ['start', 'process', 'complete'],
            ['begin', 'work', 'end'],
            ['init', 'execute', 'finish']
        ]
        
        sequence_breaks = []
        
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Check if this breaks any expected sequence
            is_break = False
            for sequence in expected_sequences:
                if current_event in sequence and next_event in sequence:
                    current_idx = sequence.index(current_event)
                    next_idx = sequence.index(next_event)
                    if next_idx != current_idx + 1:
                        is_break = True
                        break
            
            sequence_breaks.append(is_break)
        
        # Add to DataFrame
        df['sequence_break'] = [False] + sequence_breaks
        
        return df
    
    def _detect_pattern_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect violations of common event patterns."""
        events = df[self.event_column].values
        
        # Check for repeated patterns that should be unique
        pattern_violations = []
        
        for i in range(len(events)):
            violation = False
            
            # Check for duplicate start events without end
            if events[i] in ['start', 'begin', 'init']:
                # Look for corresponding end event in next few events
                look_ahead = min(10, len(events) - i - 1)
                has_end = False
                for j in range(i + 1, i + look_ahead + 1):
                    if events[j] in ['end', 'complete', 'finish']:
                        has_end = True
                        break
                
                # Check if there's another start before finding end
                for j in range(i + 1, i + look_ahead + 1):
                    if events[j] in ['start', 'begin', 'init']:
                        if not has_end or j < i + look_ahead:
                            violation = True
                            break
            
            pattern_violations.append(violation)
        
        df['pattern_violation'] = pattern_violations
        
        return df
    
    def fill_missing_events(self, df: pd.DataFrame, 
                           method: str = 'rule_based') -> pd.DataFrame:
        """
        Fill detected missing events using the specified method.
        
        Args:
            df (pd.DataFrame): Event data with missing event indicators
            method (str): Method to use for filling missing events
            
        Returns:
            pd.DataFrame: Event data with filled missing events
        """
        logger.info(f"Filling missing events using {method} method")
        
        if method == 'rule_based':
            return self.rule_based.fill_missing_events(df)
        elif method == 'probabilistic':
            return self.probabilistic.fill_missing_events(df)
        else:
            raise ValueError(f"Unknown filling method: {method}")
    
    def correct_sequence_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct sequence errors and inconsistencies.
        
        Args:
            df (pd.DataFrame): Event data with potential errors
            
        Returns:
            pd.DataFrame: Corrected event data
        """
        df_corrected = df.copy()
        
        # Correct timestamp ordering
        df_corrected = self._correct_timestamp_order(df_corrected)
        
        # Correct event sequence violations
        df_corrected = self._correct_event_sequence(df_corrected)
        
        # Remove duplicate events
        df_corrected = self._remove_duplicate_events(df_corrected)
        
        logger.info(f"Sequence correction completed. Events: {len(df)} -> {len(df_corrected)}")
        return df_corrected
    
    def _correct_timestamp_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct timestamp ordering issues."""
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Detect and fix out-of-order timestamps within same second
        timestamps = pd.to_datetime(df_sorted[self.timestamp_column])
        
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                # Add small increment to make it later
                timestamps[i] = timestamps[i-1] + pd.Timedelta(milliseconds=1)
        
        df_sorted[self.timestamp_column] = timestamps
        
        return df_sorted
    
    def _correct_event_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct event sequence violations."""
        events = df[self.event_column].values
        
        # Simple correction: add missing intermediate events
        corrected_events = []
        
        for i in range(len(events)):
            corrected_events.append(events[i])
            
            # Check if we need to insert missing events
            if i < len(events) - 1:
                current_event = events[i]
                next_event = events[i + 1]
                
                # Common missing event patterns
                if current_event == 'start' and next_event == 'complete':
                    corrected_events.append('process')
                elif current_event == 'begin' and next_event == 'end':
                    corrected_events.append('work')
                elif current_event == 'init' and next_event == 'finish':
                    corrected_events.append('execute')
        
        # Create corrected DataFrame
        if len(corrected_events) > len(events):
            # Add new events with interpolated timestamps
            df_corrected = df.copy()
            original_timestamps = pd.to_datetime(df[self.timestamp_column])
            
            new_rows = []
            event_idx = 0
            corrected_idx = 0
            
            while event_idx < len(events) and corrected_idx < len(corrected_events):
                if corrected_events[corrected_idx] == events[event_idx]:
                    # Original event
                    corrected_idx += 1
                    event_idx += 1
                else:
                    # Inserted event
                    if event_idx > 0:
                        prev_timestamp = original_timestamps.iloc[event_idx - 1]
                        next_timestamp = original_timestamps.iloc[event_idx]
                        new_timestamp = prev_timestamp + (next_timestamp - prev_timestamp) / 2
                    else:
                        new_timestamp = original_timestamps.iloc[0]
                    
                    new_row = {
                        self.timestamp_column: new_timestamp,
                        self.event_column: corrected_events[corrected_idx],
                        'reconstruction_method': 'sequence_correction'
                    }
                    new_rows.append(new_row)
                    corrected_idx += 1
            
            # Add new rows to DataFrame
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_corrected = pd.concat([df_corrected, df_new], ignore_index=True)
                df_corrected = df_corrected.sort_values(self.timestamp_column).reset_index(drop=True)
            
            return df_corrected
        else:
            return df
    
    def _remove_duplicate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate events with same timestamp and type."""
        # Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=[self.timestamp_column, self.event_column])
        
        # Handle events with same timestamp but different types
        timestamp_counts = df_dedup[self.timestamp_column].value_counts()
        duplicate_timestamps = timestamp_counts[timestamp_counts > 1].index
        
        for timestamp in duplicate_timestamps:
            events_at_timestamp = df_dedup[df_dedup[self.timestamp_column] == timestamp]
            if len(events_at_timestamp) > 1:
                # Keep the first event, add small time offsets to others
                for i, idx in enumerate(events_at_timestamp.index[1:]):
                    df_dedup.loc[idx, self.timestamp_column] = (
                        pd.to_datetime(timestamp) + pd.Timedelta(milliseconds=i+1)
                    )
        
        return df_dedup.sort_values(self.timestamp_column).reset_index(drop=True)
    
    def _record_reconstruction_stats(self, df_original: pd.DataFrame,
                                  df_reconstructed: pd.DataFrame,
                                  method: str):
        """Record reconstruction statistics."""
        stats = {
            'method': method,
            'original_events': len(df_original),
            'reconstructed_events': len(df_reconstructed),
            'events_added': len(df_reconstructed) - len(df_original),
            'reconstruction_ratio': len(df_reconstructed) / len(df_original),
            'timestamp': pd.Timestamp.now()
        }
        
        self.reconstruction_stats = stats
        self.reconstruction_history.append(stats)
    
    def get_reconstruction_summary(self) -> Dict:
        """
        Get summary of reconstruction operations.
        
        Returns:
            Dict: Summary of reconstruction statistics
        """
        if not self.reconstruction_history:
            return {'message': 'No reconstruction performed yet'}
        
        summary = {
            'total_reconstructions': len(self.reconstruction_history),
            'latest_stats': self.reconstruction_stats,
            'methods_used': list(set(stats['method'] for stats in self.reconstruction_history)),
            'avg_events_added': np.mean([stats['events_added'] for stats in self.reconstruction_history]),
            'avg_reconstruction_ratio': np.mean([stats['reconstruction_ratio'] for stats in self.reconstruction_history])
        }
        
        return summary
    
    def validate_reconstruction(self, df_original: pd.DataFrame,
                             df_reconstructed: pd.DataFrame) -> Dict:
        """
        Validate reconstruction quality.
        
        Args:
            df_original (pd.DataFrame): Original event data
            df_reconstructed (pd.DataFrame): Reconstructed event data
            
        Returns:
            Dict: Validation metrics
        """
        validation = {
            'original_events': len(df_original),
            'reconstructed_events': len(df_reconstructed),
            'events_preserved': len(df_original[df_original[self.timestamp_column].isin(df_reconstructed[self.timestamp_column])]),
            'new_events': len(df_reconstructed) - len(df_original),
            'time_range_preserved': (
                df_original[self.timestamp_column].min() == df_reconstructed[self.timestamp_column].min() and
                df_original[self.timestamp_column].max() == df_reconstructed[self.timestamp_column].max()
            ),
            'event_types_preserved': set(df_original[self.event_column].unique()).issubset(
                set(df_reconstructed[self.event_column].unique())
            )
        }
        
        # Calculate preservation rate
        validation['preservation_rate'] = validation['events_preserved'] / validation['original_events']
        
        # Check for logical consistency
        validation['is_logically_consistent'] = self._check_logical_consistency(df_reconstructed)
        
        return validation
    
    def _check_logical_consistency(self, df: pd.DataFrame) -> bool:
        """Check if the reconstructed sequence is logically consistent."""
        events = df[self.event_column].values
        
        # Basic consistency checks
        consistent = True
        
        # Check for impossible sequences
        impossible_sequences = [
            ('end', 'start'),
            ('complete', 'begin'),
            ('finish', 'init')
        ]
        
        for i in range(len(events) - 1):
            for impossible_pair in impossible_sequences:
                if events[i] == impossible_pair[0] and events[i+1] == impossible_pair[1]:
                    consistent = False
                    break
        
        return consistent
