"""
Rule-based Event Reconstructor Module

Implements rule-based reconstruction using domain knowledge and heuristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RuleBasedReconstructor:
    """
    Rule-based event reconstructor.
    
    Uses predefined rules and domain knowledge to reconstruct missing events
    and correct sequence errors.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the RuleBasedReconstructor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for reconstruction
        """
        self.config = config.get('reconstruction', {})
        self.rule_config = self.config.get('rule_based', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Define reconstruction rules
        self.reconstruction_rules = self._define_reconstruction_rules()
        
    def reconstruct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform rule-based event reconstruction.
        
        Args:
            df (pd.DataFrame): Original event data
            
        Returns:
            pd.DataFrame: Reconstructed event data
        """
        logger.info("Starting rule-based reconstruction")
        
        df_reconstructed = df.copy()
        
        # Step 1: Detect missing events
        df_reconstructed = self._detect_missing_events(df_reconstructed)
        
        # Step 2: Fill missing events based on rules
        df_reconstructed = self._fill_missing_events(df_reconstructed)
        
        # Step 3: Correct sequence errors
        df_reconstructed = self._correct_sequence_errors(df_reconstructed)
        
        # Step 4: Validate and clean
        df_reconstructed = self._validate_reconstruction(df_reconstructed)
        
        logger.info(f"Rule-based reconstruction completed: {len(df)} -> {len(df_reconstructed)} events")
        return df_reconstructed
    
    def _define_reconstruction_rules(self) -> Dict:
        """Define reconstruction rules based on domain knowledge."""
        rules = {
            'event_sequences': {
                # Common event sequences that should be complete
                'start_process_complete': ['start', 'process', 'complete'],
                'begin_work_end': ['begin', 'work', 'end'],
                'init_execute_finish': ['init', 'execute', 'finish'],
                'open_process_close': ['open', 'process', 'close'],
                'create_edit_save': ['create', 'edit', 'save'],
                'login_activity_logout': ['login', 'activity', 'logout']
            },
            'mandatory_pairs': {
                # Events that should come in pairs
                'start_end': ('start', 'end'),
                'begin_finish': ('begin', 'finish'),
                'open_close': ('open', 'close'),
                'login_logout': ('login', 'logout'),
                'create_delete': ('create', 'delete')
            },
            'time_constraints': {
                # Maximum time gaps between related events (in seconds)
                'start_to_process': 60,
                'process_to_complete': 300,
                'login_to_activity': 30,
                'activity_to_logout': 3600,
                'default_max_gap': self.rule_config.get('max_gap', 300)
            },
            'frequency_rules': {
                # Expected frequency patterns
                'heartbeat_interval': 30,  # seconds
                'status_update_interval': 60,  # seconds
                'error_retry_delay': 5  # seconds
            }
        }
        
        return rules
    
    def _detect_missing_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect missing events based on rules."""
        df_missing = df.copy()
        
        # Detect missing events in sequences
        df_missing = self._detect_missing_sequence_events(df_missing)
        
        # Detect missing paired events
        df_missing = self._detect_missing_paired_events(df_missing)
        
        # Detect missing periodic events
        df_missing = self._detect_missing_periodic_events(df_missing)
        
        return df_missing
    
    def _detect_missing_sequence_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect missing events in expected sequences."""
        events = df[self.event_column].values
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        missing_indicators = []
        
        for i in range(len(events)):
            missing_events = []
            
            # Check each defined sequence
            for sequence_name, sequence_events in self.reconstruction_rules['event_sequences'].items():
                if events[i] == sequence_events[0]:
                    # Look for complete sequence starting here
                    sequence_found = self._find_sequence_at_position(
                        events, timestamps, i, sequence_events
                    )
                    if sequence_found['missing_events']:
                        missing_events.extend(sequence_found['missing_events'])
            
            missing_indicators.append(missing_events)
        
        df['missing_sequence_events'] = missing_indicators
        return df
    
    def _find_sequence_at_position(self, events: List, timestamps: List,
                                 start_idx: int, sequence: List) -> Dict:
        """Find if a sequence exists starting at a given position."""
        result = {
            'found': False,
            'missing_events': [],
            'expected_timestamps': []
        }
        
        if start_idx + len(sequence) > len(events):
            return result
        
        # Check if sequence exists
        current_events = events[start_idx:start_idx + len(sequence)]
        if current_events == sequence:
            result['found'] = True
            return result
        
        # Find missing events
        missing_events = []
        expected_timestamps = []
        event_idx = start_idx
        seq_idx = 0
        
        while event_idx < len(events) and seq_idx < len(sequence):
            if events[event_idx] == sequence[seq_idx]:
                # Found expected event
                expected_timestamps.append(timestamps[event_idx])
                event_idx += 1
                seq_idx += 1
            else:
                # Missing event
                missing_events.append(sequence[seq_idx])
                
                # Estimate timestamp for missing event
                if seq_idx > 0 and len(expected_timestamps) > 0:
                    prev_timestamp = expected_timestamps[-1]
                    next_timestamp = timestamps[event_idx] if event_idx < len(timestamps) else prev_timestamp + timedelta(minutes=1)
                    estimated_timestamp = prev_timestamp + (next_timestamp - prev_timestamp) / 2
                else:
                    estimated_timestamp = timestamps[event_idx] if event_idx < len(timestamps) else timestamps[start_idx]
                
                expected_timestamps.append(estimated_timestamp)
                seq_idx += 1
        
        # Check for remaining missing events at the end
        while seq_idx < len(sequence):
            missing_events.append(sequence[seq_idx])
            if expected_timestamps:
                estimated_timestamp = expected_timestamps[-1] + timedelta(minutes=1)
            else:
                estimated_timestamp = timestamps[start_idx]
            expected_timestamps.append(estimated_timestamp)
            seq_idx += 1
        
        result['missing_events'] = missing_events
        result['expected_timestamps'] = expected_timestamps
        
        return result
    
    def _detect_missing_paired_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect missing paired events."""
        events = df[self.event_column].values
        
        missing_pairs = []
        
        # Track unmatched events
        unmatched = {}
        
        for pair_name, (start_event, end_event) in self.reconstruction_rules['mandatory_pairs'].items():
            # Find unmatched start events
            start_indices = [i for i, event in enumerate(events) if event == start_event]
            end_indices = [i for i, event in enumerate(events) if event == end_event]
            
            # Simple matching (could be improved with time constraints)
            unmatched_starts = len(start_indices) - min(len(start_indices), len(end_indices))
            unmatched_ends = len(end_indices) - min(len(start_indices), len(end_indices))
            
            missing_pairs.append({
                'pair_type': pair_name,
                'unmatched_starts': unmatched_starts,
                'unmatched_ends': unmatched_ends
            })
        
        df['missing_pair_events'] = missing_pairs
        return df
    
    def _detect_missing_periodic_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect missing periodic events like heartbeats."""
        events = df[self.event_column].values
        timestamps = pd.to_datetime(df[self.timestamp_column])
        
        missing_periodic = []
        
        # Check for missing heartbeat events
        heartbeat_interval = self.reconstruction_rules['frequency_rules']['heartbeat_interval']
        heartbeat_events = [i for i, event in enumerate(events) if event == 'heartbeat']
        
        if len(heartbeat_events) > 1:
            # Check time gaps between heartbeats
            for i in range(len(heartbeat_events) - 1):
                current_idx = heartbeat_events[i]
                next_idx = heartbeat_events[i + 1]
                
                time_gap = (timestamps[next_idx] - timestamps[current_idx]).total_seconds()
                expected_count = int(time_gap / heartbeat_interval)
                
                if expected_count > 1:
                    missing_count = expected_count - 1
                    missing_periodic.append({
                        'event_type': 'heartbeat',
                        'start_index': current_idx,
                        'end_index': next_idx,
                        'missing_count': missing_count
                    })
        
        df['missing_periodic_events'] = missing_periodic
        return df
    
    def _fill_missing_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill detected missing events."""
        df_filled = df.copy()
        
        # Fill missing sequence events
        df_filled = self._fill_missing_sequence_events(df_filled)
        
        # Fill missing paired events
        df_filled = self._fill_missing_paired_events(df_filled)
        
        # Fill missing periodic events
        df_filled = self._fill_missing_periodic_events(df_filled)
        
        return df_filled
    
    def _fill_missing_sequence_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing sequence events."""
        new_events = []
        
        for idx, row in df.iterrows():
            missing_events = row.get('missing_sequence_events', [])
            expected_timestamps = row.get('expected_timestamps', [])
            
            if missing_events is not None and len(missing_events) > 0 and expected_timestamps is not None and len(expected_timestamps) > 0:
                for missing_event, timestamp in zip(missing_events, expected_timestamps):
                    new_event = {
                        self.timestamp_column: timestamp,
                        self.event_column: missing_event,
                        'reconstruction_method': 'rule_based_sequence',
                        'reconstruction_confidence': 0.8
                    }
                    new_events.append(new_event)
        
        if new_events:
            df_new = pd.DataFrame(new_events)
            df_filled = pd.concat([df, df_new], ignore_index=True)
            df_filled = df_filled.sort_values(self.timestamp_column).reset_index(drop=True)
            return df_filled
        
        return df
    
    def _fill_missing_paired_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing paired events."""
        new_events = []
        
        missing_pair_events = df.get('missing_pair_events', [])
        if isinstance(missing_pair_events, list) and len(missing_pair_events) > 0:
            # Find the positions where missing pairs should be added
            for pair_info in missing_pair_events:
                if pair_info['unmatched_starts'] > 0:
                    # Add missing end events
                    start_events = df[df[self.event_column] == pair_info['pair_type'].split('_')[0]]
                    if len(start_events) > 0:
                        last_start = start_events.iloc[-1]
                        new_timestamp = pd.to_datetime(last_start[self.timestamp_column]) + timedelta(minutes=30)
                        
                        new_event = {
                            self.timestamp_column: new_timestamp,
                            self.event_column: pair_info['pair_type'].split('_')[1],
                            'reconstruction_method': 'rule_based_pair',
                            'reconstruction_confidence': 0.7
                        }
                        new_events.append(new_event)
        
        if new_events:
            df_new = pd.DataFrame(new_events)
            df_filled = pd.concat([df, df_new], ignore_index=True)
            df_filled = df_filled.sort_values(self.timestamp_column).reset_index(drop=True)
            return df_filled
        
        return df
    
    def _fill_missing_periodic_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing periodic events."""
        new_events = []
        
        missing_periodic = df.get('missing_periodic_events', [])
        if isinstance(missing_periodic, list) and len(missing_periodic) > 0:
            for missing_info in missing_periodic:
                event_type = missing_info['event_type']
                start_idx = missing_info['start_index']
                end_idx = missing_info['end_index']
                missing_count = missing_info['missing_count']
                
                # Get timestamps for interpolation
                start_timestamp = pd.to_datetime(df.iloc[start_idx][self.timestamp_column])
                end_timestamp = pd.to_datetime(df.iloc[end_idx][self.timestamp_column])
                
                # Add missing events at regular intervals
                interval = (end_timestamp - start_timestamp) / (missing_count + 1)
                
                for i in range(1, missing_count + 1):
                    new_timestamp = start_timestamp + interval * i
                    
                    new_event = {
                        self.timestamp_column: new_timestamp,
                        self.event_column: event_type,
                        'reconstruction_method': 'rule_based_periodic',
                        'reconstruction_confidence': 0.6
                    }
                    new_events.append(new_event)
        
        if new_events:
            df_new = pd.DataFrame(new_events)
            df_filled = pd.concat([df, df_new], ignore_index=True)
            df_filled = df_filled.sort_values(self.timestamp_column).reset_index(drop=True)
            return df_filled
        
        return df
    
    def _correct_sequence_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct sequence errors using rules."""
        df_corrected = df.copy()
        
        # Correct out-of-order events
        df_corrected = self._correct_event_order(df_corrected)
        
        # Correct duplicate events
        df_corrected = self._correct_duplicate_events(df_corrected)
        
        # Correct impossible transitions
        df_corrected = self._correct_impossible_transitions(df_corrected)
        
        return df_corrected
    
    def _correct_event_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct event order based on logical constraints."""
        # Sort by timestamp first
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Apply additional ordering rules
        events = df_sorted[self.event_column].values
        
        # Look for events that should be reordered
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Check for impossible ordering
            if (current_event in ['end', 'complete', 'finish'] and 
                next_event in ['start', 'begin', 'init']):
                # Swap these events
                df_sorted.iloc[i], df_sorted.iloc[i + 1] = df_sorted.iloc[i + 1].copy(), df_sorted.iloc[i].copy()
        
        return df_sorted
    
    def _correct_duplicate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct duplicate events."""
        # Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=[self.timestamp_column, self.event_column])
        
        # Handle events with same timestamp but different types
        timestamp_counts = df_dedup[self.timestamp_column].value_counts()
        duplicate_timestamps = timestamp_counts[timestamp_counts > 1].index
        
        for timestamp in duplicate_timestamps:
            events_at_timestamp = df_dedup[df_dedup[self.timestamp_column] == timestamp]
            if len(events_at_timestamp) > 1:
                # Sort by event priority and keep the most important
                event_priority = {
                    'error': 1, 'start': 2, 'begin': 2, 'init': 2,
                    'process': 3, 'work': 3, 'execute': 3,
                    'complete': 4, 'end': 4, 'finish': 4
                }
                
                events_at_timestamp['priority'] = events_at_timestamp[self.event_column].map(event_priority).fillna(5)
                events_at_timestamp = events_at_timestamp.sort_values('priority')
                
                # Keep the highest priority event, add time offsets to others
                keep_event = events_at_timestamp.iloc[0]
                other_events = events_at_timestamp.iloc[1:]
                
                for i, (_, event_row) in enumerate(other_events.iterrows()):
                    new_timestamp = pd.to_datetime(timestamp) + pd.Timedelta(milliseconds=i+1)
                    df_dedup.loc[event_row.name, self.timestamp_column] = new_timestamp
        
        return df_dedup.sort_values(self.timestamp_column).reset_index(drop=True)
    
    def _correct_impossible_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct impossible event transitions."""
        events = df[self.event_column].values
        
        # Define impossible transitions
        impossible_transitions = [
            ('end', 'start'),
            ('complete', 'begin'),
            ('finish', 'init'),
            ('close', 'open'),
            ('logout', 'login')
        ]
        
        corrected_events = []
        
        for i in range(len(events)):
            corrected_events.append(events[i])
            
            if i < len(events) - 1:
                current_event = events[i]
                next_event = events[i + 1]
                
                # Check if transition is impossible
                for impossible_pair in impossible_transitions:
                    if current_event == impossible_pair[0] and next_event == impossible_pair[1]:
                        # Insert intermediate event
                        intermediate_events = {
                            ('end', 'start'): 'reset',
                            ('complete', 'begin'): 'prepare',
                            ('finish', 'init'): 'cleanup',
                            ('close', 'open'): 'setup',
                            ('logout', 'login'): 'session_end'
                        }
                        
                        if impossible_pair in intermediate_events:
                            corrected_events.append(intermediate_events[impossible_pair])
        
        # Update DataFrame if events were added
        if len(corrected_events) > len(events):
            # Create new rows for added events
            new_rows = []
            event_idx = 0
            corrected_idx = 0
            
            while event_idx < len(events) and corrected_idx < len(corrected_events):
                if corrected_events[corrected_idx] == events[event_idx]:
                    corrected_idx += 1
                    event_idx += 1
                else:
                    # Inserted event
                    if event_idx > 0:
                        prev_timestamp = pd.to_datetime(df.iloc[event_idx - 1][self.timestamp_column])
                        next_timestamp = pd.to_datetime(df.iloc[event_idx][self.timestamp_column])
                        new_timestamp = prev_timestamp + (next_timestamp - prev_timestamp) / 2
                    else:
                        new_timestamp = pd.to_datetime(df.iloc[event_idx][self.timestamp_column])
                    
                    new_row = {
                        self.timestamp_column: new_timestamp,
                        self.event_column: corrected_events[corrected_idx],
                        'reconstruction_method': 'rule_based_transition_correction',
                        'reconstruction_confidence': 0.7
                    }
                    new_rows.append(new_row)
                    corrected_idx += 1
            
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_corrected = pd.concat([df, df_new], ignore_index=True)
                df_corrected = df_corrected.sort_values(self.timestamp_column).reset_index(drop=True)
                return df_corrected
        
        return df
    
    def _validate_reconstruction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate reconstruction quality."""
        # Check for logical consistency
        events = df[self.event_column].values
        
        # Remove any events that violate basic rules
        valid_indices = []
        
        for i, event in enumerate(events):
            # Basic validation: event should not be None or empty
            if event and str(event).strip():
                valid_indices.append(i)
        
        df_valid = df.iloc[valid_indices].copy()
        df_valid = df_valid.sort_values(self.timestamp_column).reset_index(drop=True)
        
        return df_valid
    
    def fill_missing_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to fill missing events.
        
        Args:
            df (pd.DataFrame): Event data with missing indicators
            
        Returns:
            pd.DataFrame: Event data with filled missing events
        """
        return self._fill_missing_events(df)
