"""
Sequence-based Feature Extraction Module

Extracts features related to event sequences, patterns, and transitions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from collections import Counter, defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)


class SequenceFeatureExtractor:
    """
    Extracts sequence-based features from event sequences.
    
    This class specializes in extracting features related to event patterns,
    transitions, sequences, and structural characteristics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SequenceFeatureExtractor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for feature extraction
        """
        self.config = config.get('features', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Store transition matrix and other computed features
        self.transition_matrix = None
        self.pattern_cache = {}
        
    def extract_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transition-based features from event sequences.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with transition features
        """
        df_trans = df.copy()
        
        # Calculate transition matrix
        self.transition_matrix = self._calculate_transition_matrix(df)
        
        # Add transition probability features
        df_trans = self._add_transition_probabilities(df_trans)
        
        # Add transition history features
        df_trans = self._add_transition_history(df_trans)
        
        # Add state persistence features
        df_trans = self._add_state_persistence(df_trans)
        
        logger.info("Transition features extracted successfully")
        return df_trans
    
    def extract_sliding_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features using sliding windows over the sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with sliding window features
        """
        df_window = df.copy()
        
        # Get sliding window configuration
        window_config = self.config.get('sliding_window', {})
        window_size = window_config.get('window_size', 20)
        step_size = window_config.get('step_size', 5)
        
        # Extract window-based features
        df_window = self._extract_window_diversity(df_window, window_size)
        df_window = self._extract_window_patterns(df_window, window_size)
        df_window = self._extract_window_entropy(df_window, window_size)
        
        logger.info("Sliding window features extracted successfully")
        return df_window
    
    def extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pattern-based features from event sequences.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with pattern features
        """
        df_pattern = df.copy()
        
        # Extract n-gram patterns
        df_pattern = self._extract_ngram_features(df_pattern)
        
        # Extract sequential patterns
        df_pattern = self._extract_sequential_patterns(df_pattern)
        
        # Extract repetition patterns
        df_pattern = self._extract_repetition_features(df_pattern)
        
        logger.info("Pattern features extracted successfully")
        return df_pattern
    
    def _calculate_transition_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transition probability matrix."""
        events = df[self.event_column].values
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        total_transitions = defaultdict(int)
        
        for i in range(len(events) - 1):
            from_event = events[i]
            to_event = events[i + 1]
            
            transition_counts[from_event][to_event] += 1
            total_transitions[from_event] += 1
        
        # Calculate probabilities
        transition_matrix = {}
        for from_event in transition_counts:
            transition_matrix[from_event] = {}
            total = total_transitions[from_event]
            
            for to_event in transition_counts[from_event]:
                transition_matrix[from_event][to_event] = transition_counts[from_event][to_event] / total
        
        # Convert to DataFrame for easier access
        all_events = list(set(events))
        matrix_df = pd.DataFrame(0.0, index=all_events, columns=all_events)
        
        for from_event in transition_matrix:
            for to_event in transition_matrix[from_event]:
                matrix_df.loc[from_event, to_event] = transition_matrix[from_event][to_event]
        
        return matrix_df
    
    def _add_transition_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transition probability features to each event."""
        events = df[self.event_column].values
        
        # Previous and next event transition probabilities
        prev_trans_probs = []
        next_trans_probs = []
        
        for i in range(len(events)):
            if i == 0:
                prev_trans_probs.append(0.0)
            else:
                prev_event = events[i-1]
                current_event = events[i]
                if self.transition_matrix is not None and prev_event in self.transition_matrix.index:
                    prev_trans_probs.append(self.transition_matrix.loc[prev_event, current_event])
                else:
                    prev_trans_probs.append(0.0)
            
            if i == len(events) - 1:
                next_trans_probs.append(0.0)
            else:
                current_event = events[i]
                next_event = events[i+1]
                if self.transition_matrix is not None and current_event in self.transition_matrix.index:
                    next_trans_probs.append(self.transition_matrix.loc[current_event, next_event])
                else:
                    next_trans_probs.append(0.0)
        
        df['prev_transition_prob'] = prev_trans_probs
        df['next_transition_prob'] = next_trans_probs
        
        # Average transition probability
        df['avg_transition_prob'] = (df['prev_transition_prob'] + df['next_transition_prob']) / 2
        
        return df
    
    def _add_transition_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transition history features."""
        events = df[self.event_column].values
        
        # Count unique transitions seen so far
        unique_transitions_seen = []
        transition_pairs_seen = set()
        
        for i in range(len(events)):
            if i > 0:
                transition_pair = (events[i-1], events[i])
                transition_pairs_seen.add(transition_pair)
            unique_transitions_seen.append(len(transition_pairs_seen))
        
        df['unique_transitions_seen'] = unique_transitions_seen
        
        # Transition diversity (unique transitions / total transitions)
        total_transitions = np.arange(len(events))
        df['transition_diversity'] = np.array(unique_transitions_seen) / (total_transitions + 1)
        
        return df
    
    def _add_state_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add state persistence features."""
        events = df[self.event_column].values
        
        # Count consecutive same events
        consecutive_counts = []
        current_count = 1
        current_event = events[0] if len(events) > 0 else None
        
        for i in range(len(events)):
            if i == 0:
                consecutive_counts.append(1)
            else:
                if events[i] == current_event:
                    current_count += 1
                else:
                    current_count = 1
                    current_event = events[i]
                consecutive_counts.append(current_count)
        
        df['consecutive_count'] = consecutive_counts
        
        # Is this the start of a new state?
        df['is_state_start'] = (df['consecutive_count'] == 1).astype(int)
        
        # State persistence probability
        state_persistence = []
        for event in events:
            if self.transition_matrix is not None and event in self.transition_matrix.index:
                persistence_prob = self.transition_matrix.loc[event, event]
                state_persistence.append(persistence_prob)
            else:
                state_persistence.append(0.0)
        
        df['state_persistence_prob'] = state_persistence
        
        return df
    
    def _extract_window_diversity(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Extract diversity features within sliding windows."""
        events = df[self.event_column].values
        
        for window in [window_size, window_size//2, window_size*2]:
            if len(df) >= window:
                # Rolling unique event count
                df[f'unique_events_{window}'] = df[self.event_column].rolling(window=window).apply(lambda x: len(set(x)), raw=False)
                
                # Rolling Shannon entropy
                def calculate_entropy(window_events):
                    if len(window_events) == 0:
                        return 0.0
                    counts = Counter(window_events)
                    probs = np.array(list(counts.values())) / len(window_events)
                    return -np.sum(probs * np.log(probs + 1e-10))
                
                entropy_values = []
                for i in range(len(events)):
                    start_idx = max(0, i - window + 1)
                    window_events = events[start_idx:i+1]
                    entropy_values.append(calculate_entropy(window_events))
                
                df[f'entropy_{window}'] = entropy_values
        
        return df
    
    def _extract_window_patterns(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Extract pattern features within sliding windows."""
        events = df[self.event_column].values
        
        if len(df) < window_size:
            return df
        
        # Most common pattern in window
        def most_common_pattern(window_events):
            if len(window_events) < 2:
                return ""
            
            patterns = []
            for i in range(len(window_events) - 1):
                pattern = f"{window_events[i]}->{window_events[i+1]}"
                patterns.append(pattern)
            
            if patterns:
                return Counter(patterns).most_common(1)[0][0]
            return ""
        
        common_patterns = []
        for i in range(len(events)):
            start_idx = max(0, i - window_size + 1)
            window_events = events[start_idx:i+1]
            common_patterns.append(most_common_pattern(window_events))
        
        df['common_pattern'] = common_patterns
        
        return df
    
    def _extract_window_entropy(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Extract entropy features within sliding windows."""
        events = df[self.event_column].values
        
        if len(df) < window_size:
            return df
        
        # Calculate entropy for different window sizes
        for window in [window_size, window_size//2, window_size*2]:
            if len(df) >= window:
                entropy_values = []
                
                for i in range(len(events)):
                    start_idx = max(0, i - window + 1)
                    window_events = events[start_idx:i+1]
                    
                    # Calculate entropy
                    if len(window_events) > 0:
                        counts = Counter(window_events)
                        probs = np.array(list(counts.values())) / len(window_events)
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                    else:
                        entropy = 0.0
                    
                    entropy_values.append(entropy)
                
                df[f'window_entropy_{window}'] = entropy_values
        
        return df
    
    def _extract_ngram_features(self, df: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
        """Extract n-gram features from event sequences."""
        events = df[self.event_column].values
        
        # Extract n-grams for different n values
        for n in range(2, min(max_n + 1, len(events))):
            ngram_freqs = []
            ngrams = []
            
            for i in range(len(events) - n + 1):
                ngram = tuple(events[i:i+n])
                ngrams.append(ngram)
            
            # Add n-gram frequency features
            ngram_counts = Counter(ngrams)
            total_ngrams = len(ngrams)
            
            for i, event in enumerate(events):
                # Look for n-grams starting at this position
                if i <= len(events) - n:
                    ngram = tuple(events[i:i+n])
                    ngram_freq = ngram_counts[ngram] / total_ngrams if total_ngrams > 0 else 0
                    ngram_freqs.append(ngram_freq)
                else:
                    ngram_freqs.append(0.0)
            
            df[f'{n}gram_freq'] = ngram_freqs
        
        return df
    
    def _extract_sequential_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sequential pattern features."""
        events = df[self.event_column].values
        
        # Pattern length (sequences of same event)
        pattern_lengths = []
        current_pattern = [events[0]] if len(events) > 0 else []
        
        for i in range(len(events)):
            if i == 0:
                pattern_lengths.append(1)
            else:
                if events[i] == events[i-1]:
                    current_pattern.append(events[i])
                else:
                    pattern_lengths.append(len(current_pattern))
                    current_pattern = [events[i]]
        
        # Add final pattern length
        if len(events) > 0:
            pattern_lengths[-1] = len(current_pattern)
        
        df['pattern_length'] = pattern_lengths
        
        # Pattern complexity (number of unique events in recent history)
        for window in [5, 10, 20]:
            if len(df) >= window:
                complexity = []
                for i in range(len(events)):
                    start_idx = max(0, i - window + 1)
                    window_events = events[start_idx:i+1]
                    complexity.append(len(set(window_events)))
                
                df[f'pattern_complexity_{window}'] = complexity
        
        return df
    
    def _extract_repetition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract repetition-based features."""
        events = df[self.event_column].values
        
        # Time since last occurrence of same event
        time_since_same_event = []
        last_occurrence = {}
        
        for i, event in enumerate(events):
            if event in last_occurrence:
                time_since_same_event.append(i - last_occurrence[event])
            else:
                time_since_same_event.append(i)  # First occurrence
            
            last_occurrence[event] = i
        
        df['time_since_same_event'] = time_since_same_event
        
        # Repetition rate (how often events repeat)
        repetition_rates = []
        event_counts = Counter()
        
        for i, event in enumerate(events):
            event_counts[event] += 1
            total_events = i + 1
            repetition_rates.append(event_counts[event] / total_events)
        
        df['repetition_rate'] = repetition_rates
        
        # Is this a repeated event?
        df['is_repeated'] = (df['time_since_same_event'] > 0).astype(int)
        
        return df
    
    def find_frequent_patterns(self, df: pd.DataFrame, 
                             min_support: float = 0.1,
                             max_pattern_length: int = 5) -> List[Tuple]:
        """
        Find frequent patterns in the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            min_support (float): Minimum support threshold
            max_pattern_length (int): Maximum pattern length
            
        Returns:
            List[Tuple]: List of frequent patterns with their support
        """
        events = df[self.event_column].values
        total_events = len(events)
        
        frequent_patterns = []
        
        # Generate patterns of different lengths
        for length in range(2, min(max_pattern_length + 1, len(events))):
            pattern_counts = Counter()
            
            for i in range(len(events) - length + 1):
                pattern = tuple(events[i:i+length])
                pattern_counts[pattern] += 1
            
            # Filter by minimum support
            min_count = int(min_support * total_events)
            for pattern, count in pattern_counts.items():
                if count >= min_count:
                    support = count / total_events
                    frequent_patterns.append((pattern, support, count))
        
        # Sort by support
        frequent_patterns.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(frequent_patterns)} frequent patterns")
        return frequent_patterns
    
    def calculate_sequence_complexity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various complexity metrics for the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            Dict[str, float]: Dictionary of complexity metrics
        """
        events = df[self.event_column].values
        
        complexity_metrics = {}
        
        # Event type diversity (Shannon entropy)
        event_counts = Counter(events)
        total_events = len(events)
        probs = np.array(list(event_counts.values())) / total_events
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        complexity_metrics['entropy'] = entropy
        
        # Normalized entropy (0 to 1)
        max_entropy = np.log(len(event_counts))
        complexity_metrics['normalized_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
        
        # Lempel-Ziv complexity (approximation)
        complexity_metrics['lempel_ziv_complexity'] = self._lempel_ziv_complexity(events)
        
        # Transition complexity
        if self.transition_matrix is not None:
            # Average transition entropy
            transition_entropies = []
            for from_event in self.transition_matrix.index:
                transitions = self.transition_matrix.loc[from_event].values
                transitions = transitions[transitions > 0]  # Remove zero probabilities
                if len(transitions) > 0:
                    trans_entropy = -np.sum(transitions * np.log(transitions + 1e-10))
                    transition_entropies.append(trans_entropy)
            
            complexity_metrics['avg_transition_entropy'] = np.mean(transition_entropies) if transition_entropies else 0
        else:
            complexity_metrics['avg_transition_entropy'] = 0
        
        return complexity_metrics
    
    def _lempel_ziv_complexity(self, sequence: List) -> float:
        """Calculate Lempel-Ziv complexity (approximation)."""
        if len(sequence) == 0:
            return 0.0
        
        # Simple LZ complexity approximation
        seen_substrings = set()
        complexity = 0
        
        for i in range(len(sequence)):
            for j in range(i + 1, min(i + 10, len(sequence) + 1)):  # Limit substring length
                substring = tuple(sequence[i:j])
                if substring not in seen_substrings:
                    seen_substrings.add(substring)
                    complexity += 1
                    break
        
        # Normalize by sequence length
        return complexity / len(sequence)
