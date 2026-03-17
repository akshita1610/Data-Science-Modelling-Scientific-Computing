"""
Probabilistic Event Reconstructor Module

Implements probabilistic reconstruction using Markov models and statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProbabilisticReconstructor:
    """
    Probabilistic event reconstructor.
    
    Uses statistical models and probabilistic methods to reconstruct missing events
    and predict likely event sequences.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ProbabilisticReconstructor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for reconstruction
        """
        self.config = config.get('reconstruction', {})
        self.prob_config = self.config.get('probabilistic', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Model parameters
        self.markov_order = self.prob_config.get('markov_order', 1)
        self.smoothing_factor = self.prob_config.get('smoothing_factor', 0.01)
        
        # Trained models
        self.markov_model = None
        self.transition_probabilities = None
        self.event_probabilities = None
        self.time_gap_model = None
        
    def reconstruct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform probabilistic event reconstruction.
        
        Args:
            df (pd.DataFrame): Original event data
            
        Returns:
            pd.DataFrame: Reconstructed event data
        """
        logger.info("Starting probabilistic reconstruction")
        
        # Train models on existing data
        self._train_models(df)
        
        df_reconstructed = df.copy()
        
        # Step 1: Detect gaps and missing events probabilistically
        df_reconstructed = self._detect_probabilistic_gaps(df_reconstructed)
        
        # Step 2: Fill missing events using probabilistic prediction
        df_reconstructed = self._fill_probabilistic_events(df_reconstructed)
        
        # Step 3: Optimize sequence using probabilistic methods
        df_reconstructed = self._optimize_sequence(df_reconstructed)
        
        # Step 4: Add confidence scores
        df_reconstructed = self._add_confidence_scores(df_reconstructed)
        
        logger.info(f"Probabilistic reconstruction completed: {len(df)} -> {len(df_reconstructed)} events")
        return df_reconstructed
    
    def _train_models(self, df: pd.DataFrame):
        """Train probabilistic models on the event data."""
        logger.info("Training probabilistic models")
        
        # Train Markov model
        self._train_markov_model(df)
        
        # Train event probability model
        self._train_event_probabilities(df)
        
        # Train time gap model
        self._train_time_gap_model(df)
        
    def _train_markov_model(self, df: pd.DataFrame):
        """Train Markov model for event transitions."""
        events = df[self.event_column].values
        
        # Build transition counts
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        if self.markov_order == 1:
            # First-order Markov model
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]
                transition_counts[current_event][next_event] += 1
        else:
            # Higher-order Markov model
            for i in range(len(events) - self.markov_order):
                current_state = tuple(events[i:i+self.markov_order])
                next_event = events[i + self.markov_order]
                transition_counts[current_state][next_event] += 1
        
        # Convert to probabilities with smoothing
        self.markov_model = {}
        
        for from_state in transition_counts:
            total_transitions = sum(transition_counts[from_state].values())
            self.markov_model[from_state] = {}
            
            for to_event in transition_counts[from_state]:
                # Add Laplace smoothing
                smoothed_count = transition_counts[from_state][to_event] + self.smoothing_factor
                smoothed_total = total_transitions + self.smoothing_factor * len(transition_counts[from_state])
                self.markov_model[from_state][to_event] = smoothed_count / smoothed_total
    
    def _train_event_probabilities(self, df: pd.DataFrame):
        """Train event probability model."""
        events = df[self.event_column].values
        event_counts = Counter(events)
        total_events = len(events)
        
        # Calculate probabilities with smoothing
        unique_events = len(event_counts)
        self.event_probabilities = {}
        
        for event in event_counts:
            smoothed_count = event_counts[event] + self.smoothing_factor
            smoothed_total = total_events + self.smoothing_factor * unique_events
            self.event_probabilities[event] = smoothed_count / smoothed_total
    
    def _train_time_gap_model(self, df: pd.DataFrame):
        """Train time gap distribution model."""
        timestamps = pd.to_datetime(df[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        
        # Model time gaps by event type
        self.time_gap_model = {}
        
        for event_type in df[self.event_column].unique():
            event_timestamps = timestamps[df[self.event_column] == event_type]
            if len(event_timestamps) > 1:
                event_gaps = event_timestamps.diff().dt.total_seconds().dropna()
                self.time_gap_model[event_type] = {
                    'mean': event_gaps.mean(),
                    'std': event_gaps.std(),
                    'median': event_gaps.median(),
                    'distribution': event_gaps.values
                }
            else:
                self.time_gap_model[event_type] = {
                    'mean': 60.0,  # Default 1 minute
                    'std': 30.0,
                    'median': 60.0,
                    'distribution': np.array([60.0])
                }
    
    def _detect_probabilistic_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect gaps using probabilistic methods."""
        df_gaps = df.copy()
        
        # Calculate time gaps
        timestamps = pd.to_datetime(df[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds()
        
        # Detect statistically significant gaps
        gap_scores = []
        
        for i in range(1, len(time_gaps)):
            gap = time_gaps.iloc[i]
            prev_event = df.iloc[i-1][self.event_column]
            
            # Get expected gap for this event type
            if prev_event in self.time_gap_model:
                expected_gap = self.time_gap_model[prev_event]['mean']
                gap_std = self.time_gap_model[prev_event]['std']
                
                # Calculate z-score
                if gap_std > 0:
                    z_score = (gap - expected_gap) / gap_std
                else:
                    z_score = 0
                
                # Gap is significant if z-score > 2
                is_significant = abs(z_score) > 2
            else:
                z_score = 0
                is_significant = False
            
            gap_scores.append({
                'gap_size': gap,
                'z_score': z_score,
                'is_significant': is_significant
            })
        
        df_gaps['gap_analysis'] = [None] + gap_scores
        
        return df_gaps
    
    def _fill_probabilistic_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing events using probabilistic prediction."""
        df_filled = df.copy()
        
        # Identify positions where events should be inserted
        insertion_positions = self._identify_insertion_positions(df_filled)
        
        # Generate events for each position
        new_events = []
        
        for position_info in insertion_positions:
            predicted_events = self._predict_events_at_position(df_filled, position_info)
            new_events.extend(predicted_events)
        
        # Add new events to DataFrame
        if new_events:
            df_new = pd.DataFrame(new_events)
            df_filled = pd.concat([df_filled, df_new], ignore_index=True)
            df_filled = df_filled.sort_values(self.timestamp_column).reset_index(drop=True)
        
        return df_filled
    
    def _identify_insertion_positions(self, df: pd.DataFrame) -> List[Dict]:
        """Identify positions where events should be inserted."""
        positions = []
        
        gap_analysis = df.get('gap_analysis', [])
        if isinstance(gap_analysis, list):
            for i, gap_info in enumerate(gap_analysis):
                if gap_info and gap_info.get('is_significant', False):
                    positions.append({
                        'index': i + 1,  # Position after the gap
                        'gap_size': gap_info['gap_size'],
                        'z_score': gap_info['z_score']
                    })
        
        return positions
    
    def _predict_events_at_position(self, df: pd.DataFrame, 
                                  position_info: Dict) -> List[Dict]:
        """Predict events to insert at a specific position."""
        index = position_info['index']
        gap_size = position_info['gap_size']
        
        if index >= len(df):
            return []
        
        # Get context events
        prev_event = df.iloc[index - 1][self.event_column] if index > 0 else None
        next_event = df.iloc[index][self.event_column] if index < len(df) else None
        
        # Predict number of events to insert
        num_events = self._predict_num_events(gap_size, prev_event, next_event)
        
        # Generate predicted events
        predicted_events = []
        
        if num_events > 0:
            # Get timestamps for insertion
            prev_timestamp = pd.to_datetime(df.iloc[index - 1][self.timestamp_column]) if index > 0 else pd.Timestamp.now()
            next_timestamp = pd.to_datetime(df.iloc[index][self.timestamp_column]) if index < len(df) else prev_timestamp + timedelta(minutes=gap_size/60)
            
            # Generate events
            current_state = tuple([prev_event] * self.markov_order) if prev_event else None
            
            for i in range(num_events):
                # Predict next event
                predicted_event = self._predict_next_event(current_state, next_event)
                
                # Calculate timestamp
                event_time = prev_timestamp + (next_timestamp - prev_timestamp) * (i + 1) / (num_events + 1)
                
                predicted_events.append({
                    self.timestamp_column: event_time,
                    self.event_column: predicted_event,
                    'reconstruction_method': 'probabilistic',
                    'reconstruction_confidence': self._calculate_confidence(current_state, predicted_event),
                    'gap_size': gap_size
                })
                
                # Update state for next prediction
                current_state = self._update_state(current_state, predicted_event)
        
        return predicted_events
    
    def _predict_num_events(self, gap_size: float, 
                          prev_event: Optional[str], 
                          next_event: Optional[str]) -> int:
        """Predict number of events to insert in a gap."""
        # Simple heuristic based on gap size and typical event intervals
        if prev_event and prev_event in self.time_gap_model:
            typical_gap = self.time_gap_model[prev_event]['mean']
            if typical_gap > 0:
                estimated_events = max(0, int(gap_size / typical_gap) - 1)
                return min(estimated_events, 5)  # Cap at 5 events
        
        return 1  # Default to 1 event
    
    def _predict_next_event(self, current_state: Optional[Tuple], 
                          next_event: Optional[str]) -> str:
        """Predict the next event using the Markov model."""
        if current_state and current_state in self.markov_model:
            # Get probabilities for next events
            next_probs = self.markov_model[current_state]
            
            # If we know the actual next event, bias towards it
            if next_event and next_event in next_probs:
                # Increase probability of the actual next event
                adjusted_probs = next_probs.copy()
                adjusted_probs[next_event] = min(1.0, adjusted_probs[next_event] * 2)
                
                # Renormalize
                total_prob = sum(adjusted_probs.values())
                adjusted_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
                
                # Sample from adjusted distribution
                events = list(adjusted_probs.keys())
                probs = list(adjusted_probs.values())
                return np.random.choice(events, p=probs)
            else:
                # Sample from original distribution
                events = list(next_probs.keys())
                probs = list(next_probs.values())
                return np.random.choice(events, p=probs)
        else:
            # Fall back to overall event probabilities
            if self.event_probabilities:
                events = list(self.event_probabilities.keys())
                probs = list(self.event_probabilities.values())
                return np.random.choice(events, p=probs)
            else:
                return 'unknown'
    
    def _update_state(self, current_state: Optional[Tuple], 
                     new_event: str) -> Tuple:
        """Update the state for the next prediction."""
        if current_state is None:
            return (new_event,)
        
        # Create new state by adding new event and maintaining order
        new_state = list(current_state[1:]) + [new_event]
        return tuple(new_state)
    
    def _calculate_confidence(self, current_state: Optional[Tuple], 
                            predicted_event: str) -> float:
        """Calculate confidence score for a prediction."""
        if current_state and current_state in self.markov_model:
            if predicted_event in self.markov_model[current_state]:
                return self.markov_model[current_state][predicted_event]
        
        # Fall back to event probability
        if predicted_event in self.event_probabilities:
            return self.event_probabilities[predicted_event]
        
        return 0.1  # Default low confidence
    
    def _optimize_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize the reconstructed sequence using probabilistic methods."""
        df_optimized = df.copy()
        
        # Apply simulated annealing or similar optimization
        df_optimized = self._local_sequence_optimization(df_optimized)
        
        return df_optimized
    
    def _local_sequence_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform local sequence optimization."""
        events = df[self.event_column].values
        
        # Look for locally suboptimal sequences
        for i in range(len(events) - 2):
            window = events[i:i+3]
            
            # Calculate probability of this window
            window_prob = self._calculate_sequence_probability(window)
            
            # Try alternative events
            alternatives = self.event_probabilities.keys() if self.event_probabilities else []
            
            best_window = window
            best_prob = window_prob
            
            for alt_event in alternatives:
                if alt_event != window[1]:  # Try changing middle event
                    alt_window = [window[0], alt_event, window[2]]
                    alt_prob = self._calculate_sequence_probability(alt_window)
                    
                    if alt_prob > best_prob:
                        best_window = alt_window
                        best_prob = alt_prob
            
            # Update if improvement is significant
            if best_prob > window_prob * 1.5:  # 50% improvement threshold
                events[i+1] = best_window[1]
        
        df_optimized = df.copy()
        df_optimized[self.event_column] = events
        
        return df_optimized
    
    def _calculate_sequence_probability(self, sequence: List) -> float:
        """Calculate probability of a sequence using the Markov model."""
        if len(sequence) < 2:
            return 1.0
        
        prob = 1.0
        
        for i in range(len(sequence) - 1):
            current_event = sequence[i]
            next_event = sequence[i + 1]
            
            if self.markov_order == 1:
                if current_event in self.markov_model:
                    if next_event in self.markov_model[current_event]:
                        prob *= self.markov_model[current_event][next_event]
                    else:
                        prob *= self.smoothing_factor  # Very low probability
                else:
                    prob *= self.smoothing_factor
            else:
                # Higher-order Markov
                state_start = max(0, i - self.markov_order + 1)
                state = tuple(sequence[state_start:i+1])
                
                if state in self.markov_model:
                    if next_event in self.markov_model[state]:
                        prob *= self.markov_model[state][next_event]
                    else:
                        prob *= self.smoothing_factor
                else:
                    prob *= self.smoothing_factor
        
        return prob
    
    def _add_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add confidence scores to reconstructed events."""
        df_confidence = df.copy()
        
        # Calculate confidence for each event
        confidences = []
        
        for idx, row in df_confidence.iterrows():
            if 'reconstruction_confidence' in row:
                confidences.append(row['reconstruction_confidence'])
            else:
                # Original events have high confidence
                confidences.append(1.0)
        
        df_confidence['reconstruction_confidence'] = confidences
        
        return df_confidence
    
    def fill_missing_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to fill missing events using probabilistic methods.
        
        Args:
            df (pd.DataFrame): Event data with missing indicators
            
        Returns:
            pd.DataFrame: Event data with filled missing events
        """
        # Train models if not already trained
        if self.markov_model is None:
            self._train_models(df)
        
        return self._fill_probabilistic_events(df)
    
    def get_model_statistics(self) -> Dict:
        """
        Get statistics about the trained models.
        
        Returns:
            Dict: Model statistics
        """
        stats = {
            'markov_order': self.markov_order,
            'smoothing_factor': self.smoothing_factor,
            'num_states': len(self.markov_model) if self.markov_model else 0,
            'unique_events': len(self.event_probabilities) if self.event_probabilities else 0,
            'avg_transition_probability': 0.0,
            'model_complexity': 0.0
        }
        
        if self.markov_model:
            all_probs = []
            for state in self.markov_model:
                all_probs.extend(self.markov_model[state].values())
            
            if all_probs:
                stats['avg_transition_probability'] = np.mean(all_probs)
                stats['model_complexity'] = -np.sum([p * np.log(p) for p in all_probs if p > 0])
        
        return stats
