"""
Correlation Analyzer Module

Analyzes correlations between events and features in the sequence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes correlations in event sequences."""
    
    def __init__(self, config: Dict):
        self.config = config.get('analysis', {}).get('correlation_analysis', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive correlation analysis."""
        correlations = {
            'event_correlations': self._analyze_event_correlations(df),
            'temporal_correlations': self._analyze_temporal_correlations(df),
            'feature_correlations': self._analyze_feature_correlations(df),
            'cross_correlations': self._analyze_cross_correlations(df)
        }
        
        return correlations
    
    def _analyze_event_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between different event types."""
        events = df[self.event_column].values
        event_types = list(set(events))
        
        # Create event co-occurrence matrix
        cooccurrence = np.zeros((len(event_types), len(event_types)))
        
        for i in range(len(events) - 1):
            event1_idx = event_types.index(events[i])
            event2_idx = event_types.index(events[i + 1])
            cooccurrence[event1_idx, event2_idx] += 1
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(cooccurrence)
        
        return {
            'event_types': event_types,
            'cooccurrence_matrix': cooccurrence.tolist(),
            'correlation_matrix': correlation_matrix.tolist(),
            'strongest_correlations': self._find_strongest_correlations(correlation_matrix, event_types)
        }
    
    def _analyze_temporal_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal correlations."""
        timestamps = pd.to_datetime(df[self.timestamp_column])
        events = df[self.event_column].values
        
        # Extract temporal features
        temporal_features = {
            'hour': timestamps.dt.hour,
            'day_of_week': timestamps.dt.dayofweek,
            'month': timestamps.dt.month
        }
        
        correlations = {}
        
        for feature_name, feature_values in temporal_features.items():
            # Calculate correlation between temporal feature and event types
            event_encoded = pd.factorize(events)[0]
            
            if len(set(feature_values)) > 1:
                corr, p_value = spearmanr(feature_values, event_encoded)
                correlations[feature_name] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return correlations
    
    def _analyze_feature_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in [self.timestamp_column]]
        
        if len(feature_cols) < 2:
            return {'message': 'Insufficient numeric features for correlation analysis'}
        
        # Calculate correlation matrix
        correlation_matrix = df[feature_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'feature1': feature_cols[i],
                        'feature2': feature_cols[j],
                        'correlation': corr_val
                    })
        
        return {
            'features': feature_cols,
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _analyze_cross_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze cross-correlations between events and features."""
        events = df[self.event_column].values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in [self.timestamp_column]]
        
        cross_correlations = {}
        
        for col in feature_cols:
            feature_values = df[col].values
            event_encoded = pd.factorize(events)[0]
            
            if len(set(feature_values)) > 1:
                corr, p_value = pearsonr(feature_values, event_encoded)
                cross_correlations[col] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return cross_correlations
    
    def _find_strongest_correlations(self, correlation_matrix: np.ndarray, 
                                   event_types: List[str]) -> List[Dict]:
        """Find the strongest correlations in the matrix."""
        strongest = []
        
        for i in range(len(event_types)):
            for j in range(len(event_types)):
                if i != j:
                    corr_val = correlation_matrix[i, j]
                    if abs(corr_val) > 0.5:  # Threshold for strong correlation
                        strongest.append({
                            'event1': event_types[i],
                            'event2': event_types[j],
                            'correlation': corr_val
                        })
        
        # Sort by absolute correlation value
        strongest.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return strongest[:10]  # Return top 10
    
    def calculate_lag_correlations(self, df: pd.DataFrame, 
                                 max_lag: int = 10) -> Dict:
        """Calculate lagged correlations between events."""
        events = df[self.event_column].values
        event_types = list(set(events))
        
        lag_correlations = {}
        
        for event_type in event_types:
            # Create binary series for this event type
            event_series = (events == event_type).astype(int)
            
            correlations = []
            for lag in range(1, min(max_lag + 1, len(event_series))):
                if len(event_series) > lag:
                    corr = np.corrcoef(event_series[:-lag], event_series[lag:])[0, 1]
                    correlations.append(corr)
                else:
                    correlations.append(0)
            
            lag_correlations[event_type] = correlations
        
        return lag_correlations
