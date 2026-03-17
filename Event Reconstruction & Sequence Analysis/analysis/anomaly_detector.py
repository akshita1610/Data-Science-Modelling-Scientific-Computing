"""
Anomaly Detection Module

Detects anomalies in event sequences using various statistical and ML methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in event sequences."""
    
    def __init__(self, config: Dict):
        self.config = config.get('analysis', {}).get('anomaly_detection', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using multiple methods."""
        df_anomaly = df.copy()
        
        method = self.config.get('method', 'isolation_forest')
        
        if method == 'isolation_forest':
            df_anomaly = self._detect_isolation_forest_anomalies(df_anomaly)
        elif method == 'zscore':
            df_anomaly = self._detect_zscore_anomalies(df_anomaly)
        
        df_anomaly = self._detect_temporal_anomalies(df_anomaly)
        df_anomaly = self._detect_sequence_anomalies(df_anomaly)
        
        return df_anomaly
    
    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        # Create feature matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in [self.timestamp_column, 'event_index']]
        
        if not feature_cols:
            return df
        
        X = df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        contamination = self.config.get('contamination', 0.1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        df['isolation_anomaly'] = anomaly_labels == -1
        df['isolation_score'] = anomaly_scores
        
        return df
    
    def _detect_zscore_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_threshold = 3.0
        
        for col in numeric_cols:
            if col not in [self.timestamp_column, 'event_index']:
                z_scores = np.abs(stats.zscore(df[col].fillna(0)))
                df[f'zscore_anomaly_{col}'] = z_scores > z_threshold
        
        return df
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect temporal anomalies."""
        timestamps = pd.to_datetime(df[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds()
        
        # Detect unusually large gaps
        gap_threshold = np.percentile(time_gaps.dropna(), 95)
        df['temporal_gap_anomaly'] = time_gaps > gap_threshold
        
        # Detect rapid succession
        rapid_threshold = np.percentile(time_gaps.dropna(), 5)
        df['temporal_rapid_anomaly'] = time_gaps < rapid_threshold
        
        return df
    
    def _detect_sequence_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect sequence anomalies."""
        events = df[self.event_column].values
        
        # Detect rare event transitions
        transition_counts = {}
        for i in range(len(events) - 1):
            transition = (events[i], events[i+1])
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
        
        # Find rare transitions
        total_transitions = sum(transition_counts.values())
        rare_threshold = 0.01  # Less than 1% of transitions
        
        sequence_anomalies = []
        for i in range(len(events) - 1):
            transition = (events[i], events[i+1])
            freq = transition_counts.get(transition, 0) / total_transitions
            sequence_anomalies.append(freq < rare_threshold)
        
        df['sequence_anomaly'] = [False] + sequence_anomalies
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of detected anomalies."""
        anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower()]
        
        summary = {}
        for col in anomaly_cols:
            summary[col] = {
                'count': df[col].sum(),
                'percentage': df[col].mean() * 100
            }
        
        return summary
