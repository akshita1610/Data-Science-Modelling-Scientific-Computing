"""
Main Feature Extractor Module

Coordinates extraction of all features from event sequences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .time_features import TimeFeatureExtractor
from .sequence_features import SequenceFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Main feature extractor that coordinates all feature extraction processes.
    
    This class serves as the main interface for extracting features from event
    sequences, coordinating between different specialized extractors.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FeatureExtractor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for feature extraction
        """
        self.config = config.get('features', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Initialize specialized extractors
        self.time_extractor = TimeFeatureExtractor(config)
        self.sequence_extractor = SequenceFeatureExtractor(config)
        
        # Feature storage
        self.extracted_features = {}
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all available features from the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with all extracted features
        """
        logger.info(f"Extracting features from {len(df)} events")
        
        df_features = df.copy()
        
        # Extract time-based features
        if self.config.get('time_intervals', {}).get('enabled', True):
            df_features = self.time_extractor.extract_time_features(df_features)
        
        # Extract frequency-based features
        if self.config.get('event_frequency', {}).get('enabled', True):
            df_features = self.time_extractor.extract_frequency_features(df_features)
        
        # Extract transition features
        if self.config.get('transition_probabilities', {}).get('enabled', True):
            df_features = self.sequence_extractor.extract_transition_features(df_features)
        
        # Extract sliding window features
        if self.config.get('sliding_window', {}).get('enabled', True):
            df_features = self.sequence_extractor.extract_sliding_window_features(df_features)
        
        # Extract pattern-based features (temporarily disabled for demo)
        # df_features = self.sequence_extractor.extract_pattern_features(df_features)
        
        # Extract statistical features (temporarily disabled for demo)
        # df_features = self._extract_statistical_features(df_features)
        
        logger.info(f"Feature extraction completed. Total features: {len(df_features.columns)}")
        return df_features
    
    def extract_features_for_segments(self, segments: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Extract features for each segment individually.
        
        Args:
            segments (List[pd.DataFrame]): List of segment DataFrames
            
        Returns:
            List[pd.DataFrame]: List of segments with extracted features
        """
        logger.info(f"Extracting features for {len(segments)} segments")
        
        segments_with_features = []
        
        for i, segment in enumerate(segments):
            try:
                segment_features = self.extract_all_features(segment)
                segments_with_features.append(segment_features)
            except Exception as e:
                logger.warning(f"Error extracting features for segment {i}: {e}")
                segments_with_features.append(segment)  # Keep original segment
        
        return segments_with_features
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features from the event sequence.
        
        Args:
            df (pd.DataFrame): Event data
            
        Returns:
            pd.DataFrame: Data with statistical features
        """
        df_stats = df.copy()
        
        # Event type statistics
        event_counts = df_stats[self.event_column].value_counts()
        df_stats['event_frequency_rank'] = df_stats[self.event_column].map(event_counts.rank(method='dense', ascending=False))
        
        # Time-based statistics
        if 'time_since_previous' in df_stats.columns:
            df_stats['time_gap_percentile'] = df_stats['time_since_previous'].rank(pct=True)
        
        # Rolling statistics
        if len(df_stats) >= 5:
            window_size = min(5, len(df_stats))
            
            # Rolling event frequency
            df_stats['rolling_event_count'] = df_stats[self.event_column].rolling(window=window_size).count()
            
            # Rolling unique event types
            df_stats['rolling_unique_events'] = df_stats[self.event_column].rolling(window=window_size).apply(lambda x: len(set(x)), raw=False)
            
            # Rolling time gap statistics
            if 'time_since_previous' in df_stats.columns:
                df_stats['rolling_mean_gap'] = df_stats['time_since_previous'].rolling(window=window_size).mean()
                df_stats['rolling_std_gap'] = df_stats['time_since_previous'].rolling(window=window_size).std()
        
        return df_stats
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a feature matrix suitable for machine learning.
        
        Args:
            df (pd.DataFrame): Event data with features
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns
        exclude_columns = [self.timestamp_column, 'segment_id', 'window_id', 'event_index']
        feature_columns = [col for col in numeric_features if col not in exclude_columns]
        
        feature_matrix = df[feature_columns].copy()
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(0)
        
        logger.info(f"Created feature matrix with {len(feature_columns)} features")
        return feature_matrix
    
    def get_feature_importance(self, df: pd.DataFrame, 
                             target_column: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate feature importance using various methods.
        
        Args:
            df (pd.DataFrame): Event data with features
            target_column (Optional[str]): Target column for supervised importance
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        feature_matrix = self.create_feature_matrix(df)
        
        if target_column and target_column in df.columns:
            # Supervised feature importance
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                
                # Prepare target
                target = df[target_column]
                if target.dtype == 'object':
                    le = LabelEncoder()
                    target = le.fit_transform(target)
                
                # Train model
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(feature_matrix, target)
                
                importance = dict(zip(feature_matrix.columns, rf.feature_importances_))
                logger.info("Calculated supervised feature importance using Random Forest")
                
            except Exception as e:
                logger.warning(f"Could not calculate supervised importance: {e}")
                importance = self._calculate_unsupervised_importance(feature_matrix)
        else:
            # Unsupervised feature importance
            importance = self._calculate_unsupervised_importance(feature_matrix)
        
        return importance
    
    def _calculate_unsupervised_importance(self, feature_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate unsupervised feature importance."""
        importance = {}
        
        for col in feature_matrix.columns:
            # Variance-based importance
            variance = feature_matrix[col].var()
            # Range-based importance
            range_val = feature_matrix[col].max() - feature_matrix[col].min()
            # Combined score
            importance[col] = variance * range_val
        
        # Normalize importance scores
        max_importance = max(importance.values()) if importance.values() else 1
        importance = {k: v/max_importance for k, v in importance.items()}
        
        logger.info("Calculated unsupervised feature importance")
        return importance
    
    def select_top_features(self, df: pd.DataFrame, 
                          top_k: int = 10) -> List[str]:
        """
        Select top k features based on importance.
        
        Args:
            df (pd.DataFrame): Event data with features
            top_k (int): Number of top features to select
            
        Returns:
            List[str]: List of top feature names
        """
        importance = self.get_feature_importance(df)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        top_features = [feature for feature, _ in sorted_features[:top_k]]
        
        logger.info(f"Selected top {len(top_features)} features")
        return top_features
    
    def reduce_features(self, df: pd.DataFrame, 
                       method: str = 'pca',
                       n_components: int = 10) -> pd.DataFrame:
        """
        Reduce feature dimensionality.
        
        Args:
            df (pd.DataFrame): Event data with features
            method (str): Reduction method ('pca', 'variance_threshold')
            n_components (int): Number of components to keep
            
        Returns:
            pd.DataFrame: Data with reduced features
        """
        feature_matrix = self.create_feature_matrix(df)
        
        if method == 'pca':
            try:
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=n_components, random_state=42)
                reduced_features = pca.fit_transform(feature_matrix)
                
                # Create new DataFrame with reduced features
                reduced_df = df.copy()
                for i in range(n_components):
                    reduced_df[f'pca_component_{i}'] = reduced_features[:, i]
                
                logger.info(f"Reduced features using PCA to {n_components} components")
                return reduced_df
                
            except Exception as e:
                logger.error(f"PCA reduction failed: {e}")
                return df
        
        elif method == 'variance_threshold':
            try:
                from sklearn.feature_selection import VarianceThreshold
                
                selector = VarianceThreshold(threshold=0.01)
                selected_features = selector.fit_transform(feature_matrix)
                
                # Get selected feature names
                selected_mask = selector.get_support()
                selected_names = feature_matrix.columns[selected_mask]
                
                # Create new DataFrame with selected features
                reduced_df = df.copy()
                for i, name in enumerate(selected_names):
                    reduced_df[f'selected_{name}'] = selected_features[:, i]
                
                logger.info(f"Reduced features using variance threshold to {len(selected_names)} features")
                return reduced_df
                
            except Exception as e:
                logger.error(f"Variance threshold reduction failed: {e}")
                return df
        
        else:
            logger.warning(f"Unknown reduction method: {method}")
            return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of extracted features.
        
        Args:
            df (pd.DataFrame): Event data with features
            
        Returns:
            Dict: Feature summary statistics
        """
        feature_matrix = self.create_feature_matrix(df)
        
        summary = {
            'total_features': len(feature_matrix.columns),
            'missing_values': feature_matrix.isnull().sum().sum(),
            'feature_types': {
                'numeric': len(feature_matrix.select_dtypes(include=[np.number]).columns),
                'categorical': len(feature_matrix.select_dtypes(include=['object', 'category']).columns)
            },
            'feature_stats': {
                'mean_variance': feature_matrix.var().mean(),
                'max_variance': feature_matrix.var().max(),
                'min_variance': feature_matrix.var().min(),
                'mean_range': (feature_matrix.max() - feature_matrix.min()).mean()
            }
        }
        
        return summary
