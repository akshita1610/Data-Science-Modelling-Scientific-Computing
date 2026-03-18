"""
Feature Engineer Module

Handles feature engineering for housing price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineer for creating and selecting optimal features.
    
    Features:
    - Housing-specific feature creation
    - Feature selection methods
    - Correlation analysis
    - Interaction features
    """
    
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []
        self.correlation_matrix = None
        
    def create_housing_features(self, df: pd.DataFrame, dataset_type: str = 'generic') -> pd.DataFrame:
        """
        Create housing-specific features based on dataset type.
        
        Args:
            df (pd.DataFrame): Input dataset
            dataset_type (str): Type of dataset ('hdb', 'portland', 'generic')
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        logger.info("Creating engineered features...")
        
        data = df.copy()
        
        if dataset_type.lower() == 'hdb':
            data = self._create_hdb_features(data)
        elif dataset_type.lower() == 'portland':
            data = self._create_portland_features(data)
        else:
            data = self._create_generic_features(data)
        
        logger.info(f"Created engineered features. New shape: {data.shape}")
        return data
    
    def _create_hdb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to HDB data."""
        data = df.copy()
        
        # Price per square meter
        if 'floor_area_sqm' in data.columns and 'resale_price' in data.columns:
            data['price_per_sqm'] = data['resale_price'] / data['floor_area_sqm']
        
        # Property age at sale time
        if 'lease_commence_date' in data.columns and 'year' in data.columns:
            data['property_age_at_sale'] = data['year'] - data['lease_commence_date']
        
        # Storey range processing
        if 'storey_range' in data.columns:
            # Extract lower and upper bounds
            storey_parts = data['storey_range'].str.split(' TO ', expand=True)
            if storey_parts.shape[1] == 2:
                data['storey_lower'] = pd.to_numeric(storey_parts[0], errors='coerce')
                data['storey_upper'] = pd.to_numeric(storey_parts[1], errors='coerce')
                data['storey_mid'] = (data['storey_lower'] + data['storey_upper']) / 2
        
        # Town price ranking (based on average prices)
        if 'town' in data.columns and 'resale_price' in data.columns:
            town_prices = data.groupby('town')['resale_price'].mean().sort_values(ascending=False)
            town_ranking = {town: rank for rank, town in enumerate(town_prices.index)}
            data['town_price_rank'] = data['town'].map(town_ranking)
        
        return data
    
    def _create_portland_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to Portland data."""
        data = df.copy()
        
        # Price per square foot
        if 'sqft_living' in data.columns and 'price' in data.columns:
            data['price_per_sqft'] = data['price'] / data['sqft_living']
        
        # Age of house
        if 'yr_built' in data.columns:
            current_year = 2024
            data['house_age'] = current_year - data['yr_built']
        
        # Total square footage
        sqft_cols = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
        available_sqft = [col for col in sqft_cols if col in data.columns]
        if available_sqft:
            data['total_sqft'] = data[available_sqft].sum(axis=1)
        
        return data
    
    def _create_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create generic features for any housing dataset."""
        data = df.copy()
        
        # Look for common housing columns and create features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Price per area (if both price and area columns exist)
        price_cols = [col for col in numeric_cols if 'price' in col.lower()]
        area_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['sqft', 'sqm', 'area'])]
        
        if price_cols and area_cols:
            price_col = price_cols[0]
            area_col = area_cols[0]
            data['price_per_area'] = data[price_col] / data[area_col]
        
        return data
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'univariate',
                       k: int = 10) -> Dict[str, Any]:
        """
        Select best features using various methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Selection method ('univariate', 'rfe', 'correlation')
            k (int): Number of features to select
            
        Returns:
            Dict: Feature selection results
        """
        logger.info(f"Selecting features using {method} method...")
        
        if method == 'univariate':
            results = self._univariate_selection(X, y, k)
        elif method == 'rfe':
            results = self._rfe_selection(X, y, k)
        elif method == 'correlation':
            results = self._correlation_selection(X, y, k)
        else:
            raise ValueError("method must be 'univariate', 'rfe', or 'correlation'")
        
        self.selected_features = results['selected_features']
        return results
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Dict[str, Any]:
        """Univariate feature selection using F-test."""
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(selected_features, feature_scores))
        self.feature_importance = feature_importance
        
        return {
            'method': 'univariate',
            'selected_features': selected_features,
            'feature_scores': feature_importance,
            'X_selected': X_selected
        }
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Dict[str, Any]:
        """Recursive Feature Elimination."""
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        feature_ranking = selector.ranking_
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, (feature, rank) in enumerate(zip(X.columns, feature_ranking)):
            if rank == 1:  # Selected features
                feature_importance[feature] = 1.0 / rank
        
        self.feature_importance = feature_importance
        
        return {
            'method': 'rfe',
            'selected_features': selected_features,
            'feature_ranking': dict(zip(X.columns, feature_ranking)),
            'X_selected': X_selected
        }
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Dict[str, Any]:
        """Feature selection based on correlation with target."""
        correlations = X.apply(lambda col: abs(col.corr(y)))
        
        # Sort by correlation and select top k
        top_features = correlations.nlargest(k).index.tolist()
        feature_importance = correlations[top_features].to_dict()
        
        self.feature_importance = feature_importance
        
        return {
            'method': 'correlation',
            'selected_features': top_features,
            'correlations': feature_importance,
            'X_selected': X[top_features]
        }
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between features.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Dict: Correlation analysis results
        """
        logger.info("Analyzing feature correlations...")
        
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        self.correlation_matrix = numeric_df.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_val = self.correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    feature1 = self.correlation_matrix.columns[i]
                    feature2 = self.correlation_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_val))
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'average_correlation': self.correlation_matrix.abs().mean().mean()
        }
    
    def create_interaction_features(self, X: pd.DataFrame, max_interactions: int = 5) -> pd.DataFrame:
        """
        Create interaction features between top correlated variables.
        
        Args:
            X (pd.DataFrame): Feature matrix
            max_interactions (int): Maximum number of interaction features
            
        Returns:
            pd.DataFrame: Feature matrix with interaction features
        """
        logger.info("Creating interaction features...")
        
        data = X.copy()
        
        # Select top features based on variance
        feature_variances = X.var().sort_values(ascending=False)
        top_features = feature_variances.head(min(10, len(X.columns))).index.tolist()
        
        interaction_count = 0
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                if interaction_count >= max_interactions:
                    break
                
                feat1, feat2 = top_features[i], top_features[j]
                interaction_name = f"{feat1}_x_{feat2}"
                
                # Create interaction feature
                data[interaction_name] = X[feat1] * X[feat2]
                interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features.copy()
