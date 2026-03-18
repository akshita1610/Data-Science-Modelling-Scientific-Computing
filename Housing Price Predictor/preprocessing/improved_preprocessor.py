"""
Improved Data Preprocessor Module

Enhanced preprocessing with advanced techniques for better performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import logging
from typing import Dict, Any, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedDataPreprocessor:
    """
    Enhanced data preprocessor with advanced preprocessing techniques.
    
    Features:
    - Outlier detection and removal
    - Robust scaling for better outlier handling
    - Mutual information feature selection
    - Frequency encoding for high-cardinality features
    - Smart missing value handling
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.frequency_encoders = {}
        self.feature_names = None
        self.target_column = None
        self.categorical_columns = []
        self.numeric_columns = []
        self.selected_features = []
        self.outlier_bounds = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str,
                    categorical_columns: Optional[List[str]] = None,
                    numeric_columns: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    scaling_method: str = 'robust',
                    feature_selection: bool = True,
                    k_best_features: int = 15,
                    outlier_removal: bool = True,
                    outlier_method: str = 'iqr') -> Dict[str, Any]:
        """
        Prepare data with enhanced preprocessing techniques.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable
            categorical_columns (List[str], optional): Categorical column names
            numeric_columns (List[str], optional): Numeric column names
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            scaling_method (str): 'standard', 'minmax', or 'robust'
            feature_selection (bool): Whether to perform feature selection
            k_best_features (int): Number of features to select
            outlier_removal (bool): Whether to remove outliers
            outlier_method (str): 'iqr' or 'zscore'
            
        Returns:
            Dict: Processed data including train/test splits
        """
        logger.info("Starting enhanced data preprocessing...")
        
        # Store target column
        self.target_column = target_column
        
        # Detect column types if not provided
        if categorical_columns is None or numeric_columns is None:
            categorical_columns, numeric_columns = self._detect_column_types(df)
        
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Step 1: Outlier removal
        if outlier_removal:
            data = self._remove_outliers(data, target_column, outlier_method)
        
        # Step 2: Handle missing values
        data = self._handle_missing_values(data)
        
        # Step 3: Encode categorical variables
        data = self._encode_categorical_variables(data, categorical_columns)
        
        # Step 4: Feature selection
        if feature_selection:
            data = self._select_features(data, target_column, k_best_features)
        
        # Step 5: Scale numeric variables
        data = self._scale_numeric_variables(data, numeric_columns, scaling_method)
        
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Enhanced preprocessing completed!")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'categorical_columns': categorical_columns,
            'numeric_columns': numeric_columns,
            'scaler_type': scaling_method,
            'selected_features': self.selected_features,
            'outliers_removed': outlier_removal
        }
    
    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect categorical and numeric columns."""
        categorical_columns = []
        numeric_columns = []
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if df[col].dtype == 'object':
                categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_columns.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col])
                    numeric_columns.append(col)
                except:
                    categorical_columns.append(col)
        
        return categorical_columns, numeric_columns
    
    def _remove_outliers(self, df: pd.DataFrame, target_column: str, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the target variable."""
        logger.info(f"Removing outliers using {method} method...")
        
        original_size = len(df)
        
        if method == 'iqr':
            Q1 = df[target_column].quantile(0.25)
            Q3 = df[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_bounds = {'lower': lower_bound, 'upper': upper_bound}
            df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[target_column] - df[target_column].mean()) / df[target_column].std())
            df = df[z_scores < 3]
        
        removed_count = original_size - len(df)
        logger.info(f"Removed {removed_count} outliers ({removed_count/original_size*100:.1f}% of data)")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with smart imputation."""
        logger.info("Handling missing values...")
        
        # For numeric columns, use median imputation
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        # For categorical columns, use mode imputation
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables with frequency encoding for high-cardinality features."""
        logger.info("Encoding categorical variables...")
        
        for col in categorical_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                
                if unique_count > 10:  # High cardinality - use frequency encoding
                    freq_map = df[col].value_counts().to_dict()
                    df[col + '_freq'] = df[col].map(freq_map)
                    self.frequency_encoders[col] = freq_map
                    
                    # Drop original column
                    df = df.drop(columns=[col])
                    
                    # Update column name
                    if col in self.categorical_columns:
                        self.categorical_columns.remove(col)
                        self.categorical_columns.append(col + '_freq')
                    
                    logger.info(f"Frequency encoded {col} with {unique_count} unique values")
                    
                else:  # Low cardinality - use label encoding
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[col] = encoder
                    
                    logger.info(f"Label encoded {col} with {unique_count} unique values")
        
        return df
    
    def _select_features(self, df: pd.DataFrame, target_column: str, k_best: int) -> pd.DataFrame:
        """Select features using mutual information."""
        logger.info(f"Selecting top {k_best} features using mutual information...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance dictionary
        feature_scores = dict(zip(X.columns, mi_scores))
        
        # Select top k features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feature for feature, score in sorted_features[:k_best]]
        
        # Keep only selected features
        selected_columns = self.selected_features + [target_column]
        df = df[selected_columns]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        # Show top features
        top_features = sorted_features[:10]
        logger.info("Top 10 features by mutual information:")
        for i, (feature, score) in enumerate(top_features, 1):
            logger.info(f"  {i:2d}. {feature:<25} {score:.4f}")
        
        return df
    
    def _scale_numeric_variables(self, df: pd.DataFrame, numeric_columns: List[str], scaling_method: str) -> pd.DataFrame:
        """Scale numeric variables using robust scaling."""
        logger.info(f"Scaling {len(numeric_columns)} numeric features using {scaling_method} scaling")
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
        
        # Get numeric columns that exist in the dataframe
        available_numeric = [col for col in numeric_columns if col in df.columns]
        
        if available_numeric:
            df[available_numeric] = scaler.fit_transform(df[available_numeric])
            self.scalers['numeric'] = scaler
        
        return df
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders and scalers.
        
        Args:
            df (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.feature_names or not self.encoders:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        # Create a copy
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col in data.columns:
                # Handle unknown categories
                unique_values = set(data[col].unique())
                known_values = set(encoder.classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    logger.warning(f"Unknown categories in {col}: {unknown_values}")
                    # Map unknown values to most common class
                    most_common = encoder.classes_[0]
                    data[col] = data[col].apply(lambda x: most_common if x not in known_values else x)
                
                data[col] = encoder.transform(data[col].astype(str))
        
        # Apply frequency encoding
        for col, freq_map in self.frequency_encoders.items():
            if col in data.columns:
                data[col + '_freq'] = data[col].map(freq_map).fillna(0)
                data = data.drop(columns=[col])
        
        # Scale numeric variables
        if 'numeric' in self.scalers:
            available_numeric = [col for col in self.numeric_columns if col in data.columns]
            if available_numeric:
                data[available_numeric] = self.scalers['numeric'].transform(data[available_numeric])
        
        # Ensure all selected features are present
        for col in self.selected_features:
            if col not in data.columns:
                data[col] = 0  # Default value for missing columns
        
        # Reorder columns to match training data
        data = data[self.selected_features]
        
        return data
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps performed."""
        return {
            'target_column': self.target_column,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'outlier_bounds': self.outlier_bounds,
            'frequency_encoders': list(self.frequency_encoders.keys()),
            'label_encoders': list(self.encoders.keys())
        }
