"""
Data Preprocessor Module

Handles data preprocessing including encoding, scaling, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessor for housing price prediction.
    
    Features:
    - Automatic column type detection
    - Missing value handling
    - Categorical encoding
    - Numeric scaling
    - Train-test splitting
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.target_column = None
        self.categorical_columns = []
        self.numeric_columns = []
        
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str,
                    categorical_columns: Optional[List[str]] = None,
                    numeric_columns: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    scaling_method: str = 'standard') -> Dict[str, Any]:
        """
        Prepare data for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable
            categorical_columns (List[str], optional): Categorical column names
            numeric_columns (List[str], optional): Numeric column names
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            scaling_method (str): 'standard' or 'minmax'
            
        Returns:
            Dict: Processed data including train/test splits
        """
        logger.info("Starting data preprocessing...")
        
        # Store target column
        self.target_column = target_column
        
        # Detect column types if not provided
        if categorical_columns is None or numeric_columns is None:
            categorical_columns, numeric_columns = self._detect_column_types(df)
        
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Encode categorical variables
        data = self._encode_categorical_variables(data, categorical_columns)
        
        # Scale numeric variables
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
        
        logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
        logger.info(f"Features: {len(self.feature_names)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'categorical_columns': categorical_columns,
            'numeric_columns': numeric_columns,
            'scaler_type': scaling_method
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
        
        logger.info(f"Detected {len(categorical_columns)} categorical features")
        logger.info(f"Detected {len(numeric_columns)} numeric features")
        
        return categorical_columns, numeric_columns
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # For numeric columns, fill with median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding."""
        logger.info("Encoding categorical variables...")
        
        for col in categorical_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
                
                logger.info(f"Encoded {col} with {len(encoder.classes_)} unique values")
        
        return df
    
    def _scale_numeric_variables(self, df: pd.DataFrame, numeric_columns: List[str], scaling_method: str) -> pd.DataFrame:
        """Scale numeric variables."""
        logger.info(f"Scaled {len(numeric_columns)} numeric features using {scaling_method} scaling")
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        
        if numeric_columns:
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
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
        
        # Scale numeric variables
        if 'numeric' in self.scalers and self.numeric_columns:
            available_numeric = [col for col in self.numeric_columns if col in data.columns]
            if available_numeric:
                data[available_numeric] = self.scalers['numeric'].transform(data[available_numeric])
        
        # Ensure all expected columns are present
        for col in self.feature_names:
            if col not in data.columns:
                data[col] = 0  # Default value for missing columns
        
        # Reorder columns to match training data
        data = data[self.feature_names]
        
        return data
    
    def get_feature_importance_names(self) -> List[str]:
        """Get the names of features after preprocessing."""
        return self.feature_names.copy() if self.feature_names else []
    
    def get_categorical_mappings(self) -> Dict[str, Dict]:
        """Get mappings for categorical encoders."""
        mappings = {}
        for col, encoder in self.encoders.items():
            mappings[col] = {label: idx for idx, label in enumerate(encoder.classes_)}
        return mappings
