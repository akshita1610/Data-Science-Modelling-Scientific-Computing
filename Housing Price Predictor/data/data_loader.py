"""
Data Loader Module

Handles loading and cleaning of housing datasets from various sources.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A comprehensive data loader for housing datasets with automatic cleaning and preprocessing.
    
    Supports multiple dataset types:
    - Singapore HDB resale data
    - Portland housing data
    - Generic CSV datasets
    """
    
    def __init__(self):
        self.data = None
        self.dataset_type = None
        self.original_shape = None
        self.cleaned_shape = None
        
    def load_csv(self, file_path: str, dataset_type: str = 'generic') -> pd.DataFrame:
        """
        Load CSV file with automatic dataset-specific cleaning.
        
        Args:
            file_path (str): Path to the CSV file
            dataset_type (str): Type of dataset ('hdb', 'portland', 'generic')
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            # Load the data
            self.data = pd.read_csv(file_path)
            self.dataset_type = dataset_type
            self.original_shape = self.data.shape
            
            logger.info(f"Original dataset shape: {self.original_shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            # Remove duplicates
            self.data = self.data.drop_duplicates()
            
            # Dataset-specific cleaning
            if dataset_type.lower() == 'hdb':
                self.data = self._clean_hdb_data(self.data)
            elif dataset_type.lower() == 'portland':
                self.data = self._clean_portland_data(self.data)
            else:
                self.data = self._clean_generic_data(self.data)
            
            self.cleaned_shape = self.data.shape
            logger.info(f"Cleaned dataset shape: {self.cleaned_shape}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_hdb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Singapore HDB resale data."""
        logger.info("Cleaning HDB dataset...")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Extract year and month from date column if present
        if 'month' in df.columns:
            df['year'] = pd.to_datetime(df['month'], format='%Y-%m').dt.year
            df['month_num'] = pd.to_datetime(df['month'], format='%Y-%m').dt.month
        
        return df
    
    def _clean_portland_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Portland housing data."""
        logger.info("Cleaning Portland dataset...")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _clean_generic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean generic CSV data."""
        logger.info("Cleaning generic dataset...")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded dataset.
        
        Returns:
            Dict: Dataset information including shape, columns, types, etc.
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            'dataset_type': self.dataset_type,
            'shape': self.data.shape,
            'original_shape': self.original_shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
        }
        
        return info
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for numeric columns.
        
        Returns:
            Dict: Summary statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        numeric_stats = self.data.describe().to_dict()
        
        return {
            'numeric_statistics': numeric_stats,
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
        }
    
    def preview_data(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview the first n rows of the dataset.
        
        Args:
            n_rows (int): Number of rows to preview
            
        Returns:
            pd.DataFrame: Preview of the data
        """
        if self.data is None:
            return pd.DataFrame()
        
        return self.data.head(n_rows)
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data for common issues.
        
        Returns:
            Dict: Validation results
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        validation_results = {
            'has_duplicates': self.data.duplicated().any(),
            'duplicate_count': self.data.duplicated().sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'total_missing': self.data.isnull().sum().sum(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'constant_columns': [],
            'high_cardinality_columns': [],
        }
        
        # Check for constant columns
        for col in self.data.columns:
            if self.data[col].nunique() == 1:
                validation_results['constant_columns'].append(col)
        
        # Check for high cardinality categorical columns
        for col in validation_results['categorical_columns']:
            if self.data[col].nunique() > 50:
                validation_results['high_cardinality_columns'].append(col)
        
        return validation_results
