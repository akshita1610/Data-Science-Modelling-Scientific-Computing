"""
Model Trainer Module

Handles training and evaluation of multiple regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model trainer for housing price prediction.
    
    Features:
    - Multiple regression algorithms
    - Hyperparameter tuning
    - Cross-validation
    - Model comparison
    - Model persistence
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.best_model_name = None
        self.model_results = {}
        self.training_history = []
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all regression models with default parameters.
        
        Returns:
            Dict: Dictionary of initialized models
        """
        logger.info("Initializing regression models...")
        
        self.models = {
            'linear_regression': LinearRegression(),
            'lasso': Lasso(random_state=42),
            'ridge': Ridge(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100)
        }
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self.models
    
    def train_single_model(self, 
                          model_name: str, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          hyperparameter_tuning: bool = False,
                          cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Training {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized")
        
        model = self.models[model_name]
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            best_params = self._tune_hyperparameters(model, X_train, y_train, cv_folds)
            model.set_params(**best_params)
        else:
            best_params = {}
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        result = {
            'model': model,
            'model_name': model_name,
            'best_params': best_params,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"Successfully trained {model_name}")
        return result
    
    def train_all_models(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        hyperparameter_tuning: bool = False,
                        cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results for all models
        """
        logger.info("Training all models...")
        
        if not self.models:
            self.initialize_models()
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, hyperparameter_tuning, cv_folds
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.model_results = results
        logger.info(f"Completed training {len(results)} models")
        
        return results
    
    def _tune_hyperparameters(self, 
                             model: Any, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             cv_folds: int) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            model: sklearn model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Best hyperparameters
        """
        param_grids = {
            'linear_regression': {},  # No hyperparameters to tune
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            },
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        model_name = type(model).__name__.lower()
        if 'linear' in model_name:
            model_name = 'linear_regression'
        elif 'lasso' in model_name:
            model_name = 'lasso'
        elif 'ridge' in model_name:
            model_name = 'ridge'
        elif 'randomforest' in model_name:
            model_name = 'random_forest'
        
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            return {}
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def evaluate_models(self, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Evaluation results
        """
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                evaluation_results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
                logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        # Find best model based on RMSE
        valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}
        if valid_results:
            best_model = min(valid_results.items(), key=lambda x: x[1]['rmse'])
            self.best_model_name = best_model[0]
            logger.info(f"Best model: {self.best_model_name} with RMSE: {best_model[1]['rmse']:.2f}")
        
        return evaluation_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison table of all models.
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        if not self.trained_models:
            return pd.DataFrame()
        
        # Get evaluation results
        comparison_data = []
        
        for model_name, result in self.model_results.items():
            if 'error' not in result:
                row = {
                    'Model': model_name.replace('_', ' ').title(),
                    'CV RMSE Mean': result.get('cv_rmse_mean', np.nan),
                    'CV RMSE Std': result.get('cv_rmse_std', np.nan)
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the trained model
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        return self.trained_models[model_name].predict(X)
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict[str, Any]:
        """
        Get feature importance for a model.
        
        Args:
            model_name (str): Name of the model
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict: Feature importance information
        """
        if model_name not in self.trained_models:
            return {'error': f'Model {model_name} not trained'}
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': feature_importance,
                'top_10_features': sorted_features[:10],
                'model_type': 'tree_based'
            }
        
        # For linear models, use coefficients
        elif hasattr(model, 'coef_'):
            coefficients = np.abs(model.coef_)
            feature_importance = dict(zip(feature_names, coefficients))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': feature_importance,
                'top_10_features': sorted_features[:10],
                'model_type': 'linear'
            }
        
        else:
            return {'error': 'Feature importance not available for this model type'}
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to the saved model
        """
        self.trained_models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns:
            Tuple: (model_name, model_instance)
        """
        if not self.best_model_name or self.best_model_name not in self.trained_models:
            raise ValueError("No trained models available")
        
        return self.best_model_name, self.trained_models[self.best_model_name]
