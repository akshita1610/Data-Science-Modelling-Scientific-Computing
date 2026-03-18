"""
Model Evaluator Module

Comprehensive evaluation and analysis of machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for housing price prediction.
    
    Features:
    - Cross-validation analysis
    - Learning curve analysis
    - Residual analysis
    - Prediction intervals
    - Error analysis
    - Model comparison
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}
        
    def perform_cross_validation(self, 
                                model: Any, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                cv_folds: int = 5,
                                scoring_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation analysis.
        
        Args:
            model: Trained model instance
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv_folds (int): Number of cross-validation folds
            scoring_metrics (List[str]): Metrics to compute
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        if scoring_metrics is None:
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
            
            # Convert negative scores to positive for error metrics
            if 'neg_' in metric:
                scores = -scores
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric
            
            cv_results[metric_name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        # Calculate RMSE from MSE
        if 'mean_squared_error' in cv_results:
            mse_scores = cv_results['mean_squared_error']['scores']
            cv_results['rmse'] = {
                'scores': np.sqrt(mse_scores),
                'mean': np.sqrt(mse_scores).mean(),
                'std': np.sqrt(mse_scores).std(),
                'min': np.sqrt(mse_scores).min(),
                'max': np.sqrt(mse_scores).max()
            }
        
        return cv_results
    
    def generate_learning_curve(self, 
                              model: Any, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              cv_folds: int = 5,
                              train_sizes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate learning curve analysis.
        
        Args:
            model: Model instance
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv_folds (int): Number of cross-validation folds
            train_sizes (np.ndarray): Training sizes to evaluate
            
        Returns:
            Dict: Learning curve data
        """
        logger.info("Generating learning curve...")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv_folds, train_sizes=train_sizes,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Convert to RMSE
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_rmse.mean(axis=1),
            'train_scores_std': train_rmse.std(axis=1),
            'val_scores_mean': val_rmse.mean(axis=1),
            'val_scores_std': val_rmse.std(axis=1),
            'train_scores_all': train_rmse,
            'val_scores_all': val_rmse
        }
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Residual analysis results
        """
        logger.info("Performing residual analysis...")
        
        residuals = y_true - y_pred
        
        # Basic statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }
        
        # Normality test (Shapiro-Wilk)
        if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limit
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            is_normal = shapiro_p > 0.05
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            shapiro_stat, shapiro_p = stats.kstest(residuals, 'norm')
            is_normal = shapiro_p > 0.05
        
        # Heteroscedasticity test (visual inspection would be better)
        # Here we'll compute correlation between predicted values and absolute residuals
        heteroscedasticity_corr = np.corrcoef(y_pred, np.abs(residuals))[0, 1]
        
        # Autocorrelation (Durbin-Watson statistic)
        n = len(residuals)
        diff_residuals = np.diff(residuals)
        durbin_watson = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        return {
            'residuals': residuals,
            'statistics': residual_stats,
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': is_normal
            },
            'heteroscedasticity': {
                'correlation': heteroscedasticity_corr,
                'is_heteroscedastic': abs(heteroscedasticity_corr) > 0.3
            },
            'autocorrelation': {
                'durbin_watson': durbin_watson,
                'is_autocorrelated': durbin_watson < 1.5 or durbin_watson > 2.5
            }
        }
    
    def calculate_prediction_intervals(self, 
                                      model: Any, 
                                      X: pd.DataFrame, 
                                      y: pd.Series,
                                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate prediction intervals for model predictions.
        
        Args:
            model: Trained model instance
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            confidence_level (float): Confidence level for intervals
            
        Returns:
            Dict: Prediction interval results
        """
        logger.info(f"Calculating {confidence_level*100}% prediction intervals...")
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        residual_std = np.std(residuals)
        
        # Calculate critical value for confidence level
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, len(residuals) - 1)
        
        # Calculate prediction intervals
        margin_of_error = t_critical * residual_std
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        return {
            'predictions': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_of_error': margin_of_error,
            'confidence_level': confidence_level,
            'coverage_percentage': np.mean((y >= lower_bound) & (y <= upper_bound)) * 100
        }
    
    def generate_evaluation_report(self, 
                                  model_name: str,
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  X: Optional[pd.DataFrame] = None,
                                  model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_name (str): Name of the model
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            X (pd.DataFrame, optional): Feature matrix
            model (Any, optional): Model instance
            
        Returns:
            Dict: Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate Adjusted R² if X is provided
        if X is not None:
            n, p = X.shape
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = r2
        
        # Max error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Basic metrics
        basic_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'mape': mape,
            'max_error': max_error
        }
        
        # Residual analysis
        residual_analysis = self.analyze_residuals(y_true, y_pred)
        
        # Error analysis
        errors = np.abs(y_true - y_pred)
        error_analysis = {
            'mean_absolute_error': np.mean(errors),
            'median_absolute_error': np.median(errors),
            'std_absolute_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'errors_25th_percentile': np.percentile(errors, 25),
            'errors_75th_percentile': np.percentile(errors, 75),
            'errors_90th_percentile': np.percentile(errors, 90),
            'errors_95th_percentile': np.percentile(errors, 95)
        }
        
        # Prediction intervals
        prediction_intervals = {}
        if model is not None and X is not None:
            prediction_intervals = self.calculate_prediction_intervals(model, X, y_true)
        
        return {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'residual_analysis': residual_analysis,
            'error_analysis': error_analysis,
            'prediction_intervals': prediction_intervals
        }
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their evaluation results.
        
        Args:
            model_results (Dict): Dictionary of model evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        logger.info("Comparing models...")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'basic_metrics' in results:
                metrics = results['basic_metrics']
                row = {
                    'Model': model_name.replace('_', ' ').title(),
                    'RMSE': metrics.get('rmse', np.nan),
                    'MAE': metrics.get('mae', np.nan),
                    'R²': metrics.get('r2', np.nan),
                    'Adj R²': metrics.get('adj_r2', np.nan),
                    'MAPE': metrics.get('mape', np.nan),
                    'Max Error': metrics.get('max_error', np.nan)
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df
    
    def rank_models(self, model_results: Dict[str, Dict[str, Any]], 
                   ranking_metric: str = 'rmse') -> List[Tuple[str, float]]:
        """
        Rank models based on a specific metric.
        
        Args:
            model_results (Dict): Dictionary of model evaluation results
            ranking_metric (str): Metric to rank by
            
        Returns:
            List: Ranked list of (model_name, metric_value) tuples
        """
        rankings = []
        
        for model_name, results in model_results.items():
            if 'basic_metrics' in results and ranking_metric in results['basic_metrics']:
                value = results['basic_metrics'][ranking_metric]
                rankings.append((model_name, value))
        
        # Sort based on metric (lower is better for error metrics, higher for R²)
        if ranking_metric in ['rmse', 'mae', 'mape', 'max_error']:
            rankings.sort(key=lambda x: x[1])
        else:  # R², Adj R²
            rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def get_model_summary(self, model_name: str, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a text summary of model evaluation results.
        
        Args:
            model_name (str): Name of the model
            evaluation_results (Dict): Evaluation results
            
        Returns:
            str: Summary text
        """
        if 'basic_metrics' not in evaluation_results:
            return f"No evaluation results available for {model_name}"
        
        metrics = evaluation_results['basic_metrics']
        
        summary = f"""
=== {model_name.upper()} EVALUATION SUMMARY ===

Performance Metrics:
- RMSE: ${metrics.get('rmse', 0):,.2f}
- MAE: ${metrics.get('mae', 0):,.2f}
- R²: {metrics.get('r2', 0):.4f}
- Adjusted R²: {metrics.get('adj_r2', 0):.4f}
- MAPE: {metrics.get('mape', 0):.2f}%
- Max Error: ${metrics.get('max_error', 0):,.2f}

Residual Analysis:
- Mean Residual: {evaluation_results.get('residual_analysis', {}).get('statistics', {}).get('mean', 0):.4f}
- Std Residual: {evaluation_results.get('residual_analysis', {}).get('statistics', {}).get('std', 0):.4f}
- Normality Test: {'Passed' if evaluation_results.get('residual_analysis', {}).get('normality_test', {}).get('is_normal', False) else 'Failed'}

Error Distribution:
- Median Error: ${evaluation_results.get('error_analysis', {}).get('median_absolute_error', 0):,.2f}
- 95th Percentile Error: ${evaluation_results.get('error_analysis', {}).get('errors_95th_percentile', 0):,.2f}
"""
        
        return summary
