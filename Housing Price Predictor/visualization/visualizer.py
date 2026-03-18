"""
Visualizer Module

Comprehensive visualization tools for housing price prediction analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualizer for housing price prediction.
    
    Features:
    - Price distribution analysis
    - Correlation heatmaps
    - Feature importance plots
    - Actual vs predicted comparisons
    - Residual analysis plots
    - Model comparison charts
    - Learning curves
    - Prediction intervals
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_price_distribution(self, prices: pd.Series, title: str = "Price Distribution") -> plt.Figure:
        """
        Create comprehensive price distribution analysis.
        
        Args:
            prices (pd.Series): Price data
            title (str): Plot title
            
        Returns:
            plt.Figure: Distribution plots
        """
        logger.info("Creating price distribution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(prices, bins=30, alpha=0.7, color=self.color_palette[0], edgecolor='black')
        axes[0, 0].set_title('Price Histogram')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(prices.mean(), color='red', linestyle='--', label=f'Mean: ${prices.mean():,.0f}')
        axes[0, 0].axvline(prices.median(), color='green', linestyle='--', label=f'Median: ${prices.median():,.0f}')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(prices)
        axes[0, 1].set_title('Price Box Plot')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(prices, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics table
        stats_data = {
            'Mean': [f"${prices.mean():,.2f}"],
            'Median': [f"${prices.median():,.2f}"],
            'Std Dev': [f"${prices.std():,.2f}"],
            'Min': [f"${prices.min():,.2f}"],
            'Max': [f"${prices.max():,.2f}"],
            'Skewness': [f"{prices.skew():.3f}"],
            'Kurtosis': [f"{prices.kurtosis():.3f}"]
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=list(stats_data.values()),
                                colLabels=list(stats_data.keys()),
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Descriptive Statistics', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, title: str = "Feature Correlation Heatmap") -> plt.Figure:
        """
        Create correlation heatmap of features.
        
        Args:
            df (pd.DataFrame): Feature matrix
            title (str): Plot title
            
        Returns:
            plt.Figure: Correlation heatmap
        """
        logger.info("Creating correlation heatmap...")
        
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                               title: str = "Feature Importance") -> plt.Figure:
        """
        Create feature importance plot.
        
        Args:
            feature_importance (Dict): Feature importance scores
            title (str): Plot title
            
        Returns:
            plt.Figure: Feature importance plot
        """
        logger.info("Creating feature importance plot...")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:15])  # Top 15 features
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importance, color=self.color_palette[0])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, importance)):
            ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str = "Model") -> plt.Figure:
        """
        Create actual vs predicted comparison plot.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for title
            
        Returns:
            plt.Figure: Comparison plot
        """
        logger.info("Creating actual vs predicted plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name} - Actual vs Predicted', fontsize=16, fontweight='bold')
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, color=self.color_palette[0])
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price ($)')
        axes[0].set_ylabel('Predicted Price ($)')
        axes[0].set_title('Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[1])
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Price ($)')
        axes[1].set_ylabel('Residuals ($)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model") -> plt.Figure:
        """
        Create comprehensive residual analysis plots.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for title
            
        Returns:
            plt.Figure: Residual analysis plots
        """
        logger.info("Creating residual analysis plots...")
        
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[0])
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=self.color_palette[1], edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].axvline(0, color='r', linestyle='--', label='Zero')
        axes[1, 0].axvline(residuals.mean(), color='g', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scale-Location plot
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), 
                          alpha=0.6, color=self.color_palette[2])
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('√|Standardized Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_comparison: pd.DataFrame, 
                            metric: str = 'RMSE') -> plt.Figure:
        """
        Create model comparison charts.
        
        Args:
            model_comparison (pd.DataFrame): Model comparison data
            metric (str): Primary metric to compare
            
        Returns:
            plt.Figure: Comparison charts
        """
        logger.info("Creating model comparison plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Bar chart of primary metric
        models = model_comparison['Model']
        values = model_comparison[metric]
        
        bars = axes[0].bar(models, values, color=self.color_palette[:len(models)])
        axes[0].set_title(f'{metric} Comparison')
        axes[0].set_ylabel(metric)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,.0f}', ha='center', va='bottom')
        
        # Multiple metrics comparison
        metrics_to_plot = ['RMSE', 'MAE', 'R²']
        available_metrics = [m for m in metrics_to_plot if m in model_comparison.columns]
        
        if len(available_metrics) > 1:
            x = np.arange(len(models))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = model_comparison[metric]
                axes[1].bar(x + i*width, values, width, label=metric, 
                           color=self.color_palette[i])
            
            axes[1].set_title('Multiple Metrics Comparison')
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score')
            axes[1].set_xticks(x + width * (len(available_metrics) - 1) / 2)
            axes[1].set_xticklabels(models, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Not enough metrics\nfor comparison', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Multiple Metrics Comparison')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, learning_curve_data: Dict[str, Any], 
                          model_name: str = "Model") -> plt.Figure:
        """
        Create learning curve visualization.
        
        Args:
            learning_curve_data (Dict): Learning curve data
            model_name (str): Model name for title
            
        Returns:
            plt.Figure: Learning curve plot
        """
        logger.info("Creating learning curve plot...")
        
        train_sizes = learning_curve_data['train_sizes']
        train_scores_mean = learning_curve_data['train_scores_mean']
        train_scores_std = learning_curve_data['train_scores_std']
        val_scores_mean = learning_curve_data['val_scores_mean']
        val_scores_std = learning_curve_data['val_scores_std']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color=self.color_palette[0], 
               label='Training Score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, 
                        color=self.color_palette[0])
        
        # Plot validation scores
        ax.plot(train_sizes, val_scores_mean, 'o-', color=self.color_palette[1], 
               label='Validation Score')
        ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, 
                        color=self.color_palette[1])
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{model_name} - Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                lower_bound: np.ndarray, upper_bound: np.ndarray,
                                model_name: str = "Model") -> plt.Figure:
        """
        Create prediction interval visualization.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            lower_bound (np.ndarray): Lower bounds of intervals
            upper_bound (np.ndarray): Upper bounds of intervals
            model_name (str): Model name for title
            
        Returns:
            plt.Figure: Prediction interval plot
        """
        logger.info("Creating prediction interval plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by predicted values for better visualization
        sort_idx = np.argsort(y_pred)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        lower_sorted = lower_bound[sort_idx]
        upper_sorted = upper_bound[sort_idx]
        
        x = range(len(y_true_sorted))
        
        # Plot prediction intervals
        ax.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, 
                       color=self.color_palette[0], label='95% Prediction Interval')
        
        # Plot predictions
        ax.plot(x, y_pred_sorted, 'o-', color=self.color_palette[1], 
               label='Predictions', markersize=4)
        
        # Plot actual values
        ax.plot(x, y_true_sorted, 'o-', color=self.color_palette[2], 
               label='Actual Values', markersize=4)
        
        ax.set_xlabel('Sample Index (sorted by prediction)')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{model_name} - Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, model_results: Dict[str, Any], 
                                    feature_importance: Optional[Dict] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            model_results (Dict): Model evaluation results
            feature_importance (Dict, optional): Feature importance scores
            
        Returns:
            plt.Figure: Comprehensive dashboard
        """
        logger.info("Creating comprehensive dashboard...")
        
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Price distribution (top left)
        if 'y_true' in model_results and 'y_pred' in model_results:
            ax1 = plt.subplot(3, 3, 1)
            prices = np.concatenate([model_results['y_true'], model_results['y_pred']])
            ax1.hist(prices, bins=30, alpha=0.7, color=self.color_palette[0])
            ax1.set_title('Price Distribution')
            ax1.set_xlabel('Price ($)')
            ax1.set_ylabel('Frequency')
        
        # Actual vs Predicted (top middle)
        if 'y_true' in model_results and 'y_pred' in model_results:
            ax2 = plt.subplot(3, 3, 2)
            y_true, y_pred = model_results['y_true'], model_results['y_pred']
            ax2.scatter(y_true, y_pred, alpha=0.6, color=self.color_palette[1])
            ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax2.set_title('Actual vs Predicted')
            ax2.set_xlabel('Actual Price ($)')
            ax2.set_ylabel('Predicted Price ($)')
        
        # Residuals (top right)
        if 'y_true' in model_results and 'y_pred' in model_results:
            ax3 = plt.subplot(3, 3, 3)
            residuals = y_true - y_pred
            ax3.scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[2])
            ax3.axhline(y=0, color='r', linestyle='--')
            ax3.set_title('Residual Plot')
            ax3.set_xlabel('Predicted Price ($)')
            ax3.set_ylabel('Residuals ($)')
        
        # Feature importance (middle row)
        if feature_importance:
            ax4 = plt.subplot(3, 3, 4)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*sorted_features)
            ax4.barh(range(len(features)), importance, color=self.color_palette[3])
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 10 Feature Importance')
        
        # Metrics table (middle center)
        ax5 = plt.subplot(3, 3, 5)
        ax5.axis('tight')
        ax5.axis('off')
        
        if 'basic_metrics' in model_results:
            metrics = model_results['basic_metrics']
            metrics_text = f"""
Performance Metrics:
RMSE: ${metrics.get('rmse', 0):,.2f}
MAE: ${metrics.get('mae', 0):,.2f}
R²: {metrics.get('r2', 0):.4f}
Adj R²: {metrics.get('adj_r2', 0):.4f}
MAPE: {metrics.get('mape', 0):.2f}%
Max Error: ${metrics.get('max_error', 0):,.2f}
"""
            ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes, 
                    fontsize=10, verticalalignment='center')
            ax5.set_title('Model Performance', pad=20)
        
        # Residual distribution (middle right)
        if 'y_true' in model_results and 'y_pred' in model_results:
            ax6 = plt.subplot(3, 3, 6)
            ax6.hist(residuals, bins=20, alpha=0.7, color=self.color_palette[4], edgecolor='black')
            ax6.set_title('Residual Distribution')
            ax6.set_xlabel('Residuals ($)')
            ax6.set_ylabel('Frequency')
            ax6.axvline(0, color='r', linestyle='--')
        
        # Learning curve (bottom left) - placeholder
        ax7 = plt.subplot(3, 3, 7)
        ax7.text(0.5, 0.5, 'Learning Curve\n(Provide data to visualize)', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Learning Curve')
        
        # Model comparison (bottom middle) - placeholder
        ax8 = plt.subplot(3, 3, 8)
        ax8.text(0.5, 0.5, 'Model Comparison\n(Provide multiple models)', 
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Model Comparison')
        
        # Summary statistics (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        if 'error_analysis' in model_results:
            error_stats = model_results['error_analysis']
            stats_text = f"""
Error Analysis:
Mean Error: ${error_stats.get('mean_absolute_error', 0):,.2f}
Median Error: ${error_stats.get('median_absolute_error', 0):,.2f}
Std Error: ${error_stats.get('std_absolute_error', 0):,.2f}
95th Percentile: ${error_stats.get('errors_95th_percentile', 0):,.2f}
"""
            ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='center')
            ax9.set_title('Error Statistics', pad=20)
        
        plt.suptitle('Housing Price Predictor - Comprehensive Dashboard', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """
        Save plot to file.
        
        Args:
            fig (plt.Figure): Figure to save
            filepath (str): File path
            dpi (int): Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        plt.close(fig)
