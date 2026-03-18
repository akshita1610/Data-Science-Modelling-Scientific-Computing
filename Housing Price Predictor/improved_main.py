"""
Improved Housing Price Predictor - Enhanced Main Application

Enhanced version with advanced preprocessing and modeling techniques.
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataLoader
from preprocessing import ImprovedDataPreprocessor, FeatureEngineer
from models import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedHousingPricePredictor:
    """
    Enhanced Housing Price Predictor with advanced features.
    
    This class provides improved performance through:
    - Advanced preprocessing with outlier removal
    - Enhanced feature selection using mutual information
    - Multiple regression algorithms
    - Comprehensive evaluation metrics
    - Performance optimization
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = ImprovedDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        self.processed_data = None
        self.feature_engineered_data = None
        self.training_results = None
        
    def load_and_preprocess_data(self, 
                                file_path: str, 
                                dataset_type: str = 'generic',
                                target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and preprocess data with enhanced techniques.
        
        Args:
            file_path (str): Path to the data file
            dataset_type (str): Type of dataset
            target_column (str, optional): Name of the target column
            
        Returns:
            Dict: Processed data
        """
        logger.info(f"Loading data from {file_path}")
        
        # Load data
        data = self.data_loader.load_csv(file_path, dataset_type)
        
        # Auto-detect target column if not provided
        if target_column is None:
            target_column = self._detect_target_column(data, dataset_type)
        
        logger.info(f"Using '{target_column}' as target variable")
        
        # Enhanced feature engineering
        logger.info("Performing enhanced feature engineering...")
        engineered_data = self.feature_engineer.create_housing_features(data, dataset_type)
        self.feature_engineered_data = engineered_data
        
        # Enhanced preprocessing
        logger.info("Preprocessing data with optimizations...")
        processed_data = self.preprocessor.prepare_data(
            engineered_data, 
            target_column=target_column,
            scaling_method='robust',  # Use robust scaling
            feature_selection=True,     # Enable feature selection
            k_best_features=15,         # Select top 15 features
            outlier_removal=True,        # Remove outliers
            outlier_method='iqr'         # Use IQR method
        )
        self.processed_data = processed_data
        
        logger.info("Data preprocessing completed!")
        return processed_data
    
    def _detect_target_column(self, data: pd.DataFrame, dataset_type: str) -> str:
        """Auto-detect the target column based on dataset type."""
        if dataset_type.lower() == 'hdb':
            return 'resale_price'
        elif dataset_type.lower() == 'portland':
            return 'price'
        else:
            # Look for common price-related columns
            price_columns = [col for col in data.columns if 'price' in col.lower()]
            if price_columns:
                return price_columns[0]
            else:
                raise ValueError("Could not detect target column. Please specify target_column parameter.")
    
    def train_models(self, hyperparameter_tuning: bool = False, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train models with enhanced algorithms and evaluation.
        
        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        logger.info("Training enhanced models...")
        
        # Train all models
        training_results = self.model_trainer.train_all_models(
            self.processed_data['X_train'],
            self.processed_data['y_train'],
            hyperparameter_tuning=hyperparameter_tuning,
            cv_folds=cv_folds
        )
        
        # Enhanced evaluation
        logger.info("Evaluating models...")
        evaluation_results = self.model_trainer.evaluate_models(
            self.processed_data['X_test'],
            self.processed_data['y_test']
        )
        
        # Generate comprehensive evaluation reports
        enhanced_results = {}
        for model_name, eval_data in evaluation_results.items():
            if 'error' not in eval_data:
                enhanced_results[model_name] = self.evaluator.generate_evaluation_report(
                    model_name, 
                    eval_data['actual'], 
                    eval_data['predictions'],
                    self.processed_data['X_test'],
                    self.model_trainer.trained_models.get(model_name)
                )
        
        # Get model comparison
        comparison = self.evaluator.compare_models(enhanced_results)
        
        # Store results
        self.training_results = {
            'training_results': training_results,
            'evaluation_results': enhanced_results,
            'model_comparison': comparison,
            'best_model': self.model_trainer.best_model_name
        }
        
        # Print results
        self._print_enhanced_results(comparison)
        
        return self.training_results
    
    def _print_enhanced_results(self, comparison: pd.DataFrame) -> None:
        """Print enhanced model results."""
        print("\nEnhanced Model Performance Comparison:")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("=" * 80)
        
        if self.model_trainer.best_model_name:
            best_row = comparison[comparison['Model'] == self.model_trainer.best_model_name.replace('_', ' ').title()]
            if not best_row.empty:
                print(f"\nBest model: {self.model_trainer.best_model_name}")
                print(f"RMSE: ${best_row['RMSE'].iloc[0]:.2f}")
                print(f"Adjusted R²: {best_row['Adj R²'].iloc[0]:.4f}")
    
    def predict_price(self, model_name: str, features: Dict[str, Any]) -> float:
        """
        Make price prediction with smart feature completion.
        
        Args:
            model_name (str): Name of the trained model
            features (Dict): Feature values for prediction
            
        Returns:
            float: Predicted price
        """
        if self.processed_data is None:
            raise ValueError("Models must be trained first")
        
        # Create DataFrame from features
        feature_df = pd.DataFrame([features])
        
        # Smart feature completion
        completed_features = self._complete_features(feature_df)
        
        # Transform features using preprocessor
        try:
            transformed_features = self.preprocessor.transform_new_data(completed_features)
        except Exception as e:
            logger.warning(f"Error transforming features: {e}")
            # Fallback: ensure all required columns are present
            required_columns = self.processed_data['feature_names']
            for col in required_columns:
                if col not in transformed_features.columns:
                    transformed_features[col] = 0
            transformed_features = transformed_features[required_columns]
        
        # Make prediction
        prediction = self.model_trainer.predict(model_name, transformed_features)
        
        return float(prediction[0])
    
    def _complete_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Complete missing features with smart defaults."""
        # Add common features that might be missing
        if 'price_per_sqm' not in feature_df.columns and 'resale_price' in feature_df.columns and 'floor_area_sqm' in feature_df.columns:
            feature_df['price_per_sqm'] = feature_df['resale_price'] / feature_df['floor_area_sqm']
        
        if 'property_age_at_sale' not in feature_df.columns and 'lease_commence_date' in feature_df.columns:
            feature_df['property_age_at_sale'] = 2024 - feature_df['lease_commence_date']
        
        # Fill missing numeric features with median values (if we have training data)
        if self.processed_data and hasattr(self, '_feature_medians'):
            for col in self.processed_data['feature_names']:
                if col not in feature_df.columns:
                    feature_df[col] = self._feature_medians.get(col, 0)
        
        return feature_df
    
    def generate_enhanced_visualizations(self, save_plots: bool = True, output_dir: str = "improved_visualizations") -> Dict[str, str]:
        """
        Generate enhanced visualizations.
        
        Args:
            save_plots (bool): Whether to save plots to files
            output_dir (str): Directory to save plots
            
        Returns:
            Dict: Paths to saved plots
        """
        if self.processed_data is None:
            raise ValueError("Data must be processed first")
        
        logger.info("Generating enhanced visualizations...")
        
        # Create output directory
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_paths = {}
        
        # Enhanced price distribution
        if 'y_train' in self.processed_data:
            fig = self.visualizer.plot_price_distribution(
                self.processed_data['y_train'], 
                "Enhanced Price Distribution Analysis"
            )
            if save_plots:
                path = os.path.join(output_dir, "enhanced_price_distribution.png")
                self.visualizer.save_plot(fig, path)
                plot_paths['enhanced_price_distribution'] = path
            else:
                plot_paths['enhanced_price_distribution'] = "Displayed"
        
        # Enhanced feature importance
        if self.model_trainer.best_model_name:
            feature_importance = self.model_trainer.get_feature_importance(
                self.model_trainer.best_model_name,
                self.processed_data['feature_names']
            )
            if 'error' not in feature_importance:
                fig = self.visualizer.plot_feature_importance(
                    feature_importance['feature_importance'],
                    f"Enhanced Feature Importance - {self.model_trainer.best_model_name}"
                )
                if save_plots:
                    path = os.path.join(output_dir, "enhanced_feature_importance.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['enhanced_feature_importance'] = path
                else:
                    plot_paths['enhanced_feature_importance'] = "Displayed"
        
        # Enhanced model comparison
        if self.training_results and 'model_comparison' in self.training_results:
            comparison = self.training_results['model_comparison']
            if not comparison.empty:
                fig = self.visualizer.plot_model_comparison(comparison, 'RMSE')
                if save_plots:
                    path = os.path.join(output_dir, "enhanced_model_comparison.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['enhanced_model_comparison'] = path
                else:
                    plot_paths['enhanced_model_comparison'] = "Displayed"
        
        # Comprehensive dashboard
        if self.training_results and 'evaluation_results' in self.training_results:
            best_model_results = self.training_results['evaluation_results'].get(self.model_trainer.best_model_name, {})
            if 'error' not in best_model_results:
                dashboard_data = {
                    'y_true': best_model_results.get('actual', []),
                    'y_pred': best_model_results.get('predictions', []),
                    'basic_metrics': best_model_results.get('basic_metrics', {}),
                    'error_analysis': best_model_results.get('error_analysis', {})
                }
                
                feature_importance_data = None
                if 'error' not in feature_importance:
                    feature_importance_data = feature_importance['feature_importance']
                
                fig = self.visualizer.create_comprehensive_dashboard(dashboard_data, feature_importance_data)
                if save_plots:
                    path = os.path.join(output_dir, "comprehensive_dashboard.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['comprehensive_dashboard'] = path
                else:
                    plot_paths['comprehensive_dashboard'] = "Displayed"
        
        logger.info(f"Generated {len(plot_paths)} enhanced visualizations")
        return plot_paths
    
    def run_enhanced_pipeline(self, 
                            file_path: str, 
                            dataset_type: str = 'generic',
                            target_column: Optional[str] = None,
                            hyperparameter_tuning: bool = False,
                            generate_viz: bool = True) -> Dict[str, Any]:
        """
        Run the complete enhanced pipeline.
        
        Args:
            file_path (str): Path to dataset
            dataset_type (str): Type of dataset
            target_column (str, optional): Target column name
            hyperparameter_tuning (bool): Whether to tune hyperparameters
            generate_viz (bool): Whether to generate visualizations
            
        Returns:
            Dict: Complete pipeline results
        """
        print("=" * 80)
        print("ENHANCED HOUSING PRICE PREDICTOR")
        print("=" * 80)
        
        total_start_time = time.time()
        
        # Step 1: Load and preprocess data
        print("\n=== Step 1: Enhanced Data Loading & Preprocessing ===")
        start_time = time.time()
        processed_data = self.load_and_preprocess_data(file_path, dataset_type, target_column)
        load_time = time.time() - start_time
        print(f"Data loading completed in {load_time:.2f} seconds")
        print(f"Training set shape: {processed_data['X_train'].shape}")
        print(f"Test set shape: {processed_data['X_test'].shape}")
        
        # Step 2: Train enhanced models
        print("\n=== Step 2: Enhanced Model Training ===")
        start_time = time.time()
        results = self.train_models(hyperparameter_tuning)
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Step 3: Generate visualizations
        if generate_viz:
            print("\n=== Step 3: Enhanced Visualization Generation ===")
            start_time = time.time()
            plot_paths = self.generate_enhanced_visualizations()
            viz_time = time.time() - start_time
            print(f"Visualizations generated in {viz_time:.2f} seconds")
        else:
            plot_paths = {}
        
        total_time = time.time() - total_start_time
        
        # Final summary
        print("\n" + "=" * 80)
        print("=== Enhanced Pipeline Complete ===")
        print(f"Best model: {results['best_model']}")
        print(f"RMSE: ${results['model_comparison'].iloc[0]['RMSE']:.2f}")
        print(f"Adjusted R²: {results['model_comparison'].iloc[0]['Adj R²']:.4f}")
        print(f"Total time: {total_time:.2f} seconds")
        print("=" * 80)
        
        return {
            'best_model': results['best_model'],
            'model_results': results,
            'plot_paths': plot_paths,
            'total_time': total_time,
            'preprocessing_summary': self.preprocessor.get_preprocessing_summary()
        }


def main():
    """Main function for enhanced command-line interface."""
    parser = argparse.ArgumentParser(description="Enhanced Housing Price Predictor")
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    
    # Optional arguments
    parser.add_argument('--type', type=str, default='generic', 
                       choices=['hdb', 'portland', 'generic'],
                       help='Dataset type')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--tuning', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Initialize enhanced predictor
    predictor = ImprovedHousingPricePredictor()
    
    try:
        # Run enhanced pipeline
        results = predictor.run_enhanced_pipeline(
            file_path=args.data,
            dataset_type=args.type,
            target_column=args.target,
            hyperparameter_tuning=args.tuning,
            generate_viz=not args.no_viz
        )
        
        logger.info("Enhanced pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
