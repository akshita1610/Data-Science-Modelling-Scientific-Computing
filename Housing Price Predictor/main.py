"""
Housing Price Predictor - Main Application

A comprehensive machine learning system for predicting housing prices.
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataLoader
from preprocessing import DataPreprocessor, FeatureEngineer
from models import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingPricePredictor:
    """
    Main class for the Housing Price Predictor system.
    
    This class orchestrates the complete machine learning pipeline from
    data loading to model training and prediction.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        self.processed_data = None
        self.feature_engineered_data = None
        
    def load_and_preprocess_data(self, 
                                file_path: str, 
                                dataset_type: str = 'generic',
                                target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and preprocess housing data.
        
        Args:
            file_path (str): Path to the data file
            dataset_type (str): Type of dataset ('hdb', 'portland', 'generic')
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
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        engineered_data = self.feature_engineer.create_housing_features(data, dataset_type)
        self.feature_engineered_data = engineered_data
        
        # Preprocessing
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.prepare_data(
            engineered_data, 
            target_column=target_column
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
        Train multiple regression models.
        
        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        logger.info("Training models...")
        
        # Train all models
        training_results = self.model_trainer.train_all_models(
            self.processed_data['X_train'],
            self.processed_data['y_train'],
            hyperparameter_tuning=hyperparameter_tuning,
            cv_folds=cv_folds
        )
        
        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = self.model_trainer.evaluate_models(
            self.processed_data['X_test'],
            self.processed_data['y_test']
        )
        
        # Get model comparison
        comparison = self.model_trainer.get_model_comparison()
        
        # Print results
        self._print_model_results(comparison, evaluation_results)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_comparison': comparison,
            'best_model': self.model_trainer.best_model_name
        }
    
    def _print_model_results(self, comparison: pd.DataFrame, evaluation_results: Dict[str, Any]) -> None:
        """Print model training results."""
        print("\nModel Performance Comparison:")
        print("=" * 60)
        print(comparison.to_string(index=False))
        print("=" * 60)
        
        if self.model_trainer.best_model_name:
            best_results = evaluation_results.get(self.model_trainer.best_model_name, {})
            if 'error' not in best_results:
                print(f"\nBest model: {self.model_trainer.best_model_name}")
                print(f"RMSE: ${best_results.get('rmse', 0):.2f}")
                print(f"R²: {best_results.get('r2', 0):.4f}")
    
    def predict_price(self, model_name: str, features: Dict[str, Any]) -> float:
        """
        Make price prediction using a trained model.
        
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
        
        # Transform features using preprocessor
        try:
            transformed_features = self.preprocessor.transform_new_data(feature_df)
        except Exception as e:
            logger.warning(f"Error transforming features: {e}")
            # Fallback: ensure all required columns are present
            required_columns = self.processed_data['feature_names']
            transformed_features = pd.DataFrame(columns=required_columns)
            for col in required_columns:
                if col in feature_df.columns:
                    transformed_features[col] = feature_df[col]
                else:
                    transformed_features[col] = 0
            transformed_features = transformed_features[required_columns]
        
        # Make prediction
        prediction = self.model_trainer.predict(model_name, transformed_features)
        
        return float(prediction[0])
    
    def generate_visualizations(self, save_plots: bool = True, output_dir: str = "visualizations") -> Dict[str, str]:
        """
        Generate comprehensive visualizations.
        
        Args:
            save_plots (bool): Whether to save plots to files
            output_dir (str): Directory to save plots
            
        Returns:
            Dict: Paths to saved plots
        """
        if self.processed_data is None:
            raise ValueError("Data must be processed first")
        
        logger.info("Generating visualizations...")
        
        # Create output directory
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_paths = {}
        
        # Price distribution
        if 'y_train' in self.processed_data:
            fig = self.visualizer.plot_price_distribution(
                self.processed_data['y_train'], 
                "Price Distribution (Training Data)"
            )
            if save_plots:
                path = os.path.join(output_dir, "price_distribution.png")
                self.visualizer.save_plot(fig, path)
                plot_paths['price_distribution'] = path
            else:
                plot_paths['price_distribution'] = "Displayed"
        
        # Correlation heatmap
        if self.feature_engineered_data is not None:
            fig = self.visualizer.plot_correlation_heatmap(
                self.feature_engineered_data,
                "Feature Correlation Heatmap"
            )
            if save_plots:
                path = os.path.join(output_dir, "correlation_heatmap.png")
                self.visualizer.save_plot(fig, path)
                plot_paths['correlation_heatmap'] = path
            else:
                plot_paths['correlation_heatmap'] = "Displayed"
        
        # Feature importance (if available)
        if self.model_trainer.best_model_name:
            feature_importance = self.model_trainer.get_feature_importance(
                self.model_trainer.best_model_name,
                self.processed_data['feature_names']
            )
            if 'error' not in feature_importance:
                fig = self.visualizer.plot_feature_importance(
                    feature_importance['feature_importance'],
                    f"Feature Importance - {self.model_trainer.best_model_name}"
                )
                if save_plots:
                    path = os.path.join(output_dir, "feature_importance.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['feature_importance'] = path
                else:
                    plot_paths['feature_importance'] = "Displayed"
        
        # Model comparison
        if hasattr(self.model_trainer, 'model_results') and self.model_trainer.model_results:
            comparison = self.model_trainer.get_model_comparison()
            if not comparison.empty:
                fig = self.visualizer.plot_model_comparison(comparison)
                if save_plots:
                    path = os.path.join(output_dir, "model_comparison.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['model_comparison'] = path
                else:
                    plot_paths['model_comparison'] = "Displayed"
        
        # Actual vs predicted for best model
        if self.model_trainer.best_model_name and 'evaluation_results' in self.model_trainer.model_results:
            best_eval = self.model_trainer.model_results['evaluation_results'].get(self.model_trainer.best_model_name, {})
            if 'error' not in best_eval:
                fig = self.visualizer.plot_actual_vs_predicted(
                    best_eval['actual'],
                    best_eval['predictions'],
                    self.model_trainer.best_model_name
                )
                if save_plots:
                    path = os.path.join(output_dir, "actual_vs_predicted.png")
                    self.visualizer.save_plot(fig, path)
                    plot_paths['actual_vs_predicted'] = path
                else:
                    plot_paths['actual_vs_predicted'] = "Displayed"
        
        logger.info(f"Generated {len(plot_paths)} visualizations")
        return plot_paths
    
    def run_interactive_prediction(self) -> None:
        """Run interactive prediction mode."""
        if not self.model_trainer.best_model_name:
            print("No trained model available. Please train models first.")
            return
        
        print(f"\n=== Interactive Prediction Mode ===")
        print(f"Using model: {self.model_trainer.best_model_name}")
        print("Enter 'quit' to exit\n")
        
        while True:
            try:
                print("Enter feature values:")
                features = {}
                
                # Get feature values based on dataset
                if self.feature_engineered_data is not None:
                    numeric_cols = self.feature_engineered_data.select_dtypes(include=[np.number]).columns
                    target_col = self.processed_data.get('target_column', 'price')
                    
                    for col in numeric_cols:
                        if col != target_col and col in self.processed_data['feature_names']:
                            value = input(f"{col}: ")
                            if value.lower() == 'quit':
                                return
                            try:
                                features[col] = float(value)
                            except ValueError:
                                print(f"Invalid value for {col}, using 0")
                                features[col] = 0
                else:
                    # Default features for generic prediction
                    features['floor_area_sqm'] = float(input("Floor area (sqm): ") or "90")
                    features['rooms'] = int(input("Number of rooms: ") or "3")
                    features['age'] = int(input("Property age (years): ") or "10")
                
                # Make prediction
                prediction = self.predict_price(self.model_trainer.best_model_name, features)
                print(f"\nPredicted price: ${prediction:,.2f}\n")
                
            except KeyboardInterrupt:
                print("\nExiting prediction mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def save_models(self, output_dir: str = "models") -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Args:
            output_dir (str): Directory to save models
            
        Returns:
            Dict: Paths to saved models
        """
        if not self.model_trainer.trained_models:
            raise ValueError("No trained models to save")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_models = {}
        
        for model_name in self.model_trainer.trained_models:
            filepath = os.path.join(output_dir, f"{model_name}.pkl")
            self.model_trainer.save_model(model_name, filepath)
            saved_models[model_name] = filepath
        
        logger.info(f"Saved {len(saved_models)} models to {output_dir}")
        return saved_models


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Housing Price Predictor")
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    
    # Optional arguments
    parser.add_argument('--type', type=str, default='generic', 
                       choices=['hdb', 'portland', 'generic'],
                       help='Dataset type')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--tuning', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--no-save', action='store_true', help='Skip model saving')
    parser.add_argument('--interactive', action='store_true', help='Start interactive prediction mode')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    try:
        # Load and preprocess data
        start_time = time.time()
        processed_data = predictor.load_and_preprocess_data(
            args.data, args.type, args.target
        )
        
        # Train models
        results = predictor.train_models(args.tuning, args.cv_folds)
        
        # Generate visualizations
        if not args.no_viz:
            predictor.generate_visualizations(save_plots=True)
        
        # Save models
        if not args.no_save:
            predictor.save_models()
        
        # Interactive mode
        if args.interactive:
            predictor.run_interactive_prediction()
        
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
