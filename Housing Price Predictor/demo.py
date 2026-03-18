"""
Housing Price Predictor - Complete Demonstration

This script demonstrates all features of the Housing Price Predictor system.
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import HousingPricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_demo():
    """Run complete demonstration of the Housing Price Predictor."""
    print("=" * 80)
    print("HOUSING PRICE PREDICTOR - COMPLETE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    try:
        # Step 1: Load and preprocess data
        print("\n1. LOADING AND PREPROCESSING DATA")
        print("-" * 50)
        
        start_time = time.time()
        processed_data = predictor.load_and_preprocess_data(
            file_path="sample_data.csv",
            dataset_type="hdb"
        )
        load_time = time.time() - start_time
        
        print("SUCCESS: Data loading and preprocessing successful")
        print(f"Training samples: {processed_data['X_train'].shape[0]}")
        print(f"Test samples: {processed_data['X_test'].shape[0]}")
        print(f"Features: {len(processed_data['feature_names'])}")
        
        # Step 2: Train models
        print("\n2. TRAINING MACHINE LEARNING MODELS")
        print("-" * 50)
        
        start_time = time.time()
        results = predictor.train_models(hyperparameter_tuning=False)
        training_time = time.time() - start_time
        
        print(f"Models trained in {training_time:.2f} seconds")
        print(f"Best model: {results['best_model']}")
        
        # Step 3: Make predictions
        print("\n3. MAKING PREDICTIONS")
        print("-" * 50)
        
        # Test with sample data
        sample_features = {
            'floor_area_sqm': 90.0,
            'town': 'BISHAN',
            'flat_type': '4 ROOM',
            'flat_model': 'Model A'
        }
        
        try:
            prediction = predictor.predict_price(results['best_model'], sample_features)
            print("SUCCESS: Sample prediction: ${prediction:,.2f}")
        except Exception as e:
            print("WARNING: Prediction error: {e}")
        
        # Test with a sample from the training data
        if len(processed_data['X_test']) > 0:
            test_sample = processed_data['X_test'].iloc[0:1]
            try:
                test_prediction = predictor.model_trainer.predict(results['best_model'], test_sample)
                actual_value = processed_data['y_test'].iloc[0]
                print("SUCCESS: Test sample prediction: ${test_prediction[0]:,.2f}")
                print(f"SUCCESS: Actual value: ${actual_value:,.2f}")
                print(f"SUCCESS: Prediction error: ${abs(test_prediction[0] - actual_value):,.2f}")
            except Exception as e:
                print("WARNING: Test prediction error: {e}")
        
        # Step 4: Feature importance analysis
        print("\n4. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        try:
            feature_importance = predictor.model_trainer.get_feature_importance(
                results['best_model'], 
                processed_data['feature_names']
            )
            
            if 'error' not in feature_importance:
                print("✅ Top 10 Most Important Features:")
                for i, (feature, importance) in enumerate(feature_importance['top_10_features'][:10], 1):
                    print(f"   {i:2d}. {feature:<25} {importance:.4f}")
            else:
                print("⚠ Feature importance not available for this model type")
        except Exception as e:
            print(f"⚠ Feature importance error: {e}")
        
        # Step 5: Generate visualizations
        print("\n5. GENERATING VISUALIZATIONS")
        print("-" * 50)
        
        try:
            start_time = time.time()
            plot_paths = predictor.generate_visualizations(save_plots=True)
            viz_time = time.time() - start_time
            
            print(f"✓ Visualizations generated in {viz_time:.2f} seconds")
            for plot_type, path in plot_paths.items():
                print(f"✓ {plot_type.replace('_', ' ').title()}: {path}")
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
        
        # Step 6: Save models
        print("\n6. SAVING MODELS")
        print("-" * 50)
        
        try:
            start_time = time.time()
            saved_models = predictor.save_models()
            save_time = time.time() - start_time
            
            print(f"✓ Models saved in {save_time:.2f} seconds")
            for model_name, path in saved_models.items():
                print(f"✓ {model_name}: {path}")
        except Exception as e:
            print(f"⚠ Model saving error: {e}")
        
        # Step 7: Performance summary
        print("\n7. PERFORMANCE SUMMARY")
        print("-" * 50)
        
        try:
            comparison = results['model_comparison']
            if not comparison.empty:
                print("\n📊 Model Performance Ranking:")
                for i, (_, row) in enumerate(comparison.iterrows(), 1):
                    print(f"   {i}. {row['Model']}: RMSE=${row['CV RMSE Mean']:,.0f}")
                
                best_row = comparison.iloc[0]
                print(f"\n🏆 Best Model: {best_row['Model']}")
                print(f"   Cross-Validation RMSE: ${best_row['CV RMSE Mean']:,.2f}")
                print(f"   Cross-Validation Std: ${best_row['CV RMSE Std']:,.2f}")
        except Exception as e:
            print(f"⚠ Performance summary error: {e}")
        
        # Step 8: Dataset information
        print("\n8. DATASET INFORMATION")
        print("-" * 50)
        
        try:
            data_info = predictor.data_loader.get_data_info()
            print(f"✓ Dataset type: {data_info['dataset_type']}")
            print(f"✓ Original shape: {data_info['original_shape']}")
            print(f"✓ Cleaned shape: {data_info['shape']}")
            print(f"✓ Numeric columns: {len(data_info['numeric_columns'])}")
            print(f"✓ Categorical columns: {len(data_info['categorical_columns'])}")
            print(f"✓ Memory usage: {data_info['memory_usage'] / 1024:.1f} KB")
        except Exception as e:
            print(f"WARNING: Dataset info error: {e}")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📈 Models trained: {len(predictor.model_trainer.trained_models)}")
        print(f"SUCCESS: Best model: {results['best_model']}")
        print(f"📊 Features used: {len(processed_data['feature_names'])}")
        print(f"📁 Output files generated:")
        
        # Count output files
        output_files = []
        if os.path.exists("visualizations"):
            output_files.extend([f for f in os.listdir("visualizations") if f.endswith('.png')])
        if os.path.exists("models"):
            output_files.extend([f for f in os.listdir("models") if f.endswith('.pkl')])
        
        for file in output_files:
            print(f"   ✓ {file}")
        
        print("\n🎉 Your Housing Price Predictor is ready for use!")
        print("💡 Try the interactive mode: python main.py --data sample_data.csv --interactive")
        
    except Exception as e:
        print(f"WARNING: Demo failed with error: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        return False
    
    return True


def run_quick_demo():
    """Run a quick demonstration with minimal output."""
    print("Housing Price Predictor - Quick Demo")
    print("-" * 50)
    
    predictor = HousingPricePredictor()
    
    try:
        # Load and train
        predictor.load_and_preprocess_data("sample_data.csv", "hdb")
        results = predictor.train_models()
        
        # Show best model
        print(f"✓ Best model: {results['best_model']}")
        
        # Quick prediction
        sample_features = {'floor_area_sqm': 90, 'town': 'BISHAN', 'flat_type': '4 ROOM'}
        prediction = predictor.predict_price(results['best_model'], sample_features)
        print("SUCCESS: Sample prediction: ${prediction:,.2f}")
        
        print("SUCCESS: Quick demo completed successfully!")
        
    except Exception as e:
        print(f"WARNING: Quick demo failed: {e}")
        return False
    
    return True


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Housing Price Predictor Demo")
    parser.add_argument('--quick', action='store_true', help='Run quick demo')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_demo()
    elif args.interactive:
        # Run complete demo then start interactive mode
        success = run_complete_demo()
        if success:
            predictor = HousingPricePredictor()
            try:
                predictor.load_and_preprocess_data("sample_data.csv", "hdb")
                predictor.train_models()
                predictor.run_interactive_prediction()
            except Exception as e:
                print(f"Interactive mode error: {e}")
    else:
        success = run_complete_demo()
    
    if success:
        print("\n🎓 Perfect for graduate school applications!")
        print("📁 Check the generated files in 'visualizations/' and 'models/' folders")
    else:
        print("\nDemo failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
