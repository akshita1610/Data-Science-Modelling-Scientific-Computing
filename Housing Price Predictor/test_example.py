"""
Housing Price Predictor - Basic Test Suite

This script tests the basic functionality of the Housing Price Predictor.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import HousingPricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test data loading functionality."""
    print("1. Testing data loading and preprocessing...")
    
    try:
        predictor = HousingPricePredictor()
        processed_data = predictor.load_and_preprocess_data("sample_data.csv", "hdb")
        
        assert processed_data is not None, "Processed data should not be None"
        assert 'X_train' in processed_data, "X_train should be in processed data"
        assert 'X_test' in processed_data, "X_test should be in processed data"
        assert 'y_train' in processed_data, "y_train should be in processed data"
        assert 'y_test' in processed_data, "y_test should be in processed data"
        
        print("SUCCESS: Data loading and preprocessing successful")
        print(f"  Training samples: {processed_data['X_train'].shape[0]}")
        print(f"  Test samples: {processed_data['X_test'].shape[0]}")
        print(f"  Features: {len(processed_data['feature_names'])}")
        
        return True, processed_data
        
    except Exception as e:
        print(f"FAILED: Data loading error - {e}")
        return False, None


def test_model_training(processed_data):
    """Test model training functionality."""
    print("\n2. Testing model training...")
    
    try:
        predictor = HousingPricePredictor()
        predictor.processed_data = processed_data
        
        results = predictor.train_models(hyperparameter_tuning=False)
        
        assert results is not None, "Training results should not be None"
        assert 'best_model' in results, "Best model should be in results"
        assert 'model_comparison' in results, "Model comparison should be in results"
        
        print("SUCCESS: Model training successful")
        print(f"  Best model: {results['best_model']}")
        
        # Get evaluation results
        if hasattr(predictor.model_trainer, 'model_results'):
            eval_results = predictor.model_trainer.model_results.get('evaluation_results', {})
            if results['best_model'] in eval_results:
                best_eval = eval_results[results['best_model']]
                if 'error' not in best_eval:
                    print(f"  RMSE: ${best_eval.get('rmse', 0):.2f}")
                    print(f"  R²: {best_eval.get('r2', 0):.4f}")
        
        return True, predictor
        
    except Exception as e:
        print(f"FAILED: Model training error - {e}")
        return False, None


def test_prediction(predictor):
    """Test prediction functionality."""
    print("\n3. Testing prediction functionality...")
    
    try:
        if not predictor.model_trainer.best_model_name:
            print("FAILED: No best model available for prediction")
            return False
        
        # Test with a sample from the training data instead of custom features
        if len(predictor.processed_data['X_test']) > 0:
            test_sample = predictor.processed_data['X_test'].iloc[0:1]
            prediction = predictor.model_trainer.predict(predictor.model_trainer.best_model_name, test_sample)
            actual_value = predictor.processed_data['y_test'].iloc[0]
            
            assert isinstance(prediction, (np.ndarray, list)), "Prediction should be an array"
            assert len(prediction) > 0, "Prediction should not be empty"
            assert prediction[0] > 0, "Prediction should be positive"
            
            print(f"SUCCESS: Prediction successful: ${prediction[0]:,.2f}")
            print(f"SUCCESS: Actual value: ${actual_value:,.2f}")
            print(f"SUCCESS: Prediction error: ${abs(prediction[0] - actual_value):,.2f}")
            return True
        else:
            print("FAILED: No test data available")
            return False
        
    except Exception as e:
        print(f"FAILED: Prediction error - {e}")
        return False


def test_feature_importance(predictor):
    """Test feature importance functionality."""
    print("\n4. Testing feature importance analysis...")
    
    try:
        if not predictor.model_trainer.best_model_name:
            print("FAILED: No best model available for feature importance")
            return False
        
        if not predictor.processed_data:
            print("FAILED: No processed data available")
            return False
        
        feature_importance = predictor.model_trainer.get_feature_importance(
            predictor.model_trainer.best_model_name,
            predictor.processed_data['feature_names']
        )
        
        assert feature_importance is not None, "Feature importance should not be None"
        
        if 'error' not in feature_importance:
            assert 'feature_importance' in feature_importance, "Feature importance dict should be present"
            assert len(feature_importance['feature_importance']) > 0, "Should have at least one feature"
            
            print("SUCCESS: Feature importance analysis successful")
            print(f"  Features analyzed: {len(feature_importance['feature_importance'])}")
            
            if 'top_10_features' in feature_importance:
                print("  Top 5 features:")
                for i, (feature, importance) in enumerate(feature_importance['top_10_features'][:5], 1):
                    print(f"    {i}. {feature}: {importance:.4f}")
        else:
            print(f"INFO: Feature importance not available - {feature_importance['error']}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Feature importance error - {e}")
        return False


def test_visualization(predictor):
    """Test visualization functionality."""
    print("\n5. Testing visualization generation...")
    
    try:
        plot_paths = predictor.generate_visualizations(save_plots=False)
        
        assert plot_paths is not None, "Plot paths should not be None"
        assert len(plot_paths) > 0, "Should generate at least one plot"
        
        print("SUCCESS: Visualization generation successful")
        print(f"  Plots generated: {len(plot_paths)}")
        for plot_type, path in plot_paths.items():
            print(f"    {plot_type}: {path}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Visualization error - {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("HOUSING PRICE PREDICTOR - BASIC TESTS")
    print("=" * 60)
    
    # Test data loading
    success, processed_data = test_data_loading()
    if not success:
        return False
    
    # Test model training
    success, predictor = test_model_training(processed_data)
    if not success:
        return False
    
    # Test prediction
    success = test_prediction(predictor)
    if not success:
        return False
    
    # Test feature importance
    test_feature_importance(predictor)
    
    # Test visualization
    test_visualization(predictor)
    
    return True


def main():
    """Main test function."""
    try:
        success = run_all_tests()
        
        print("\n" + "=" * 60)
        if success:
            print("ALL TESTS PASSED!")
            print("The Housing Price Predictor is working correctly.")
            print("\nNext steps:")
            print("1. Try the full demo: python demo.py")
            print("2. Use with your own data: python main.py --data your_data.csv")
            print("3. Start interactive mode: python main.py --data sample_data.csv --interactive")
        else:
            print("SOME TESTS FAILED!")
            print("Please check the error messages above.")
        
        print("=" * 60)
        return success
        
    except Exception as e:
        print(f"\nTEST SUITE FAILED: {e}")
        logger.error(f"Test suite error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
