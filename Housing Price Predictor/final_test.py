"""
Final Test - Housing Price Predictor

Simple test to confirm the project is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import HousingPricePredictor

def main():
    print("Housing Price Predictor - Final Test")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = HousingPricePredictor()
        
        # Load and preprocess data
        print("1. Loading data...")
        processed_data = predictor.load_and_preprocess_data("sample_data.csv", "hdb")
        print(f"   SUCCESS: Loaded {processed_data['X_train'].shape[0]} training samples")
        
        # Train models
        print("2. Training models...")
        results = predictor.train_models()
        print(f"   SUCCESS: Best model: {results['best_model']}")
        
        # Make prediction
        print("3. Making prediction...")
        test_sample = processed_data['X_test'].iloc[0:1]
        prediction = predictor.model_trainer.predict(results['best_model'], test_sample)
        actual = processed_data['y_test'].iloc[0]
        print(f"   SUCCESS: Predicted: ${prediction[0]:,.2f}, Actual: ${actual:,.2f}")
        
        print("\n" + "=" * 50)
        print("FINAL TEST PASSED!")
        print("Your Housing Price Predictor is ready!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\nFINAL TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
