"""
Housing Price Predictor Package

A comprehensive machine learning system for predicting housing prices
using multiple regression algorithms and advanced preprocessing techniques.

Author: Housing Price Predictor Team
License: MIT License
Version: 1.1.0
"""

__version__ = "1.1.0"
__author__ = "Housing Price Predictor Team"
__email__ = "contact@example.com"
__license__ = "MIT"

from .main import HousingPricePredictor
from .improved_main import ImprovedHousingPricePredictor

__all__ = [
    "HousingPricePredictor",
    "ImprovedHousingPricePredictor",
]
