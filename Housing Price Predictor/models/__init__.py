"""
Models module for Housing Price Predictor

This module handles machine learning model training, evaluation, and selection.
"""

from .model_trainer import ModelTrainer
from .improved_model_trainer import ImprovedModelTrainer

__all__ = [
    "ModelTrainer",
    "ImprovedModelTrainer"
]
