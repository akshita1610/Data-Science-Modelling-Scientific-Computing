"""
Evaluation Module
Contains metrics and evaluation tools for detector performance analysis.
"""

from .metrics import EvaluationMetrics, PerformanceMetrics
from .comparison import SignalComparator, CalibrationComparator
from .analysis import StatisticalAnalyzer, QualityAssessment

__all__ = [
    'EvaluationMetrics', 'PerformanceMetrics',
    'SignalComparator', 'CalibrationComparator',
    'StatisticalAnalyzer', 'QualityAssessment'
]
