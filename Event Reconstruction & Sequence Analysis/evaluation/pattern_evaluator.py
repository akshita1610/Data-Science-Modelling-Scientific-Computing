"""
Pattern Evaluator Module

Evaluates pattern detection quality and performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


class PatternEvaluator:
    """
    Evaluates pattern detection results.
    
    This class provides metrics for assessing the quality and usefulness
    of detected patterns in event sequences.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PatternEvaluator with configuration.
        
        Args:
            config (Dict): Configuration dictionary for evaluation
        """
        self.config = config.get('evaluation', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
    def evaluate_patterns(self, detected_patterns: Dict,
                         ground_truth_patterns: Optional[Dict] = None) -> Dict:
        """
        Evaluate pattern detection results.
        
        Args:
            detected_patterns (Dict): Detected patterns
            ground_truth_patterns (Optional[Dict]): Ground truth patterns for comparison
            
        Returns:
            Dict: Pattern evaluation results
        """
        logger.info("Starting pattern evaluation")
        
        evaluation = {
            'basic_metrics': self._calculate_pattern_basic_metrics(detected_patterns),
            'diversity_metrics': self._calculate_pattern_diversity_metrics(detected_patterns),
            'quality_metrics': self._calculate_pattern_quality_metrics(detected_patterns),
            'coverage_metrics': self._calculate_pattern_coverage_metrics(detected_patterns),
            'accuracy_metrics': self._calculate_pattern_accuracy_metrics(
                detected_patterns, ground_truth_patterns) if ground_truth_patterns else {}
        }
        
        # Overall score
        evaluation['overall_score'] = self._calculate_pattern_overall_score(evaluation)
        
        logger.info(f"Pattern evaluation completed. Overall score: {evaluation['overall_score']:.3f}")
        return evaluation
    
    def _calculate_pattern_basic_metrics(self, patterns: Dict) -> Dict:
        """Calculate basic pattern metrics."""
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values() if isinstance(pattern_list, list))
        
        # Pattern type distribution
        pattern_types = {}
        for pattern_type, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                pattern_types[pattern_type] = len(pattern_list)
        
        # Length distribution
        all_lengths = []
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    if 'length' in pattern:
                        all_lengths.append(pattern['length'])
        
        length_stats = {}
        if all_lengths:
            length_stats = {
                'mean_length': np.mean(all_lengths),
                'median_length': np.median(all_lengths),
                'min_length': min(all_lengths),
                'max_length': max(all_lengths),
                'std_length': np.std(all_lengths)
            }
        
        # Support distribution
        all_supports = []
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    if 'support' in pattern:
                        all_supports.append(pattern['support'])
        
        support_stats = {}
        if all_supports:
            support_stats = {
                'mean_support': np.mean(all_supports),
                'median_support': np.median(all_supports),
                'min_support': min(all_supports),
                'max_support': max(all_supports),
                'std_support': np.std(all_supports)
            }
        
        return {
            'total_patterns': total_patterns,
            'pattern_types': pattern_types,
            'length_distribution': length_stats,
            'support_distribution': support_stats
        }
    
    def _calculate_pattern_diversity_metrics(self, patterns: Dict) -> Dict:
        """Calculate pattern diversity metrics."""
        # Collect all unique patterns
        all_patterns = []
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    if 'pattern' in pattern:
                        all_patterns.append(tuple(pattern['pattern']))
        
        if not all_patterns:
            return {'pattern_diversity': 0, 'unique_patterns': 0}
        
        unique_patterns = set(all_patterns)
        pattern_diversity = len(unique_patterns) / len(all_patterns) if all_patterns else 0
        
        # Event diversity in patterns
        all_events_in_patterns = []
        for pattern in all_patterns:
            all_events_in_patterns.extend(pattern)
        
        unique_events_in_patterns = set(all_events_in_patterns)
        event_diversity = len(unique_events_in_patterns) / len(all_events_in_patterns) if all_events_in_patterns else 0
        
        return {
            'pattern_diversity': pattern_diversity,
            'unique_patterns': len(unique_patterns),
            'total_pattern_instances': len(all_patterns),
            'event_diversity_in_patterns': event_diversity,
            'unique_events_in_patterns': len(unique_events_in_patterns)
        }
    
    def _calculate_pattern_quality_metrics(self, patterns: Dict) -> Dict:
        """Calculate pattern quality metrics."""
        quality_scores = []
        
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    score = self._calculate_individual_pattern_quality(pattern)
                    quality_scores.append(score)
        
        if not quality_scores:
            return {'mean_quality': 0, 'quality_distribution': {}}
        
        quality_stats = {
            'mean_quality': np.mean(quality_scores),
            'median_quality': np.median(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'std_quality': np.std(quality_scores)
        }
        
        # Quality distribution
        quality_ranges = {'Low (0-0.3)': 0, 'Medium (0.3-0.7)': 0, 'High (0.7-1.0)': 0}
        for score in quality_scores:
            if score < 0.3:
                quality_ranges['Low (0-0.3)'] += 1
            elif score < 0.7:
                quality_ranges['Medium (0.3-0.7)'] += 1
            else:
                quality_ranges['High (0.7-1.0)'] += 1
        
        return {
            'quality_stats': quality_stats,
            'quality_distribution': quality_ranges
        }
    
    def _calculate_individual_pattern_quality(self, pattern: Dict) -> float:
        """Calculate quality score for an individual pattern."""
        score = 0.0
        weights = {
            'support': 0.4,
            'length': 0.2,
            'uniqueness': 0.2,
            'complexity': 0.2
        }
        
        # Support component
        support = pattern.get('support', 0)
        score += weights['support'] * support
        
        # Length component (moderate length is better)
        length = pattern.get('length', 1)
        length_score = 1.0 - abs(length - 5) / 5  # Ideal length around 5
        length_score = max(0, length_score)
        score += weights['length'] * length_score
        
        # Uniqueness component (harder to quantify, use length as proxy)
        uniqueness_score = min(1.0, length / 10)  # Longer patterns tend to be more unique
        score += weights['uniqueness'] * uniqueness_score
        
        # Complexity component (based on pattern diversity)
        if 'pattern' in pattern:
            events = pattern['pattern']
            complexity_score = len(set(events)) / len(events) if events else 0
            score += weights['complexity'] * complexity_score
        
        return score
    
    def _calculate_pattern_coverage_metrics(self, patterns: Dict) -> Dict:
        """Calculate pattern coverage metrics."""
        # Collect all events covered by patterns
        covered_events = set()
        total_pattern_instances = 0
        
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    if 'pattern' in pattern:
                        covered_events.update(pattern['pattern'])
                        total_pattern_instances += 1
        
        # Calculate coverage (this would need the original data for accurate calculation)
        # For now, return what we can calculate
        return {
            'unique_events_covered': len(covered_events),
            'total_pattern_instances': total_pattern_instances,
            'avg_events_per_pattern': len(covered_events) / total_pattern_instances if total_pattern_instances > 0 else 0
        }
    
    def _calculate_pattern_accuracy_metrics(self, detected_patterns: Dict,
                                          ground_truth_patterns: Dict) -> Dict:
        """Calculate pattern accuracy metrics against ground truth."""
        # Convert patterns to comparable format
        detected_set = self._patterns_to_set(detected_patterns)
        ground_truth_set = self._patterns_to_set(ground_truth_patterns)
        
        # Calculate precision, recall, F1
        true_positives = len(detected_set & ground_truth_set)
        false_positives = len(detected_set - ground_truth_set)
        false_negatives = len(ground_truth_set - detected_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        }
    
    def _patterns_to_set(self, patterns: Dict) -> set:
        """Convert patterns dictionary to a set of tuples for comparison."""
        pattern_set = set()
        
        for pattern_list in patterns.values():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    if 'pattern' in pattern:
                        pattern_tuple = tuple(pattern['pattern'])
                        pattern_set.add(pattern_tuple)
        
        return pattern_set
    
    def _calculate_pattern_overall_score(self, evaluation: Dict) -> float:
        """Calculate overall pattern evaluation score."""
        weights = {
            'diversity': 0.3,
            'quality': 0.3,
            'coverage': 0.2,
            'accuracy': 0.2
        }
        
        scores = {
            'diversity': evaluation['diversity_metrics'].get('pattern_diversity', 0),
            'quality': evaluation['quality_metrics']['quality_stats'].get('mean_quality', 0),
            'coverage': min(1.0, evaluation['coverage_metrics'].get('avg_events_per_pattern', 0) / 5),  # Normalize
            'accuracy': evaluation.get('accuracy_metrics', {}).get('f1_score', 0)
        }
        
        overall_score = sum(weights[key] * scores[key] for key in weights)
        
        return overall_score
    
    def compare_pattern_detection_methods(self, 
                                        results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple pattern detection methods.
        
        Args:
            results_dict (Dict[str, Dict]): Dictionary of method names to evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for method_name, results in results_dict.items():
            row = {
                'Method': method_name,
                'Overall Score': results.get('overall_score', 0),
                'Total Patterns': results['basic_metrics']['total_patterns'],
                'Pattern Diversity': results['diversity_metrics']['pattern_diversity'],
                'Mean Quality': results['quality_metrics']['quality_stats']['mean_quality'],
                'Unique Events Covered': results['coverage_metrics']['unique_events_covered']
            }
            
            # Add accuracy metrics if available
            if 'accuracy_metrics' in results:
                row.update({
                    'Precision': results['accuracy_metrics'].get('precision', 0),
                    'Recall': results['accuracy_metrics'].get('recall', 0),
                    'F1 Score': results['accuracy_metrics'].get('f1_score', 0)
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
        
        return comparison_df
    
    def generate_pattern_report(self, evaluation: Dict) -> str:
        """
        Generate a human-readable pattern evaluation report.
        
        Args:
            evaluation (Dict): Evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        basic_metrics = evaluation['basic_metrics']
        diversity_metrics = evaluation['diversity_metrics']
        quality_metrics = evaluation['quality_metrics']
        coverage_metrics = evaluation['coverage_metrics']
        
        report = f"""
        Pattern Detection Evaluation Report
        ==================================
        
        Overall Score: {evaluation['overall_score']:.3f}
        
        Basic Metrics:
        --------------
        • Total Patterns: {basic_metrics['total_patterns']}
        • Pattern Types: {len(basic_metrics['pattern_types'])}
        """
        
        # Add pattern type details
        for pattern_type, count in basic_metrics['pattern_types'].items():
            report += f"  - {pattern_type}: {count}\n"
        
        report += f"""
        Length Distribution:
        -------------------
        """
        
        if basic_metrics['length_distribution']:
            length_stats = basic_metrics['length_distribution']
            report += f"""
        • Mean Length: {length_stats['mean_length']:.2f}
        • Median Length: {length_stats['median_length']:.2f}
        • Min/Max Length: {length_stats['min_length']} / {length_stats['max_length']}
        """
        
        report += f"""
        Support Distribution:
        --------------------
        """
        
        if basic_metrics['support_distribution']:
            support_stats = basic_metrics['support_distribution']
            report += f"""
        • Mean Support: {support_stats['mean_support']:.3f}
        • Median Support: {support_stats['median_support']:.3f}
        • Min/Max Support: {support_stats['min_support']:.3f} / {support_stats['max_support']:.3f}
        """
        
        report += f"""
        Diversity Metrics:
        ------------------
        • Pattern Diversity: {diversity_metrics['pattern_diversity']:.3f}
        • Unique Patterns: {diversity_metrics['unique_patterns']}
        • Event Diversity in Patterns: {diversity_metrics['event_diversity_in_patterns']:.3f}
        
        Quality Metrics:
        ----------------
        • Mean Quality: {quality_metrics['quality_stats']['mean_quality']:.3f}
        • Median Quality: {quality_metrics['quality_stats']['median_quality']:.3f}
        • Min/Max Quality: {quality_metrics['quality_stats']['min_quality']:.3f} / {quality_metrics['quality_stats']['max_quality']:.3f}
        
        Quality Distribution:
        --------------------
        """
        
        quality_dist = quality_metrics['quality_distribution']
        for range_name, count in quality_dist.items():
            report += f"• {range_name}: {count}\n"
        
        report += f"""
        Coverage Metrics:
        ------------------
        • Unique Events Covered: {coverage_metrics['unique_events_covered']}
        • Total Pattern Instances: {coverage_metrics['total_pattern_instances']}
        • Avg Events per Pattern: {coverage_metrics['avg_events_per_pattern']:.2f}
        """
        
        # Add accuracy metrics if available
        if 'accuracy_metrics' in evaluation:
            accuracy = evaluation['accuracy_metrics']
            report += f"""
        Accuracy Metrics:
        -----------------
        • Precision: {accuracy['precision']:.3f}
        • Recall: {accuracy['recall']:.3f}
        • F1 Score: {accuracy['f1_score']:.3f}
        • Accuracy: {accuracy['accuracy']:.3f}
        """
        
        return report
