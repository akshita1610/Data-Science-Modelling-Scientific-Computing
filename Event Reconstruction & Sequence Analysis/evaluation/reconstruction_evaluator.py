"""
Reconstruction Evaluator Module

Evaluates the quality and accuracy of event reconstruction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReconstructionEvaluator:
    """
    Evaluates event reconstruction quality.
    
    This class provides comprehensive metrics for assessing reconstruction accuracy,
    completeness, and quality compared to ground truth or expected patterns.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ReconstructionEvaluator with configuration.
        
        Args:
            config (Dict): Configuration dictionary for evaluation
        """
        self.config = config.get('evaluation', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Evaluation parameters
        self.tolerance_window = self.config.get('reconstruction_accuracy', {}).get('tolerance_window', 5)
        
    def evaluate_reconstruction(self, df_original: pd.DataFrame,
                             df_reconstructed: pd.DataFrame,
                             df_ground_truth: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate reconstruction quality comprehensively.
        
        Args:
            df_original (pd.DataFrame): Original event data
            df_reconstructed (pd.DataFrame): Reconstructed event data
            df_ground_truth (Optional[pd.DataFrame]): Ground truth data for comparison
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        logger.info("Starting reconstruction evaluation")
        
        evaluation = {
            'basic_metrics': self._calculate_basic_metrics(df_original, df_reconstructed),
            'accuracy_metrics': self._calculate_accuracy_metrics(df_original, df_reconstructed, df_ground_truth),
            'completeness_metrics': self._calculate_completeness_metrics(df_original, df_reconstructed),
            'temporal_metrics': self._calculate_temporal_metrics(df_original, df_reconstructed),
            'quality_metrics': self._calculate_quality_metrics(df_original, df_reconstructed),
            'confidence_metrics': self._calculate_confidence_metrics(df_reconstructed)
        }
        
        # Overall score
        evaluation['overall_score'] = self._calculate_overall_score(evaluation)
        
        logger.info(f"Reconstruction evaluation completed. Overall score: {evaluation['overall_score']:.3f}")
        return evaluation
    
    def _calculate_basic_metrics(self, df_original: pd.DataFrame,
                               df_reconstructed: pd.DataFrame) -> Dict:
        """Calculate basic reconstruction metrics."""
        original_count = len(df_original)
        reconstructed_count = len(df_reconstructed)
        added_events = reconstructed_count - original_count
        
        # Event type analysis
        original_events = set(df_original[self.event_column].unique())
        reconstructed_events = set(df_reconstructed[self.event_column].unique())
        common_events = original_events & reconstructed_events
        new_events = reconstructed_events - original_events
        lost_events = original_events - reconstructed_events
        
        return {
            'original_events': original_count,
            'reconstructed_events': reconstructed_count,
            'events_added': added_events,
            'reconstruction_ratio': reconstructed_count / original_count if original_count > 0 else 0,
            'original_event_types': len(original_events),
            'reconstructed_event_types': len(reconstructed_events),
            'common_event_types': len(common_events),
            'new_event_types': len(new_events),
            'lost_event_types': len(lost_events),
            'event_type_preservation_rate': len(common_events) / len(original_events) if original_events else 0
        }
    
    def _calculate_accuracy_metrics(self, df_original: pd.DataFrame,
                                 df_reconstructed: pd.DataFrame,
                                 df_ground_truth: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate reconstruction accuracy metrics."""
        if df_ground_truth is None:
            # Use original as proxy for ground truth
            df_ground_truth = df_original
        
        # Match events between reconstructed and ground truth
        matches = self._match_events(df_reconstructed, df_ground_truth)
        
        # Calculate basic accuracy metrics
        total_ground_truth = len(df_ground_truth)
        total_reconstructed = len(df_reconstructed)
        true_positives = len(matches)
        false_positives = total_reconstructed - true_positives
        false_negatives = total_ground_truth - true_positives
        
        precision = true_positives / total_reconstructed if total_reconstructed > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Event type accuracy
        event_type_accuracy = self._calculate_event_type_accuracy(matches, df_reconstructed, df_ground_truth)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'event_type_accuracy': event_type_accuracy,
            'exact_match_rate': true_positives / total_ground_truth if total_ground_truth > 0 else 0
        }
    
    def _match_events(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[Tuple]:
        """Match events between two DataFrames within tolerance window."""
        matches = []
        timestamps1 = pd.to_datetime(df1[self.timestamp_column])
        timestamps2 = pd.to_datetime(df2[self.timestamp_column])
        events1 = df1[self.event_column].values
        events2 = df2[self.event_column].values
        
        used_indices2 = set()
        
        for i, (timestamp1, event1) in enumerate(zip(timestamps1, events1)):
            for j, (timestamp2, event2) in enumerate(zip(timestamps2, events2)):
                if j in used_indices2:
                    continue
                
                # Check if events match within tolerance window
                time_diff = abs((timestamp1 - timestamp2).total_seconds())
                if time_diff <= self.tolerance_window and event1 == event2:
                    matches.append((i, j))
                    used_indices2.add(j)
                    break
        
        return matches
    
    def _calculate_event_type_accuracy(self, matches: List[Tuple],
                                     df_reconstructed: pd.DataFrame,
                                     df_ground_truth: pd.DataFrame) -> float:
        """Calculate event type accuracy for matched events."""
        if not matches:
            return 0.0
        
        correct_types = 0
        for recon_idx, truth_idx in matches:
            recon_event = df_reconstructed.iloc[recon_idx][self.event_column]
            truth_event = df_ground_truth.iloc[truth_idx][self.event_column]
            if recon_event == truth_event:
                correct_types += 1
        
        return correct_types / len(matches)
    
    def _calculate_completeness_metrics(self, df_original: pd.DataFrame,
                                      df_reconstructed: pd.DataFrame) -> Dict:
        """Calculate completeness metrics."""
        # Time range completeness
        orig_start = pd.to_datetime(df_original[self.timestamp_column]).min()
        orig_end = pd.to_datetime(df_original[self.timestamp_column]).max()
        recon_start = pd.to_datetime(df_reconstructed[self.timestamp_column]).min()
        recon_end = pd.to_datetime(df_reconstructed[self.timestamp_column]).max()
        
        time_coverage = min(recon_end, orig_end) - max(recon_start, orig_start)
        total_time = orig_end - orig_start
        time_completeness = time_coverage.total_seconds() / total_time.total_seconds() if total_time.total_seconds() > 0 else 0
        
        # Event sequence completeness
        original_events = df_original[self.event_column].values
        reconstructed_events = df_reconstructed[self.event_column].values
        
        # Find longest common subsequence (simplified)
        lcs_length = self._longest_common_subsequence(original_events, reconstructed_events)
        sequence_completeness = lcs_length / len(original_events) if len(original_events) > 0 else 0
        
        return {
            'time_completeness': time_completeness,
            'sequence_completeness': sequence_completeness,
            'time_range_preserved': recon_start <= orig_start and recon_end >= orig_end,
            'event_order_preserved': self._check_event_order_preservation(original_events, reconstructed_events)
        }
    
    def _longest_common_subsequence(self, seq1: List, seq2: List) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _check_event_order_preservation(self, original: List, reconstructed: List) -> float:
        """Check how well the original event order is preserved."""
        if not original or not reconstructed:
            return 0.0
        
        # Create position mapping for original events
        event_positions = {}
        for i, event in enumerate(original):
            if event not in event_positions:
                event_positions[event] = []
            event_positions[event].append(i)
        
        # Check order consistency for common events
        consistent_pairs = 0
        total_pairs = 0
        
        recon_event_positions = {}
        for i, event in enumerate(reconstructed):
            if event not in recon_event_positions:
                recon_event_positions[event] = []
            recon_event_positions[event].append(i)
        
        # Check pairwise order consistency
        for event1 in event_positions:
            for event2 in event_positions:
                if event1 == event2:
                    continue
                
                if event1 in recon_event_positions and event2 in recon_event_positions:
                    # Get first occurrences
                    orig_pos1 = event_positions[event1][0]
                    orig_pos2 = event_positions[event2][0]
                    recon_pos1 = recon_event_positions[event1][0]
                    recon_pos2 = recon_event_positions[event2][0]
                    
                    # Check if order is preserved
                    if (orig_pos1 < orig_pos2 and recon_pos1 < recon_pos2) or \
                       (orig_pos1 > orig_pos2 and recon_pos1 > recon_pos2):
                        consistent_pairs += 1
                    
                    total_pairs += 1
        
        return consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_temporal_metrics(self, df_original: pd.DataFrame,
                                 df_reconstructed: pd.DataFrame) -> Dict:
        """Calculate temporal accuracy metrics."""
        orig_timestamps = pd.to_datetime(df_original[self.timestamp_column])
        recon_timestamps = pd.to_datetime(df_reconstructed[self.timestamp_column])
        
        # Time gap analysis
        orig_gaps = orig_timestamps.diff().dt.total_seconds().dropna()
        recon_gaps = recon_timestamps.diff().dt.total_seconds().dropna()
        
        # Calculate gap statistics
        orig_gap_stats = {
            'mean': orig_gaps.mean(),
            'std': orig_gaps.std(),
            'median': orig_gaps.median()
        }
        
        recon_gap_stats = {
            'mean': recon_gaps.mean(),
            'std': recon_gaps.std(),
            'median': recon_gaps.median()
        }
        
        # Gap similarity metrics
        gap_mean_diff = abs(orig_gap_stats['mean'] - recon_gap_stats['mean'])
        gap_std_diff = abs(orig_gap_stats['std'] - recon_gap_stats['std'])
        
        return {
            'original_gap_stats': orig_gap_stats,
            'reconstructed_gap_stats': recon_gap_stats,
            'gap_mean_difference': gap_mean_diff,
            'gap_std_difference': gap_std_diff,
            'temporal_consistency': 1.0 / (1.0 + gap_mean_diff + gap_std_diff)
        }
    
    def _calculate_quality_metrics(self, df_original: pd.DataFrame,
                                 df_reconstructed: pd.DataFrame) -> Dict:
        """Calculate overall quality metrics."""
        # Logical consistency
        logical_consistency = self._check_logical_consistency(df_reconstructed)
        
        # Redundancy check
        redundancy_score = self._calculate_redundancy_score(df_reconstructed)
        
        # Plausibility check
        plausibility_score = self._calculate_plausibility_score(df_original, df_reconstructed)
        
        return {
            'logical_consistency': logical_consistency,
            'redundancy_score': redundancy_score,
            'plausibility_score': plausibility_score,
            'overall_quality': (logical_consistency + redundancy_score + plausibility_score) / 3
        }
    
    def _check_logical_consistency(self, df: pd.DataFrame) -> float:
        """Check logical consistency of reconstructed events."""
        events = df[self.event_column].values
        consistency_score = 1.0
        
        # Check for impossible transitions
        impossible_transitions = [
            ('end', 'start'),
            ('complete', 'begin'),
            ('finish', 'init'),
            ('close', 'open'),
            ('logout', 'login')
        ]
        
        violations = 0
        total_transitions = 0
        
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            for impossible_pair in impossible_transitions:
                if current_event == impossible_pair[0] and next_event == impossible_pair[1]:
                    violations += 1
                    break
            
            total_transitions += 1
        
        if total_transitions > 0:
            consistency_score = 1.0 - (violations / total_transitions)
        
        return max(0.0, consistency_score)
    
    def _calculate_redundancy_score(self, df: pd.DataFrame) -> float:
        """Calculate redundancy score (lower is better, so we return 1 - redundancy)."""
        events = df[self.event_column].values
        
        # Calculate event type distribution
        event_counts = {}
        for event in events:
            event_counts[event] = event_counts.get(event, 0) + 1
        
        # Calculate entropy (higher entropy = less redundancy)
        total_events = len(events)
        entropy = 0.0
        
        for count in event_counts.values():
            if count > 0:
                prob = count / total_events
                entropy -= prob * np.log2(prob)
        
        max_entropy = np.log2(len(event_counts)) if len(event_counts) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy  # Higher entropy = better diversity
    
    def _calculate_plausibility_score(self, df_original: pd.DataFrame,
                                    df_reconstructed: pd.DataFrame) -> float:
        """Calculate plausibility score based on similarity to original patterns."""
        # Compare event type frequencies
        orig_freq = df_original[self.event_column].value_counts(normalize=True)
        recon_freq = df_reconstructed[self.event_column].value_counts(normalize=True)
        
        # Calculate frequency similarity
        all_events = set(orig_freq.index) | set(recon_freq.index)
        freq_diff = 0.0
        
        for event in all_events:
            orig_prob = orig_freq.get(event, 0)
            recon_prob = recon_freq.get(event, 0)
            freq_diff += abs(orig_prob - recon_prob)
        
        frequency_similarity = 1.0 - (freq_diff / 2.0)  # Max possible diff is 2.0
        
        return max(0.0, frequency_similarity)
    
    def _calculate_confidence_metrics(self, df_reconstructed: pd.DataFrame) -> Dict:
        """Calculate confidence-related metrics."""
        if 'reconstruction_confidence' not in df_reconstructed.columns:
            return {'mean_confidence': 0.0, 'confidence_distribution': 'N/A'}
        
        confidences = df_reconstructed['reconstruction_confidence']
        
        # Filter out original events (confidence = 1.0)
        reconstructed_only = confidences[confidences < 1.0]
        
        if len(reconstructed_only) > 0:
            return {
                'mean_confidence': reconstructed_only.mean(),
                'median_confidence': reconstructed_only.median(),
                'min_confidence': reconstructed_only.min(),
                'max_confidence': reconstructed_only.max(),
                'std_confidence': reconstructed_only.std(),
                'high_confidence_ratio': (reconstructed_only >= 0.8).mean(),
                'low_confidence_ratio': (reconstructed_only < 0.5).mean()
            }
        else:
            return {
                'mean_confidence': 1.0,
                'median_confidence': 1.0,
                'min_confidence': 1.0,
                'max_confidence': 1.0,
                'std_confidence': 0.0,
                'high_confidence_ratio': 1.0,
                'low_confidence_ratio': 0.0
            }
    
    def _calculate_overall_score(self, evaluation: Dict) -> float:
        """Calculate overall reconstruction score."""
        weights = {
            'accuracy': 0.3,
            'completeness': 0.25,
            'temporal': 0.2,
            'quality': 0.15,
            'confidence': 0.1
        }
        
        scores = {
            'accuracy': evaluation['accuracy_metrics'].get('f1_score', 0),
            'completeness': evaluation['completeness_metrics'].get('sequence_completeness', 0),
            'temporal': evaluation['temporal_metrics'].get('temporal_consistency', 0),
            'quality': evaluation['quality_metrics'].get('overall_quality', 0),
            'confidence': evaluation['confidence_metrics'].get('mean_confidence', 0)
        }
        
        overall_score = sum(weights[key] * scores[key] for key in weights)
        
        return overall_score
    
    def compare_reconstruction_methods(self, 
                                     results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple reconstruction methods.
        
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
                'F1 Score': results['accuracy_metrics'].get('f1_score', 0),
                'Precision': results['accuracy_metrics'].get('precision', 0),
                'Recall': results['accuracy_metrics'].get('recall', 0),
                'Completeness': results['completeness_metrics'].get('sequence_completeness', 0),
                'Temporal Consistency': results['temporal_metrics'].get('temporal_consistency', 0),
                'Quality Score': results['quality_metrics'].get('overall_quality', 0),
                'Mean Confidence': results['confidence_metrics'].get('mean_confidence', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
        
        return comparison_df
    
    def generate_evaluation_report(self, evaluation: Dict) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation (Dict): Evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        report = f"""
        Event Reconstruction Evaluation Report
        =====================================
        
        Overall Score: {evaluation['overall_score']:.3f}
        
        Basic Metrics:
        --------------
        • Original Events: {evaluation['basic_metrics']['original_events']:,}
        • Reconstructed Events: {evaluation['basic_metrics']['reconstructed_events']:,}
        • Events Added: {evaluation['basic_metrics']['events_added']:,}
        • Reconstruction Ratio: {evaluation['basic_metrics']['reconstruction_ratio']:.3f}
        • Event Type Preservation: {evaluation['basic_metrics']['event_type_preservation_rate']:.3f}
        
        Accuracy Metrics:
        ----------------
        • Precision: {evaluation['accuracy_metrics']['precision']:.3f}
        • Recall: {evaluation['accuracy_metrics']['recall']:.3f}
        • F1 Score: {evaluation['accuracy_metrics']['f1_score']:.3f}
        • Event Type Accuracy: {evaluation['accuracy_metrics']['event_type_accuracy']:.3f}
        
        Completeness Metrics:
        --------------------
        • Time Completeness: {evaluation['completeness_metrics']['time_completeness']:.3f}
        • Sequence Completeness: {evaluation['completeness_metrics']['sequence_completeness']:.3f}
        • Event Order Preservation: {evaluation['completeness_metrics']['event_order_preserved']:.3f}
        
        Temporal Metrics:
        ----------------
        • Temporal Consistency: {evaluation['temporal_metrics']['temporal_consistency']:.3f}
        • Gap Mean Difference: {evaluation['temporal_metrics']['gap_mean_difference']:.3f}s
        • Gap Std Difference: {evaluation['temporal_metrics']['gap_std_difference']:.3f}s
        
        Quality Metrics:
        ---------------
        • Logical Consistency: {evaluation['quality_metrics']['logical_consistency']:.3f}
        • Redundancy Score: {evaluation['quality_metrics']['redundancy_score']:.3f}
        • Plausibility Score: {evaluation['quality_metrics']['plausibility_score']:.3f}
        • Overall Quality: {evaluation['quality_metrics']['overall_quality']:.3f}
        
        Confidence Metrics:
        ------------------
        • Mean Confidence: {evaluation['confidence_metrics']['mean_confidence']:.3f}
        • High Confidence Ratio: {evaluation['confidence_metrics']['high_confidence_ratio']:.3f}
        • Low Confidence Ratio: {evaluation['confidence_metrics']['low_confidence_ratio']:.3f}
        """
        
        return report
