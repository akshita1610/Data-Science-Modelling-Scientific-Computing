"""
Sequence Analyzer Module

Provides comprehensive analysis of event sequences including alignment,
similarity measures, and sequence statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter
from itertools import combinations

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """Analyzes event sequences with various metrics and comparisons."""
    
    def __init__(self, config: Dict):
        self.config = config.get('analysis', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
    
    def analyze_sequence(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive sequence analysis."""
        events = df[self.event_column].values
        
        analysis = {
            'basic_stats': self._calculate_basic_stats(events),
            'complexity_metrics': self._calculate_complexity_metrics(events),
            'transition_analysis': self._analyze_transitions(events),
            'repetition_analysis': self._analyze_repetitions(events),
            'entropy_analysis': self._calculate_entropy_metrics(events)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, events: List) -> Dict:
        """Calculate basic sequence statistics."""
        return {
            'length': len(events),
            'unique_events': len(set(events)),
            'most_common_event': Counter(events).most_common(1)[0] if events else None,
            'event_diversity': len(set(events)) / len(events) if events else 0
        }
    
    def _calculate_complexity_metrics(self, events: List) -> Dict:
        """Calculate complexity metrics."""
        if len(events) < 2:
            return {'entropy': 0, 'compression_ratio': 1}
        
        # Shannon entropy
        event_counts = Counter(events)
        probs = np.array([count/len(events) for count in event_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Simple compression ratio approximation
        unique_substrings = len(set(tuple(events[i:i+3]) for i in range(len(events)-2)))
        compression_ratio = len(events) / unique_substrings if unique_substrings > 0 else 1
        
        return {
            'entropy': entropy,
            'max_entropy': np.log2(len(set(events))),
            'normalized_entropy': entropy / np.log2(len(set(events))) if len(set(events)) > 1 else 0,
            'compression_ratio': compression_ratio
        }
    
    def _analyze_transitions(self, events: List) -> Dict:
        """Analyze event transitions."""
        if len(events) < 2:
            return {'total_transitions': 0}
        
        transitions = [(events[i], events[i+1]) for i in range(len(events)-1)]
        transition_counts = Counter(transitions)
        
        return {
            'total_transitions': len(transitions),
            'unique_transitions': len(set(transitions)),
            'most_common_transition': transition_counts.most_common(1)[0] if transitions else None,
            'transition_diversity': len(set(transitions)) / len(transitions)
        }
    
    def _analyze_repetitions(self, events: List) -> Dict:
        """Analyze repetition patterns."""
        if not events:
            return {'total_repetitions': 0}
        
        # Find consecutive repetitions
        consecutive_reps = []
        current_count = 1
        
        for i in range(1, len(events)):
            if events[i] == events[i-1]:
                current_count += 1
            else:
                if current_count > 1:
                    consecutive_reps.append(current_count)
                current_count = 1
        
        if current_count > 1:
            consecutive_reps.append(current_count)
        
        return {
            'total_repetitions': len(consecutive_reps),
            'avg_repetition_length': np.mean(consecutive_reps) if consecutive_reps else 0,
            'max_repetition_length': max(consecutive_reps) if consecutive_reps else 0
        }
    
    def _calculate_entropy_metrics(self, events: List) -> Dict:
        """Calculate various entropy metrics."""
        if len(events) < 2:
            return {'shannon_entropy': 0}
        
        # Shannon entropy
        event_counts = Counter(events)
        probs = np.array([count/len(events) for count in event_counts.values()])
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Sample entropy (simplified)
        sample_entropy = self._calculate_sample_entropy(events, m=2, r=0.2)
        
        return {
            'shannon_entropy': shannon_entropy,
            'sample_entropy': sample_entropy
        }
    
    def _calculate_sample_entropy(self, events: List, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (simplified version)."""
        if len(events) < m + 1:
            return 0
        
        # Convert events to numerical codes
        event_to_code = {event: i for i, event in enumerate(set(events))}
        coded_events = [event_to_code[event] for event in events]
        
        # Create template vectors
        templates_m = [coded_events[i:i+m] for i in range(len(coded_events)-m+1)]
        templates_m1 = [coded_events[i:i+m+1] for i in range(len(coded_events)-m)]
        
        # Count matches
        tolerance = r * np.std(coded_events)
        
        def count_matches(templates):
            matches = 0
            for i, template1 in enumerate(templates):
                for template2 in templates[i+1:]:
                    if np.max(np.abs(np.array(template1) - np.array(template2))) <= tolerance:
                        matches += 1
            return matches
        
        matches_m = count_matches(templates_m)
        matches_m1 = count_matches(templates_m1)
        
        if matches_m == 0 or matches_m1 == 0:
            return 0
        
        return -np.log(matches_m1 / matches_m)
    
    def compare_sequences(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Compare two event sequences."""
        events1 = df1[self.event_column].values
        events2 = df2[self.event_column].values
        
        comparison = {
            'length_ratio': len(events1) / len(events2) if len(events2) > 0 else 0,
            'event_overlap': len(set(events1) & set(events2)) / len(set(events1) | set(events2)) if events1 or events2 else 0,
            'jaccard_similarity': self._calculate_jaccard_similarity(events1, events2),
            'edit_distance': self._calculate_edit_distance(events1, events2),
            'cosine_similarity': self._calculate_cosine_similarity(events1, events2)
        }
        
        return comparison
    
    def _calculate_jaccard_similarity(self, seq1: List, seq2: List) -> float:
        """Calculate Jaccard similarity between sequences."""
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0
    
    def _calculate_edit_distance(self, seq1: List, seq2: List) -> int:
        """Calculate edit distance between sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _calculate_cosine_similarity(self, seq1: List, seq2: List) -> float:
        """Calculate cosine similarity between sequences."""
        # Create frequency vectors
        all_events = list(set(seq1) | set(seq2))
        
        vec1 = [seq1.count(event) for event in all_events]
        vec2 = [seq2.count(event) for event in all_events]
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def find_similar_subsequences(self, df: pd.DataFrame, 
                                 min_length: int = 3, 
                                 min_similarity: float = 0.7) -> List[Dict]:
        """Find similar subsequences within the sequence."""
        events = df[self.event_column].values
        similar_pairs = []
        
        for i in range(len(events) - min_length):
            for j in range(i + min_length, len(events) - min_length):
                # Compare subsequences of different lengths
                for length in range(min_length, min(len(events) - i, len(events) - j) + 1):
                    subseq1 = events[i:i+length]
                    subseq2 = events[j:j+length]
                    
                    similarity = self._calculate_jaccard_similarity(subseq1, subseq2)
                    
                    if similarity >= min_similarity:
                        similar_pairs.append({
                            'start1': i,
                            'end1': i + length - 1,
                            'start2': j,
                            'end2': j + length - 1,
                            'length': length,
                            'similarity': similarity,
                            'subsequence1': subseq1,
                            'subsequence2': subseq2
                        })
        
        return similar_pairs
