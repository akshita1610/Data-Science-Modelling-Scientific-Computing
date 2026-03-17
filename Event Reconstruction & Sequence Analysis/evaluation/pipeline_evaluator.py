"""
Pipeline Evaluator Module

Evaluates the overall performance of the event reconstruction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """
    Evaluates the complete event reconstruction pipeline.
    
    This class provides comprehensive metrics for assessing the overall
    performance, efficiency, and effectiveness of the pipeline.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PipelineEvaluator with configuration.
        
        Args:
            config (Dict): Configuration dictionary for evaluation
        """
        self.config = config.get('evaluation', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Performance tracking
        self.performance_metrics = {}
        
    def evaluate_pipeline(self, 
                         input_data: pd.DataFrame,
                         pipeline_results: Dict,
                         processing_times: Optional[Dict] = None) -> Dict:
        """
        Evaluate the complete pipeline performance.
        
        Args:
            input_data (pd.DataFrame): Original input data
            pipeline_results (Dict): Results from all pipeline stages
            processing_times (Optional[Dict]): Processing times for each stage
            
        Returns:
            Dict: Comprehensive pipeline evaluation
        """
        logger.info("Starting comprehensive pipeline evaluation")
        
        evaluation = {
            'performance_metrics': self._evaluate_performance(processing_times),
            'data_quality_metrics': self._evaluate_data_quality(input_data, pipeline_results),
            'stage_metrics': self._evaluate_pipeline_stages(pipeline_results),
            'efficiency_metrics': self._evaluate_efficiency(input_data, pipeline_results, processing_times),
            'robustness_metrics': self._evaluate_robustness(input_data, pipeline_results)
        }
        
        # Overall pipeline score
        evaluation['overall_score'] = self._calculate_pipeline_overall_score(evaluation)
        
        logger.info(f"Pipeline evaluation completed. Overall score: {evaluation['overall_score']:.3f}")
        return evaluation
    
    def _evaluate_performance(self, processing_times: Optional[Dict]) -> Dict:
        """Evaluate pipeline performance metrics."""
        if not processing_times:
            return {
                'total_time': 0,
                'stage_times': {},
                'bottleneck_stage': 'N/A',
                'performance_score': 0
            }
        
        total_time = sum(processing_times.values())
        stage_times = processing_times.copy()
        
        # Find bottleneck
        bottleneck_stage = max(stage_times, key=stage_times.get)
        bottleneck_time = stage_times[bottleneck_stage]
        bottleneck_ratio = bottleneck_time / total_time if total_time > 0 else 0
        
        # Performance score based on efficiency
        # Lower time = higher score (normalized)
        performance_score = 1.0 / (1.0 + total_time / 100)  # Normalize by expected time of 100s
        
        return {
            'total_time': total_time,
            'stage_times': stage_times,
            'bottleneck_stage': bottleneck_stage,
            'bottleneck_ratio': bottleneck_ratio,
            'performance_score': performance_score
        }
    
    def _evaluate_data_quality(self, input_data: pd.DataFrame,
                             pipeline_results: Dict) -> Dict:
        """Evaluate data quality throughout the pipeline."""
        quality_metrics = {}
        
        # Input data quality
        quality_metrics['input_quality'] = self._assess_data_quality(input_data)
        
        # Quality after preprocessing
        if 'preprocessing' in pipeline_results:
            preprocessed_data = pipeline_results['preprocessing'].get('data')
            if preprocessed_data is not None:
                quality_metrics['preprocessed_quality'] = self._assess_data_quality(preprocessed_data)
            else:
                quality_metrics['preprocessed_quality'] = 0
        
        # Quality after reconstruction
        if 'reconstruction' in pipeline_results:
            reconstructed_data = pipeline_results['reconstruction'].get('data')
            if reconstructed_data is not None:
                quality_metrics['reconstructed_quality'] = self._assess_data_quality(reconstructed_data)
            else:
                quality_metrics['reconstructed_quality'] = 0
        
        # Quality improvement
        if 'preprocessed_quality' in quality_metrics and 'input_quality' in quality_metrics:
            quality_metrics['preprocessing_improvement'] = (
                quality_metrics['preprocessed_quality'] - quality_metrics['input_quality']
            )
        
        if 'reconstructed_quality' in quality_metrics and 'preprocessed_quality' in quality_metrics:
            quality_metrics['reconstruction_impact'] = (
                quality_metrics['reconstructed_quality'] - quality_metrics['preprocessed_quality']
            )
        
        return quality_metrics
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess the quality of a DataFrame."""
        if df.empty:
            return 0.0
        
        quality_score = 1.0
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_score -= missing_ratio * 0.3
        
        # Check for duplicates
        duplicate_ratio = df.duplicated().sum() / len(df)
        quality_score -= duplicate_ratio * 0.2
        
        # Check timestamp consistency
        if self.timestamp_column in df.columns:
            timestamps = pd.to_datetime(df[self.timestamp_column])
            if not timestamps.is_monotonic_increasing:
                quality_score -= 0.2
        
        # Check event type consistency
        if self.event_column in df.columns:
            unique_events = df[self.event_column].nunique()
            if unique_events == 0:
                quality_score -= 0.3
        
        return max(0.0, quality_score)
    
    def _evaluate_pipeline_stages(self, pipeline_results: Dict) -> Dict:
        """Evaluate individual pipeline stages."""
        stage_metrics = {}
        
        for stage_name, stage_results in pipeline_results.items():
            stage_metric = {
                'success': stage_results is not None,
                'output_size': len(stage_results.get('data', pd.DataFrame())) if stage_results and 'data' in stage_results else 0,
                'has_metadata': 'metadata' in stage_results if stage_results else False,
                'has_errors': 'errors' in stage_results if stage_results else False
            }
            
            # Stage-specific metrics
            if stage_name == 'ingestion':
                stage_metric.update(self._evaluate_ingestion_stage(stage_results))
            elif stage_name == 'preprocessing':
                stage_metric.update(self._evaluate_preprocessing_stage(stage_results))
            elif stage_name == 'features':
                stage_metric.update(self._evaluate_features_stage(stage_results))
            elif stage_name == 'reconstruction':
                stage_metric.update(self._evaluate_reconstruction_stage(stage_results))
            elif stage_name == 'analysis':
                stage_metric.update(self._evaluate_analysis_stage(stage_results))
            
            stage_metrics[stage_name] = stage_metric
        
        return stage_metrics
    
    def _evaluate_ingestion_stage(self, stage_results: Dict) -> Dict:
        """Evaluate ingestion stage."""
        return {
            'formats_supported': len(stage_results.get('supported_formats', [])),
            'data_loaded': stage_results.get('data') is not None
        }
    
    def _evaluate_preprocessing_stage(self, stage_results: Dict) -> Dict:
        """Evaluate preprocessing stage."""
        return {
            'cleaning_applied': stage_results.get('cleaning_stats', {}).get('events_removed', 0) > 0,
            'normalization_applied': stage_results.get('normalization_stats', {}) != {},
            'segmentation_applied': 'segments' in stage_results
        }
    
    def _evaluate_features_stage(self, stage_results: Dict) -> Dict:
        """Evaluate feature extraction stage."""
        feature_stats = stage_results.get('feature_summary', {})
        return {
            'features_extracted': feature_stats.get('total_features', 0),
            'feature_diversity': feature_stats.get('feature_diversity', 0)
        }
    
    def _evaluate_reconstruction_stage(self, stage_results: Dict) -> Dict:
        """Evaluate reconstruction stage."""
        recon_stats = stage_results.get('reconstruction_stats', {})
        return {
            'events_added': recon_stats.get('events_added', 0),
            'reconstruction_ratio': recon_stats.get('reconstruction_ratio', 0),
            'method_used': recon_stats.get('method', 'unknown')
        }
    
    def _evaluate_analysis_stage(self, stage_results: Dict) -> Dict:
        """Evaluate analysis stage."""
        return {
            'patterns_detected': len(stage_results.get('patterns', {})),
            'anomalies_detected': stage_results.get('anomaly_summary', {}) != {},
            'correlations_computed': stage_results.get('correlations', {}) != {}
        }
    
    def _evaluate_efficiency(self, input_data: pd.DataFrame,
                            pipeline_results: Dict,
                            processing_times: Optional[Dict]) -> Dict:
        """Evaluate pipeline efficiency."""
        input_size = len(input_data)
        
        # Calculate throughput (events per second)
        total_time = sum(processing_times.values()) if processing_times else 1
        throughput = input_size / total_time if total_time > 0 else 0
        
        # Memory efficiency (simplified)
        final_size = 0
        for stage_results in pipeline_results.values():
            if stage_results and 'data' in stage_results:
                final_size = max(final_size, len(stage_results['data']))
        
        memory_efficiency = input_size / final_size if final_size > 0 else 1
        
        # Stage efficiency
        stage_efficiency = {}
        if processing_times:
            for stage, time_taken in processing_times.items():
                # Efficiency inversely proportional to time taken
                stage_efficiency[stage] = 1.0 / (1.0 + time_taken)
        
        return {
            'throughput': throughput,
            'memory_efficiency': memory_efficiency,
            'stage_efficiency': stage_efficiency,
            'size_change_ratio': final_size / input_size if input_size > 0 else 0
        }
    
    def _evaluate_robustness(self, input_data: pd.DataFrame,
                           pipeline_results: Dict) -> Dict:
        """Evaluate pipeline robustness."""
        # Check for errors in any stage
        total_errors = 0
        error_stages = []
        
        for stage_name, stage_results in pipeline_results.items():
            if stage_results and 'errors' in stage_results:
                stage_errors = len(stage_results['errors'])
                total_errors += stage_errors
                if stage_errors > 0:
                    error_stages.append(stage_name)
        
        # Check data consistency across stages
        consistency_score = self._check_data_consistency(pipeline_results)
        
        # Check handling of edge cases
        edge_case_handling = self._evaluate_edge_case_handling(input_data, pipeline_results)
        
        return {
            'total_errors': total_errors,
            'error_stages': error_stages,
            'error_rate': total_errors / len(pipeline_results) if pipeline_results else 0,
            'consistency_score': consistency_score,
            'edge_case_handling': edge_case_handling,
            'robustness_score': (1.0 - total_errors / len(pipeline_results)) * consistency_score if pipeline_results else 0
        }
    
    def _check_data_consistency(self, pipeline_results: Dict) -> float:
        """Check data consistency across pipeline stages."""
        data_sizes = []
        
        for stage_results in pipeline_results.values():
            if stage_results and 'data' in stage_results:
                data_sizes.append(len(stage_results['data']))
        
        if not data_sizes:
            return 0.0
        
        # Check for reasonable size changes
        consistency_score = 1.0
        
        for i in range(1, len(data_sizes)):
            size_change = abs(data_sizes[i] - data_sizes[i-1]) / data_sizes[i-1]
            if size_change > 0.5:  # More than 50% change is suspicious
                consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _evaluate_edge_case_handling(self, input_data: pd.DataFrame,
                                   pipeline_results: Dict) -> float:
        """Evaluate handling of edge cases."""
        edge_case_score = 1.0
        
        # Check empty data handling
        if len(input_data) == 0:
            # Pipeline should handle empty data gracefully
            for stage_results in pipeline_results.values():
                if stage_results is None:
                    edge_case_score -= 0.2
        
        # Check single event handling
        if len(input_data) == 1:
            # Should handle single event without errors
            for stage_results in pipeline_results.values():
                if stage_results and 'errors' in stage_results and stage_results['errors']:
                    edge_case_score -= 0.1
        
        return max(0.0, edge_case_score)
    
    def _calculate_pipeline_overall_score(self, evaluation: Dict) -> float:
        """Calculate overall pipeline score."""
        weights = {
            'performance': 0.25,
            'data_quality': 0.25,
            'stage_success': 0.20,
            'efficiency': 0.15,
            'robustness': 0.15
        }
        
        scores = {
            'performance': evaluation['performance_metrics'].get('performance_score', 0),
            'data_quality': np.mean(list(evaluation['data_quality_metrics'].values())) if evaluation['data_quality_metrics'] else 0,
            'stage_success': sum(1 for stage in evaluation['stage_metrics'].values() if stage.get('success', False)) / len(evaluation['stage_metrics']) if evaluation['stage_metrics'] else 0,
            'efficiency': evaluation['efficiency_metrics'].get('throughput', 0) / 100,  # Normalize
            'robustness': evaluation['robustness_metrics'].get('robustness_score', 0)
        }
        
        overall_score = sum(weights[key] * scores[key] for key in weights)
        
        return overall_score
    
    def benchmark_pipeline(self, 
                         test_datasets: List[pd.DataFrame],
                         pipeline_function) -> Dict:
        """
        Benchmark the pipeline on multiple test datasets.
        
        Args:
            test_datasets (List[pd.DataFrame]): List of test datasets
            pipeline_function: Function to run the pipeline
            
        Returns:
            Dict: Benchmark results
        """
        logger.info(f"Starting pipeline benchmark on {len(test_datasets)} datasets")
        
        benchmark_results = {
            'dataset_results': [],
            'aggregate_metrics': {},
            'performance_summary': {}
        }
        
        total_time = 0
        total_score = 0
        successful_runs = 0
        
        for i, dataset in enumerate(test_datasets):
            logger.info(f"Benchmarking dataset {i+1}/{len(test_datasets)}")
            
            start_time = time.time()
            
            try:
                # Run pipeline
                results = pipeline_function(dataset)
                
                end_time = time.time()
                run_time = end_time - start_time
                total_time += run_time
                
                # Evaluate results
                evaluation = self.evaluate_pipeline(dataset, results, {'total': run_time})
                
                dataset_result = {
                    'dataset_id': i,
                    'dataset_size': len(dataset),
                    'processing_time': run_time,
                    'overall_score': evaluation['overall_score'],
                    'success': True
                }
                
                benchmark_results['dataset_results'].append(dataset_result)
                total_score += evaluation['overall_score']
                successful_runs += 1
                
            except Exception as e:
                logger.error(f"Pipeline failed on dataset {i+1}: {e}")
                
                dataset_result = {
                    'dataset_id': i,
                    'dataset_size': len(dataset),
                    'processing_time': 0,
                    'overall_score': 0,
                    'success': False,
                    'error': str(e)
                }
                
                benchmark_results['dataset_results'].append(dataset_result)
        
        # Calculate aggregate metrics
        if successful_runs > 0:
            benchmark_results['aggregate_metrics'] = {
                'success_rate': successful_runs / len(test_datasets),
                'avg_processing_time': total_time / successful_runs,
                'avg_overall_score': total_score / successful_runs,
                'total_processing_time': total_time
            }
        else:
            benchmark_results['aggregate_metrics'] = {
                'success_rate': 0,
                'avg_processing_time': 0,
                'avg_overall_score': 0,
                'total_processing_time': total_time
            }
        
        # Performance summary by dataset size
        successful_results = [r for r in benchmark_results['dataset_results'] if r['success']]
        
        if successful_results:
            sizes = [r['dataset_size'] for r in successful_results]
            times = [r['processing_time'] for r in successful_results]
            scores = [r['overall_score'] for r in successful_results]
            
            benchmark_results['performance_summary'] = {
                'size_correlation': np.corrcoef(sizes, times)[0, 1] if len(sizes) > 1 else 0,
                'score_correlation': np.corrcoef(sizes, scores)[0, 1] if len(sizes) > 1 else 0,
                'min_dataset_size': min(sizes),
                'max_dataset_size': max(sizes),
                'min_processing_time': min(times),
                'max_processing_time': max(times)
            }
        
        logger.info(f"Pipeline benchmark completed. Success rate: {benchmark_results['aggregate_metrics']['success_rate']:.2f}")
        
        return benchmark_results
    
    def generate_pipeline_report(self, evaluation: Dict) -> str:
        """
        Generate a comprehensive pipeline evaluation report.
        
        Args:
            evaluation (Dict): Pipeline evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        performance = evaluation['performance_metrics']
        data_quality = evaluation['data_quality_metrics']
        stage_metrics = evaluation['stage_metrics']
        efficiency = evaluation['efficiency_metrics']
        robustness = evaluation['robustness_metrics']
        
        report = f"""
        Pipeline Evaluation Report
        ==========================
        
        Overall Score: {evaluation['overall_score']:.3f}
        
        Performance Metrics:
        -------------------
        • Total Processing Time: {performance['total_time']:.2f} seconds
        • Performance Score: {performance['performance_score']:.3f}
        • Bottleneck Stage: {performance['bottleneck_stage']}
        • Bottleneck Ratio: {performance['bottleneck_ratio']:.2f}
        
        Stage Times:
        """
        
        for stage, time_taken in performance['stage_times'].items():
            report += f"  - {stage}: {time_taken:.3f}s\n"
        
        report += f"""
        Data Quality Metrics:
        ---------------------
        • Input Quality: {data_quality.get('input_quality', 0):.3f}
        • Preprocessed Quality: {data_quality.get('preprocessed_quality', 0):.3f}
        • Reconstructed Quality: {data_quality.get('reconstructed_quality', 0):.3f}
        """
        
        if 'preprocessing_improvement' in data_quality:
            report += f"• Preprocessing Improvement: {data_quality['preprocessing_improvement']:.3f}\n"
        
        if 'reconstruction_impact' in data_quality:
            report += f"• Reconstruction Impact: {data_quality['reconstruction_impact']:.3f}\n"
        
        report += f"""
        Stage Metrics:
        --------------
        """
        
        for stage_name, stage_metric in stage_metrics.items():
            report += f"\n{stage_name.capitalize()} Stage:\n"
            report += f"  • Success: {stage_metric.get('success', False)}\n"
            report += f"  • Output Size: {stage_metric.get('output_size', 0)}\n"
            report += f"  • Has Metadata: {stage_metric.get('has_metadata', False)}\n"
            report += f"  • Has Errors: {stage_metric.get('has_errors', False)}\n"
        
        report += f"""
        Efficiency Metrics:
        -------------------
        • Throughput: {efficiency.get('throughput', 0):.2f} events/second
        • Memory Efficiency: {efficiency.get('memory_efficiency', 0):.3f}
        • Size Change Ratio: {efficiency.get('size_change_ratio', 0):.3f}
        
        Stage Efficiency:
        """
        
        for stage, eff_score in efficiency.get('stage_efficiency', {}).items():
            report += f"  - {stage}: {eff_score:.3f}\n"
        
        report += f"""
        Robustness Metrics:
        -------------------
        • Total Errors: {robustness['total_errors']}
        • Error Rate: {robustness['error_rate']:.3f}
        • Consistency Score: {robustness['consistency_score']:.3f}
        • Edge Case Handling: {robustness['edge_case_handling']:.3f}
        • Robustness Score: {robustness['robustness_score']:.3f}
        """
        
        if robustness['error_stages']:
            report += f"• Error Stages: {', '.join(robustness['error_stages'])}\n"
        
        return report
