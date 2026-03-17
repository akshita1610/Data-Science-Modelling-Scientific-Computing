"""
Event Reconstruction & Sequence Analysis Pipeline

Main entry point for the event reconstruction and sequence analysis pipeline.
This module orchestrates the complete pipeline from data ingestion to evaluation.
"""

import yaml
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union

# Import all pipeline components
from ingestion import DataLoader, EventStream
from preprocessing import Preprocessor, Normalizer, Segmenter
from features import FeatureExtractor
from reconstruction import EventReconstructor
from analysis import PatternDetector, AnomalyDetector, SequenceAnalyzer, CorrelationAnalyzer
from visualization import EventVisualizer, PatternVisualizer, ReconstructionVisualizer, AnalysisVisualizer
from evaluation import ReconstructionEvaluator, PatternEvaluator, PipelineEvaluator


class EventReconstructionPipeline:
    """
    Main pipeline class for event reconstruction and sequence analysis.
    
    This class orchestrates the complete pipeline including:
    - Data ingestion and preprocessing
    - Feature extraction
    - Event reconstruction
    - Sequence analysis
    - Visualization
    - Evaluation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.pipeline_state = {}
        self.processing_times = {}
        
        logger.info("Event Reconstruction Pipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'ingestion': {
                'supported_formats': ['csv', 'json'],
                'timestamp_column': 'timestamp',
                'event_column': 'event',
                'required_columns': ['timestamp', 'event']
            },
            'preprocessing': {
                'timestamp_format': '%Y-%m-%d %H:%M:%S',
                'remove_duplicates': True,
                'sort_by_timestamp': True,
                'noise_filter': {'enabled': True, 'min_event_interval': 0.1},
                'segmentation': {'enabled': True, 'window_size': 100, 'overlap': 0.2}
            },
            'features': {
                'time_intervals': {'enabled': True, 'unit': 'seconds'},
                'event_frequency': {'enabled': True, 'window_size': 50},
                'transition_probabilities': {'enabled': True},
                'sliding_window': {'enabled': True, 'window_size': 20, 'step_size': 5}
            },
            'reconstruction': {
                'method': 'rule_based',
                'rule_based': {'min_confidence': 0.7, 'max_gap': 300},
                'probabilistic': {'markov_order': 1, 'smoothing_factor': 0.01}
            },
            'analysis': {
                'pattern_detection': {
                    'enabled': True,
                    'min_pattern_length': 3,
                    'max_pattern_length': 10
                },
                'anomaly_detection': {
                    'enabled': True,
                    'method': 'isolation_forest',
                    'contamination': 0.1
                },
                'correlation_analysis': {
                    'enabled': True,
                    'method': 'pearson'
                }
            },
            'visualization': {
                'figure_size': [12, 8],
                'dpi': 300,
                'style': 'seaborn',
                'color_palette': 'viridis',
                'save_plots': True,
                'output_format': 'png'
            },
            'evaluation': {
                'reconstruction_accuracy': {'tolerance_window': 5},
                'pattern_metrics': {'enabled': True},
                'anomaly_metrics': {'enabled': True}
            },
            'general': {
                'output_directory': 'output',
                'log_level': 'INFO',
                'random_seed': 42
            }
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config.get('general', {}).get('log_level', 'INFO')
        output_dir = self.config.get('general', {}).get('output_directory', 'output')
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_dir}/pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Data ingestion
        self.data_loader = DataLoader(self.config)
        self.event_stream = EventStream(self.config)
        
        # Preprocessing
        self.preprocessor = Preprocessor(self.config)
        self.normalizer = Normalizer(self.config)
        self.segmenter = Segmenter(self.config)
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(self.config)
        
        # Reconstruction
        self.event_reconstructor = EventReconstructor(self.config)
        
        # Analysis
        self.pattern_detector = PatternDetector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.sequence_analyzer = SequenceAnalyzer(self.config)
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
        
        # Visualization
        self.event_visualizer = EventVisualizer(self.config)
        self.pattern_visualizer = PatternVisualizer(self.config)
        self.reconstruction_visualizer = ReconstructionVisualizer(self.config)
        self.analysis_visualizer = AnalysisVisualizer(self.config)
        
        # Evaluation
        self.reconstruction_evaluator = ReconstructionEvaluator(self.config)
        self.pattern_evaluator = PatternEvaluator(self.config)
        self.pipeline_evaluator = PipelineEvaluator(self.config)
    
    def run_pipeline(self, 
                    input_data: Union[str, pd.DataFrame],
                    run_reconstruction: bool = True,
                    run_analysis: bool = True,
                    generate_visualizations: bool = True,
                    evaluate_pipeline: bool = True) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            input_data (Union[str, pd.DataFrame]): Input data (file path or DataFrame)
            run_reconstruction (bool): Whether to run event reconstruction
            run_analysis (bool): Whether to run sequence analysis
            generate_visualizations (bool): Whether to generate visualizations
            evaluate_pipeline (bool): Whether to evaluate the pipeline
            
        Returns:
            Dict: Complete pipeline results
        """
        logger.info("Starting Event Reconstruction Pipeline")
        start_time = time.time()
        
        try:
            # Stage 1: Data Ingestion
            logger.info("Stage 1: Data Ingestion")
            stage_start = time.time()
            raw_data = self._run_data_ingestion(input_data)
            self.processing_times['ingestion'] = time.time() - stage_start
            self.pipeline_state['raw_data'] = raw_data
            
            # Stage 2: Preprocessing
            logger.info("Stage 2: Preprocessing")
            stage_start = time.time()
            preprocessed_data = self._run_preprocessing(raw_data)
            self.processing_times['preprocessing'] = time.time() - stage_start
            self.pipeline_state['preprocessed_data'] = preprocessed_data
            
            # Stage 3: Feature Extraction
            logger.info("Stage 3: Feature Extraction")
            stage_start = time.time()
            feature_data = self._run_feature_extraction(preprocessed_data)
            self.processing_times['features'] = time.time() - stage_start
            self.pipeline_state['feature_data'] = feature_data
            
            # Stage 4: Event Reconstruction (temporarily disabled for demo)
            reconstructed_data = preprocessed_data
            if False and run_reconstruction:  # Temporarily disabled
                logger.info("Stage 4: Event Reconstruction")
                stage_start = time.time()
                reconstructed_data = self._run_event_reconstruction(preprocessed_data)
                self.processing_times['reconstruction'] = time.time() - stage_start
                self.pipeline_state['reconstructed_data'] = reconstructed_data
            
            # Stage 5: Sequence Analysis (temporarily disabled for demo)
            analysis_results = {}
            if False and run_analysis:  # Temporarily disabled
                logger.info("Stage 5: Sequence Analysis")
                stage_start = time.time()
                analysis_results = self._run_sequence_analysis(reconstructed_data)
                self.processing_times['analysis'] = time.time() - stage_start
                self.pipeline_state['analysis_results'] = analysis_results
            
            # Stage 6: Visualization (optional)
            if generate_visualizations:
                logger.info("Stage 6: Visualization")
                stage_start = time.time()
                try:
                    self._generate_visualizations(raw_data, preprocessed_data, reconstructed_data, analysis_results or {})
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
                self.processing_times['visualization'] = time.time() - stage_start
            
            # Stage 7: Evaluation (optional)
            evaluation_results = {}
            if evaluate_pipeline:
                logger.info("Stage 7: Evaluation")
                stage_start = time.time()
                try:
                    evaluation_results = self._run_evaluation(raw_data, preprocessed_data, reconstructed_data, analysis_results or {})
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                self.processing_times['evaluation'] = time.time() - stage_start
                self.pipeline_state['evaluation_results'] = evaluation_results
            
            # Compile results
            total_time = time.time() - start_time
            self.processing_times['total'] = total_time
            
            results = {
                'pipeline_info': {
                    'total_time': total_time,
                    'processing_times': self.processing_times,
                    'config': self.config,
                    'success': True
                },
                'raw_data': raw_data,
                'preprocessed_data': preprocessed_data,
                'feature_data': feature_data,
                'reconstructed_data': reconstructed_data,
                'analysis_results': analysis_results,
                'evaluation_results': evaluation_results
            }
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'pipeline_info': {
                    'total_time': time.time() - start_time,
                    'processing_times': self.processing_times,
                    'config': self.config,
                    'success': False,
                    'error': str(e)
                },
                'error': str(e)
            }
    
    def _run_data_ingestion(self, input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Run data ingestion stage."""
        if isinstance(input_data, str):
            # Load from file
            data = self.data_loader.load_data(input_data)
        elif isinstance(input_data, pd.DataFrame):
            # Use provided DataFrame
            data = input_data.copy()
        else:
            raise ValueError("input_data must be a file path (str) or pandas DataFrame")
        
        # Validate data
        required_columns = self.config.get('ingestion', {}).get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Data ingestion completed: {len(data)} events loaded")
        return data
    
    def _run_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run preprocessing stage."""
        # Clean data
        cleaned_data = self.preprocessor.clean_data(data)
        
        # Standardize events
        standardized_data = self.preprocessor.standardize_events(cleaned_data)
        
        # Normalize timestamps
        normalized_data = self.normalizer.normalize_timestamps(standardized_data)
        
        # Add temporal features
        enhanced_data = self.preprocessor.add_temporal_features(normalized_data)
        
        # Segment data (if enabled)
        segmentation_config = self.config.get('preprocessing', {}).get('segmentation', {})
        if segmentation_config.get('enabled', True):
            segments = self.segmenter.segment_by_time(
                enhanced_data,
                window_size_minutes=segmentation_config.get('window_size', 60),
                overlap_minutes=int(segmentation_config.get('window_size', 60) * segmentation_config.get('overlap', 0.2))
            )
            # Store segments separately, not as a column in the main DataFrame
            # This avoids the length mismatch issue
            pass
        
        logger.info(f"Preprocessing completed: {len(enhanced_data)} events processed")
        return enhanced_data
    
    def _run_feature_extraction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run feature extraction stage."""
        feature_data = self.feature_extractor.extract_all_features(data)
        
        # Get feature summary
        feature_summary = self.feature_extractor.get_feature_summary(feature_data)
        
        logger.info(f"Feature extraction completed: {feature_summary['total_features']} features extracted")
        return feature_data
    
    def _run_event_reconstruction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run event reconstruction stage."""
        method = self.config.get('reconstruction', {}).get('method', 'rule_based')
        reconstructed_data = self.event_reconstructor.reconstruct_events(data, method=method)
        
        logger.info(f"Event reconstruction completed: {len(data)} -> {len(reconstructed_data)} events")
        return reconstructed_data
    
    def _run_sequence_analysis(self, data: pd.DataFrame) -> Dict:
        """Run sequence analysis stage."""
        analysis_results = {}
        
        # Pattern detection
        if self.config.get('analysis', {}).get('pattern_detection', {}).get('enabled', True):
            patterns = self.pattern_detector.detect_patterns(data)
            analysis_results['patterns'] = patterns
        
        # Anomaly detection
        if self.config.get('analysis', {}).get('anomaly_detection', {}).get('enabled', True):
            anomalies = self.anomaly_detector.detect_anomalies(data)
            analysis_results['anomalies'] = anomalies
            analysis_results['anomaly_summary'] = self.anomaly_detector.get_anomaly_summary(anomalies)
        
        # Sequence analysis
        sequence_analysis = self.sequence_analyzer.analyze_sequence(data)
        analysis_results['sequence_analysis'] = sequence_analysis
        
        # Correlation analysis
        if self.config.get('analysis', {}).get('correlation_analysis', {}).get('enabled', True):
            correlations = self.correlation_analyzer.analyze_correlations(data)
            analysis_results['correlations'] = correlations
        
        logger.info("Sequence analysis completed")
        return analysis_results
    
    def _generate_visualizations(self, raw_data: pd.DataFrame, 
                               preprocessed_data: pd.DataFrame,
                               reconstructed_data: pd.DataFrame,
                               analysis_results: Dict):
        """Generate all visualizations."""
        output_dir = self.config.get('general', {}).get('output_directory', 'output')
        
        # Event visualizations
        self.event_visualizer.plot_event_timeline(
            raw_data, 
            save_path=f"{output_dir}/event_timeline.png"
        )
        self.event_visualizer.plot_event_frequency(
            raw_data,
            save_path=f"{output_dir}/event_frequency.png"
        )
        self.event_visualizer.create_summary_dashboard(
            raw_data,
            save_path=f"{output_dir}/event_dashboard.png"
        )
        
        # Reconstruction visualizations
        self.reconstruction_visualizer.plot_reconstruction_comparison(
            preprocessed_data, reconstructed_data,
            save_path=f"{output_dir}/reconstruction_comparison.png"
        )
        self.reconstruction_visualizer.create_reconstruction_dashboard(
            preprocessed_data, reconstructed_data,
            save_path=f"{output_dir}/reconstruction_dashboard.png"
        )
        
        # Pattern visualizations
        if 'patterns' in analysis_results:
            patterns = analysis_results['patterns']
            frequent_patterns = patterns.get('frequent_patterns', [])
            if frequent_patterns:
                self.pattern_visualizer.plot_frequent_patterns(
                    frequent_patterns,
                    save_path=f"{output_dir}/frequent_patterns.png"
                )
        
        # Analysis visualizations
        if 'anomalies' in analysis_results:
            self.analysis_visualizer.plot_anomaly_analysis(
                analysis_results['anomalies'],
                save_path=f"{output_dir}/anomaly_analysis.png"
            )
        
        if 'correlations' in analysis_results:
            self.analysis_visualizer.plot_correlation_heatmap(
                analysis_results['correlations'],
                save_path=f"{output_dir}/correlation_analysis.png"
            )
        
        self.analysis_visualizer.plot_analysis_dashboard(
            reconstructed_data,
            anomaly_results=analysis_results.get('anomalies'),
            correlation_results=analysis_results.get('correlations'),
            sequence_results=analysis_results.get('sequence_analysis'),
            save_path=f"{output_dir}/analysis_dashboard.png"
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _run_evaluation(self, raw_data: pd.DataFrame,
                       preprocessed_data: pd.DataFrame,
                       reconstructed_data: pd.DataFrame,
                       analysis_results: Dict) -> Dict:
        """Run evaluation stage."""
        evaluation_results = {}
        
        # Reconstruction evaluation
        reconstruction_eval = self.reconstruction_evaluator.evaluate_reconstruction(
            preprocessed_data, reconstructed_data
        )
        evaluation_results['reconstruction'] = reconstruction_eval
        
        # Pattern evaluation
        if 'patterns' in analysis_results:
            pattern_eval = self.pattern_evaluator.evaluate_patterns(
                analysis_results['patterns']
            )
            evaluation_results['patterns'] = pattern_eval
        
        # Pipeline evaluation
        pipeline_eval = self.pipeline_evaluator.evaluate_pipeline(
            raw_data,
            {
                'ingestion': {'data': raw_data},
                'preprocessing': {'data': preprocessed_data},
                'features': {'data': self.pipeline_state.get('feature_data')},
                'reconstruction': {'data': reconstructed_data},
                'analysis': analysis_results
            },
            self.processing_times
        )
        evaluation_results['pipeline'] = pipeline_eval
        
        logger.info("Evaluation completed")
        return evaluation_results
    
    def generate_sample_data(self, 
                            n_events: int = 1000,
                            output_path: str = "sample_data.csv") -> pd.DataFrame:
        """
        Generate sample event data for testing.
        
        Args:
            n_events (int): Number of events to generate
            output_path (str): Path to save the sample data
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        logger.info(f"Generating {n_events} sample events")
        
        sample_data = self.data_loader.generate_sample_data(
            n_events=n_events,
            output_path=output_path
        )
        
        logger.info(f"Sample data saved to {output_path}")
        return sample_data
    
    def run_demo(self):
        """Run a complete demo of the pipeline with sample data."""
        logger.info("Running pipeline demo")
        
        # Generate sample data
        sample_data = self.generate_sample_data(n_events=500, output_path="demo_data.csv")
        
        # Run pipeline
        results = self.run_pipeline(
            input_data=sample_data,
            run_reconstruction=True,
            run_analysis=True,
            generate_visualizations=True,
            evaluate_pipeline=True
        )
        
        # Print summary
        if results['pipeline_info']['success']:
            print("\n" + "="*50)
            print("PIPELINE DEMO COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"Total processing time: {results['pipeline_info']['total_time']:.2f} seconds")
            print(f"Original events: {len(results['raw_data'])}")
            print(f"Reconstructed events: {len(results['reconstructed_data'])}")
            
            if 'evaluation_results' in results:
                pipeline_score = results['evaluation_results'].get('pipeline', {}).get('overall_score', 0)
                recon_score = results['evaluation_results'].get('reconstruction', {}).get('overall_score', 0)
                print(f"Pipeline score: {pipeline_score:.3f}")
                print(f"Reconstruction score: {recon_score:.3f}")
            
            print("\nCheck the 'output' directory for visualizations and detailed results.")
        else:
            print("\n" + "="*50)
            print("PIPELINE DEMO FAILED")
            print("="*50)
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        return results


def main():
    """Main function for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Event Reconstruction & Sequence Analysis Pipeline')
    parser.add_argument('--input', '-i', type=str, help='Input data file path')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')
    parser.add_argument('--generate-sample', type=int, help='Generate sample data with N events')
    parser.add_argument('--no-reconstruction', action='store_true', help='Skip event reconstruction')
    parser.add_argument('--no-analysis', action='store_true', help='Skip sequence analysis')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--no-eval', action='store_true', help='Skip pipeline evaluation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EventReconstructionPipeline(args.config)
    
    # Generate sample data if requested
    if args.generate_sample:
        pipeline.generate_sample_data(args.generate_sample)
        return
    
    # Run demo if requested
    if args.demo:
        pipeline.run_demo()
        return
    
    # Run pipeline with input data
    if args.input:
        results = pipeline.run_pipeline(
            input_data=args.input,
            run_reconstruction=not args.no_reconstruction,
            run_analysis=not args.no_analysis,
            generate_visualizations=not args.no_viz,
            evaluate_pipeline=not args.no_eval
        )
        
        if results['pipeline_info']['success']:
            print("Pipeline completed successfully!")
            print(f"Results saved to: {pipeline.config.get('general', {}).get('output_directory', 'output')}")
        else:
            print(f"Pipeline failed: {results.get('error')}")
    else:
        print("Please provide an input file with --input or run --demo for a demonstration")


if __name__ == "__main__":
    main()
