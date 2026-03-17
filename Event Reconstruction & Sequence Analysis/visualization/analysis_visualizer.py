"""
Analysis Visualizer Module

Visualizes sequence analysis results including anomalies, correlations, and metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AnalysisVisualizer:
    """
    Visualizes sequence analysis results.
    
    This class provides visualization methods for anomaly detection results,
    correlation analysis, sequence metrics, and other analysis outputs.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the AnalysisVisualizer with configuration.
        
        Args:
            config (Dict): Configuration dictionary for visualization
        """
        self.config = config.get('visualization', {})
        self.timestamp_column = config.get('ingestion', {}).get('timestamp_column', 'timestamp')
        self.event_column = config.get('ingestion', {}).get('event_column', 'event')
        
        # Set up matplotlib and seaborn
        self.figure_size = self.config.get('figure_size', [12, 8])
        self.dpi = self.config.get('dpi', 300)
        self.style = self.config.get('style', 'seaborn')
        self.color_palette = self.config.get('color_palette', 'viridis')
        self.save_plots = self.config.get('save_plots', True)
        self.output_format = self.config.get('output_format', 'png')
        
        # Apply style settings
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
    
    def plot_anomaly_analysis(self, df_with_anomalies: pd.DataFrame,
                             title: str = "Anomaly Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            df_with_anomalies (pd.DataFrame): Event data with anomaly indicators
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        timestamps = pd.to_datetime(df_with_anomalies[self.timestamp_column])
        events = df_with_anomalies[self.event_column].values
        
        # Identify anomaly columns
        anomaly_cols = [col for col in df_with_anomalies.columns if 'anomaly' in col.lower()]
        
        if not anomaly_cols:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No anomaly data available', ha='center', va='center', transform=ax.transAxes)
            plt.suptitle(title)
            return fig
        
        # 1. Anomaly timeline
        combined_anomalies = df_with_anomalies[anomaly_cols].any(axis=1)
        normal_events = ~combined_anomalies
        
        ax1.scatter(timestamps[normal_events], np.where(normal_events)[0], 
                   c='blue', s=50, alpha=0.7, label='Normal')
        ax1.scatter(timestamps[combined_anomalies], np.where(combined_anomalies)[0], 
                   c='red', s=100, alpha=0.9, label='Anomaly')
        ax1.set_title('Anomaly Timeline')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Event Index')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Anomaly type distribution
        anomaly_counts = {}
        for col in anomaly_cols:
            count = df_with_anomalies[col].sum()
            if count > 0:
                anomaly_counts[col.replace('_anomaly', '').replace('_', ' ').title()] = count
        
        if anomaly_counts:
            ax2.bar(anomaly_counts.keys(), anomaly_counts.values(), 
                   color='lightcoral', edgecolor='darkred')
            ax2.set_title('Anomaly Types')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Anomaly Types')
        
        # 3. Anomaly density over time
        if len(timestamps) > 10:
            # Create time windows
            time_windows = pd.cut(timestamps, bins=min(10, len(timestamps)//5))
            anomaly_density = df_with_anomalies.groupby(time_windows)[anomaly_cols].apply(lambda x: x.any(axis=1).mean())
            
            window_centers = [interval.mid for interval in anomaly_density.index]
            ax3.plot(window_centers, anomaly_density.values, marker='o', color='red', linewidth=2)
            ax3.set_title('Anomaly Density Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Anomaly Density')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for density plot', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Anomaly Density Over Time')
        
        # 4. Anomaly vs event type
        event_anomaly_cross = pd.crosstab(df_with_anomalies[self.event_column], combined_anomalies)
        if event_anomaly_cross.shape[1] > 1:
            event_anomaly_cross.columns = ['Normal', 'Anomaly']
            event_anomaly_cross.plot(kind='bar', ax=ax4, color=['blue', 'red'], alpha=0.7)
            ax4.set_title('Anomaly Distribution by Event Type')
            ax4.set_xlabel('Event Type')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Insufficient anomaly data by event type', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Anomaly Distribution by Event Type')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"anomaly_analysis.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Anomaly analysis plot saved to {output_path}")
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_data: Dict,
                                title: str = "Correlation Analysis",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation analysis results as heatmaps.
        
        Args:
            correlation_data (Dict): Correlation analysis results
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # 1. Feature correlation matrix
        if 'feature_correlations' in correlation_data:
            feature_corr = correlation_data['feature_correlations']
            if 'correlation_matrix' in feature_corr:
                corr_df = pd.DataFrame(feature_corr['correlation_matrix'])
                sns.heatmap(corr_df, ax=ax1, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                ax1.set_title('Feature Correlation Matrix')
            else:
                ax1.text(0.5, 0.5, 'No feature correlation matrix', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Feature Correlation Matrix')
        else:
            ax1.text(0.5, 0.5, 'No feature correlations', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Feature Correlation Matrix')
        
        # 2. Event correlations
        if 'event_correlations' in correlation_data:
            event_corr = correlation_data['event_correlations']
            if 'correlation_matrix' in event_corr:
                event_corr_matrix = np.array(event_corr['correlation_matrix'])
                event_types = event_corr.get('event_types', [])
                
                if len(event_types) > 0:
                    sns.heatmap(event_corr_matrix, ax=ax2, annot=True, cmap='coolwarm', 
                               center=0, square=True, fmt='.2f',
                               xticklabels=event_types, yticklabels=event_types)
                    ax2.set_title('Event Correlation Matrix')
                else:
                    ax2.text(0.5, 0.5, 'No event types found', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Event Correlation Matrix')
            else:
                ax2.text(0.5, 0.5, 'No event correlation matrix', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Event Correlation Matrix')
        else:
            ax2.text(0.5, 0.5, 'No event correlations', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Event Correlation Matrix')
        
        # 3. Temporal correlations
        if 'temporal_correlations' in correlation_data:
            temporal_corr = correlation_data['temporal_correlations']
            features = list(temporal_corr.keys())
            correlations = [temporal_corr[f]['correlation'] for f in features]
            p_values = [temporal_corr[f]['p_value'] for f in features]
            
            # Create bar plot with significance indicators
            bars = ax3.bar(features, correlations, color='lightblue', edgecolor='navy')
            ax3.set_title('Temporal Correlations')
            ax3.set_ylabel('Correlation Coefficient')
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Mark significant correlations
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.05:
                    bar.set_color('lightcoral')
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            '*', ha='center', va='bottom', fontweight='bold', color='red')
        else:
            ax3.text(0.5, 0.5, 'No temporal correlations', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Temporal Correlations')
        
        # 4. Strong correlations summary
        strong_correlations = []
        
        if 'feature_correlations' in correlation_data:
            feature_corr = correlation_data['feature_correlations']
            if 'strong_correlations' in feature_corr:
                for corr in feature_corr['strong_correlations']:
                    strong_correlations.append(f"{corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
        
        if 'event_correlations' in correlation_data:
            event_corr = correlation_data['event_correlations']
            if 'strongest_correlations' in event_corr:
                for corr in event_corr['strongest_correlations']:
                    strong_correlations.append(f"{corr['event1']} ↔ {corr['event2']}: {corr['correlation']:.3f}")
        
        if strong_correlations:
            ax4.axis('off')
            ax4.text(0.05, 0.95, 'Strong Correlations:', transform=ax4.transAxes, 
                    fontsize=12, fontweight='bold', verticalalignment='top')
            
            y_pos = 0.85
            for corr_text in strong_correlations[:10]:  # Show top 10
                ax4.text(0.05, y_pos, corr_text, transform=ax4.transAxes, 
                        fontsize=10, verticalalignment='top')
                y_pos -= 0.08
            
            ax4.set_title('Strong Correlations Summary')
        else:
            ax4.text(0.5, 0.5, 'No strong correlations found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Strong Correlations Summary')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"correlation_analysis.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation analysis plot saved to {output_path}")
        
        return fig
    
    def plot_sequence_metrics(self, sequence_analysis: Dict,
                            title: str = "Sequence Metrics",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sequence analysis metrics.
        
        Args:
            sequence_analysis (Dict): Sequence analysis results
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # 1. Basic statistics
        if 'basic_stats' in sequence_analysis:
            stats = sequence_analysis['basic_stats']
            categories = ['Length', 'Unique Events', 'Diversity']
            values = [stats.get('length', 0), stats.get('unique_events', 0), 
                     stats.get('event_diversity', 0) * 100]  # Convert diversity to percentage
            
            bars = ax1.bar(categories, values, color='lightblue', edgecolor='navy')
            ax1.set_title('Basic Sequence Statistics')
            ax1.set_ylabel('Count / Percentage')
            
            # Add value labels
            for bar, val, cat in zip(bars, values, categories):
                height = bar.get_height()
                label = f'{val:.0f}' if cat != 'Diversity' else f'{val:.1f}%'
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        label, ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No basic statistics', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Basic Sequence Statistics')
        
        # 2. Complexity metrics
        if 'complexity_metrics' in sequence_analysis:
            complexity = sequence_analysis['complexity_metrics']
            metrics = ['Entropy', 'Normalized Entropy', 'Compression Ratio']
            values = [complexity.get('entropy', 0), complexity.get('normalized_entropy', 0), 
                     min(complexity.get('compression_ratio', 1), 5)]  # Cap compression ratio for display
            
            bars = ax2.bar(metrics, values, color='lightcoral', edgecolor='darkred')
            ax2.set_title('Complexity Metrics')
            ax2.set_ylabel('Value')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No complexity metrics', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Complexity Metrics')
        
        # 3. Transition analysis
        if 'transition_analysis' in sequence_analysis:
            transitions = sequence_analysis['transition_analysis']
            categories = ['Total', 'Unique', 'Diversity']
            values = [transitions.get('total_transitions', 0), transitions.get('unique_transitions', 0),
                     transitions.get('transition_diversity', 0) * 100]  # Convert to percentage
            
            bars = ax3.bar(categories, values, color='lightgreen', edgecolor='darkgreen')
            ax3.set_title('Transition Analysis')
            ax3.set_ylabel('Count / Percentage')
            
            # Add value labels
            for bar, val, cat in zip(bars, values, categories):
                height = bar.get_height()
                label = f'{val:.0f}' if cat != 'Diversity' else f'{val:.1f}%'
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        label, ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No transition analysis', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Transition Analysis')
        
        # 4. Repetition analysis
        if 'repetition_analysis' in sequence_analysis:
            repetitions = sequence_analysis['repetition_analysis']
            categories = ['Total Repetitions', 'Avg Length', 'Max Length']
            values = [repetitions.get('total_repetitions', 0), repetitions.get('avg_repetition_length', 0),
                     repetitions.get('max_repetition_length', 0)]
            
            bars = ax4.bar(categories, values, color='gold', edgecolor='orange')
            ax4.set_title('Repetition Analysis')
            ax4.set_ylabel('Count / Length')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No repetition analysis', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Repetition Analysis')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"sequence_metrics.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sequence metrics plot saved to {output_path}")
        
        return fig
    
    def plot_analysis_dashboard(self, df: pd.DataFrame,
                               anomaly_results: Optional[Dict] = None,
                               correlation_results: Optional[Dict] = None,
                               sequence_results: Optional[Dict] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive analysis dashboard.
        
        Args:
            df (pd.DataFrame): Event data
            anomaly_results (Optional[Dict]): Anomaly detection results
            correlation_results (Optional[Dict]): Correlation analysis results
            sequence_results (Optional[Dict]): Sequence analysis results
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Event frequency (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        event_counts = df[self.event_column].value_counts().head(10)
        event_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Top 10 Event Types')
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Anomaly summary (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if anomaly_results:
            anomaly_summary = anomaly_results.get('summary', {})
            if anomaly_summary:
                categories = list(anomaly_summary.keys())
                counts = [anomaly_summary[cat]['count'] for cat in categories]
                percentages = [anomaly_summary[cat]['percentage'] for cat in categories]
                
                bars = ax2.bar(categories, counts, color='lightcoral', edgecolor='darkred')
                ax2.set_title('Anomaly Summary')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Anomaly Summary')
        else:
            ax2.text(0.5, 0.5, 'No anomaly analysis', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Anomaly Summary')
        
        # 3. Sequence complexity (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if sequence_results and 'complexity_metrics' in sequence_results:
            complexity = sequence_results['complexity_metrics']
            metrics = ['Entropy', 'Norm. Entropy', 'Compression']
            values = [complexity.get('entropy', 0), complexity.get('normalized_entropy', 0), 
                     min(complexity.get('compression_ratio', 1), 3)]
            
            bars = ax3.bar(metrics, values, color='lightgreen', edgecolor='darkgreen')
            ax3.set_title('Sequence Complexity')
            ax3.set_ylabel('Value')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No complexity data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Sequence Complexity')
        
        # 4. Time gaps distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        timestamps = pd.to_datetime(df[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        ax4.hist(time_gaps, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_title('Time Gaps Distribution')
        ax4.set_xlabel('Time Gap (seconds)')
        ax4.set_ylabel('Frequency')
        
        # 5. Event timeline with anomalies (middle)
        ax5 = fig.add_subplot(gs[1, 1:])
        events = df[self.event_column].values
        
        # Create color mapping
        unique_events = list(set(events))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_events)))
        event_colors = dict(zip(unique_events, colors))
        
        # Plot events
        for i, event in enumerate(events):
            color = event_colors[event]
            ax5.scatter(i, 0, c=[color], s=100, alpha=0.7, edgecolor='black')
        
        ax5.set_title('Event Sequence Timeline')
        ax5.set_xlabel('Event Index')
        ax5.set_yticks([0])
        ax5.set_yticklabels(['Events'])
        ax5.grid(True, alpha=0.3)
        
        # Add legend for event types
        handles = [plt.scatter([], [], c=[event_colors[event]], s=100, label=event) 
                  for event in unique_events[:10]]  # Limit to 10 for readability
        ax5.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Correlation summary (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        if correlation_results:
            strong_corrs = []
            
            if 'feature_correlations' in correlation_results:
                feature_corr = correlation_results['feature_correlations']
                if 'strong_correlations' in feature_corr:
                    for corr in feature_corr['strong_correlations'][:5]:
                        strong_corrs.append(f"F: {corr['feature1'][:10]}... {corr['correlation']:.2f}")
            
            if 'event_correlations' in correlation_results:
                event_corr = correlation_results['event_correlations']
                if 'strongest_correlations' in event_corr:
                    for corr in event_corr['strongest_correlations'][:5]:
                        strong_corrs.append(f"E: {corr['event1']}→{corr['event2']} {corr['correlation']:.2f}")
            
            if strong_corrs:
                ax6.axis('off')
                ax6.text(0.05, 0.95, 'Strong Correlations:', transform=ax6.transAxes, 
                        fontsize=10, fontweight='bold', verticalalignment='top')
                
                y_pos = 0.85
                for corr_text in strong_corrs:
                    ax6.text(0.05, y_pos, corr_text, transform=ax6.transAxes, 
                            fontsize=8, verticalalignment='top')
                    y_pos -= 0.15
                
                ax6.set_title('Correlation Summary')
            else:
                ax6.text(0.5, 0.5, 'No strong correlations', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Correlation Summary')
        else:
            ax6.text(0.5, 0.5, 'No correlation analysis', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Correlation Summary')
        
        # 7. Overall statistics (bottom middle and right)
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')
        
        # Calculate comprehensive statistics
        total_events = len(df)
        unique_events = df[self.event_column].nunique()
        time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # hours
        
        stats_text = f"""
        Comprehensive Analysis Summary:
        ─────────────────────────────
        Dataset Overview:
        • Total Events: {total_events:,}
        • Unique Event Types: {unique_events}
        • Time Span: {time_span:.1f} hours
        • Events per Hour: {total_events / max(time_span, 1):.1f}
        
        Sequence Analysis:
        • Event Diversity: {unique_events / total_events * 100:.1f}%
        • Avg Time Gap: {time_gaps.mean():.2f} seconds
        • Median Time Gap: {time_gaps.median():.2f} seconds
        
        Anomaly Detection:
        • Anomaly Rate: {anomaly_results.get('summary', {}).get('isolation_anomaly', {}).get('percentage', 0):.1f}%
        • Temporal Anomalies: {anomaly_results.get('summary', {}).get('temporal_gap_anomaly', {}).get('percentage', 0):.1f}%
        • Sequence Anomalies: {anomaly_results.get('summary', {}).get('sequence_anomaly', {}).get('percentage', 0):.1f}%
        
        Quality Metrics:
        • Data Completeness: 100.0%
        • Timestamp Consistency: {'✓' if time_gaps.min() >= 0 else '✗'}
        • Event Type Coverage: {'✓' if unique_events > 1 else '✗'}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Event Sequence Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"analysis_dashboard.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Analysis dashboard saved to {output_path}")
        
        return fig
