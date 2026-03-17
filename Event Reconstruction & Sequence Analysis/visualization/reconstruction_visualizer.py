"""
Reconstruction Visualizer Module

Visualizes event reconstruction results and comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ReconstructionVisualizer:
    """
    Visualizes event reconstruction results.
    
    This class provides visualization methods for comparing original and
    reconstructed event sequences, showing reconstruction confidence,
    and displaying reconstruction statistics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ReconstructionVisualizer with configuration.
        
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
    
    def plot_reconstruction_comparison(self, df_original: pd.DataFrame, 
                                    df_reconstructed: pd.DataFrame,
                                    title: str = "Original vs Reconstructed Events",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between original and reconstructed event sequences.
        
        Args:
            df_original (pd.DataFrame): Original event data
            df_reconstructed (pd.DataFrame): Reconstructed event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Original events
        timestamps_orig = pd.to_datetime(df_original[self.timestamp_column])
        events_orig = df_original[self.event_column].values
        
        # Reconstructed events
        timestamps_recon = pd.to_datetime(df_reconstructed[self.timestamp_column])
        events_recon = df_reconstructed[self.event_column].values
        
        # Color mapping
        unique_events = list(set(events_orig) | set(events_recon))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_events)))
        event_colors = dict(zip(unique_events, colors))
        
        # Plot original events
        for i, (timestamp, event) in enumerate(zip(timestamps_orig, events_orig)):
            ax1.scatter(timestamp, i, c=[event_colors[event]], s=100, alpha=0.7, 
                       edgecolor='black', linewidth=1)
        
        ax1.set_title('Original Events')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Event Index')
        ax1.grid(True, alpha=0.3)
        
        # Plot reconstructed events with reconstruction method indicators
        reconstruction_method = df_reconstructed.get('reconstruction_method', 'original')
        methods = list(set(reconstruction_method))
        method_markers = {'original': 'o', 'rule_based': 's', 'probabilistic': '^'}
        
        for i, (timestamp, event, method) in enumerate(zip(timestamps_recon, events_recon, reconstruction_method)):
            marker = method_markers.get(method, 'o')
            alpha = 1.0 if method == 'original' else 0.6
            ax2.scatter(timestamp, i, c=[event_colors[event]], s=100, alpha=alpha,
                       marker=marker, edgecolor='black', linewidth=1)
        
        ax2.set_title('Reconstructed Events')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Event Index')
        ax2.grid(True, alpha=0.3)
        
        # Add legend for reconstruction methods
        method_handles = []
        for method in methods:
            marker = method_markers.get(method, 'o')
            alpha = 1.0 if method == 'original' else 0.6
            handle = plt.scatter([], [], c='gray', s=100, marker=marker, alpha=alpha, label=method)
            method_handles.append(handle)
        
        ax2.legend(handles=method_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"reconstruction_comparison.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Reconstruction comparison plot saved to {output_path}")
        
        return fig
    
    def plot_reconstruction_statistics(self, df_original: pd.DataFrame,
                                     df_reconstructed: pd.DataFrame,
                                     title: str = "Reconstruction Statistics",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot reconstruction statistics and metrics.
        
        Args:
            df_original (pd.DataFrame): Original event data
            df_reconstructed (pd.DataFrame): Reconstructed event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Calculate statistics
        original_count = len(df_original)
        reconstructed_count = len(df_reconstructed)
        added_events = reconstructed_count - original_count
        
        # Event type comparison
        original_events = set(df_original[self.event_column].unique())
        reconstructed_events = set(df_reconstructed[self.event_column].unique())
        common_events = original_events & reconstructed_events
        new_events = reconstructed_events - original_events
        
        # 1. Event count comparison
        categories = ['Original', 'Reconstructed', 'Added']
        counts = [original_count, reconstructed_count, added_events]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax1.bar(categories, counts, color=colors, edgecolor='black')
        ax1.set_title('Event Count Comparison')
        ax1.set_ylabel('Number of Events')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Event type comparison (Venn-like bar chart)
        event_categories = ['Common', 'New', 'Lost']
        event_counts = [len(common_events), len(new_events), len(original_events - common_events)]
        event_colors = ['gold', 'lightgreen', 'lightcoral']
        
        ax2.bar(event_categories, event_counts, color=event_colors, edgecolor='black')
        ax2.set_title('Event Type Changes')
        ax2.set_ylabel('Number of Event Types')
        
        # 3. Time range comparison
        orig_start = pd.to_datetime(df_original[self.timestamp_column]).min()
        orig_end = pd.to_datetime(df_original[self.timestamp_column]).max()
        recon_start = pd.to_datetime(df_reconstructed[self.timestamp_column]).min()
        recon_end = pd.to_datetime(df_reconstructed[self.timestamp_column]).max()
        
        time_ranges = [
            (orig_start, orig_end, 'Original'),
            (recon_start, recon_end, 'Reconstructed')
        ]
        
        for i, (start, end, label) in enumerate(time_ranges):
            ax3.barh(i, (end - start).total_seconds() / 3600, 0.5, 
                    left=start.timestamp() / 3600, label=label, alpha=0.7)
        
        ax3.set_title('Time Range Comparison')
        ax3.set_xlabel('Time (hours from epoch)')
        ax3.set_ylabel('Dataset')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Reconstruction method distribution
        if 'reconstruction_method' in df_reconstructed.columns:
            method_counts = df_reconstructed['reconstruction_method'].value_counts()
            method_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', 
                             colors=plt.cm.Set3(np.linspace(0, 1, len(method_counts))))
            ax4.set_title('Reconstruction Method Distribution')
        else:
            ax4.text(0.5, 0.5, 'No reconstruction method data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Reconstruction Method Distribution')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"reconstruction_statistics.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Reconstruction statistics plot saved to {output_path}")
        
        return fig
    
    def plot_confidence_scores(self, df_reconstructed: pd.DataFrame,
                              title: str = "Reconstruction Confidence Scores",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confidence scores for reconstructed events.
        
        Args:
            df_reconstructed (pd.DataFrame): Reconstructed event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        if 'reconstruction_confidence' not in df_reconstructed.columns:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No confidence scores available', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.title(title)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        confidence_scores = df_reconstructed['reconstruction_confidence']
        timestamps = pd.to_datetime(df_reconstructed[self.timestamp_column])
        
        # Filter out original events (confidence = 1.0)
        reconstructed_only = df_reconstructed[confidence_scores < 1.0]
        
        # 1. Confidence score distribution
        ax1.hist(confidence_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Confidence Score Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(confidence_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidence_scores.mean():.3f}')
        ax1.legend()
        
        # 2. Confidence over time
        ax2.scatter(timestamps, confidence_scores, alpha=0.6, s=50)
        ax2.set_title('Confidence Scores Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Confidence by reconstruction method
        if 'reconstruction_method' in df_reconstructed.columns:
            methods = df_reconstructed['reconstruction_method'].unique()
            for method in methods:
                method_data = df_reconstructed[df_reconstructed['reconstruction_method'] == method]
                method_confidence = method_data['reconstruction_confidence']
                ax3.hist(method_confidence, bins=15, alpha=0.7, label=method, 
                        edgecolor='black')
            
            ax3.set_title('Confidence by Reconstruction Method')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No method data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Confidence by Reconstruction Method')
        
        # 4. Confidence vs event type
        event_confidence = df_reconstructed.groupby(self.event_column)['reconstruction_confidence'].mean()
        event_confidence.plot(kind='bar', ax=ax4, color='lightcoral', edgecolor='darkred')
        ax4.set_title('Average Confidence by Event Type')
        ax4.set_xlabel('Event Type')
        ax4.set_ylabel('Average Confidence')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"confidence_scores.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confidence scores plot saved to {output_path}")
        
        return fig
    
    def plot_gap_analysis(self, df_original: pd.DataFrame,
                         title: str = "Gap Analysis",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot gap analysis showing where events were added during reconstruction.
        
        Args:
            df_original (pd.DataFrame): Original event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Calculate time gaps in original data
        timestamps = pd.to_datetime(df_original[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        
        # 1. Time gap distribution
        ax1.hist(time_gaps, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Original Time Gap Distribution')
        ax1.set_xlabel('Time Gap (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(time_gaps.mean(), color='red', linestyle='--', 
                   label=f'Mean: {time_gaps.mean():.2f}s')
        ax1.axvline(time_gaps.median(), color='green', linestyle='--', 
                   label=f'Median: {time_gaps.median():.2f}s')
        ax1.legend()
        
        # 2. Gap analysis over sequence
        gap_threshold = time_gaps.quantile(0.95)  # Large gaps as potential missing events
        
        # Identify large gaps
        large_gaps = time_gaps > gap_threshold
        large_gap_indices = time_gaps[large_gaps].index
        
        # Plot gaps over sequence
        ax2.plot(range(len(time_gaps)), time_gaps, alpha=0.7, color='blue', linewidth=1)
        ax2.scatter(range(len(time_gaps)), time_gaps, c=large_gaps.map({True: 'red', False: 'blue'}), 
                   s=20, alpha=0.8)
        ax2.axhline(gap_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Gap Threshold: {gap_threshold:.2f}s')
        ax2.set_title('Time Gaps Over Sequence')
        ax2.set_xlabel('Gap Index')
        ax2.set_ylabel('Time Gap (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for largest gaps
        if len(large_gap_indices) > 0:
            top_gaps = time_gaps[large_gap_indices].nlargest(5)
            for idx, gap_size in top_gaps.items():
                ax2.annotate(f'{gap_size:.1f}s', (idx, gap_size), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"gap_analysis.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Gap analysis plot saved to {output_path}")
        
        return fig
    
    def create_reconstruction_dashboard(self, df_original: pd.DataFrame,
                                      df_reconstructed: pd.DataFrame,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive reconstruction dashboard.
        
        Args:
            df_original (pd.DataFrame): Original event data
            df_reconstructed (pd.DataFrame): Reconstructed event data
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Event count comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        categories = ['Original', 'Reconstructed', 'Added']
        counts = [len(df_original), len(df_reconstructed), len(df_reconstructed) - len(df_original)]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = ax1.bar(categories, counts, color=colors, edgecolor='black')
        ax1.set_title('Event Count Comparison')
        ax1.set_ylabel('Number of Events')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Time range comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        orig_start = pd.to_datetime(df_original[self.timestamp_column]).min()
        orig_end = pd.to_datetime(df_original[self.timestamp_column]).max()
        recon_start = pd.to_datetime(df_reconstructed[self.timestamp_column]).min()
        recon_end = pd.to_datetime(df_reconstructed[self.timestamp_column]).max()
        
        ax2.barh(0, (orig_end - orig_start).total_seconds() / 3600, 0.3, 
                label='Original', alpha=0.7, color='lightblue')
        ax2.barh(0.4, (recon_end - recon_start).total_seconds() / 3600, 0.3, 
                label='Reconstructed', alpha=0.7, color='lightgreen')
        ax2.set_title('Time Range Comparison')
        ax2.set_xlabel('Duration (hours)')
        ax2.set_yticks([0, 0.4])
        ax2.set_yticklabels(['Original', 'Reconstructed'])
        ax2.legend()
        
        # 3. Event type changes (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        original_events = set(df_original[self.event_column].unique())
        reconstructed_events = set(df_reconstructed[self.event_column].unique())
        common_events = len(original_events & reconstructed_events)
        new_events = len(reconstructed_events - original_events)
        
        ax3.bar(['Common', 'New'], [common_events, new_events], 
               color=['gold', 'lightgreen'], edgecolor='black')
        ax3.set_title('Event Type Changes')
        ax3.set_ylabel('Number of Event Types')
        
        # 4. Confidence score distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'reconstruction_confidence' in df_reconstructed.columns:
            confidence_scores = df_reconstructed['reconstruction_confidence']
            ax4.hist(confidence_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax4.set_title('Confidence Score Distribution')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
        else:
            ax4.text(0.5, 0.5, 'No confidence scores', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Confidence Score Distribution')
        
        # 5. Reconstruction timeline (middle)
        ax5 = fig.add_subplot(gs[1, 1:])
        timestamps_orig = pd.to_datetime(df_original[self.timestamp_column])
        timestamps_recon = pd.to_datetime(df_reconstructed[self.timestamp_column])
        
        # Plot original events
        ax5.scatter(timestamps_orig, [1] * len(timestamps_orig), 
                   c='blue', s=50, alpha=0.7, label='Original')
        
        # Plot reconstructed events
        if 'reconstruction_method' in df_reconstructed.columns:
            reconstructed_mask = df_reconstructed['reconstruction_method'] != 'original'
            recon_timestamps = timestamps_recon[reconstructed_mask]
            ax5.scatter(recon_timestamps, [1.1] * len(recon_timestamps), 
                       c='red', s=50, alpha=0.7, label='Reconstructed')
        
        ax5.set_title('Reconstruction Timeline')
        ax5.set_xlabel('Time')
        ax5.set_yticks([1, 1.1])
        ax5.set_yticklabels(['Original', 'Reconstructed'])
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Statistics summary (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate statistics
        stats_text = f"""
        Reconstruction Summary:
        ─────────────────────
        Original Events: {len(df_original):,}
        Reconstructed Events: {len(df_reconstructed):,}
        Events Added: {len(df_reconstructed) - len(df_original):,}
        Reconstruction Ratio: {len(df_reconstructed) / len(df_original):.2f}
        
        Event Types:
        ─────────────────────
        Original Types: {len(original_events)}
        Reconstructed Types: {len(reconstructed_events)}
        Common Types: {common_events}
        New Types: {new_events}
        
        Time Analysis:
        ─────────────────────
        Original Duration: {(orig_end - orig_start).total_seconds() / 3600:.1f} hours
        Reconstructed Duration: {(recon_end - recon_start).total_seconds() / 3600:.1f} hours
        Time Coverage: {100 * (recon_end - recon_start).total_seconds() / (orig_end - orig_start).total_seconds():.1f}%
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Event Reconstruction Dashboard', fontsize=16, fontweight='bold')
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"reconstruction_dashboard.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Reconstruction dashboard saved to {output_path}")
        
        return fig
