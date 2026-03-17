"""
Event Visualizer Module

Visualizes event sequences, timelines, and basic event distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EventVisualizer:
    """
    Visualizes event sequences and timelines.
    
    This class provides various visualization methods for event data including
    timeline plots, frequency distributions, and temporal patterns.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the EventVisualizer with configuration.
        
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
        
    def plot_event_timeline(self, df: pd.DataFrame, 
                          title: str = "Event Timeline",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot event timeline showing events over time.
        
        Args:
            df (pd.DataFrame): Event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Prepare data
        timestamps = pd.to_datetime(df[self.timestamp_column])
        events = df[self.event_column].values
        
        # Create color mapping for events
        unique_events = list(set(events))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_events)))
        event_colors = dict(zip(unique_events, colors))
        
        # Plot events
        for i, (timestamp, event) in enumerate(zip(timestamps, events)):
            ax.scatter(timestamp, i, c=[event_colors[event]], s=100, alpha=0.7, label=event if i == 0 else "")
            
            # Add event labels for first few events
            if i < min(10, len(events)):
                ax.annotate(event, (timestamp, i), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Event Index')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [plt.scatter([], [], c=[event_colors[event]], s=100, label=event) 
                  for event in unique_events]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"event_timeline.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Timeline plot saved to {output_path}")
        
        return fig
    
    def plot_event_frequency(self, df: pd.DataFrame,
                           title: str = "Event Frequency Distribution",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot event frequency distribution.
        
        Args:
            df (pd.DataFrame): Event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Count events
        event_counts = df[self.event_column].value_counts()
        
        # Bar plot
        event_counts.plot(kind='bar', ax=ax1, color=plt.cm.Set3(np.linspace(0, 1, len(event_counts))))
        ax1.set_title('Event Counts')
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', 
                colors=plt.cm.Set3(np.linspace(0, 1, len(event_counts))))
        ax2.set_title('Event Distribution')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"event_frequency.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Frequency plot saved to {output_path}")
        
        return fig
    
    def plot_temporal_patterns(self, df: pd.DataFrame,
                             title: str = "Temporal Patterns",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot temporal patterns in event data.
        
        Args:
            df (pd.DataFrame): Event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Extract temporal features
        timestamps = pd.to_datetime(df[self.timestamp_column])
        events = df[self.event_column].values
        
        # Hour of day distribution
        hour_counts = timestamps.dt.hour.value_counts().sort_index()
        hour_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Events by Hour of Day')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Count')
        
        # Day of week distribution
        day_counts = timestamps.dt.dayofweek.value_counts().sort_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts.index = [day_names[i] for i in day_counts.index]
        day_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Events by Day of Week')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Count')
        
        # Events over time (line plot)
        daily_counts = timestamps.dt.date.value_counts().sort_index()
        daily_counts.plot(kind='line', ax=ax3, marker='o', color='green')
        ax3.set_title('Events Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Event type by hour heatmap
        hour_event_cross = pd.crosstab(timestamps.dt.hour, events)
        sns.heatmap(hour_event_cross, ax=ax4, cmap='YlOrRd', annot=True, fmt='d')
        ax4.set_title('Event Types by Hour')
        ax4.set_xlabel('Event Type')
        ax4.set_ylabel('Hour')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"temporal_patterns.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Temporal patterns plot saved to {output_path}")
        
        return fig
    
    def plot_time_gaps(self, df: pd.DataFrame,
                      title: str = "Time Gaps Between Events",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of time gaps between events.
        
        Args:
            df (pd.DataFrame): Event data
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Calculate time gaps
        timestamps = pd.to_datetime(df[self.timestamp_column])
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        
        # Histogram of time gaps
        ax1.hist(time_gaps, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Time Gaps')
        ax1.set_xlabel('Time Gap (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Time gaps over time
        ax2.plot(range(len(time_gaps)), time_gaps, marker='o', markersize=3, alpha=0.7)
        ax2.set_title('Time Gaps Over Sequence')
        ax2.set_xlabel('Event Index')
        ax2.set_ylabel('Time Gap (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {time_gaps.mean():.2f}s\nMedian: {time_gaps.median():.2f}s\nStd: {time_gaps.std():.2f}s'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"time_gaps.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Time gaps plot saved to {output_path}")
        
        return fig
    
    def plot_event_sequence(self, df: pd.DataFrame,
                          max_events: int = 100,
                          title: str = "Event Sequence",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot event sequence as a connected timeline.
        
        Args:
            df (pd.DataFrame): Event data
            max_events (int): Maximum number of events to display
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Limit events for readability
        df_plot = df.head(max_events)
        timestamps = pd.to_datetime(df_plot[self.timestamp_column])
        events = df_plot[self.event_column].values
        
        # Create color mapping
        unique_events = list(set(events))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_events)))
        event_colors = dict(zip(unique_events, colors))
        
        # Plot sequence
        x_positions = range(len(events))
        
        # Draw connections
        ax.plot(x_positions, [1] * len(events), 'k-', alpha=0.3, linewidth=1)
        
        # Draw events
        for i, (x, event) in enumerate(zip(x_positions, events)):
            ax.scatter(x, 1, c=[event_colors[event]], s=200, zorder=5, edgecolor='black', linewidth=1)
            ax.text(x, 1.2, event, ha='center', va='bottom', fontsize=8, rotation=45)
            
            # Add timestamp for selected events
            if i % 5 == 0:  # Show every 5th timestamp
                ax.text(x, 0.8, timestamps.iloc[i].strftime('%H:%M'), 
                       ha='center', va='top', fontsize=7, alpha=0.7)
        
        # Formatting
        ax.set_xlim(-0.5, len(events) - 0.5)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel('Event Sequence Position')
        ax.set_ylabel('Events')
        ax.set_title(f'{title} (First {len(events)} events)')
        ax.set_yticks([1])
        ax.set_yticklabels(['Event Sequence'])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [plt.scatter([], [], c=[event_colors[event]], s=100, label=event) 
                  for event in unique_events]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"event_sequence.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Event sequence plot saved to {output_path}")
        
        return fig
    
    def create_summary_dashboard(self, df: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df (pd.DataFrame): Event data
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
        
        # 2. Hourly distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        timestamps = pd.to_datetime(df[self.timestamp_column])
        hour_counts = timestamps.dt.hour.value_counts().sort_index()
        hour_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Events by Hour')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Count')
        
        # 3. Daily trend (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        daily_counts = timestamps.dt.date.value_counts().sort_index().tail(30)
        daily_counts.plot(kind='line', ax=ax3, marker='o', color='green')
        ax3.set_title('Daily Event Trend (Last 30 Days)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Time gaps histogram (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        ax4.hist(time_gaps, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_title('Time Gap Distribution')
        ax4.set_xlabel('Time Gap (seconds)')
        ax4.set_ylabel('Frequency')
        
        # 5. Event type heatmap (middle)
        ax5 = fig.add_subplot(gs[1, 1])
        hour_event_cross = pd.crosstab(timestamps.dt.hour, df[self.event_column])
        sns.heatmap(hour_event_cross, ax=ax5, cmap='YlOrRd', cbar=True)
        ax5.set_title('Event Types by Hour')
        ax5.set_xlabel('Event Type')
        ax5.set_ylabel('Hour')
        
        # 6. Day of week distribution (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        day_counts = timestamps.dt.dayofweek.value_counts().sort_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts.index = [day_names[i] for i in day_counts.index]
        day_counts.plot(kind='bar', ax=ax6, color='purple')
        ax6.set_title('Events by Day of Week')
        ax6.set_xlabel('Day')
        ax6.set_ylabel('Count')
        
        # 7. Statistics summary (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate statistics
        stats_text = f"""
        Event Sequence Summary:
        ─────────────────────
        Total Events: {len(df):,}
        Unique Event Types: {df[self.event_column].nunique()}
        Time Range: {timestamps.min()} to {timestamps.max()}
        Average Time Gap: {time_gaps.mean():.2f} seconds
        Median Time Gap: {time_gaps.median():.2f} seconds
        Most Common Event: {df[self.event_column].mode().iloc[0]} ({df[self.event_column].value_counts().iloc[0]} occurrences)
        Events per Day: {len(df) / ((timestamps.max() - timestamps.min()).days + 1):.1f}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Event Sequence Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"event_dashboard.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Dashboard saved to {output_path}")
        
        return fig
