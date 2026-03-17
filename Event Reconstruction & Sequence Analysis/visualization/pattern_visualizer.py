"""
Pattern Visualizer Module

Visualizes detected patterns, sequences, and pattern analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class PatternVisualizer:
    """
    Visualizes patterns and pattern analysis results.
    
    This class provides visualization methods for frequent patterns,
    sequential patterns, motifs, and other pattern analysis outputs.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PatternVisualizer with configuration.
        
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
    
    def plot_frequent_patterns(self, patterns: List[Dict],
                             top_k: int = 10,
                             title: str = "Frequent Patterns",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot frequent patterns with their support values.
        
        Args:
            patterns (List[Dict]): List of frequent patterns
            top_k (int): Number of top patterns to display
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Filter and sort patterns
        frequent_patterns = [p for p in patterns if p.get('type') == 'frequent']
        frequent_patterns.sort(key=lambda x: x.get('support', 0), reverse=True)
        top_patterns = frequent_patterns[:top_k]
        
        if not top_patterns:
            ax1.text(0.5, 0.5, 'No frequent patterns found', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No frequent patterns found', ha='center', va='center', transform=ax2.transAxes)
            plt.suptitle(title)
            return fig
        
        # Prepare data for plotting
        pattern_names = [' -> '.join(p['pattern']) for p in top_patterns]
        supports = [p['support'] for p in top_patterns]
        counts = [p['count'] for p in top_patterns]
        
        # Bar plot of support values
        bars = ax1.bar(range(len(pattern_names)), supports, color='skyblue', edgecolor='navy')
        ax1.set_title('Pattern Support Values')
        ax1.set_xlabel('Patterns')
        ax1.set_ylabel('Support')
        ax1.set_xticks(range(len(pattern_names)))
        ax1.set_xticklabels(pattern_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, support in zip(bars, supports):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{support:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Scatter plot of pattern length vs support
        pattern_lengths = [len(p['pattern']) for p in top_patterns]
        scatter = ax2.scatter(pattern_lengths, supports, c=counts, s=100, alpha=0.7, cmap='viridis')
        ax2.set_title('Pattern Length vs Support')
        ax2.set_xlabel('Pattern Length')
        ax2.set_ylabel('Support')
        
        # Add colorbar for counts
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Pattern Count')
        
        # Add pattern labels to scatter plot
        for i, (x, y, pattern) in enumerate(zip(pattern_lengths, supports, pattern_names)):
            if i < 5:  # Label only first 5 to avoid clutter
                ax2.annotate(pattern[:20] + '...' if len(pattern) > 20 else pattern,
                           (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"frequent_patterns.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Frequent patterns plot saved to {output_path}")
        
        return fig
    
    def plot_pattern_network(self, patterns: List[Dict],
                           title: str = "Pattern Network",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize patterns as a network graph.
        
        Args:
            patterns (List[Dict]): List of patterns
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges from patterns
        for pattern in patterns:
            if 'pattern' in pattern:
                events = pattern['pattern']
                support = pattern.get('support', 1)
                
                # Add nodes
                for event in events:
                    if event not in G.nodes:
                        G.add_node(event, weight=0)
                    G.nodes[event]['weight'] += support
                
                # Add edges
                for i in range(len(events) - 1):
                    G.add_edge(events[i], events[i+1], weight=support)
        
        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, 'No patterns to visualize', ha='center', va='center', transform=ax.transAxes)
            plt.title(title)
            return fig
        
        # Set up layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_sizes = [G.nodes[node]['weight'] * 1000 for node in G.nodes()]
        node_colors = [G.nodes[node]['weight'] for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                      node_color=node_colors, cmap='viridis',
                                      alpha=0.7)
        
        # Draw edges
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w*3 for w in edge_weights],
                              alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=min(node_colors), 
                                                   vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pattern Support')
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"pattern_network.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Pattern network plot saved to {output_path}")
        
        return fig
    
    def plot_motif_analysis(self, motifs: List[Dict],
                          title: str = "Motif Analysis",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot motif analysis results.
        
        Args:
            motifs (List[Dict]): List of detected motifs
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        if not motifs:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No motifs found', ha='center', va='center', transform=ax.transAxes)
            plt.title(title)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Prepare data
        motif_names = [' -> '.join(m['pattern']) for m in motifs]
        motif_counts = [m['count'] for m in motifs]
        avg_spacings = [m.get('avg_spacing', 0) for m in motifs]
        spacing_stds = [m.get('spacing_std', 0) for m in motifs]
        
        # 1. Motif counts
        bars = ax1.bar(range(len(motif_names)), motif_counts, color='lightcoral', edgecolor='darkred')
        ax1.set_title('Motif Counts')
        ax1.set_xlabel('Motifs')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(motif_names)))
        ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in motif_names], 
                           rotation=45, ha='right')
        
        # 2. Average spacing
        ax2.bar(range(len(motif_names)), avg_spacings, color='lightblue', edgecolor='darkblue')
        ax2.set_title('Average Spacing Between Motif Occurrences')
        ax2.set_xlabel('Motifs')
        ax2.set_ylabel('Average Spacing')
        ax2.set_xticks(range(len(motif_names)))
        ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in motif_names], 
                           rotation=45, ha='right')
        
        # 3. Spacing variability
        ax3.bar(range(len(motif_names)), spacing_stds, color='lightgreen', edgecolor='darkgreen')
        ax3.set_title('Spacing Variability (Std Dev)')
        ax3.set_xlabel('Motifs')
        ax3.set_ylabel('Spacing Std Dev')
        ax3.set_xticks(range(len(motif_names)))
        ax3.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in motif_names], 
                           rotation=45, ha='right')
        
        # 4. Motif positions timeline
        for i, motif in enumerate(motifs[:5]):  # Show first 5 motifs
            positions = motif.get('positions', [])
            if positions:
                ax4.scatter([i] * len(positions), positions, alpha=0.7, s=50, label=f'Motif {i+1}')
        
        ax4.set_title('Motif Positions in Sequence')
        ax4.set_xlabel('Motif Index')
        ax4.set_ylabel('Position in Sequence')
        if len(motifs) <= 5:
            ax4.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"motif_analysis.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Motif analysis plot saved to {output_path}")
        
        return fig
    
    def plot_conditional_patterns(self, conditional_patterns: List[Dict],
                                title: str = "Conditional Patterns",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot conditional (if-then) patterns.
        
        Args:
            conditional_patterns (List[Dict]): List of conditional patterns
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        if not conditional_patterns:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No conditional patterns found', ha='center', va='center', transform=ax.transAxes)
            plt.title(title)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Prepare data
        pattern_names = [f"{p['condition']} → {p['consequence']}" for p in conditional_patterns]
        probabilities = [p['conditional_probability'] for p in conditional_patterns]
        counts = [p['count'] for p in conditional_patterns]
        
        # Sort by probability
        sorted_data = sorted(zip(pattern_names, probabilities, counts), key=lambda x: x[1], reverse=True)
        pattern_names, probabilities, counts = zip(*sorted_data)
        
        # 1. Conditional probabilities
        bars = ax1.bar(range(len(pattern_names)), probabilities, color='gold', edgecolor='orange')
        ax1.set_title('Conditional Probabilities')
        ax1.set_xlabel('Patterns')
        ax1.set_ylabel('Probability')
        ax1.set_xticks(range(len(pattern_names)))
        ax1.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in pattern_names], 
                           rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Add probability labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Pattern counts
        ax2.bar(range(len(pattern_names)), counts, color='lightcyan', edgecolor='teal')
        ax2.set_title('Pattern Occurrence Counts')
        ax2.set_xlabel('Patterns')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(pattern_names)))
        ax2.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in pattern_names], 
                           rotation=45, ha='right')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"conditional_patterns.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Conditional patterns plot saved to {output_path}")
        
        return fig
    
    def plot_pattern_comparison(self, patterns1: List[Dict], patterns2: List[Dict],
                              set1_name: str = "Set 1", set2_name: str = "Set 2",
                              title: str = "Pattern Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare two sets of patterns.
        
        Args:
            patterns1 (List[Dict]): First set of patterns
            patterns2 (List[Dict]): Second set of patterns
            set1_name (str): Name for first set
            set2_name (str): Name for second set
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Extract pattern statistics
        def extract_stats(patterns):
            frequent = [p for p in patterns if p.get('type') == 'frequent']
            sequential = [p for p in patterns if p.get('type') == 'sequential_with_gaps']
            motifs = [p for p in patterns if p.get('type') == 'motif']
            
            return {
                'total': len(patterns),
                'frequent': len(frequent),
                'sequential': len(sequential),
                'motifs': len(motifs),
                'avg_support': np.mean([p.get('support', 0) for p in frequent]) if frequent else 0,
                'avg_length': np.mean([len(p.get('pattern', [])) for p in patterns if 'pattern' in p]) if patterns else 0
            }
        
        stats1 = extract_stats(patterns1)
        stats2 = extract_stats(patterns2)
        
        # 1. Pattern type distribution comparison
        categories = ['Total', 'Frequent', 'Sequential', 'Motifs']
        values1 = [stats1['total'], stats1['frequent'], stats1['sequential'], stats1['motifs']]
        values2 = [stats2['total'], stats2['frequent'], stats2['sequential'], stats2['motifs']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, values1, width, label=set1_name, color='skyblue', alpha=0.7)
        ax1.bar(x + width/2, values2, width, label=set2_name, color='lightcoral', alpha=0.7)
        ax1.set_title('Pattern Type Distribution')
        ax1.set_xlabel('Pattern Type')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        
        # 2. Average support comparison
        ax2.bar([set1_name, set2_name], [stats1['avg_support'], stats2['avg_support']], 
               color=['skyblue', 'lightcoral'], alpha=0.7)
        ax2.set_title('Average Pattern Support')
        ax2.set_ylabel('Average Support')
        
        # 3. Average pattern length comparison
        ax3.bar([set1_name, set2_name], [stats1['avg_length'], stats2['avg_length']], 
               color=['skyblue', 'lightcoral'], alpha=0.7)
        ax3.set_title('Average Pattern Length')
        ax3.set_ylabel('Average Length')
        
        # 4. Venn diagram of overlapping patterns (simplified)
        set1_patterns = set(str(p.get('pattern', '')) for p in patterns1 if 'pattern' in p)
        set2_patterns = set(str(p.get('pattern', '')) for p in patterns2 if 'pattern' in p)
        
        overlap = len(set1_patterns & set2_patterns)
        only1 = len(set1_patterns - set2_patterns)
        only2 = len(set2_patterns - set1_patterns)
        
        # Simple bar representation of Venn diagram
        ax4.bar(['Only Set 1', 'Overlap', 'Only Set 2'], [only1, overlap, only2], 
               color=['skyblue', 'purple', 'lightcoral'], alpha=0.7)
        ax4.set_title('Pattern Overlap')
        ax4.set_ylabel('Number of Patterns')
        
        # Add overlap percentage
        total_unique = len(set1_patterns | set2_patterns)
        overlap_pct = (overlap / total_unique * 100) if total_unique > 0 else 0
        ax4.text(1, overlap + max(only1, overlap, only2) * 0.05, f'{overlap_pct:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots or save_path:
            output_path = save_path or f"pattern_comparison.{self.output_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Pattern comparison plot saved to {output_path}")
        
        return fig
