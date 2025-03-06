# Visualization utilities for Yanomami translation model training
#
# This module provides visualization tools for tracking and analyzing model performance
# during training, including loss curves, learning rate schedules, and comparison with baselines

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Class for creating visualizations of model training progress and performance
    """
    
    def __init__(self, output_dir="./visualization_results"):
        """
        Initialize the visualizer
        
        Args:
            output_dir (str): Directory to save visualization results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.loss_history = []
        self.lr_history = []
        self.precision_history = []
        self.baseline_precision = {}
        
    def record_training_metrics(self, phase, epoch, batch=None, loss=None, learning_rate=None, precision=None):
        """
        Record training metrics for later visualization
        
        Args:
            phase (int): Current training phase
            epoch (int): Current epoch
            batch (int, optional): Current batch number
            loss (float, optional): Current loss value
            learning_rate (float, optional): Current learning rate
            precision (float, optional): Current precision metric
        """
        timestamp = datetime.now().isoformat()
        
        # Record loss and learning rate
        if loss is not None or learning_rate is not None:
            entry = {
                "timestamp": timestamp,
                "phase": phase,
                "epoch": epoch,
                "batch": batch
            }
            
            if loss is not None:
                entry["loss"] = loss
                
            if learning_rate is not None:
                entry["learning_rate"] = learning_rate
                
            self.loss_history.append(entry)
            
        # Record precision
        if precision is not None:
            self.precision_history.append({
                "timestamp": timestamp,
                "phase": phase,
                "epoch": epoch,
                "batch": batch,
                "precision": precision
            })
    
    def set_baseline_precision(self, model_name, precision):
        """
        Set baseline precision values for comparison
        
        Args:
            model_name (str): Name of the baseline model
            precision (float): Precision value for the baseline model
        """
        self.baseline_precision[model_name] = precision
    
    def plot_loss_and_lr(self, title_suffix=""):
        """
        Generate a plot showing loss and learning rate over time
        
        Args:
            title_suffix (str): Optional suffix for the plot title
            
        Returns:
            str: Path to the saved chart file
        """
        if not self.loss_history:
            logger.warning("No loss history data available for plotting")
            return None
        
        # Extract data for plotting
        phases = [entry.get("phase", 0) for entry in self.loss_history]
        epochs = [entry.get("epoch", 0) for entry in self.loss_history]
        batches = [entry.get("batch", 0) for entry in self.loss_history]
        losses = [entry.get("loss", 0) for entry in self.loss_history if "loss" in entry]
        learning_rates = [entry.get("learning_rate", 0) for entry in self.loss_history if "learning_rate" in entry]
        
        # Create x-axis labels based on available data
        has_batch_info = any(batch is not None for batch in batches)
        
        if has_batch_info:
            x_labels = [f"P{p}E{e}B{b}" for p, e, b in zip(phases, epochs, batches)]
        else:
            x_labels = [f"P{p}E{e}" for p, e in zip(phases, epochs)]
            
        # Use indices for x-axis positioning
        x_positions = list(range(len(x_labels)))
        
        # Create figure with two subplots (loss and learning rate)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot loss history
        if losses:
            ax1.plot(x_positions, losses, 'o-', color='red', linewidth=2, markersize=5)
            ax1.set_title('Training Loss Over Time', fontsize=14)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add moving average trendline if we have enough data points
            if len(losses) > 5:
                window_size = min(5, len(losses) // 3)
                moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                # Plot moving average with offset to align with original data
                offset = (window_size - 1) // 2
                ax1.plot(x_positions[offset:offset+len(moving_avg)], moving_avg, 
                         '-', color='darkred', linewidth=2, label=f'{window_size}-point Moving Avg')
                ax1.legend()
        
        # Plot learning rate history
        if learning_rates:
            ax2.plot(x_positions, learning_rates, 'o-', color='blue', linewidth=2, markersize=5)
            ax2.set_title('Learning Rate Schedule', fontsize=14)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Use scientific notation for small learning rates
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        # Set x-axis labels
        # Only show a subset of labels if there are too many
        if len(x_labels) > 20:
            step = len(x_labels) // 20 + 1
            visible_positions = x_positions[::step]
            visible_labels = x_labels[::step]
        else:
            visible_positions = x_positions
            visible_labels = x_labels
            
        ax2.set_xticks(visible_positions)
        ax2.set_xticklabels(visible_labels, rotation=45, ha='right')
        ax2.set_xlabel('Training Progress (Phase, Epoch, Batch)', fontsize=12)
        
        # Add overall title
        title = f'Yanomami Translation Model Training Progress{title_suffix}'
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)  # Make room for the suptitle
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(self.output_dir, f"loss_lr_history_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss and learning rate chart saved to {chart_file}")
        return chart_file
    
    def plot_precision_comparison(self, title_suffix=""):
        """
        Generate a plot showing precision improvement compared to baselines
        
        Args:
            title_suffix (str): Optional suffix for the plot title
            
        Returns:
            str: Path to the saved chart file
        """
        if not self.precision_history:
            logger.warning("No precision history data available for plotting")
            return None
        
        # Extract data for plotting
        phases = [entry.get("phase", 0) for entry in self.precision_history]
        epochs = [entry.get("epoch", 0) for entry in self.precision_history]
        precision_values = [entry.get("precision", 0) for entry in self.precision_history]
        
        # Create x-axis labels
        x_labels = [f"P{p}E{e}" for p, e in zip(phases, epochs)]
        x_positions = list(range(len(x_labels)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot precision history
        ax.plot(x_positions, precision_values, 'o-', color='green', linewidth=2, markersize=8, label='Yanomami Model')
        
        # Add baseline model precision values as horizontal lines
        colors = ['blue', 'red', 'purple', 'orange', 'brown']
        for i, (model_name, precision) in enumerate(self.baseline_precision.items()):
            color = colors[i % len(colors)]
            ax.axhline(y=precision, color=color, linestyle='--', linewidth=2, label=f'{model_name} Baseline')
        
        # Add labels and title
        ax.set_title('Precision Comparison with Baseline Models', fontsize=14)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlabel('Training Progress (Phase, Epoch)', fontsize=12)
        
        # Set x-axis labels
        # Only show a subset of labels if there are too many
        if len(x_labels) > 15:
            step = len(x_labels) // 15 + 1
            visible_positions = x_positions[::step]
            visible_labels = x_labels[::step]
        else:
            visible_positions = x_positions
            visible_labels = x_labels
            
        ax.set_xticks(visible_positions)
        ax.set_xticklabels(visible_labels, rotation=45, ha='right')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Add annotations for final value
        if precision_values:
            final_precision = precision_values[-1]
            ax.annotate(f'Final: {final_precision:.4f}', 
                        xy=(x_positions[-1], final_precision),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Add overall title
        title = f'Yanomami Translation Model Precision Comparison{title_suffix}'
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)  # Make room for the suptitle
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(self.output_dir, f"precision_comparison_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision comparison chart saved to {chart_file}")
        return chart_file
    
    def save_history_data(self):
        """
        Save all recorded history data to JSON files
        
        Returns:
            tuple: Paths to the saved JSON files (loss_history_file, precision_history_file)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save loss and learning rate history
        loss_history_file = None
        if self.loss_history:
            loss_history_file = os.path.join(self.output_dir, f"loss_lr_history_{timestamp}.json")
            with open(loss_history_file, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
            logger.info(f"Loss and learning rate history saved to {loss_history_file}")
        
        # Save precision history
        precision_history_file = None
        if self.precision_history:
            precision_data = {
                "yanomami_model": self.precision_history,
                "baseline_models": self.baseline_precision
            }
            precision_history_file = os.path.join(self.output_dir, f"precision_history_{timestamp}.json")
            with open(precision_history_file, 'w') as f:
                json.dump(precision_data, f, indent=2)
            logger.info(f"Precision history saved to {precision_history_file}")
        
        return loss_history_file, precision_history_file
    
    def generate_training_report(self, final_metrics=None, translation_results=None):
        """
        Generate a comprehensive HTML report with all training metrics and visualizations
        
        Args:
            final_metrics (dict, optional): Final evaluation metrics to include in the report
            translation_results (dict, optional): Translation test results to include in the report
                
        Returns:
            str: Path to the saved HTML report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"training_report_{timestamp}.html")
        
        # Generate charts if they don't exist yet
        loss_chart = self.plot_loss_and_lr(title_suffix=" (Report)")
        precision_chart = self.plot_precision_comparison(title_suffix=" (Report)")
        
        # Calculate statistics from loss history
        loss_stats = {}
        if self.loss_history:
            losses = [entry.get("loss") for entry in self.loss_history if "loss" in entry]
            if losses:
                loss_stats = {
                    "initial": losses[0],
                    "final": losses[-1],
                    "min": min(losses),
                    "max": max(losses),
                    "avg": sum(losses) / len(losses),
                    "improvement": losses[0] - losses[-1] if len(losses) > 1 else 0,
                    "improvement_percent": ((losses[0] - losses[-1]) / losses[0] * 100) if len(losses) > 1 and losses[0] > 0 else 0
                }
        
        # Calculate statistics from precision history
        precision_stats = {}
        if self.precision_history:
            precisions = [entry.get("precision") for entry in self.precision_history if "precision" in entry]
            if precisions:
                precision_stats = {
                    "initial": precisions[0],
                    "final": precisions[-1],
                    "max": max(precisions),
                    "improvement": precisions[-1] - precisions[0] if len(precisions) > 1 else 0
                }
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Yanomami Translation Model Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
                .metric-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; }}
                .chart-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Yanomami Translation Model Training Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="metric-box">
                    <h2>Training Summary</h2>
                    <div class="metric-grid">
        """
        
        # Add loss statistics
        if loss_stats:
            html_content += f"""
                        <div class="metric-item">
                            <h3>Initial Loss</h3>
                            <p>{loss_stats['initial']:.4f}</p>
                        </div>
                        <div class="metric-item">
                            <h3>Final Loss</h3>
                            <p>{loss_stats['final']:.4f}</p>
                        </div>
                        <div class="metric-item">
                            <h3>Loss Improvement</h3>
                            <p>{loss_stats['improvement']:.4f} ({loss_stats['improvement_percent']:.2f}%)</p>
                        </div>
            """
        
        # Add precision statistics
        if precision_stats:
            html_content += f"""
                        <div class="metric-item">
                            <h3>Initial Precision</h3>
                            <p>{precision_stats['initial']:.4f}</p>
                        </div>
                        <div class="metric-item">
                            <h3>Final Precision</h3>
                            <p>{precision_stats['final']:.4f}</p>
                        </div>
                        <div class="metric-item">
                            <h3>Precision Improvement</h3>
                            <p>{precision_stats['improvement']:.4f}</p>
                        </div>
            """
        
        # Add baseline comparison
        if self.baseline_precision:
            for model_name, baseline in self.baseline_precision.items():
                if precision_stats:
                    improvement = precision_stats['final'] - baseline
                    improvement_percent = (improvement / baseline * 100) if baseline > 0 else 0
                    html_content += f"""
                        <div class="metric-item">
                            <h3>vs {model_name}</h3>
                            <p>{improvement:.4f} ({improvement_percent:.2f}%)</p>
                        </div>
                    """
        
        # Close the metric grid
        html_content += """
                    </div>
                </div>
        """
        
        # Add charts
        if loss_chart and os.path.exists(loss_chart):
            loss_chart_rel_path = os.path.relpath(loss_chart, self.output_dir)
            html_content += f"""
                <div class="chart-container">
                    <h2>Loss and Learning Rate</h2>
                    <img src="{loss_chart_rel_path}" alt="Loss and Learning Rate Chart" style="max-width: 100%;">
                </div>
            """
        
        if precision_chart and os.path.exists(precision_chart):
            precision_chart_rel_path = os.path.relpath(precision_chart, self.output_dir)
            html_content += f"""
                <div class="chart-container">
                    <h2>Precision Comparison</h2>
                    <img src="{precision_chart_rel_path}" alt="Precision Comparison Chart" style="max-width: 100%;">
                </div>
            """
        
        # Add final metrics if provided
        if final_metrics:
            html_content += """
                <div class="metric-box">
                    <h2>Final Evaluation Metrics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
            """
            
            for metric_name, metric_value in final_metrics.items():
                html_content += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{metric_value}</td>
                        </tr>
                """
                
            html_content += """
                    </table>
                </div>
            """
        
        # Add data tables
        if self.loss_history:
            html_content += """
                <div class="metric-box">
                    <h2>Training History</h2>
                    <table>
                        <tr>
                            <th>Phase</th>
                            <th>Epoch</th>
                            <th>Batch</th>
                            <th>Loss</th>
                            <th>Learning Rate</th>
                        </tr>
            """
            
            # Limit to last 50 entries if there are too many
            display_history = self.loss_history[-50:] if len(self.loss_history) > 50 else self.loss_history
            
            for entry in display_history:
                phase = entry.get("phase", "-")
                epoch = entry.get("epoch", "-")
                batch = entry.get("batch", "-")
                loss = f"{entry.get('loss', '-'):.6f}" if "loss" in entry else "-"
                lr = f"{entry.get('learning_rate', '-'):.8f}" if "learning_rate" in entry else "-"
                
                html_content += f"""
                        <tr>
                            <td>{phase}</td>
                            <td>{epoch}</td>
                            <td>{batch}</td>
                            <td>{loss}</td>
                            <td>{lr}</td>
                        </tr>
                """
                
            html_content += """
                    </table>
                </div>
            """
        
        # Add translation results if provided
        if translation_results and 'translations' in translation_results:
            html_content += """
                <div class="metric-box">
                    <h2>Translation Examples</h2>
                    <table>
                        <tr>
                            <th>Input</th>
                            <th>Direction</th>
                            <th>Translation</th>
                        </tr>
            """
            
            for translation in translation_results['translations']:
                input_text = translation.get('input', '-')
                direction = translation.get('direction', '-')
                output_text = translation.get('output', '-')
                
                # Format the direction for display
                if direction == 'english_to_yanomami':
                    direction_display = 'English → Yanomami'
                elif direction == 'yanomami_to_english':
                    direction_display = 'Yanomami → English'
                else:
                    direction_display = direction
                
                html_content += f"""
                        <tr>
                            <td>{input_text}</td>
                            <td>{direction_display}</td>
                            <td>{output_text}</td>
                        </tr>
                """
                
            html_content += """
                    </table>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Training report generated at {report_file}")
        return report_file
