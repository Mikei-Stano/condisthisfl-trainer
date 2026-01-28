from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TreeVisualizationPiece(BasePiece):
    """
    Tree Visualization Piece
    
    Visualizes federated learning training results from TensorBoard logs.
    Creates plots for:
    - Training/validation loss curves per client and global
    - Accuracy curves
    - Metrics summary
    """

    def piece_function(self, input_model: InputModel) -> OutputModel:
        """
        Parse TensorBoard logs and create visualization plots
        
        Args:
            input_model: Contains workspace_dir with TensorBoard logs
            
        Returns:
            OutputModel with plot_paths and metrics_summary
        """
        workspace_dir = Path(input_model.workspace_dir)
        
        # Create output directory for plots
        plots_dir = Path("/app/outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Visualizing training results from: {workspace_dir}")
        
        # Parse TensorBoard event files
        metrics = self._parse_tensorboard_logs(workspace_dir)
        
        # Create plots
        plot_paths = {}
        
        # Plot training loss
        if 'train_loss' in metrics:
            loss_plot = self._plot_training_curves(
                metrics['train_loss'], 
                title='Training Loss',
                ylabel='Loss',
                output_path=plots_dir / 'training_loss.png'
            )
            plot_paths['training_loss'] = str(loss_plot)
        
        # Plot validation accuracy
        if 'val_accuracy' in metrics:
            acc_plot = self._plot_training_curves(
                metrics['val_accuracy'],
                title='Validation Accuracy', 
                ylabel='Accuracy',
                output_path=plots_dir / 'validation_accuracy.png'
            )
            plot_paths['validation_accuracy'] = str(acc_plot)
        
        # Create metrics summary
        metrics_summary = self._create_metrics_summary(metrics)
        
        # Save metrics to JSON
        metrics_json_path = plots_dir / 'metrics_summary.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        self.logger.info(f"Created {len(plot_paths)} plots")
        self.logger.info(f"Final metrics: {metrics_summary}")
        
        return OutputModel(
            plot_paths=plot_paths,
            metrics_summary=metrics_summary
        )
    
    def _parse_tensorboard_logs(self, workspace_dir: Path) -> dict:
        """
        Parse TensorBoard event files from workspace
        
        Returns dictionary with metrics: {'train_loss': {'site-1': [...], 'global': [...]}, ...}
        """
        metrics = {
            'train_loss': {},
            'val_accuracy': {},
            'val_loss': {}
        }
        
        # Look for TensorBoard logs in workspace/*/local/logs or workspace/server
        # For now, return mock data structure
        # TODO: Implement actual TensorBoard event file parsing using tensorboard or tensorflow
        
        self.logger.warning("TensorBoard parsing not yet implemented, returning placeholder metrics")
        
        # Placeholder data
        metrics['train_loss']['global'] = list(range(10))
        metrics['val_accuracy']['global'] = list(range(10))
        
        return metrics
    
    def _plot_training_curves(self, data: dict, title: str, ylabel: str, output_path: Path) -> Path:
        """
        Create training curve plot
        
        Args:
            data: Dictionary mapping client names to metric values over rounds
            title: Plot title
            ylabel: Y-axis label
            output_path: Where to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for client_name, values in data.items():
            plt.plot(values, label=client_name, marker='o')
        
        plt.xlabel('Round')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def _create_metrics_summary(self, metrics: dict) -> dict:
        """
        Create summary of final metrics
        """
        summary = {}
        
        # Get final values from each metric type
        for metric_name, clients_data in metrics.items():
            if 'global' in clients_data and len(clients_data['global']) > 0:
                summary[f'final_{metric_name}'] = float(clients_data['global'][-1])
                summary[f'best_{metric_name}'] = float(min(clients_data['global']) if 'loss' in metric_name else max(clients_data['global']))
        
        return summary
