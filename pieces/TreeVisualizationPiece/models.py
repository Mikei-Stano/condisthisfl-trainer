from pydantic import BaseModel, Field
from typing import Dict


class InputModel(BaseModel):
    """
    Tree Visualization Input Model
    """
    workspace_dir: str = Field(
        description="Training workspace directory containing TensorBoard logs"
    )


class OutputModel(BaseModel):
    """
    Tree Visualization Output Model
    """
    plot_paths: Dict[str, str] = Field(
        description="Dictionary mapping plot types to file paths. Example: {'training_loss': 'plots/loss.png', 'accuracy': 'plots/accuracy.png'}"
    )
    metrics_summary: Dict[str, float] = Field(
        description="Summary of final training metrics. Example: {'final_train_loss': 0.25, 'final_val_accuracy': 0.89, 'best_round': 45}"
    )
