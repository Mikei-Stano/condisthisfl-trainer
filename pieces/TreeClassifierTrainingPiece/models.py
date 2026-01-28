from pydantic import BaseModel, Field
from typing import Optional, Dict


class InputModel(BaseModel):
    """
    Tree Classifier Training Input Model
    """
    client_datalists: Dict[str, Dict[str, str]] = Field(
        description="Dictionary mapping client names to their train/val datalist CSV paths. Example: {'site-1': {'train': 'site-1_train_datalist.csv', 'val': 'site-1_val_datalist.csv'}}"
    )
    num_rounds: int = Field(
        default=50,
        description="Number of federated learning rounds"
    )
    steps_per_round: int = Field(
        default=100,
        description="Training steps per round"
    )
    workspace_dir: str = Field(
        default="/app/workspace",
        description="Directory to save training workspace and results"
    )


class OutputModel(BaseModel):
    """
    Tree Classifier Training Output Model
    """
    workspace_dir: str = Field(
        description="Directory containing training results and models"
    )
    best_global_model_path: str = Field(
        description="Path to the best global model checkpoint"
    )
    global_model_path: str = Field(
        description="Path to the final global model checkpoint"
    )
    best_local_models: Dict[str, str] = Field(
        description="Paths to best local models for each client site",
        default_factory=dict
    )
    training_complete: bool = Field(
        description="Whether training completed successfully"
    )
    num_rounds_completed: int = Field(
        description="Number of federated learning rounds completed"
    )
    validation_metrics: Dict[str, float] = Field(
        description="Summary of validation metrics (accuracy, loss, etc.) per client",
        default_factory=dict
    )
    message: str = Field(
        description="Status message about training completion"
    )
