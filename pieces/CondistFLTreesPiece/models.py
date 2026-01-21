from pydantic import BaseModel, Field
from typing import Optional, Dict


class InputModel(BaseModel):
    """
    CondistFL Tree Classification Training Input Model
    """
    num_rounds: int = Field(
        default=50,
        description="Number of federated learning rounds",
        json_schema_extra={"from_upstream": "never"}
    )
    steps_per_round: int = Field(
        default=100,
        description="Training steps per round",
        json_schema_extra={"from_upstream": "never"}
    )
    clients: str = Field(
        default="site-1,site-2,site-3",
        description="Comma-separated list of client names (site-1, site-2, site-3)",
        json_schema_extra={"from_upstream": "never"}
    )
    gpus: str = Field(
        default="0,1,2",
        description="Comma-separated GPU IDs to use (one per client)",
        json_schema_extra={"from_upstream": "never"}
    )
    workspace_dir: str = Field(
        default="/app/workspace",
        description="Directory to save training workspace and results",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Tree Classification Training Output Model
    """
    workspace_dir: Optional[str] = Field(
        default=None,
        description="Directory containing training results and models"
    )
    best_global_model_path: Optional[str] = Field(
        default=None,
        description="Path to the best global model checkpoint"
    )
    global_model_path: Optional[str] = Field(
        default=None,
        description="Path to the final global model checkpoint"
    )
    best_local_models: Optional[Dict[str, str]] = Field(
        description="Paths to best local models for each client site",
        default_factory=dict
    )
    training_complete: Optional[bool] = Field(
        default=False,
        description="Whether training completed successfully"
    )
    num_rounds_completed: Optional[int] = Field(
        default=0,
        description="Number of federated learning rounds completed"
    )
    validation_metrics: Optional[Dict[str, float]] = Field(
        description="Summary of validation metrics (accuracy, loss, etc.) per client",
        default_factory=dict
    )
    message: Optional[str] = Field(
        default="",
        description="Status message about training completion"
    )
