from pydantic import BaseModel, Field
from typing import List, Dict


class InputModel(BaseModel):
    """
    Tree Inference Input Model
    """
    model_path: str = Field(
        description="Path to trained PyTorch model (.pt file)"
    )
    input_bsq_paths: List[str] = Field(
        description="List of BSQ file paths to run inference on"
    )
    batch_size: int = Field(
        default=32,
        description="Inference batch size"
    )


class OutputModel(BaseModel):
    """
    Tree Inference Output Model
    """
    predictions: List[Dict] = Field(
        description="List of predictions for each input file. Each dict contains: {'file': str, 'predicted_class': str, 'class_id': int, 'confidence': float}"
    )
    predictions_csv: str = Field(
        description="Path to CSV file with all predictions"
    )
