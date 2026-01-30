from pydantic import BaseModel, Field
from typing import List, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_models import TreeSampleInfo


class InputModel(BaseModel):
    """Tree Data Split Input Model"""
    samples: List[TreeSampleInfo] = Field(
        description="List of tree samples from TreeDataLoaderPiece"
    )
    n_clients: int = Field(
        default=3,
        description="Number of federated learning clients",
        ge=1,
        le=10
    )
    val_split: float = Field(
        default=0.2,
        description="Validation split ratio (0.0-0.5)",
        ge=0.0,
        le=0.5
    )
    random_seed: int = Field(
        default=123,
        description="Random seed for reproducible splits"
    )


class OutputModel(BaseModel):
    """Tree Data Split Output Model"""
    client_datalists: Dict[str, str] = Field(
        description="Mapping of client names to their train/val datalist paths"
    )
    split_summary: dict = Field(
        description="Summary of data distribution across clients"
    )
    output_dir: str = Field(
        description="Directory containing all datalist CSV files"
    )
    total_samples: int = Field(
        description="Total number of samples split"
    )
