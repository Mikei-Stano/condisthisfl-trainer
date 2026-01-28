from pydantic import BaseModel, Field
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_models import TreeSampleInfo, DatasetInfo


class InputModel(BaseModel):
    """Tree Data Loader Input Model"""
    dataset_path: str = Field(
        default="/app/tree_bsq_chips_sampled",
        description="Path to tree_bsq_chips dataset directory"
    )
    labels_file: str = Field(
        default="labels_all_projects.csv",
        description="Name of labels CSV file (relative to dataset_path or absolute path)"
    )


class OutputModel(BaseModel):
    """Tree Data Loader Output Model"""
    samples: List[TreeSampleInfo] = Field(
        description="List of discovered tree samples with BSQ paths and labels"
    )
    dataset_info: DatasetInfo = Field(
        description="Summary information about the dataset"
    )
    total_samples: int = Field(
        description="Total number of samples found"
    )
    class_distribution: dict = Field(
        description="Count of samples per class"
    )
