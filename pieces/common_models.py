"""Common data models shared across tree classification pieces"""
from pydantic import BaseModel, Field
from typing import Optional


class TreeSampleInfo(BaseModel):
    """Information about a single tree sample"""
    sample_id: str = Field(description="Unique identifier for the sample")
    bsq_path: str = Field(description="Path to BSQ hyperspectral file")
    class_label: str = Field(description="Tree height class: nizka_vegetacia, stredna_vegetacia, or vysoka_vegetacia")
    class_id: int = Field(description="Numeric class ID: 0=nizka, 1=stredna, 2=vysoka")
    project: Optional[str] = Field(default=None, description="Project ID (e.g., P101, P212)")


class DatasetInfo(BaseModel):
    """Information about the complete dataset"""
    dataset_path: str = Field(description="Root path to tree_bsq_chips dataset")
    total_samples: int = Field(description="Total number of samples")
    class_distribution: dict = Field(description="Count of samples per class")
    labels_file: str = Field(description="Path to labels CSV file")
