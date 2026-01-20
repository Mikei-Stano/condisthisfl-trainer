# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import math
import numpy as np

from .dataset import BSQTreeDataset, find_bsq_with_meta, LABEL_MAP
from torch.utils.data import DataLoader


def create_dataset(app_root: str, config: Dict, split: str, mode: str, local_class_id: int = None):
    """
    Create a dataset for tree classification.
    
    Args:
        app_root: Application root directory
        config: Configuration dictionary with data settings
        split: "training" or "validation"
        mode: "train" or "validate"
        local_class_id: The class ID that this client specializes in
    
    Returns:
        BSQTreeDataset
    """
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = Path(app_root) / data_root
    
    # Support two modes:
    # 1. Single data_list with split column (old ConDist approach)
    # 2. Separate train_datalist and val_datalist (per-client approach)
    if "train_datalist" in config and "val_datalist" in config:
        # Per-client approach
        datalist_key = "train_datalist" if split == "training" else "val_datalist"
        datalist_file = Path(config[datalist_key])
    else:
        # ConDist approach with single datalist
        datalist_file = Path(config["data_list"])
    
    if not datalist_file.is_absolute():
        datalist_file = Path(app_root) / datalist_file
    
    # Load the datalist
    if not datalist_file.exists():
        raise FileNotFoundError(
            f"Datalist file {datalist_file} not found. "
            "Please run prepare_data.py or prepare_data_per_client.py first."
        )
    
    df = pd.read_csv(datalist_file)
    
    # Filter by split column if present (ConDist approach)
    if "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    
    print(f"Loaded {len(df)} samples for {split} split")
    
    # Get settings
    band_norm = config.get("band_norm", None)
    force_k = config.get("force_k", None)
    target_hw = config.get("target_hw", None)
    if target_hw is not None:
        target_hw = tuple(target_hw)
    
    # Create dataset
    augment = (mode == "train")
    dataset = BSQTreeDataset(
        df, 
        band_norm=band_norm,
        augment=augment,
        force_k=force_k,
        target_hw=target_hw,
        local_class_id=local_class_id,
    )
    
    return dataset


def create_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False):
    """Create a data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class DataManager(object):
    """
    Data manager for tree classification.
    Handles dataset creation and data loaders.
    """
    def __init__(self, app_root: str, config: Dict, local_class_id: int = None):
        self.app_root = app_root
        self.config = config
        self.local_class_id = local_class_id

        self._dataset = {}
        self._data_loader = {}

    def _build_dataset(self, stage: str):
        if stage == "train":
            mode = "train"
            split = "training"
        elif stage == "validate":
            mode = "validate"
            split = "validation"
        elif stage == "test":
            mode = "validate"
            split = "testing"
        else:
            raise ValueError(f"Unknown stage {stage} for dataset")
        return create_dataset(self.app_root, self.config, split, mode, self.local_class_id)

    def _build_data_loader(self, stage: str):
        ds = self._dataset.get(stage)
        if stage == "train":
            dl = create_data_loader(
                ds,
                batch_size=self.config["data_loader"].get("batch_size", 1),
                num_workers=self.config["data_loader"].get("num_workers", 0),
                shuffle=True,
            )
        else:
            dl = create_data_loader(
                ds, 
                batch_size=self.config["data_loader"].get("batch_size", 1),
                num_workers=self.config["data_loader"].get("num_workers", 0)
            )
        return dl

    def setup(self, stage: Optional[str] = None):
        if stage is None:
            for s in ["train", "validate", "test"]:
                self._dataset[s] = self._build_dataset(s)
                self._data_loader[s] = self._build_data_loader(s)
        elif stage in ["train", "validate", "test"]:
            self._dataset[stage] = self._build_dataset(stage)
            self._data_loader[stage] = self._build_data_loader(stage)

    def get_dataset(self, stage: str):
        return self._dataset.get(stage, None)

    def get_data_loader(self, stage: str):
        return self._data_loader.get(stage, None)

    def teardown(self):
        self._dataset = {}
        self._data_loader = {}
