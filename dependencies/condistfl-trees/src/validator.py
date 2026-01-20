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

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator(object):
    """
    Validator for tree classification.
    Computes accuracy and per-class metrics on validation set.
    """
    def __init__(self, task_config: Dict):
        self.classes = task_config["classes"]
        self.num_classes = len(self.classes)

    @torch.no_grad()
    def run(self, model: nn.Module, data_loader: DataLoader, device: str = "cuda:0") -> Dict:
        """
        Run validation.
        
        Args:
            model: Model to validate
            data_loader: Validation data loader
            device: Device to run on
        
        Returns:
            Dictionary of metrics
        """
        model = model.to(device)
        model.eval()

        total = 0
        correct = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(data_loader, desc="Validation", dynamic_ncols=True):
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            logits = model(image)
            pred = logits.argmax(dim=1)

            # Accumulate statistics
            total += label.size(0)
            correct += (pred == label).sum().item()
            
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

        # Compute overall accuracy
        accuracy = correct / max(1, total)

        # Compute per-class metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.classes):
            class_mask = (all_labels == i)
            class_correct = ((all_preds == i) & class_mask).sum().item()
            class_total = class_mask.sum().item()
            class_acc = class_correct / max(1, class_total)
            per_class_metrics[f"val_acc_{class_name}"] = class_acc

        metrics = {
            "val_acc": accuracy,
            **per_class_metrics
        }

        return metrics
