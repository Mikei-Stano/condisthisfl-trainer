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

import torch
from torch import nn


class SmallConvNet(nn.Module):
    """
    Small Convolutional Neural Network for hyperspectral tree classification.
    
    This is adapted from simple_classification.ipynb.
    Input: (batch, in_channels, H, W) where in_channels is the number of hyperspectral bands
    Output: (batch, n_classes) logits
    """
    def __init__(self, in_channels: int, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_classes)
        
    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(x.size(0), -1)
        return self.head(feats)


def get_model(config: dict) -> nn.Module:
    """
    Factory function to create a model from configuration.
    
    Args:
        config: Dictionary with model configuration
            - name: Model class name (e.g., "SmallConvNet")
            - args: Dictionary of arguments to pass to the model constructor
    
    Returns:
        nn.Module: The instantiated model
    """
    model_name = config.get("name", "SmallConvNet")
    model_args = config.get("args", {})
    
    if model_name == "SmallConvNet":
        return SmallConvNet(**model_args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
