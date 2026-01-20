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

from typing import Optional, Sequence, Union, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class ClsConDistLoss(_Loss):
    """
    Conditional Distillation Loss for Classification.
    
    This loss is adapted for tree classification where each client specializes in one class.
    - num_classes: Total number of classes (3 for our tree case: nizka, stredna, vysoka)
    - foreground: The class ID that this client specializes in (e.g., [0] for nizka)
    - background: The other class IDs that this client treats as "background" (e.g., [1, 2] for stredna, vysoka)
    """
    def __init__(
        self,
        num_classes: int,
        foreground: Sequence[int],
        background: Sequence[Union[int, Sequence[int]]],
        temperature: float = 2.0,
        loss: Literal["ce", "kl"] = "ce"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.foreground = foreground
        self.background = background

        if temperature < 0.0:
            raise ValueError("Softmax temperature must be a positive number!")
        self.temperature = temperature

        if loss not in ["ce", "kl"]:
            raise ValueError(f"Unsupported loss type: {loss}")
        self.loss = loss

    def softmax(self, data: Tensor):
        return torch.softmax(data / self.temperature, dim=1)

    def ce(
        self,
        preds: Tensor,
        targets: Tensor,
        eps: float = 1e-6,
        mask: Optional[Tensor] = None
    ):
        log_p = torch.log(preds.clamp(min=eps))
        loss = -torch.sum(targets * log_p, dim=1)
        if mask is not None:
            return (mask * loss).sum() / (mask.sum() + eps)
        return loss.mean()

    def kl_div(
        self,
        preds: Tensor,
        targets: Tensor,
        eps: float = 1e-6,
        mask: Optional[Tensor] = None
    ):
        log_p = torch.log(preds.clamp(min=eps))
        log_q = torch.log(targets.clamp(min=eps))

        loss = torch.sum(targets * (log_q - log_p), dim=1)
        if mask is not None:
            return (mask * loss).sum() / (mask.sum() + eps)
        return loss.mean()

    def reduce_channels(self, data: Tensor, eps: float = 1e-5):
        """
        Reduce the classification probabilities to conditional probabilities.
        - data: (batch, num_classes) tensor of probabilities
        Returns: (batch, num_background_groups) tensor of conditional probabilities
        """
        batch, channels = data.shape
        if channels != self.num_classes:
            raise ValueError(f"Expect input with {self.num_classes} channels, get {channels}")

        fg_shape = [batch, 1]
        bg_shape = [batch, len(self.background)]

        # Compute the probability for the union of local foreground
        fg = torch.zeros(fg_shape, dtype=torch.float32, device=data.device)
        for c in self.foreground:
            fg += data[:, c].view(*fg_shape)

        # Clamp foreground probability to prevent numerical issues
        fg = fg.clamp(max=1.0 - eps)

        # Compute the raw probabilities for each background group
        bg = torch.zeros(bg_shape, dtype=torch.float32, device=data.device)
        for i, g in enumerate(self.background):
            if isinstance(g, int):
                bg[:, i] = data[:, g]
            else:
                for c in g:
                    bg[:, i] += data[:, c]

        # Compute conditional probability for background groups with better numerical stability
        denominator = (1.0 - fg).clamp(min=eps)
        conditional_probs = bg / denominator
        
        # Normalize to ensure they sum to 1 (prevent numerical drift)
        conditional_probs = conditional_probs / (conditional_probs.sum(dim=1, keepdim=True) + eps)
        
        return conditional_probs

    def generate_mask(self, targets: Tensor, ground_truth: Tensor):
        """
        Generate a mask that covers the background but excludes false positive samples.
        - targets: (batch, num_classes) soft predictions from global model
        - ground_truth: (batch,) or (batch, 1) hard labels
        Returns: (batch,) mask tensor
        """
        # Get the predicted class from global model
        targets_pred = torch.argmax(targets, dim=1, keepdim=True)  # (batch, 1)
        
        # Reshape ground_truth to match
        if ground_truth.dim() == 1:
            ground_truth = ground_truth.view(-1, 1)  # (batch, 1)
        
        # The mask covers the background but excludes false positive areas
        # condition = 1 if sample belongs to our foreground class (either predicted or ground truth)
        condition = torch.zeros_like(targets_pred, device=targets_pred.device)
        for c in self.foreground:
            condition = torch.where(
                torch.logical_or(targets_pred == c, ground_truth == c), 
                1, 
                condition
            )
        mask = 1 - condition  # mask = 1 for background samples
        mask = mask.view(-1)  # Required to be flattened in classification task

        return mask.to(torch.float32)

    def forward(self, preds, targets, ground_truth):
        """
        Forward pass for ConDist loss.
        - preds: (batch, num_classes) logits from local model
        - targets: (batch, num_classes) logits from global model
        - ground_truth: (batch,) or (batch, 1) hard labels
        """
        mask = self.generate_mask(targets, ground_truth)

        preds = self.softmax(preds)
        preds = self.reduce_channels(preds)

        targets = self.softmax(targets)
        targets = self.reduce_channels(targets)

        if self.loss == "ce":
            return self.ce(preds, targets, mask=mask)
        return self.kl_div(preds, targets, mask=mask)


class MarginalCELoss(_Loss):
    """
    Marginal Cross-Entropy Loss for the local foreground class.
    This is the supervised loss on the labeled data.
    """
    def __init__(
        self,
        foreground: Sequence[int],
    ):
        super().__init__()
        self.foreground = foreground

    def forward(self, preds, targets):
        """
        - preds: (batch, num_classes) logits
        - targets: (batch,) hard labels
        """
        # Standard cross-entropy loss
        return F.cross_entropy(preds, targets)
