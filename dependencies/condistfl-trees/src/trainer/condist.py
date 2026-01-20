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

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from losses import ClsConDistLoss, MarginalCELoss
from model import get_model
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ConDistTrainer(object):
    """
    Conditional Distillation Trainer for Tree Classification.
    
    Each client specializes in one tree class (nizka, stredna, or vysoka).
    The client receives all data but only considers its class as labeled.
    It learns from the global model for other classes (background) using ConDist loss.
    """
    def __init__(self, task_config: Dict):
        self.init_lr = task_config["training"].get("lr", 1e-3)
        self.max_steps = task_config["training"]["max_steps"]
        self.max_rounds = task_config["training"]["max_rounds"]
        self.grad_clip_norm = task_config["training"].get("grad_clip_norm", 1.0)

        num_classes = len(task_config["classes"])
        foreground = task_config["condist_config"]["foreground"]
        background = task_config["condist_config"]["background"]
        temperature = task_config["condist_config"].get("temperature", 2.0)
        self.model_config = task_config["model"]
        self.weight_range = task_config["condist_config"]["weight_schedule_range"]

        # ConDist loss for learning from global model on background classes
        self.condist_loss_fn = ClsConDistLoss(
            num_classes, foreground, background, temperature=temperature, loss="ce"
        )
        
        # Marginal loss for supervised learning on foreground class
        self.marginal_loss_fn = MarginalCELoss(foreground)

        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None

    def update_condist_weight(self):
        """Update the weight of ConDist loss based on the current round."""
        left = min(self.weight_range)
        right = max(self.weight_range)
        intv = (right - left) / max(1, self.max_rounds - 1)
        self.weight = left + intv * self.current_round

    def configure_optimizer(self):
        """Configure optimizer and learning rate scheduler."""
        self.opt = Adam(self.model.parameters(), lr=self.init_lr, weight_decay=1e-4)
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        self.sch = CosineAnnealingLR(self.opt, T_max=self.max_steps, eta_min=1e-7)
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def training_step(self, model: nn.Module, batch, device: str = "cuda:0"):
        """
        Single training step.
        
        Args:
            model: Local model
            batch: Tuple of (image, label)
            device: Device to run on
        
        Returns:
            Total loss
        """
        image, label = batch
        image = image.to(device)
        label = label.to(device)

        # Forward pass on local model
        preds = model(image)  # (batch, num_classes)
        
        # Supervised loss on foreground class
        marginal_loss = self.marginal_loss_fn(preds, label)

        # ConDist loss: learn from global model on background classes
        with torch.no_grad():
            targets = self.global_model(image)  # (batch, num_classes)
        
        condist_loss = self.condist_loss_fn(preds, targets, label)

        # Total loss
        loss = marginal_loss + self.weight * condist_loss

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf detected! marginal_loss={marginal_loss.item()}, condist_loss={condist_loss.item()}, weight={self.weight}")
            # Return a safe fallback - just use marginal loss
            loss = marginal_loss

        # Log training information
        if self.logger is not None:
            step = self.current_step
            self.logger.add_scalar("loss", loss.item(), step)
            self.logger.add_scalar("loss_marginal", marginal_loss.item(), step)
            self.logger.add_scalar("loss_condist", condist_loss.item(), step)
            self.logger.add_scalar("lr", self.sch.get_last_lr()[-1], step)
            self.logger.add_scalar("condist_weight", self.weight, step)

        return loss

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        """Generator to iterate through batches."""
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        """
        Training loop for num_steps.
        
        Args:
            data_loader: Training data loader
            num_steps: Number of training steps
            device: Device to run on
        """
        self.model = self.model.to(device)
        self.global_model = self.global_model.to(device)
        self.model.train()

        target_step = self.current_step + num_steps
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for batch in self.get_batch(data_loader, num_steps):
                # Forward
                loss = self.training_step(self.model, batch, device)

                # Backward
                self.opt.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm, norm_type=2.0)

                # Apply gradient
                self.opt.step()
                self.sch.step()

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

    def setup(self, model: nn.Module, logger: SummaryWriter, abort_signal: Any):
        """
        Setup trainer for a new round.
        
        Args:
            model: The model to train
            logger: TensorBoard logger
            abort_signal: Signal to abort training
        """
        self.model = model
        # Create a copy of the model as global model (will be updated with global weights)
        self.global_model = get_model(self.model_config)
        self.global_model.load_state_dict(deepcopy(model.state_dict()))
        self.global_model.eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()
        self.update_condist_weight()

    def cleanup(self):
        """Cleanup after training."""
        # Save opt & sch states
        if self.opt is not None:
            self.opt_state = deepcopy(self.opt.state_dict())
        if self.sch is not None:
            self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None
        self.global_model = None

        self.logger = None
        self.abort = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def run(self, model: nn.Module, data_loader: DataLoader, num_steps: int, 
            logger: SummaryWriter = None, abort_signal: Any = None, device: str = "cuda:0"):
        """
        Run training for num_steps.
        
        Args:
            model: Model to train
            data_loader: Training data loader
            num_steps: Number of training steps
            logger: TensorBoard logger
            abort_signal: Signal to abort training
            device: Device to run on
        """
        self.setup(model, logger, abort_signal)
        self.training_loop(data_loader, num_steps, device)
        self.cleanup()
        self.current_round += 1

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        """Save training checkpoint."""
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        """Load training checkpoint."""
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model
