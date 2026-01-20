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

import json
import os
import traceback
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
# import wandb  # Disabled for Domino piece
from codecarbon import EmissionsTracker
from data import DataManager
from model import get_model
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from trainer import ConDistTrainer
from utils.model_weights import extract_weights, load_weights
from validator import Validator

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType


class ConDistLearner(Learner):
    """
    ConDist Learner for tree classification with NVFlare.
    
    Each client specializes in one tree class but receives all data.
    The client learns from the global model for other classes using ConDist loss.
    
    Args:
        task_config: Path to task configuration JSON file
        data_config: Path to data configuration JSON file
        aggregation_steps: Number of training steps between aggregations
        local_class_id: The class ID that this client specializes in (0=nizka, 1=stredna, 2=vysoka)
        method: Training method (default: "ConDist")
        seed: Random seed
        max_retry: Maximum number of retries on errors
        train_task_name: Name of the training task
        submit_model_task_name: Name of the submit model task
    """
    def __init__(
        self,
        task_config: str,
        data_config: str,
        aggregation_steps: int,
        local_class_id: int,
        method: Literal["ConDist"] = "ConDist",
        device: str = "cuda:0",
        # use_wandb: bool = False,  # Disabled for Domino piece
        # wandb_project: str = "condistfl-trees",
        # wandb_api_key: Optional[str] = None,
        seed: Optional[int] = None,
        max_retry: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        super().__init__()

        self.task_config = task_config
        self.data_config = data_config
        self.aggregation_steps = aggregation_steps
        self.local_class_id = local_class_id
        self.device = device
        # self.use_wandb = False  # Disabled for Domino piece
        # self.wandb_project = wandb_project
        # self.wandb_api_key = wandb_api_key
        
        # Power tracking
        self.emissions_tracker = None

        self._method = method
        self._seed = seed
        self._max_retry = max_retry

        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name

    def initialize(self, parts: Dict, fl_ctx: FLContext) -> None:
        """Initialize the learner."""
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load configurations
        prefix = Path(self.app_root)
        with open(prefix / self.task_config) as f:
            task_config = json.load(f)

        with open(prefix / self.data_config) as f:
            data_config = json.load(f)

        # Initialize variables
        self.key_metric = "val_acc"
        self.best_metric = -np.inf
        self.best_model_path = "models/best_model.pt"
        self.last_model_path = "models/last.pt"

        # Create data manager
        self.dm = DataManager(self.app_root, data_config, local_class_id=self.local_class_id)

        # Create model
        self.model = get_model(task_config["model"])

        # Configure trainer & validator
        if self._method == "ConDist":
            self.trainer = ConDistTrainer(task_config)
        self.validator = Validator(task_config)

        # Create logger
        self.tb_logger = SummaryWriter(log_dir=prefix / "logs")
        
        # Start power tracking
        self.emissions_tracker = EmissionsTracker(
            project_name=f"condistfl-client-{self.local_class_id}",
            measure_power_secs=15,  # Measure every 15 seconds
            save_to_file=False,  # Don't save to file, only track in memory
            logging_logger=None,  # Disable codecarbon's own logging
        )
        self.emissions_tracker.start()
        self.log_info(fl_ctx, "Power tracking started")
        
        # Initialize WandB - Disabled for Domino piece
        # if self.use_wandb:
        #     # Set API key if provided
        #     if self.wandb_api_key:
        #         os.environ["WANDB_API_KEY"] = self.wandb_api_key
        #     
        #     client_name = fl_ctx.get_identity_name()
        #     class_names = ["nizka", "stredna", "vysoka"]
        #     wandb.init(
        #         project=self.wandb_project,
        #         name=f"{client_name}",
        #         tags=[f"client_{self.local_class_id}", class_names[self.local_class_id]],
        #         config={
        #             "method": self._method,
        #             "local_class_id": self.local_class_id,
        #             "aggregation_steps": self.aggregation_steps,
        #             "model": task_config["model"]["name"],
        #             "lr": task_config["training"]["lr"],
        #             "device": self.device,
        #         },
        #         reinit=True,
        #     )

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Train the model for aggregation_steps.
        
        Args:
            data: Shareable containing global model weights
            fl_ctx: Federated learning context
            abort_signal: Signal to abort training
        
        Returns:
            Shareable containing updated model weights
        """
        # Log training info
        num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{num_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        self.log_info(fl_ctx, f"Local class ID: {self.local_class_id}")

        # Get global weights
        dxo = from_shareable(data)
        global_weights = dxo.data

        # Create dataset & data loader (if necessary)
        if self.dm.get_data_loader("train") is None:
            self.dm.setup("train")
        if self.dm.get_data_loader("validate") is None:
            self.dm.setup("validate")

        # Load global weights into local model
        load_weights(self.model, global_weights)

        # Run training
        for i in range(self._max_retry + 1):
            try:
                self.trainer.run(
                    self.model,
                    self.dm.get_data_loader("train"),
                    num_steps=self.aggregation_steps,
                    logger=self.tb_logger,
                    abort_signal=abort_signal,
                    device=self.device,
                )
                break
            except Exception as e:
                if i < self._max_retry:
                    self.log_warning(fl_ctx, f"Something wrong in training, retrying ({i+1}/{self._max_retry}).")
                    self.log_warning(fl_ctx, str(e))
                    # Restore to global weights
                    load_weights(self.model, global_weights)
                    # Reset dataset & dataloader
                    self.dm._data_loader["train"] = None
                    self.dm._dataset["train"] = None
                    self.dm.setup("train")
                else:
                    self.log_error(fl_ctx, traceback.format_exc())
                    raise RuntimeError(traceback.format_exc())

        # Run validation
        for i in range(self._max_retry + 1):
            try:
                metrics = self.validator.run(self.model, self.dm.get_data_loader("validate"), device=self.device)
                break
            except Exception as e:
                if i < self._max_retry:
                    self.log_warning(fl_ctx, f"Something wrong in validation, retrying ({i+1}/{self._max_retry}).")
                    self.log_warning(fl_ctx, str(e))
                    # Reset dataset & dataloader
                    self.dm._data_loader["validate"] = None
                    self.dm._dataset["validate"] = None
                    self.dm.setup("validate")
                else:
                    self.log_error(fl_ctx, traceback.format_exc())
                    raise RuntimeError(traceback.format_exc())

        # Log metrics
        self.log_info(fl_ctx, f"Validation metrics: {metrics}")
        for key, value in metrics.items():
            self.tb_logger.add_scalar(key, value, current_round)
        
        # Log to WandB - Disabled for Domino piece
        # if self.use_wandb:
        #     wandb_metrics = {f"train/{k}": v for k, v in metrics.items()}
        #     wandb_metrics["train/round"] = current_round + 1
        #     
        #     # Add power consumption metrics if available
        #     if self.emissions_tracker:
        #         # Get current emissions data
        #         emissions_data = self.emissions_tracker._prepare_emissions_data()
        #         if emissions_data:
        #             wandb_metrics["power/energy_consumed_kwh"] = emissions_data.energy_consumed
        #             wandb_metrics["power/emissions_kg_co2"] = emissions_data.emissions
        #             wandb_metrics["power/cpu_power_w"] = emissions_data.cpu_power
        #             wandb_metrics["power/gpu_power_w"] = emissions_data.gpu_power
        #             wandb_metrics["power/ram_power_w"] = emissions_data.ram_power
        #     
        #     wandb.log(wandb_metrics)

        # Save best model
        if metrics[self.key_metric] > self.best_metric:
            self.best_metric = metrics[self.key_metric]
            self.trainer.save_checkpoint(self.best_model_path, self.model)
            self.log_info(fl_ctx, f"New best model saved with {self.key_metric}={self.best_metric:.4f}")
            if self.use_wandb:
                wandb.log({"train/best_val_acc": self.best_metric})

        # Save last model
        self.trainer.save_checkpoint(self.last_model_path, self.model)

        # Create output DXO
        weights = extract_weights(self.model)
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.aggregation_steps},
        )
        
        return outgoing_dxo.to_shareable()

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Validate the model on validation set.
        
        Args:
            data: Shareable containing model weights to validate
            fl_ctx: Federated learning context
            abort_signal: Signal to abort validation
        
        Returns:
            Shareable containing validation metrics
        """
        # Get model type
        model_owner = data.get_header(AppConstants.MODEL_OWNER, "?")
        validate_type = data.get_header(AppConstants.VALIDATE_TYPE)

        self.log_info(fl_ctx, f"Validating model from {model_owner}")

        # Get model weights
        dxo = from_shareable(data)
        global_weights = dxo.data

        # Load weights
        load_weights(self.model, global_weights)

        # Create validation data loader if necessary
        if self.dm.get_data_loader("validate") is None:
            self.dm.setup("validate")

        # Run validation
        metrics = self.validator.run(self.model, self.dm.get_data_loader("validate"), device=self.device)

        # Log metrics
        self.log_info(fl_ctx, f"Validation metrics: {metrics}")
        
        # Log global model validation to WandB
        if self.use_wandb:
            wandb_metrics = {f"global/{k}": v for k, v in metrics.items()}
            wandb_metrics["global/model"] = model_owner
            wandb.log(wandb_metrics)

        # Create output DXO
        outgoing_dxo = DXO(data_kind=DataKind.METRICS, data=metrics)
        return outgoing_dxo.to_shareable()

    def submit_model(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Submit local model for cross-site validation.
        This is called by CrossSiteModelEval to collect local models.
        """
        # We don't submit local models in this setup - only validate global models
        # Return empty shareable to avoid error
        return make_reply(ReturnCode.OK)

    def finalize(self, fl_ctx: FLContext) -> None:
        """Finalize the learner."""
        self.log_info(fl_ctx, "Finalizing learner")
        
        # Stop power tracking and log final stats
        if self.emissions_tracker:
            try:
                emissions = self.emissions_tracker.stop()
                self.log_info(fl_ctx, f"Total energy consumed: {emissions:.6f} kWh")
                if self.use_wandb and emissions:
                    wandb.log({
                        "power/total_energy_kwh": emissions,
                    })
            except Exception as e:
                self.log_warning(fl_ctx, f"Failed to stop emissions tracker: {e}")
        
        if self.tb_logger:
            self.tb_logger.close()
        if self.use_wandb:
            wandb.finish()
