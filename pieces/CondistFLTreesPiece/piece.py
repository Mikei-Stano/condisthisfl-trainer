import os
import json
import subprocess
from pathlib import Path
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLTreesPiece(BasePiece):
    """
    CondistFL Federated Learning Trainer for Tree Classification
    
    This piece trains a federated learning model for hyperspectral tree height
    classification using the CondistFL framework with NVFlare.
    
    The training process:
    1. Uses pre-configured BSQ data and datalists (already in container)
    2. Launches NVFlare simulator with 3 clients (site-1, site-2, site-3)
    3. Trains for specified number of rounds
    4. Saves best global and local models
    5. Performs validation on each client
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        """
        Execute the CondistFL federated training for tree classification
        
        Args:
            input_data: InputModel with training configuration
            
        Returns:
            OutputModel with training results
        """
        # Base directory inside the container where code and jobs were copied
        base_dir = Path("/app")
        
        # Ensure workspace directory exists
        workspace_path = Path(input_data.workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting CondistFL Tree Classification training")
        self.logger.info(f"Workspace: {workspace_path}")
        self.logger.info(f"Clients: {input_data.clients}")
        self.logger.info(f"GPUs: {input_data.gpus}")
        self.logger.info(f"Rounds: {input_data.num_rounds}")
        self.logger.info(f"Steps per round: {input_data.steps_per_round}")
        
        # Note: Data paths are already configured in the datalist CSVs
        # They point to /app/tree_bsq_chips_sampled which is in the container
        
        clients = [c.strip() for c in input_data.clients.split(',')]
        jobs_dir = base_dir / "jobs" / "condist"
        
        # Prepare the training command
        cmd = [
            "nvflare", "simulator",
            "-w", str(workspace_path.absolute()),
            "-c", input_data.clients,
            "-gpu", input_data.gpus,
            "-n", str(input_data.num_rounds),
            str(jobs_dir.absolute())
        ]
        
        # Set environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{base_dir}/src:{env.get('PYTHONPATH', '')}"
        
        self.logger.info(f"Running training command: {' '.join(cmd)}")
        
        # Execute the training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(base_dir),
                env=env,
                timeout=7200  # 2 hour timeout
            )
            self.logger.info("Training completed successfully")
            self.logger.info(f"Training stdout:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"Training stderr:\n{result.stderr}")
            
            training_complete = True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Training timed out after 2 hours")
            training_complete = False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed with error: {e.stderr}")
            training_complete = False
        
        # Look for the best global model
        best_model_path = workspace_path / "server" / "simulate_job" / "app_server" / "best_FL_global_model.pt"
        global_model_path = workspace_path / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
        
        if best_model_path.exists():
            best_model = str(best_model_path.absolute())
            self.logger.info(f"Found best global model: {best_model}")
        else:
            best_model = "Not found - training may have failed"
            self.logger.warning("Best global model not found")
            
        if global_model_path.exists():
            global_model = str(global_model_path.absolute())
            self.logger.info(f"Found final global model: {global_model}")
        else:
            global_model = "Not found - training may have failed"
            self.logger.warning("Final global model not found")
        
        # Look for best local models for each client
        best_local_models = {}
        for client in clients:
            local_model_path = workspace_path / client / "models" / "best_model.pt"
            if local_model_path.exists():
                best_local_models[client] = str(local_model_path.absolute())
                self.logger.info(f"Found best local model for {client}: {local_model_path}")
            else:
                self.logger.warning(f"Best local model not found for {client}")
        
        # Parse validation metrics from TensorBoard logs
        validation_metrics = {}
        for client in clients:
            log_file = workspace_path / client / "local" / "log.txt"
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        # Try to extract metrics from log
                        # This is a simple approach - metrics are in TensorBoard format
                        # Could parse them more sophisticatedly if needed
                        if "val_accuracy" in log_content:
                            self.logger.info(f"Found validation metrics for {client} in log")
                except Exception as e:
                    self.logger.warning(f"Could not parse log file for {client}: {e}")
        
        # Try to read validation metrics from each client's TensorBoard events
        for client in clients:
            tb_dir = workspace_path / client / "local" / "tb_events"
            if tb_dir.exists():
                self.logger.info(f"TensorBoard events found for {client}: {tb_dir}")
                # Could parse TensorBoard events here if needed
        
        # Count completed rounds
        num_rounds_completed = input_data.num_rounds if training_complete else 0
        
        message = "Training completed successfully" if training_complete else "Training failed or incomplete"
        
        return OutputModel(
            workspace_dir=str(workspace_path.absolute()),
            best_global_model_path=best_model,
            global_model_path=global_model,
            best_local_models=best_local_models,
            training_complete=training_complete,
            num_rounds_completed=num_rounds_completed,
            validation_metrics=validation_metrics,
            message=message
        )

    # Override default container resources for federated learning training
    container_resources = {
        "requests": {
            "cpu": 4000,
            "memory": 8192
        },
        "limits": {
            "cpu": 16000,
            "memory": 32768
        },
        "use_gpu": True,
        "shm_size": 8192  # 8GB shared memory for PyTorch DataLoader multiprocessing
    }
