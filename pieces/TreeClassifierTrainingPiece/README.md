# CondistFL Tree Classification Trainer

## Overview

This Domino piece trains a federated learning model for tree height classification from hyperspectral BSQ imagery using the CondistFL (Continual Distillation for Federated Learning) framework with NVIDIA FLARE.

## Features

- **Hyperspectral Image Classification**: Processes BSQ format hyperspectral chips (50 bands, 32x32 pixels)
- **3-Class Tree Height Classification**: 
  - Nízka vegetácia (low vegetation)
  - Stredná vegetácia (medium vegetation)
  - Vysoká vegetácia (tall vegetation)
- **Federated Learning with ConDist**: Each client specializes in one class while learning from others
- **NVFlare Integration**: Uses NVIDIA FLARE simulator for distributed training
- **Pre-packaged Data**: Includes ~900 sampled BSQ chips ready for training
- **Power Tracking**: Monitors energy consumption during training with CodeCarbon

## Architecture

### Model
- **SmallConvNet**: Lightweight 2D CNN for hyperspectral classification
- Input: (batch, 50, 32, 32) - 50 spectral bands, 32x32 spatial
- Output: (batch, 3) - 3 tree height classes

### Federated Setup
- **3 Clients**: site-1, site-2, site-3
- Each client gets balanced data from all classes
- Client specialization via `local_class_id` parameter
- ConDist loss for learning from global model on non-specialized classes

## Input Parameters

- `num_rounds` (default: 50): Number of federated learning rounds
- `steps_per_round` (default: 100): Training steps per round
- `clients` (default: "site-1,site-2,site-3"): Comma-separated client names
- `gpus` (default: "0,1,2"): Comma-separated GPU IDs (one per client)
- `workspace_dir` (default: "/app/workspace"): Output directory for results

## Outputs

- `workspace_dir`: Directory with all training artifacts
- `best_global_model_path`: Path to best global model checkpoint (.pt)
- `global_model_path`: Path to final global model checkpoint
- `best_local_models`: Dictionary of best local models per client
- `training_complete`: Boolean indicating successful completion
- `num_rounds_completed`: Actual number of rounds completed
- `validation_metrics`: Validation accuracy/loss per client
- `message`: Status message

## Data

The piece includes pre-sampled BSQ chips:
- 300 chips per class (900 total)
- 80/20 train/validation split per client
- Data location in container: `/app/tree_bsq_chips_sampled`
- Datalists pre-configured with correct paths

## Usage Example

```python
from domino import workflow
from CondistFLTreesPiece import CondistFLTreesPiece

# Run federated training
result = CondistFLTreesPiece(
    num_rounds=50,
    steps_per_round=100,
    clients="site-1,site-2,site-3",
    gpus="0,1,2"
)

print(f"Training complete: {result.training_complete}")
print(f"Best model: {result.best_global_model_path}")
print(f"Validation metrics: {result.validation_metrics}")
```

## Technical Details

### Dependencies
- PyTorch 2.x
- NVFlare 2.6.2
- CodeCarbon (power tracking)
- TensorBoard (logging)

### Container Resources
- CPU: 4-16 cores
- Memory: 8-32 GB
- GPU: Required (multi-GPU supported)
- Shared Memory: 8 GB (for PyTorch DataLoader)

### Training Time
- ~2-5 minutes per round (depending on GPU)
- Total time for 50 rounds: ~2-4 hours

## References

- [CondistFL Framework](https://github.com/NVIDIA/NVFlare)
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)

## Version

0.0.1 - Initial release
