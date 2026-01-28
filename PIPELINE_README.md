# CondistFL Trees Multi-Piece Pipeline

Modular federated learning pipeline for hyperspectral tree height classification using Domino Pieces.

## Architecture

This repository contains 6 Domino pieces organized into 2 dependency groups:

### Group 0 (Lightweight - Dockerfile_base)
- **TreeDataLoaderPiece**: Discovers BSQ files and loads tree height class labels
- **TreeDataSplitPiece**: Splits data across federated clients with train/val datalists
- **TreePreprocessingPiece**: Placeholder for BSQ preprocessing (future: normalization, augmentation)
- **TreeVisualizationPiece**: Visualizes training results from TensorBoard logs

### Group 1 (Heavy - Dockerfile_condistfl_trees)
- **TreeClassifierTrainingPiece**: Runs NVFlare federated learning training
- **TreeInferencePiece**: Runs inference on new BSQ files using trained model

## Pipeline Flow

```
TreeDataLoaderPiece 
  → TreeDataSplitPiece 
    → TreePreprocessingPiece (optional)
      → TreeClassifierTrainingPiece 
        → TreeVisualizationPiece (analyze results)
        → TreeInferencePiece (deploy model)
```

## Data Models

Pieces communicate using typed Pydantic models defined in `pieces/common_models.py`:

- **TreeSampleInfo**: Represents a single tree sample (BSQ path, class label, project)
- **DatasetInfo**: Summary of dataset (total samples, class distribution, labels file)

## Building

Build both dependency groups:

```bash
domino piece organize
```

This creates two Docker images:
- `ghcr.io/mikei-stano/condisthisfl-trainer:v0.1.0-group0` (lightweight)
- `ghcr.io/mikei-stano/condisthisfl-trainer:v0.1.0-group1` (PyTorch + NVFlare)

## Usage in Domino

1. **Data Loading**: Use TreeDataLoaderPiece to discover BSQ files from `/app/tree_bsq_chips_sampled`
2. **Data Splitting**: Split data across N clients (e.g., 3) with train/val ratios
3. **Training**: Run federated learning with NVFlare across clients
4. **Visualization**: Analyze training curves and metrics
5. **Inference**: Deploy trained model on new data

## Configuration

### TreeClassifierTrainingPiece Parameters
- `client_datalists`: Dictionary of client train/val datalist paths (from TreeDataSplitPiece)
- `num_rounds`: Number of federated learning rounds (default: 50)
- `steps_per_round`: Training steps per round (default: 100)
- `workspace_dir`: Where to save training outputs (default: /app/workspace)

### TreeDataSplitPiece Parameters
- `tree_samples`: List of TreeSampleInfo from TreeDataLoaderPiece
- `n_clients`: Number of federated clients (default: 3)
- `val_split`: Validation split ratio (default: 0.2)
- `client_names`: Client identifiers (default: ["site-1", "site-2", "site-3"])

## Version History

- **v0.1.0**: Multi-piece architecture with 6 modular pieces
- **v0.0.13**: Single training piece with fixed nvflare command
- **v0.0.11**: Removed wandb/codecarbon dependencies
- **v0.0.8-v0.0.10**: Fixed Domino schema validation errors

## Development

### Adding New Pieces

1. Create piece directory in `pieces/`
2. Add `metadata.json`, `models.py`, `piece.py`
3. Update `dependencies_map.json` to assign to group0 or group1
4. Run `domino piece organize`

### Testing Locally

```bash
# Test data loader
domino piece run TreeDataLoaderPiece --input '{"data_dir": "/app/tree_bsq_chips_sampled"}'

# Test data split
domino piece run TreeDataSplitPiece --input '{"tree_samples": [...], "n_clients": 3}'
```

## References

- NVFlare: https://github.com/NVIDIA/NVFlare
- Domino Pieces: https://docs.dominodatalab.com/
- CondistFL Paper: [Add link to paper if available]
