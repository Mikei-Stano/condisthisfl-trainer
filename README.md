# CondistFL Tree Classification Trainer - Domino Piece

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/Mikei-Stano/condisthisfl-trainer)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Domino piece for federated learning training of hyperspectral tree height classification using CondistFL (Continual Distillation) framework with NVIDIA FLARE.

## Overview

This piece enables federated learning across multiple sites for classifying tree height from hyperspectral BSQ imagery into three categories:
- **Nízka vegetácia** (low vegetation)
- **Stredná vegetácia** (medium vegetation)  
- **Vysoká vegetácia** (tall vegetation)

### Key Features

- 🌲 **Hyperspectral Classification**: 50-band BSQ imagery processing
- 🤝 **Federated Learning**: 3-client training with NVIDIA FLARE 2.6.2
- 🎯 **ConDist Loss**: Client specialization with continual distillation
- 📦 **Pre-packaged Data**: Sampled BSQ chips included in container
- ⚡ **GPU Accelerated**: Multi-GPU support for parallel training
- 📊 **Power Tracking**: CodeCarbon integration for sustainability metrics

## Architecture

### Model
- **SmallConvNet**: Lightweight 2D CNN for hyperspectral data
- **Input**: (batch, 50, 32, 32) - 50 spectral bands, 32×32 spatial
- **Output**: (batch, 3) - 3 tree height classes

### Federated Setup
- **3 Clients**: site-1, site-2, site-3
- Each client trains on balanced data from all classes
- ConDist loss enables learning from global model on non-specialized classes
- 80/20 train/validation split per client

## Quick Start

### Prerequisites
- Docker with GPU support
- NVIDIA GPU with CUDA support
- 25+ GB disk space
- Domino CLI (for deployment)

### Building the Docker Image

```bash
# Clone the repository
git clone https://github.com/Mikei-Stano/condisthisfl-trainer.git
cd condisthisfl-trainer

# Prepare sampled data (300 chips per class)
cd /path/to/dicris
python prepare_sampled_data.py --subset 300 --dst condisthisfl-trainer

# Build with Domino CLI
cd condisthisfl-trainer
domino piece organize
```

### Usage in Domino Workflows

```python
from CondistFLTreesPiece import CondistFLTreesPiece

# Execute federated training
result = CondistFLTreesPiece(
    num_rounds=50,
    steps_per_round=100,
    clients="site-1,site-2,site-3",
    gpus="0,1,2",
    workspace_dir="/app/workspace"
)

# Access results
print(f"Training complete: {result.training_complete}")
print(f"Best model: {result.best_global_model_path}")
print(f"Metrics: {result.validation_metrics}")
```

## Project Structure

```
condisthisfl-trainer/
├── config.toml                    # Repository configuration
├── .gitignore                     # Git ignore rules
├── prepare_sampled_data.py        # Data sampling script (run locally)
├── dependencies/
│   ├── Dockerfile_condistfl_trees # Container build file
│   └── condistfl-trees/
│       ├── src/                   # Source code
│       ├── jobs/                  # NVFlare job configs
│       ├── tree_bsq_chips_sampled/ # Sampled data (not in git)
│       └── run_validate.py
└── pieces/
    └── CondistFLTreesPiece/
        ├── piece.py               # Main execution logic
        ├── models.py              # Input/output schemas
        ├── metadata.json          # Domino piece metadata
        └── README.md              # Piece documentation
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | int | 50 | Number of federated learning rounds |
| `steps_per_round` | int | 100 | Training steps per round |
| `clients` | str | "site-1,site-2,site-3" | Comma-separated client names |
| `gpus` | str | "0,1,2" | GPU IDs (one per client) |
| `workspace_dir` | str | "/app/workspace" | Output directory |

## Outputs

- `workspace_dir`: Training artifacts and logs
- `best_global_model_path`: Best global model checkpoint (.pt)
- `global_model_path`: Final global model
- `best_local_models`: Per-client best models (dict)
- `training_complete`: Success status (bool)
- `num_rounds_completed`: Actual rounds finished (int)
- `validation_metrics`: Per-client accuracy/loss (dict)
- `message`: Status message (str)

## Data Preparation

The sampled BSQ data is **not included in git** due to size (1.3GB). To prepare the data:

```bash
# From the dicris directory with tree_bsq_chips data
python prepare_sampled_data.py \
    --src tree_bsq_chips \
    --dst condisthisfl-trainer \
    --subset 300 \
    --n_clients 3
```

This will:
1. Sample 300 chips per class (900 total)
2. Copy BSQ and HDR files to `dependencies/condistfl-trees/tree_bsq_chips_sampled/`
3. Generate datalist CSVs with container paths (`/app/tree_bsq_chips_sampled/`)
4. Split data across 3 clients with 80/20 train/val

## Dependencies

### Container Base
- `nvcr.io/nvidia/pytorch:24.12-py3`

### Python Packages
- `nvflare==2.6.2` - Federated learning framework
- `torch>=2.0.0` - Deep learning
- `pandas>=1.3.0` - Data handling
- `tensorboard==2.16.2` - Logging
- `codecarbon==2.3.0` - Power tracking
- `domino-py[cli]>=0.9.0` - Domino integration

## Resource Requirements

- **CPU**: 4-16 cores
- **Memory**: 8-32 GB RAM
- **GPU**: Required (3 GPUs recommended for parallel clients)
- **Shared Memory**: 8 GB (for PyTorch DataLoader)
- **Disk**: ~25 GB (image + workspace)

## Development

### Local Testing

```bash
# Build Docker image
docker build -f dependencies/Dockerfile_condistfl_trees \
    -t condisthisfl-trainer:test .

# Run interactively
docker run --gpus all -it condisthisfl-trainer:test bash

# Inside container, verify setup
python -c "from pieces.CondistFLTreesPiece import CondistFLTreesPiece; print('OK')"
ls -lh /app/tree_bsq_chips_sampled/ | head
head /app/jobs/condist/site-1/config/train_datalist.csv
```

### Modifying the Piece

1. Edit source code in `dependencies/condistfl-trees/src/`
2. Update piece logic in `pieces/CondistFLTreesPiece/piece.py`
3. Modify input/output schemas in `pieces/CondistFLTreesPiece/models.py`
4. Update metadata in `pieces/CondistFLTreesPiece/metadata.json`
5. Rebuild with `domino piece organize`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details

## References

- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [CondistFL Framework](https://github.com/NVIDIA/NVFlare)
- [Domino Documentation](https://docs.dominodatalab.com/)

## Citation

If you use this work, please cite:

```bibtex
@software{condisthisfl_trainer,
  title = {CondistFL Tree Classification Trainer for Domino},
  author = {Stanojević, Michal},
  year = {2026},
  url = {https://github.com/Mikei-Stano/condisthisfl-trainer}
}
```

## Support

For issues and questions:
- Open an [issue](https://github.com/Mikei-Stano/condisthisfl-trainer/issues)
- Contact: [your-email@example.com]

## Acknowledgments

- Built on NVIDIA FLARE federated learning framework
- Inspired by the medical imaging CondistFL trainer
- Part of the dicris tree classification project

---

**Version**: v0.0.1  
**Last Updated**: January 2026
