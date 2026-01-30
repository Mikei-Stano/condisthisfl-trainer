from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_models import TreeSampleInfo
import pandas as pd
import numpy as np
import math


class TreeDataSplitPiece(BasePiece):
    """
    Tree Data Split Piece
    
    Splits tree samples across federated learning clients using round-robin
    distribution to ensure balanced class distribution. Creates train/validation
    datalist CSVs for each client that can be used by NVFlare training.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        self.logger.info("=" * 60)
        self.logger.info("Starting TreeDataSplitPiece execution")
        self.logger.info("=" * 60)
        
        samples = input_data.samples
        n_clients = input_data.n_clients
        val_split = input_data.val_split
        seed = input_data.random_seed
        
        # Generate client names
        client_names = [f"site-{i+1}" for i in range(n_clients)]
        output_dir = Path("/tmp/tree_datalists")
        
        self.logger.info(f"Input configuration:")
        self.logger.info(f"  Total samples: {len(samples)}")
        self.logger.info(f"  Number of clients: {n_clients}")
        self.logger.info(f"  Client names: {client_names}")
        self.logger.info(f"  Validation split: {val_split}")
        self.logger.info(f"  Random seed: {seed}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle samples
        rng = np.random.default_rng(seed)
        shuffled_samples = samples.copy()
        rng.shuffle(shuffled_samples)
        
        # Assign samples to clients using round-robin
        client_samples = {name: [] for name in client_names[:n_clients]}
        for idx, sample in enumerate(shuffled_samples):
            client_idx = idx % n_clients
            client_samples[client_names[client_idx]].append(sample)
        
        client_datalists = {}
        split_summary = {}
        
        for client_name, client_data in client_samples.items():
            self.logger.info(f"\nProcessing client: {client_name}")
            self.logger.info(f"  Total samples: {len(client_data)}")
            
            # Split into train/val
            n_val = max(1, int(math.ceil(len(client_data) * val_split)))
            val_idxs = rng.choice(len(client_data), size=n_val, replace=False)
            val_mask = np.zeros(len(client_data), dtype=bool)
            val_mask[val_idxs] = True
            
            train_samples = [s for i, s in enumerate(client_data) if not val_mask[i]]
            val_samples = [s for i, s in enumerate(client_data) if val_mask[i]]
            
            self.logger.info(f"  Train samples: {len(train_samples)}")
            self.logger.info(f"  Val samples: {len(val_samples)}")
            
            # Create DataFrames
            train_df = pd.DataFrame([
                {"bsq": s.bsq_path, "class": s.class_label}
                for s in train_samples
            ])
            val_df = pd.DataFrame([
                {"bsq": s.bsq_path, "class": s.class_label}
                for s in val_samples
            ])
            
            # Save CSVs
            train_path = output_dir / f"{client_name}_train_datalist.csv"
            val_path = output_dir / f"{client_name}_val_datalist.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            self.logger.info(f"  Saved train datalist: {train_path}")
            self.logger.info(f"  Saved val datalist: {val_path}")
            
            client_datalists[client_name] = {
                "train": str(train_path),
                "val": str(val_path)
            }
            
            # Class distribution
            train_dist = train_df["class"].value_counts().to_dict()
            val_dist = val_df["class"].value_counts().to_dict()
            
            split_summary[client_name] = {
                "total": len(client_data),
                "train": len(train_samples),
                "val": len(val_samples),
                "train_distribution": train_dist,
                "val_distribution": val_dist
            }
        
        self.logger.info("=" * 60)
        self.logger.info("TreeDataSplitPiece completed successfully")
        self.logger.info("=" * 60)
        
        return OutputModel(
            client_datalists=client_datalists,
            split_summary=split_summary,
            output_dir=str(output_dir),
            total_samples=len(samples)
        )
