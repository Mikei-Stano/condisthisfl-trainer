from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_models import TreeSampleInfo, DatasetInfo
import pandas as pd
import glob
import os


class TreeDataLoaderPiece(BasePiece):
    """
    Tree Data Loader Piece
    
    Discovers BSQ hyperspectral files and loads tree height class labels.
    Currently reads from the existing dataset location. In the future, this
    could be extended to download from remote storage or cloud buckets.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        self.logger.info("=" * 60)
        self.logger.info("Starting TreeDataLoaderPiece execution")
        self.logger.info("=" * 60)
        
        dataset_path = Path(input_data.dataset_path)
        labels_file = input_data.labels_file
        
        self.logger.info(f"Dataset path: {dataset_path}")
        self.logger.info(f"Labels file: {labels_file}")
        
        # Validate dataset exists
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Find labels file
        if Path(labels_file).is_absolute():
            labels_path = Path(labels_file)
        else:
            labels_path = dataset_path / labels_file
        
        if not labels_path.exists():
            raise ValueError(f"Labels file not found: {labels_path}")
        
        self.logger.info(f"Loading labels from: {labels_path}")
        
        # Read labels CSV
        df = pd.read_csv(labels_path)
        self.logger.info(f"Loaded {len(df)} label records")
        
        # Map class names to IDs
        class_map = {
            "nizka_vegetacia": 0,
            "stredna_vegetacia": 1,
            "vysoka_vegetacia": 2
        }
        
        samples = []
        for idx, row in df.iterrows():
            bsq_path = row['bsq']
            class_label = row['class']
            
            # Extract sample ID and project from path
            bsq_file = Path(bsq_path)
            sample_id = bsq_file.stem
            project = bsq_file.parts[-3] if len(bsq_file.parts) >= 3 else None
            
            samples.append(TreeSampleInfo(
                sample_id=sample_id,
                bsq_path=str(bsq_path),
                class_label=class_label,
                class_id=class_map.get(class_label, -1),
                project=project
            ))
        
        # Calculate class distribution
        class_dist = {}
        for sample in samples:
            class_dist[sample.class_label] = class_dist.get(sample.class_label, 0) + 1
        
        self.logger.info(f"Dataset summary:")
        self.logger.info(f"  Total samples: {len(samples)}")
        self.logger.info(f"  Class distribution:")
        for cls, count in class_dist.items():
            self.logger.info(f"    {cls}: {count} ({count/len(samples)*100:.1f}%)")
        
        dataset_info = DatasetInfo(
            dataset_path=str(dataset_path),
            total_samples=len(samples),
            class_distribution=class_dist,
            labels_file=str(labels_path)
        )
        
        self.logger.info("=" * 60)
        self.logger.info("TreeDataLoaderPiece completed successfully")
        self.logger.info("=" * 60)
        
        return OutputModel(
            samples=samples,
            dataset_info=dataset_info,
            total_samples=len(samples),
            class_distribution=class_dist
        )
