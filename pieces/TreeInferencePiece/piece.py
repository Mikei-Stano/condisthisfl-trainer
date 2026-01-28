from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import torch
import numpy as np
from pathlib import Path
import csv
import sys
sys.path.append('/app/condistfl-trees/src')
from model import TreeClassifier


class TreeInferencePiece(BasePiece):
    """
    Tree Inference Piece
    
    Runs inference on hyperspectral BSQ files using trained tree height classification model.
    Loads PyTorch model and predicts tree height classes (nizka, stredna, vysoka vegetacia).
    """
    
    CLASS_NAMES = ['nizka_vegetacia', 'stredna_vegetacia', 'vysoka_vegetacia']

    def piece_function(self, input_model: InputModel) -> OutputModel:
        """
        Run inference on BSQ files
        
        Args:
            input_model: Contains model_path and input_bsq_paths
            
        Returns:
            OutputModel with predictions list and predictions_csv path
        """
        model_path = Path(input_model.model_path)
        bsq_paths = [Path(p) for p in input_model.input_bsq_paths]
        
        self.logger.info(f"Loading model from: {model_path}")
        self.logger.info(f"Running inference on {len(bsq_paths)} files")
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self._load_model(model_path, device)
        model.eval()
        
        # Run inference
        predictions = []
        
        with torch.no_grad():
            for bsq_path in bsq_paths:
                pred_dict = self._infer_single_file(
                    model=model,
                    bsq_path=bsq_path,
                    device=device,
                    batch_size=input_model.batch_size
                )
                predictions.append(pred_dict)
                
                self.logger.info(f"Predicted {bsq_path.name}: {pred_dict['predicted_class']} (confidence: {pred_dict['confidence']:.3f})")
        
        # Save predictions to CSV
        output_dir = Path("/app/outputs/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "predictions.csv"
        self._save_predictions_csv(predictions, csv_path)
        
        self.logger.info(f"Saved predictions to: {csv_path}")
        
        return OutputModel(
            predictions=predictions,
            predictions_csv=str(csv_path)
        )
    
    def _load_model(self, model_path: Path, device: torch.device) -> torch.nn.Module:
        """
        Load trained PyTorch model
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model architecture
        # TODO: Get actual model config from checkpoint or config file
        model = TreeClassifier(
            in_channels=50,  # Typical hyperspectral bands
            num_classes=3,
            hidden_dim=128
        )
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        
        self.logger.info(f"Loaded model on device: {device}")
        return model
    
    def _infer_single_file(
        self, 
        model: torch.nn.Module, 
        bsq_path: Path, 
        device: torch.device,
        batch_size: int
    ) -> dict:
        """
        Run inference on a single BSQ file
        
        Returns dictionary with prediction results
        """
        # Load BSQ file
        # TODO: Implement actual BSQ loading logic
        # For now, create dummy data
        self.logger.warning(f"BSQ loading not implemented, using dummy data for {bsq_path.name}")
        
        # Dummy inference
        dummy_input = torch.randn(1, 50, 32, 32).to(device)  # (batch, channels, height, width)
        
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        
        confidence, predicted_class_id = torch.max(probs, dim=1)
        
        predicted_class_id = predicted_class_id.item()
        confidence = confidence.item()
        predicted_class = self.CLASS_NAMES[predicted_class_id]
        
        return {
            'file': str(bsq_path),
            'predicted_class': predicted_class,
            'class_id': int(predicted_class_id),
            'confidence': float(confidence)
        }
    
    def _save_predictions_csv(self, predictions: list, csv_path: Path):
        """
        Save predictions to CSV file
        """
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'predicted_class', 'class_id', 'confidence'])
            writer.writeheader()
            writer.writerows(predictions)
