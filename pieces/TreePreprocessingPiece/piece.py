from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class TreePreprocessingPiece(BasePiece):
    """
    Tree Preprocessing Piece (Placeholder)
    
    Future home for BSQ hyperspectral preprocessing:
    - Normalization
    - Band selection
    - Noise filtering
    - Data augmentation
    
    Currently passes data through unchanged.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        self.logger.info("TreePreprocessingPiece: Currently a placeholder")
        self.logger.info("Data passed through unchanged")
        self.logger.info("Future: Add BSQ preprocessing here")
        
        return OutputModel(
            client_datalists=input_data.client_datalists,
            output_dir=input_data.output_dir,
            preprocessing_applied=False
        )
