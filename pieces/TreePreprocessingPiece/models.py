from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """Tree Preprocessing Input Model"""
    client_datalists: dict = Field(
        description="Client datalist paths from TreeDataSplitPiece"
    )
    output_dir: str = Field(
        description="Directory containing datalist files"
    )


class OutputModel(BaseModel):
    """Tree Preprocessing Output Model"""
    client_datalists: dict = Field(
        description="Client datalist paths (currently unchanged)"
    )
    output_dir: str = Field(
        description="Directory containing datalist files"
    )
    preprocessing_applied: bool = Field(
        description="Whether preprocessing was applied"
    )
