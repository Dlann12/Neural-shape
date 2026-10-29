"""Request models for the NeuralShape API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

ReconstructMethod = Literal["baseline", "ml_multiview", "ml_singleview", "both"]


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: str = "chair"
    auto_generate_synthetic_views: bool = True
    seed: Optional[int] = None


class ReconstructRequest(BaseModel):
    method: ReconstructMethod = "both"


class TrainExperimentRequest(BaseModel):
    name: str = "Experiment C (Multi-View ML)"
    epochs: int = Field(12, ge=1, le=200)
    batch_size: int = Field(8, ge=1, le=64)
    lr: float = Field(2e-3, gt=0, le=0.1)
    seed: int = 42
    num_train_objects: int = Field(60, ge=5, le=2000)
    num_val_objects: int = Field(15, ge=2, le=500)
    num_test_objects: int = Field(15, ge=2, le=500)
    train_augmented: bool = False
    promote_to_default: bool = Field(
        False, description="Replace the checkpoint used by /reconstruct if this run beats it on validation IoU."
    )


class ActionFigureGenerateRequest(BaseModel):
    name: str = Field("Action Figure", min_length=1, max_length=120)
    mc_resolution: int = Field(128, ge=64, le=256)
    remove_bg: bool = True
    foreground_ratio: float = Field(0.85, ge=0.5, le=1.0)
    sample_preset: Optional[str] = None
