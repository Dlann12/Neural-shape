"""Checkpoint loading and inference helpers shared by the benchmark, CLI scripts and the API."""
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ml.camera import resolve_view_indices
from ml.model import ModelConfig, MultiViewOccupancyNet


class IncompatibleCheckpointError(RuntimeError):
    pass


def load_checkpoint(path: Union[str, Path], device: Optional[torch.device] = None) -> Tuple[MultiViewOccupancyNet, Dict[str, Any]]:
    """Rebuild the model from the architecture stored in the checkpoint and load its weights."""
    device = device or torch.device("cpu")
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if "model_config" not in ckpt:
        raise IncompatibleCheckpointError(
            f"{path} was produced by the old architecture (no 'model_config'). Retrain with scripts/train.py."
        )
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = MultiViewOccupancyNet.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


class OccupancyPredictor:
    """Wraps a trained checkpoint: numpy images in, occupancy probabilities out."""

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.meta = load_checkpoint(self.checkpoint_path, self.device)

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))

    @property
    def grid_size(self) -> int:
        return int(self.meta["model_config"]["grid_size"])

    @torch.no_grad()
    def predict(self, images_vchw: np.ndarray, view_indices: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, float]:
        """images_vchw: (V, 3, H, W) float in [0, 1]. Returns (probabilities (G, G, G), inference ms)."""
        views = resolve_view_indices(images_vchw.shape[0], view_indices)
        t0 = time.perf_counter()
        imgs = torch.from_numpy(np.ascontiguousarray(images_vchw, dtype=np.float32)).unsqueeze(0).to(self.device)
        _, probs = self.model(imgs, view_indices=views)
        out = probs[0].cpu().numpy()
        return out, (time.perf_counter() - t0) * 1000.0
