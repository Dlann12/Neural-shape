"""Runtime settings for the NeuralShape backend (overridable through environment variables)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Settings:
    storage_dir: Path = field(default_factory=lambda: Path(os.environ.get("NEURALSHAPE_STORAGE_DIR", ROOT_DIR / "storage")))
    frontend_dir: Path = ROOT_DIR / "frontend"
    max_workers: int = field(default_factory=lambda: _env_int("NEURALSHAPE_MAX_WORKERS", 2))
    grid_size: int = 24
    img_size: int = 64
    # Used only when no trained checkpoint exists yet and a reconstruction needs the ML model.
    bootstrap_epochs: int = field(default_factory=lambda: _env_int("NEURALSHAPE_BOOTSTRAP_EPOCHS", 10))
    bootstrap_train_objects: int = field(default_factory=lambda: _env_int("NEURALSHAPE_BOOTSTRAP_OBJECTS", 60))

    @property
    def projects_dir(self) -> Path:
        return self.storage_dir / "projects"

    @property
    def checkpoints_dir(self) -> Path:
        return self.storage_dir / "checkpoints"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "neuralshape.db"

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def sample_images_dir(self) -> Path:
        return ROOT_DIR / "sample_images"

    @property
    def default_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_model_multiview.pt"

    def ensure_dirs(self) -> None:
        for d in (self.projects_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)
