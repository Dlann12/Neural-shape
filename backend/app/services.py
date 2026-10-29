"""Background workers: 3D reconstruction jobs and ML training experiments."""
import json
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.config import Settings
from backend.app.db import Database
from ml.baseline import reconstruct_classical_multiview
from ml.camera import CAMERA_VIEWS, NUM_VIEWS
from ml.dataset import CATEGORIES, SyntheticMultiViewDataset, generate_3d_shape, render_multiview_object
from ml.inference import OccupancyPredictor
from ml.mesh_utils import export_obj, export_ply, occupancy_to_mesh
from ml.metrics import evaluate_reconstruction
from ml.model import ModelConfig
from ml.train import SuiteConfig, run_experiment_suite, train_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class ModelRegistry:
    """Caches loaded checkpoints and reloads them when the file on disk changes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._cache: Dict[Path, Tuple[float, OccupancyPredictor]] = {}

    def get(self, path: Optional[Path] = None) -> OccupancyPredictor:
        path = Path(path or self.settings.default_checkpoint)
        with self._lock:
            if not path.exists():
                if path != self.settings.default_checkpoint:
                    raise FileNotFoundError(path)
                self._bootstrap(path)
            mtime = path.stat().st_mtime
            cached = self._cache.get(path)
            if cached is None or cached[0] != mtime:
                self._cache[path] = (mtime, OccupancyPredictor(path))
            return self._cache[path][1]

    def _bootstrap(self, path: Path) -> None:
        """Train a small model so the API works out of the box. Run scripts/train.py for a proper one."""
        s = self.settings
        common = dict(grid_size=s.grid_size, img_size=s.img_size, seed_offset=100)
        train_ds = SyntheticMultiViewDataset(num_objects=s.bootstrap_train_objects, split="train", **common)
        val_ds = SyntheticMultiViewDataset(num_objects=max(4, s.bootstrap_train_objects // 5), split="val", **common)
        train_model(train_ds, val_ds, path, epochs=s.bootstrap_epochs, batch_size=8,
                    model_config=ModelConfig(grid_size=s.grid_size), experiment_name="bootstrap")


def save_synthetic_views(proj_dir: Path, category: str, seed: int, settings: Settings) -> None:
    """Seed a project with 8 calibrated synthetic views + ground truth for immediate testing."""
    rng = np.random.default_rng(seed)
    occ_gt, color = generate_3d_shape(category, grid_size=settings.grid_size, rng=rng)
    imgs, _ = render_multiview_object(occ_gt, color, img_size=settings.img_size, rng=rng)
    np.save(proj_dir / "gt_occupancy.npy", occ_gt)
    np.save(proj_dir / "images_tensor.npy", imgs)
    for v_idx in range(imgs.shape[0]):
        rgb_u8 = np.clip(np.transpose(imgs[v_idx], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(proj_dir / "images" / f"view_{v_idx:02d}.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def to_model_input(bgr: np.ndarray, img_size: int) -> np.ndarray:
    """Center-crop to a square (keeps aspect ratio), resize and convert to CHW float RGB in [0, 1]."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = bgr[y0:y0 + s, x0:x0 + s]
    rgb = cv2.cvtColor(cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def reset_project_inputs(proj_dir: Path) -> None:
    """Uploaded photos replace any synthetic views, so the synthetic ground truth no longer applies."""
    img_dir = proj_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    for name in ("gt_occupancy.npy", "images_tensor.npy"):
        (proj_dir / name).unlink(missing_ok=True)


def _mesh_payload(occ: np.ndarray, threshold: float, out_dir: Path, prefix: str) -> Dict[str, Any]:
    v, f, n = occupancy_to_mesh(occ, threshold=threshold)
    return {
        "obj_export": export_obj(out_dir / f"{prefix}_mesh.obj", v, f, n),
        "ply_export": export_ply(out_dir / f"{prefix}_mesh.ply", v, f, n),
        "vertices": v.tolist(),
        "faces": f.tolist(),
    }


def run_reconstruction_job(settings: Settings, db: Database, registry: ModelRegistry,
                           job_id: str, project_id: str, method: str) -> None:
    t0 = time.perf_counter()
    proj_dir = settings.projects_dir / project_id
    try:
        db.execute("UPDATE jobs SET status = 'Processing', progress = 10 WHERE id = ?", (job_id,))
        db.execute("UPDATE projects SET status = 'Processing', updated_at = ? WHERE id = ?", (time.time(), project_id))

        imgs = np.load(proj_dir / "images_tensor.npy")
        notes: List[str] = []
        if imgs.shape[0] > NUM_VIEWS:
            notes.append(f"{imgs.shape[0]} images supplied; only the first {NUM_VIEWS} are used.")
            imgs = imgs[:NUM_VIEWS]
        views = list(range(imgs.shape[0]))
        gt_path = proj_dir / "gt_occupancy.npy"
        gt_occ = np.load(gt_path) if gt_path.exists() else None

        results: Dict[str, Any] = {
            "view_mapping": [CAMERA_VIEWS[v][2] for v in views],
            "notes": notes,
        }
        metrics: Dict[str, Any] = {}

        if method in ("baseline", "both"):
            occ, ms, meta = reconstruct_classical_multiview(imgs, grid_size=settings.grid_size, view_indices=views)
            results["baseline"] = {"metadata": meta, **_mesh_payload(occ, 0.5, proj_dir, "baseline")}
            if gt_occ is not None:
                metrics["baseline"] = evaluate_reconstruction(occ, gt_occ, ms)
            db.execute("UPDATE jobs SET progress = 40 WHERE id = ?", (job_id,))

        if method in ("ml_multiview", "ml_singleview", "both"):
            predictor = registry.get()
            use_views = [0] if method == "ml_singleview" else views
            occ, ms = predictor.predict(imgs[use_views], view_indices=use_views)
            results["ml"] = {
                "checkpoint_epoch": predictor.meta.get("epoch"),
                "checkpoint_val_iou": predictor.meta.get("val_iou"),
                "threshold": predictor.threshold,
                "views_used": len(use_views),
                **_mesh_payload(occ, predictor.threshold, proj_dir, "ml"),
            }
            if gt_occ is not None:
                metrics["ml"] = evaluate_reconstruction(occ, gt_occ, ms, threshold=predictor.threshold)

        dur = (time.perf_counter() - t0) * 1000.0
        db.execute("UPDATE jobs SET status = 'Completed', progress = 100, duration_ms = ? WHERE id = ?", (dur, job_id))
        db.execute(
            "UPDATE projects SET status = 'Completed', results_json = ?, metrics_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(results), json.dumps(metrics), time.time(), project_id),
        )
    except Exception as exc:
        traceback.print_exc()
        db.execute("UPDATE jobs SET status = 'Failed', error = ? WHERE id = ?", (f"{type(exc).__name__}: {exc}", job_id))
        db.execute("UPDATE projects SET status = 'Failed', updated_at = ? WHERE id = ?", (time.time(), project_id))


def run_experiment_job(settings: Settings, db: Database, exp_id: str, cfg: Dict[str, Any]) -> None:
    out_dir = settings.checkpoints_dir / "experiments" / exp_id

    def on_progress(info: Dict[str, Any]) -> None:
        db.execute("UPDATE ml_experiments SET progress_json = ? WHERE id = ?", (json.dumps(info), exp_id))

    try:
        suite = SuiteConfig(
            name="exp", num_train_objects=cfg["num_train_objects"], num_val_objects=cfg["num_val_objects"],
            num_test_objects=cfg["num_test_objects"], grid_size=settings.grid_size, img_size=settings.img_size,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], seed=cfg["seed"],
            train_augmented=cfg["train_augmented"],
        )
        report = run_experiment_suite(suite, out_dir, progress_callback=on_progress)
        report["promoted"] = cfg.get("promote_to_default", False) and _maybe_promote(
            settings, out_dir / "exp_multiview.pt", report["training"]["standard"]["best_val_iou"]
        )
        db.execute("UPDATE ml_experiments SET status = 'Completed', result_json = ? WHERE id = ?", (json.dumps(report), exp_id))
    except Exception as exc:
        traceback.print_exc()
        db.execute("UPDATE ml_experiments SET status = 'Failed', result_json = ? WHERE id = ?",
                   (json.dumps({"error": f"{type(exc).__name__}: {exc}"}), exp_id))


def _maybe_promote(settings: Settings, candidate: Path, candidate_val_iou: float) -> bool:
    current = settings.default_checkpoint
    if current.exists():
        try:
            current_iou = float(OccupancyPredictor(current).meta.get("val_iou", -1))
        except Exception:
            current_iou = -1.0
        if candidate_val_iou <= current_iou:
            return False
    tmp = current.with_suffix(".promote.tmp")
    shutil.copy2(candidate, tmp)
    tmp.replace(current)
    return True


def resolve_category(category: str) -> str:
    return category if category in CATEGORIES else "chair"


def run_action_figure_job(
    settings: Settings,
    db: Database,
    figure_id: str,
    image_path: Path,
    name: str,
    mc_resolution: int = 128,
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
) -> None:
    from ml.action_figure import generate_action_figure_3d

    fig_dir = settings.projects_dir / "action_figures" / figure_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(pct: int, msg: str) -> None:
        db.execute(
            "UPDATE action_figures SET progress = ?, progress_msg = ? WHERE id = ?",
            (pct, msg, figure_id),
        )

    try:
        on_progress(5, "Memulai pipeline pembuatan 3D...")
        res = generate_action_figure_3d(
            input_image=image_path,
            output_dir=fig_dir,
            base_name="model",
            mc_resolution=mc_resolution,
            remove_bg=remove_bg,
            foreground_ratio=foreground_ratio,
            progress_callback=on_progress,
        )
        now = time.time()
        db.execute(
            """
            UPDATE action_figures
            SET status = 'Completed', progress = 100, progress_msg = 'Selesai!',
                vertex_count = ?, face_count = ?, duration_seconds = ?,
                metadata_json = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                res["vertex_count"],
                res["face_count"],
                res["timing"]["total_seconds"],
                json.dumps(res),
                now,
                figure_id,
            ),
        )
    except Exception as exc:
        traceback.print_exc()
        db.execute(
            "UPDATE action_figures SET status = 'Failed', error = ?, progress_msg = 'Gagal membuat 3D' WHERE id = ?",
            (f"{type(exc).__name__}: {exc}", figure_id),
        )
