"""Reproducible training and benchmarking pipeline for NeuralShape (Experiments A, B, C, D)."""
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.baseline import reconstruct_classical_multiview
from ml.camera import NUM_VIEWS
from ml.dataset import SyntheticMultiViewDataset
from ml.inference import OccupancyPredictor
from ml.metrics import compute_volumetric_iou, evaluate_reconstruction
from ml.model import ModelConfig, MultiViewOccupancyNet, count_parameters, occupancy_loss

CHECKPOINT_FORMAT_VERSION = 2
THRESHOLD_CANDIDATES = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]

ProgressCallback = Callable[[Dict[str, Any]], None]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_view_subset(rng: random.Random, single_view_prob: float, full_view_prob: float = 0.5) -> List[int]:
    """View dropout: train on a random camera subset so one model serves 1..8 view inference."""
    r = rng.random()
    if r < single_view_prob:
        return [rng.randrange(NUM_VIEWS)]
    if r < single_view_prob + full_view_prob:
        return list(range(NUM_VIEWS))
    k = rng.randint(2, NUM_VIEWS - 1)
    return sorted(rng.sample(range(NUM_VIEWS), k))


def _batch_ious(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> List[float]:
    pred = probs.detach().cpu().numpy()
    gt = targets.detach().cpu().numpy()
    return [compute_volumetric_iou(pred[b], gt[b], threshold) for b in range(len(pred))]


@torch.no_grad()
def _predict_dataset(model: MultiViewOccupancyNet, loader: DataLoader, device: torch.device, view_indices: List[int]):
    probs_all, gts_all, losses = [], [], []
    for batch in loader:
        imgs = batch["images"][:, view_indices].to(device)
        targets = batch["occupancy"].to(device)
        logits, probs = model(imgs, view_indices=view_indices)
        losses.append(float(occupancy_loss(logits, targets).item()))
        probs_all.append(probs.cpu().numpy())
        gts_all.append(targets.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(gts_all), float(np.mean(losses))


def calibrate_threshold(probs: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    """Pick the occupancy threshold that maximises mean validation IoU."""
    scores = {t: float(np.mean([compute_volumetric_iou(p, g, t) for p, g in zip(probs, gts)])) for t in THRESHOLD_CANDIDATES}
    best = max(scores, key=scores.get)
    return {"threshold": float(best), "val_iou_at_threshold": round(scores[best], 4), "val_iou_at_0.5": round(scores[0.5], 4)}


def train_model(
    train_ds: SyntheticMultiViewDataset,
    val_ds: SyntheticMultiViewDataset,
    checkpoint_path: Path,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-3,
    seed: int = 42,
    experiment_name: str = "exp_c_multiview_ml",
    model_config: Optional[ModelConfig] = None,
    weight_decay: float = 1e-4,
    single_view_prob: float = 0.2,
    grad_clip: float = 1.0,
    patience: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train MultiViewOccupancyNet, keep the best checkpoint by 8-view validation IoU, then calibrate
    the occupancy threshold on the validation set and store it in the checkpoint."""
    set_seed(seed)
    view_rng = random.Random(seed)
    device = get_device()
    checkpoint_path = Path(checkpoint_path)
    model_config = model_config or ModelConfig(grid_size=train_ds.grid_size)
    if model_config.grid_size != train_ds.grid_size:
        raise ValueError("model_config.grid_size must match dataset grid_size")

    model = MultiViewOccupancyNet.from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Linear warmup for the first epoch, then cosine decay to 5% of the peak LR (per optimizer step).
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(steps_per_epoch, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_cfg = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "weight_decay": weight_decay,
        "single_view_prob": single_view_prob, "grad_clip": grad_clip, "patience": patience,
        "num_train_objects": len(train_ds), "num_val_objects": len(val_ds),
        "photometric_aug": bool(getattr(train_ds, "photometric_aug", False)),
    }

    history: List[Dict[str, Any]] = []
    best_val_iou, best_epoch, epochs_no_improve = -1.0, 0, 0
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_ious = [], []
        for batch in train_loader:
            views = sample_view_subset(view_rng, single_view_prob)
            imgs = batch["images"][:, views].to(device)
            targets = batch["occupancy"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, probs = model(imgs, view_indices=views)
            loss = occupancy_loss(logits, targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(float(loss.item()))
            train_ious.extend(_batch_ious(probs, targets))

        model.eval()
        val_probs, val_gts, val_loss = _predict_dataset(model, val_loader, device, list(range(NUM_VIEWS)))
        val_iou = float(np.mean([compute_volumetric_iou(p, g) for p, g in zip(val_probs, val_gts)]))
        val_probs_1v, _, _ = _predict_dataset(model, val_loader, device, [0])
        val_iou_1v = float(np.mean([compute_volumetric_iou(p, g) for p, g in zip(val_probs_1v, val_gts)]))

        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "train_iou": round(float(np.mean(train_ious)), 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(val_iou, 4),
            "val_iou_1view": round(val_iou_1v, 4),
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
        }
        history.append(record)
        if verbose:
            print(json.dumps(record), flush=True)

        if val_iou > best_val_iou:
            best_val_iou, best_epoch, epochs_no_improve = val_iou, epoch, 0
            _save_checkpoint(checkpoint_path, model, model_config, train_cfg, experiment_name, epoch, val_iou, val_loss)
        else:
            epochs_no_improve += 1

        if progress_callback:
            progress_callback({"epoch": epoch, "epochs": epochs, **record})
        if patience is not None and epochs_no_improve >= patience:
            break

    # Threshold calibration on the best weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_probs, val_gts, _ = _predict_dataset(model, val_loader, device, list(range(NUM_VIEWS)))
    calib = calibrate_threshold(val_probs, val_gts)
    ckpt["threshold"] = calib["threshold"]
    ckpt["calibration"] = calib
    torch.save(ckpt, checkpoint_path)

    total_sec = time.perf_counter() - t_start
    result = {
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "best_val_iou": round(best_val_iou, 4),
        "calibration": calib,
        "num_parameters": count_parameters(model),
        "training_time_sec": round(total_sec, 2),
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config.to_dict(),
        "train_config": train_cfg,
        "history": history,
    }
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _save_checkpoint(path: Path, model, model_config: ModelConfig, train_cfg: Dict[str, Any], name: str,
                     epoch: int, val_iou: float, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "experiment_name": name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_cfg,
        "val_iou": round(val_iou, 4),
        "val_loss": round(val_loss, 4),
        "threshold": 0.5,
    }, tmp)
    tmp.replace(path)  # atomic swap so concurrent readers never see a half-written file


def _summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key: str, nd: int = 4) -> float:
        return round(float(np.mean([r[key] for r in records])), nd)

    by_cat: Dict[str, List[float]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["iou"])
    return {
        "num_test_objects": len(records),
        "mean_iou": mean("iou"),
        "mean_chamfer_distance": mean("chamfer_distance"),
        "mean_f_score": mean("f_score"),
        "mean_inference_time_ms": mean("inference_time_ms", 2),
        "success_rate": round(float(np.mean([1.0 if r["valid_output"] else 0.0 for r in records])) * 100.0, 1),
        "iou_per_category": {c: round(float(np.mean(v)), 4) for c, v in sorted(by_cat.items())},
        "per_object": records,
    }


EXP_A = "Experiment A (Classical Multi-View Baseline)"
EXP_B = "Experiment B (Single-View ML - 1 View)"
EXP_C = "Experiment C (Multi-View ML - 8 Views)"
EXP_D = "Experiment D (Multi-View ML + Augmentation)"


def evaluate_all_experiments(
    test_ds: SyntheticMultiViewDataset,
    ckpt_standard: Path,
    ckpt_aug: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate the PRD Section 7 benchmark on the held-out test set:
    - A: classical space carving (8 views)
    - B: ML model with a single front view
    - C: ML model with all 8 views
    - D: ML model trained with photometric augmentation, 8 views (skipped when `ckpt_aug` is None)
    Each ML model uses the occupancy threshold calibrated on its validation set.
    """
    device = get_device()
    std = OccupancyPredictor(ckpt_standard, device=device)
    aug = OccupancyPredictor(ckpt_aug, device=device) if ckpt_aug else None
    all_views = list(range(NUM_VIEWS))

    results: Dict[str, List[Dict[str, Any]]] = {EXP_A: [], EXP_B: [], EXP_C: []}
    if aug:
        results[EXP_D] = []

    for i in range(len(test_ds)):
        item = test_ds[i]
        imgs = item["images"].numpy()
        gt = item["occupancy"].numpy()
        tag = {"object_id": item["object_id"], "category": item["category"]}

        pred, ms, _ = reconstruct_classical_multiview(imgs, grid_size=test_ds.grid_size)
        results[EXP_A].append({**evaluate_reconstruction(pred, gt, ms), **tag})

        pred, ms = std.predict(imgs[:1], view_indices=[0])
        results[EXP_B].append({**evaluate_reconstruction(pred, gt, ms, threshold=std.threshold), **tag})

        pred, ms = std.predict(imgs, view_indices=all_views)
        results[EXP_C].append({**evaluate_reconstruction(pred, gt, ms, threshold=std.threshold), **tag})

        if aug:
            pred, ms = aug.predict(imgs, view_indices=all_views)
            results[EXP_D].append({**evaluate_reconstruction(pred, gt, ms, threshold=aug.threshold), **tag})

    return {name: _summarise(recs) for name, recs in results.items()}


@dataclass
class SuiteConfig:
    """Datasets + training settings for a full benchmark run (used by the CLI and the API)."""
    name: str = "neuralshape"
    num_train_objects: int = 150
    num_val_objects: int = 25
    num_test_objects: int = 30
    grid_size: int = 24
    img_size: int = 64
    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-3
    seed: int = 42
    train_augmented: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_experiment_suite(
    cfg: SuiteConfig,
    out_dir: Path,
    progress_callback: Optional[ProgressCallback] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build splits, train the standard (+ augmented) model and evaluate experiments A-D."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.model.grid_size = cfg.grid_size
    common = dict(grid_size=cfg.grid_size, img_size=cfg.img_size, seed_offset=cfg.seed)
    train_ds = SyntheticMultiViewDataset(num_objects=cfg.num_train_objects, split="train", **common)
    val_ds = SyntheticMultiViewDataset(num_objects=cfg.num_val_objects, split="val", **common)
    test_ds = SyntheticMultiViewDataset(num_objects=cfg.num_test_objects, split="test", **common)

    train_kwargs = dict(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
                        model_config=cfg.model, verbose=verbose)

    def stage_cb(stage: str):
        if not progress_callback:
            return None
        return lambda info: progress_callback({"stage": stage, **info})

    ckpt_std = out_dir / f"{cfg.name}_multiview.pt"
    training = {"standard": train_model(train_ds, val_ds, ckpt_std, experiment_name=f"{cfg.name} (standard)",
                                        progress_callback=stage_cb("standard"), **train_kwargs)}
    ckpt_aug = None
    if cfg.train_augmented:
        ckpt_aug = out_dir / f"{cfg.name}_multiview_aug.pt"
        train_ds.photometric_aug = True
        training["augmented"] = train_model(train_ds, val_ds, ckpt_aug, experiment_name=f"{cfg.name} (augmented)",
                                            progress_callback=stage_cb("augmented"), **train_kwargs)

    benchmarks = evaluate_all_experiments(test_ds, ckpt_std, ckpt_aug)
    report = {"config": cfg.to_dict(), "training": training, "benchmarks": benchmarks}
    (out_dir / f"{cfg.name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
