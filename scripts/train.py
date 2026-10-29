"""Train the NeuralShape model(s) and run the A-D benchmark.

Examples:
    python -m scripts.train                      # default preset
    python -m scripts.train --preset quick       # ~2 min smoke run
    python -m scripts.train --epochs 30 --train-objects 300 --no-augmented
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.model import ModelConfig  # noqa: E402
from ml.train import SuiteConfig, run_experiment_suite  # noqa: E402

PRESETS = {
    "quick": dict(num_train_objects=30, num_val_objects=10, num_test_objects=10, epochs=4, train_augmented=False),
    "default": dict(num_train_objects=150, num_val_objects=25, num_test_objects=30, epochs=20, train_augmented=True),
    "full": dict(num_train_objects=400, num_val_objects=50, num_test_objects=50, epochs=35, train_augmented=True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="default")
    p.add_argument("--name", default="neuralshape")
    p.add_argument("--out", type=Path, default=ROOT / "storage" / "checkpoints")
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-objects", type=int, dest="num_train_objects")
    p.add_argument("--val-objects", type=int, dest="num_val_objects")
    p.add_argument("--test-objects", type=int, dest="num_test_objects")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--feat-channels", type=int, default=32)
    p.add_argument("--hidden-3d", type=int, default=32)
    p.add_argument("--augmented", dest="train_augmented", action="store_true", default=None,
                   help="also train the photometric-augmentation model (Experiment D)")
    p.add_argument("--no-augmented", dest="train_augmented", action="store_false")
    p.add_argument("--no-promote", action="store_true",
                   help="do not copy the trained model to storage/checkpoints/best_model_multiview.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k in SuiteConfig.__dataclass_fields__ and v is not None and k not in ("name",)}
    cfg = SuiteConfig(name=args.name, model=ModelConfig(feat_channels=args.feat_channels, hidden_3d=args.hidden_3d),
                      **{**PRESETS[args.preset], **overrides})
    print(f"Config: {json.dumps(cfg.to_dict())}", flush=True)

    report = run_experiment_suite(cfg, args.out, verbose=True)

    print("\n=== Benchmark (held-out test set) ===")
    print(f"{'Experiment':48s} {'IoU':>7s} {'Chamfer':>8s} {'F-score':>8s} {'ms':>7s}")
    for name, s in report["benchmarks"].items():
        print(f"{name:48s} {s['mean_iou']:7.4f} {s['mean_chamfer_distance']:8.4f} {s['mean_f_score']:8.4f} {s['mean_inference_time_ms']:7.1f}")
    print(f"\nReport: {args.out / f'{cfg.name}_report.json'}")

    if not args.no_promote:
        src = args.out / f"{cfg.name}_multiview.pt"
        dst = args.out / "best_model_multiview.pt"
        shutil.copy2(src, dst)
        print(f"API checkpoint updated: {dst}")


if __name__ == "__main__":
    main()
