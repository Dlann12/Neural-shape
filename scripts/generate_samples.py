import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ml.camera import CAMERA_VIEWS
from ml.dataset import generate_3d_shape, render_multiview_object

SAMPLE_DIR = ROOT / "sample_images"


def generate_sample_set(category: str, folder_name: str, seed: int = 42, img_size: int = 256):
    out_dir = SAMPLE_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    occ, color = generate_3d_shape(category, grid_size=32, rng=rng)
    imgs, _ = render_multiview_object(occ, color, img_size=img_size, rng=rng)

    saved_paths = []
    for idx, (az, el, name) in enumerate(CAMERA_VIEWS):
        chw = imgs[idx]
        rgb_u8 = np.clip(np.transpose(chw, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

        fname = f"{idx + 1:02d}_{name}_{int(az)}deg.png"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), bgr_u8)
        saved_paths.append(fpath)

    print(f"Generated {len(saved_paths)} images in {out_dir}")
    return saved_paths


def main():
    print("=== Generating NeuralShape Test Image Sets ===")
    generate_sample_set("chair", "1_kursi_chair", seed=42)
    generate_sample_set("hollow_vessel", "2_mangkuk_vessel", seed=105)
    generate_sample_set("arch_monument", "3_monumen_arch", seed=202)
    print("\nDone! Sample image sets are ready in folder: sample_images/")


if __name__ == "__main__":
    main()
