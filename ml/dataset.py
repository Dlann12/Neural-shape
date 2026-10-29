"""Synthetic multi-view 3D dataset: procedural shapes, z-buffered renderer and a PyTorch Dataset.

Splits are generated from independent seed streams per (split, base_seed, index), so no object can
appear in more than one split.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.camera import (  # noqa: F401  (re-exported for backward compatibility)
    CAMERA_VIEWS,
    NUM_VIEWS,
    get_rotation_matrix,
    norm_to_pixel,
    pose_vector,
    project_points,
)

CATEGORIES = ["chair", "table", "arch_monument", "hollow_vessel", "stepped_sculpture"]
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
BACKGROUND_LEVEL = 0.06


def generate_3d_shape(category: str, grid_size: int = 24, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3D binary occupancy grid (D, H, W) in [-1, 1]^3 and an RGB albedo color (3,).
    Coordinates: X (left-right), Y (up-down), Z (front-back).
    """
    if rng is None:
        rng = np.random.default_rng()

    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    occ = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    if category == "chair":
        w = rng.uniform(0.42, 0.65)
        d = rng.uniform(0.42, 0.65)
        seat_y = rng.uniform(-0.15, 0.05)
        seat_t = 0.12
        back_h = rng.uniform(0.55, 0.82)
        leg_t = rng.uniform(0.11, 0.16)
        seat = (np.abs(X) <= w) & (np.abs(Z) <= d) & (Y >= seat_y - seat_t) & (Y <= seat_y + seat_t)
        back = (np.abs(X) <= w) & (Z >= -d) & (Z <= -d + 2.0 * leg_t) & (Y >= seat_y) & (Y <= back_h)
        leg_x = (np.abs(X) >= w - 2.0 * leg_t) & (np.abs(X) <= w)
        leg_z = (np.abs(Z) >= d - 2.0 * leg_t) & (np.abs(Z) <= d)
        legs = leg_x & leg_z & (Y >= -0.82) & (Y <= seat_y)
        if rng.random() > 0.35:
            arms = leg_x & (np.abs(Z) <= d * 0.85) & (Y >= seat_y) & (Y <= seat_y + 0.32)
            occ[(seat | back | legs | arms)] = 1.0
        else:
            occ[(seat | back | legs)] = 1.0
        color = np.array([0.82, 0.46, 0.24], dtype=np.float32)

    elif category == "table":
        w = rng.uniform(0.55, 0.82)
        d = rng.uniform(0.45, 0.78)
        top_y = rng.uniform(0.25, 0.55)
        top_t = 0.12
        leg_t = rng.uniform(0.11, 0.16)
        top = (np.abs(X) <= w) & (np.abs(Z) <= d) & (Y >= top_y - top_t) & (Y <= top_y + top_t)
        if rng.random() > 0.45:
            leg_x = (np.abs(X) >= w - 2.0 * leg_t) & (np.abs(X) <= w)
            leg_z = (np.abs(Z) >= d - 2.0 * leg_t) & (np.abs(Z) <= d)
            legs = leg_x & leg_z & (Y >= -0.80) & (Y <= top_y)
            occ[(top | legs)] = 1.0
        else:
            ped = (X**2 + Z**2 <= (min(w, d) * 0.38)**2) & (Y >= -0.78) & (Y <= top_y)
            base = (X**2 + Z**2 <= (min(w, d) * 0.72)**2) & (Y >= -0.82) & (Y <= -0.62)
            occ[(top | ped | base)] = 1.0
        # Recessed tray on top (concave feature only visible through shading / top view)
        recess = (np.abs(X) <= w * 0.6) & (np.abs(Z) <= d * 0.6) & (Y > top_y + 0.02)
        occ[recess] = 0.0
        color = np.array([0.28, 0.65, 0.78], dtype=np.float32)

    elif category == "arch_monument":
        w = rng.uniform(0.55, 0.80)
        d = rng.uniform(0.32, 0.58)
        h = rng.uniform(0.60, 0.84)
        body = (np.abs(X) <= w) & (np.abs(Z) <= d) & (Y >= -0.82) & (Y <= h)
        arch_w = w * rng.uniform(0.42, 0.58)
        arch_h = rng.uniform(0.05, 0.30)
        tunnel_rect = (np.abs(X) <= arch_w) & (Y >= -0.85) & (Y <= arch_h)
        tunnel_top = ((X / max(arch_w, 0.1))**2 + ((Y - arch_h) / 0.28)**2 <= 1.0)
        tunnel = (tunnel_rect | tunnel_top) & (np.abs(Z) <= d + 0.1)
        alcove = (np.abs(Z) <= d * 0.55) & (np.abs(Y - 0.2) <= 0.25) & (np.abs(X) >= w - 0.16)
        occ[body & (~tunnel) & (~alcove)] = 1.0
        color = np.array([0.76, 0.72, 0.62], dtype=np.float32)

    elif category == "hollow_vessel":
        rx = rng.uniform(0.55, 0.82)
        rz = rng.uniform(0.55, 0.82)
        h_top = rng.uniform(0.45, 0.78)
        h_bot = -0.78
        wall = rng.uniform(0.20, 0.28)
        if rng.random() > 0.5:
            dist_xz = np.sqrt((X / rx)**2 + (Z / rz)**2)
            outer = (dist_xz <= 1.0) & (Y >= h_bot) & (Y <= h_top)
            inner = (dist_xz <= (1.0 - wall)) & (Y >= h_bot + 0.22) & (Y <= h_top + 0.1)
        else:
            # Box crate with deep hollow interior (classic visual-hull failure case)
            outer = (np.abs(X) <= rx) & (np.abs(Z) <= rz) & (Y >= h_bot) & (Y <= h_top)
            inner = (np.abs(X) <= rx - wall) & (np.abs(Z) <= rz - wall) & (Y >= h_bot + 0.22) & (Y <= h_top + 0.1)
        occ[outer & (~inner)] = 1.0
        color = np.array([0.85, 0.38, 0.45], dtype=np.float32)

    elif category == "stepped_sculpture":
        w1 = rng.uniform(0.62, 0.84)
        d1 = rng.uniform(0.62, 0.84)
        tier1 = (np.abs(X) <= w1) & (np.abs(Z) <= d1) & (Y >= -0.82) & (Y <= -0.20)
        tier2 = (np.abs(X) <= w1 * 0.70) & (np.abs(Z) <= d1 * 0.70) & (Y >= -0.20) & (Y <= 0.32)
        tier3 = (np.abs(X) <= w1 * 0.42) & (np.abs(Z) <= d1 * 0.42) & (Y >= 0.32) & (Y <= 0.78)
        courtyard = (np.abs(X) <= w1 * 0.35) & (np.abs(Z) <= d1 * 0.35) & (Y >= 0.05) & (Y <= 0.85)
        side_recess = (np.abs(X) <= 0.22) & (Z >= 0.20) & (Y >= -0.50) & (Y <= 0.25)
        occ[(tier1 | tier2 | tier3) & (~courtyard) & (~side_recess)] = 1.0
        color = np.array([0.48, 0.76, 0.44], dtype=np.float32)

    else:
        raise ValueError(f"Unknown category '{category}'. Expected one of {CATEGORIES}")

    color = np.clip(color + rng.uniform(-0.08, 0.08, size=3).astype(np.float32), 0.15, 0.95)
    return occ, color


def _surface_normals(occ_grid: np.ndarray) -> np.ndarray:
    """Outward normals (D, H, W, 3) from the occupancy gradient (occupancy grows toward the inside)."""
    grads = np.stack(np.gradient(occ_grid), axis=-1)
    return -grads / (np.linalg.norm(grads, axis=-1, keepdims=True) + 1e-6)


def render_multiview_object(
    occ_grid: np.ndarray,
    base_color: np.ndarray,
    img_size: int = 64,
    photometric_aug: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render the 8 calibrated views (8, 3, H, W) in [0, 1] plus pose vectors (8, 4).

    Voxels are splatted with a per-pixel z-buffer, so the closest surface always wins. Lambertian shading
    (headlight-style) plus a depth cue encodes concavities in the RGB appearance.
    """
    if rng is None:
        rng = np.random.default_rng()

    G = occ_grid.shape[0]
    poses = np.stack([pose_vector(i) for i in range(NUM_VIEWS)], axis=0)
    idx = np.argwhere(occ_grid > 0.5)
    if len(idx) == 0:
        return np.full((NUM_VIEWS, 3, img_size, img_size), BACKGROUND_LEVEL, dtype=np.float32), poses

    coords = (idx.astype(np.float32) / (G - 1.0)) * 2.0 - 1.0
    normals = _surface_normals(occ_grid)[idx[:, 0], idx[:, 1], idx[:, 2]]

    light_dir = np.array([0.35, 0.65, 0.67], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    radius = max(1, img_size // 28)
    offsets = np.array([(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)], dtype=np.int32)

    images = []
    for view_idx, (az_deg, el_deg, _) in enumerate(CAMERA_VIEWS):
        u_n, v_n, depth = project_points(coords, view_idx)
        cam_norms = normals @ get_rotation_matrix(az_deg, el_deg).T

        diffuse = np.clip(cam_norms @ light_dir, 0.15, 1.0)
        depth_factor = np.clip((depth + 1.2) / 2.4, 0.20, 1.0)
        intensity = 0.55 * diffuse + 0.45 * depth_factor

        u_px = np.round(norm_to_pixel(u_n, img_size)).astype(np.int32)
        v_px = np.round(norm_to_pixel(v_n, img_size)).astype(np.int32)

        # Expand every voxel into a small square splat, then resolve visibility with a z-buffer.
        uu = (u_px[None, :] + offsets[:, 1:2]).ravel()
        vv = (v_px[None, :] + offsets[:, 0:1]).ravel()
        dd = np.tile(depth, len(offsets))
        ii = np.tile(intensity, len(offsets))
        inside = (uu >= 0) & (uu < img_size) & (vv >= 0) & (vv < img_size)
        pix, dd, ii = (vv * img_size + uu)[inside], dd[inside], ii[inside]

        zbuf = np.full(img_size * img_size, -np.inf, dtype=np.float32)
        np.maximum.at(zbuf, pix, dd)
        front = dd >= zbuf[pix]

        canvas = np.full((img_size * img_size, 3), BACKGROUND_LEVEL, dtype=np.float32)
        canvas[pix[front]] = base_color[None, :] * ii[front, None]
        canvas = canvas.reshape(img_size, img_size, 3)

        if photometric_aug:
            canvas = _photometric_jitter(canvas, rng)
        images.append(np.transpose(canvas, (2, 0, 1)))

    return np.stack(images, axis=0).astype(np.float32), poses


def _photometric_jitter(img_hwc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness / contrast / colour-gain / sensor-noise jitter on an (H, W, 3) image in [0, 1]."""
    brightness = rng.uniform(-0.06, 0.06)
    contrast = rng.uniform(0.85, 1.15)
    gain = rng.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
    noise = rng.normal(0.0, 0.015, size=img_hwc.shape).astype(np.float32)
    out = (img_hwc - 0.5) * contrast + 0.5 + brightness
    return np.clip(out * gain + noise, 0.0, 1.0)


def object_seed(split: str, base_seed: int, index: int) -> np.random.SeedSequence:
    """Independent random stream per object; different splits can never share a stream."""
    if split not in SPLIT_IDS:
        raise ValueError(f"split must be one of {list(SPLIT_IDS)}, got '{split}'")
    return np.random.SeedSequence([SPLIT_IDS[split], int(base_seed), int(index)])


class SyntheticMultiViewDataset(Dataset):
    """
    Multi-view 3D shape reconstruction dataset.

    Each item: 8 calibrated RGB views (8, 3, H, W), pose vectors (8, 4) and the ground-truth occupancy
    (G, G, G). When `photometric_aug=True` and split == "train", a fresh photometric jitter is applied
    per view every time an item is fetched.
    """

    def __init__(
        self,
        num_objects: int = 80,
        grid_size: int = 24,
        img_size: int = 64,
        split: str = "train",
        seed_offset: int = 1000,
        photometric_aug: bool = False,
        categories: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.img_size = img_size
        self.photometric_aug = photometric_aug
        self.categories = list(categories) if categories else list(CATEGORIES)
        self._aug_rng = np.random.default_rng(object_seed(split, seed_offset, 10**6))
        self.records: List[Dict[str, Any]] = []

        for i in range(num_objects):
            rng = np.random.default_rng(object_seed(split, seed_offset, i))
            category = self.categories[i % len(self.categories)]
            occ, color = generate_3d_shape(category, grid_size=grid_size, rng=rng)
            imgs, poses = render_multiview_object(occ, color, img_size=img_size, rng=rng)
            self.records.append({
                "object_id": f"{split}_{category}_{seed_offset}_{i:03d}",
                "category": category,
                "images": imgs,
                "poses": poses,
                "occupancy": occ,
                "color": color,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        imgs = rec["images"]
        if self.split == "train" and self.photometric_aug:
            imgs = np.stack([
                np.transpose(_photometric_jitter(np.transpose(v, (1, 2, 0)), self._aug_rng), (2, 0, 1))
                for v in imgs
            ]).astype(np.float32)
        return {
            "object_id": rec["object_id"],
            "category": rec["category"],
            "images": torch.from_numpy(np.ascontiguousarray(imgs)),
            "poses": torch.from_numpy(rec["poses"]),
            "occupancy": torch.from_numpy(rec["occupancy"]),
        }
