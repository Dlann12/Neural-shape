"""Classical multi-view 3D reconstruction baseline (space carving / visual hull)."""
import shutil
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from ml.camera import norm_to_pixel, project_points, resolve_view_indices, voxel_centers


def extract_foreground_silhouette(img_chw: np.ndarray, bg_tolerance: float = 0.035) -> np.ndarray:
    """
    Binary foreground silhouette (H, W) from an RGB image (3, H, W) in [0, 1].

    The background colour is estimated from the image border (median), and a pixel is foreground if any
    channel deviates from it by more than `bg_tolerance`. This keeps dark, shadowed surfaces that a fixed
    luminance threshold would discard, and adapts to exposure changes.
    """
    img = np.asarray(img_chw, dtype=np.float32)
    border = np.concatenate([img[:, 0, :], img[:, -1, :], img[:, :, 0], img[:, :, -1]], axis=1)
    bg = np.median(border, axis=1)[:, None, None]
    diff = np.max(np.abs(img - bg), axis=0)
    mask = (diff > bg_tolerance).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 127).astype(np.float32)


def reconstruct_classical_multiview(
    images_vchw: np.ndarray,
    grid_size: int = 24,
    carve_tolerance: float = 0.85,
    view_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Space carving: project every voxel into each calibrated view and keep voxels that fall inside the
    foreground silhouette in (almost) all views. `carve_tolerance` controls how many views may disagree.
    Returns (occupancy_prob_grid, inference_time_ms, metadata).
    """
    t0 = time.perf_counter()
    num_views, _, H, W = images_vchw.shape
    view_indices = resolve_view_indices(num_views, view_indices)
    coords = voxel_centers(grid_size)

    consistency = np.zeros(len(coords), dtype=np.float32)
    for slot, cam in enumerate(view_indices):
        sil = extract_foreground_silhouette(images_vchw[slot])
        u_n, v_n, _ = project_points(coords, cam)
        u = np.clip(np.round(norm_to_pixel(u_n, W)).astype(np.int32), 0, W - 1)
        v = np.clip(np.round(norm_to_pixel(v_n, H)).astype(np.int32), 0, H - 1)
        consistency += sil[v, u]

    frac_consistent = consistency / max(float(num_views), 1.0)
    # Soft score centred so that marching cubes at 0.5 extracts the hull at the requested tolerance
    occ_prob = np.clip((frac_consistent - (carve_tolerance - 0.25)) / 0.35, 0.0, 1.0)
    occ_grid = occ_prob.reshape((grid_size, grid_size, grid_size)).astype(np.float32)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    meta = {
        "method": "Classical Multi-View Space Carving (Visual Hull)",
        "colmap_available": shutil.which("colmap") is not None,
        "num_views_used": int(num_views),
        "view_indices": view_indices,
        "carve_tolerance": float(carve_tolerance),
        "note": (
            "Visual Hull reconstructs the outer silhouette intersection across calibrated views, "
            "serving as the deterministic classical geometry baseline (cannot carve self-occluded interior concavities)."
        ),
    }
    return occ_grid, elapsed_ms, meta
