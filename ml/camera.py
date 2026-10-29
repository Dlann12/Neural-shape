"""Calibrated orthographic camera rig shared by the renderer, the visual-hull baseline and the network.

Keeping the projection in one place guarantees that the synthetic renderer, the classical baseline and
the learned unprojection all agree on where a 3D voxel lands in each image.
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 8 calibrated viewpoints around the object: (azimuth_deg, elevation_deg, name)
CAMERA_VIEWS: List[Tuple[float, float, str]] = [
    (0.0, 15.0, "front"),
    (45.0, 20.0, "front_right"),
    (90.0, 15.0, "right"),
    (135.0, 20.0, "back_right"),
    (180.0, 15.0, "back"),
    (225.0, 20.0, "back_left"),
    (270.0, 15.0, "left"),
    (315.0, 55.0, "top_diagonal"),
]
NUM_VIEWS = len(CAMERA_VIEWS)

# Half-extent of the orthographic image plane in world units. The object lives in [-1, 1]^3, so a value
# slightly above sqrt(1.5) keeps rotated shapes inside the frame.
ORTHO_SCALE = 1.28


def get_rotation_matrix(azim_deg: float, elev_deg: float) -> np.ndarray:
    """3x3 world-to-camera rotation: rotate around Y by azimuth, then around X by elevation."""
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    ry = np.array([
        [math.cos(az), 0.0, math.sin(az)],
        [0.0, 1.0, 0.0],
        [-math.sin(az), 0.0, math.cos(az)],
    ], dtype=np.float32)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(el), -math.sin(el)],
        [0.0, math.sin(el), math.cos(el)],
    ], dtype=np.float32)
    return rx @ ry


def project_points(points: np.ndarray, view_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points (N, 3) into view `view_idx`.

    Returns normalized image coordinates (u, v) in [-1, 1] (grid_sample convention, v pointing down)
    and camera-space depth, where larger depth means closer to the camera.
    """
    az, el, _ = CAMERA_VIEWS[view_idx]
    cam = points @ get_rotation_matrix(az, el).T
    return cam[:, 0] / ORTHO_SCALE, -cam[:, 1] / ORTHO_SCALE, cam[:, 2]


def norm_to_pixel(coord_norm: np.ndarray, size: int) -> np.ndarray:
    """Map normalized [-1, 1] coordinates to pixel centers (matches grid_sample align_corners=True)."""
    return (coord_norm * 0.5 + 0.5) * (size - 1)


def pose_vector(view_idx: int) -> np.ndarray:
    """Compact pose encoding [sin az, cos az, sin el, cos el] for one view."""
    az, el, _ = CAMERA_VIEWS[view_idx]
    az_r, el_r = math.radians(az), math.radians(el)
    return np.array([math.sin(az_r), math.cos(az_r), math.sin(el_r), math.cos(el_r)], dtype=np.float32)


def resolve_view_indices(num_images: int, view_indices: Optional[Sequence[int]] = None) -> List[int]:
    """Validate the mapping from image slots to calibrated cameras."""
    if view_indices is None:
        view_indices = list(range(num_images))
    view_indices = [int(v) for v in view_indices]
    if len(view_indices) != num_images:
        raise ValueError(f"Got {num_images} images but {len(view_indices)} view indices")
    if any(v < 0 or v >= NUM_VIEWS for v in view_indices):
        raise ValueError(f"View indices must be in [0, {NUM_VIEWS - 1}], got {view_indices}")
    return view_indices


def voxel_centers(grid_size: int) -> np.ndarray:
    """Voxel center coordinates (G^3, 3) in [-1, 1]^3 using 'ij' (X, Y, Z) ordering."""
    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
