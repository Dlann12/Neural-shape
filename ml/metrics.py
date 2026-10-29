"""3D Reconstruction Quantitative Evaluation Metrics (IoU, Chamfer Distance, F-Score)."""
from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial import cKDTree
from ml.mesh_utils import occupancy_to_mesh


def compute_volumetric_iou(pred_prob: np.ndarray, gt_occ: np.ndarray, threshold: float = 0.5) -> float:
    """Compute 3D Volumetric Intersection over Union (IoU)."""
    pred_bin = (pred_prob >= threshold).astype(bool)
    gt_bin = (gt_occ >= 0.5).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def sample_surface_points(vertices: np.ndarray, faces: np.ndarray, num_points: int = 1024, seed: int = 42) -> np.ndarray:
    """Uniformly sample 3D points on triangle mesh surface for Chamfer & F-Score evaluation."""
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(faces) == 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(vertices), size=min(num_points, len(vertices)), replace=False)
        return vertices[idx].astype(np.float32)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = areas.sum()
    if total_area <= 1e-10:
        return vertices[:min(num_points, len(vertices))].astype(np.float32)

    probs = areas / total_area
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=num_points, replace=True, p=probs)
    r1 = np.sqrt(rng.random(num_points, dtype=np.float32))[:, None]
    r2 = rng.random(num_points, dtype=np.float32)[:, None]
    pts = (1.0 - r1) * v0[face_indices] + r1 * (1.0 - r2) * v1[face_indices] + (r1 * r2) * v2[face_indices]
    return pts.astype(np.float32)


def compute_chamfer_and_fscore(
    pred_verts: np.ndarray,
    pred_faces: np.ndarray,
    gt_verts: np.ndarray,
    gt_faces: np.ndarray,
    tau: float = 0.08,
    num_samples: int = 1024,
) -> Tuple[float, float]:
    """
    Compute the symmetric Chamfer distance (mean Euclidean nearest-neighbour distance, averaged over both
    directions) and F-Score@tau between predicted and ground-truth surfaces in normalized [-1, 1]^3 space.
    """
    if len(pred_verts) == 0 or len(gt_verts) == 0:
        return 2.0, 0.0

    pts_pred = sample_surface_points(pred_verts, pred_faces, num_points=num_samples)
    pts_gt = sample_surface_points(gt_verts, gt_faces, num_points=num_samples)
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 2.0, 0.0

    tree_pred = cKDTree(pts_pred)
    tree_gt = cKDTree(pts_gt)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred, k=1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt, k=1)

    chamfer_l2 = float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) * 0.5
    precision = float(np.mean(dist_pred_to_gt < tau))
    recall = float(np.mean(dist_gt_to_pred < tau))
    if precision + recall <= 1e-8:
        fscore = 0.0
    else:
        fscore = float(2.0 * precision * recall / (precision + recall))

    return chamfer_l2, fscore


def evaluate_reconstruction(
    pred_prob: np.ndarray,
    gt_occ: np.ndarray,
    inference_time_ms: float,
    threshold: float = 0.5,
    tau: float = 0.08,
) -> Dict[str, Any]:
    """Full geometric evaluation of a single 3D reconstruction against ground truth."""
    iou = compute_volumetric_iou(pred_prob, gt_occ, threshold=threshold)
    pred_v, pred_f, _ = occupancy_to_mesh(pred_prob, threshold=threshold, adaptive_threshold=False)
    gt_v, gt_f, _ = occupancy_to_mesh(gt_occ, threshold=0.5, adaptive_threshold=False)

    valid_output = bool(len(pred_v) >= 4 and len(pred_f) >= 4)
    if valid_output:
        cd, fscore = compute_chamfer_and_fscore(pred_v, pred_f, gt_v, gt_f, tau=tau)
    else:
        cd, fscore = 2.0, 0.0

    return {
        "iou": round(float(iou), 4),
        "chamfer_distance": round(float(cd), 4),
        "f_score": round(float(fscore), 4),
        "inference_time_ms": round(float(inference_time_ms), 2),
        "valid_output": valid_output,
        "vertex_count": int(len(pred_v)),
        "face_count": int(len(pred_f)),
    }
