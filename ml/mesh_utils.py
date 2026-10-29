"""3D Mesh Extraction, Post-Processing, and Export (.OBJ / .PLY)."""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth per-vertex normals from triangle faces."""
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros_like(vertices)
    normals = np.zeros_like(vertices, dtype=np.float64)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    face_normals = face_normals / lengths
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    v_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    v_lengths = np.maximum(v_lengths, 1e-8)
    return (normals / v_lengths).astype(np.float32)


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 2, lam: float = 0.35) -> np.ndarray:
    """Apply lightweight Laplacian smoothing to reduce voxel staircase artifacts."""
    if len(vertices) == 0 or len(faces) == 0 or iterations <= 0:
        return vertices
    verts = vertices.copy().astype(np.float64)
    n_verts = len(verts)
    adj_sum = np.zeros_like(verts)
    adj_count = np.zeros((n_verts, 1), dtype=np.float64)

    edges = np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
        faces[:, [2, 0]], faces[:, [0, 2]],
    ], axis=0)

    for _ in range(iterations):
        adj_sum.fill(0.0)
        adj_count.fill(0.0)
        np.add.at(adj_sum, edges[:, 0], verts[edges[:, 1]])
        np.add.at(adj_count, edges[:, 0], 1.0)
        mask = adj_count[:, 0] > 0
        verts[mask] = (1.0 - lam) * verts[mask] + lam * (adj_sum[mask] / adj_count[mask])
    return verts.astype(np.float32)


def occupancy_to_mesh(
    occupancy_grid: np.ndarray,
    threshold: float = 0.5,
    smooth_sigma: float = 0.25,
    laplacian_iters: int = 1,
    adaptive_threshold: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a 3D occupancy probability grid (D, H, W) into a triangle mesh (vertices, faces, normals)
    normalized to [-1, 1]^3 coordinate space.

    With `adaptive_threshold=True` a soft prediction that never crosses `threshold` is still meshed at the
    mid-range level (useful for previews). Disable it for evaluation.
    """
    grid = occupancy_grid.astype(np.float32)
    # Pad boundary with zeros so marching cubes closes surfaces at grid boundaries
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=0.0)
    if smooth_sigma > 0:
        padded = gaussian_filter(padded, sigma=smooth_sigma)

    max_val = float(padded.max())
    min_val = float(padded.min())
    if max_val <= threshold or min_val >= threshold:
        # Fallback threshold if prediction is soft but non-empty
        if adaptive_threshold and max_val > 0.15 and min_val < max_val:
            threshold = 0.5 * (max_val + min_val)
        else:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )

    try:
        verts, faces, normals, _ = marching_cubes(padded, level=threshold)
        # Map from padded grid indices [1 .. D] back to [-1, 1]^3
        dims = np.array(grid.shape, dtype=np.float32)
        verts = ((verts - 1.0) / np.maximum(dims - 1.0, 1.0)) * 2.0 - 1.0
        verts = verts.astype(np.float32)
        faces = faces.astype(np.int32)
        if laplacian_iters > 0:
            verts = laplacian_smooth(verts, faces, iterations=laplacian_iters)
        normals = compute_vertex_normals(verts, faces)
        return verts, faces, normals
    except Exception:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
        )


def export_obj(filepath: Path, vertices: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Export mesh to Wavefront .OBJ format (geometry + vertex normals, no fake texture claims)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if normals is None or len(normals) != len(vertices):
        normals = compute_vertex_normals(vertices, faces)

    lines = [
        "# NeuralShape 3D Reconstruction Export (OBJ)",
        "# Representation: Triangle Mesh (Geometry + Vertex Normals, Untextured)",
        f"# Vertices: {len(vertices)} | Faces: {len(faces)}",
    ]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for n in normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for f in faces:
        i0, i1, i2 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
        lines.append(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}")

    content = "\n".join(lines) + "\n"
    filepath.write_text(content, encoding="utf-8")
    return {
        "format": "obj",
        "path": str(filepath),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "file_size_bytes": int(filepath.stat().st_size),
        "has_texture": False,
    }


def export_ply(
    filepath: Path,
    vertices: np.ndarray,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Export mesh or point cloud to ASCII Stanford .PLY format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    has_faces = faces is not None and len(faces) > 0
    if normals is None and has_faces:
        normals = compute_vertex_normals(vertices, faces)
    has_normals = normals is not None and len(normals) == len(vertices)

    header = [
        "ply",
        "format ascii 1.0",
        "comment NeuralShape 3D Reconstruction Export",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    if has_faces:
        header.extend([
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
        ])
    header.append("end_header")

    body = []
    for idx, v in enumerate(vertices):
        if has_normals:
            n = normals[idx]
            body.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        else:
            body.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if has_faces:
        for f in faces:
            body.append(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")

    filepath.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return {
        "format": "ply",
        "path": str(filepath),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)) if has_faces else 0,
        "file_size_bytes": int(filepath.stat().st_size),
        "has_texture": False,
    }
