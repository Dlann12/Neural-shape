"""Anime & Object Single-Image to 3D Action Figure Generator.

Uses feed-forward transformer architecture (TripoSR) with background segmentation
and marching-cubes surface extraction to convert single 2D images into textured,
colored 3D meshes (OBJ, PLY, GLB).
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rembg
import torch
import trimesh
from PIL import Image

from ml.triposr import TSR
from ml.triposr.utils import remove_background, resize_foreground

logger = logging.getLogger("neuralshape.action_figure")

_MODEL: Optional[TSR] = None
_REMBG_SESSION: Optional[Any] = None


def get_action_figure_model(
    device: str = "cpu",
    chunk_size: int = 4096,
    repo_id: str = "stabilityai/TripoSR",
) -> TSR:
    """Singleton getter for the TripoSR Image-to-3D model."""
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading TripoSR model from {repo_id} onto {device}...")
        t0 = time.time()
        model = TSR.from_pretrained(
            repo_id,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
        model.eval()
        _MODEL = model
        logger.info(f"TripoSR initialized in {time.time() - t0:.2f}s.")
    return _MODEL


def get_rembg_session(model_name: Optional[str] = None) -> Any:
    """Singleton getter for background removal session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        if model_name:
            _REMBG_SESSION = rembg.new_session(model_name=model_name)
        else:
            _REMBG_SESSION = rembg.new_session()
    return _REMBG_SESSION


def prepare_input_image(
    input_image: Union[str, Path, Image.Image],
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
    session: Optional[Any] = None,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """Segment, center, and blend input image onto neutral gray background."""
    if isinstance(input_image, (str, Path)):
        img = Image.open(input_image)
    else:
        img = input_image

    cutout_img: Optional[Image.Image] = None

    if remove_bg:
        sess = session or get_rembg_session()
        cutout = remove_background(img, sess)
        cutout_img = cutout
        fg = resize_foreground(cutout, foreground_ratio)
        np_fg = np.array(fg).astype(np.float32) / 255.0
        # Blend foreground over 50% neutral gray for TripoSR conditioning
        blended = np_fg[:, :, :3] * np_fg[:, :, 3:4] + (1.0 - np_fg[:, :, 3:4]) * 0.5
        ready_img = Image.fromarray((blended * 255.0).astype(np.uint8))
    else:
        rgb = img.convert("RGB")
        ready_img = rgb

    return ready_img, cutout_img


def normalize_mesh_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center mesh at origin, set bottom contact at y=0, and scale to unit bounds."""
    verts = mesh.vertices.copy()
    if len(verts) == 0:
        return mesh

    # Center x and z at origin
    min_b, max_b = verts.min(axis=0), verts.max(axis=0)
    center = (min_b + max_b) / 2.0
    verts[:, 0] -= center[0]
    verts[:, 2] -= center[2]

    # Align lowest point (feet/base) to y=0
    verts[:, 1] -= min_b[1]

    # Scale so max bounding dimension is ~1.0
    extent = (max_b - min_b).max()
    if extent > 1e-6:
        verts /= extent

    mesh.vertices = verts
    return mesh


def generate_action_figure_3d(
    input_image: Union[str, Path, Image.Image],
    output_dir: Union[str, Path],
    base_name: str = "action_figure",
    mc_resolution: int = 128,
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
    device: str = "cpu",
    chunk_size: int = 4096,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Union[str, int, float, dict]]:
    """End-to-end generation of a colored 3D action figure mesh from a 2D image."""
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def report(pct: int, msg: str):
        logger.info(f"[{pct}%] {msg}")
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    # Step 1: Preprocessing & Background removal
    report(10, "Menghapus latar belakang & segmentasi objek...")
    ready_img, cutout_img = prepare_input_image(
        input_image,
        remove_bg=remove_bg,
        foreground_ratio=foreground_ratio,
    )

    ready_path = out_path / f"{base_name}_input.png"
    ready_img.save(ready_path)

    cutout_path: Optional[Path] = None
    if cutout_img is not None:
        cutout_path = out_path / f"{base_name}_cutout.png"
        cutout_img.save(cutout_path)

    # Step 2: Initialize & run model
    report(30, "Memuat model Image-to-3D Transformer...")
    model = get_action_figure_model(device=device, chunk_size=chunk_size)

    report(45, "Mengekstrak representasi 3D Triplane dari gambar...")
    t_tf = time.time()
    with torch.no_grad():
        scene_codes = model([ready_img], device=device)
    tf_duration = time.time() - t_tf

    # Step 3: Marching Cubes Isosurface & Vertex Color extraction
    report(65, f"Ekstraksi geometri 3D (Marching Cubes resolution={mc_resolution})...")
    t_mc = time.time()
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
    raw_mesh = meshes[0]
    mc_duration = time.time() - t_mc

    report(85, "Normalisasi orientasi 3D & pewarnaan tekstur...")
    mesh = normalize_mesh_orientation(raw_mesh)
    # Ensure vertex normals are computed so GLTF/GLB exports have surface normals
    try:
        _ = mesh.vertex_normals
    except Exception as e:
        logger.warning(f"Error computing vertex normals: {e}")

    # Step 4: Export meshes (OBJ, PLY, GLB)
    report(92, "Mengekspor file 3D (OBJ, PLY, GLB)...")
    obj_path = out_path / f"{base_name}.obj"
    ply_path = out_path / f"{base_name}.ply"
    glb_path = out_path / f"{base_name}.glb"

    try:
        mesh.export(str(obj_path))
    except Exception as e:
        logger.warning(f"Error exporting OBJ: {e}")

    try:
        mesh.export(str(ply_path))
    except Exception as e:
        logger.warning(f"Error exporting PLY: {e}")

    try:
        mesh.export(str(glb_path))
    except Exception as e:
        logger.warning(f"Error exporting GLB: {e}")

    total_duration = time.time() - t_start
    report(100, f"Selesai! 3D Action Figure berhasil dibuat dalam {total_duration:.1f} detik.")

    return {
        "status": "completed",
        "name": base_name,
        "obj_path": str(obj_path),
        "ply_path": str(ply_path),
        "glb_path": str(glb_path),
        "ready_image_path": str(ready_path),
        "cutout_image_path": str(cutout_path) if cutout_path else None,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "mc_resolution": mc_resolution,
        "timing": {
            "total_seconds": round(total_duration, 2),
            "transformer_seconds": round(tf_duration, 2),
            "marching_cubes_seconds": round(mc_duration, 2),
        },
    }
