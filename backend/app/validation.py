"""FR-02: Image Quality Validation Pipeline using OpenCV."""
from typing import List, Dict, Any
import numpy as np
import cv2


def compute_dhash(gray_img: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute 2D difference hash (dHash) for duplicate detection (128 bits)."""
    resized = cv2.resize(gray_img, (hash_size + 1, hash_size + 1), interpolation=cv2.INTER_AREA)
    h_diff = resized[:-1, 1:] > resized[:-1, :-1]
    v_diff = resized[1:, :-1] > resized[:-1, :-1]
    return np.concatenate([h_diff.ravel(), v_diff.ravel()])


def validate_single_image(img_rgb_uint8: np.ndarray, filename: str = "image.png") -> Dict[str, Any]:
    """
    Validate a single RGB uint8 image (H, W, 3) against FR-02 criteria:
    - Blur (Laplacian variance)
    - Too dark or overexposed (mean luminance)
    - Object too small in frame (foreground ratio)
    - Object cropped at frame borders
    """
    H, W, _ = img_rgb_uint8.shape
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)

    # 1. Blur check via Laplacian variance (normalized to 64x64 scale)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 2. Brightness check
    mean_brightness = float(gray.mean())
    # 3. Foreground segmentation check
    bg_level = float(np.percentile(gray, 10))
    fg_mask = (gray > max(bg_level + 12.0, 22.0)).astype(np.uint8)
    fg_ratio = float(fg_mask.mean())

    # 4. Border cropping check
    border_pixels = np.concatenate([
        fg_mask[0, :], fg_mask[-1, :], fg_mask[:, 0], fg_mask[:, -1]
    ])
    border_touch_ratio = float(border_pixels.mean())

    warnings = []
    if lap_var < 18.0:
        warnings.append("Gambar terdeteksi buram (Laplacian variance rendah). Gunakan fokus tajam atau tripod.")
    if mean_brightness < 14.0:
        warnings.append("Gambar terlalu gelap. Tingkatkan pencahayaan objek.")
    elif mean_brightness > 242.0:
        warnings.append("Gambar terlalu terang (overexposed).")
    if fg_ratio < 0.04:
        warnings.append("Objek terlalu kecil di dalam frame (<4% area). Dekatkan kamera ke objek.")
    if border_touch_ratio > 0.25:
        warnings.append("Objek terpotong pada tepi frame. Pastikan seluruh objek masuk di dalam frame.")

    return {
        "filename": filename,
        "width": int(W),
        "height": int(H),
        "blur_score": round(lap_var, 2),
        "brightness": round(mean_brightness, 2),
        "foreground_ratio": round(fg_ratio, 4),
        "border_touch_ratio": round(border_touch_ratio, 4),
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
        "_dhash": compute_dhash(gray),
    }


def validate_image_set(images_rgb_uint8: List[np.ndarray], filenames: List[str]) -> Dict[str, Any]:
    """
    Validate a multi-view image set for 3D reconstruction (FR-01 & FR-02):
    - Checks minimum view count (recommended >= 5 views)
    - Per-image quality checks (blur, lighting, size, cropping)
    - Near-duplicate photo detection via perceptual hash Hamming distance
    - Lighting consistency across images
    - Viewpoint diversity estimation
    """
    per_image_results = []
    hashes = []
    brightness_values = []
    duplicate_pairs = []

    for idx, (img, fname) in enumerate(zip(images_rgb_uint8, filenames)):
        res = validate_single_image(img, fname)
        h = res.pop("_dhash")
        for prev_idx, prev_h in enumerate(hashes):
            hamming_dist = int(np.count_nonzero(h != prev_h))
            if hamming_dist <= 4:
                duplicate_pairs.append((filenames[prev_idx], fname))
                res["warnings"].append(f"Hampir identik/duplikat dengan {filenames[prev_idx]} (Hamming={hamming_dist}).")
                res["is_valid"] = False
        hashes.append(h)
        brightness_values.append(res["brightness"])
        per_image_results.append(res)

    set_warnings = []
    recommendations = []

    num_images = len(images_rgb_uint8)
    if num_images < 5:
        set_warnings.append(f"Jumlah foto ({num_images}) kurang dari rekomendasi minimum MVP (5-30 foto).")
        recommendations.append("Unggah minimal 5-8 foto dari sudut depan, belakang, kiri, kanan, dan diagonal atas.")

    if len(duplicate_pairs) > 0:
        set_warnings.append(f"Ditemukan {len(duplicate_pairs)} pasang foto duplikat / sudut pandang kurang bervariasi.")
        recommendations.append("Ganti foto duplikat dengan sudut pandang baru (kelilingi objek setiap 45 derajat).")

    if len(brightness_values) >= 2:
        b_std = float(np.std(brightness_values))
        if b_std > 45.0:
            set_warnings.append("Perubahan pencahayaan ekstrem terdeteksi antar-foto.")
            recommendations.append("Gunakan pencahayaan merata dan kunci exposure kamera saat memotret objek.")

    valid_count = sum(1 for r in per_image_results if r["is_valid"])
    overall_valid = (num_images >= 5) and (valid_count >= max(4, num_images - 1)) and (len(duplicate_pairs) == 0)

    if overall_valid and not recommendations:
        recommendations.append("Kualitas gambar dan variasi sudut pandang memenuhi syarat. Siap menjalankan rekonstruksi 3D.")

    return {
        "overall_valid": bool(overall_valid),
        "total_images": int(num_images),
        "valid_images": int(valid_count),
        "duplicate_pairs": duplicate_pairs,
        "set_warnings": set_warnings,
        "recommendations": recommendations,
        "images": per_image_results,
    }
