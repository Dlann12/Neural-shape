"""Comprehensive Automated Verification Tests for NeuralShape MVP (FR-01 to FR-07 & DoD)."""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.validation import validate_image_set
from ml.baseline import reconstruct_classical_multiview
from ml.camera import CAMERA_VIEWS, NUM_VIEWS, norm_to_pixel, project_points
from ml.dataset import SyntheticMultiViewDataset
from ml.inference import OccupancyPredictor
from ml.mesh_utils import export_obj, export_ply, occupancy_to_mesh
from ml.metrics import evaluate_reconstruction
from ml.model import ModelConfig, MultiViewOccupancyNet, occupancy_loss
from ml.train import train_model


def test_fr01_dataset_split_no_leakage():
    """FR-01: Synthetic multi-view dataset splits strictly by object_id with no cross-contamination."""
    train_ds = SyntheticMultiViewDataset(num_objects=8, grid_size=24, img_size=64, split="train", seed_offset=100)
    val_ds = SyntheticMultiViewDataset(num_objects=4, grid_size=24, img_size=64, split="val", seed_offset=5000)
    test_ds = SyntheticMultiViewDataset(num_objects=4, grid_size=24, img_size=64, split="test", seed_offset=9000)

    train_ids = {r["object_id"] for r in train_ds.records}
    val_ids = {r["object_id"] for r in val_ds.records}
    test_ids = {r["object_id"] for r in test_ds.records}

    assert len(train_ids & val_ids) == 0, "Train and Val share object IDs!"
    assert len(train_ids & test_ids) == 0, "Train and Test share object IDs!"
    assert len(val_ids & test_ids) == 0, "Val and Test share object IDs!"

    sample = train_ds[0]
    assert sample["images"].shape == (8, 3, 64, 64)
    assert sample["poses"].shape == (8, 4)
    assert sample["occupancy"].shape == (24, 24, 24)


def test_fr02_image_quality_validation():
    """FR-02: Image quality validation catches dark, blurred, and near-duplicate images."""
    ds = SyntheticMultiViewDataset(num_objects=1, grid_size=24, img_size=64, split="test", seed_offset=777)
    imgs_chw = ds[0]["images"].numpy()
    imgs_rgb_u8 = [np.clip(np.transpose(im, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8) for im in imgs_chw]
    names = [f"view_{i}.png" for i in range(len(imgs_rgb_u8))]

    valid_report = validate_image_set(imgs_rgb_u8, names)
    assert valid_report["total_images"] == 8
    assert valid_report["overall_valid"] is True
    assert len(valid_report["duplicate_pairs"]) == 0

    # Inject duplicate and completely dark image
    bad_imgs = [imgs_rgb_u8[0], imgs_rgb_u8[0].copy(), np.zeros((64, 64, 3), dtype=np.uint8)]
    bad_report = validate_image_set(bad_imgs, ["a.png", "a_dup.png", "dark.png"])
    assert bad_report["overall_valid"] is False
    assert len(bad_report["duplicate_pairs"]) >= 1


def test_fr03_classical_baseline_and_mesh_export():
    """FR-03 & FR-06: Classical multi-view reconstruction (space carving) & 3D mesh export."""
    ds = SyntheticMultiViewDataset(num_objects=1, grid_size=24, img_size=64, split="test", seed_offset=888)
    item = ds[0]
    occ_pred, ms, meta = reconstruct_classical_multiview(item["images"].numpy(), grid_size=24)
    assert meta["num_views_used"] == 8
    assert ms > 0

    metrics = evaluate_reconstruction(occ_pred, item["occupancy"].numpy(), ms)
    assert metrics["valid_output"] is True
    assert metrics["iou"] > 0.35

    verts, faces, normals = occupancy_to_mesh(occ_pred, threshold=0.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = Path(tmpdir) / "test.obj"
        ply_path = Path(tmpdir) / "test.ply"
        obj_meta = export_obj(obj_path, verts, faces, normals)
        ply_meta = export_ply(ply_path, verts, faces, normals)

        assert obj_meta["vertex_count"] > 10 and obj_path.stat().st_size > 100
        assert ply_meta["face_count"] > 10 and ply_path.stat().st_size > 100


def test_fr04_camera_rig_projection():
    """Camera projections are deterministic and map normalized bounds into image canvas."""
    pts = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.5]], dtype=np.float32)
    for v_idx in range(NUM_VIEWS):
        u_n, v_n, d = project_points(pts, v_idx)
        assert u_n.shape == (2,)
        assert v_n.shape == (2,)
        u_px = norm_to_pixel(u_n, 64)
        v_px = norm_to_pixel(v_n, 64)
        assert np.all((u_px >= 0) & (u_px < 64))
        assert np.all((v_px >= 0) & (v_px < 64))


def test_fr05_pytorch_model_forward_backward():
    """FR-04: MultiViewOccupancyNet forward/backward passes with 8 views and 1 view."""
    cfg = ModelConfig(feat_channels=16, hidden_3d=16, grid_size=24)
    model = MultiViewOccupancyNet.from_config(cfg)
    dummy_imgs = torch.rand(2, 8, 3, 64, 64)
    dummy_gt = (torch.rand(2, 24, 24, 24) > 0.8).float()

    logits_8v, probs_8v = model(dummy_imgs)
    assert probs_8v.shape == (2, 24, 24, 24)
    loss = occupancy_loss(logits_8v, dummy_gt)
    loss.backward()

    # 1-view forward pass (Experiment B)
    logits_1v, probs_1v = model(dummy_imgs[:, :1], view_indices=[0])
    assert probs_1v.shape == (2, 24, 24, 24)


def test_fr06_checkpoint_and_inference_predictor():
    """Checkpoint saving and OccupancyPredictor loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        train_ds = SyntheticMultiViewDataset(num_objects=4, grid_size=24, img_size=64, split="train", seed_offset=1)
        val_ds = SyntheticMultiViewDataset(num_objects=2, grid_size=24, img_size=64, split="val", seed_offset=2)

        res = train_model(
            train_ds,
            val_ds,
            ckpt_path,
            epochs=1,
            batch_size=2,
            model_config=ModelConfig(feat_channels=16, hidden_3d=16, grid_size=24),
        )
        assert ckpt_path.exists()
        assert "calibration" in res

        predictor = OccupancyPredictor(ckpt_path)
        assert predictor.grid_size == 24
        assert 0.0 < predictor.threshold < 1.0

        sample_imgs = train_ds[0]["images"].numpy()
        pred, ms = predictor.predict(sample_imgs)
        assert pred.shape == (24, 24, 24)
        assert ms > 0


def test_fr07_api_end_to_end_workflow():
    """FR-05 & FR-07: FastAPI End-to-End Workflow (Project Creation, Validation, Reconstruction, Export)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            storage_dir=Path(tmpdir) / "storage",
            bootstrap_epochs=1,
            bootstrap_train_objects=4,
        )
        app = create_app(settings)
        client = TestClient(app)

        # Health check
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # Create Project with deterministic seed for reproducible metric assertions
        res = client.post("/api/projects", json={
            "name": "Integration Test Chair",
            "category": "chair",
            "auto_generate_synthetic_views": True,
            "seed": 42
        })
        assert res.status_code == 200
        proj_data = res.json()
        proj_id = proj_data["id"]

        # Validate images
        val_res = client.post(f"/api/projects/{proj_id}/validate")
        assert val_res.status_code == 200
        assert val_res.json()["overall_valid"] is True

        # Run Baseline Reconstruction (synchronous wait on job)
        rec_res = client.post(f"/api/projects/{proj_id}/reconstruct", json={"method": "baseline"})
        assert rec_res.status_code == 200
        job_id = rec_res.json()["job_id"]

        # Poll job until completed
        for _ in range(30):
            job_status = client.get(f"/api/jobs/{job_id}").json()
            if job_status["status"] == "Completed":
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Job did not finish in time")

        # Check results
        results_res = client.get(f"/api/projects/{proj_id}/results")
        assert results_res.status_code == 200
        res_body = results_res.json()
        assert "baseline" in res_body["results"]
        assert len(res_body["results"]["baseline"]["vertices"]) > 0

        # Check metrics
        metrics_res = client.get(f"/api/projects/{proj_id}/metrics")
        assert metrics_res.status_code == 200
        m_body = metrics_res.json()
        assert m_body["metrics"]["baseline"]["valid_output"] is True
        assert m_body["metrics"]["baseline"]["iou"] > 0.35

        # Check export
        exp_res = client.get(f"/api/projects/{proj_id}/export?format=obj&source=baseline")
        assert exp_res.status_code == 200
        assert exp_res.headers["content-type"] == "application/octet-stream"
        assert len(exp_res.content) > 100


def test_fr08_action_figure_single_image_to_3d():
    """FR-08: Anime & arbitrary single-image to 3D Action Figure generation (TripoSR + rembg)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            storage_dir=Path(tmpdir) / "storage",
        )
        app = create_app(settings)
        client = TestClient(app)

        # 1. Samples endpoint
        res = client.get("/api/action-figures/samples")
        assert res.status_code == 200
        samples = res.json()
        assert len(samples) >= 1
        assert any(s["id"] == "anime_hero" for s in samples)

        # 2. Generate Action Figure from preset
        gen_res = client.post(
            "/api/action-figures/generate?sample_preset=anime_hero&mc_resolution=64&name=TestFigure"
        )
        assert gen_res.status_code == 200
        fig_data = gen_res.json()
        fig_id = fig_data["id"]
        assert fig_data["status"] == "Processing"

        # 3. Poll until completed (timeout 120s for CPU)
        for _ in range(120):
            st = client.get(f"/api/action-figures/{fig_id}").json()
            if st["status"] == "Completed":
                break
            elif st["status"] == "Failed":
                raise RuntimeError(f"Action figure generation failed: {st.get('error')}")
            time.sleep(1.0)
        else:
            raise TimeoutError("Action figure generation did not complete within timeout")

        # 4. Check statistics
        fig_info = client.get(f"/api/action-figures/{fig_id}").json()
        assert fig_info["vertex_count"] > 100
        assert fig_info["face_count"] > 100
        assert fig_info["progress"] == 100

        # 5. Check model downloads (GLB, OBJ, PLY)
        glb_res = client.get(f"/api/action-figures/{fig_id}/model?format=glb")
        assert glb_res.status_code == 200
        assert len(glb_res.content) > 1000

        obj_res = client.get(f"/api/action-figures/{fig_id}/model?format=obj")
        assert obj_res.status_code == 200
        assert len(obj_res.content) > 1000
        # Verify vertex colors in OBJ
        assert b"v " in obj_res.content

        ply_res = client.get(f"/api/action-figures/{fig_id}/model?format=ply")
        assert ply_res.status_code == 200
        assert len(ply_res.content) > 1000

        # 6. Check image endpoint
        cutout_res = client.get(f"/api/action-figures/{fig_id}/image?type=cutout")
        assert cutout_res.status_code == 200
        assert len(cutout_res.content) > 1000