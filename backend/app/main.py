"""FastAPI backend and asynchronous job runner for NeuralShape MVP (PRD Sections 10 & 11)."""
import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from backend.app import services
from backend.app.config import Settings
from backend.app.db import Database, loads
from backend.app.schemas import CreateProjectRequest, ReconstructRequest, TrainExperimentRequest
from backend.app.validation import validate_image_set


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    registry = services.ModelRegistry(settings)
    executor = ThreadPoolExecutor(max_workers=settings.max_workers)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="NeuralShape API", version="1.1.0", lifespan=lifespan)
    app.state.settings, app.state.db, app.state.registry, app.state.executor = settings, db, registry, executor
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def project_dir(project_id: str) -> Path:
        if not db.fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,)):
            raise HTTPException(status_code=404, detail="Project not found")
        return settings.projects_dir / project_id

    @app.get("/", response_class=HTMLResponse)
    def serve_studio():
        html_path = settings.frontend_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>NeuralShape Backend Active</h1><p>Open /docs for OpenAPI specification.</p>")

    @app.get("/api/health")
    def health():
        ckpt = settings.default_checkpoint
        return {"status": "ok", "checkpoint_ready": ckpt.exists(), "checkpoint": str(ckpt)}

    @app.post("/api/projects")
    def create_project(req: CreateProjectRequest):
        proj_id = f"proj_{uuid.uuid4().hex[:8]}"
        now = time.time()
        proj_dir = settings.projects_dir / proj_id
        (proj_dir / "images").mkdir(parents=True, exist_ok=True)
        category = services.resolve_category(req.category)
        if req.auto_generate_synthetic_views:
            seed = req.seed if req.seed is not None else (uuid.uuid4().int % 2**31)
            services.save_synthetic_views(proj_dir, category, seed=seed, settings=settings)
        db.execute(
            "INSERT INTO projects (id, name, category, status, created_at, updated_at) VALUES (?, ?, ?, 'Draft', ?, ?)",
            (proj_id, req.name, category, now, now),
        )
        return {"id": proj_id, "name": req.name, "category": category, "status": "Draft"}

    @app.get("/api/projects")
    def list_projects():
        return db.fetch_all(
            "SELECT id, name, category, status, created_at, updated_at FROM projects ORDER BY created_at DESC"
        )

    @app.post("/api/projects/{project_id}/images")
    async def upload_images(project_id: str, files: List[UploadFile] = File(...)):
        proj_dir = project_dir(project_id)
        decoded = []
        for f in files:
            ext = Path(f.filename or "img.png").suffix.lower()
            if ext not in services.IMAGE_EXTS:
                raise HTTPException(status_code=400, detail=f"Format tidak didukung: {ext}")
            raw = await f.read()
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(status_code=400, detail=f"File gambar rusak: {f.filename}")
            decoded.append((ext, raw, bgr))

        # Validate everything before touching the project so a bad file cannot leave it half-updated.
        services.reset_project_inputs(proj_dir)
        saved, tensors = [], []
        for idx, (ext, raw, bgr) in enumerate(decoded):
            dest = proj_dir / "images" / f"upload_{idx:02d}{ext}"
            dest.write_bytes(raw)
            saved.append(dest.name)
            tensors.append(services.to_model_input(bgr, settings.img_size))
        np.save(proj_dir / "images_tensor.npy", np.stack(tensors, axis=0))

        db.execute("UPDATE projects SET status = 'Uploaded', results_json = NULL, metrics_json = NULL, updated_at = ? WHERE id = ?",
                   (time.time(), project_id))
        warnings = []
        if len(saved) != services.NUM_VIEWS:
            warnings.append(
                f"Model dikalibrasi untuk {services.NUM_VIEWS} sudut kamera tetap "
                f"({', '.join(v[2] for v in services.CAMERA_VIEWS)}); foto diurutkan sesuai urutan tersebut."
            )
        return {"project_id": project_id, "uploaded_count": len(saved), "files": saved, "warnings": warnings}

    @app.post("/api/projects/{project_id}/validate")
    def validate_project_images(project_id: str):
        img_dir = project_dir(project_id) / "images"
        paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in services.IMAGE_EXTS) if img_dir.exists() else []
        imgs_rgb, names = [], []
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is not None:
                imgs_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                names.append(p.name)
        if not imgs_rgb:
            raise HTTPException(status_code=400, detail="Belum ada gambar untuk divalidasi.")

        report = validate_image_set(imgs_rgb, names)
        db.execute(
            "UPDATE projects SET status = ?, validation_json = ?, updated_at = ? WHERE id = ?",
            ("Validated" if report["overall_valid"] else "Validation Warning", json.dumps(report), time.time(), project_id),
        )
        return report

    @app.post("/api/projects/{project_id}/reconstruct")
    def start_reconstruction(project_id: str, req: ReconstructRequest):
        proj_dir = project_dir(project_id)
        if not (proj_dir / "images_tensor.npy").exists():
            raise HTTPException(status_code=400, detail="Project belum memiliki gambar. Unggah foto terlebih dahulu.")
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        now = time.time()
        db.execute(
            "INSERT INTO jobs (id, project_id, job_type, method, status, progress, created_at) VALUES (?, ?, 'reconstruct', ?, 'Queued', 0, ?)",
            (job_id, project_id, req.method, now),
        )
        db.execute("UPDATE projects SET status = 'Queued', updated_at = ? WHERE id = ?", (now, project_id))
        executor.submit(services.run_reconstruction_job, settings, db, registry, job_id, project_id, req.method)
        return {"job_id": job_id, "project_id": project_id, "status": "Queued", "method": req.method}

    @app.get("/api/jobs/{job_id}")
    def get_job_status(job_id: str):
        row = db.fetch_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return row

    def _project_json(project_id: str, column: str, key: str):
        row = db.fetch_one(f"SELECT id, status, {column} FROM projects WHERE id = ?", (project_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"project_id": row["id"], "status": row["status"], key: loads(row[column])}

    @app.get("/api/projects/{project_id}/results")
    def get_project_results(project_id: str):
        return _project_json(project_id, "results_json", "results")

    @app.get("/api/projects/{project_id}/metrics")
    def get_project_metrics(project_id: str):
        return _project_json(project_id, "metrics_json", "metrics")

    @app.get("/api/projects/{project_id}/export")
    def export_project_model(
        project_id: str,
        format: str = Query("obj", pattern="^(obj|ply)$"),
        source: str = Query("ml", pattern="^(ml|baseline)$"),
    ):
        fname = f"{source}_mesh.{format}"
        fpath = project_dir(project_id) / fname
        if not fpath.exists():
            raise HTTPException(status_code=404, detail=f"Export file {fname} belum tersedia. Jalankan rekonstruksi terlebih dahulu.")
        return FileResponse(fpath, filename=f"{project_id}_{fname}", media_type="application/octet-stream")

    @app.post("/api/ml/experiments")
    def create_ml_experiment(req: TrainExperimentRequest):
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        cfg = req.model_dump()
        db.execute(
            "INSERT INTO ml_experiments (id, name, status, config_json, created_at) VALUES (?, ?, 'Running', ?, ?)",
            (exp_id, req.name, json.dumps(cfg), time.time()),
        )
        executor.submit(services.run_experiment_job, settings, db, exp_id, cfg)
        return {"experiment_id": exp_id, "status": "Running", "config": cfg}

    @app.get("/api/ml/experiments/{exp_id}")
    def get_ml_experiment(exp_id: str):
        row = db.fetch_one("SELECT * FROM ml_experiments WHERE id = ?", (exp_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {
            "id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "config": loads(row["config_json"]),
            "progress": loads(row["progress_json"]),
            "result": loads(row["result_json"]),
        }

    # =========================================================================
    # Action Figure & Single-Image 3D Endpoints
    # =========================================================================
    @app.get("/api/action-figures/samples")
    def get_action_figure_samples():
        sample_dir = settings.sample_images_dir / "4_anime_action_figure"
        samples = []
        if (sample_dir / "anime_hero.jpg").exists():
            samples.append({
                "id": "anime_hero",
                "name": "Samurai Anime Hero",
                "description": "Japanese warrior figure with spiky blue hair and red-gold armor",
                "image_url": "/api/action-figures/samples/anime_hero/image",
            })
        if (sample_dir / "anime_chibi.jpg").exists():
            samples.append({
                "id": "anime_chibi",
                "name": "Magical Girl Chibi",
                "description": "Cute anime chibi collectible figure with pink hair and magical wand",
                "image_url": "/api/action-figures/samples/anime_chibi/image",
            })
        return samples

    @app.get("/api/action-figures/samples/{sample_id}/image")
    def get_action_figure_sample_image(sample_id: str):
        sample_dir = settings.sample_images_dir / "4_anime_action_figure"
        path = sample_dir / f"{sample_id}.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Sample image not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/action-figures/generate")
    async def create_action_figure(
        name: str = "Action Figure",
        mc_resolution: int = 128,
        remove_bg: bool = True,
        foreground_ratio: float = 0.85,
        sample_preset: Optional[str] = None,
        file: Optional[UploadFile] = File(None),
    ):
        fig_id = f"fig_{uuid.uuid4().hex[:8]}"
        fig_dir = settings.projects_dir / "action_figures" / fig_id
        fig_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()

        src_img_path = fig_dir / "source.png"
        if file is not None:
            raw = await file.read()
            src_img_path.write_bytes(raw)
        elif sample_preset:
            sample_file = settings.sample_images_dir / "4_anime_action_figure" / f"{sample_preset}.jpg"
            if not sample_file.exists():
                raise HTTPException(status_code=400, detail=f"Preset sample '{sample_preset}' tidak ditemukan.")
            shutil.copy2(sample_file, src_img_path)
            if name == "Action Figure":
                name = sample_preset.replace("_", " ").title()
        else:
            raise HTTPException(status_code=400, detail="Harap unggah gambar atau pilih sample preset.")

        db.execute(
            """
            INSERT INTO action_figures (id, name, status, mc_resolution, progress, progress_msg, created_at)
            VALUES (?, ?, 'Processing', ?, 0, 'Antrean pembuatan 3D...', ?)
            """,
            (fig_id, name, mc_resolution, now),
        )

        executor.submit(
            services.run_action_figure_job,
            settings,
            db,
            fig_id,
            src_img_path,
            name,
            mc_resolution,
            remove_bg,
            foreground_ratio,
        )

        return {
            "id": fig_id,
            "name": name,
            "status": "Processing",
            "mc_resolution": mc_resolution,
            "created_at": now,
        }

    @app.get("/api/action-figures")
    def list_action_figures():
        return db.fetch_all(
            """
            SELECT id, name, status, mc_resolution, progress, progress_msg,
                   error, vertex_count, face_count, duration_seconds,
                   metadata_json, created_at, completed_at
            FROM action_figures
            ORDER BY created_at DESC
            """
        )

    @app.get("/api/action-figures/{figure_id}")
    def get_action_figure(figure_id: str):
        row = db.fetch_one("SELECT * FROM action_figures WHERE id = ?", (figure_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Action Figure not found")
        row_dict = dict(row)
        row_dict["metadata"] = loads(row_dict.pop("metadata_json", None))
        return row_dict

    @app.get("/api/action-figures/{figure_id}/model")
    def download_action_figure_model(
        figure_id: str,
        format: str = Query("glb", pattern="^(glb|obj|ply)$"),
    ):
        fig_dir = settings.projects_dir / "action_figures" / figure_id
        path = fig_dir / f"model.{format}"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model .{format} belum selesai atau tidak ditemukan.")
        media_types = {
            "glb": "model/gltf-binary",
            "obj": "text/plain",
            "ply": "application/octet-stream",
        }
        return FileResponse(
            path,
            filename=f"{figure_id}_model.{format}",
            media_type=media_types.get(format, "application/octet-stream"),
        )

    @app.get("/api/action-figures/{figure_id}/image")
    def get_action_figure_image(
        figure_id: str,
        type: str = Query("source", pattern="^(source|cutout|ready)$"),
    ):
        fig_dir = settings.projects_dir / "action_figures" / figure_id
        mapping = {
            "source": fig_dir / "source.png",
            "cutout": fig_dir / "model_cutout.png",
            "ready": fig_dir / "model_input.png",
        }
        path = mapping.get(type, fig_dir / "source.png")
        if not path.exists():
            path = fig_dir / "source.png"
            if not path.exists():
                raise HTTPException(status_code=404, detail="Gambar tidak ditemukan.")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()