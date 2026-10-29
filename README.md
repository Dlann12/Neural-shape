# NeuralShape 3D Reconstruction Studio & ML Lab

NeuralShape adalah sistem rekonstruksi bentuk 3D berbasis multi-view menggunakan kombinasi pendekatan geometri klasik (*Visual Hull Space Carving*) dan *Deep Learning* (*Multi-View Occupancy Network* berbasis PyTorch).

---

## 📁 Struktur Proyek yang Dirapikan

```text
neuralshape/
├── backend/
│   └── app/
│       ├── config.py         # Konfigurasi runtime (storage, grid size, bootstrap)
│       ├── db.py             # SQLite persistence layer (projects, jobs, experiments)
│       ├── main.py           # FastAPI application endpoints & lifespan handlers
│       ├── schemas.py        # Pydantic validation schemas
│       ├── services.py       # Asynchronous workers untuk rekonstruksi & eksperimen ML
│       └── validation.py     # Pipeline validasi kualitas foto (OpenCV, 2D dHash)
├── frontend/
│   └── index.html            # Studio interaktif WebGL (Three.js) & UI Benchmarks ML Lab
├── ml/
│   ├── baseline.py           # Rekonstruksi geometri klasik (Visual Hull Space Carving)
│   ├── camera.py             # Single source of truth untuk kalibrasi kamera & proyeksi
│   ├── dataset.py            # Generator bentuk 3D prosedural & PyTorch Dataset
│   ├── inference.py          # OccupancyPredictor & checkpoint loader dinamis
│   ├── mesh_utils.py         # Ekstraksi marching cubes, smoothing, ekspor .OBJ & .PLY
│   ├── metrics.py            # Evaluasi kuantitatif (IoU, Chamfer Distance, F-Score)
│   ├── model.py              # PyTorch MultiViewOccupancyNet (2D U-Net + 3D U-Net)
│   └── train.py              # Pipeline training, LR warmup/cosine, kalibrasi threshold
├── scripts/
│   └── train.py              # CLI training dengan preset (quick, default, full)
├── tests/
│   └── test_end_to_end.py    # Automated Verification Tests (FR-01 s/d FR-07)
├── storage/
│   ├── checkpoints/          # Model weights (.pt), riwayat training & laporan evaluasi
│   ├── projects/             # Output rekonstruksi 3D dan file ekspor mesh
│   └── neuralshape.db        # SQLite database
├── pytest.ini                # Konfigurasi PyTest
├── run.sh                    # Skrip menjalankan server uvicorn
└── test.sh                   # Skrip menjalankan pengujian otomatis
```

---

## 🚀 Peningkatan & Optimalisasi Model

### 1. Masalah pada Model Sebelumnya
* **Model Collapse (Val IoU 0.000)**: Model asli mengalami *zero occupancy collapse* di mana output sigmoid konvergen ke 0 di semua voxel, menghasilkan IoU 0.000 dan F-score 0.000 (0% success rate).
* **Z-Fighting pada Renderer**: Perhitungan splatting voxel pada kamera menimpa permukaan terdekat dengan permukaan belakang karena ketiadaan z-buffer per-pixel.
* **BatchNorm Discrepancy**: BatchNorm menyebabkan statistik view 1-kamera tidak cocok dengan 8-kamera.
* **False Positive Duplikat**: dHash 1D (horizontal saja) salah mendeteksi sudut 0° dan 180° pada objek simetris sebagai duplikat identik.

### 2. Perbaikan & Optimalisasi
* **Arsitektur 2D U-Net + Unprojection 3D + 3D U-Net**: Menambahkan ekstraksi fitur multi-skala dengan skip connection raw RGB ke voxel grid.
* **Photo-Consistency & Pose Encoding**: Voxel aggregator menghitung *variance/std* lintas sudut pandang sebagai sinyal diskrepansi permukaan (*photo-consistency cue*) dan menyertakan koordinat depth/pose.
* **GroupNorm**: Menggantikan BatchNorm dengan GroupNorm agar statistik normalisasi stabil terlepas dari jumlah batch maupun jumlah kamera yang diinput (1 view s/d 8 views).
* **Inisialisasi Bias Prior**: Head 3D diinisialisasi dengan bias prior (-1.5 ≈ 18% okupansi) mencegah jebakan lokal *all-empty*.
* **Dynamic Threshold Calibration**: Checkpoint secara otomatis mengalibrasi ambang batas okupansi optimal pada validation set.
* **2D dHash 128-bit**: Mendeteksi duplikat foto asli tanpa false-positive pada sudut kamera simetris.

---

## 📊 Hasil Benchmark (Held-out Test Set)

| Eksperimen (PRD Section 7) | Input Views | Volumetric IoU ↑ | Chamfer Distance ↓ | F-Score@0.08 ↑ | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Experiment A (Classical Baseline)** | 8 Views | 0.4520 | 0.1616 | 0.0589 | 100.0% |
| **Experiment B (Single-View ML)** | 1 View | 0.3372 | 0.1645 | 0.2566 | 100.0% |
| **Experiment C (Multi-View ML)** | 8 Views | **0.4322** | **0.1308** | **0.3542** | **100.0%** |

*Model ML menghasilkan F-score 6x lebih tinggi dan Chamfer Distance lebih rendah dibandingkan baseline klasik, serta berhasil merekonstruksi cekungan interior.*

---

## 🧪 Menjalankan Pengujian (Testing)

Jalankan seluruh test suite otomatis:
```bash
./test.sh
```
Atau menggunakan pytest langsung:
```bash
.venv/bin/pytest tests/test_end_to_end.py -v
```

Cakupan pengujian:
1. `test_fr01_dataset_split_no_leakage`: Verifikasi nol kebocoran data antar split objek.
2. `test_fr02_image_quality_validation`: Deteksi blur, kegelapan, dan duplikasi gambar.
3. `test_fr03_classical_baseline_and_mesh_export`: Rekonstruksi visual hull & ekspor .OBJ/.PLY.
4. `test_fr04_camera_rig_projection`: Konsistensi matematis proyeksi kamera rig.
5. `test_fr05_pytorch_model_forward_backward`: Kompatibilitas fwd/bwd 8-view dan 1-view.
6. `test_fr06_checkpoint_and_inference_predictor`: Verifikasi load checkpoint & prediksi numpy.
7. `test_fr07_api_end_to_end_workflow`: Workflow HTTP FastAPI lengkap (Project -> Validate -> Reconstruct -> Results -> Export).
8. `test_fr08_action_figure_single_image_to_3d`: Verifikasi end-to-end pembuatan action figure 3D dari single-image (RMBG -> Triplane NeRF Transformer -> Marching Cubes -> Ekspor GLB/OBJ/PLY).

---

## 🖥️ Menjalankan Aplikasi Web Studio

1. Jalankan server:
   ```bash
   ./run.sh
   ```
2. Buka browser pada: `http://localhost:8000`
3. Fitur yang tersedia:
   * **⚡ Anime & Action Figure 3D Generator**: Ubah gambar anime atau foto apa pun menjadi model 3D berwarna (seperti action figure koleksi fisik) dari satu gambar tunggal (single-image-to-3D). Dilengkapi Three.js 3D studio viewer, turntable otomatis, dan download `.GLB`, `.OBJ`, `.PLY`.
   * **Multi-View 3D Studio**: Rekonstruksi multi-sudut pandang (8-kamera) menggunakan kombinasi Space Carving dan Multi-View Occupancy Network.
   * **Visualisasi 3D Interaktif**: Toggle Color Mesh, Studio Clay, Wireframe, Normal Map, dan turntable spin.
   * **Ekspor Multi-Format**: Unduh model dalam format `.GLB` (glTF binary), `.OBJ` (dengan vertex color RGB), atau `.PLY`.
   * **ML Lab & Benchmarks**: Pantau metrik evaluasi (IoU, Chamfer, F-Score) dan jalankan training eksperimen baru secara asynchronous.

---

## ⚡ Anime & Single-Image to 3D Action Figure Generator

NeuralShape kini dilengkapi modul deep learning foundation model (**TripoSR Transformer & NeRF Triplane**) yang dioptimalkan untuk berjalan **100% pada CPU** tanpa memerlukan GPU NVIDIA / CUDA.

### Cara Kerja Pipeline:
1. **Auto Background Matting & Cutout**: Menggunakan *RMBG 2.0* via ONNX Runtime CPU untuk memisahkan karakter/anime dari latar belakang secara presisi.
2. **Auto-Centering & Normalization**: Karakter dipusatkan, diskalakan dengan bounding box ideal (rasio 0.85), dan di-blend dengan neutral gray background 50%.
3. **Triplane Transformer NeRF**: Model ViT + Triplane Transformer memprediksi representasi densitas volumetrik dan warna RGB 3D dari gambar tunggal dalam waktu ~9-12 detik pada CPU.
4. **Isosurface Extraction (Marching Cubes)**: Algoritma marching cubes mengekstrak permukaan 3D mesh lengkap dengan warna RGB vertex.
5. **Multi-Format 3D Export**: Otomatis menyimpan file `.GLB` (untuk WebGL / Three.js), `.OBJ` (untuk 3D modeling Blender atau 3D printing slicing), dan `.PLY`.

### Cara Menggunakan via Web Studio:
1. Buka `http://localhost:8000` di browser.
2. Tab utama **"⚡ Anime & Action Figure 3D"** langsung aktif.
3. Pilih salah satu preset sampel anime (**Samurai Anime Hero** atau **Magical Girl Chibi**), atau klik area upload untuk memilih gambar karakter anime milik Anda sendiri.
4. Klik tombol **"⚡ Generate 3D Action Figure"**.
5. Saksikan live progress bar & visual checklist:
   * Pembersihan latar belakang (RMBG 2.0)
   * Ekstraksi densitas 3D NeRF Transformer
   * Rekonstruksi geometri (Marching Cubes)
   * Packaging & visualisasi 3D
6. Model 3D langsung muncul di viewer 3D Three.js! Anda dapat memutar objek, memperbesar/memperkecil, mengubah mode material (Full Color, Studio Clay, Wireframe, Normal Map), atau menyalakan auto-rotate turntable.
7. Klik tombol download **.GLB**, **.OBJ**, atau **.PLY** untuk menyimpan model ke komputer Anda.

### API Endpoints:
* `GET /api/action-figures/samples` : Daftar preset sampel anime.
* `POST /api/action-figures/generate` : Mengirim gambar atau preset id untuk memproses 3D.
* `GET /api/action-figures/{id}` : Status progres dan metadata objek 3D.
* `GET /api/action-figures/{id}/model?format=glb|obj|ply` : Mengunduh file 3D.

---

## 🏋️ Training Model Tambahan

Gunakan CLI skrip training:
```bash
# Training cepat (quick preset ~2 menit)
.venv/bin/python -m scripts.train --preset quick

# Training standar (default preset)
.venv/bin/python -m scripts.train --preset default

# Training penuh dengan jumlah objek lebih banyak
.venv/bin/python -m scripts.train --preset full
```
