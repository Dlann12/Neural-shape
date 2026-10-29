# Sample Multi-View Test Images (Foto Uji Coba Langsung)

Folder ini berisi 3 set foto multi-view resolusi 256×256 yang sudah diurutkan berdasarkan 8 sudut kamera terkalibrasi (*front, front_right, right, back_right, back, back_left, left, top_diagonal*).

---

## 📁 Daftar Set Foto Uji

### 1. `sample_images/1_kursi_chair/` (Rekomendasi Utama)
* **Kategori**: `chair`
* **Isi**: 8 foto sudut pandang 360° mengelilingi kursi dengan sandaran lengan (*armrests*) dan cekungan dudukan.
* **File**:
  - `01_front_0deg.png`
  - `02_front_right_45deg.png`
  - `03_right_90deg.png`
  - `04_back_right_135deg.png`
  - `05_back_180deg.png`
  - `06_back_left_225deg.png`
  - `07_left_270deg.png`
  - `08_top_diagonal_315deg.png`

### 2. `sample_images/2_mangkuk_vessel/`
* **Kategori**: `hollow_vessel`
* **Isi**: 8 foto mangkuk berongga cekung (menguji kemampuan AI memotong rongga dalam mangkuk).

### 3. `sample_images/3_monumen_arch/`
* **Kategori**: `arch_monument`
* **Isi**: 8 foto monumen dengan terowongan tembus pandang.

---

## 🚀 Cara Mengunggah di Web Studio

1. Buka browser: `http://localhost:8000`
2. Di panel kiri:
   - Pilih Kategori yang sesuai (misal: `Chair`).
   - Pada input **"Unggah Foto Sendiri"**, klik tombol pilih file.
   - Buka folder `sample_images/1_kursi_chair/` lalu pilih seluruh 8 foto sekaligus (`Ctrl + A` lalu klik *Open*).
   - Klik tombol **"Buat Proyek + Jalankan Rekonstruksi"**.
3. Sistem akan otomatis:
   - Mengunggah 8 foto.
   - Menjalankan analisis kualitas gambar OpenCV.
   - Menjalankan rekonstruksi 3D (*PyTorch Multi-View ML* dan *Classical Visual Hull*).
   - Menampilkan model 3D interaktif yang bisa diputar 360° di layar.
