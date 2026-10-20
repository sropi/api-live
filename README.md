# Sistem Deteksi Kebakaran Real-Time 🔥

Aplikasi web pendeteksi api (*fire*) dan bara api (*ember*) secara langsung (*real-time*) dari kamera atau video stream menggunakan model Vision AI **Ultralytics YOLO** (`best.pt`), backend **Flask**, dan antarmuka web modern dengan sistem peringatan visual (background berkedip merah) serta audio sirine nada ganda via **Web Audio API**.

---

## 📁 Struktur Folder

```text
fire-detector/
├── app.py                 # Backend Flask, inferensi YOLO & streaming MJPEG
├── best.pt                # Bobot model YOLO hasil training (Fire & Ember)
├── best_fire_ember.pt     # Backup file bobot model asli
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi & panduan penggunaan
└── templates/
    └── index.html         # Antarmuka web vanilla HTML/CSS/JS + Web Audio API
```

---

## 🛠️ Kebutuhan Sistem (Tech Stack)

- **Python**: 3.10 atau lebih baru
- **Backend**: Flask
- **Vision AI**: Ultralytics YOLO (`best.pt`) & OpenCV (`cv2`)
- **Frontend**: HTML5, Vanilla CSS3, Vanilla JavaScript
- **Audio Alert**: Web Audio API (Sintesis sirine murni tanpa file MP3)

---

## 🚀 Panduan Menjalankan

### 1. Masuk ke Direktori Proyek
Buka terminal (Command Prompt / PowerShell / Bash) lalu arahkan ke folder proyek:
```bash
cd fire-ember-web
```

### 2. (Opsional namun Disarankan) Buat Virtual Environment
```bash
python -m venv .venv
# Di Windows:
.\.venv\Scripts\activate
# Di Linux/macOS:
source .venv/bin/activate
```

### 3. Instalasi Dependensi
Jalankan perintah berikut:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
python app.py
```

### 5. Buka di Browser
Akses aplikasi melalui peramban:
- Lokal: [http://localhost:5000](http://localhost:5000) atau [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Dari perangkat lain di jaringan yang sama: `http://<IP-LAPTOP-KAMU>:5000`

---

## ⚙️ Konfigurasi Sumber Video & Sensitivitas

Buka file `app.py` untuk menyesuaikan pengaturan pada baris atas:

```python
# 1. Sumber Video (FR-1):
VIDEO_SOURCE = 0              # 0 untuk webcam bawaan
# VIDEO_SOURCE = 'demo.mp4'   # Atau ganti dengan path file video MP4
# VIDEO_SOURCE = 'rtsp://...' # Atau URL RTSP CCTV

# 2. Ambang Batas Kepercayaan / Confidence (FR-2):
CONF_THRESHOLD = 0.5          # Nilai 0.0 sampai 1.0

# 3. Anti False-Alarm Filter (FR-5):
CONSECUTIVE_FRAMES_THRESHOLD = 5  # Jumlah frame berurutan sebelum alarm berbunyi
```

---

## 🧪 Skenario Pengujian Demo (Kriteria Evaluasi)

1. **Uji Coba Kondisi Normal (Aman):**
   - Buka halaman web. Pastikan video live menyala dengan label status hijau `✅ Aman`.
2. **Aktifkan Izin Suara:**
   - Klik tombol **"Aktifkan Alarm"** (sesuai aturan Autoplay Browser). Anda juga dapat menekan tombol **"Tes Suara"** untuk memastikan output speaker berfungsi normal.
3. **Uji Coba Deteksi Api (Alarm Menyala):**
   - Tampilkan gambar atau video api pada layar smartphone di hadapan webcam (atau gunakan file `demo.mp4`).
   - Saat model mendeteksi api minimal 5 frame berturut-turut:
     - 🚨 Status berubah menjadi `🚨 API TERDETEKSI!`.
     - 🔴 Background halaman berkedip merah-hitam secara intensif.
     - 🔊 Suara sirine nada ganda (900 Hz dan 650 Hz bergantian tiap 300 ms) menyala terus-menerus.
4. **Uji Coba Saat Api Hilang:**
   - Jauhkan gambar/video api dari kamera.
   - Kedipan merah dan suara sirine akan berhenti secara otomatis dalam hitungan milidetik, lalu status kembali menjadi `✅ Aman`.

---

## 🌐 Endpoint API

- `GET /` : Menampilkan antarmuka monitor real-time.
- `GET /video` : Streaming MJPEG frame kamera dengan bounding box deteksi.
- `GET /status` : Mengembalikan format JSON status kondisi ruangan, contoh:
  ```json
  {
    "fire": true,
    "consecutive_frames": 12
  }
  ```
