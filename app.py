import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# ==========================================================
# Inisialisasi Aplikasi Flask
# ==========================================================
app = Flask(__name__)

# ==========================================================
# Konfigurasi Utama (FR-1, FR-2, FR-5)
# ==========================================================
# FR-1: Sumber video default (0 = webcam default).
# Dapat diganti ke file video lokal (contoh: 'video.mp4') atau URL RTSP (contoh: 'rtsp://...')
VIDEO_SOURCE = 0

# FR-2: Ambang batas kepercayaan (confidence threshold) deteksi YOLO
CONF_THRESHOLD = 0.5

# FR-5: Anti false-alarm, harus terdeteksi minimal 5 frame berurutan
CONSECUTIVE_FRAMES_THRESHOLD = 5

# Path file model YOLO hasil training
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "best_fire_ember.pt"

# ==========================================================
# Load Model Ultralytics YOLO
# ==========================================================
print(f"[*] Memuat model YOLO dari: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"[+] Model berhasil dimuat. Kelas terdaftar: {model.names}")

# ==========================================================
# State Global Sistem (Thread-Safe)
# ==========================================================
state_lock = threading.Lock()
consecutive_fire_count = 0
fire_detected_state = False
last_frame = None


def get_camera_stream():
    """
    Membuka koneksi ke sumber video OpenCV.
    Jika berupa webcam index, gunakan cv2.CAP_DSHOW untuk inisialisasi cepat di Windows.
    """
    if isinstance(VIDEO_SOURCE, int):
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    return cap


def generate_frames():
    """
    Generator streaming MJPEG untuk endpoint /video (FR-3).
    Membaca frame, menjalankan deteksi YOLO, menggambar bounding box,
    serta memperbarui status anti false-alarm (FR-5).
    """
    global consecutive_fire_count, fire_detected_state, last_frame

    cap = get_camera_stream()
    synthetic_counter = 0

    while True:
        success = False
        frame = None

        if cap.isOpened():
            success, frame = cap.read()
            # Jika sumber video berupa file lokal dan mencapai akhir, putar ulang (loop)
            if not success and isinstance(VIDEO_SOURCE, str) and os.path.exists(VIDEO_SOURCE):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()

        # Fallback jika kamera fisik tidak terdeteksi atau terputus
        if not success or frame is None:
            # Buat frame simulasi placeholder jika kamera tidak dapat dibuka
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            synthetic_counter += 1
            text = "Kamera Tidak Terdeteksi / Menunggu Sumber Video"
            cv2.putText(frame, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Sumber: {VIDEO_SOURCE} | Status: Standby", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            time.sleep(0.05)
            annotated_frame = frame
            has_fire_in_frame = False
        else:
            # FR-2: Jalankan inferensi deteksi YOLO pada frame saat ini
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=640,
                verbose=False
            )

            has_fire_in_frame = False

            # Cek apakah ada objek 'fire' atau 'ember' yang melampaui confidence threshold
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names.get(cls_id, "").lower()

                        if conf >= CONF_THRESHOLD and cls_name in ["fire", "ember"]:
                            has_fire_in_frame = True
                            break

                # FR-3: Gambar bounding box pada frame
                annotated_frame = result.plot()
            else:
                annotated_frame = frame

        # FR-5: Logika Anti False-Alarm (Minimal 5 frame berurutan)
        with state_lock:
            if has_fire_in_frame:
                consecutive_fire_count += 1
                if consecutive_fire_count >= CONSECUTIVE_FRAMES_THRESHOLD:
                    fire_detected_state = True
            else:
                consecutive_fire_count = 0
                fire_detected_state = False

            # Tambahkan visual status overlay pada sudut atas frame
            if fire_detected_state:
                cv2.rectangle(annotated_frame, (10, 10), (320, 50), (0, 0, 200), -1)
                cv2.putText(annotated_frame, "STATUS: API TERDETEKSI!", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (10, 10), (220, 50), (30, 120, 30), -1)
                cv2.putText(annotated_frame, "STATUS: AMAN", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Kompres frame ke format JPEG untuk streaming MJPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Format multipart stream MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ==========================================================
# Rute dan Endpoint Aplikasi Web
# ==========================================================

@app.route('/')
def index():
    """
    FR-6: Halaman utama menampilkan judul, tombol alarm, status, dan video.
    """
    return render_template('index.html')


@app.route('/video')
def video_feed():
    """
    FR-3: Endpoint streaming video MJPEG dengan bounding box deteksi.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """
    FR-4: Endpoint status yang mengembalikan JSON {"fire": true/false}.
    """
    with state_lock:
        current_state = fire_detected_state
        current_count = consecutive_fire_count

    return jsonify({
        "fire": current_state,
        "consecutive_frames": current_count
    })


# ==========================================================
# Menjalankan Server Flask (Kebutuhan Non-Fungsional FR-11)
# ==========================================================
if __name__ == '__main__':
    print("[*] Menjalankan server Deteksi Kebakaran pada http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
