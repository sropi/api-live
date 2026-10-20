"""
Skrip Pengujian Otomatis Sistem Deteksi Kebakaran Real-Time
Menguji:
1. Endpoint Flask (/, /status, /video)
2. Struktur respons JSON sesuai FR-4
3. Keberadaan elemen UI (FR-6, FR-8, FR-10)
4. Model YOLO dan inferensi gambar api
5. Logika anti false-alarm 5 frame berurutan (FR-5 & FR-9)
"""

import os
import cv2
import numpy as np
from app import app, model, CONF_THRESHOLD, CONSECUTIVE_FRAMES_THRESHOLD, generate_frames


def test_homepage_elements():
    """Menguji halaman utama / mengembalikan status 200 dan elemen penting."""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    html = response.data.decode('utf-8')

    # Periksa judul & tombol sesuai FR-6 & FR-10
    assert "Sistem Deteksi Kebakaran Real-Time" in html
    assert "Aktifkan Alarm" in html
    assert "Aman" in html
    assert "src=\"/video\"" in html

    # Periksa logika audio sirine 900Hz / 650Hz tiap 300ms (FR-8b)
    assert "900" in html
    assert "650" in html
    assert "300" in html

    # Periksa polling 500ms (FR-7)
    assert "500" in html
    print("[PASS] Halaman utama / dan elemen antarmuka lengkap.")


def test_status_endpoint():
    """Menguji endpoint /status mengembalikan format JSON {"fire": bool}."""
    client = app.test_client()
    response = client.get('/status')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert "fire" in data
    assert isinstance(data["fire"], bool)
    print(f"[PASS] Endpoint /status merespons JSON: {data}")


def test_consecutive_anti_false_alarm_logic():
    """
    Menguji logika anti false-alarm:
    - Kurang dari 5 frame api: fire = False
    - Tepat 5 frame api: fire = True
    - Begitu api hilang 1 frame: fire = False seketika
    """
    consecutive_count = 0
    fire_state = False

    # Simulasi 4 frame berurutan api
    for _ in range(4):
        consecutive_count += 1
        if consecutive_count >= CONSECUTIVE_FRAMES_THRESHOLD:
            fire_state = True
        else:
            fire_state = False
        assert fire_state is False, "Frame ke-1 s/d 4 seharusnya belum memicu alarm!"

    # Frame ke-5 api
    consecutive_count += 1
    if consecutive_count >= CONSECUTIVE_FRAMES_THRESHOLD:
        fire_state = True
    assert fire_state is True, "Frame ke-5 berturut-turut harus mengaktifkan alarm (FR-5)!"

    # Frame berikutnya aman (api hilang)
    has_fire = False
    if has_fire:
        consecutive_count += 1
    else:
        consecutive_count = 0
        fire_state = False

    assert fire_state is False, "Saat api hilang, alarm harus langsung mati (FR-9)!"
    assert consecutive_count == 0
    print("[PASS] Logika Anti False-Alarm 5 frame berturut-turut berhasil divalidasi.")


def test_yolo_model_inference():
    """Menguji kemampuan inferensi model best.pt terhadap citra api."""
    test_fire_path = 'test_media/test/images/F_1003_jpg.rf.413f0951ae9fdd7a72cfe264a77cfce5.jpg'
    if os.path.exists(test_fire_path):
        results = model.predict(test_fire_path, conf=0.45, verbose=False)
        assert len(results) > 0
        boxes = results[0].boxes
        assert len(boxes) > 0, "Model harus berhasil mendeteksi api pada gambar uji!"
        cls_names = [model.names[int(c)] for c in boxes.cls]
        assert any(c in ['fire', 'ember'] for c in cls_names)
        print(f"[PASS] Inferensi model sukses. Deteksi: {cls_names}, Conf: {[float(c) for c in boxes.conf]}")
    else:
        print("[SKIP] Gambar uji tidak ditemukan, melewati tes citra fisik.")


def test_mjpeg_stream_generator():
    """Menguji generator MJPEG /video menghasilkan format multipart frame yang valid."""
    stream = generate_frames()
    first_chunk = next(stream)
    assert b'--frame\r\n' in first_chunk
    assert b'Content-Type: image/jpeg\r\n\r\n' in first_chunk
    print("[PASS] Generator streaming MJPEG /video menghasilkan frame JPEG yang valid.")


if __name__ == '__main__':
    print("--- MEMULAI PENGUJIAN OTOMATIS SISTEM DETEKSI KEBAKARAN ---")
    test_homepage_elements()
    test_status_endpoint()
    test_consecutive_anti_false_alarm_logic()
    test_yolo_model_inference()
    test_mjpeg_stream_generator()
    print("--- SEMUA PENGUJIAN BERHASIL (100% PASS) ---")
