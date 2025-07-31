# Indonesian License Plate Recognition — UAS Computer Vision

**Proyek UAS — OCR Plat Nomor**  
**Integrasi Visual Language Model (VLM) dan Python untuk Inferensi Teks Otomatis**

🎓 **Proyek Ujian Akhir Semester UAS Computer Vision oleh Zain Daffa**  
🤖 **Menggunakan LM Studio & Model BakLLaVA untuk mengenali plat nomor kendaraan**  
📊 **Evaluasi dengan Character Error Rate (CER) secara otomatis**

---

## 🧠 Deskripsi Singkat

Proyek ini bertujuan untuk melakukan inferensi **OCR (Optical Character Recognition)** pada gambar plat nomor kendaraan menggunakan **Visual Language Model (VLM)** dari **LM Studio**.

Evaluasi hasil prediksi dilakukan dengan menghitung **Character Error Rate (CER)**, sehingga dapat diketahui tingkat akurasi sistem secara objektif.

---

## 📁 Struktur Folder

```
UAS-RE604-COMPUTER-VISION/
├── test/
│   ├── *.jpg                # 📸 Gambar plat nomor
│   ├── *.txt                # 📝 Label YOLO format (class_id x_center y_center width height)
│   ├── classes.txt          # 🏷️ Mapping ID kelas ke karakter
│   └── ground_truth.csv     # 📄 Data hasil konversi dari .txt (auto-generated)
│
├── generate_ground_truth_csv.py  # 🔄 Script untuk konversi .txt ke CSV
├── main.py                       # 🚀 Script utama untuk inferensi dan evaluasi VLM
├── ocr_results.csv              # 📊 Hasil akhir: image, ground_truth, prediction, CER_score
└── README.md                    # 📖 Dokumentasi proyek
```

---

## 🚀 Langkah Eksekusi

### 🧾 1. Persiapan Dataset
* Siapkan folder `test` yang berisi:
  * Gambar plat nomor (.jpg, .png, .bmp)
  * Label plat nomor dalam format `.txt` dengan format YOLO
* **Format file .txt**: Setiap baris berisi `class_id x_center y_center width height`
* **File `classes.txt`**: Berisi mapping ID kelas ke karakter (0-9, A-Z)

### 🔄 2. Generate Ground Truth CSV
**PENTING**: Sebelum menjalankan `main.py`, jalankan terlebih dahulu script untuk mengkonversi label `.txt` menjadi `ground_truth.csv`:

```bash
python generate_ground_truth_csv.py
```

**✅ Fungsi script:**
* Membaca semua file `.txt` di folder dataset
* Mengkonversi ID kelas menjadi karakter menggunakan label mapping
* Mengurutkan karakter berdasarkan posisi x_center (kiri ke kanan)
* Menggabungkan karakter menjadi string plat nomor lengkap
* Menyimpan hasil ke `ground_truth.csv`

**Konfigurasi di `generate_ground_truth_csv.py`:**
```python
# Sesuaikan path dataset Anda
dataset_folder = r"C:\Users\ASUS_TUF_GAMING\Documents\Dataset\test"
```

### 🤖 3. Setup LM Studio
Pastikan **LM Studio** telah terinstall dan model **BakLLaVA** telah di-load:

1. Buka **LM Studio**
2. Load model `bakllava1-mistralllava-7b` atau model VLM lainnya
3. Jalankan server lokal
4. 📡 Server akan berjalan di: `http://localhost:1234`

### ⚙️ 4. Konfigurasi Script
Edit bagian konfigurasi di `main.py` sesuai dengan setup Anda:

```python
# Konfigurasi
lmstudio_url = "http://localhost:1234/v1/chat/completions"
image_dir = r"C:\path\to\your\test\folder"  # Sesuaikan path
ground_truth_file = os.path.join(image_dir, "ground_truth.csv")
model_name = "bakllava1-mistralllava-7b"  # Sesuaikan nama model
```

### ▶️ 5. Jalankan Program Utama
Untuk menjalankan inferensi dan evaluasi otomatis:

```bash
python main.py
```

**✅ Fungsi program:**
* Membaca gambar dari folder `test/image/`
* Melakukan preprocessing gambar (grayscale, contrast enhancement, histogram equalization, blur)
* Mengirim permintaan OCR ke LM Studio dengan model VLM
* Menghitung CER untuk setiap gambar
* Memilih prediksi terbaik dari berbagai varian preprocessing
* Menyimpan hasil ke `ocr_results.csv`

---

## 📊 Output dan Hasil

### 📄 File Output: `ocr_results.csv`
**Contoh hasil:**

| image | ground_truth | prediction | CER_score |
|-------|-------------|------------|-----------|
| test001_1.jpg | B9140BCD | B9140BCD | 0.0000 |
| test001_2.jpg | B2407UZO | B2407UZ | 0.1250 |
| test003_3.jpg | D1234ABC | D1284ABC | 0.1250 |

### 📈 Metrik Evaluasi
Program akan menampilkan summary hasil evaluasi:
* **Total Images Processed**: Jumlah gambar yang diproses
* **Average CER**: Rata-rata Character Error Rate
* **Accuracy**: Persentase prediksi yang benar 100%
* **Error Analysis**: Detail substitusi, deletion, dan insertion

### 🧮 Formula CER (Character Error Rate)
```
CER = (Substitutions + Deletions + Insertions) / Total_Characters_in_Ground_Truth
```

**Contoh perhitungan:**
* Ground Truth: `B2407UZO` (8 karakter)
* Prediction: `B2407UZ` (7 karakter)
* Deletions: 1 (karakter 'O' hilang)
* CER = (0 + 1 + 0) / 8 = 0.125

---

## 🛠️ Fitur Utama

### 🖼️ Image Preprocessing
Script melakukan 4 varian preprocessing untuk meningkatkan akurasi:
1. **Grayscale + Resize**: Konversi ke abu-abu dan resize ke 224x224
2. **Contrast Enhancement**: Peningkatan kontras 2x
3. **Histogram Equalization**: Pemerataan histogram untuk pencahayaan
4. **Gaussian Blur**: Sedikit blur untuk mengurangi noise

### 🎯 Best Result Selection
Sistem memilih prediksi dengan CER terendah dari semua varian preprocessing.

### 📝 Comprehensive Logging
Program menampilkan progress detail untuk setiap gambar yang diproses.

---

## 🔧 Requirements

```python
# Library yang dibutuhkan:
import os
import json
import csv
import base64
import requests
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import difflib
```

**Install dependencies:**
```bash
pip install pillow opencv-python numpy requests
```

---

## 🚨 Troubleshooting

**Jika mengalami error:**
1. **LM Studio tidak berjalan**: Pastikan server LM Studio aktif di `localhost:1234`
2. **Model tidak ter-load**: Cek apakah model sudah di-load dengan benar di LM Studio
3. **Path dataset salah**: Periksa kembali path folder `test/`
4. **Format ground truth**: Pastikan file CSV memiliki kolom `image` dan `ground_truth`

---

## 📧 Kontak
**Nama**: Zain Daffa  
**Mata Kuliah**: Computer Vision - UAS RE604  
**Repository**: [UAS-RE604-COMPUTER-VISION](https://github.com/Zain-Daffa/UAS-RE604-COMPUTER-VISION)

---
*Dibuat untuk keperluan Ujian Akhir Semester Computer Vision* 🎓
