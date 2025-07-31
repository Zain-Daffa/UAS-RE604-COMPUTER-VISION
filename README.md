# Indonesian License Plate Recognition — UAS Computer Vision

 Deskripsi Singkat
Proyek ini bertujuan untuk melakukan inferensi OCR (Optical Character Recognition) pada gambar plat nomor kendaraan menggunakan Visual Language Model (VLM) dari LM Studio.
Evaluasi hasil prediksi dilakukan dengan menghitung Character Error Rate (CER), sehingga dapat diketahui tingkat akurasi sistem secara objektif.
📁 Struktur Folder
UAS-RE604-COMPUTER-VISION/
├── test/
│   ├── image/           # 📸 Gambar plat nomor (.jpg, .png)
│   ├── label/           # 📝 Label ground truth (.txt)
│   └── ground_truth.csv # 📄 Data hasil penggabungan dari .txt dan jpg
│
├── main.py              # 🚀 Script utama untuk inferensi dan evaluasi VLM
├── ocr_results.csv      # 📊 Hasil akhir: image, ground_truth, prediction, CER_score
└── README.md            # 📖 Dokumentasi proyek
🚀 Langkah Eksekusi
🧾 1. Persiapan Dataset

Siapkan folder test yang berisi:

image/ → gambar plat nomor (.jpg, .png, .bmp)
label/ → label plat nomor dalam format .txt (opsional)


Jika memiliki file ground_truth.csv, letakkan di dalam folder test/

🤖 2. Setup LM Studio
Pastikan LM Studio telah terinstall dan model BakLLaVA telah di-load:

Buka LM Studio
Load model bakllava1-mistralllava-7b atau model VLM lainnya
Jalankan server lokal
📡 Server akan berjalan di: http://localhost:1234

⚙️ 3. Konfigurasi Script
Edit bagian konfigurasi di main.py sesuai dengan setup Anda:
python# Konfigurasi
lmstudio_url = "http://localhost:1234/v1/chat/completions"
image_dir = r"C:\path\to\your\test\folder"  # Sesuaikan path
ground_truth_file = os.path.join(image_dir, "ground_truth.csv")
model_name = "bakllava1-mistralllava-7b"  # Sesuaikan nama model
▶️ 4. Jalankan Program Utama
Untuk menjalankan inferensi dan evaluasi otomatis:
bashpython main.py
✅ Fungsi program:

Membaca gambar dari folder test/image/
Melakukan preprocessing gambar (grayscale, contrast enhancement, histogram equalization, blur)
Mengirim permintaan OCR ke LM Studio dengan model VLM
Menghitung CER untuk setiap gambar
Memilih prediksi terbaik dari berbagai varian preprocessing
Menyimpan hasil ke ocr_results.csv

📊 Output dan Hasil
📄 File Output: ocr_results.csv
Contoh hasil:
imageground_truthpredictionCER_scoretest001_1.jpgB9140BCDB9140BCD0.0000test001_2.jpgB2407UZOB2407UZ0.1250test003_3.jpgD1234ABCD1284ABC0.1250
📈 Metrik Evaluasi
Program akan menampilkan summary hasil evaluasi:

Total Images Processed: Jumlah gambar yang diproses
Average CER: Rata-rata Character Error Rate
Accuracy: Persentase prediksi yang benar 100%
Error Analysis: Detail substitusi, deletion, dan insertion

🧮 Formula CER (Character Error Rate)
CER = (Substitutions + Deletions + Insertions) / Total_Characters_in_Ground_Truth
Contoh perhitungan:

Ground Truth: B2407UZO (8 karakter)
Prediction: B2407UZ (7 karakter)
Deletions: 1 (karakter 'O' hilang)
CER = (0 + 1 + 0) / 8 = 0.125

🛠️ Fitur Utama
🖼️ Image Preprocessing
Script melakukan 4 varian preprocessing untuk meningkatkan akurasi:

Grayscale + Resize: Konversi ke abu-abu dan resize ke 224x224
Contrast Enhancement: Peningkatan kontras 2x
Histogram Equalization: Pemerataan histogram untuk pencahayaan
Gaussian Blur: Sedikit blur untuk mengurangi noise

🎯 Best Result Selection
Sistem memilih prediksi dengan CER terendah dari semua varian preprocessing.
