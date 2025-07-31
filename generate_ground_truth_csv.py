import os
import csv

# KONFIGURASI FOLDER
dataset_folder = r"C:\Users\ASUS_TUF_GAMING\Documents\Dataset\test"
output_csv_path = os.path.join(dataset_folder, "ground_truth.csv")

# MAPPING ID KELAS KE KARAKTER
label_map = {
    0: '0',  1: '1',  2: '2',  3: '3',  4: '4',
    5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

# PROSES SETIAP FILE .txt
data_rows = []

for filename in os.listdir(dataset_folder):
    if filename.endswith(".txt") and filename != "classes.txt":
        txt_path = os.path.join(dataset_folder, filename)
        image_name = filename.replace(".txt", ".jpg")

        # Baca isi file anotasi
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        character_boxes = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id = int(parts[0])
                x_center = float(parts[1])
                character = label_map.get(class_id, '?')  # gunakan '?' jika class_id tidak ditemukan
                character_boxes.append((x_center, character))

        # Urutkan karakter dari kiri ke kanan (berdasarkan posisi x_center)
        character_boxes.sort(key=lambda item: item[0])

        # Gabungkan karakter menjadi string plat nomor
        plate_number = ''.join([char for _, char in character_boxes])

        # Simpan hasil
        data_rows.append([image_name, plate_number])

# SIMPAN KE FILE CSV
with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "ground_truth"]) 
    writer.writerows(data_rows)

print(f"✅ ground_truth.csv berhasil dibuat di: {output_csv_path}")
