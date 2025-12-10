# 🚗 Implementasi Sistem Automatic Number Plate Recognition (ANPR) Indonesia

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)](https://ultralytics.com/)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-yellow)](https://github.com/JaidedAI/EasyOCR)

## Anggota Kelompok
1. Hana Azizah Nurhadi - 5025231134
2. Aqila Zahira Naia Puteri Arifin - 5025231138
3. Muhammad Dawanis Baihaqi - 5025231177

## 📝 Deskripsi 

Proyek ini merupakan **Final Project** implementasi **Sistem Automatic Number Plate Recognition (ANPR)** berbasis **Deep Learning**, yang dirancang secara spesifik untuk mengenali dan memverifikasi plat nomor kendaraan Indonesia. Fokus utama dari proyek ini adalah meningkatkan **Robustness** sistem terhadap tantangan pada skenario *real-world*, seperti variasi pencahayaan, *shadow*, dan resolusi gambar yang rendah.

Metodologi yang diusulkan mengintegrasikan dua teknologi utama: **YOLOv8** untuk tahap *Object Detection* dan **EasyOCR** untuk tahap *Optical Character Recognition* (OCR). Seluruh sistem diimplementasikan dalam sebuah **Streamlit App** yang bersifat *interactive* bagi pengguna.

---

## ✨ Fitur 

1.  **Pendekatan Multi-Model untuk Deteksi (YOLOv8):**
    * Sistem ini memanfaatkan **dua model YOLOv8n** yang telah melalui proses *training* khusus, yaitu **`front.pt`** dan **`side.pt`**.
    * Penggunaan model terpisah ini bertujuan untuk mengoptimalkan *Performance* dalam melokalisasi *Bounding Box* plat nomor dari berbagai *view angle* (sudut pandang depan dan samping).

2.  **Modul Interaktif Preprocessing dan Tuning:**
    * Kami menyediakan *interface* Streamlit yang memungkinkan **Manual Tuning** terhadap parameter *Image Processing* ketika *Auto OCR* memberikan hasil yang tidak akurat. Fitur ini krusial untuk mitigasi *Noise*.
    * **Manual Thresholding:** Penyesuaian batas binarisasi untuk mengoptimalkan kontras teks.
    * **Gamma Correction:** Teknik *enhancement* untuk mencerahkan area *under-exposed* atau berbayangan.
    * **Smart Cropping:** Fitur untuk menghilangkan *frame* atau *date stamp* pada plat yang dapat mengganggu *Accuracy* OCR.

3.  **Visualisasi Debug:**
    * Tersedia **Debug View** yang menampilkan *crop* gambar hasil *preprocessing*. Hal ini membantu pengguna memverifikasi kualitas input yang diterima oleh mesin OCR.

---

## 📖 Prosedur Penggunaan Aplikasi Web

Aplikasi *live deployment* dapat diakses melalui *link* berikut: [Streamlit Cloud link](https://anpr-pcv.streamlit.app)

1.  **Input Data:** Lakukan *Upload* gambar kendaraan (format JPG/PNG) melalui *interface*.
2.  **Eksekusi:** Klik tombol "Start Detection" untuk memulai alur deteksi plat nomor dan *initial reading*.
3.  **Tuning (Opsional):** Jika hasil pembacaan gagal atau tidak valid, akses **Sidebar** dan lakukan **Tuning** parameter *preprocessing* (Threshold, Crop Frame, Gamma) secara *real-time*.
4.  **Verifikasi Hasil:** Hasil akan ditandai dengan kotak:
    * **Hijau:** *Output* plat terdeteksi dan formatnya valid.
    * **Kuning/Merah:** Deteksi berhasil, namun hasil *reading* OCR memerlukan *Tuning* lebih lanjut.

---

## 💻 Panduan Instalasi Lokal

Untuk menjalankan proyek ini di komputer lokal, ikuti *step-by-step* instalasi di bawah ini:

### Prasyarat Sistem

* Python versi 3.8 atau yang lebih tinggi.
* Git harus sudah terinstal.

### Tahapan Instalasi

1.  **Clone Repositori:**
    ```bash
    git clone [https://github.com/citylighxts/ANPR.git](https://github.com/citylighxts/ANPR.git)
    cd ANPR
    ```
2.  **Inisiasi dan Aktivasi Virtual Environment:**
    ```bash
    python3 -m venv venv
    
    # Linux/macOS
    source venv/bin/activate
    
    # Windows
    venv\Scripts\activate
    ```
3.  **Instalasi Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Deployment Aplikasi:**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Teknologi

| Komponen | Teknologi | Keterangan |
| :--- | :--- | :--- |
| **Bahasa Pemrograman** | Python 3.9+ | *Core Language* untuk seluruh *Logic* sistem. |
| **Antarmuka Pengguna** | Streamlit | *Framework* yang digunakan untuk *rapid deployment* antarmuka web interaktif. |
| **Deteksi Objek** | YOLOv8 (Ultralytics) | *State-of-the-art model* untuk *detection* lokasi plat nomor. |
| **Pengenalan Teks** | EasyOCR | Pustaka OCR yang efisien untuk mengekstraksi karakter dari plat. |
| **Pemrosesan Gambar** | OpenCV | Pustaka *Computer Vision* untuk operasi *Preprocessing* seperti Thresholding dan *Morphology*. |
