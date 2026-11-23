# 🚆 Prediksi Pilihan Moda Transportasi (Kasus 9)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

Repositori ini berisi implementasi proyek **Ujian Tengah Semester (UTS) Mata Kuliah Intelegensi Buatan** untuk **Kasus 9: Pilihan Moda Transportasi** menggunakan *Machine Learning* klasifikasi multikelas berbasis `scikit-learn` dan antarmuka web interaktif berbasis **Streamlit**.

---

## 👤 Identitas Mahasiswa

* **Nama:** Radya Anantia Divena
* **NIM:** `23552011415` (Jumlah digit = 29 -> 29 (mod10) = 9 -> (Kasus 9)
* **Kelas:** TIF RP 23D — Teknik Informatika, Universitas Teknologi Bandung
* **Dosen Pengampu:** Muhammad Ikhwan Fathulloh, S.Kom.

---

## 📊 Ringkasan Kasus & Dataset

Dataset `kasus09_mode_choice.csv` berasal dari survei perjalanan antar-kota (Sydney, Canberra, dan Melbourne) dengan total **840 baris dan 9 kolom**, yang merepresentasikan **210 individu** dengan **4 alternatif moda transportasi** (Moda 1–4).

### Tugas Utama & Metodologi

1. **Audit Data & Transformasi Format (*Long-to-Wide*):**
   * Mengonversi data dari format *long* (4 baris per individu) menjadi format *wide* (1 baris per individu, total 210 baris) agar seluruh atribut biaya dan waktu dari keempat moda dapat dibandingkan secara simultan dalam satu vektor fitur.
   * Mengekstrak target `target_mode` dari kolom `mode` saat `choice == 1` serta memvalidasi bahwa setiap individu memiliki tepat satu pilihan.
   * Membuang kolom `individual` (ID unik) dan `choice` (sumber target) untuk mencegah *data leakage*.

2. **Rekayasa Fitur (*Feature Engineering*):**
   * Menambahkan 16 fitur baru berbasis domain transportasi dari 18 fitur mentah (total 34 fitur):
     * `total_time_mode{1..4}`: Total waktu perjalanan (`ttme + invt`).
     * `invc_per_person_mode{1..4}`: Biaya dalam kendaraan per anggota rombongan (`invc / psize`).
     * `gc_diff_min_mode{1..4}`: Selisih biaya gabungan terhadap moda termurah.
     * `time_diff_min_mode{1..4}`: Selisih total waktu perjalanan terhadap moda tercepat.

3. **Pelatihan & Evaluasi Model:**
   * Pembagian data latih (`80%`, 168 individu) dan data uji (`20%`, 42 individu) secara *stratified* dilakukan sebelum proses *imputasi* dan *scaling*.
   * Membandingkan **Logistic Regression** dan **Random Forest** menggunakan `Pipeline` scikit-learn dan **5-Fold Stratified Cross-Validation**.

---

## 📁 Struktur Direktori Proyek

```text
UTS_Kasus09_ModeChoice/
├── dataset/
│   └── kasus09_mode_choice.csv       # Dataset survei pilihan moda transportasi
├── models/
│   └── model_mode_choice.joblib      # Pipeline model terbaik & metadata evaluasi
├── notebook/
│   └── kasus09_colab.ipynb           # Notebook Google Colab (training & deploy Streamlit)
├── fitur.py                          # Modul audit, reshaping long-to-wide, & feature engineering
├── latih.py                          # Skrip training, cross-validation, evaluasi test, & simpan model
├── app.py                            # Aplikasi web interaktif Streamlit
└── README.md                         # Dokumentasi repositori
```

---

## 📈 Ringkasan Performa Model

### 1. Perbandingan *5-Fold Stratified Cross-Validation* (Data Train)

| Algoritma Model | CV Accuracy (Mean ± Std) | CV Macro F1 (Mean ± Std) | Status |
| :--- | :---: | :---: | :---: |
| Logistic Regression (Multinomial) | `0.7865 ± 0.0716` | `0.7797 ± 0.0922` | Pembanding |
| **Random Forest (200 Estimators)** | **`0.9048 ± 0.0113`** | **`0.9043 ± 0.0360`** | **Model Terpilih** |

### 2. Hasil Evaluasi Akhir pada Data Test (1x Evaluasi)

* **Test Accuracy:** `92.86%` (39 dari 42 sampel uji diprediksi tepat)
* **Test Macro F1-Score:** `0.9357`
* **Test Weighted F1-Score:** `0.9288`

---

## 🚀 Cara Menjalankan Proyek Secara Lokal

1. **Kloning Repositori & Masuk ke Direktori Proyek**
   ```bash
   git clone https://github.com/<username-github-anda>/<nama-repositori>.git
   cd <nama-repositori>
   ```

2. **Instal Dependensi**
   ```bash
   pip install scikit-learn pandas numpy joblib streamlit
   ```

3. **Latih Ulang & Simpan Model (Opsional)**
   ```bash
   python latih.py
   ```
   > Perintah ini akan membaca `dataset/kasus09_mode_choice.csv`, melakukan *cross-validation*, mengevaluasi data uji, dan menyimpan artefak terbaru ke `models/model_mode_choice.joblib`.

4. **Jalankan Aplikasi Streamlit**
   ```bash
   streamlit run app.py
   ```
   > Buka alamat lokal yang muncul di terminal (umumnya `http://localhost:8501`) untuk melakukan simulasi prediksi dan melihat rincian metrik evaluasi model.
