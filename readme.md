# Prediksi Harga Saham Bank Central Asia (BCA) Menggunakan Support Vector Regression (SVR)

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi harga saham Bank Central Asia (BCA) menggunakan algoritma machine learning Support Vector Regression (SVR). Prediksi dilakukan dengan menganalisis data historis harga saham dan mengidentifikasi pola yang dapat digunakan untuk memperkirakan harga di masa depan.

## Fitur Utama

- Prediksi harga saham BCA menggunakan SVR
- Visualisasi data historis dan hasil prediksi
- Evaluasi performa model dengan berbagai metrik
- Analisis fitur yang berpengaruh terhadap harga saham
- Dashboard interaktif untuk menampilkan hasil prediksi

## Teknologi yang Digunakan

- Python 3.x
- Scikit-learn untuk implementasi SVR
- Pandas untuk manipulasi dan analisis data
- NumPy untuk komputasi numerik
- Matplotlib dan Seaborn untuk visualisasi data
- Streamlit untuk dashboard interaktif
- Yfinance untuk pengambilan data saham

## Struktur Proyek

```
prediksi-saham-BCA-SVR/
├── img/                         # menyimpan Gambar
├── bca_prediction_svm.py        # aplikasi
├── requirements.txt             # Dependensi proyek
└── README.md                    # Dokumentasi proyek
```

## Cara Instalasi

1. Clone repository ini:

   ```
   git clone https://github.com/ahmadseloabadi/prediksi-saham-BCA-SVR.git
   ```

2. Masuk ke direktori proyek:

   ```
   cd prediksi-saham-BCA-SVR
   ```

3. Instal dependensi yang diperlukan:
   ```
   pip install -r requirements.txt
   ```

## Cara Penggunaan

### Menjalankan Dashboard

1. Jalankan aplikasi Streamlit melalui command line:

   ```
   streamlit run bca_prediction_svm.py
   ```

2. Buka browser dan akses `http://localhost:xxxx`

3. Gunakan dashboard untuk:
   - Melihat data historis saham BCA
   - Melihat hasil prediksi harga saham
   - Menyesuaikan parameter model
   - Mengunduh laporan prediksi

## Metode Prediksi

### Support Vector Regression (SVR)

Support Vector Regression adalah algoritma machine learning yang menggunakan prinsip Support Vector Machine untuk masalah regresi. SVR bekerja dengan mencari fungsi dengan deviasi maksimum ε dari target sebenarnya untuk semua data pelatihan, dan pada saat yang sama mencari fungsi yang sedatar mungkin.

Dalam proyek ini, SVR digunakan untuk memprediksi harga saham BCA dengan memanfaatkan fitur-fitur seperti:

- Harga pembukaan (open)
- Harga tertinggi (high)
- Harga terendah (low)

### Pengolahan Data dan Feature Engineering

1. **Pengumpulan Data**: Data historis saham BCA diambil menggunakan API Yahoo Finance
2. **Pembersihan Data**: Menangani missing values dan outliers
3. **Normalisasi**: Menskalakan fitur menggunakan StandardScaler untuk meningkatkan performa model
4. **Train-Test Split**: Membagi data menjadi set pelatihan dan pengujian

## Dataset

Dataset terdiri dari data historis harga saham BCA (BBCA.JK) selama beberapa tahun terakhir, dengan fitur:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Data diambil menggunakan API Yahoo Finance dan diproses untuk analisis lebih lanjut.

## Hasil dan Evaluasi

Performa model SVR dievaluasi menggunakan metrik:

- Mean Absolute Error (MAE): 245.67
- Mean Squared Error (MSE): 97234.89
- Root Mean Squared Error (RMSE): 311.82
- R-squared (R²): 0.89

Visualisasi hasil prediksi menunjukkan bahwa model cukup baik dalam mengikuti tren harga saham BCA, dengan beberapa penyimpangan pada periode volatilitas tinggi.

## Limitasi dan Pengembangan Lebih Lanjut

### Limitasi

- Model saat ini hanya fokus pada data historis harga saham dan indikator teknikal
- Tidak memperhitungkan faktor eksternal seperti berita ekonomi, kebijakan perusahaan, atau sentimen pasar
- Akurasi prediksi menurun pada periode volatilitas tinggi

### Pengembangan Lebih Lanjut

- Integrasi analisis sentimen dari berita dan media sosial
- Implementasi model ensemble atau deep learning untuk meningkatkan akurasi
- Penambahan fitur makroekonomi sebagai variabel prediktor
- Pengembangan sistem alert untuk pergerakan harga saham yang signifikan

## Disclaimer

Proyek ini dibuat untuk tujuan pendidikan dan penelitian. Prediksi harga saham memiliki risiko dan ketidakpastian yang tinggi. Prediksi yang dihasilkan tidak dimaksudkan sebagai saran investasi. Selalu lakukan analisis dan pertimbangan sendiri sebelum membuat keputusan investasi.

## Kontribusi

Kontribusi untuk pengembangan proyek ini sangat diterima. Silakan fork repository ini dan kirimkan pull request dengan perubahan yang Anda usulkan.

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

## Kontak

Ahmad Selo Abadi - [ahmadseloabadi@gmail.com](mailto:ahmadseloabadi@gmail.com)

Link Proyek: [https://github.com/ahmadseloabadi/prediksi-saham-BCA-SVR](https://github.com/ahmadseloabadi/prediksi-saham-BCA-SVR)
