# RAM Price Prediction: Machine Learning Regresi Prediksi Harga per GB

Proyek Machine Learning untuk memprediksi harga RAM per gigabyte (`price_per_gb`) dan mengestimasi harga total kit memori berdasarkan spesifikasi teknis perangkat keras dan indikator rantai pasok global semikonduktor.

---

## 1. Pertanyaan Penelitian

> **"Seberapa baik spesifikasi RAM dan indikator pasokan dapat memprediksi harga per GB?"**

### Temuan Utama:
- **Akurasi Model**: Model non-linear **Random Forest Regressor** berhasil menjelaskan **~95% varians harga** ($R^2 = 0.9503$) dengan kesalahan rata-rata ($MAE$) sebesar **$7.17 / GB** pada pengujian kronologis (*out-of-time test set*).
- **Perbandingan Model Linier vs Non-Linier**: Model linier (Linear Regression & Ridge) hanya mencapai $R^2 \approx 0.64$ ($MAE \approx \$15.51 / GB$). Hal ini membuktikan bahwa interaksi antara spesifikasi perangkat keras (generasi, frekuensi, latensi) dan indikator rantai pasok makro bersifat sangat non-linear.
- **Prediktor Dominan**: Indikator rantai pasok semikonduktor (**`fab_utilization_rate`**, **`global_inventory_weeks`**, **`gpu_hbm_trend_gb`**) bersama dengan spesifikasi perangkat keras (**`generation`**, **`speed_mts`**, **`capacity_gb`**) memiliki pengaruh terbesar terhadap lonjakan harga memori.

---

## 2. Struktur Direktori Proyek

```text
ram-price-prediction/
├── data/
│   └── ram_dataset.csv               # Salinan dataset RAM (10.000 baris x 28 kolom)
├── notebooks/
│   └── ram_price_analysis.ipynb      # Notebook analisis lengkap dan tereksekusi
├── src/
│   ├── train.py                      # Skrip pelatihan pipeline 4 model regresi
│   └── predict.py                    # CLI inferensi prediksi harga per GB & total
├── models/
│   ├── best_model.joblib             # Pipeline Scikit-Learn model terbaik terlatih
│   └── feature_meta.joblib           # Metadata fitur numerik & kategorikal
├── reports/
│   ├── evaluation_metrics.csv        # Tabel ringkasan metrik MAE, RMSE, R2
│   └── figures/                      # Grafik visualisasi hasil evaluasi
│       ├── model_comparison.png      # Bar chart komparasi 4 model
│       ├── actual_vs_predicted.png   # Scatter plot aktual vs prediksi
│       ├── residuals.png             # Analisis residual
│       └── feature_importance.png    # Top 20 fitur paling berpengaruh
├── sample_input.json                 # Contoh file input untuk pengujian prediksi
├── requirements.txt                  # Dependensi minimal pustaka Python
└── README.md                         # Dokumentasi lengkap proyek
```

---

## 3. Strategi Fitur & Pencegahan Data Leakage

Untuk memastikan model benar-benar dapat digunakan di dunia nyata dan tidak mengalami bias optimisme semu:

### Fitur yang Dikecualikan:
1. **`price_usd`**: Mengandung kebocoran langsung karena $\text{price\_usd} / \text{capacity\_gb} = \text{price\_per\_gb}$.
2. **`bandwidth_per_dollar`, `latency_value_index`, `speed_premium_ratio`**: Rasio ini mengikutsertakan harga pasar dalam formulasinya.
3. **`price_status`**: Label kualitatif (misal `'Shortage-Inflated'`) yang ditentukan berdasarkan deviasi harga pasar.
4. **`kit_id` & `model_name`**: Identifier acak dan teks nama model dengan kardinalitas tinggi yang sudah terwakili secara spesifik oleh kolom brand, generasi, dan frekuensi.
5. **`timing_string`**: Format teks yang redundan dengan `cas_latency` dan `true_latency_ns`.

### Fitur yang Digunakan:
- **Numerik (9 fitur)**: `capacity_gb`, `module_count`, `speed_mts`, `cas_latency`, `voltage`, `true_latency_ns`, `global_inventory_weeks`, `fab_utilization_rate`, `gpu_hbm_trend_gb`.
- **Kategorikal (9 fitur)**: `brand`, `region`, `generation`, `form_factor`, `is_ecc`, `die_manufacturer`, `die_type`, `market_segment`, `recommended_usage`.

### Pembagian Data Kronologis (*Chronological Split*):
- Data diurutkan berdasarkan kolom `timestamp`.
- **Train Set (80%)**: 8.000 baris awal (Januari 2024 s.d. Mei 2026).
- **Test Set (20%)**: 2.000 baris terbaru (Mei 2026 s.d. Januari 2027).
- **Penanganan Tanggal Masa Depan**: Terdapat 682 baris bertanggal setelah 19 Oktober 2026. Baris ini tetap dipertahankan utuh dan jatuh ke dalam Test Set secara alami sebagai uji generalisasi proyeksi masa depan (*future out-of-time forecasting*).
- Preprocessing (`ColumnTransformer`) dipelajari (**fit**) **hanya** pada data latih.

---

## 4. Instalasi Dependensi

Proyek ini dirancang ringan agar dapat dijalankan secara efisien pada laptop Windows dengan spesifikasi standar.

1. Buka PowerShell atau Command Prompt di folder proyek `ram-price-prediction`:
   ```powershell
   cd "ram-price-prediction"
   ```

2. (Opsional) Buat dan aktifkan virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Pasang dependensi dari `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 5. Cara Menjalankan Pelatihan Model (`train.py`)

Skrip pelatihan akan:
- Memuat dataset dan melakukan validasi konsistensi matematis target.
- Membagi data secara kronologis (80% Train, 20% Test).
- Membangun `Pipeline` Scikit-Learn dengan `ColumnTransformer` (median imputer untuk numerik, mode imputer + One-Hot Encoding untuk kategorikal).
- Melatih dan membandingkan 4 model: `DummyRegressor`, `LinearRegression`, `Ridge`, dan `RandomForestRegressor`.
- Menyimpan metrik evaluasi ke `reports/evaluation_metrics.csv`.
- Menyimpan grafik komparasi, actual vs predicted, residual, dan feature importance ke `reports/figures/`.
- Menyimpan pipeline model terbaik ke `models/best_model.joblib`.

Perintah eksekusi:
```powershell
python src/train.py
```

Opsi argumen:
```powershell
# Melatih dengan path CSV khusus:
python src/train.py --csv data/ram_dataset.csv
```

---

## 6. Cara Menjalankan Prediksi (`predict.py`)

Skrip inferensi memuat model tersimpan dan menghitung perkiraan harga per GB serta harga total kit RAM.

Rumus harga total:
$$\text{Estimated Total Price} = \text{Predicted Price per GB} \times \text{capacity\_gb}$$

### Mode 1: Menggunakan File JSON (Direkomendasikan)
Gunakan file input yang sudah disiapkan atau buat file JSON baru:
```powershell
python src/predict.py --file sample_input.json
```

Contoh output:
```text
+------------------------------------------+
|          PREDICTION RESULTS              |
+------------------------------------------+
|  Predicted price_per_gb : $     18.15    |
|  Capacity (GB)          :         32     |
|  Estimated total price  : $    580.76    |
+------------------------------------------+

  Formula: $18.15/GB x 32 GB = $580.76
```

### Mode 2: Mode Interaktif
Jalankan skrip tanpa argumen, program akan meminta masukan spesifikasi baris per baris:
```powershell
python src/predict.py
```

### Mode 3: String JSON Langsung
```powershell
python src/predict.py --json "{\"brand\":\"Kingston\",\"region\":\"North America\",\"generation\":\"DDR5\",\"form_factor\":\"DIMM\",\"capacity_gb\":32,\"module_count\":2,\"is_ecc\":\"False\",\"speed_mts\":6000,\"cas_latency\":36,\"voltage\":1.35,\"die_manufacturer\":\"Samsung\",\"die_type\":\"B-Die\",\"true_latency_ns\":12.0,\"market_segment\":\"Consumer\",\"recommended_usage\":\"Gaming (AAA)\",\"global_inventory_weeks\":10.0,\"fab_utilization_rate\":0.72,\"gpu_hbm_trend_gb\":160.0}"
```

---

## 7. Cara Menjalankan Notebook (`ram_price_analysis.ipynb`)

1. Buka folder `ram-price-prediction` di VS Code:
   ```powershell
   code .
   ```
2. Buka berkas `notebooks/ram_price_analysis.ipynb`.
3. Pilih Python kernel (Python 3.12).
4. Klik **Run All** atau jalankan setiap sel berurutan dari atas ke bawah.
5. Notebook sudah dilengkapi output teks terstruktur, tabel statistik, dan visualisasi grafik terintegrasi.

---

## 8. Ringkasan Hasil Evaluasi Model Aktual

Berikut adalah metrik performa objektif pada **2.000 sampel data uji kronologis** (Mei 2026 s.d. Januari 2027):

| Model | MAE ($/GB) | RMSE ($/GB) | $R^2$ Score | Catatan Evaluasi |
|---|---|---|---|---|
| **RandomForestRegressor** | **$7.1703** | **$10.3967** | **0.9503** | **Pemenang (Model Terbaik)**: Menangkap pola non-linear dan interaksi supply shock secara akurat |
| **LinearRegression** | $15.5064 | $27.9618 | 0.6402 | Cukup baik namun gagal menangkap korelasi non-linear pada periode kelangkaan ekstrem |
| **Ridge Regression** | $15.5160 | $27.9891 | 0.6395 | Performa serupa OLS linier dengan sedikit peredaman koefisien (L2 penalty) |
| **DummyRegressor (Mean)** | $35.5626 | $57.3484 | -0.5136 | Baseline naif; performa $R^2 < 0$ menunjukkan target harga bergeser jauh dari rata-rata historis |

> **Catatan**: Seluruh metrik berasal dari pengujian kronologis murni (*out-of-time test set*), bukan dari data pelatihan (*train set*). Tidak ada nilai metrik yang direkayasa.
