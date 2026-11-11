# -*- coding: utf-8 -*-
"""
===============================================================================
FILE UTAMA : identifikasi_9_dataset.py
DESKRIPSI  : Program Identifikasi dan Validasi 9 Dataset Klasifikasi
             (Binary, Multiclass, dan Multilabel Classification)
             menggunakan Python dan Library Pandas.
CARA PAKAI : 1. Letakkan file 'identifikasi_9_dataset.py' ini:
                - Di luar (sejajar dengan) folder 'datasets', ATAU
                - Di dalam folder 'datasets' (sejajar dengan 'archive', 'biner', dll.)
             2. Jalankan melalui Terminal VS Code di Windows:
                python identifikasi_9_dataset.py
===============================================================================
"""

import os
import sys
import pandas as pd

# Mengatur tampilan Pandas agar seluruh kolom dan lebar tabel terlihat jelas di terminal
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 1000)


# =============================================================================
# 1. KONFIGURASI LOKASI FILE DAN KOLOM LABEL (SESUAI ISI DATASETS.ZIP ANDA)
# =============================================================================
# Seluruh lokasi file dan nama kolom target di bawah ini sudah disesuaikan
# 100% dengan struktur folder dan isi file di dalam datasets.zip Anda.

KONFIGURASI_DATASET = {
    "binary": [
        {
            "kode": "A1",
            "nama_dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "asal_dataset": "UCI Machine Learning Repository / Kaggle",
            "url_sumber": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "lokasi_file": "datasets/biner/breast-cancer-wisconsin-data.csv",
            "nama_file_alternatif": ["breast-cancer-wisconsin-data.csv", "wdbc.data"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": "diagnosis"
        },
        {
            "kode": "A2",
            "nama_dataset": "Rice (Cammeo and Osmancik)",
            "asal_dataset": "UCI Machine Learning Repository",
            "url_sumber": "https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik",
            "lokasi_file": "datasets/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff",
            "nama_file_alternatif": ["Rice_Cammeo_Osmancik.arff"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": "Class"
        },
        {
            "kode": "A3",
            "nama_dataset": "Iranian Churn (Telecom Customer Churn)",
            "asal_dataset": "UCI Machine Learning Repository",
            "url_sumber": "https://archive.ics.uci.edu/dataset/563/iranian+churn",
            "lokasi_file": "datasets/iranian+churn+dataset/Customer Churn.csv",
            "nama_file_alternatif": ["Customer Churn.csv"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": "Churn"
        }
    ],
    "multiclass": [
        {
            "kode": "B1",
            "nama_dataset": "Wine",
            "asal_dataset": "UCI Machine Learning Repository",
            "url_sumber": "https://archive.ics.uci.edu/dataset/109/wine",
            "lokasi_file": "datasets/wine/wine.data",
            "nama_file_alternatif": ["wine.data"],
            "tanpa_header": True,
            "nama_kolom_manual": [
                "class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash",
                "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",
                "Proanthocyanins", "Color_intensity", "Hue",
                "0D280_0D315_of_diluted_wines", "Proline"
            ],
            "kolom_target": "class"
        },
        {
            "kode": "B2",
            "nama_dataset": "Car Evaluation",
            "asal_dataset": "UCI Machine Learning Repository",
            "url_sumber": "https://archive.ics.uci.edu/dataset/19/car+evaluation",
            "lokasi_file": "datasets/car+evaluation/car.data",
            "nama_file_alternatif": ["car.data"],
            "tanpa_header": True,
            "nama_kolom_manual": [
                "buying", "maint", "doors", "persons", "lug_boot", "safety", "class"
            ],
            "kolom_target": "class"
        },
        {
            "kode": "B3",
            "nama_dataset": "Dry Bean",
            "asal_dataset": "UCI Machine Learning Repository",
            "url_sumber": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
            "lokasi_file": "datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx",
            "nama_file_alternatif": ["Dry_Bean_Dataset.xlsx", "Dry_Bean_Dataset.arff"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": "Class"
        }
    ],
    "multilabel": [
        {
            "kode": "C1",
            "nama_dataset": "Yeast Multi-Label Dataset (yeast)",
            "asal_dataset": "OpenML Benchmark Repository (ID: 40597)",
            "url_sumber": "https://www.openml.org/d/40597",
            "lokasi_file": "datasets/file2754771351f4.arff",
            "nama_file_alternatif": ["file2754771351f4.arff", "yeast.arff", "yeast.csv"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": [
                "Class1", "Class2", "Class3", "Class4", "Class5",
                "Class6", "Class7", "Class8", "Class9", "Class10",
                "Class11", "Class12", "Class13", "Class14"
            ]
        },
        {
            "kode": "C2",
            "nama_dataset": "Natural Scene Image Multi-Label Dataset (image)",
            "asal_dataset": "OpenML Benchmark Repository (ID: 41468)",
            "url_sumber": "https://www.openml.org/d/41468",
            "lokasi_file": "datasets/file3a6c46689b2.arff",
            "nama_file_alternatif": ["file3a6c46689b2.arff", "image.arff", "image.csv"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": [
                "desert", "mountains", "sea", "sunset", "trees"
            ]
        },
        {
            "kode": "C3",
            "nama_dataset": "Multi-Label Classification Dataset (Research Articles)",
            "asal_dataset": "Kaggle / Analytics Vidhya Janatahack",
            "url_sumber": "https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset",
            "lokasi_file": "datasets/archive/train.csv",
            "nama_file_alternatif": ["train.csv"],
            "tanpa_header": False,
            "nama_kolom_manual": None,
            "kolom_target": [
                "Computer Science", "Physics", "Mathematics",
                "Statistics", "Quantitative Biology", "Quantitative Finance"
            ]
        }
    ]
}


# =============================================================================
# 2. FUNGSI PENCARI & PEMBACA FILE DATASET (.CSV, .DATA, .XLSX, .ARFF)
# =============================================================================

def temukan_lokasi_file(konfig):
    """
    Mencari lokasi file dataset secara otomatis baik ketika skrip diletakkan
    di luar folder 'datasets/' maupun di dalam folder 'datasets/' (atau 'dataset/').
    """
    jalur_utama = konfig["lokasi_file"]
    tanpa_prefix_datasets = jalur_utama.replace("datasets/", "", 1)
    folder_skrip = os.path.dirname(os.path.abspath(__file__))

    kandidat_jalur = [
        jalur_utama,
        tanpa_prefix_datasets,
        os.path.join("dataset", tanpa_prefix_datasets),
        os.path.join(folder_skrip, jalur_utama),
        os.path.join(folder_skrip, tanpa_prefix_datasets),
        os.path.join(folder_skrip, "dataset", tanpa_prefix_datasets),
        os.path.join(folder_skrip, "datasets", tanpa_prefix_datasets),
    ]

    for jalur in kandidat_jalur:
        if os.path.isfile(jalur):
            return jalur

    # Pencarian otomatis di seluruh subfolder berdasarkan nama file
    daftar_nama_dicari = [os.path.basename(jalur_utama)] + konfig.get("nama_file_alternatif", [])
    daftar_nama_lower = [n.lower() for n in daftar_nama_dicari]

    for akar_pencarian in [os.getcwd(), folder_skrip]:
        for root, _, files in os.walk(akar_pencarian):
            for nama_file in files:
                if nama_file.lower() in daftar_nama_lower:
                    return os.path.join(root, nama_file)

    return None


def baca_file_arff(lokasi_file):
    """Membaca file format .arff secara langsung menjadi Pandas DataFrame."""
    nama_kolom = []
    baris_data = []
    sedang_baca_data = False

    with open(lokasi_file, "r", encoding="utf-8", errors="ignore") as f:
        for baris in f:
            baris_bersih = baris.strip()
            if not baris_bersih or baris_bersih.startswith("%"):
                continue

            baris_lower = baris_bersih.lower()
            if baris_lower.startswith("@attribute"):
                bagian = baris_bersih.split()
                if len(bagian) >= 2:
                    nama_atribut = bagian[1].strip("'\"")
                    nama_kolom.append(nama_atribut)
            elif baris_lower.startswith("@data"):
                sedang_baca_data = True
            elif sedang_baca_data:
                nilai = [x.strip().strip("'\"") for x in baris_bersih.split(",")]
                baris_data.append(nilai)

    df = pd.DataFrame(baris_data, columns=nama_kolom if nama_kolom else None)
    for kol in df.columns:
        try:
            df[kol] = pd.to_numeric(df[kol])
        except (ValueError, TypeError):
            pass
    return df


def muat_dataset(konfig):
    """
    Memuat dataset ke dalam Pandas DataFrame berdasarkan ekstensi file.
    Memberikan pesan error yang jelas apabila file tidak ditemukan.
    """
    lokasi_ditemukan = temukan_lokasi_file(konfig)

    if lokasi_ditemukan is None:
        print(f"\n[ERROR] File dataset tidak ditemukan untuk '{konfig['nama_dataset']}'!")
        print(f"        Lokasi default yang dicari : {konfig['lokasi_file']}")
        print(f"        Nama file yang dicari      : {konfig.get('nama_file_alternatif', [])}")
        print("        Solusi                     : Pastikan folder hasil ekstrak berada satu folder")
        print("                                     dengan file Python ini, atau sesuaikan 'lokasi_file'.\n")
        return None, None

    ekstensi = os.path.splitext(lokasi_ditemukan)[1].lower()

    try:
        if ekstensi in [".xlsx", ".xls"]:
            df = pd.read_excel(lokasi_ditemukan)
        elif ekstensi == ".arff":
            df = baca_file_arff(lokasi_ditemukan)
        else:
            if konfig.get("tanpa_header", False):
                df = pd.read_csv(
                    lokasi_ditemukan,
                    header=None,
                    names=konfig.get("nama_kolom_manual")
                )
            else:
                df = pd.read_csv(lokasi_ditemukan)
        return df, lokasi_ditemukan
    except Exception as e:
        print(f"\n[ERROR] Gagal membaca file '{lokasi_ditemukan}': {e}\n")
        return None, None


# =============================================================================
# 3. FUNGSI PEMERIKSAAN DASAR DATASET (POIN A.1 s/d A.7 & A.11)
# =============================================================================

def tampilkan_informasi_umum(df, konfig, lokasi_aktual, label_jenis):
    """Menampilkan pemeriksaan umum dataset (poin A.1 s/d A.7 dan A.11)."""
    jumlah_baris, jumlah_kolom = df.shape
    jumlah_duplikat = int(df.duplicated().sum())
    jumlah_baris_bersih = jumlah_baris - jumlah_duplikat

    print("=" * 80)
    print(f"[{konfig['kode']}] IDENTIFIKASI DATASET {label_jenis.upper()}")
    print("=" * 80)
    print(f"1. Nama Dataset             : {konfig['nama_dataset']}")
    print(f"2. Asal Dataset             : {konfig['asal_dataset']}")
    print(f"   URL Sumber               : {konfig['url_sumber']}")
    print(f"   Lokasi File Terbaca      : {lokasi_aktual}")
    print(f"3. Dimensi Dataset (Mentah) : {jumlah_baris} baris x {jumlah_kolom} kolom")
    print(f"   Jumlah Baris Duplikat    : {jumlah_duplikat} baris")
    print(f"   Dimensi Setelah Bersih   : {jumlah_baris_bersih} baris x {jumlah_kolom} kolom")

    # Poin A.11: Validasi jumlah baris minimal 150 baris
    if jumlah_baris >= 150:
        print(f"   Status Jumlah Baris      : LOLOS SYARAT WAJIB ({jumlah_baris} >= 150 baris)")
    else:
        print(f"   Status Jumlah Baris      : [PERINGATAN!] TIDAK LOLOS SYARAT WAJIB ({jumlah_baris} < 150 baris)")

    # Poin A.4: Nama seluruh kolom
    print(f"\n4. Daftar Nama Kolom ({jumlah_kolom} Kolom):")
    print(f"   {list(df.columns)}")

    # Poin A.5: Tipe data setiap kolom
    print("\n5. Tipe Data Setiap Kolom:")
    ringkasan_tipe = df.dtypes.to_string().replace("\n", "\n   ")
    print(f"   {ringkasan_tipe}")

    # Poin A.6: Tampilkan 5 data pertama
    print("\n6. Lima Baris Data Pertama (df.head(5)):")
    print(df.head(5))

    # Poin A.7: Periksa missing values
    missing_per_kolom = df.isna().sum()
    total_missing = int(missing_per_kolom.sum())
    print(f"\n7. Pemeriksaan Missing Values:")
    print(f"   Total nilai kosong (NaN/Null) di seluruh tabel: {total_missing}")
    if total_missing > 0:
        print("   Rincian kolom yang memiliki missing values:")
        print(missing_per_kolom[missing_per_kolom > 0].to_string())
    else:
        print("   Tidak ditemukan missing values pada seluruh kolom.")


# =============================================================================
# 4. FUNGSI IDENTIFIKASI KHUSUS SETIAP JENIS KLASIFIKASI (POIN A.8-10 & B.1-3)
# =============================================================================

def identifikasi_binary(konfig):
    """
    Melakukan identifikasi dan validasi untuk dataset Binary Classification.
    Memastikan target hanya berupa 1 kolom dengan tepat 2 kategori.
    """
    df, lokasi_aktual = muat_dataset(konfig)
    if df is None:
        return

    tampilkan_informasi_umum(df, konfig, lokasi_aktual, "Binary Classification")
    kolom_target = konfig["kolom_target"]

    print("\n" + "-" * 80)
    print("8–10 & B.1. VALIDASI TARGET BINARY CLASSIFICATION")
    print("-" * 80)

    peta_kolom = {k.strip().lower(): k for k in df.columns}
    if kolom_target not in df.columns and kolom_target.lower() in peta_kolom:
        kolom_target = peta_kolom[kolom_target.lower()]

    if kolom_target not in df.columns:
        print(f"[PERINGATAN] Kolom target '{kolom_target}' tidak ditemukan di dalam dataset!")
        print(f"             Kolom yang tersedia: {list(df.columns)}\n")
        return

    seri_target = df[kolom_target].dropna()
    kategori_unik = seri_target.unique().tolist()
    jumlah_kategori = len(kategori_unik)
    distribusi = seri_target.value_counts()
    persentase = (seri_target.value_counts(normalize=True) * 100).round(2)

    print(f"- Nama Kolom Target         : '{kolom_target}'")
    print(f"- Jumlah Kategori           : {jumlah_kategori} kategori")
    print(f"- Daftar Kedua Kategori     : {kategori_unik}")
    print("\n- Distribusi Setiap Kategori:")
    tabel_dist = pd.DataFrame({
        "Jumlah Baris": distribusi,
        "Persentase (%)": persentase
    })
    print(tabel_dist)

    if jumlah_kategori == 2:
        print("\n[KESIMPULAN VALIDASI] Dataset VALID sebagai Binary Classification")
        print("                      (Memiliki 1 kolom target tunggal dengan tepat 2 kelas saling lepas).")
    else:
        print(f"\n[PERINGATAN VALIDASI] Dataset TIDAK SESUAI untuk Binary Classification!")
        print(f"                      Ditemukan {jumlah_kategori} kategori (seharusnya tepat 2 kategori).")
    print("\n")


def identifikasi_multiclass(konfig):
    """
    Melakukan identifikasi dan validasi untuk dataset Multiclass Classification.
    Memastikan setiap baris hanya memiliki 1 label dari > 2 pilihan kategori.
    """
    df, lokasi_aktual = muat_dataset(konfig)
    if df is None:
        return

    tampilkan_informasi_umum(df, konfig, lokasi_aktual, "Multiclass Classification")
    kolom_target = konfig["kolom_target"]

    print("\n" + "-" * 80)
    print("8–10 & B.2. VALIDASI TARGET MULTICLASS CLASSIFICATION")
    print("-" * 80)

    if isinstance(kolom_target, list):
        print("[PERINGATAN] Target berupa beberapa kolom. Pastikan ini bukan dataset Multilabel!")
        return

    peta_kolom = {k.strip().lower(): k for k in df.columns}
    if kolom_target not in df.columns and kolom_target.lower() in peta_kolom:
        kolom_target = peta_kolom[kolom_target.lower()]

    if kolom_target not in df.columns:
        print(f"[PERINGATAN] Kolom target '{kolom_target}' tidak ditemukan di dalam dataset!")
        print(f"             Kolom yang tersedia: {list(df.columns)}\n")
        return

    seri_target = df[kolom_target].dropna()
    kategori_unik = seri_target.unique().tolist()
    jumlah_kategori = len(kategori_unik)
    distribusi = seri_target.value_counts()
    persentase = (seri_target.value_counts(normalize=True) * 100).round(2)

    print(f"- Nama Kolom Target         : '{kolom_target}' (1 kolom target tunggal)")
    print(f"- Jumlah Kategori           : {jumlah_kategori} kategori")
    print(f"- Daftar Seluruh Kategori   : {kategori_unik}")
    print("- Struktur Label per Sampel : Setiap baris sampel hanya memiliki tepat 1 nilai label.")
    print("\n- Distribusi Setiap Kategori:")
    tabel_dist = pd.DataFrame({
        "Jumlah Baris": distribusi,
        "Persentase (%)": persentase
    })
    print(tabel_dist)

    if jumlah_kategori > 2:
        print("\n[KESIMPULAN VALIDASI] Dataset VALID sebagai Multiclass Classification")
        print(f"                      (Memiliki {jumlah_kategori} kategori (> 2) dan tiap sampel hanya punya 1 label).")
    else:
        print(f"\n[PERINGATAN VALIDASI] Dataset TIDAK SESUAI untuk Multiclass Classification!")
        print(f"                      Hanya ditemukan {jumlah_kategori} kategori (seharusnya > 2 kategori).")
    print("\n")


def normalisasi_matriks_label_biner(df_label):
    """
    Mengubah kolom-kolom target multilabel (baik bertipe 0/1, True/False, atau string 'TRUE'/'FALSE')
    menjadi matriks integer biner 0 dan 1 agar mudah dihitung frekuensinya.
    """
    df_biner = pd.DataFrame(index=df_label.index)
    for kol in df_label.columns:
        seri = df_label[kol]
        if pd.api.types.is_bool_dtype(seri):
            df_biner[kol] = seri.astype(int)
        elif pd.api.types.is_numeric_dtype(seri):
            df_biner[kol] = (seri > 0).astype(int)
        else:
            df_biner[kol] = seri.astype(str).str.strip().str.upper().isin(["1", "TRUE", "YES", "T"]).astype(int)
    return df_biner


def identifikasi_multilabel(konfig):
    """
    Melakukan identifikasi dan validasi untuk dataset Multilabel Classification.
    Memastikan terdapat sampel yang benar-benar memiliki > 1 label aktif sekaligus.
    """
    df, lokasi_aktual = muat_dataset(konfig)
    if df is None:
        return

    tampilkan_informasi_umum(df, konfig, lokasi_aktual, "Multilabel Classification")
    daftar_kolom_label = konfig["kolom_target"]

    print("\n" + "-" * 80)
    print("8–10 & B.3. VALIDASI TARGET MULTILABEL CLASSIFICATION")
    print("-" * 80)

    if not isinstance(daftar_kolom_label, list) or len(daftar_kolom_label) < 2:
        print("[PERINGATAN] Struktur target Multilabel harus terdiri dari daftar beberapa kolom label!")
        return

    kolom_tidak_ada = [k for k in daftar_kolom_label if k not in df.columns]
    if kolom_tidak_ada:
        print(f"[PERINGATAN] Kolom label berikut tidak ditemukan di dataset: {kolom_tidak_ada}")
        print(f"             Kolom yang tersedia: {list(df.columns)}\n")
        return

    df_label_biner = normalisasi_matriks_label_biner(df[daftar_kolom_label])
    jumlah_label_tersedia = len(daftar_kolom_label)

    jumlah_label_per_baris = df_label_biner.sum(axis=1)
    baris_multi_label = df[jumlah_label_per_baris > 1]
    banyak_baris_multi = len(baris_multi_label)
    rata_rata_label = jumlah_label_per_baris.mean()

    print(f"- Daftar Kolom Label        : {daftar_kolom_label}")
    print(f"- Jumlah Label Tersedia     : {jumlah_label_tersedia} label")
    print(f"- Rata-rata Label / Sampel  : {rata_rata_label:.2f} label per baris")
    print(f"- Sampel dengan > 1 Label   : {banyak_baris_multi} dari {len(df)} baris ({(banyak_baris_multi / len(df)) * 100:.2f}%)")

    frekuensi_label = df_label_biner.sum().sort_values(ascending=False)
    persentase_label = ((frekuensi_label / len(df)) * 100).round(2)
    print("\n- Frekuensi Kemunculan Setiap Label Positif (Bernilai 1 / TRUE):")
    tabel_frek = pd.DataFrame({
        "Frekuensi Aktif": frekuensi_label,
        "Persentase Sampel (%)": persentase_label
    })
    print(tabel_frek)

    print("\n- Distribusi Jumlah Label Aktif dalam Satu Sampel:")
    print(jumlah_label_per_baris.value_counts().sort_index().rename("Jumlah Baris Sampel").to_string())

    print("\n- Contoh 5 Baris Sampel yang Memiliki Lebih dari Satu Label Sekaligus:")
    if banyak_baris_multi > 0:
        contoh_multilabel = df_label_biner[jumlah_label_per_baris > 1].head(5).copy()
        contoh_multilabel["Total_Label_Aktif"] = jumlah_label_per_baris[jumlah_label_per_baris > 1].head(5)
        print(contoh_multilabel)
        print("\n[KESIMPULAN VALIDASI] Dataset VALID sebagai Multilabel Classification")
        print(f"                      (Terdapat {banyak_baris_multi} sampel yang memiliki > 1 label sekaligus secara alami).")
    else:
        print("  Tidak ditemukan baris yang memiliki lebih dari 1 label aktif!")
        print("\n[PERINGATAN VALIDASI] Dataset TIDAK SESUAI sebagai Multilabel Classification!")
        print("                      Setiap baris hanya memiliki paling banyak 1 label (kemungkinan Multiclass One-Hot).")
    print("\n")


# =============================================================================
# 5. PROGRAM UTAMA (MAIN EXECUTION)
# =============================================================================

def jalankan_semua_identifikasi():
    """Menjalankan identifikasi berurutan untuk ke-9 dataset."""
    print("\n" + "#" * 80)
    print("   PROGRAM IDENTIFIKASI & VALIDASI 9 DATASET KLASIFIKASI MACHINE LEARNING")
    print("#" * 80 + "\n")

    print(">>> BAGIAN A: IDENTIFIKASI 3 DATASET BINARY CLASSIFICATION\n")
    for konfig in KONFIGURASI_DATASET["binary"]:
        identifikasi_binary(konfig)

    print(">>> BAGIAN B: IDENTIFIKASI 3 DATASET MULTICLASS CLASSIFICATION\n")
    for konfig in KONFIGURASI_DATASET["multiclass"]:
        identifikasi_multiclass(konfig)

    print(">>> BAGIAN C: IDENTIFIKASI 3 DATASET MULTILABEL CLASSIFICATION\n")
    for konfig in KONFIGURASI_DATASET["multilabel"]:
        identifikasi_multilabel(konfig)

    print("#" * 80)
    print("   SELESAI: SELURUH KONFIGURASI DATASET TELAH DIPERIKSA.")
    print("#" * 80)


if __name__ == "__main__":
    jalankan_semua_identifikasi()