from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
)
DATA_FILE = Path(__file__).with_name("diabetes.csv")


def load_data() -> pd.DataFrame:
    """Muat CSV lokal; unduh dari sumber dataset jika file belum ada."""
    if DATA_FILE.exists():
        print(f"Membaca dataset lokal: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print("diabetes.csv tidak ditemukan; mencoba mengunduh dataset...")
    try:
        return pd.read_csv(DATA_URL)
    except Exception as exc:
        raise RuntimeError(
            "Dataset gagal diunduh. Unduh diabetes.csv dari sumber Pima "
            "Indians Diabetes Dataset, simpan di folder yang sama dengan "
            "script, lalu jalankan kembali."
        ) from exc


def main() -> None:
    # 1. Membaca data dan membentuk DataFrame.
    df = load_data()
    print("\nUkuran dataset:", df.shape)
    print("\nLima data pertama:")
    print(df.head())
    print("\nInformasi dan tipe data:")
    df.info()
    print("\nJumlah nilai kosong per kolom:")
    print(df.isna().sum())

    # 2. Memisahkan fitur (8 atribut) dan label (Outcome).
    if "Outcome" not in df.columns:
        raise ValueError("Kolom label 'Outcome' tidak ditemukan dalam CSV.")
    X = df.drop(columns="Outcome")
    y = df["Outcome"].astype(int)

    print("\nJumlah kelas (0 = tidak diabetes, 1 = diabetes):")
    print(y.value_counts().sort_index())

    # 3. Membagi data terlebih dahulu agar scaler hanya belajar dari data latih.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )
    print(f"\nJumlah data latih: {len(X_train)}")
    print(f"Jumlah data uji: {len(X_test)}")

    # 4. Standarisasi dan pelatihan SVC. Pipeline mencegah kebocoran data uji.
    model = make_pipeline(StandardScaler(), SVC())
    model.fit(X_train, y_train)

    # 5. Evaluasi model pada data uji.
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    print("\nAkurasi:", f"{accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (baris = label aktual, kolom = prediksi):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nLaporan klasifikasi:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Tidak diabetes (0)", "Diabetes (1)"],
            zero_division=0,
        )
    )
    print("ROC-AUC:", f"{roc_auc_score(y_test, y_score):.4f}")


if __name__ == "__main__":
    main()
