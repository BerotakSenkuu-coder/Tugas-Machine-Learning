"""
train.py - RAM Price-per-GB Regression Training Pipeline
=========================================================
Trains four regressors (Dummy, Linear, Ridge, Random Forest) on the
Ultimate Memory Shortage Crisis dataset using a chronological train/test
split and scikit-learn Pipelines.

Usage:
    python src/train.py                           # default paths
    python src/train.py --csv path/to/data.csv    # custom CSV path

Outputs:
    models/best_model.joblib        - trained pipeline (best model)
    reports/evaluation_metrics.csv  - MAE / RMSE / R2 for every model
    reports/figures/                - PNG visualisations
"""

import argparse
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
# CONFIGURATION
# ======================================================================
TARGET = "price_per_gb"

# Features safe to use (no leakage)
NUMERIC_FEATURES = [
    "capacity_gb",
    "module_count",
    "speed_mts",
    "cas_latency",
    "voltage",
    "true_latency_ns",
    "global_inventory_weeks",
    "fab_utilization_rate",
    "gpu_hbm_trend_gb",
]

CATEGORICAL_FEATURES = [
    "brand",
    "region",
    "generation",
    "form_factor",
    "is_ecc",
    "die_manufacturer",
    "die_type",
    "market_segment",
    "recommended_usage",
]

# Columns excluded and reasons:
# - kit_id            : unique identifier, no predictive value
# - model_name        : high-cardinality, encodes brand+gen+speed already
# - timestamp         : used only for chronological split
# - timing_string     : redundant with cas_latency / true_latency_ns
# - price_usd         : directly computes target -> leakage
# - price_per_gb      : TARGET
# - price_status      : label derived from price behaviour -> leakage risk
# - bandwidth_per_dollar, latency_value_index, speed_premium_ratio:
#       all contain "per dollar" / price-normalised info -> leakage

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CHRONOLOGICAL_SPLIT_COL = "timestamp"
FUTURE_CUTOFF = "2026-10-19"   # current date


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV, parse dates, sort chronologically, flag future rows."""
    df = pd.read_csv(csv_path)
    df[CHRONOLOGICAL_SPLIT_COL] = pd.to_datetime(df[CHRONOLOGICAL_SPLIT_COL])
    df = df.sort_values(CHRONOLOGICAL_SPLIT_COL).reset_index(drop=True)

    # Flag future-dated rows
    future_mask = df[CHRONOLOGICAL_SPLIT_COL] > pd.Timestamp(FUTURE_CUTOFF)
    n_future = future_mask.sum()
    if n_future > 0:
        print(f"[INFO] {n_future} rows dated after {FUTURE_CUTOFF} detected.")
        print(f"       Range: {df.loc[future_mask, CHRONOLOGICAL_SPLIT_COL].min().date()} "
              f"to {df.loc[future_mask, CHRONOLOGICAL_SPLIT_COL].max().date()}")
        print("       These rows are KEPT but documented. They land in the "
              "test set as part of the chronological evaluation.\n")

    # Cast is_ecc to string for categorical encoding
    df["is_ecc"] = df["is_ecc"].astype(str)
    return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split into train/test by time - test = latest test_frac rows."""
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"[SPLIT] Train: {len(train)} rows  "
          f"({train[CHRONOLOGICAL_SPLIT_COL].min().date()} -> "
          f"{train[CHRONOLOGICAL_SPLIT_COL].max().date()})")
    print(f"[SPLIT] Test:  {len(test)} rows  "
          f"({test[CHRONOLOGICAL_SPLIT_COL].min().date()} -> "
          f"{test[CHRONOLOGICAL_SPLIT_COL].max().date()})\n")
    return train, test


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer: median impute numerics, mode impute + OHE categoricals."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_models(preprocessor: ColumnTransformer) -> dict:
    """Return dict of name -> Pipeline for each candidate model."""
    return {
        "DummyRegressor": Pipeline([
            ("pre", preprocessor),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "LinearRegression": Pipeline([
            ("pre", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("pre", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("pre", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def evaluate(models: dict, X_train, y_train, X_test, y_test, report_dir: str):
    """Train all models, evaluate on test set, and return results DataFrame."""
    results = []
    predictions = {}

    for name, pipe in models.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        predictions[name] = y_pred
        print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(os.path.join(report_dir, "evaluation_metrics.csv"), index=False)
    print(f"\n[SAVED] evaluation_metrics.csv")
    return results_df, predictions


def plot_results(y_test, predictions: dict, results_df: pd.DataFrame,
                 best_name: str, fig_dir: str):
    """Generate comparison plots and save to fig_dir."""
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.25
    ax.bar(x - width, results_df["MAE"], width, label="MAE", color="#4C72B0")
    ax.bar(x, results_df["RMSE"], width, label="RMSE", color="#DD8452")
    ax.bar(x + width, results_df["R2"] * results_df["RMSE"].max(), width,
           label=f"R2 (scaled x{results_df['RMSE'].max():.0f})", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_title("Model Comparison - Test Set Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # 2) Actual vs Predicted (best model)
    y_pred_best = predictions[best_name]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.3, s=10, color="#4C72B0")
    lims = [min(y_test.min(), y_pred_best.min()),
            max(y_test.max(), y_pred_best.max())]
    ax.plot(lims, lims, "--", color="red", linewidth=1)
    ax.set_xlabel("Actual price_per_gb ($)")
    ax.set_ylabel("Predicted price_per_gb ($)")
    ax.set_title(f"Actual vs Predicted - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # 3) Residual plot (best model)
    residuals = y_test.values - y_pred_best
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred_best, residuals, alpha=0.3, s=10, color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Predicted price_per_gb ($)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "residuals.png"), dpi=150)
    plt.close(fig)

    print(f"[SAVED] Evaluation figures to {fig_dir}/")


def plot_feature_importance(pipe, fig_dir: str):
    """Extract and plot feature importance from a trained RF pipeline."""
    preprocessor = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    # Get feature names after transformation
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            feature_names.extend(ohe.get_feature_names_out(cols))

    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]  # top 20

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(idx)), importances[idx], color="#4C72B0")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title("Top 20 Feature Importances (RandomForest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"[SAVED] feature_importance.png to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train RAM price_per_gb regression models")
    parser.add_argument("--csv", default=None,
                        help="Path to the CSV dataset. Defaults to data/ram_dataset.csv "
                             "or the original CSV location.")
    args = parser.parse_args()

    # Resolve project root (one level up from src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Find CSV
    csv_candidates = [
        args.csv,
        os.path.join(project_root, "data", "ram_dataset.csv"),
        os.path.join(project_root, "..", "Ultimate_Memory_Shortage_Crisis_Dataset_10k.csv"),
    ]
    csv_path = None
    for c in csv_candidates:
        if c and os.path.isfile(c):
            csv_path = c
            break
    if csv_path is None:
        print("[ERROR] CSV dataset not found. Provide --csv path.")
        sys.exit(1)

    print(f"[DATA]  {csv_path}")
    model_dir = os.path.join(project_root, "models")
    report_dir = os.path.join(project_root, "reports")
    fig_dir = os.path.join(report_dir, "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # -- Load & split --
    df = load_and_prepare(csv_path)
    train_df, test_df = chronological_split(df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]

    # -- Build & evaluate --
    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION (chronological test set)")
    print("=" * 60)
    results_df, predictions = evaluate(models, X_train, y_train,
                                       X_test, y_test, report_dir)

    # -- Select best model --
    best_name = results_df.iloc[0]["Model"]
    best_pipe = models[best_name]
    print(f"\n[BEST] Best model by MAE: {best_name}")
    print(f"  MAE  = {results_df.iloc[0]['MAE']:.4f}")
    print(f"  RMSE = {results_df.iloc[0]['RMSE']:.4f}")
    print(f"  R2   = {results_df.iloc[0]['R2']:.4f}")

    # -- Save best model --
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n[SAVED] {model_path}")

    # -- Save feature list (for predict.py) --
    feature_meta = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "target": TARGET,
    }
    joblib.dump(feature_meta, os.path.join(model_dir, "feature_meta.joblib"))
    print("[SAVED] feature_meta.joblib")

    # -- Plots --
    plot_results(y_test, predictions, results_df, best_name, fig_dir)
    if best_name == "RandomForest":
        plot_feature_importance(best_pipe, fig_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Metrics: {os.path.join(report_dir, 'evaluation_metrics.csv')}")
    print(f"  Figures: {fig_dir}/")
    print("\nNote: All evaluation metrics are from the chronological test set,")
    print("      NOT from training data.")


if __name__ == "__main__":
    main()
