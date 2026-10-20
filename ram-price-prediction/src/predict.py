"""
predict.py — RAM Price-per-GB Prediction CLI
=============================================
Loads the trained pipeline and predicts price_per_gb for user-supplied
RAM specifications.

Usage:
    python src/predict.py                    # interactive prompts
    python src/predict.py --json '{...}'     # single JSON record

Example JSON:
    python src/predict.py --json '{
        "brand": "Kingston",
        "region": "North America",
        "generation": "DDR5",
        "form_factor": "DIMM",
        "capacity_gb": 32,
        "module_count": 2,
        "is_ecc": "False",
        "speed_mts": 6000,
        "cas_latency": 36,
        "voltage": 1.35,
        "die_manufacturer": "Samsung",
        "die_type": "B-Die",
        "true_latency_ns": 12.0,
        "market_segment": "Consumer",
        "recommended_usage": "Gaming (AAA)",
        "global_inventory_weeks": 10.0,
        "fab_utilization_rate": 0.72,
        "gpu_hbm_trend_gb": 160.0
    }'
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd


def find_model_dir():
    """Locate models/ directory relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "models")


def load_artifacts(model_dir: str):
    """Load the trained pipeline and feature metadata."""
    model_path = os.path.join(model_dir, "best_model.joblib")
    meta_path = os.path.join(model_dir, "feature_meta.joblib")

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("        Run 'python src/train.py' first.")
        sys.exit(1)
    if not os.path.isfile(meta_path):
        print(f"[ERROR] Feature metadata not found: {meta_path}")
        sys.exit(1)

    pipeline = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipeline, meta


def validate_and_cast(record: dict, meta: dict) -> dict:
    """Validate input fields, cast types, fill defaults for missing optional fields."""
    numeric_features = meta["numeric_features"]
    categorical_features = meta["categorical_features"]
    all_features = meta["all_features"]

    errors = []
    validated = {}

    # Check required features
    for feat in all_features:
        if feat not in record or record[feat] is None or str(record[feat]).strip() == "":
            errors.append(f"Missing or empty: '{feat}'")
        else:
            validated[feat] = record[feat]

    if errors:
        print("[VALIDATION ERRORS]")
        for e in errors:
            print(f"  - {e}")
        print()
        print("Required features:")
        print(f"  Numeric:     {numeric_features}")
        print(f"  Categorical: {categorical_features}")
        sys.exit(1)

    # Cast numeric features
    for feat in numeric_features:
        try:
            validated[feat] = float(validated[feat])
        except (ValueError, TypeError):
            print(f"[ERROR] '{feat}' must be numeric, got: {validated[feat]}")
            sys.exit(1)

    # Cast categorical to string
    for feat in categorical_features:
        validated[feat] = str(validated[feat])

    # Sanity checks
    if validated.get("capacity_gb", 0) <= 0:
        print("[ERROR] capacity_gb must be positive.")
        sys.exit(1)
    if validated.get("speed_mts", 0) <= 0:
        print("[ERROR] speed_mts must be positive.")
        sys.exit(1)

    return validated


def interactive_input(meta: dict) -> dict:
    """Prompt user for each feature interactively."""
    print("\n+------------------------------------------+")
    print("|   RAM Price-per-GB Prediction -- Input   |")
    print("+------------------------------------------+\n")

    # Default values for quick testing
    defaults = {
        "brand": "Kingston",
        "region": "North America",
        "generation": "DDR5",
        "form_factor": "DIMM",
        "capacity_gb": "32",
        "module_count": "2",
        "is_ecc": "False",
        "speed_mts": "6000",
        "cas_latency": "36",
        "voltage": "1.35",
        "die_manufacturer": "Samsung",
        "die_type": "B-Die",
        "true_latency_ns": "12.0",
        "market_segment": "Consumer",
        "recommended_usage": "Gaming (AAA)",
        "global_inventory_weeks": "10.0",
        "fab_utilization_rate": "0.72",
        "gpu_hbm_trend_gb": "160.0",
    }

    record = {}
    all_features = meta["all_features"]
    for feat in all_features:
        default = defaults.get(feat, "")
        prompt = f"  {feat}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        val = input(prompt).strip()
        if val == "" and default:
            val = default
        record[feat] = val

    return record


def predict_single(pipeline, record: dict, meta: dict):
    """Run prediction and display results."""
    df = pd.DataFrame([record])

    # Ensure column order matches training
    df = df[meta["all_features"]]

    predicted_ppg = pipeline.predict(df)[0]
    capacity = float(record["capacity_gb"])
    estimated_total = predicted_ppg * capacity

    print("\n+------------------------------------------+")
    print("|          PREDICTION RESULTS              |")
    print("+------------------------------------------+")
    print(f"|  Predicted price_per_gb : ${predicted_ppg:>10.2f}    |")
    print(f"|  Capacity (GB)          : {capacity:>10.0f}     |")
    print(f"|  Estimated total price  : ${estimated_total:>10.2f}    |")
    print("+------------------------------------------+")
    print(f"\n  Formula: ${predicted_ppg:.2f}/GB x {capacity:.0f} GB = ${estimated_total:.2f}")

    return predicted_ppg, estimated_total


def main():
    parser = argparse.ArgumentParser(description="Predict RAM price_per_gb")
    parser.add_argument("--json", default=None,
                        help="JSON string with RAM specifications")
    parser.add_argument("--file", default=None,
                        help="Path to JSON file containing RAM specifications")
    parser.add_argument("--model-dir", default=None,
                        help="Path to models/ directory")
    args = parser.parse_args()

    model_dir = args.model_dir or find_model_dir()
    pipeline, meta = load_artifacts(model_dir)

    if args.file:
        if not os.path.isfile(args.file):
            print(f"[ERROR] Input file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            record = json.load(f)
    elif args.json:
        try:
            record = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            sys.exit(1)
    else:
        record = interactive_input(meta)

    validated = validate_and_cast(record, meta)
    predict_single(pipeline, validated, meta)


if __name__ == "__main__":
    main()
