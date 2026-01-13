#!/usr/bin/env python
# coding: utf-8

# In[5]:


import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
IMPUTER_PATH = ARTIFACT_DIR / "imputer.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_names.json"


def load_artifacts():
    """Load model + preprocessing artifacts + feature column ordering."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run train.py first.")
    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(f"Missing {IMPUTER_PATH}. Run train.py first.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing {SCALER_PATH}. Run train.py first.")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError("feature_names.json did not contain a non-empty list of feature names.")

    return model, imputer, scaler, feature_cols


def load_input_as_dataframe(path: str, feature_cols: list[str]) -> pd.DataFrame:
    """
    Load input file into a DataFrame. Supports:
      - JSON (one row dict OR list of dicts)
      - CSV (header row)
    Then aligns columns to feature_cols (adds missing cols as NaN, drops extras).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in (".json", ".jsonl"):
        # For .jsonl, we assume one JSON object per line.
        if suffix == ".jsonl":
            rows = []
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
        else:
            with open(p, "r") as f:
                obj = json.load(f)
            # obj can be dict (single row) or list[dict] (many rows)
            if isinstance(obj, dict):
                df = pd.DataFrame([obj])
            elif isinstance(obj, list):
                df = pd.DataFrame(obj)
            else:
                raise ValueError("JSON input must be a dict (single row) or a list of dicts (many rows).")
    else:
        raise ValueError("Unsupported file type. Please provide a .csv, .json, or .jsonl input file.")

    # Align columns: keep only the expected features; add missing as NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_cols]  # enforce ordering, drop extras

    # Ensure numeric dtype where possible (strings -> NaN if coercion fails)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def predict(df: pd.DataFrame, model, imputer, scaler) -> np.ndarray:
    """Apply preprocessing and return predictions."""
    X = imputer.transform(df)
    X = scaler.transform(X)
    preds = model.predict(X)
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Predict bike rental demand (cnt) using saved artifacts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (.json/.jsonl/.csv). JSON can be a dict (single row) or list of dicts."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save predictions as CSV with a 'prediction' column."
    )
    args = parser.parse_args()

    model, imputer, scaler, feature_cols = load_artifacts()
    df = load_input_as_dataframe(args.input, feature_cols)
    preds = predict(df, model, imputer, scaler)

    # Print predictions
    if len(preds) == 1:
        print(f"Predicted cnt: {preds[0]:.2f}")
    else:
        print("Predicted cnt (first 10):")
        for i, p in enumerate(preds[:10]):
            print(f"  row {i}: {p:.2f}")
        print(f"... total rows predicted: {len(preds)}")

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_df = df.copy()
        out_df["prediction"] = preds
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()


# In[ ]:




