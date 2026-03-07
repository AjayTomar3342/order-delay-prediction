import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import json
import yaml
from pathlib import Path


# =====================================================
# Drift Detection
# =====================================================
def detect_drift(reference_df, current_df, num_cols, cat_cols, threshold=0.05):

    drift_report = {
        "drift_detected": False,
        "details": {}
    }

    print("\n--- Starting Data Drift Audit ---")

    # ---------- Numerical ----------
    for col in num_cols:

        if col not in current_df.columns:
            print(f"Skipping missing column: {col}")
            continue

        ref = reference_df[col].dropna()
        curr = current_df[col].dropna()

        if len(ref) == 0 or len(curr) == 0:
            continue

        stat, p_value = ks_2samp(ref, curr)

        drift = bool(p_value < threshold)

        drift_report["details"][col] = {
            "type": "numerical",
            "p_value": float(p_value),
            "drift": drift
        }

        if drift:
            drift_report["drift_detected"] = True

    # ---------- Categorical ----------
    for col in cat_cols:

        if col not in current_df.columns:
            print(f"Skipping missing column: {col}")
            continue

        ref_counts = reference_df[col].value_counts()
        curr_counts = current_df[col].value_counts()

        combined = pd.concat([ref_counts, curr_counts], axis=1).fillna(0)

        if combined.shape[0] < 2:
            continue

        chi2, p_value, _, _ = chi2_contingency(combined)

        drift = bool(p_value < threshold)

        drift_report["details"][col] = {
            "type": "categorical",
            "p_value": float(p_value),
            "drift": drift
        }

        if drift:
            drift_report["drift_detected"] = True

    return drift_report


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    print("\nRunning monitor_drift.py")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    SRC_DIR = BASE_DIR.parent
    ROOT_DIR = SRC_DIR.parent

    DATA_DIR = ROOT_DIR / "data" / "processed"
    CONFIG_PATH = ROOT_DIR / "config/config.yaml"

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    num_cols = config["modeling"]["numerical_columns"]
    cat_cols = config["modeling"]["categorical_columns"]

    # Load data
    df_ref = pd.read_csv(DATA_DIR / "feature_date.csv")
    df_curr = pd.read_csv(DATA_DIR / "clean_data.csv")

    print("✅ Files loaded successfully.")
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")

    # Run drift
    report = detect_drift(df_ref, df_curr, num_cols, cat_cols)

    # Save results
    results_dir = ROOT_DIR / "monitoringresults"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "drift_report.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Drift report saved to:\n{output_path}")

    if report["drift_detected"]:
        print("⚠️ DATA DRIFT DETECTED")
    else:
        print("✅ No significant drift detected")