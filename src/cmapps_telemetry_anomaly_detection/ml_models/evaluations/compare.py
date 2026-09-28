"""
compare.py

Collects the saved metrics of every anomaly model into one comparison table.

Reads  : reports/metrics/<model>_<subset>.json
Writes : reports/tables/anomaly_model_comparison_<subset>.csv
"""

import json
import pandas as pd
from pathlib import Path

from cmapps_telemetry_anomaly_detection.ml_models.utils import load_config

ANOMALY_MODELS = [
    "isolation_forest",
    "pca_reconstruction",
    "kmeans",
    "local_outlier_factor",
    "one_class_svm",
    "gaussian_mixture",
    "autoencoder",
]


def run_comparison(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    config      = load_config(config_path)
    subset      = config["dataset"].get("subset", "FD001")
    metrics_dir = Path(config["reports"]["metrics_path"])
    tables_dir  = Path(config["reports"]["tables_path"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ANOMALY_MODELS:
        path = metrics_dir / f"{name}_{subset}.json"
        if not path.exists():
            print(f"[WARN] No metrics for {name} (run: cmapps-tad train --model {name})")
            continue
        m = json.loads(path.read_text())
        rows.append({
            "model": name,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })

    if not rows:
        raise FileNotFoundError("[ERROR] No model metrics found. Train some models first.")

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out_path = tables_dir / f"anomaly_model_comparison_{subset}.csv"
    table.to_csv(out_path, index=False)

    print(f"\n── Anomaly Model Comparison ({subset}, validation engines) ──")
    print(table.to_string(index=False))
    print(f"\n[INFO] Saved -> {out_path}")
    return table
