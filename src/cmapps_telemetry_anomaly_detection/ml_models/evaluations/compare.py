"""
compare.py

Collects saved metrics into comparison tables.

Per subset:
    reports/tables/anomaly_model_comparison_<subset>.csv
        every anomaly model + both ensembles, sorted by PR-AUC

Across all subsets:
    reports/tables/summary_all_subsets.csv
        best single anomaly model vs the ensembles, and the RUL results,
        for FD001 - FD004 side by side
"""

import json
import pandas as pd
from pathlib import Path

from cmapps_telemetry_anomaly_detection.config import SUBSETS
from cmapps_telemetry_anomaly_detection.ml_models.utils import load_config

SINGLE_MODELS = [
    "isolation_forest",
    "pca_reconstruction",
    "kmeans",
    "local_outlier_factor",
    "one_class_svm",
    "gaussian_mixture",
    "autoencoder",
]
ENSEMBLES = ["ensemble_all", "ensemble_top3"]


def load_anomaly_table(metrics_dir: Path, subset: str) -> pd.DataFrame:
    rows = []
    for name in SINGLE_MODELS + ENSEMBLES:
        path = metrics_dir / f"{name}_{subset}.json"
        if not path.exists():
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
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def run_comparison(config_path: str = "configs/data.yaml", subset: str | None = None) -> pd.DataFrame:
    """Comparison table of all anomaly models (incl. ensembles) for one subset."""
    config      = load_config(config_path, subset)
    subset      = config["dataset"]["subset"]
    metrics_dir = Path(config["reports"]["metrics_path"])
    tables_dir  = Path(config["reports"]["tables_path"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = load_anomaly_table(metrics_dir, subset)
    if table.empty:
        raise FileNotFoundError(f"[ERROR] No model metrics found for {subset}. Train some models first.")

    out_path = tables_dir / f"anomaly_model_comparison_{subset}.csv"
    table.to_csv(out_path, index=False)

    print(f"\n── Anomaly Model Comparison ({subset}, validation engines) ──")
    print(table.to_string(index=False))
    print(f"\n[INFO] Saved -> {out_path}")
    return table


def run_summary(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    """
    One row per subset: does an ensemble beat the best single model?
    Called by:  cmapps-tad compare --subset all
    """
    config      = load_config(config_path)
    metrics_dir = Path(config["reports"]["metrics_path"])
    tables_dir  = Path(config["reports"]["tables_path"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subset in SUBSETS:
        table = load_anomaly_table(metrics_dir, subset)
        rul_path = metrics_dir / f"rul_{subset}.json"
        if table.empty or not rul_path.exists():
            print(f"[WARN] Skipping {subset}: run the full pipeline for it first")
            continue

        singles = table[table["model"].isin(SINGLE_MODELS)]
        best    = singles.iloc[0]  # already sorted by PR-AUC
        pr_auc  = table.set_index("model")["pr_auc"]

        rul        = json.loads(rul_path.read_text())
        rul_models = {k: v for k, v in rul["models"].items() if k != "ensemble_average"}
        best_rul   = min(rul_models, key=lambda n: rul_models[n]["validation"]["rmse"])

        rows.append({
            "subset": subset,
            "best_single_anomaly_model": best["model"],
            "best_single_pr_auc": best["pr_auc"],
            "ensemble_all_pr_auc": pr_auc.get("ensemble_all"),
            "ensemble_top3_pr_auc": pr_auc.get("ensemble_top3"),
            "best_rul_model": best_rul,
            "best_rul_test_rmse": rul_models[best_rul]["test"]["rmse"],
            "best_rul_test_r2": rul_models[best_rul]["test"]["r2"],
            "rul_ensemble_test_rmse": rul["models"]["ensemble_average"]["test"]["rmse"],
        })

    summary = pd.DataFrame(rows)
    out_path = tables_dir / "summary_all_subsets.csv"
    summary.to_csv(out_path, index=False)

    print("\n── Summary across subsets ─────────────────────────────────")
    print(summary.to_string(index=False))
    print(f"\n[INFO] Saved -> {out_path}")
    return summary
