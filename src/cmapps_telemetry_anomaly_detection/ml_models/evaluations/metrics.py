"""
metrics.py

Shared evaluation + reporting for the unsupervised anomaly models.

Note on labels:
    CMAPSS has no real anomaly labels. 'anomaly_label' is a PROXY created in
    preprocessing: a cycle is labelled anomalous if the engine is within
    N cycles of failure (RUL <= anomaly_threshold). So these metrics measure how
    well each model's score separates near-failure cycles from healthy ones.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate(scored_df: pd.DataFrame) -> dict:
    """
    Evaluate anomaly detection performance against proxy labels.

    Metrics:
        Precision : of all rows flagged as anomalous, how many truly are?
        Recall    : of all truly anomalous rows, how many did we catch?
        F1        : harmonic mean of precision + recall
        ROC-AUC   : area under ROC curve (threshold-independent)
        PR-AUC    : area under Precision-Recall curve
                    (better than ROC-AUC for imbalanced data like this)
    """
    y_true  = scored_df["anomaly_label"].values
    y_pred  = scored_df["predicted_label"].values
    y_score = scored_df["anomaly_score"].values

    return {
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_true, y_score), 4),
        "pr_auc"    : round(average_precision_score(y_true, y_score), 4),
        "total_rows"          : int(len(scored_df)),
        "true_anomalies"      : int(y_true.sum()),
        "predicted_anomalies" : int(y_pred.sum()),
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    print(f"\n── {model_name} Evaluation (validation engines) ─────────")
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  Recall              : {metrics['recall']:.4f}")
    print(f"  F1 Score            : {metrics['f1']:.4f}")
    print(f"  ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC              : {metrics['pr_auc']:.4f}")
    print(f"  True anomalies      : {metrics['true_anomalies']:,}")
    print(f"  Predicted anomalies : {metrics['predicted_anomalies']:,}")
    print(f"  Total rows scored   : {metrics['total_rows']:,}")
    print("─────────────────────────────────────────────────────────\n")


def save_results(
    scored_df: pd.DataFrame,
    metrics: dict,
    config: dict,
    model_name: str,
) -> None:
    """Save metrics JSON to reports/metrics/ and per-row scores CSV to reports/tables/."""
    subset      = config["dataset"].get("subset", "FD001")
    metrics_dir = Path(config["reports"]["metrics_path"])
    tables_dir  = Path(config["reports"]["tables_path"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{model_name}_{subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    scores_path = tables_dir / f"{model_name}_{subset}_scores.csv"
    scored_df[["unit_id", "cycle", "RUL", "anomaly_label",
               "anomaly_score", "predicted_label"]].to_csv(scores_path, index=False)
    print(f"[INFO] Scores saved -> {scores_path}")
