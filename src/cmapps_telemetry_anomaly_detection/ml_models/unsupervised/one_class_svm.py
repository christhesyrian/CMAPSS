"""
one_class_svm.py

One-Class SVM anomaly detection model for CMAPSS.

Idea:
    Learn a boundary (in an RBF kernel space) that encloses the healthy data.
    Points that fall outside the boundary are anomalous. The signed distance to
    the boundary is used as a continuous anomaly score.

Note:
    One-Class SVM training scales roughly O(n^2) with the number of rows, so the
    healthy training data is randomly subsampled to `max_train_samples`.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

from cmapps_telemetry_anomaly_detection.ml_models.utils import (
    load_config,
    get_feature_columns,
    load_train_val,
)
from cmapps_telemetry_anomaly_detection.ml_models.evaluations.metrics import (
    evaluate,
    print_metrics,
    save_results,
)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_class_svm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    nu: float = 0.05,
    gamma: str = "scale",
    max_train_samples: int = 5000,
    random_state: int = 42,
) -> tuple[OneClassSVM, StandardScaler, np.ndarray]:
    """
    Train One-Class SVM on HEALTHY cycles only.

    Parameters:
        nu    : upper bound on the fraction of training points treated as outliers
                (plays a similar role to contamination)
        gamma : RBF kernel width ("scale" = 1 / (n_features * X.var()))
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    if len(healthy_df) > max_train_samples:
        healthy_df = healthy_df.sample(n=max_train_samples, random_state=random_state)

    print(f"[INFO] Training on healthy cycles only (subsampled)")
    print(f"[INFO] Training rows : {len(healthy_df):,}")
    print(f"[INFO] Features      : {len(feature_cols)}")
    print(f"[INFO] nu: {nu} | gamma: {gamma}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(healthy_df[feature_cols].values)

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train)

    # decision_function: positive = inside boundary (normal), so flip it
    train_scores = -model.decision_function(X_train)

    print("[INFO] One-Class SVM training complete.")
    return model, scaler, train_scores


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: OneClassSVM,
    scaler: StandardScaler,
    train_scores: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    X = scaler.transform(df[feature_cols].values)

    raw_scores  = -model.decision_function(X)
    predictions = model.predict(X)  # -1 = anomaly, 1 = normal

    min_s, max_s = train_scores.min(), train_scores.max()
    normalized = (raw_scores - min_s) / (max_s - min_s + 1e-9)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = np.clip(normalized, 0, None)
    scored_df["predicted_label"] = (predictions == -1).astype(int)
    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_one_class_svm(config_path: str = "configs/data.yaml") -> dict:
    """
    Full One-Class SVM train + score + evaluate pipeline.
    Called by:  cmapps-tad train --model one_class_svm
    """
    config       = load_config(config_path)
    subset       = config["dataset"].get("subset", "FD001")
    model_dir    = Path(config["artifacts"]["model_path"])
    params       = config["models"]["one_class_svm"]
    nu           = params.get("nu", 0.05)
    gamma        = params.get("gamma", "scale")
    max_samples  = params.get("max_train_samples", 5000)
    random_state = config["models"].get("random_state", 42)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    model, scaler, train_scores = train_one_class_svm(
        train_df, feature_cols, nu, gamma, max_samples, random_state
    )

    model_path = model_dir / f"one_class_svm_{subset}.joblib"
    joblib.dump({"model": model, "scaler": scaler, "train_scores": train_scores}, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    scored_df = score_dataframe(model, scaler, train_scores, val_df, feature_cols)

    metrics = evaluate(scored_df)
    metrics["model"]  = "one_class_svm"
    metrics["subset"] = subset
    metrics["config"] = {"nu": nu, "gamma": gamma, "max_train_samples": max_samples}

    print_metrics(metrics, "One-Class SVM")
    save_results(scored_df, metrics, config, "one_class_svm")
    print("[SUCCESS] One-Class SVM pipeline complete.")
    return metrics
