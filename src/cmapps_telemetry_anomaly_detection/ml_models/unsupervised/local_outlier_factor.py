"""
local_outlier_factor.py

Local Outlier Factor (LOF) anomaly detection model for CMAPSS.

Idea:
    LOF compares the local density around a point to the density around its
    neighbours. A point sitting in a much sparser region than its neighbours
    gets a high LOF score -> anomalous.

    Unlike Isolation Forest (global), LOF is a *local* density method, so it can
    catch points that are unusual relative to their own neighbourhood.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
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

def train_lof(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> tuple[LocalOutlierFactor, StandardScaler, np.ndarray]:
    """
    Train LOF on HEALTHY cycles only.

    novelty=True is required so the fitted model can score NEW data
    (the validation engines). Without it, LOF can only score its training set.
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_neighbors  : {n_neighbors} | contamination: {contamination}")

    # Scale first — LOF is distance-based
    scaler = StandardScaler()
    X_train = scaler.fit_transform(healthy_df[feature_cols].values)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)

    # score_samples: higher = more normal, so flip it
    train_scores = -model.score_samples(X_train)

    print("[INFO] LOF training complete.")
    return model, scaler, train_scores


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: LocalOutlierFactor,
    scaler: StandardScaler,
    train_scores: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    anomaly_score   : LOF score normalized using the TRAINING score range
    predicted_label : 1 = anomaly, 0 = normal  (from model.predict)
    """
    X = scaler.transform(df[feature_cols].values)

    raw_scores  = -model.score_samples(X)
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

def run_local_outlier_factor(config_path: str = "configs/data.yaml") -> dict:
    """
    Full LOF train + score + evaluate pipeline.
    Called by:  cmapps-tad train --model local_outlier_factor
    """
    config        = load_config(config_path)
    subset        = config["dataset"].get("subset", "FD001")
    model_dir     = Path(config["artifacts"]["model_path"])
    params        = config["models"]["local_outlier_factor"]
    n_neighbors   = params.get("n_neighbors", 20)
    contamination = params.get("contamination", 0.05)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    model, scaler, train_scores = train_lof(train_df, feature_cols, n_neighbors, contamination)

    model_path = model_dir / f"local_outlier_factor_{subset}.joblib"
    joblib.dump({"model": model, "scaler": scaler, "train_scores": train_scores}, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    scored_df = score_dataframe(model, scaler, train_scores, val_df, feature_cols)

    metrics = evaluate(scored_df)
    metrics["model"]  = "local_outlier_factor"
    metrics["subset"] = subset
    metrics["config"] = {"n_neighbors": n_neighbors, "contamination": contamination}

    print_metrics(metrics, "Local Outlier Factor")
    save_results(scored_df, metrics, config, "local_outlier_factor")
    print("[SUCCESS] Local Outlier Factor pipeline complete.")
    return metrics
