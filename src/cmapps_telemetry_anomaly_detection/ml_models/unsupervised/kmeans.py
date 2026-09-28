"""
kmeans.py

K-Means Clustering anomaly detection model for CMAPSS FD001.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
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

def train_kmeans(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler]:
    """
    Train K-Means on HEALTHY cycles only.

    Parameters:
        n_clusters   : number of cluster centers to learn.
                       More clusters = finer-grained healthy regions.
                       8 is a solid default for FD001.
        random_state : reproducibility
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_clusters   : {n_clusters}")

    X_train = healthy_df[feature_cols].values

    # Scale first — critical for distance-based models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,    # run 10 initialisations to avoid bad local minima
        max_iter=300,
    )
    model.fit(X_scaled)

    print("[INFO] K-Means training complete.")
    return model, scaler


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: KMeans,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold_percentile: float = 95.0,
    train_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Score every row by distance to nearest cluster center.

    Threshold is set at the given percentile of TRAINING distances
    so the model flags the top X% most anomalous training points.
    Scores are normalized to [0, 1] for ensemble compatibility.
    """
    X = scaler.transform(df[feature_cols].values)

    # Distance to nearest cluster center
    distances = np.min(
        np.linalg.norm(X[:, np.newaxis] - model.cluster_centers_, axis=2),
        axis=1,
    )

    # Compute threshold from healthy training distances
    if train_df is not None:
        X_train = scaler.transform(
            train_df[train_df["anomaly_label"] == 0][feature_cols].values
        )
        train_distances = np.min(
            np.linalg.norm(X_train[:, np.newaxis] - model.cluster_centers_, axis=2),
            axis=1,
        )
        threshold = np.percentile(train_distances, threshold_percentile)
        min_d, max_d = train_distances.min(), train_distances.max()
    else:
        threshold = np.percentile(distances, threshold_percentile)
        min_d, max_d = distances.min(), distances.max()

    # Normalize scores to [0, 1]
    normalized = (distances - min_d) / (max_d - min_d + 1e-9)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = np.clip(normalized, 0, None)
    scored_df["predicted_label"] = (distances > threshold).astype(int)

    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_kmeans(config_path: str = "configs/data.yaml") -> dict:
    """
    Full K-Means train + score + evaluate pipeline.

    Trains on healthy cycles of the TRAIN engines, evaluates on VALIDATION engines.

    Called by:  cmapps-tad train --model kmeans
    """
    config        = load_config(config_path)
    subset        = config["dataset"].get("subset", "FD001")
    model_dir     = Path(config["artifacts"]["model_path"])
    n_clusters    = config["models"]["kmeans"].get("n_clusters", 8)
    threshold_pct = config["models"]["kmeans"].get("threshold_percentile", 95.0)
    random_state  = config["models"].get("random_state", 42)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    # Train
    model, scaler = train_kmeans(train_df, feature_cols, n_clusters, random_state)

    # Save model + scaler together
    model_path = model_dir / f"kmeans_{subset}.joblib"
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    # Score validation engines (threshold comes from healthy TRAIN distances)
    scored_df = score_dataframe(model, scaler, val_df, feature_cols,
                                threshold_pct, train_df)

    metrics = evaluate(scored_df)
    metrics["model"]  = "kmeans"
    metrics["subset"] = subset
    metrics["config"] = {"n_clusters": n_clusters, "threshold_percentile": threshold_pct}

    print_metrics(metrics, "K-Means")
    save_results(scored_df, metrics, config, "kmeans")
    print("[SUCCESS] K-Means pipeline complete.")
    return metrics
