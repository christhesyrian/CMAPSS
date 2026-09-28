"""
gaussian_mixture.py

Gaussian Mixture Model (GMM) anomaly detection model for CMAPSS.

Idea:
    Model the distribution of healthy data as a mixture of Gaussians. Each new
    point gets a log-likelihood under that distribution. Low likelihood means
    "this doesn't look like healthy data" -> anomalous.

    Anomaly score = negative log-likelihood (higher = more anomalous).
    This is a probabilistic, density-based alternative to K-Means distance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
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

def train_gaussian_mixture(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 5,
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[GaussianMixture, StandardScaler, np.ndarray]:
    """
    Train GMM on HEALTHY cycles only.

    reg_covar adds a small value to the covariance diagonal. Many engineered
    features are highly correlated (e.g. rolling mean vs cumulative mean of the
    same sensor), which would otherwise make the covariance matrix near-singular.
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_components : {n_components} | covariance: {covariance_type}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(healthy_df[feature_cols].values)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=1e-3,
        max_iter=500,
        random_state=random_state,
    )
    model.fit(X_train)

    # score_samples = log-likelihood (higher = more normal), so flip it
    train_scores = -model.score_samples(X_train)

    print("[INFO] Gaussian Mixture training complete.")
    return model, scaler, train_scores


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: GaussianMixture,
    scaler: StandardScaler,
    train_scores: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold_percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Threshold = given percentile of the healthy TRAINING scores,
    so the top X% least-likely points get flagged.
    """
    X = scaler.transform(df[feature_cols].values)
    raw_scores = -model.score_samples(X)

    threshold = np.percentile(train_scores, threshold_percentile)
    min_s, max_s = train_scores.min(), train_scores.max()
    normalized = (raw_scores - min_s) / (max_s - min_s + 1e-9)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = np.clip(normalized, 0, None)
    scored_df["predicted_label"] = (raw_scores > threshold).astype(int)
    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_gaussian_mixture(config_path: str = "configs/data.yaml") -> dict:
    """
    Full Gaussian Mixture train + score + evaluate pipeline.
    Called by:  cmapps-tad train --model gaussian_mixture
    """
    config          = load_config(config_path)
    subset          = config["dataset"].get("subset", "FD001")
    model_dir       = Path(config["artifacts"]["model_path"])
    params          = config["models"]["gaussian_mixture"]
    n_components    = params.get("n_components", 5)
    covariance_type = params.get("covariance_type", "full")
    threshold_pct   = params.get("threshold_percentile", 95.0)
    random_state    = config["models"].get("random_state", 42)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    model, scaler, train_scores = train_gaussian_mixture(
        train_df, feature_cols, n_components, covariance_type, random_state
    )

    model_path = model_dir / f"gaussian_mixture_{subset}.joblib"
    joblib.dump({"model": model, "scaler": scaler, "train_scores": train_scores}, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    scored_df = score_dataframe(model, scaler, train_scores, val_df, feature_cols, threshold_pct)

    metrics = evaluate(scored_df)
    metrics["model"]  = "gaussian_mixture"
    metrics["subset"] = subset
    metrics["config"] = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "threshold_percentile": threshold_pct,
    }

    print_metrics(metrics, "Gaussian Mixture")
    save_results(scored_df, metrics, config, "gaussian_mixture")
    print("[SUCCESS] Gaussian Mixture pipeline complete.")
    return metrics
