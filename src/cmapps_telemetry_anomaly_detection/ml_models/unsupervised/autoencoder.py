"""
autoencoder.py

Autoencoder anomaly detection model for CMAPSS.

Idea:
    A neural network is trained to reproduce its own input through a narrow
    "bottleneck" layer (e.g. 84 features -> 32 -> 8 -> 32 -> 84). Trained only
    on healthy data, it learns to compress and reconstruct *healthy* behaviour.
    Degraded cycles reconstruct poorly -> high reconstruction error -> anomalous.

    This is the non-linear version of PCA Reconstruction: PCA can only learn a
    linear subspace, while the autoencoder can learn a curved one.

Implemented with scikit-learn's MLPRegressor (input = target = X).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
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

def reconstruction_error(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per row."""
    X_recon = model.predict(X)
    return np.mean((X - X_recon) ** 2, axis=1)


def train_autoencoder(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    hidden_layers: tuple = (32, 8, 32),
    max_iter: int = 200,
    random_state: int = 42,
) -> tuple[MLPRegressor, StandardScaler, np.ndarray]:
    """
    Train the autoencoder on HEALTHY cycles only.

    early_stopping holds out 10% of the healthy rows and stops training once the
    reconstruction loss on them stops improving (prevents overfitting).
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows  : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Architecture  : {len(feature_cols)} -> "
          f"{' -> '.join(str(h) for h in hidden_layers)} -> {len(feature_cols)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(healthy_df[feature_cols].values)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        max_iter=max_iter,
        early_stopping=True,
        random_state=random_state,
    )
    model.fit(X_train, X_train)  # input == target

    train_errors = reconstruction_error(model, X_train)

    print(f"[INFO] Epochs run            : {model.n_iter_}")
    print(f"[INFO] Mean train recon error: {train_errors.mean():.4f}")
    print("[INFO] Autoencoder training complete.")
    return model, scaler, train_errors


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: MLPRegressor,
    scaler: StandardScaler,
    train_errors: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold_percentile: float = 95.0,
) -> pd.DataFrame:
    X = scaler.transform(df[feature_cols].values)
    errors = reconstruction_error(model, X)

    threshold = np.percentile(train_errors, threshold_percentile)
    min_e, max_e = train_errors.min(), train_errors.max()
    normalized = (errors - min_e) / (max_e - min_e + 1e-9)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = np.clip(normalized, 0, None)
    scored_df["predicted_label"] = (errors > threshold).astype(int)
    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_autoencoder(config_path: str = "configs/data.yaml", subset: str | None = None) -> dict:
    """
    Full Autoencoder train + score + evaluate pipeline.
    Called by:  cmapps-tad train --model autoencoder
    """
    config        = load_config(config_path, subset)
    subset        = config["dataset"].get("subset", "FD001")
    model_dir     = Path(config["artifacts"]["model_path"])
    params        = config["models"]["autoencoder"]
    hidden_layers = tuple(params.get("hidden_layers", [32, 8, 32]))
    max_iter      = params.get("max_iter", 200)
    threshold_pct = params.get("threshold_percentile", 95.0)
    random_state  = config["models"].get("random_state", 42)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    model, scaler, train_errors = train_autoencoder(
        train_df, feature_cols, hidden_layers, max_iter, random_state
    )

    model_path = model_dir / f"autoencoder_{subset}.joblib"
    joblib.dump({"model": model, "scaler": scaler, "train_errors": train_errors}, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    scored_df = score_dataframe(model, scaler, train_errors, val_df, feature_cols, threshold_pct)

    metrics = evaluate(scored_df)
    metrics["model"]  = "autoencoder"
    metrics["subset"] = subset
    metrics["config"] = {
        "hidden_layers": list(hidden_layers),
        "epochs_run": int(model.n_iter_),
        "threshold_percentile": threshold_pct,
    }

    print_metrics(metrics, "Autoencoder")
    save_results(scored_df, metrics, config, "autoencoder")
    print("[SUCCESS] Autoencoder pipeline complete.")
    return metrics
