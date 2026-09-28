"""
isolation_forest.py

Isolation Forest anomaly detection model for CMAPSS FD001.

Training strategy:
    - Train ONLY on healthy cycles (anomaly_label == 0, RUL > threshold)
    - Should learn the "normal" behavior of engines so that deviations get high anomaly scores.
    - At scoring time, anything that deviates from normal gets a high anomaly score and is flagged as anomalous.


"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
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

def train_isolation_forest(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest on HEALTHY cycles only.

    Parameters:
        n_estimators   : number of trees (more = more stable, slower)
        contamination  : expected proportion of anomalies in scoring data
                         (used to set the decision threshold internally)
        random_state   : for reproducibility
    """
    # Filter to healthy cycles only
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total rows: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_estimators : {n_estimators} | contamination: {contamination}")

    X_train = healthy_df[feature_cols].values

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train)

    print("[INFO] Isolation Forest training complete.")
    return model


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: IsolationForest,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Generate anomaly scores for every row in the DataFrame.

    Isolation Forest outputs:
        decision_function score : more negative = more anomalous
        binary prediction       : -1 = anomaly, 1 = normal

    We convert to:
        anomaly_score : higher = more anomalous  (flipped + normalized to [0, 1])
        predicted_label : 1 = anomaly, 0 = normal  (converted from -1/1)
    """
    X = df[feature_cols].values

    raw_scores = model.decision_function(X)     # more negative = more anomalous
    predictions = model.predict(X)              # -1 = anomaly, 1 = normal

    # Flip so higher = more anomalous, then normalize to [0, 1]
    flipped = -raw_scores
    normalized = (flipped - flipped.min()) / (flipped.max() - flipped.min() + 1e-9)

    # Convert -1/1 predictions to 1/0
    binary_preds = (predictions == -1).astype(int)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = normalized
    scored_df["predicted_label"] = binary_preds

    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_isolation_forest(config_path: str = "configs/data.yaml") -> dict:
    """
    Full Isolation Forest train + score + evaluate pipeline.

    Trains on healthy cycles of the TRAIN engines, then evaluates on the
    held-out VALIDATION engines (never seen during training).

    Called by:  cmapps-tad train --model isolation_forest
    """
    config        = load_config(config_path)
    subset        = config["dataset"].get("subset", "FD001")
    model_dir     = Path(config["artifacts"]["model_path"])
    n_estimators  = config["models"]["isolation_forest"].get("n_estimators", 100)
    contamination = config["models"]["isolation_forest"].get("contamination", 0.05)
    random_state  = config["models"].get("random_state", 42)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val(config)
    feature_cols = get_feature_columns(train_df)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_isolation_forest(
        train_df,
        feature_cols,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    model_path = model_dir / f"isolation_forest_{subset}.joblib"
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    # ── Score + evaluate on validation engines ────────────────────────────────
    scored_df = score_dataframe(model, val_df, feature_cols)
    metrics = evaluate(scored_df)
    metrics["model"]  = "isolation_forest"
    metrics["subset"] = subset
    metrics["config"] = {
        "n_estimators" : n_estimators,
        "contamination": contamination,
        "random_state" : random_state,
    }

    print_metrics(metrics, "Isolation Forest")
    save_results(scored_df, metrics, config, "isolation_forest")
    print("[SUCCESS] Isolation Forest pipeline complete.")
    return metrics
