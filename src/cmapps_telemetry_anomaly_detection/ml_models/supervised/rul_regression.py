"""
rul_regression.py

Remaining Useful Life (RUL) prediction for CMAPSS.

Where the anomaly models answer "is this engine behaving abnormally?", this
answers "how many cycles does this engine have left?".

Target:
    RUL = max_cycle_for_engine - current_cycle   (computed in preprocessing)

    The target is capped at `rul.cap` (default 125). Early in an engine's life the
    sensors barely change, so there is no signal to tell "300 cycles left" from
    "200 cycles left". Capping (a "piecewise-linear" target) is the standard fix
    for CMAPSS and lets the model focus on the degradation phase.

Models compared:
    - Linear Regression            (simple baseline)
    - Random Forest                (non-linear baseline)
    - Gradient Boosting (HistGBM)  (usually the strongest on tabular data)
    - Ensemble average             (mean of the three models' predictions)

Evaluation:
    1. Validation engines  — every cycle of the held-out engines
    2. Official NASA test  — one prediction per test engine at its LAST observed
                             cycle, compared to the true RUL in RUL_FD001.txt
                             (this is the standard CMAPSS benchmark setting).
                             The true RUL is capped at `rul.cap` too, matching
                             the training target.

Metrics: RMSE, MAE, R², and the NASA scoring function.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from cmapps_telemetry_anomaly_detection.ml_models.utils import (
    load_config,
    get_feature_columns,
    load_train_val,
)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA / PHM08 asymmetric scoring function (lower is better).

        d = predicted - true
        s = sum( exp(-d/13) - 1 )  if d < 0   (predicted too early)
            sum( exp( d/10) - 1 )  if d >= 0  (predicted too late)

    Late predictions are punished harder than early ones: predicting MORE life
    than an engine has means it could fail in service.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse"      : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae"       : round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2"        : round(float(r2_score(y_true, y_pred)), 4),
        "nasa_score": round(nasa_score(y_true, y_pred), 2),
        "n"         : int(len(y_true)),
    }


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def load_test_last_cycle(config: dict) -> pd.DataFrame:
    """
    Load test features and keep only each engine's LAST observed cycle,
    joined with the official true RUL for that cycle.
    """
    features_path  = Path(config["dataset"]["features_path"])
    processed_path = Path(config["dataset"]["processed_path"])
    subset         = config["dataset"].get("subset", "FD001")

    test_df = pd.read_parquet(features_path / f"test_{subset}.parquet")
    rul_df  = pd.read_parquet(processed_path / f"rul_{subset}.parquet")

    last_idx = test_df.groupby("unit_id")["cycle"].idxmax()
    last_df  = test_df.loc[last_idx].merge(rul_df, on="unit_id")
    return last_df


def build_models(config: dict) -> dict:
    random_state = config["models"].get("random_state", 42)
    rf_params    = config["rul"].get("random_forest", {})
    gb_params    = config["rul"].get("gradient_boosting", {})

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 12),
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=gb_params.get("max_iter", 300),
            learning_rate=gb_params.get("learning_rate", 0.05),
            max_depth=gb_params.get("max_depth", 6),
            random_state=random_state,
        ),
    }


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_rul_regression(config_path: str = "configs/data.yaml", subset: str | None = None) -> dict:
    """
    Train + evaluate all RUL models.
    Called by:  cmapps-tad train-rul
    """
    config      = load_config(config_path, subset)
    subset      = config["dataset"].get("subset", "FD001")
    rul_cap     = config["rul"].get("cap", 125)
    model_dir   = Path(config["artifacts"]["model_path"])
    metrics_dir = Path(config["reports"]["metrics_path"])
    tables_dir  = Path(config["reports"]["tables_path"])
    for d in [model_dir, metrics_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df, val_df = load_train_val(config)
    test_df = load_test_last_cycle(config)

    # Same engineered features as the anomaly models, plus the current cycle.
    # Engine age is legitimate information at prediction time and a strong RUL signal
    # (the anomaly models leave it out so they flag sensor behaviour, not just age).
    feature_cols = get_feature_columns(train_df) + ["cycle"]
    print(f"[INFO] Features: {len(feature_cols)} | RUL cap: {rul_cap}")

    X_train = train_df[feature_cols].values
    y_train = np.minimum(train_df["RUL"].values, rul_cap)
    X_val   = val_df[feature_cols].values
    y_val   = np.minimum(val_df["RUL"].values, rul_cap)
    X_test  = test_df[feature_cols].values
    # Official NASA ground truth, capped the same way as the training target
    # (standard practice for CMAPSS — the model is never trained to predict > cap)
    y_test  = np.minimum(test_df["true_RUL"].values, rul_cap)

    # ── Train + evaluate each model ───────────────────────────────────────────
    results = {}
    val_preds, test_preds = {}, {}
    test_predictions = test_df[["unit_id", "cycle", "true_RUL"]].copy()

    for name, model in build_models(config).items():
        print(f"\n[INFO] Training {name} ...")
        model.fit(X_train, y_train)

        # RUL can't be negative
        val_pred  = np.clip(model.predict(X_val), 0, None)
        test_pred = np.clip(model.predict(X_test), 0, None)

        results[name] = {
            "validation": rul_metrics(y_val, val_pred),
            "test"      : rul_metrics(y_test, test_pred),
        }
        val_preds[name], test_preds[name] = val_pred, test_pred
        test_predictions[f"pred_{name}"] = test_pred.round(2)

        joblib.dump(model, model_dir / f"rul_{name}_{subset}.joblib")
        print(f"[INFO] {name:18s} val RMSE: {results[name]['validation']['rmse']:.2f} "
              f"| test RMSE: {results[name]['test']['rmse']:.2f}")

    # ── Ensemble: average the predictions of all three models ─────────────────
    val_avg  = np.mean(list(val_preds.values()), axis=0)
    test_avg = np.mean(list(test_preds.values()), axis=0)
    results["ensemble_average"] = {
        "validation": rul_metrics(y_val, val_avg),
        "test"      : rul_metrics(y_test, test_avg),
    }
    test_predictions["pred_ensemble_average"] = test_avg.round(2)
    print(f"[INFO] {'ensemble_average':18s} val RMSE: {results['ensemble_average']['validation']['rmse']:.2f} "
          f"| test RMSE: {results['ensemble_average']['test']['rmse']:.2f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    best = min(results, key=lambda n: results[n]["validation"]["rmse"])
    output = {
        "subset": subset,
        "rul_cap": rul_cap,
        "n_features": len(feature_cols),
        "n_train_engines": int(train_df["unit_id"].nunique()),
        "n_val_engines": int(val_df["unit_id"].nunique()),
        "n_test_engines": int(len(test_df)),
        "best_model_by_val_rmse": best,
        "models": results,
    }

    metrics_path = metrics_dir / f"rul_{subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Metrics saved -> {metrics_path}")

    preds_path = tables_dir / f"rul_test_predictions_{subset}.csv"
    test_predictions.to_csv(preds_path, index=False)
    print(f"[INFO] Test predictions saved -> {preds_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── RUL Results (official test set, last cycle per engine) ──")
    print(f"  {'Model':20s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'NASA':>9s}")
    for name, r in results.items():
        t = r["test"]
        print(f"  {name:20s} {t['rmse']:7.2f} {t['mae']:7.2f} {t['r2']:7.3f} {t['nasa_score']:9.1f}")
    print(f"\n  Best model (chosen on validation RMSE): {best}")
    print("─────────────────────────────────────────────────────────\n")
    print("[SUCCESS] RUL regression complete.")
    return output
