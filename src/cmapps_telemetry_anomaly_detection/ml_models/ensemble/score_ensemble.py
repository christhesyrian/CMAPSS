"""
score_ensemble.py

Ensemble anomaly detector: combines the scores of the individual models to test
whether an ensemble beats the best single model.

How the scores are combined:
    Each model's raw score lives on a different scale (tree path lengths, SVM
    distances, log-likelihoods, ...), so averaging them directly would let one
    model dominate. Instead each model's scores are converted to a percentile
    rank (0 = most normal, 1 = most anomalous) and the ranks are averaged.

    anomaly_score   = mean of the members' percentile ranks
    predicted_label = majority vote of the members' binary predictions

Two ensembles are built:
    ensemble_all  : all 7 models (no model selection at all)
    ensemble_top3 : the members listed in config (ensemble.top_members). These
                    were picked from the FD001 validation results, so FD002-FD004
                    are an unbiased test of whether the ensemble generalizes.

Input: the per-row validation scores each model saved to
       reports/tables/<model>_<subset>_scores.csv
"""

import pandas as pd
from pathlib import Path

from cmapps_telemetry_anomaly_detection.ml_models.utils import load_config
from cmapps_telemetry_anomaly_detection.ml_models.evaluations.metrics import (
    evaluate,
    print_metrics,
    save_results,
)

ALL_MODELS = [
    "isolation_forest",
    "pca_reconstruction",
    "kmeans",
    "local_outlier_factor",
    "one_class_svm",
    "gaussian_mixture",
    "autoencoder",
]


# ─────────────────────────────────────────────
# Combining member scores
# ─────────────────────────────────────────────

def load_member_scores(tables_dir: Path, members: list[str], subset: str) -> pd.DataFrame:
    """
    Load each member's validation scores and join them on (unit_id, cycle).
    Returns one row per cycle with <model>_score and <model>_label columns.
    """
    combined = None
    for name in members:
        path = tables_dir / f"{name}_{subset}_scores.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] Missing scores for {name}: {path}\n"
                f"        Run 'cmapps-tad train --model {name} --subset {subset}' first."
            )
        df = pd.read_csv(path).rename(columns={
            "anomaly_score": f"{name}_score",
            "predicted_label": f"{name}_label",
        })
        if combined is None:
            combined = df
        else:
            combined = combined.merge(
                df[["unit_id", "cycle", f"{name}_score", f"{name}_label"]],
                on=["unit_id", "cycle"],
                how="inner",
            )
            if len(combined) != len(df):
                raise ValueError(f"[ERROR] {name} scored a different set of rows than the other models")
    return combined


def combine_scores(member_df: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Rank-average the scores and majority-vote the labels."""
    ranks = pd.DataFrame({
        name: member_df[f"{name}_score"].rank(pct=True) for name in members
    })
    votes = member_df[[f"{name}_label" for name in members]].sum(axis=1)

    scored_df = member_df[["unit_id", "cycle", "RUL", "anomaly_label"]].copy()
    scored_df["anomaly_score"]   = ranks.mean(axis=1)
    scored_df["predicted_label"] = (votes > len(members) / 2).astype(int)
    return scored_df


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def build_ensemble(config: dict, name: str, members: list[str]) -> dict:
    subset     = config["dataset"].get("subset", "FD001")
    tables_dir = Path(config["reports"]["tables_path"])

    print(f"[INFO] Building {name} from {len(members)} models: {members}")
    member_df = load_member_scores(tables_dir, members, subset)
    scored_df = combine_scores(member_df, members)

    metrics = evaluate(scored_df)
    metrics["model"]   = name
    metrics["subset"]  = subset
    metrics["members"] = members

    print_metrics(metrics, name)
    save_results(scored_df, metrics, config, name)
    return metrics


def run_ensemble(config_path: str = "configs/data.yaml", subset: str | None = None) -> dict:
    """
    Build both ensembles for one subset.
    Called by:  cmapps-tad ensemble   (and automatically by 'train --model all')
    """
    config      = load_config(config_path, subset)
    top_members = config["ensemble"]["top_members"]

    results = {
        "ensemble_all": build_ensemble(config, "ensemble_all", ALL_MODELS),
        "ensemble_top3": build_ensemble(config, "ensemble_top3", top_members),
    }
    print("[SUCCESS] Ensemble pipeline complete.")
    return results
