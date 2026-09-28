"""
utils.py

Shared helpers used by every model script:
    - loading the config
    - loading the feature parquet and splitting it into train / validation engines
    - picking which columns are model inputs
"""

import pandas as pd
from pathlib import Path
import yaml


# Columns that are never model inputs (ids, labels, op settings, split flag)
EXCLUDE_COLS = {
    "unit_id", "cycle", "RUL", "anomaly_label", "split",
    "op_setting_1", "op_setting_2", "op_setting_3",
}


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the engineered feature columns used as model input
    (raw sensors + rolling / cumulative features).
    """
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_train_val(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training feature file and split it by the engine-level
    'split' column created during preprocessing.

    Returns (train_df, val_df). No engine appears in both.
    """
    features_path = Path(config["dataset"]["features_path"])
    subset        = config["dataset"].get("subset", "FD001")
    train_file    = features_path / f"train_{subset}.parquet"

    if not train_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Features not found: {train_file}\n"
            f"        Run 'cmapps-tad preprocess' and 'cmapps-tad features' first."
        )

    df = pd.read_parquet(train_file)
    if "split" not in df.columns:
        raise ValueError(
            "[ERROR] Feature file has no 'split' column. "
            "Re-run 'cmapps-tad preprocess' and 'cmapps-tad features'."
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    # Sanity check: an engine must never be in both splits
    overlap = set(train_df["unit_id"]) & set(val_df["unit_id"])
    assert not overlap, f"Engines in both train and val: {sorted(overlap)}"

    print(f"[INFO] Train: {train_df['unit_id'].nunique()} engines, {len(train_df):,} rows")
    print(f"[INFO] Val  : {val_df['unit_id'].nunique()} engines, {len(val_df):,} rows")
    return train_df, val_df
