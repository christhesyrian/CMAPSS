"""
config.py

Loads configs/data.yaml and applies a subset override, so every step of the
pipeline can be pointed at FD001, FD002, FD003 or FD004 from the CLI.
"""

import yaml

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def load_config(config_path: str = "configs/data.yaml", subset: str | None = None) -> dict:
    """
    Load the YAML config. If `subset` is given it overrides dataset.subset.

    Also fills in subset-specific values:
        - artifacts.scaler_path  ("{subset}" placeholder)
        - dataset.n_conditions   (number of operating conditions in the subset)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if subset is not None:
        if subset not in SUBSETS:
            raise ValueError(f"[ERROR] Unknown subset {subset!r}. Choose from {SUBSETS}.")
        config["dataset"]["subset"] = subset

    subset = config["dataset"].get("subset", "FD001")
    config["artifacts"]["scaler_path"] = config["artifacts"]["scaler_path"].format(subset=subset)
    config["dataset"]["n_conditions"] = config["dataset"]["operating_conditions"][subset]
    return config
