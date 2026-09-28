import argparse
from cmapps_telemetry_anomaly_detection.config import SUBSETS
from cmapps_telemetry_anomaly_detection.data_ingestion.kaggle_data_import import download_dataset
from cmapps_telemetry_anomaly_detection.data_preprocessing.data_preprocess import run_preprocessing
from cmapps_telemetry_anomaly_detection.feature_extraction.feature_extraction import run_feature_extraction
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.isolation_forest import run_isolation_forest
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.pca_reconstruction import run_pca_reconstruction
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.kmeans import run_kmeans
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.local_outlier_factor import run_local_outlier_factor
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.one_class_svm import run_one_class_svm
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.gaussian_mixture import run_gaussian_mixture
from cmapps_telemetry_anomaly_detection.ml_models.unsupervised.autoencoder import run_autoencoder
from cmapps_telemetry_anomaly_detection.ml_models.ensemble.score_ensemble import run_ensemble
from cmapps_telemetry_anomaly_detection.ml_models.supervised.rul_regression import run_rul_regression
from cmapps_telemetry_anomaly_detection.ml_models.evaluations.compare import run_comparison, run_summary

# Model name -> training function
ANOMALY_MODELS = {
    "isolation_forest": run_isolation_forest,
    "pca_reconstruction": run_pca_reconstruction,
    "kmeans": run_kmeans,
    "local_outlier_factor": run_local_outlier_factor,
    "one_class_svm": run_one_class_svm,
    "gaussian_mixture": run_gaussian_mixture,
    "autoencoder": run_autoencoder,
}


def add_subset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subset",
        default=None,
        choices=[*SUBSETS, "all"],
        help="CMAPSS subset to use (default: dataset.subset in configs/data.yaml, 'all' = FD001-FD004)",
    )


def subsets_from_arg(subset: str | None) -> list:
    """'all' -> every subset, None -> config default, otherwise just that one."""
    if subset == "all":
        return SUBSETS
    return [subset]


def train_all_anomaly_models(subset: str | None) -> None:
    for run in ANOMALY_MODELS.values():
        run(subset=subset)
    run_ensemble(subset=subset)
    run_comparison(subset=subset)


def main():
    # Create top-level parser
    p = argparse.ArgumentParser(prog="cmapps-tad")
    # Add subcommands
    sub = p.add_subparsers(dest="cmd")

    # Create subcommands ONCE
    download_parser = sub.add_parser("download", help="Download CMAPSS dataset from Kaggle")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist (ignores .gitkeep)",
    )

    # Add other subcommands
    train_parser = sub.add_parser("train", help="Train an anomaly detection model")
    train_parser.add_argument(
        "--model",
        required=True,
        choices=[*ANOMALY_MODELS, "all"],
        help="Which model to train ('all' trains every model + the ensembles)",
    )
    add_subset_arg(train_parser)

    for name, help_text in [
        ("preprocess", "Preprocess raw CMAPSS data"),
        ("features", "Extract features from processed data"),
        ("ensemble", "Build ensembles from the trained anomaly models"),
        ("train-rul", "Train + evaluate Remaining Useful Life models"),
        ("compare", "Comparison table of all anomaly models ('--subset all' adds a summary)"),
    ]:
        add_subset_arg(sub.add_parser(name, help=help_text))

    sub.add_parser("score", help="Score anomalies")

    args = p.parse_args()

    if args.cmd is None:
        print("No command provided. Use --help for usage information.")
        return
    if args.cmd == "download":
        download_dataset(force=args.force)  # pass force flag
        return
    if args.cmd == "score":
        print("TODO: score anomalies")
        return

    for subset in subsets_from_arg(args.subset):
        if subset is not None:
            print(f"\n══════════════════════ {subset} ══════════════════════")
        if args.cmd == "preprocess":
            run_preprocessing(subset=subset)
        elif args.cmd == "features":
            run_feature_extraction(subset=subset)
        elif args.cmd == "train":
            if args.model == "all":
                train_all_anomaly_models(subset)
            else:
                ANOMALY_MODELS[args.model](subset=subset)
        elif args.cmd == "ensemble":
            run_ensemble(subset=subset)
        elif args.cmd == "train-rul":
            run_rul_regression(subset=subset)
        elif args.cmd == "compare":
            run_comparison(subset=subset)

    if args.cmd == "compare" and args.subset == "all":
        run_summary()
