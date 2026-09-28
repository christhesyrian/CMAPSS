import argparse
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
from cmapps_telemetry_anomaly_detection.ml_models.supervised.rul_regression import run_rul_regression
from cmapps_telemetry_anomaly_detection.ml_models.evaluations.compare import run_comparison

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
        help="Which model to train ('all' trains every model)",
    )
    sub.add_parser("train-rul", help="Train + evaluate Remaining Useful Life models")
    sub.add_parser("compare", help="Build a comparison table of all anomaly models")
    sub.add_parser("score", help="Score anomalies")
    sub.add_parser("preprocess", help="Preprocess raw CMAPSS data")
    sub.add_parser("features", help="Extract features from processed data")

    args = p.parse_args()

    if args.cmd == "download":
        download_dataset(force=args.force)  # pass force flag
    elif args.cmd == "train":
        if args.model == "all":
            for run in ANOMALY_MODELS.values():
                run()
            run_comparison()
        else:
            ANOMALY_MODELS[args.model]()
    elif args.cmd == "train-rul":
        run_rul_regression()
    elif args.cmd == "compare":
        run_comparison()
    elif args.cmd == "score":
        print("TODO: score anomalies")
    elif args.cmd is None:
        print("No command provided. Use --help for usage information.")
    elif args.cmd == "preprocess":
        run_preprocessing()
    elif args.cmd == "features":
        run_feature_extraction()
    else:
        p.print_help()
