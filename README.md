# CMAPSS Telemetry Anomaly Detection & RUL Prediction

Anomaly detection and Remaining Useful Life (RUL) prediction on NASA's C-MAPSS
simulated jet engine data (FD001).

- **7 unsupervised anomaly detectors** trained only on healthy engine cycles and
  compared on held-out engines
- **RUL regression** (how many cycles an engine has left), evaluated on NASA's
  official test set

## Dataset

C-MAPSS contains run-to-failure sensor data for simulated turbofan engines. Each row
is one operating cycle of one engine: 3 operating settings and 21 sensors.
FD001 has 100 training engines (run until failure) and 100 test engines (cut off
before failure, with the true remaining life given in `RUL_FD001.txt`).

## Pipeline

```
data/raw  ->  preprocess  ->  data/processed  ->  features  ->  data/features  ->  models
```

1. **Preprocess** — name columns, drop the 7 constant sensors, compute RUL
   (`max_cycle - cycle`), create proxy anomaly labels, split engines into
   train/validation, and scale with MinMaxScaler.
2. **Features** — per engine: rolling mean/std (10-cycle window), rate of change,
   cumulative mean/std. 84 features in total. All rolling windows only look
   backwards, so no feature uses future cycles.
3. **Models** — anomaly detectors and RUL regressors (below).

### Train / validation split

Engines are split **by engine ID** (80 train / 20 validation), not by row.
Consecutive cycles of one engine are nearly identical, so a random row split would let
a model memorise an engine in training and then get "tested" on its neighbouring cycles.
The scaler is also fit on training engines only.

## Anomaly detection

C-MAPSS has no real anomaly labels, so a **proxy label** is used: a cycle is
"anomalous" if the engine is within 30 cycles of failure (`RUL <= 30`). Every model is
trained only on healthy cycles (`RUL > 30`) of the training engines, then scored on the
20 validation engines.

Results on the validation engines (from `reports/tables/anomaly_model_comparison_FD001.csv`):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | **0.982** | **0.851** | 0.748 | 0.915 | **0.823** |
| One-Class SVM | 0.968 | 0.835 | 0.684 | 0.860 | 0.762 |
| K-Means (distance to centroid) | 0.935 | 0.790 | 0.636 | 0.789 | 0.704 |
| Gaussian Mixture | 0.813 | 0.641 | 0.401 | 0.653 | 0.497 |
| Local Outlier Factor | 0.867 | 0.629 | 0.543 | 0.689 | 0.607 |
| PCA Reconstruction | 0.689 | 0.303 | 0.446 | 0.248 | 0.319 |
| Autoencoder (MLP, 84→32→8→32→84) | 0.647 | 0.234 | 0.311 | 0.152 | 0.204 |

PR-AUC is the main metric since only ~15% of cycles are labelled anomalous.

**Why the reconstruction models (PCA, autoencoder) do worst:** as an engine degrades,
its sensors move *along* the main direction of normal variation — on the validation
engines, position along the first principal component correlates −0.66 with RUL, while
PCA reconstruction error correlates only −0.02. The degraded cycles stay inside the
subspace the model learned, so they reconstruct fine and don't look anomalous.
Distance- and isolation-based methods don't have that blind spot.

## RUL prediction

The target RUL is capped at 125 cycles. Early in an engine's life the sensors barely
change, so there's no signal to tell "300 cycles left" from "200 cycles left"; capping
lets the model focus on the degradation phase. Features are the same 84 features plus
the current cycle.

Evaluation follows the standard C-MAPSS setup: one prediction per test engine at its
last observed cycle, compared to NASA's true RUL (capped at 125, matching training).
The best model is chosen by validation RMSE, not test RMSE.

Results on the 100 official test engines (from `reports/metrics/rul_FD001.json`):

| Model | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|
| Linear Regression | 15.93 | 12.76 | 0.842 | 393.4 |
| Random Forest | 13.72 | 9.58 | 0.883 | 383.9 |
| **Gradient Boosting** | **12.53** | **8.70** | **0.902** | **357.1** |

The NASA score is the asymmetric scoring function from the original PHM08 challenge:
predicting *later* than the true failure is penalised more than predicting early,
since overestimating remaining life is the dangerous mistake. Lower is better.

## Usage

```bash
pip install -e .

cmapps-tad download            # download from Kaggle (needs ~/.kaggle/kaggle.json)
cmapps-tad preprocess
cmapps-tad features

cmapps-tad train --model all   # train + evaluate all 7 anomaly models
cmapps-tad train --model isolation_forest   # or just one
cmapps-tad compare             # rebuild the comparison table
cmapps-tad train-rul           # train + evaluate the RUL models
```

Settings (window size, RUL cap, anomaly threshold, model hyperparameters) are in
`configs/data.yaml`.

Outputs:
- `reports/metrics/` — metrics JSON for each model
- `reports/tables/` — per-cycle anomaly scores, comparison table, RUL test predictions
- `artifacts/models/` — trained models

## Project structure

```
src/cmapps_telemetry_anomaly_detection/
├── cli.py
├── data_ingestion/kaggle_data_import.py
├── data_preprocessing/data_preprocess.py
├── feature_extraction/feature_extraction.py
└── ml_models/
    ├── utils.py                      # config, feature columns, train/val loading
    ├── unsupervised/                 # the 7 anomaly detectors
    ├── supervised/rul_regression.py  # RUL models
    └── evaluations/                  # shared metrics + comparison table
```

## Limitations

- The anomaly labels are a proxy (RUL ≤ 30), not real labelled faults, so these metrics
  measure how well each score separates near-failure cycles from healthy ones.
- C-MAPSS is simulated data; real sensor data is noisier and has missing values.
- Only FD001 (one operating condition, one fault mode) is used.
