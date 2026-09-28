# CMAPSS Telemetry Anomaly Detection & RUL Prediction

Anomaly detection and Remaining Useful Life (RUL) prediction on NASA's C-MAPSS
simulated jet engine data — all four subsets (FD001–FD004).

- **7 unsupervised anomaly detectors** trained only on healthy engine cycles, plus
  **2 ensembles** that combine them, compared on held-out engines
- **RUL regression** (how many cycles an engine has left), evaluated on NASA's
  official test sets

## Dataset

C-MAPSS contains run-to-failure sensor data for simulated turbofan engines. Each row
is one operating cycle of one engine: 3 operating settings and 21 sensors. Training
engines run until failure; test engines are cut off before failure, and the true
remaining life is given in `RUL_FDxxx.txt`.

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---:|---:|---:|---:|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

709 training engines (160k cycles) and 707 test engines in total.

## Pipeline

```
data/raw  ->  preprocess  ->  data/processed  ->  features  ->  data/features  ->  models
```

1. **Preprocess** — name columns, drop the 7 sensors with no degradation signal,
   compute RUL (`max_cycle - cycle`), create proxy anomaly labels, split engines into
   train/validation, normalize by operating condition (FD002/FD004), and scale with
   MinMaxScaler.
2. **Features** — per engine: rolling mean/std (10-cycle window), rate of change,
   cumulative mean/std. 84 features in total. All rolling windows only look
   backwards, so no feature uses future cycles.
3. **Models** — anomaly detectors, ensembles, and RUL regressors (below).

### Train / validation split

Engines are split **by engine ID** (80% train / 20% validation), not by row.
Consecutive cycles of one engine are nearly identical, so a random row split would let
a model memorise an engine in training and then get "tested" on its neighbouring cycles.
The scaler is also fit on training engines only.

### Operating conditions (FD002 / FD004)

FD002 and FD004 engines fly at 6 different altitude / Mach / throttle settings. In
FD002, 80–100% of each sensor's variation comes from *which condition the engine is
in*, not from wear — so without handling it, the models would learn to recognize
flight conditions instead of degradation.

The fix: cluster the 3 operating settings into 6 conditions (K-Means), then standardize
each sensor using the mean and standard deviation *of its condition*. A value of +2 then
means "2 standard deviations above normal for this flight condition", which is
comparable across conditions. Clusters and statistics are fit on training engines only.

## Anomaly detection

C-MAPSS has no real anomaly labels, so a **proxy label** is used: a cycle is
"anomalous" if the engine is within 30 cycles of failure (`RUL <= 30`). Every model is
trained only on healthy cycles (`RUL > 30`) of the training engines, then scored on the
validation engines.

**PR-AUC** on the validation engines (the main metric, since only ~15% of cycles are
labelled anomalous). Best per subset in bold:

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.851 | 0.824 | 0.798 | **0.760** |
| One-Class SVM | 0.835 | 0.774 | 0.774 | 0.650 |
| K-Means (distance to centroid) | 0.790 | 0.753 | 0.708 | 0.658 |
| Gaussian Mixture | 0.641 | 0.661 | 0.378 | 0.503 |
| Local Outlier Factor | 0.629 | 0.449 | 0.598 | 0.295 |
| PCA Reconstruction | 0.303 | 0.274 | 0.404 | 0.316 |
| Autoencoder (MLP, 84→32→8→32→84) | 0.234 | 0.227 | 0.264 | 0.257 |
| Ensemble (top 3) | **0.881** | **0.825** | **0.822** | 0.740 |
| Ensemble (all 7) | 0.729 | 0.702 | 0.670 | 0.581 |

**ROC-AUC** on the validation engines:

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Isolation Forest | **0.982** | **0.958** | **0.962** | **0.956** |
| One-Class SVM | 0.968 | 0.954 | 0.948 | 0.935 |
| K-Means | 0.935 | 0.929 | 0.928 | 0.922 |
| Gaussian Mixture | 0.813 | 0.852 | 0.781 | 0.852 |
| Local Outlier Factor | 0.867 | 0.781 | 0.863 | 0.727 |
| PCA Reconstruction | 0.689 | 0.713 | 0.785 | 0.767 |
| Autoencoder | 0.647 | 0.644 | 0.727 | 0.735 |
| Ensemble (top 3) | 0.973 | 0.955 | 0.958 | 0.950 |
| Ensemble (all 7) | 0.904 | 0.885 | 0.908 | 0.899 |

Full tables including precision / recall / F1 are in
`reports/tables/anomaly_model_comparison_<subset>.csv`.

**Why the reconstruction models (PCA, autoencoder) do worst:** as an engine degrades,
its sensors move *along* the main direction of normal variation — on FD001's validation
engines, position along the first principal component correlates −0.66 with RUL, while
PCA reconstruction error correlates only −0.02. The degraded cycles stay inside the
subspace the model learned, so they reconstruct fine and don't look anomalous.
Distance- and isolation-based methods don't have that blind spot.

### Is an ensemble better?

The ensembles combine the individual models' validation scores. Each model's raw scores
are on a different scale (tree path lengths, SVM distances, log-likelihoods), so each is
first converted to a percentile rank, and the ranks are averaged. The binary prediction
is a majority vote.

- **Ensemble (all 7)** — every model, no selection.
- **Ensemble (top 3)** — Isolation Forest, One-Class SVM and K-Means: the 3 best models
  on **FD001**. FD002–FD004 were not used to pick them, so those three subsets are a fair
  test of whether the ensemble generalizes. (Its FD001 score is optimistic for the same
  reason.)

Results:

- **Averaging all 7 models is worse than Isolation Forest alone on every subset.** The
  weak reconstruction and density models drag the average down.
- **The top-3 ensemble is not a consistent win.** On the three unbiased subsets it
  beats Isolation Forest's PR-AUC on FD003 (+0.024), ties on FD002 (0.825 vs 0.824),
  and loses on FD004 (−0.020). Isolation Forest also has the higher ROC-AUC everywhere.

So a single Isolation Forest is the better choice: it matches the best ensemble while
being one model instead of three.

## RUL prediction

The target RUL is capped at 125 cycles. Early in an engine's life the sensors barely
change, so there's no signal to tell "300 cycles left" from "200 cycles left"; capping
lets the model focus on the degradation phase. Features are the same 84 features plus
the current cycle.

Evaluation follows the standard C-MAPSS setup: one prediction per test engine at its
last observed cycle, compared to NASA's true RUL (capped at 125, matching training).
The best model is chosen by validation RMSE, not test RMSE — Gradient Boosting on all
four subsets.

**RMSE** on the official test sets (lower is better):

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Linear Regression | 15.93 | 15.55 | 15.76 | 17.14 |
| Random Forest | 13.72 | 12.88 | 15.06 | 14.34 |
| **Gradient Boosting** | **12.53** | **12.07** | **12.53** | 14.02 |
| Ensemble (average of the 3) | 12.63 | 12.10 | 13.34 | **13.80** |

Gradient Boosting in detail:

| Subset | Test engines | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|---:|
| FD001 | 100 | 12.53 | 8.70 | 0.902 | 357.1 |
| FD002 | 259 | 12.07 | 8.38 | 0.921 | 632.9 |
| FD003 | 100 | 12.53 | 8.37 | 0.898 | 343.4 |
| FD004 | 248 | 14.02 | 9.33 | 0.894 | 1099.3 |

FD004 is the hardest subset (6 operating conditions *and* 2 fault modes), and it's the
only one where Gradient Boosting's error rises noticeably.

**The averaged ensemble** matches Gradient Boosting on RMSE within about 0.8 cycles, but
has the best **NASA score** on FD001 (271 vs 357), FD002 (618 vs 633) and FD004 (889 vs
1099). The NASA score is the asymmetric scoring function from the original PHM08
challenge: it grows exponentially with the size of an error, and predicting *later* than
the true failure is penalised more than predicting early, since overestimating remaining
life is the dangerous mistake. Averaging likely helps here because it smooths out the
occasional large error from any single model, which the exponential penalty punishes
much harder than RMSE does.

## Usage

```bash
pip install -e .

cmapps-tad download                             # download from Kaggle (needs ~/.kaggle/kaggle.json)
cmapps-tad preprocess --subset all
cmapps-tad features   --subset all

cmapps-tad train --model all --subset all       # 7 anomaly models + both ensembles
cmapps-tad train-rul --subset all               # RUL models + averaged ensemble
cmapps-tad compare --subset all                 # comparison tables + cross-subset summary
```

`--subset` accepts `FD001`, `FD002`, `FD003`, `FD004` or `all` (default: `dataset.subset`
in the config). Single models can be trained with e.g.
`cmapps-tad train --model isolation_forest --subset FD002`.

Settings (window size, RUL cap, anomaly threshold, ensemble members, model
hyperparameters) are in `configs/data.yaml`.

### Results app

A small Streamlit app for browsing the results: a cross-subset overview, the anomaly
model comparison with each validation engine's anomaly score over its life, and
predicted vs. actual RUL for the test engines.

```bash
pip install -e ".[app]"
streamlit run app.py
```

It only reads the files in `reports/`, which are committed, so it works right after
cloning — no need to run the pipeline first.

Outputs:
- `reports/metrics/` — metrics JSON for each model and subset
- `reports/tables/` — per-cycle anomaly scores, comparison tables, RUL test predictions,
  and `summary_all_subsets.csv`
- `artifacts/models/` — trained models

## Project structure

```
app.py                                # Streamlit results app
src/cmapps_telemetry_anomaly_detection/
├── cli.py
├── config.py                         # config loading + --subset override
├── data_ingestion/kaggle_data_import.py
├── data_preprocessing/data_preprocess.py
├── feature_extraction/feature_extraction.py
└── ml_models/
    ├── utils.py                      # feature columns, train/val loading
    ├── unsupervised/                 # the 7 anomaly detectors
    ├── ensemble/score_ensemble.py    # rank-average + majority-vote ensembles
    ├── supervised/rul_regression.py  # RUL models
    └── evaluations/                  # shared metrics + comparison tables
```

## Limitations

- The anomaly labels are a proxy (RUL ≤ 30), not real labelled faults, so these metrics
  measure how well each score separates near-failure cycles from healthy ones.
- The ensembles rank each model's scores within the batch being scored, which is fine
  for evaluation but would need per-model reference distributions to score a single
  new cycle in production.
- C-MAPSS is simulated data; real sensor data is noisier and has missing values.
