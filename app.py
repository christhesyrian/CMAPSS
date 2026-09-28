"""
app.py

Bare-bones Streamlit app for browsing the pipeline results.

Run:
    pip install -e ".[app]"
    streamlit run app.py

It only reads the outputs in reports/metrics and reports/tables — no models are
trained here. To regenerate them, run the pipeline (see README).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

METRICS_DIR = Path("reports/metrics")
TABLES_DIR = Path("reports/tables")
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

MODEL_NAMES = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "kmeans": "K-Means",
    "gaussian_mixture": "Gaussian Mixture",
    "local_outlier_factor": "Local Outlier Factor",
    "pca_reconstruction": "PCA Reconstruction",
    "autoencoder": "Autoencoder",
    "ensemble_top3": "Ensemble (top 3)",
    "ensemble_all": "Ensemble (all 7)",
}
RUL_MODEL_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ensemble_average": "Ensemble (average)",
}


# ─────────────────────────────────────────────
# Data loading (cached so switching tabs is instant)
# ─────────────────────────────────────────────

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ─────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────

st.set_page_config(page_title="CMAPSS Anomaly Detection & RUL", layout="wide")
st.title("CMAPSS Anomaly Detection & RUL Prediction")
st.caption("Results on NASA's C-MAPSS simulated jet engine data. "
           "Anomaly models are scored on held-out validation engines; "
           "RUL models on NASA's official test engines.")

if not TABLES_DIR.exists() or not any(TABLES_DIR.glob("*.csv")):
    st.error("No results found in reports/tables. Run the pipeline first (see README).")
    st.stop()

subset = st.sidebar.selectbox("Dataset subset", SUBSETS)
st.sidebar.markdown(
    "- **FD001**: 1 flight condition, 1 fault mode\n"
    "- **FD002**: 6 flight conditions, 1 fault mode\n"
    "- **FD003**: 1 flight condition, 2 fault modes\n"
    "- **FD004**: 6 flight conditions, 2 fault modes"
)

tab_overview, tab_anomaly, tab_rul = st.tabs(["Overview", "Anomaly Detection", "RUL Prediction"])


# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Summary across all subsets")
    summary = load_csv(TABLES_DIR / "summary_all_subsets.csv")
    st.dataframe(
        summary.rename(columns={
            "subset": "Subset",
            "best_single_anomaly_model": "Best anomaly model",
            "best_single_pr_auc": "Best PR-AUC",
            "ensemble_all_pr_auc": "Ensemble (all 7) PR-AUC",
            "ensemble_top3_pr_auc": "Ensemble (top 3) PR-AUC",
            "best_rul_model": "Best RUL model",
            "best_rul_test_rmse": "RUL test RMSE",
            "best_rul_test_r2": "RUL test R²",
            "rul_ensemble_test_rmse": "RUL ensemble RMSE",
        }),
        hide_index=True,
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Anomaly detection — PR-AUC** (higher is better)")
        st.bar_chart(summary.set_index("subset")[
            ["best_single_pr_auc", "ensemble_top3_pr_auc", "ensemble_all_pr_auc"]
        ], stack=False)
    with right:
        st.markdown("**RUL prediction — test RMSE in cycles** (lower is better)")
        st.bar_chart(summary.set_index("subset")[
            ["best_rul_test_rmse", "rul_ensemble_test_rmse"]
        ], stack=False)


# ── Anomaly detection ─────────────────────────────────────────────────────────
with tab_anomaly:
    st.subheader(f"Anomaly model comparison — {subset}")
    st.caption("Labels are a proxy: a cycle counts as anomalous if the engine is "
               "within 30 cycles of failure (RUL ≤ 30).")

    comparison = load_csv(TABLES_DIR / f"anomaly_model_comparison_{subset}.csv")
    comparison["model"] = comparison["model"].map(MODEL_NAMES)
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    # sort=False keeps the table's order (best PR-AUC first) instead of alphabetical
    st.bar_chart(comparison.set_index("model")[["pr_auc", "roc_auc"]],
                 stack=False, horizontal=True, sort=False)

    st.subheader("Anomaly score over an engine's life")
    col_model, col_engine = st.columns(2)
    model_key = col_model.selectbox(
        "Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get
    )
    scores = load_csv(TABLES_DIR / f"{model_key}_{subset}_scores.csv")
    engine = col_engine.selectbox("Validation engine", sorted(scores["unit_id"].unique()))

    engine_df = scores[scores["unit_id"] == engine].sort_values("cycle")
    st.line_chart(
        engine_df.set_index("cycle")[["anomaly_score", "anomaly_label"]],
        x_label="Cycle",
    )

    near_failure = engine_df[engine_df["anomaly_label"] == 1]
    healthy      = engine_df[engine_df["anomaly_label"] == 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Engine life (cycles)", int(engine_df["cycle"].max()))
    c2.metric("Near-failure cycles caught",
              f"{int(near_failure['predicted_label'].sum())} / {len(near_failure)}")
    c3.metric("False alarms on healthy cycles",
              f"{int(healthy['predicted_label'].sum())} / {len(healthy)}")
    st.caption("anomaly_label steps to 1 for the last 30 cycles before failure. A good "
               "model's score rises around the same point. Scores are often high for the "
               "first few cycles too: the rolling features have almost no history yet.")


# ── RUL prediction ────────────────────────────────────────────────────────────
with tab_rul:
    st.subheader(f"Remaining Useful Life — {subset}")
    rul = load_json(METRICS_DIR / f"rul_{subset}.json")
    cap = rul["rul_cap"]

    rows = []
    for key, result in rul["models"].items():
        t = result["test"]
        rows.append({
            "Model": RUL_MODEL_NAMES.get(key, key),
            "RMSE": t["rmse"],
            "MAE": t["mae"],
            "R²": t["r2"],
            "NASA score": t["nasa_score"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Scored on {rul['n_test_engines']} official test engines "
               f"(one prediction per engine, at its last observed cycle). "
               f"Best model by validation RMSE: "
               f"{RUL_MODEL_NAMES.get(rul['best_model_by_val_rmse'])}.")

    rul_model = st.selectbox(
        "Model", list(RUL_MODEL_NAMES), index=2, format_func=RUL_MODEL_NAMES.get
    )
    preds = load_csv(TABLES_DIR / f"rul_test_predictions_{subset}.csv")
    preds["actual_rul"] = preds["true_RUL"].clip(upper=cap)
    preds["predicted_rul"] = preds[f"pred_{rul_model}"]

    st.markdown("**Predicted vs actual RUL** (points on the diagonal are perfect predictions)")
    st.scatter_chart(preds, x="actual_rul", y="predicted_rul",
                     x_label="Actual RUL (cycles)", y_label="Predicted RUL (cycles)")
    st.caption(f"Actual RUL is capped at {cap} cycles, matching the training target.")

    preds["error"] = (preds["predicted_rul"] - preds["actual_rul"]).round(2)
    st.markdown("**Largest errors**")
    st.dataframe(
        preds.reindex(preds["error"].abs().sort_values(ascending=False).index)
        [["unit_id", "actual_rul", "predicted_rul", "error"]].head(10),
        hide_index=True,
        use_container_width=True,
    )
