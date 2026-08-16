"""
BOLT / AT-CBGRU dashboard — Home page.

Run with:  streamlit run app.py

Expects a `results/` folder next to this file containing the artefacts
produced by Section 11 of BOLT_Forecasting_AnomalyDetection.ipynb, PLUS
scaler.pkl and shap_export.json — see notebook_export_addon.py for the
extra cells that generate those two files (the original notebook doesn't
save them).
"""

import streamlit as st

from utils import load_metadata, load_all_predictions, RESULTS_DIR

st.set_page_config(
    page_title="BOLT — AT-CBGRU Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

meta = load_metadata()
preds = load_all_predictions()

st.title("⚡ BOLT — Building Energy Forecasting")
st.caption("Attention-based CNN–BiLSTM–GRU hybrid, with a shared fairness protocol across 12 models.")

if not meta:
    st.warning(
        f"No `run_metadata.json` found in `{RESULTS_DIR}/`. "
        "Run the notebook through Section 11.3, download `bolt_results.zip`, "
        "and unzip it into a `results/` folder next to this app."
    )
    st.stop()

fm = meta.get("forecast_metrics", {})
proposed = meta.get("proposed_model", "AT-CBGRU (proposed)")
champion = meta.get("champion_by_mae", proposed)
proposed_metrics = fm.get(proposed, {})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MAE (kWh)", f"{proposed_metrics.get('MAE', float('nan')):.4f}")
with col2:
    st.metric("RMSE (kWh)", f"{proposed_metrics.get('RMSE', float('nan')):.4f}")
with col3:
    mape = proposed_metrics.get("MAPE (%)", proposed_metrics.get("MAPE", float("nan")))
    st.metric("MAPE", f"{mape:.2f}%")
with col4:
    r2 = proposed_metrics.get("R2", float("nan"))
    st.metric("R²", f"{r2:.4f}")

if champion != proposed:
    st.info(
        f"Heads up: the current run's champion-by-MAE is **{champion}**, not "
        f"the proposed model ({proposed}). The notebook prints this same "
        "warning — report whichever model actually won this run."
    )

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Actual vs. Predicted — test period")
    if not preds.empty:
        cols_available = [c for c in preds.columns if c not in ("timestamp", "actual")]
        default_models = [m for m in [proposed, "BiLSTM", "GRU"] if m in cols_available][:3]
        chosen = st.multiselect(
            "Models to overlay", cols_available, default=default_models or cols_available[:2]
        )
        plot_df = preds.set_index("timestamp")[["actual"] + chosen]
        st.line_chart(plot_df, height=380)
    else:
        st.info("No `all_model_predictions.csv` found yet.")

with right:
    st.subheader("Dataset")
    ds = meta.get("dataset", {})
    st.markdown(
        f"""
        - **Building:** 779
        - **Hours:** {ds.get('n_hours', '—')}
        - **Range:** {ds.get('start', '—')} → {ds.get('end', '—')}
        - **Lookback window:** {meta.get('config', {}).get('TIME_STEPS', 24)} h
        - **Split:** {meta.get('split', {})}
        """
    )
    st.caption(
        "Single building, single year (2016). Do not generalise these "
        "numbers to buildings in the plural — the notebook is explicit "
        "about this limitation."
    )

st.divider()
st.caption(
    "Use the sidebar to open **Leaderboard**, **Forecast Explorer**, "
    "**Explainability (SHAP)**, and **Anomaly Detection**."
)
