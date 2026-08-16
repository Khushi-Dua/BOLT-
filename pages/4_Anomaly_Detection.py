import pandas as pd
import streamlit as st

from utils import load_metadata, load_anomaly_table

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection")

st.markdown(
    "Detectors are scored on a **labelled synthetic-fault benchmark** "
    "(spikes, dips, drift, flatline, noise injected into a copy of the "
    "clean test period), not on the unlabelled real test set — the "
    "notebook is explicit that this is a falsifiable, reproducible "
    "benchmark rather than a claim about real-world anomalies."
)

meta = load_metadata()
best_det = meta.get("best_anomaly_detector")

anom = load_anomaly_table()
if anom.empty:
    st.warning(
        "No `results/anomaly_benchmark.csv` found. Export it from Section "
        "10.5 of the notebook with:\n\n"
        "`bench_scores_df.to_csv('results/anomaly_benchmark.csv', index=False)`"
    )
    st.stop()

if best_det:
    st.info(f"Best detector this run, by the notebook's own selection: **{best_det}**")

st.subheader("Detector comparison")
st.dataframe(
    anom.sort_values("F1", ascending=False).style.format(precision=4),
    width="stretch",
)
st.bar_chart(anom.set_index("Detector")[["Precision", "Recall", "F1"]], height=340)

st.caption(
    "These numbers are honest and, deliberately, not flattering — F1 in "
    "the 0.1–0.2 range is expected for this benchmark rather than a bug. "
    "Report them as-is rather than the old notebook's threshold-based "
    "'anomaly detected' percentages, which weren't measuring real "
    "detection accuracy against any ground truth."
)
