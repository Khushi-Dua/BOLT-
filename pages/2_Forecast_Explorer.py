import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_metadata, load_all_predictions, load_champion_model, load_scaler,
    load_raw_series, build_latest_window, recursive_forecast, TIME_STEPS,
)

st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")
st.title("🔮 Forecast Explorer")

meta = load_metadata()
preds = load_all_predictions()
proposed = meta.get("proposed_model", "AT-CBGRU (proposed)")

tab_hist, tab_live = st.tabs(["Test-period residuals", "Live multi-step forecast"])

with tab_hist:
    if preds.empty:
        st.info("No `all_model_predictions.csv` found.")
    else:
        model = st.selectbox(
            "Model", [c for c in preds.columns if c not in ("timestamp", "actual")],
            index=0,
        )
        df = preds[["timestamp", "actual", model]].copy()
        df["residual"] = df["actual"] - df[model]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Actual vs {model}**")
            st.line_chart(df.set_index("timestamp")[["actual", model]], height=320)
        with c2:
            st.markdown("**Residuals over time**")
            st.line_chart(df.set_index("timestamp")[["residual"]], height=320)

        st.markdown("**Residual distribution**")
        hist_counts = pd.cut(df["residual"], bins=30).value_counts().sort_index()
        hist_counts.index = hist_counts.index.astype(str)  # Altair can't take Interval objects
        st.bar_chart(hist_counts, height=260)

with tab_live:
    st.markdown(
        "Forecasts forward from the **champion model's** last known window "
        "using a recursive sliding window. Only the meter-reading channel is "
        "rolled forward each step — lagged/calendar features are held from "
        "the last real hour, matching the simplification used in the "
        "notebook's own multi-step cell. Treat forecasts beyond a few hours "
        "as indicative rather than exact."
    )

    model = load_champion_model()
    scaler = load_scaler()
    series_df = load_raw_series()

    missing = []
    if model is None:
        missing.append("`results/at_cbgru_champion.keras`")
    if scaler is None:
        missing.append("`results/scaler.pkl` (see notebook_export_addon.py)")
    if series_df.empty:
        missing.append("`results/building_779_updated_meter_reading.csv`")

    if missing:
        st.warning("Missing for live forecasting: " + ", ".join(missing))
    else:
        horizon = st.slider("Hours to forecast", min_value=1, max_value=48, value=14)
        if st.button("Run forecast", type="primary"):
            with st.spinner("Building the latest window and forecasting..."):
                window = build_latest_window(series_df["meter_reading"], scaler)
                forecast = recursive_forecast(model, window, scaler, horizon)

            last_ts = series_df.index[-1]
            future_idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
            hist = series_df["meter_reading"].iloc[-72:]

            plot_df = pd.DataFrame(index=hist.index.union(future_idx))
            plot_df.loc[hist.index, "Recent actual"] = hist.values
            plot_df.loc[future_idx, "Forecast"] = forecast
            st.line_chart(plot_df, height=380)

            out = pd.DataFrame({"timestamp": future_idx, "forecast_kwh": forecast})
            st.dataframe(out, width="stretch")
            st.download_button(
                "Download forecast as CSV",
                out.to_csv(index=False).encode(),
                file_name="live_forecast.csv",
                mime="text/csv",
            )