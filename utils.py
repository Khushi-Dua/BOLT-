"""
Shared loaders and helpers for the BOLT / AT-CBGRU Streamlit dashboard.

Everything here reads artefacts produced by Section 11.3 of
`BOLT_Forecasting_AnomalyDetection.ipynb` (the `results/` folder /
`bolt_results.zip`), plus two extra files you need to export from the
notebook that the original export cell did NOT save:

  results/scaler.pkl        -> the fitted StandardScaler (joblib dump)
  results/shap_export.json  -> precomputed SHAP values (see notebook_export_addon.py)

See notebook_export_addon.py in this same folder for the exact cells to
add to the notebook before you download bolt_results.zip again.
"""

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_DIR = os.environ.get("BOLT_RESULTS_DIR", "results")


# --------------------------------------------------------------------------- #
# Custom Keras layers used inside at_cbgru_champion.keras. The notebook
# defines these inline, so the saved model has no way to reconstruct them on
# load unless we redefine + register the exact same classes here first.
# Reverse-engineered from the saved config.json / weight shapes:
#   temporal_attention: input (B,24,64) -> Dense(32, tanh) -> Dense(1) ->
#                        softmax over time -> weighted sum -> (B,64)
#   persistence: window[:, -1, 0:1]  (last hour's raw 'reading' channel,
#                used as the residual-connection baseline; delta is added
#                to it via `residual_add`)
# --------------------------------------------------------------------------- #

def _register_custom_layers():
    import tensorflow as tf
    import keras

    @keras.saving.register_keras_serializable(package="atcbgru")
    class TemporalAttention(keras.layers.Layer):
        def __init__(self, units=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            hidden_units = self.units or 32
            self.hidden = keras.layers.Dense(hidden_units, activation="tanh", name="hidden")
            self.score = keras.layers.Dense(1, name="score")
            super().build(input_shape)

        def call(self, inputs):
            h = self.hidden(inputs)
            s = self.score(h)
            weights = tf.nn.softmax(s, axis=1)
            return tf.reduce_sum(weights * inputs, axis=1)

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config

    @keras.saving.register_keras_serializable(package="atcbgru")
    class PersistenceSlice(keras.layers.Layer):
        def call(self, inputs):
            return inputs[:, -1, 0:1]

    return {"TemporalAttention": TemporalAttention, "PersistenceSlice": PersistenceSlice}

FEATURE_NAMES = [
    "reading", "lag24", "lag168", "roll_mean24", "roll_std24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos", "is_weekend",
]

TIME_STEPS = 24
WEEK_LAG = 168


# --------------------------------------------------------------------------- #
# Artefact loaders (cached so the Streamlit app doesn't re-read disk on every
# interaction / page switch)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = os.path.join(RESULTS_DIR, "run_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_all_predictions() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "all_model_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_model_predictions(slug: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"pred_{slug}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_resource(show_spinner=False)
def load_champion_model():
    """Loads the saved AT-CBGRU .keras model. Requires tensorflow at runtime.

    The model contains two custom layers (TemporalAttention, PersistenceSlice)
    that aren't part of stock Keras, so they must be registered/passed as
    custom_objects or load_model raises "Could not locate class ..." even
    though the file itself is fine."""
    import tensorflow as tf
    path = os.path.join(RESULTS_DIR, "at_cbgru_champion.keras")
    if not os.path.exists(path):
        return None
    custom_objects = _register_custom_layers()
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


@st.cache_resource(show_spinner=False)
def load_autoencoder():
    import tensorflow as tf
    path = os.path.join(RESULTS_DIR, "lstm_autoencoder.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Requires you to have added `joblib.dump(D['scaler'], ...)` in the
    notebook — see notebook_export_addon.py. Falls back to None."""
    path = os.path.join(RESULTS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_shap_export() -> dict:
    """Requires the SHAP export cell from notebook_export_addon.py.
    Returns {} if not present so the Explainability page can show
    an instruction message instead of crashing."""
    path = os.path.join(RESULTS_DIR, "shap_export.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_ablation_table() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "ablation_table.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_dm_table() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "dm_significance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_anomaly_table() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "anomaly_benchmark.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_raw_series() -> pd.DataFrame:
    """Optional: only needed for the 'live' recursive forecast panel.
    Looks for the original building_779 CSV in results/ or the app root."""
    for candidate in (
        os.path.join(RESULTS_DIR, "building_779_updated_meter_reading.csv"),
        "building_779_updated_meter_reading.csv",
    ):
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df["meter_reading"] = df["meter_reading"].interpolate(method="linear").ffill()
            return df
    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Feature engineering (mirrors Section 3 of the notebook exactly, so a live
# window built here matches what the champion model was trained on)
# --------------------------------------------------------------------------- #

def calendar_matrix(idx: pd.DatetimeIndex) -> np.ndarray:
    hr, dw, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    return np.column_stack([
        np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
        np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
        np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366),
        (dw >= 5).astype("float64"),
    ])


def build_latest_window(series: pd.Series, scaler, time_steps: int = TIME_STEPS,
                         week_lag: int = WEEK_LAG) -> np.ndarray:
    """Builds ONE (time_steps, 12) window ending at the last timestamp in
    `series`, using the already-fitted scaler. Same feature order as
    FEATURE_NAMES / the training pipeline."""
    y_raw = series.values.astype("float64")
    idx = series.index
    ys = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    roll_mean = pd.Series(ys).rolling(24, min_periods=1).mean().values
    roll_std = pd.Series(ys).rolling(24, min_periods=1).std().fillna(0.0).values
    cal = calendar_matrix(idx)

    n = len(ys)
    if n < week_lag + time_steps:
        raise ValueError(
            f"Need at least {week_lag + time_steps} hours of history to build "
            f"a window (have {n}). Upload more history or lower WEEK_LAG."
        )

    end = n
    start = end - time_steps
    lag24 = ys[start - 24:end - 24]
    lag168 = ys[start - week_lag:end - week_lag]

    window = np.column_stack([
        ys[start:end], lag24, lag168,
        roll_mean[start:end], roll_std[start:end],
        cal[start:end],
    ])
    assert window.shape == (time_steps, len(FEATURE_NAMES))
    return window


def recursive_forecast(model, last_window: np.ndarray, scaler, n_steps: int) -> np.ndarray:
    """Recursive multi-step forecast.

    NOTE (same simplification the notebook's own Cell 1 multi-step helper
    uses): only the 'reading' channel (feature index 0) is rolled forward
    with the model's own prediction at each step. lag24 / lag168 / rolling
    stats / calendar features are held at their last real values rather
    than being recomputed exactly — recomputing lag168 recursively would
    require carrying a full 168-hour buffer forward. Treat forecasts beyond
    a few hours as indicative, not exact, for this reason.
    """
    current = last_window.copy()
    preds_scaled = []
    for _ in range(n_steps):
        pred = model.predict(current[np.newaxis, :, :], verbose=0)[0, 0]
        preds_scaled.append(pred)
        current = np.roll(current, -1, axis=0)
        current[-1, 0] = pred  # only the 'reading' channel is updated
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


@dataclass
class ModelChoice:
    label: str
    slug: str


# --------------------------------------------------------------------------- #
# Chatbot grounding — a compact, factual summary of this run's real results,
# built from the same artefacts every other page reads. Injected as system
# context so the chatbot answers from actual numbers instead of guessing.
# --------------------------------------------------------------------------- #

def build_project_context() -> str:
    meta = load_metadata()
    fm = meta.get("forecast_metrics", {})
    proposed = meta.get("proposed_model", "AT-CBGRU (proposed)")
    ds = meta.get("dataset", {})
    split = meta.get("split", {})

    lines = ["# BOLT / AT-CBGRU project — factual run summary\n"]

    lines.append(
        f"Dataset: single building (779), hourly meter readings, "
        f"{ds.get('n_hours', '?')} hours, {ds.get('start', '?')} to {ds.get('end', '?')}. "
        f"Split: train {split.get('train', '?')} / val {split.get('val', '?')} / "
        f"test {split.get('test', '?')} hours."
    )
    lines.append(
        "This is a single-building, single-year dataset — do not generalise "
        "results to buildings in the plural."
    )

    if fm:
        ranked = sorted(fm.items(), key=lambda kv: kv[1].get("MAE", float("inf")))
        lines.append("\n## Forecasting leaderboard (by MAE, kWh, lower is better)")
        for name, m in ranked:
            tag = " <- proposed model" if name == proposed else ""
            lines.append(
                f"- {name}: MAE={m.get('MAE'):.4f}, RMSE={m.get('RMSE'):.4f}, "
                f"MAPE={m.get('MAPE (%)', m.get('MAPE')):.2f}%, R2={m.get('R2'):.4f}{tag}"
            )

    dm = load_dm_table()
    if not dm.empty:
        lines.append(
            "\n## Diebold-Mariano significance vs. the proposed model "
            "(p<0.05 = statistically better than; p>=0.05 = only 'comparable to', not 'better than')"
        )
        for _, r in dm.iterrows():
            verdict = "significantly better than" if str(r["Significant (p<0.05)"]).lower() == "yes" else "comparable to (not significantly different from)"
            lines.append(
                f"- vs {r['Baseline']}: p={r['DM p-value']:.4f} -> proposed model is {verdict} {r['Baseline']}"
            )

    abl = load_ablation_table()
    if not abl.empty:
        lines.append("\n## Ablation study (component contribution, sorted by MAE)")
        for _, r in abl.sort_values("MAE").iterrows():
            lines.append(
                f"- {r['Variant']}: MAE={r['MAE']:.4f}, degradation vs full model={r['Degradation (%)']:.2f}%"
            )
        lines.append(
            "Interpretation: a component whose removal causes only small degradation "
            "(or a negative one) contributes little; the largest positive degradation "
            "identifies the most important component. Report this honestly rather than "
            "assuming every component helps equally."
        )

    anom = load_anomaly_table()
    best_det = meta.get("best_anomaly_detector")
    if not anom.empty:
        lines.append(
            "\n## Anomaly detection benchmark (labelled synthetic-fault test set, "
            "NOT the unlabelled real test period)"
        )
        for _, r in anom.sort_values("F1", ascending=False).iterrows():
            lines.append(
                f"- {r['Detector']}: Precision={r['Precision']:.4f}, Recall={r['Recall']:.4f}, F1={r['F1']:.4f}"
            )
        if best_det:
            lines.append(f"Best detector this run: {best_det}")
        lines.append(
            "These F1 scores are honestly weak (roughly 0.1-0.2 range) — this is "
            "expected for this benchmark and should be reported as-is, not hidden."
        )

    shap = load_shap_export()
    if shap:
        lines.append(
            "\nSHAP explainability data is available (global feature/lag importance, "
            "local per-instance explanations, and an attention profile over the 24-hour "
            "lookback window) — the app's Explainability page renders this."
        )

    return "\n".join(lines)
