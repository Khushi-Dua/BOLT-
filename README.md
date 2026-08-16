# BOLT — AT-CBGRU Streamlit Dashboard

Dashboard for the **leak-free** `BOLT_Forecasting_AnomalyDetection.ipynb`
pipeline (AT-CBGRU proposed model, 12-model leaderboard, ablation study,
Diebold–Mariano significance testing, synthetic-fault anomaly benchmark).

## 1. Get the artefacts out of the notebook

The notebook's own Section 11.3 saves most of what's needed, but **not
everything** — it's missing the scaler, the ablation/DM/anomaly tables as
plain CSVs, the SHAP values, and a copy of the raw CSV. Open
`notebook_export_addon.py` in this folder, paste each `### CELL` block into
the Colab notebook right before Section 11.3, run them, then re-run
Section 11.3 and re-download `bolt_results.zip`.

Some variable names in the addon cells (`abl_table`, `dm_table`,
`anomaly_scores`, `attn_weights_mean`) are my best guess at what Section 8 /
9.2 / 10.5 / 9.5 already named their output DataFrames — check against your
actual notebook and adjust the left-hand names if they differ. Everything
else in the addon should work unmodified.

## 2. Unzip next to this app

```
bolt_dashboard/
├── app.py
├── utils.py
├── requirements.txt
├── pages/
│   ├── 1_Leaderboard.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Explainability.py
│   ├── 4_Anomaly_Detection.py
│   └── 5_Chatbot.py
└── results/              <- unzip bolt_results.zip contents here
    ├── run_metadata.json
    ├── all_model_predictions.csv
    ├── pred_*.csv
    ├── at_cbgru_champion.keras
    ├── lstm_autoencoder.keras
    ├── scaler.pkl
    ├── ablation_table.csv
    ├── dm_significance.csv
    ├── anomaly_benchmark.csv
    ├── shap_export.json
    └── building_779_updated_meter_reading.csv
```

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy

**Streamlit Community Cloud** (free, easiest): push this whole folder
(including `results/`) to a GitHub repo, then connect it at
share.streamlit.io. Point it at `app.py`.

**Render / a VM**: same as your old bolt dashboard —
`pip install -r requirements.txt && streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Pages

- **Home** — headline AT-CBGRU metrics, actual-vs-predicted overlay for any
  subset of the 12 models, dataset summary.
- **Leaderboard** — full 12-model ranking, Diebold–Mariano significance vs.
  the proposed model, ablation table (component-by-component contribution).
- **Forecast Explorer** — per-model residual diagnostics on the historical
  test set, plus a live recursive multi-step forecast panel that runs the
  actual saved champion model.
- **Explainability** — SHAP global feature/lag importance, two fully
  worked local explanations, and the model's learned attention profile.
- **Anomaly Detection** — the labelled synthetic-fault benchmark results
  (precision/recall/F1/ROC-AUC per detector), reported honestly rather than
  the old notebook's threshold-based "% anomalies" figures.
- **Chatbot** — free-form chat about the project, grounded in this run's
  actual leaderboard/ablation/DM/anomaly numbers so it doesn't invent
  figures. Uses your own Gemini API key (free tier available at
  https://aistudio.google.com/apikey), configured by you as
  `GEMINI_API_KEY` — see the comment at the top of `pages/5_Chatbot.py`
  for the two ways to set it (`.streamlit/secrets.toml` or an environment
  variable). It is never entered by whoever opens the app.

## Known simplification

The live multi-step forecast (Forecast Explorer → "Live multi-step
forecast" tab) only rolls the raw meter-reading channel forward at each
step; the other 11 engineered features (lag168, rolling stats, calendar
encodings) are held at their last real value rather than being fully
recomputed recursively. This matches the same simplification the
notebook's own multi-step cell (Section 12, Cell 1) uses — it's not a bug
introduced here, just something worth being upfront about if asked in a
viva. Forecasts more than a few hours out should be read as indicative,
not exact.

## Bugs fixed since the last handoff

- `at_cbgru_champion.keras` uses two custom layers (`TemporalAttention`,
  `PersistenceSlice`) that the notebook defines inline and never registers
  with `@keras.saving.register_keras_serializable`. That meant
  `load_champion_model()` crashed with `Could not locate class
  'TemporalAttention'` every time — the "Live multi-step forecast" tab was
  broken even once the raw CSV existed. `utils.py` now redefines and
  registers both layers (reverse-engineered from the saved weight shapes,
  verified to load the real trained weights and produce correct-shaped
  predictions) and passes them as `custom_objects` on load. The autoencoder
  (`lstm_autoencoder.keras`) uses only stock Keras layers and needed no fix.
- Replaced the deprecated `use_container_width=True` (removed after
  2025-12-31) with `width="stretch"` across all `st.dataframe` calls.

**Still outstanding:** `building_779_updated_meter_reading.csv` is still
missing from `results/` — copy it in yourself (see the note in the
Forecast Explorer page) before the "Live multi-step forecast" tab will work.
