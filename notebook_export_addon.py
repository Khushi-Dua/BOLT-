# =============================================================================
# ADD THESE CELLS TO BOLT_Forecasting_AnomalyDetection.ipynb, right before
# Section 11.3 ("Save artefacts and download"), THEN re-run Section 11.3
# and re-download bolt_results.zip. The original notebook doesn't save any
# of this, and the Streamlit dashboard needs all of it.
#
# Paste each ### block below in as its own cell, in order.
# =============================================================================

### CELL A — save the fitted scaler ##########################################
import joblib
joblib.dump(D["scaler"], f"{CFG.OUTDIR}/scaler.pkl")
print("saved scaler.pkl")


### CELL B — save ablation and DM-significance tables as plain CSVs ##########
# (Section 8 and 9.2 already compute these as DataFrames in-memory —
#  this just writes them out. Variable names must match what those
#  sections actually assigned; adjust the left-hand names below if yours
#  differ from `abl_table` / `dm_table`.)
abl_table.to_csv(f"{CFG.OUTDIR}/ablation_table.csv", index=False)
dm_table.to_csv(f"{CFG.OUTDIR}/dm_significance.csv", index=False)
print("saved ablation_table.csv, dm_significance.csv")


### CELL C — save the anomaly-detection benchmark table ######################
bench_table = pd.DataFrame([
    {"Detector": name, **{k: v for k, v in row.items()}}
    for name, row in anomaly_scores.items()   # replace `anomaly_scores` with
])                                              # whatever dict/DataFrame
bench_table.to_csv(f"{CFG.OUTDIR}/anomaly_benchmark.csv", index=False)
print("saved anomaly_benchmark.csv")
# If Section 10.5 already builds a single DataFrame with columns
# Detector/Precision/Recall/F1/ROC-AUC/PR-AUC (the printed table in Cell 75
# looks exactly like this), just do:
#   detector_results_df.to_csv(f"{CFG.OUTDIR}/anomaly_benchmark.csv", index=False)
# and delete the block above.


### CELL D — SHAP export (GradientExplainer, not KernelExplainer) ###########
# GradientExplainer is far faster than KernelExplainer on a 288-dim
# flattened (24 steps x 12 features) input and works natively on the
# 3-D tensor, so no wrapper/reshape trick is needed.
import shap
import numpy as np

background = Xtr[np.random.choice(len(Xtr), 100, replace=False)]
explain_set = Xte[:20]

explainer = shap.GradientExplainer(champion_model, background)
shap_values = explainer.shap_values(explain_set)
shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
shap_vals = np.squeeze(shap_vals)          # -> (n_instances, TIME_STEPS, n_features)

FEATURE_NAMES = ["reading", "lag24", "lag168", "roll_mean24", "roll_std24",
                  "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                  "doy_sin", "doy_cos", "is_weekend"]

# --- global importance: mean |SHAP| per (feature, lag) ---
mean_abs = np.abs(shap_vals).mean(axis=0)     # (TIME_STEPS, n_features)
global_importance = [
    {"feature": FEATURE_NAMES[f], "lag": f"t-{CFG.TIME_STEPS - t}", "importance": float(mean_abs[t, f])}
    for t in range(CFG.TIME_STEPS) for f in range(len(FEATURE_NAMES))
]

# --- local examples: a couple of individual predictions, fully explained ---
preds_for_explained = champion_model.predict(explain_set, verbose=0).flatten()
base_value = float(np.mean(champion_model.predict(background, verbose=0)))

local_examples = []
for i in [0, 5]:
    contributions = [
        {
            "name": f"{FEATURE_NAMES[f]} (t-{CFG.TIME_STEPS - t})",
            "shap_value": float(shap_vals[i, t, f]),
        }
        for t in range(CFG.TIME_STEPS) for f in range(len(FEATURE_NAMES))
    ]
    local_examples.append({
        "index": i,
        "prediction": float(preds_for_explained[i]),
        "base_value": base_value,
        "contributions": contributions,
    })

# --- attention profile (Section 9.5) ---
# Reuse whatever variable Section 9.5 already computed the averaged
# attention weights into — commonly something like `attn_weights_mean`,
# a length-TIME_STEPS array. Adjust the name if yours differs.
attention_profile = attn_weights_mean.tolist() if "attn_weights_mean" in dir() else None

shap_export = {
    "global_importance": global_importance,
    "local_examples": local_examples,
    "attention_profile": attention_profile,
}

import json
with open(f"{CFG.OUTDIR}/shap_export.json", "w", encoding="utf-8") as fh:
    json.dump(shap_export, fh)
print("saved shap_export.json")


### CELL E — also copy the raw CSV into results/ for the live-forecast panel #
import shutil
shutil.copy(CSV_PATH, f"{CFG.OUTDIR}/building_779_updated_meter_reading.csv")
print("copied raw CSV into results/")

# After all of the above, re-run Section 11.3 and re-download
# bolt_results.zip — it will now include everything the dashboard needs:
#   scaler.pkl, ablation_table.csv, dm_significance.csv,
#   anomaly_benchmark.csv, shap_export.json,
#   building_779_updated_meter_reading.csv
