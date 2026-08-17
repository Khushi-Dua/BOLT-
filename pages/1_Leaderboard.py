import pandas as pd
import streamlit as st

from utils import load_metadata, load_ablation_table, load_dm_table

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 Model Leaderboard")

meta = load_metadata()
fm = meta.get("forecast_metrics", {})
proposed = meta.get("proposed_model", "AT-CBGRU (proposed)")

if not fm:
    st.warning("No metrics found in `run_metadata.json`.")
    st.stop()

df = pd.DataFrame(fm).T
df = df.sort_values("MAE")
df.insert(0, "Rank", range(1, len(df) + 1))

# --- Plain-language answer to "which model is best", not just a chart ---
best_model = df.index[0]
best_row = df.iloc[0]
runner_up = df.index[1] if len(df) > 1 else None
runner_row = df.iloc[1] if len(df) > 1 else None
st.success(
    f"**Best model: {best_model}** — MAE {best_row['MAE']:.4f}, "
    f"RMSE {best_row['RMSE']:.4f}, MAPE {best_row['MAPE (%)']:.2f}%, "
    f"R² {best_row['R2']:.4f}, Accuracy {best_row['Accuracy (%)']:.2f}%"
    + (
        f"  \n*(vs. runner-up {runner_up}: MAE {runner_row['MAE']:.4f} — "
        f"{((runner_row['MAE'] - best_row['MAE']) / runner_row['MAE'] * 100):.1f}% lower error)*"
        if runner_up is not None else ""
    )
)

st.subheader("Forecasting — every model, same test window")
st.dataframe(
    df.style.format(precision=4).apply(
        lambda row: ["background-color: rgba(46,160,67,0.18)" if row.name == proposed else "" for _ in row],
        axis=1,
    ),
    width="stretch",
)

st.bar_chart(df["MAE"], height=320)

st.divider()

st.subheader("Statistical significance vs. the proposed model (Diebold–Mariano)")
dm = load_dm_table()
if not dm.empty:
    st.dataframe(dm, width="stretch")
    st.caption(
        "A negative DM statistic means the proposed model has lower error. "
        "p < 0.05 → the difference is unlikely to be chance ('better than'). "
        "p ≥ 0.05 → report as 'comparable to', not 'better than' — this "
        "notebook is explicit about that distinction."
    )
else:
    st.info(
        "No `dm_significance.csv` found. Export it from Section 9.2 of the "
        "notebook with:\n\n`out.to_csv('results/dm_significance.csv', index=False)`"
    )

st.divider()

st.subheader("Ablation — component-by-component contribution")
abl = load_ablation_table()
if not abl.empty:
    abl_sorted = abl.sort_values("MAE")
    st.dataframe(abl_sorted, width="stretch")
    st.bar_chart(abl_sorted.set_index("Variant")["Degradation (%)"], height=320)
    st.caption(
        "Positive degradation % = removing that component hurts MAE. "
        "A negative value means the full model is actually *worse* than "
        "without that component — worth addressing directly in the paper "
        "rather than omitting."
    )
else:
    st.info(
        "No `ablation_table.csv` found. Export it from Section 8 of the "
        "notebook with:\n\n`abl.to_csv('results/ablation_table.csv', index=False)`"
    )
