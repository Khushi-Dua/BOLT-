import numpy as np
import pandas as pd
import streamlit as st

from utils import load_shap_export, FEATURE_NAMES, TIME_STEPS

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")
st.title("🧠 Explainability")

shap_data = load_shap_export()

if not shap_data:
    st.warning(
        "No `results/shap_export.json` found. This isn't produced by the "
        "notebook as-is — add the export cell from `notebook_export_addon.py` "
        "(computes SHAP once in the notebook using GradientExplainer, which is "
        "far faster than KernelExplainer on a 288-feature flattened window, "
        "then saves the values so the dashboard doesn't need to recompute "
        "them live)."
    )
    st.stop()

tab_global, tab_local, tab_attn = st.tabs(
    ["Global feature importance", "Local (single prediction)", "Attention weights"]
)

with tab_global:
    st.markdown(
        "Mean |SHAP value| per **(feature, lag)** pair, averaged across the "
        "explained test instances."
    )
    global_imp = pd.DataFrame(shap_data["global_importance"])  # columns: feature, lag, importance
    pivot = global_imp.pivot(index="feature", columns="lag", values="importance")
    pivot = pivot.reindex(FEATURE_NAMES)

    # Styler.background_gradient() is lazy — it doesn't touch matplotlib
    # until Streamlit actually renders it, so a try/except around the call
    # itself never catches a missing-matplotlib ImportError. Check up front
    # instead.
    try:
        import matplotlib  # noqa: F401
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    if has_matplotlib:
        styled = pivot.style.background_gradient(cmap="Reds", axis=None)
    else:
        styled = pivot
        st.caption("Install `matplotlib` to see this table with a heatmap gradient.")
    st.dataframe(styled, width="stretch")

    top_by_feature = global_imp.groupby("feature")["importance"].sum().sort_values(ascending=False)
    st.markdown("**Total importance by feature (summed over all 24 lags)**")
    st.bar_chart(top_by_feature, height=320)

with tab_local:
    instances = shap_data.get("local_examples", [])
    if not instances:
        st.info("No local examples in the export.")
    else:
        idx = st.selectbox(
            "Test instance", range(len(instances)),
            format_func=lambda i: f"Instance {instances[i]['index']}",
        )
        ex = instances[idx]
        st.markdown(f"**Prediction:** {ex['prediction']:.3f} kWh &nbsp;|&nbsp; **Base value:** {ex['base_value']:.3f} kWh")

        contrib = pd.DataFrame(ex["contributions"]).sort_values("shap_value")
        top = pd.concat([contrib.head(5), contrib.tail(5)]).drop_duplicates()
        st.markdown("**Top positive / negative contributors (this prediction)**")
        st.bar_chart(top.set_index("name")["shap_value"], height=340)

        top_pos = contrib.iloc[-1]
        top_neg = contrib.iloc[0]
        st.write(
            f"Prediction increased mainly because **{top_pos['name']}** was high "
            f"(SHAP +{top_pos['shap_value']:.4f}). It was offset mainly by "
            f"**{top_neg['name']}** (SHAP {top_neg['shap_value']:.4f})."
        )

with tab_attn:
    attn = shap_data.get("attention_profile")
    if not attn:
        st.info(
            "No attention profile in the export. Add the `attention_profile` "
            "block from `notebook_export_addon.py` (reads Section 9.5's "
            "averaged attention weights)."
        )
    else:
        s = pd.Series(attn, index=[f"t-{TIME_STEPS - i}" for i in range(TIME_STEPS)])
        st.markdown(
            "Attention weight per lag, averaged over the test set. Peaks "
            "here should line up with the ACF plot from Section 2.3 of the "
            "notebook — that agreement is the independent check that the "
            "model learned real structure, not noise."
        )
        st.bar_chart(s, height=340)