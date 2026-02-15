"""
Streamlit app: Personal Health Drift Detector (PHDD).
Run on Nebius: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.pipeline import run_pipeline
from src.config import NLSY97_CSV, SAMPLE_CSV, ID_COL
from src.explainability import get_top_contributors

st.set_page_config(page_title="Personal Health Drift Detector", layout="centered")
st.title("Personal Health Drift Detector (PHDD)")
st.caption("Detect early weak signals from longitudinal, self-reported health data.")

# Sidebar: data source
data_source = st.sidebar.radio(
    "Data source",
    ["Synthetic demo (no file)", "Upload CSV", "Use path (NLSY97 or /s3/...)"],
)
df_input = None
data_path = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload longitudinal CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded, nrows=5000, low_memory=False)
        if ID_COL not in df_input.columns and len(df_input.columns) > 0:
            df_input = df_input.rename(columns={df_input.columns[0]: ID_COL})
elif data_source == "Use path (NLSY97 or /s3/...)":
    default_path = str(SAMPLE_CSV) if SAMPLE_CSV.exists() else str(NLSY97_CSV)
    path_str = st.sidebar.text_input("Path to CSV", value=default_path, help="Use data/sample_longitudinal.csv (in repo) or nlsy97_all_1997-2019.csv for real NLSY97.")
    data_path = Path(path_str)
    is_nlsy97 = data_path.exists() and "nlsy97" in path_str.lower()
    max_rows_val = 8984 if is_nlsy97 else 2000
    sample_n_path = st.sidebar.number_input(
        "Max rows (persons for NLSY97)" if is_nlsy97 else "Max rows (memory-safe)",
        min_value=300, max_value=max_rows_val, value=min(2000, max_rows_val), step=200 if not is_nlsy97 else 500,
        help="NLSY97: number of persons (wide→long, 10 waves × 6 vars). Else: rows, first 50 cols."
    )
    if data_path.exists():
        st.sidebar.success(f"Found: {path_str}")
    else:
        st.sidebar.warning("Path not found. Will fall back to synthetic demo.")

run_clicked = st.sidebar.button("Run pipeline")
if run_clicked or (data_source == "Synthetic demo (no file)" and "pipeline_result" not in st.session_state):
    with st.spinner("Building baselines and training model..."):
        try:
            if df_input is not None:
                result = run_pipeline(df_preloaded=df_input)
            elif data_path and data_path.exists():
                result = run_pipeline(data_path=data_path, sample_n=int(sample_n_path))
            else:
                result = run_pipeline()  # synthetic
            st.session_state["pipeline_result"] = result
        except Exception as e:
            st.error(str(e))
            raise

result = st.session_state.get("pipeline_result")
if result:
    st.success("Pipeline finished.")
    m = result["metrics"]
    st.subheader("Model performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("F2-Score", f"{m['f2']:.3f}")
    col2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    col3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    if result.get("fairness"):
        with st.expander("Fairness by group (stratified metrics)"):
            fa = result["fairness"]
            if fa.get("by_group"):
                st.caption("Demographics are never model inputs; used only to check for disparity.")
                rows = []
                for g, v in fa["by_group"].items():
                    rows.append({"Group": g, "F2": f"{v['f2']:.3f}", "PR-AUC": f"{v['pr_auc']:.3f}", "ROC-AUC": f"{v['roc_auc']:.3f}", "n": v["n"]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if fa.get("disparity", {}).get("f2_max_min") is not None:
                    st.metric("F2 max − min across groups", f"{fa['disparity']['f2_max_min']:.3f}")

    st.subheader("Sample outputs (last wave per person)")
    df = result["df"]
    last_all = df.groupby(ID_COL).tail(1)[[ID_COL, "risk_score", "risk_band", "risk_category"]]
    # Filter by risk band and category
    bands = st.multiselect("Filter by risk band", options=["Low", "Moderate", "High"], default=["Low", "Moderate", "High"], key="bands")
    cats = st.multiselect("Filter by category", options=last_all["risk_category"].dropna().unique().tolist() or ["Psycho-emotional", "Metabolic", "Cardiovascular"], default=last_all["risk_category"].dropna().unique().tolist(), key="cats")
    last = last_all[(last_all["risk_band"].isin(bands)) & (last_all["risk_category"].isin(cats))].head(50)
    st.dataframe(last, use_container_width=True, hide_index=True)
    # Export to CSV
    csv = last_all.to_csv(index=False)
    export_name = f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_export.csv"
    st.download_button("Download full results (last wave) as CSV", data=csv, file_name=export_name, mime="text/csv")

    # Feature importance (top contributors)
    with st.expander("Feature importance (top factors driving the model)"):
        top_contrib = get_top_contributors(result["model"], result["model_feat"], model_type="logistic", top_k=8)
        imp_df = pd.DataFrame({"Feature": top_contrib.index, "Importance (|coef|)": top_contrib.values.round(4)})
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.subheader("Example: explanation and follow-up question")
    score_one = result["score_one"]
    last_wave = df.groupby(ID_COL).tail(1)
    person_ids = last_wave[ID_COL].astype(str).tolist()
    selected_id = st.selectbox("Choose a person (by ID) to see their explanation", options=person_ids[:100], index=0, key="person_id")
    sample_row = last_wave[last_wave[ID_COL].astype(str) == str(selected_id)].iloc[0]
    score, band, cat, expl, follow_up = score_one(sample_row)
    st.write(f"**Risk score:** {score:.0f}/100 ({band}) — **Category:** {cat}")
    st.write("**Why we flagged:**", expl)
    st.info("**Follow-up question:** " + follow_up)

st.sidebar.markdown("---")
st.sidebar.markdown("Hea Hackathon · PHDD · No diagnosis, supportive questions only.")
