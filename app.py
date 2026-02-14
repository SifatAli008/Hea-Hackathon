"""
Streamlit app: Personal Health Drift Detector (PHDD).
Run on Nebius: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.pipeline import run_pipeline
from src.config import NLSY97_CSV, ID_COL

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
    path_str = st.sidebar.text_input("Path to CSV", value=str(NLSY97_CSV))
    data_path = Path(path_str)
    sample_n_path = st.sidebar.number_input("Max rows (memory-safe)", min_value=500, max_value=5000, value=2000, step=500, help="Large NLSY97: use 2000–3000 to avoid OOM.")
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

    st.subheader("Sample outputs (last wave per person)")
    df = result["df"]
    last = df.groupby(ID_COL).tail(1)[[ID_COL, "risk_score", "risk_band", "risk_category"]].head(15)
    st.dataframe(last, use_container_width=True)

    st.subheader("Example: explanation and follow-up question")
    score_one = result["score_one"]
    sample_row = df.groupby(ID_COL).tail(1).iloc[0]
    score, band, cat, expl, follow_up = score_one(sample_row)
    st.write(f"**Risk score:** {score:.0f}/100 ({band}) — **Category:** {cat}")
    st.write("**Why we flagged:**", expl)
    st.info("**Follow-up question:** " + follow_up)

st.sidebar.markdown("---")
st.sidebar.markdown("Hea Hackathon · PHDD · No diagnosis, supportive questions only.")
