import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipeline import run_experiment, make_demo_data, validate_dataset

st.set_page_config(page_title="BusBoard AI", page_icon="🚌", layout="wide")
st.title("🚌 BusBoard AI — Hourly Boarding Demand Prediction")
st.caption("Imbalanced smart-card data • GAN oversampling • DNN / CNN2D comparison")

with st.sidebar:
    st.header("Experiment settings")
    epochs = st.slider("GAN / neural-network epochs", 5, 60, 15, 5)
    gan_samples = st.slider("Synthetic samples per minority class", 50, 1000, 200, 50)
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
    st.info("CPU-friendly defaults are selected. First run may take longer while libraries initialize.")

st.markdown("""
Upload a CSV containing smart-card boarding records. Or use the built-in **demo dataset** to test the complete workflow.
The target column is `NumberOfBoardings` (as used in the supplied project document).
""")
uploaded = st.file_uploader("Upload BusBoard CSV", type=["csv"])
use_demo = st.checkbox("Use generated demo dataset", value=uploaded is None)

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
elif use_demo:
    df = make_demo_data()

if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head(12), use_container_width=True)
    st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")
    errors = validate_dataset(df)
    if errors:
        for e in errors: st.error(e)
    else:
        with st.expander("Target class distribution", expanded=True):
            counts = df["NumberOfBoardings"].value_counts().sort_index()
            st.bar_chart(counts)
        if st.button("Run complete experiment", type="primary", use_container_width=True):
            with st.spinner("Preprocessing, training baselines and evaluating models..."):
                try:
                    result = run_experiment(df, epochs=int(epochs), gan_samples=int(gan_samples),
                                            test_size=float(test_size), random_state=int(random_state))
                    st.session_state["result"] = result
                except Exception as e:
                    st.exception(e)

if "result" in st.session_state:
    r = st.session_state["result"]
    st.divider()
    st.subheader("Model performance")
    st.caption("Metrics are calculated on the untouched hold-out test set. They are not the paper's reported results.")
    st.dataframe(r["metrics"].style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1"]}),
                 use_container_width=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Original training rows", f'{r["original_rows"]:,}')
    c2.metric("GAN-generated rows", f'{r["generated_rows"]:,}')
    c3.metric("Test rows", f'{r["test_rows"]:,}')
    st.subheader("Class distribution after resampling")
    st.bar_chart(r["class_counts"])
    st.subheader("Confusion matrices")
    for name, fig in r["confusion"].items():
        with st.expander(name, expanded=(name=="GAN + DNN")):
            st.pyplot(fig, clear_figure=True)
    st.subheader("Download outputs")
    st.download_button("Download predictions CSV", r["predictions"].to_csv(index=False).encode(),
                       "busboard_predictions.csv", "text/csv")
    st.download_button("Download metrics CSV", r["metrics"].to_csv(index=False).encode(),
                       "busboard_metrics.csv", "text/csv")
    st.success("Experiment completed.")
st.divider()
st.markdown("**Expected CSV fields:** `StopName`, `WeekBeginning`, `NumberOfBoardings`. Additional numeric/categorical columns are allowed. "
            "The app extracts calendar features from `WeekBeginning` and encodes categorical predictors.")
