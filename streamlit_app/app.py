"""Streamlit UI for Credit Risk PD & Stress Testing Engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk PD Engine", layout="wide")
st.title("Credit Risk PD & Stress Testing Engine")
st.caption("Streamlit alternative UI — connects to FastAPI backend")

page = st.sidebar.radio("Navigation", ["Train", "Single Predict", "Batch Predict", "Stress Dashboard"])


def _api_post(path: str, payload: dict | None = None):
    url = f"{API_URL}{path}"
    resp = requests.post(url, json=payload or {}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _api_get(path: str):
    url = f"{API_URL}{path}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


if page == "Train":
    st.header("Train Models")
    if st.button("Run Training", type="primary"):
        with st.spinner("Training LR and XGBoost..."):
            try:
                result = _api_post("/train", {})
                st.success(f"Best model: {result['best_model']}")
                for name, m in result["metrics"].items():
                    st.subheader(name.upper())
                    st.json(m)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

elif page == "Single Predict":
    st.header("Loan PD Calculator")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Person Age", 18, 100, 35)
        person_income = st.number_input("Person Income", 1000, 1000000, 60000)
        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
        emp_length = st.number_input("Employment Length (years)", 0.0, 50.0, 5.0)
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    with col2:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount", 500, 500000, 10000)
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 12.0)
        loan_pct_income = st.number_input("Loan % Income", 0.0, 1.0, 0.3)
        default_file = st.selectbox("Prior Default on File", ["Y", "N"])
        cred_hist = st.number_input("Credit History Length", 0, 30, 3)

    model_name = st.selectbox("Model", ["xgb", "lr"])
    if st.button("Calculate PD", type="primary"):
        payload = {
            "loan": {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": home,
                "person_emp_length": emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_pct_income,
                "cb_person_default_on_file": default_file,
                "cb_person_cred_hist_length": cred_hist,
            },
            "model_name": model_name,
        }
        try:
            result = _api_post("/predict", payload)
            pred = result["predictions"][0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PD", f"{pred['pd']:.2%}")
            c2.metric("Risk Score", pred["risk_score"])
            c3.metric("Risk Band", pred["risk_band"])
            c4.metric("ECL", f"${pred['ecl']:,.2f}")
            st.json(pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Batch Predict":
    st.header("Batch Predict")
    uploaded = st.file_uploader("Upload CSV with loan columns", type=["csv"])
    model_name = st.selectbox("Model", ["xgb", "lr"], key="batch_model")
    if uploaded and st.button("Run Batch Predict", type="primary"):
        df = pd.read_csv(uploaded)
        loans = df.to_dict(orient="records")
        try:
            result = _api_post("/predict", {"loans": loans, "model_name": model_name})
            out = pd.DataFrame(result["predictions"])
            st.dataframe(out)
            st.download_button("Download Results", out.to_csv(index=False), "predictions.csv")
        except Exception as exc:
            st.error(f"Batch predict failed: {exc}")

else:
    st.header("Stress Test Dashboard")
    sample_size = st.slider("Portfolio Sample Size", 100, 2000, 500)
    scenarios = st.multiselect("Scenarios", ["Normal", "Boom", "Recession"], default=["Normal", "Boom", "Recession"])
    model_name = st.selectbox("Model", ["xgb", "lr"], key="stress_model")

    if st.button("Run Stress Test", type="primary"):
        payload = {"sample_size": sample_size, "scenarios": scenarios, "model_name": model_name}
        try:
            result = _api_post("/stress_test", payload)
            st.metric("Loans Tested", result["loan_count"])

            comp_df = pd.DataFrame(result["comparison"])
            st.subheader("Scenario Comparison")
            st.dataframe(comp_df)

            results_df = pd.DataFrame(result["results"])
            st.subheader("Portfolio Results")
            st.bar_chart(results_df.set_index("scenario")[["avg_pd", "total_ecl"]])

            st.subheader("IFRS 9 Stage Breakdown")
            stage_cols = ["stage_1_count", "stage_2_count", "stage_3_count"]
            st.bar_chart(results_df.set_index("scenario")[stage_cols])
        except Exception as exc:
            st.error(f"Stress test failed: {exc}")
