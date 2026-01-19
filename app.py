import streamlit as st
import pandas as pd

from src.predict import predict_fraud
from src.input_schema import build_input_df

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡",
    layout="centered"
)

# --------------------------------------------------
# Title & description
# --------------------------------------------------
st.title("🛡 Insurance Fraud Detection System")
st.write(
    """
    This application predicts whether an insurance claim is **potentially fraudulent**
    using a machine learning model trained on historical insurance claims data.
    """
)

st.divider()

# --------------------------------------------------
# User Inputs (Simplified but meaningful)
# --------------------------------------------------
st.subheader("Enter Claim Details")

age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=35
)

policy_premium = st.number_input(
    "Annual Policy Premium (₹)",
    min_value=5000,
    max_value=300000,
    value=45000
)

total_claim = st.number_input(
    "Total Claim Amount (₹)",
    min_value=1000,
    max_value=2000000,
    value=120000
)

incident_severity = st.selectbox(
    "Incident Severity",
    [
        "Trivial Damage",
        "Minor Damage",
        "Major Damage"
    ]
)
police_report = st.selectbox(
    "Police Report Available?",
    ["YES", "NO"]
)

witnesses = st.number_input(
    "Number of Witnesses",
    min_value=0,
    max_value=5,
    value=0
)


# --------------------------------------------------
# Build full input dataframe (CRITICAL STEP)
# --------------------------------------------------
user_inputs = {
    "age": age,
    "policy_annual_premium": policy_premium,
    "total_claim_amount": total_claim,
    "incident_severity": incident_severity,
    "police_report_available": police_report,
    "witnesses": witnesses
}


input_df = build_input_df(user_inputs)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("🔍 Predict Fraud Risk"):
    with st.spinner("Analyzing claim..."):
        prediction, probability = predict_fraud(input_df)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **High Fraud Risk Detected**\n\n"
            f"Fraud Probability: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **Low Fraud Risk**\n\n"
            f"Fraud Probability: **{probability:.2%}**"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Built using Machine Learning & Streamlit | Insurance Fraud Detection Project"
)
