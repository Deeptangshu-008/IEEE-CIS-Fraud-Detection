from inference import predict_fraud
import streamlit as st
import pandas as pd
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="IEEE-CIS Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ IEEE-CIS Fraud Detection")
st.write("Upload transaction data to predict fraud probabilities.")


uploaded_file = st.file_uploader(
    "Upload a CSV file containing raw transaction data",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(f"Number of transactions: {len(df)}")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Predict Fraud", type="primary"):
        with st.spinner("Predicting fraud probabilities..."):
            try:
                results = predict_fraud(df)

                if len(results) != len(df):
                    st.error(
                        "The number of predictions does not match "
                        "the uploaded data."
                    )
                    st.stop()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        df["fraud_probability"] = results
        df["predicted_fraud"] = (
            df["fraud_probability"] >= 0.5
        ).astype(int)

        # Move prediction columns to 4th and 5th positions
        cols = list(df.columns)
        cols.remove("fraud_probability")
        cols.remove("predicted_fraud")

        df = df[
            cols[:3]
            + ["fraud_probability", "predicted_fraud"]
            + cols[3:]
        ]

        st.success("Predictions completed!")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Transactions", len(df))

        with col2:
            st.metric(
                "Predicted Fraud",
                int(df["predicted_fraud"].sum())
            )

        with col3:
            st.metric(
                "Average Fraud Probability",
                f"{df['fraud_probability'].mean():.2%}"
            )

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions CSV",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )