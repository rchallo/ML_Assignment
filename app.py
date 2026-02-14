!pip install streamlit
pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Breast Cancer Classification App")

st.title("Breast Cancer Classification using Multiple ML Models")

st.write("Upload a CSV file (test dataset only) to make predictions.")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model dictionary
models = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

# File upload
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    if "diagnosis" in df.columns:
        y_true = df["diagnosis"].map({"M": 1, "B": 0})
        X = df.drop("diagnosis", axis=1)
    else:
        st.error("Uploaded dataset must contain 'diagnosis' column.")
        st.stop()

    # Scale features
    X_scaled = scaler.transform(X)

    # Model selection
    selected_model = st.selectbox("Select Model", list(models.keys()))

    model = joblib.load(models[selected_model])

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
