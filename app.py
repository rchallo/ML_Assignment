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

# ---------- Safe Path Handling ----------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load scaler and feature names
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

st.set_page_config(page_title="Breast Cancer Classification App")
st.title("Breast Cancer Classification using Multiple ML Models")
st.write("Upload a CSV file (test dataset only) to make predictions.")

# Model dictionary (FILE PATHS only)
models = {
    "Logistic Regression": os.path.join(MODEL_DIR, "Logistic_Regression.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "Decision_Tree.pkl"),
    "KNN": os.path.join(MODEL_DIR, "KNN.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "Naive_Bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "Random_Forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "XGBoost.pkl")
}

# ---------- Download Sample Test File ----------
DATA_DIR = os.path.join(BASE_DIR, "data")
test_file_path = os.path.join(DATA_DIR, "test_data.csv")

st.markdown("### 📥 Download Sample Test Dataset")

if os.path.exists(test_file_path):
    with open(test_file_path, "rb") as file:
        st.download_button(
            label="Download test_data.csv",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )
else:
    st.warning("Sample test_data.csv not found in data folder.")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    if "diagnosis" in df.columns:
        y_true = df["diagnosis"].map({"M": 1, "B": 0})
        X = df.drop("diagnosis", axis=1)

        # 🔥 Drop id column if present
       # if "id" in X.columns:
         #   X = X.drop("id", axis=1)

        # 🔥 Ensure correct feature order
        X = X[feature_names]

    else:
        st.error("Uploaded dataset must contain 'diagnosis' column.")
        st.stop()

    # Scale features
    X_scaled = scaler.transform(X)

    selected_model = st.selectbox("Select Model", list(models.keys()))

    model = joblib.load(models[selected_model])

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    st.write(f"AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
    st.write(f"Precision: {precision_score(y_true, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_true, y_pred):.4f}")
    st.write(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    st.write(f"MCC Score: {matthews_corrcoef(y_true, y_pred):.4f}")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
