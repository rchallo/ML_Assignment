🧬 Breast Cancer Classification using Machine Learning:

📌 Problem Statement:
The objective of this project is to build and compare multiple Machine Learning classification models to predict whether a breast tumor is Malignant (M) or Benign (B) based on diagnostic features.
The goal is to evaluate and compare model performance using multiple evaluation metrics and deploy the best-performing model using Streamlit.

📊 Dataset Description:
Dataset: Breast Cancer Wisconsin Dataset
Source: Kaggle
Instances: 569
Features: 30 numerical features
Target Variable: diagnosis
M → Malignant (1)
B → Benign (0)
The dataset contains computed features from digitized images of fine needle aspirate (FNA) of breast masses.

🤖 Machine Learning Models Implemented:
The following six classification algorithms were implemented and compared:
Logistic Regression,
Decision Tree,
K-Nearest Neighbors (KNN),
Naive Bayes,
Random Forest,
XGBoost

All models were trained on the same dataset split and evaluated using identical metrics.

📈 Model Evaluation Metrics:
Each model was evaluated using:
Accuracy
AUC (Area Under ROC Curve)
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

📊 Model Performance Comparison:

| Model                | Accuracy | Precision | Recall | F1 Score | AUC  | MCC  |
|----------------------|----------|-----------|--------|----------|------|------|
| Logistic Regression  | 0.97     | 0.96      | 0.95   | 0.96     | 0.99 | 0.94 |
| Decision Tree        | 0.93     | 0.92      | 0.91   | 0.91     | 0.94 | 0.87 |
| KNN                  | 0.95     | 0.94      | 0.93   | 0.94     | 0.97 | 0.90 |
| Naive Bayes          | 0.94     | 0.93      | 0.92   | 0.92     | 0.96 | 0.89 |
| Random Forest        | 0.98     | 0.97      | 0.96   | 0.97     | 0.99 | 0.95 |
| XGBoost              | 0.99     | 0.98      | 0.97   | 0.98     | 0.99 | 0.96 |


## 📌 Observations

| Model                | Key Observation | Strength | Weakness | Overall Remark |
|----------------------|----------------|----------|----------|----------------|
| Logistic Regression  | Balanced and stable performance across metrics | Interpretable | Slightly lower accuracy than ensemble models | Good baseline model |
| Decision Tree        | Lower accuracy compared to others | Easy to interpret | Prone to overfitting | Not the best performer |
| KNN                  | Good performance with moderate stability | Simple algorithm | Sensitive to scaling | Decent model |
| Naive Bayes          | Fast and efficient | Works well with small datasets | Assumes feature independence | Moderate performance |
| Random Forest        | High accuracy and strong AUC score | Handles overfitting well | Less interpretable | Strong performer |
| XGBoost              | Highest accuracy and AUC score | Excellent predictive power | Slightly complex | Best overall model |


***Overall, Random Forest and XGBoost are the most reliable models for this classification task.

