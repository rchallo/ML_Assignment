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
Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Naive Bayes
Random Forest
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

| Model| 	| Accuracy| |AUC|	|Precision|	|Recall|	|F1 Score|	|MCC|
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9298	0.9246	0.9048	0.9048	0.9048	0.8492
KNN	0.9561	0.9823	0.9744	0.9048	0.9383	0.9058
Naive Bayes	0.9211	0.9891	0.9231	0.8571	0.8889	0.8292
Random Forest	0.9737	0.9929	1.0000	0.9286	0.9630	0.9442
XGBoost	0.9737	0.9940	1.0000	0.9286	0.9630	0.9442
🔎 Observations
Random Forest and XGBoost achieved the highest accuracy (97.36%) and MCC (0.9442), indicating superior predictive performance and balanced classification capability.
Logistic Regression also performed strongly with high AUC (0.9960), demonstrating excellent class separability.
Naive Bayes showed comparatively lower recall, indicating slightly weaker performance in detecting malignant cases.
Ensemble methods (Random Forest & XGBoost) performed better than individual classifiers, confirming that ensemble learning improves generalization and stability.
Overall, Random Forest and XGBoost are the most reliable models for this classification task.
🚀 Streamlit Web Application
A Streamlit web application was developed to:
Download a sample test dataset
Upload test dataset (.csv)
Select any of the six models
View evaluation metrics
View confusion matrix
View classification report

