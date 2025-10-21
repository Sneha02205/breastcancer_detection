Overview

This project aims to develop a machine learning-based breast cancer detection system that can accurately classify whether a tumor is malignant (cancerous) or benign (non-cancerous) based on various medical features.
The model helps assist doctors and medical researchers by providing quick, data-driven insights for early diagnosis and treatment planning.

 Objectives

Analyze and preprocess breast cancer data.

Train and evaluate multiple ML models for classification.

Achieve high accuracy and sensitivity in cancer detection.

Build a simple and interactive web interface using Flask for real-time predictions.

 Technologies Used

Python

Flask (for web deployment)

Pandas, NumPy (for data handling)

Matplotlib, Seaborn (for visualization)

Scikit-learn (for model training and evaluation)

 Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, available from:

UCI Machine Learning Repository

Features include:

Mean radius, texture, perimeter, area, smoothness, compactness, concavity, etc.

Target Variable:
M = Malignant
B = Benign

 Model Workflow

Data Preprocessing

Handle missing values

Feature normalization

Data visualization for correlation analysis

Model Training

Algorithms used:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Model selection based on highest accuracy

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Deployment

Flask-based web app for predicting cancer type from user input
