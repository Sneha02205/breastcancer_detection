# Breast Cancer Detection App

A clean ML pipeline and Streamlit frontend for the sklearn Wisconsin Diagnostic Breast Cancer dataset.

## Features
- Standardized features + Logistic Regression pipeline
- 5-fold CV, metrics (accuracy, ROC-AUC, classification report)
- Saved model with metadata
- Streamlit UI for single and batch predictions

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Train the model
```bash
python bc_app/train.py
```
This saves the model and feature names to `bc_app/models/`.

## Run the UI
```bash
streamlit run bc_app/streamlit_app.py
```

## CSV Batch Schema
Upload a CSV with the exact 30 feature columns named as in sklearn's dataset (e.g., `mean radius`, `mean texture`, ..., `worst fractal dimension`).

## Notes
- This is for educational use only. Not a medical device.
- If you prefer using your `Breast_cancer_dataset.csv`, add a loader function to read and map its columns to sklearn names before prediction.
