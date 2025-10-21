import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "breast_cancer_model.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"

def read_metadata():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(selected_model_path: str):
    model = joblib.load(selected_model_path)
    meta = read_metadata()
    return model, meta["feature_names"], meta["target_names"], meta.get("models", {})


def predict_df(model, df):
    # Probability of class label 0 (malignant) per sklearn's breast_cancer dataset
    proba = model.predict_proba(df)
    classes = list(model.classes_)
    malignant_idx = classes.index(0)
    malignant_prob = proba[:, malignant_idx]
    preds = (malignant_prob >= 0.5).astype(int)
    return preds, malignant_prob


def try_map_known_schemas(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame | None:
    """
    Attempt to map a common Kaggle/UCI breast cancer CSV schema to sklearn feature names.
    Returns a DataFrame with exactly expected_features columns in order, or None if mapping fails.
    """
    mapping = {
        # means
        "radius_mean": "mean radius",
        "texture_mean": "mean texture",
        "perimeter_mean": "mean perimeter",
        "area_mean": "mean area",
        "smoothness_mean": "mean smoothness",
        "compactness_mean": "mean compactness",
        "concavity_mean": "mean concavity",
        "concave points_mean": "mean concave points",
        "symmetry_mean": "mean symmetry",
        "fractal_dimension_mean": "mean fractal dimension",
        # errors
        "radius_se": "radius error",
        "texture_se": "texture error",
        "perimeter_se": "perimeter error",
        "area_se": "area error",
        "smoothness_se": "smoothness error",
        "compactness_se": "compactness error",
        "concavity_se": "concavity error",
        "concave points_se": "concave points error",
        "symmetry_se": "symmetry error",
        "fractal_dimension_se": "fractal dimension error",
        # worst
        "radius_worst": "worst radius",
        "texture_worst": "worst texture",
        "perimeter_worst": "worst perimeter",
        "area_worst": "worst area",
        "smoothness_worst": "worst smoothness",
        "compactness_worst": "worst compactness",
        "concavity_worst": "worst concavity",
        "concave points_worst": "worst concave points",
        "symmetry_worst": "worst symmetry",
        "fractal_dimension_worst": "worst fractal dimension",
    }

    # Work on a copy to avoid mutating the original df
    df2 = df.copy()
    # Drop likely non-feature columns if present
    for col in ["id", "diagnosis"]:
        if col in df2.columns:
            df2 = df2.drop(columns=[col])

    # Only rename columns that exist
    applicable = {old: new for old, new in mapping.items() if old in df2.columns}
    if not applicable:
        return None
    df2 = df2.rename(columns=applicable)

    # After renaming, check if we can produce the full expected set
    if all(col in df2.columns for col in expected_features):
        return df2[expected_features]
    return None


st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("Breast Cancer Detection (Wisconsin Diagnostic)")

# Check model availability
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.warning("Model not found. Please run `python train.py` first to train and save the model.")
    st.stop()

# Read metadata and available models
meta = read_metadata()
available = meta.get("models", {})

with st.sidebar:
    st.subheader("Model Selection")
    if available:
        options = []
        labels = []
        for key, info in available.items():
            options.append(info["path"])
            labels.append(f"{key} (acc={info['acc']:.3f}, auc={info['roc_auc']:.3f})")
        selected_label = st.selectbox("Choose trained model", options=labels, index=0)
        selected_idx = labels.index(selected_label)
        selected_model_path = options[selected_idx]
    else:
        st.info("Only default model found. Train again to generate multiple models.")
        selected_model_path = str(MODEL_PATH)

model, feature_names, target_names, _ = load_model(selected_model_path)

with st.sidebar:
    st.header("Single Prediction Input")
    st.caption("Adjust feature values and click Predict.")

    inputs = {}
    # Create number inputs; typical ranges are guided by dataset statistics
    # We'll set defaults to dataset means for a reasonable starting point
    defaults = {
        'mean radius': 14.13, 'mean texture': 19.29, 'mean perimeter': 91.97, 'mean area': 654.89,
        'mean smoothness': 0.096, 'mean compactness': 0.104, 'mean concavity': 0.089, 'mean concave points': 0.048,
        'mean symmetry': 0.181, 'mean fractal dimension': 0.063, 'radius error': 0.405, 'texture error': 1.22,
        'perimeter error': 2.87, 'area error': 40.34, 'smoothness error': 0.007, 'compactness error': 0.025,
        'concavity error': 0.031, 'concave points error': 0.011, 'symmetry error': 0.020, 'fractal dimension error': 0.003,
        'worst radius': 16.27, 'worst texture': 25.68, 'worst perimeter': 107.26, 'worst area': 880.58,
        'worst smoothness': 0.132, 'worst compactness': 0.254, 'worst concavity': 0.272, 'worst concave points': 0.115,
        'worst symmetry': 0.290, 'worst fractal dimension': 0.084
    }

    for feat in feature_names:
        val = defaults.get(feat, 0.0)
        # Use number_input with reasonable bounds
        inputs[feat] = st.number_input(feat, value=float(val), step=0.01, format="%f")

    threshold = st.slider("Decision threshold (malignant probability)", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    if st.button("Predict (Single)"):
        df_in = pd.DataFrame([{k: inputs[k] for k in feature_names}])
        preds, probs = predict_df(model, df_in)
        malignant_prob = float(probs[0])
        pred_label = int(malignant_prob >= threshold)
        # sklearn target_names order is ['malignant','benign'] with 0=malignant, 1=benign
        # We will override using thresholded probability for clarity
        diagnosis_text = "Malignant — has breast cancer" if pred_label == 1 else "Benign — no breast cancer"
        if pred_label == 1:
            st.error(f"Diagnosis: {diagnosis_text}\n\nMalignant probability: {malignant_prob:.3f} (threshold {threshold:.2f})")
        else:
            st.success(f"Diagnosis: {diagnosis_text}\n\nMalignant probability: {malignant_prob:.3f} (threshold {threshold:.2f})")

st.header("No CSV? Try one of these")
tab1, tab2 = st.tabs(["Pick a sample", "Paste values"])

with tab1:
    st.caption("Use a real sample from the sklearn dataset to see how the model predicts.")
    bc = datasets.load_breast_cancer()
    idx = st.slider("Sample index", min_value=0, max_value=len(bc.data)-1, value=0)
    sample_df = pd.DataFrame([bc.data[idx]], columns=bc.feature_names)
    preds, probs = predict_df(model, sample_df)
    malignant_prob = float(probs[0])
    diagnosis_text = "Malignant — has breast cancer" if malignant_prob >= 0.5 else "Benign — no breast cancer"
    st.write(pd.DataFrame(sample_df.T).rename(columns={0: "value"}))
    if malignant_prob >= 0.5:
        st.error(f"Diagnosis: {diagnosis_text} | Malignant probability: {malignant_prob:.3f}")
    else:
        st.success(f"Diagnosis: {diagnosis_text} | Malignant probability: {malignant_prob:.3f}")

with tab2:
    st.caption("Paste either: (1) 30 comma-separated numbers in sklearn feature order, or (2) JSON mapping of feature names to values.")
    txt = st.text_area("Paste values here", height=120, placeholder="e.g. 14.13,19.29,91.97,... (30 numbers) OR {\"mean radius\":14.13,...}")
    if st.button("Predict (Pasted values)"):
        try:
            df_in = None
            # Try JSON first
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    # Accept Kaggle names too via existing mapping helper
                    df_tmp = pd.DataFrame([obj])
                    mapped = try_map_known_schemas(df_tmp, feature_names)
                    if mapped is None:
                        # If already sklearn names
                        if all(k in df_tmp.columns for k in feature_names):
                            df_in = df_tmp[feature_names]
                    else:
                        df_in = mapped
            except Exception:
                pass

            # Fallback: CSV of 30 numbers
            if df_in is None:
                parts = [p.strip() for p in txt.split(',') if p.strip()]
                if len(parts) == len(feature_names):
                    values = [float(p) for p in parts]
                    df_in = pd.DataFrame([values], columns=feature_names)

            if df_in is None:
                st.error("Could not parse input. Provide 30 numbers in sklearn feature order or a JSON of feature:value pairs.")
            else:
                preds, probs = predict_df(model, df_in)
                malignant_prob = float(probs[0])
                diagnosis_text = "Malignant — has breast cancer" if malignant_prob >= 0.5 else "Benign — no breast cancer"
                if malignant_prob >= 0.5:
                    st.error(f"Diagnosis: {diagnosis_text} | Malignant probability: {malignant_prob:.3f}")
                else:
                    st.success(f"Diagnosis: {diagnosis_text} | Malignant probability: {malignant_prob:.3f}")
        except Exception as e:
            st.exception(e)
st.header("Batch Prediction via CSV")
st.write(
    "Upload a CSV to predict for each row. "
    "If your file uses the common schema (e.g., radius_mean, texture_mean, ...), we'll auto-map to sklearn names."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            # Try auto-mapping common schemas
            mapped = try_map_known_schemas(df, feature_names)
            if mapped is None:
                st.error(
                    "Uploaded CSV is missing required columns and could not be auto-mapped.\n\n"
                    f"Required sklearn columns: {feature_names}\n\n"
                    "Tip: If your columns look like radius_mean, texture_mean, ..., upload that file — auto-mapping is supported."
                )
                st.stop()
            df_features = mapped
            st.info("Auto-mapped CSV columns to sklearn feature names.")
        else:
            # Use provided columns directly
            df_features = df[feature_names]

        preds, probs = predict_df(model, df_features)
        out = df_features.copy()
        out["malignant_probability"] = probs
        out["has_breast_cancer"] = (out["malignant_probability"] >= 0.50).astype(int)
        out["diagnosis_label"] = out["has_breast_cancer"].map({1: "Malignant — has breast cancer", 0: "Benign — no breast cancer"})

        # Inline summary metrics
        total = len(out)
        malignant_count = int(out["has_breast_cancer"].sum())
        benign_count = int(total - malignant_count)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", f"{total}")
        col2.metric("Malignant (has breast cancer)", f"{malignant_count}")
        col3.metric("Benign (no breast cancer)", f"{benign_count}")

        # Full table inline
        st.subheader("Predictions Table")
        st.dataframe(out, use_container_width=True)

        # Top-N highest malignant probability preview
        st.subheader("Highest Risk Cases (by malignant probability)")
        top_n = st.slider("Show top N", min_value=5, max_value=min(50, total), value=min(10, total))
        top_df = out.sort_values("malignant_probability", ascending=False).head(top_n)
        st.dataframe(top_df, use_container_width=True)
        st.download_button(
            label="Download Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
        st.success("Batch prediction completed.")
    except Exception as e:
        st.exception(e)

st.caption("Model: Logistic Regression with StandardScaler. Dataset: sklearn breast_cancer. This tool is not a substitute for medical diagnosis.")
