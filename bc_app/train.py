import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# Backward-compatible default path for Logistic Regression
MODEL_PATH = MODELS_DIR / "breast_cancer_model.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"


def load_data():
    data = datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")
    return X, y, list(data.feature_names), list(data.target_names)


def build_pipeline(estimator):
    """Create a pipeline with StandardScaler and the provided estimator."""
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
    return pipe


def main():
    X, y, feature_names, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=5000, solver="lbfgs"),
        "svm_rbf": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
    }

    results = {}
    for key, estimator in models.items():
        print(f"\n=== Training model: {key} ===")
        pipe = build_pipeline(estimator)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        print(f"CV Accuracy: mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test ROC-AUC: {auc:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Save model under unique filename
        out_path = MODELS_DIR / f"breast_cancer_{key}.joblib"
        joblib.dump(pipe, out_path)
        print(f"Saved model to: {out_path}")
        results[key] = {"cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()), "acc": float(acc), "roc_auc": float(auc), "path": str(out_path)}

    # Keep backward-compatible default (point to logistic regression)
    if (MODELS_DIR / "breast_cancer_logreg.joblib").exists():
        joblib.dump(joblib.load(MODELS_DIR / "breast_cancer_logreg.joblib"), MODEL_PATH)
        print(f"\nSaved default model (logreg) to: {MODEL_PATH}")

    # Persist shared metadata
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "target_names": target_names, "models": results}, f, indent=2)
    print(f"Saved metadata to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
