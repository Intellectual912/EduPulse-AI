
"""Controlled XGBoost tuning using training and validation data only."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "student_features.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "feature_manifest.json"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "target_at_risk"
TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATION = "2014B"

CATEGORICAL = [
    "gender", "region", "highest_education",
    "imd_band", "age_band", "disability"
]

THRESHOLD = 0.40

CONFIGS = [
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.85,
     "colsample_bytree": 0.85, "reg_lambda": 2.0},
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.85,
     "colsample_bytree": 0.85, "reg_lambda": 2.0},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.85,
     "colsample_bytree": 0.85, "reg_lambda": 3.0},
    {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.90,
     "colsample_bytree": 0.90, "reg_lambda": 3.0},
]


def get_feature_columns(df, manifest):
    """Read predictor names from the feature manifest."""

    candidates = [
        manifest.get("feature_columns"),
        manifest.get("features"),
        manifest.get("predictor_columns"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            if isinstance(candidate[0], dict):
                names = [
                    item.get("name") or item.get("feature")
                    for item in candidate
                ]
                names = [name for name in names if name]
            else:
                names = candidate

            if names and all(name in df.columns for name in names):
                return names

    raise ValueError(
        "Could not identify predictor columns in feature_manifest.json. "
        "Do not guess the feature list; inspect the manifest."
    )


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if TARGET not in df.columns or "code_presentation" not in df.columns:
        raise ValueError("Required target or presentation column is missing.")

    features = get_feature_columns(df, manifest)

    forbidden = {
        TARGET, "final_result", "id_student",
        "code_module", "code_presentation"
    }

    if forbidden.intersection(features):
        raise ValueError("Forbidden columns found in predictors.")

    if not set(features).issubset(df.columns):
        raise ValueError("Some manifest features are missing from the dataset.")

    train = df[df["code_presentation"].isin(TRAIN_PRESENTATIONS)].copy()
    valid = df[df["code_presentation"] == VALIDATION_PRESENTATION].copy()

    X_train = train[features]
    y_train = train[TARGET].astype(int)
    X_valid = valid[features]
    y_valid = valid[TARGET].astype(int)

    if y_train.nunique() != 2 or y_valid.nunique() != 2:
        raise ValueError("Training and validation must contain both classes.")

    categorical = [
        col for col in features
        if col in CATEGORICAL or not pd.api.types.is_numeric_dtype(X_train[col])
    ]
    numeric = [col for col in features if col not in categorical]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric
        ))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical
        ))

    preprocessor = ColumnTransformer(transformers)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    results = []
    best_model = None
    best_f1 = -1
    best_config = None

    for number, config in enumerate(CONFIGS, start=1):
        print(f"\nTraining configuration {number}/{len(CONFIGS)}")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            early_stopping_rounds=25,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
            **config,
        )

        model.fit(
            X_train_processed,
            y_train,
            eval_set=[(X_valid_processed, y_valid)],
            verbose=False,
        )

        probabilities = model.predict_proba(X_valid_processed)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        scores = {
            "configuration": number,
            **config,
            "best_iteration": int(model.best_iteration),
            "accuracy": accuracy_score(y_valid, predictions),
            "precision": precision_score(y_valid, predictions, zero_division=0),
            "recall": recall_score(y_valid, predictions, zero_division=0),
            "f1": f1_score(y_valid, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, probabilities),
            "average_precision": average_precision_score(y_valid, probabilities),
        }

        results.append(scores)

        print(
            f"F1: {scores['f1']:.4f} | "
            f"ROC-AUC: {scores['roc_auc']:.4f} | "
            f"AP: {scores['average_precision']:.4f}"
        )

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_model = model
            best_config = config.copy()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "xgboost_tuning_results.csv",
        index=False
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": best_model,
            "features": features,
            "threshold": THRESHOLD,
            "configuration": best_config,
        },
        MODEL_DIR / "xgboost_tuned_candidate.joblib",
    )

    print("\nTUNING COMPLETE")
    print(results_df.to_string(index=False))
    print("\nBest configuration:", best_config)
    print("Best validation F1:", round(best_f1, 4))
    print("Saved candidate: models/xgboost_tuned_candidate.joblib")
    print("Saved report: reports/xgboost_tuning_results.csv")
    print("\nThe 2014J test set was NOT used.")


if __name__ == "__main__":
    main()
