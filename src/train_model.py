import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MANIFEST_PATH = DATA_DIR / "feature_manifest.json"
DATA_PATH = DATA_DIR / "student_features.csv"

TARGET = "target_at_risk"
TEST_PRESENTATION = "2014J"
RANDOM_STATE = 42


def main():
    # 1. Load the dataset and the authoritative feature manifest.
    if not DATA_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Required processed dataset or feature manifest is missing."
        )

    df = pd.read_csv(DATA_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    features = manifest["feature_columns"]

    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")

    if "code_presentation" not in df.columns:
        raise ValueError("Cannot perform chronological split: "
                         "code_presentation is missing.")

    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing predictor columns: {missing_features}")

    forbidden = set(manifest.get("forbidden_predictors", []))
    leakage_features = sorted(set(features) & forbidden)
    if leakage_features:
        raise ValueError(
            f"Forbidden predictors found in manifest feature list: "
            f"{leakage_features}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # 2. Keep the latest presentation completely separate for testing.
    test_mask = df["code_presentation"].astype(str).eq(TEST_PRESENTATION)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test partition is empty.")

    if train_df[TARGET].nunique() != 2 or test_df[TARGET].nunique() != 2:
        raise ValueError(
            "Both training and test partitions must contain both target classes."
        )

    X_train = train_df[features].copy()
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int)

    # 3. Identify categorical and numeric predictors.
    known_categorical = {
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    }

    categorical_features = [
        col for col in features
        if col in known_categorical
        or pd.api.types.is_object_dtype(X_train[col])
        or isinstance(X_train[col].dtype, pd.CategoricalDtype)
    ]

    numeric_features = [
        col for col in features if col not in categorical_features
    ]

    # 4. Handle missing values and convert categories into numeric columns.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    # 5. Define the XGBoost binary classifier.
    classifier = xgboost.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )

    # Keep preprocessing and model together in one deployable pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("=== TRAINING CONFIGURATION ===")
    print("Dataset rows:", len(df))
    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Test presentation:", TEST_PRESENTATION)
    print("Manifest predictors:", len(features))
    print("Numeric predictors:", len(numeric_features))
    print("Categorical predictors:", len(categorical_features))
    print("Training presentations:",
          sorted(train_df["code_presentation"].astype(str).unique()))
    print("Test target distribution:")
    print(y_test.value_counts().sort_index().to_dict())

    # 6. Fit the model only on the earlier presentations.
    print("\nTraining XGBoost. Please wait...")
    model.fit(X_train, y_train)

    # 7. Evaluate once on the held-out presentation.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "XGBoostClassifier",
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "positive_class_meaning": "Fail or Withdrawn",
        "test_presentation": TEST_PRESENTATION,
        "training_presentations": sorted(
            train_df["code_presentation"].astype(str).unique().tolist()
        ),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "manifest_predictor_count": int(len(features)),
        "numeric_predictor_count": int(len(numeric_features)),
        "categorical_predictor_count": int(len(categorical_features)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["On track", "At risk"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # 8. Save the complete pipeline and evaluation evidence.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "risk_classifier.joblib"
    metrics_path = REPORT_DIR / "risk_model_metrics.json"
    importance_path = REPORT_DIR / "risk_feature_importance.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    transformed_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    if len(transformed_names) != len(importances):
        raise RuntimeError(
            "Feature importance count does not match transformed features."
        )

    importance_df = pd.DataFrame({
        "feature": transformed_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(importance_path, index=False)

    print("\n=== HELD-OUT TEST RESULTS ===")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted on-track | Predicted at-risk")
    print(f"Actual on-track: {metrics['confusion_matrix'][0]}")
    print(f"Actual at-risk:  {metrics['confusion_matrix'][1]}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["On track", "At risk"],
        zero_division=0,
    ))

    print("Top 15 transformed features:")
    print(importance_df.head(15).to_string(index=False))

    print("\n=== SAVED FILES ===")
    print("Model pipeline:", model_path)
    print("Evaluation report:", metrics_path)
    print("Feature importance:", importance_path)


if __name__ == "__main__":
    main()
