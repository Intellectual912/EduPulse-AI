
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


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_PATH = DATA_DIR / "student_features.csv"
MANIFEST_PATH = DATA_DIR / "feature_manifest.json"

TARGET = "target_at_risk"
TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATION = "2014B"
TEST_PRESENTATION = "2014J"

RANDOM_STATE = 42
MAX_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 20


# ---------------------------------------------------------
# 2. EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["On-track", "At-risk"],
            output_dict=True,
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# 3. LOAD DATA AND VALIDATE INPUTS
# ---------------------------------------------------------

def main():
    if not DATA_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset or feature manifest is missing."
        )

    df = pd.read_csv(DATA_PATH)
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    features = manifest["feature_columns"]

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if "code_presentation" not in df.columns:
        raise ValueError("Missing code_presentation column.")

    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise ValueError(
            f"Missing predictor columns: {missing_features}"
        )

    forbidden = set(manifest.get("forbidden_predictors", []))
    leakage_features = sorted(set(features) & forbidden)

    if leakage_features:
        raise ValueError(
            f"Forbidden predictors detected: {leakage_features}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    df["code_presentation"] = (
        df["code_presentation"].astype(str)
    )

    required_presentations = (
        TRAIN_PRESENTATIONS
        + [VALIDATION_PRESENTATION, TEST_PRESENTATION]
    )

    available = set(df["code_presentation"].unique())
    missing_presentations = (
        set(required_presentations) - available
    )

    if missing_presentations:
        raise ValueError(
            f"Missing presentations: {sorted(missing_presentations)}"
        )

    # -----------------------------------------------------
    # 4. CHRONOLOGICAL DATA SPLIT
    # -----------------------------------------------------

    train_df = df[
        df["code_presentation"].isin(TRAIN_PRESENTATIONS)
    ].copy()

    validation_df = df[
        df["code_presentation"].eq(VALIDATION_PRESENTATION)
    ].copy()

    test_df = df[
        df["code_presentation"].eq(TEST_PRESENTATION)
    ].copy()

    partitions = {
        "training": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for name, part in partitions.items():
        if part.empty or part[TARGET].nunique() != 2:
            raise ValueError(
                f"{name} partition is empty or lacks a target class."
            )

    X_train = train_df[features].copy()
    y_train = train_df[TARGET].astype(int)

    X_validation = validation_df[features].copy()
    y_validation = validation_df[TARGET].astype(int)

    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int)

    # -----------------------------------------------------
    # 5. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
    # -----------------------------------------------------

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
        if (
            col in known_categorical
            or pd.api.types.is_object_dtype(X_train[col])
            or isinstance(X_train[col].dtype, pd.CategoricalDtype)
        )
    ]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    # -----------------------------------------------------
    # 6. BUILD PREPROCESSING PIPELINE
    # -----------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # -----------------------------------------------------
    # 7. BUILD XGBOOST CLASSIFIER
    # -----------------------------------------------------

    classifier = xgboost.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=MAX_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # -----------------------------------------------------
    # 8. FIT PREPROCESSOR ON TRAINING DATA ONLY
    # -----------------------------------------------------

    print("Preparing training data...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(
        X_validation
    )

    X_test_processed = preprocessor.transform(X_test)

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Test rows:", len(X_test))
    print("Predictors:", len(features))
    print("Numeric predictors:", len(numeric_features))
    print("Categorical predictors:", len(categorical_features))

    print(
        "Training presentations:",
        sorted(train_df["code_presentation"].unique()),
    )
    print("Validation presentation:", VALIDATION_PRESENTATION)
    print("Test presentation:", TEST_PRESENTATION)

    # -----------------------------------------------------
    # 9. TRAIN WITH VALIDATION-BASED EARLY STOPPING
    # -----------------------------------------------------

    print("\nStarting XGBoost training...")

    model.fit(
        X_train,
        y_train,
        classifier__eval_set=[
            (X_validation_processed, y_validation)
        ],
        classifier__verbose=False,
    )

    best_iteration = int(
        model.named_steps["classifier"].best_iteration
    )

    print("\nBest boosting iteration:", best_iteration + 1)

    # -----------------------------------------------------
    # 10. EVALUATE VALIDATION AND HELD-OUT TEST
    # -----------------------------------------------------

    print("\nEvaluating validation presentation...")

    validation_metrics = evaluate_model(
        model, X_validation, y_validation
    )

    print("Evaluating final test presentation...")

    test_metrics = evaluate_model(
        model, X_test, y_test
    )

    # -----------------------------------------------------
    # 11. SAVE SEPARATE MODEL AND REPORTS
    # -----------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        MODEL_DIR / "xgboost_comparison_model.joblib"
    )

    metrics_path = (
        REPORT_DIR / "xgboost_comparison_metrics.json"
    )

    importance_path = (
        REPORT_DIR / "xgboost_comparison_feature_importance.csv"
    )

    joblib.dump(model, model_path)

    metrics = {
        "model": "XGBoost",
        "dataset": "OULAD",
        "target": TARGET,
        "positive_class_meaning": "Fail or Withdrawn",
        "training_presentations": TRAIN_PRESENTATIONS,
        "validation_presentation": VALIDATION_PRESENTATION,
        "test_presentation": TEST_PRESENTATION,
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "feature_count": len(features),
        "best_iteration": best_iteration + 1,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_STATE,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    booster = model.named_steps["classifier"]
    transformed_names = (
        preprocessor.get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "feature": transformed_names,
        "importance": booster.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # -----------------------------------------------------
    # 12. DISPLAY FINAL TEST RESULTS
    # -----------------------------------------------------

    print("\n=== XGBOOST HELD-OUT TEST RESULTS ===")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        print(f"{metric}: {test_metrics[metric]:.4f}")

    print("\nConfusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
            labels=[0, 1],
            target_names=["On-track", "At-risk"],
            zero_division=0,
        )
    )

    print("\nSaved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved feature importance:", importance_path)

    print("\nXGBoost comparison training complete.")


if __name__ == "__main__":
    main()
