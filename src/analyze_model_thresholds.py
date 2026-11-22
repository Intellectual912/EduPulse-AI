
"""Compare XGBoost and LSTM thresholds using validation data only."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train_lstm import WeeklyLSTM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATION = "2014B"


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation labels and XGBoost features.
    df = pd.read_csv(DATA_DIR / "student_features.csv")
    manifest = json.loads(
        (DATA_DIR / "feature_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    features = manifest["feature_columns"]
    validation_df = df[
        df["code_presentation"].astype(str)
        == VALIDATION_PRESENTATION
    ].copy()

    y = validation_df["target_at_risk"].to_numpy().astype(int)

    xgb_model = joblib.load(
        MODEL_DIR / "xgboost_comparison_model.joblib"
    )

    xgb_probabilities = xgb_model.predict_proba(
        validation_df[features]
    )[:, 1]

    # Load LSTM sequences and validation metadata.
    data = np.load(DATA_DIR / "lstm_sequences.npz")
    X = data["X"].astype(np.float32)
    score_mask = data["score_missing_mask"].astype(np.float32)

    metadata = pd.read_csv(
        DATA_DIR / "lstm_sequence_metadata.csv"
    )

    presentations = metadata["code_presentation"].astype(str).to_numpy()

    train_idx = np.flatnonzero(
        np.isin(presentations, TRAIN_PRESENTATIONS)
    )
    val_idx = np.flatnonzero(
        presentations == VALIDATION_PRESENTATION
    )

    lstm_y = data["y"][val_idx].astype(int)

    if not np.array_equal(y, lstm_y):
        raise ValueError(
            "XGBoost and LSTM validation labels/order do not match."
        )

    # Reproduce the exact LSTM preprocessing.
    X_transformed = X.copy()
    X_transformed[:, :, :7] = np.log1p(
        np.maximum(X_transformed[:, :, :7], 0)
    )

    checkpoint = torch.load(
        MODEL_DIR / "lstm_risk_classifier.pt",
        map_location="cpu",
        weights_only=False,
    )

    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]

    X_normalized = (
        X_transformed
        - feature_mean.reshape(1, 1, 10)
    ) / feature_std.reshape(1, 1, 10)

    X_final = np.concatenate(
        [X_normalized, score_mask], axis=2
    ).astype(np.float32)

    model = WeeklyLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_final[val_idx])
        lstm_probabilities = torch.sigmoid(
            model(inputs)
        ).numpy()

    # Evaluate both models at identical thresholds.
    results = []

    for model_name, probabilities in [
        ("XGBoost", xgb_probabilities),
        ("LSTM", lstm_probabilities),
    ]:
        for threshold in THRESHOLDS:
            metrics = calculate_metrics(
                y, probabilities, threshold
            )
            metrics["model"] = model_name
            results.append(metrics)

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "model_threshold_analysis.csv"
    results_df.to_csv(output_path, index=False)

    print("Validation presentation:", VALIDATION_PRESENTATION)
    print("Validation rows:", len(y))
    print("LSTM training rows:", len(train_idx))
    print("Positive class: Fail or Withdrawn")
    print("\nThreshold analysis:")
    print(
        results_df[
            [
                "model", "threshold", "precision", "recall",
                "f1", "specificity", "false_positive_rate",
                "balanced_accuracy",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
