
"""Final held-out test evaluation using validation-selected thresholds."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from train_lstm import WeeklyLSTM


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATION = "2014B"
TEST_PRESENTATION = "2014J"

THRESHOLDS = {
    "XGBoost": 0.40,
    "LSTM": 0.45,
}


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "students_flagged": int(predictions.sum()),
        "test_students": int(len(y_true)),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tabular dataset and manifest.
    df = pd.read_csv(DATA_DIR / "student_features.csv")
    manifest = json.loads(
        (DATA_DIR / "feature_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    features = manifest["feature_columns"]

    presentations = df["code_presentation"].astype(str)
    test_df = df[presentations == TEST_PRESENTATION].copy()
    y_xgb = test_df["target_at_risk"].to_numpy().astype(int)

    if test_df.empty:
        raise ValueError("No XGBoost test rows found.")

    # XGBoost probabilities.
    xgb_model = joblib.load(
        MODEL_DIR / "xgboost_comparison_model.joblib"
    )
    xgb_probabilities = xgb_model.predict_proba(
        test_df[features]
    )[:, 1]

    # Load LSTM sequence dataset.
    data = np.load(DATA_DIR / "lstm_sequences.npz")
    X = data["X"].astype(np.float32)
    score_mask = data["score_missing_mask"].astype(np.float32)
    y = data["y"].astype(int)

    metadata = pd.read_csv(
        DATA_DIR / "lstm_sequence_metadata.csv"
    )
    sequence_presentations = (
        metadata["code_presentation"].astype(str).to_numpy()
    )

    train_idx = np.flatnonzero(
        np.isin(sequence_presentations, TRAIN_PRESENTATIONS)
    )
    val_idx = np.flatnonzero(
        sequence_presentations == VALIDATION_PRESENTATION
    )
    test_idx = np.flatnonzero(
        sequence_presentations == TEST_PRESENTATION
    )

    # Verify that both models evaluate the same test students.
    lstm_test_metadata = metadata.iloc[test_idx].reset_index(drop=True)
    xgb_test_metadata = test_df.reset_index(drop=True)

    if len(test_idx) != len(test_df):
        raise ValueError("XGBoost and LSTM test row counts differ.")

    if not np.array_equal(
        lstm_test_metadata["id_student"].to_numpy(),
        xgb_test_metadata["id_student"].to_numpy(),
    ):
        raise ValueError("Test student ordering does not match.")

    y_lstm = y[test_idx]

    if not np.array_equal(y_xgb, y_lstm):
        raise ValueError("Test labels do not match.")

    # Reproduce exact LSTM preprocessing.
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
        test_inputs = torch.tensor(
            X_final[test_idx], dtype=torch.float32
        )
        lstm_probabilities = torch.sigmoid(
            model(test_inputs)
        ).numpy()

    # Evaluate both models at fixed validation-selected thresholds.
    results = {
        "dataset": "OULAD",
        "test_presentation": TEST_PRESENTATION,
        "test_students": len(y_xgb),
        "positive_class": "Fail or Withdrawn",
        "threshold_selection_presentation": VALIDATION_PRESENTATION,
        "thresholds": THRESHOLDS,
        "models": {
            "XGBoost": evaluate(
                y_xgb, xgb_probabilities, THRESHOLDS["XGBoost"]
            ),
            "LSTM": evaluate(
                y_lstm, lstm_probabilities, THRESHOLDS["LSTM"]
            ),
        },
    }

    output_path = REPORT_DIR / "final_test_evaluation.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("FINAL HELD-OUT TEST EVALUATION")
    print("Test presentation:", TEST_PRESENTATION)
    print("Test students:", len(y_xgb))
    print("Thresholds:", THRESHOLDS)

    for name, metrics in results["models"].items():
        print(f"\n=== {name} ===")
        for key in [
            "threshold", "accuracy", "balanced_accuracy",
            "precision", "recall", "f1", "roc_auc",
            "average_precision", "students_flagged",
            "true_positives", "false_positives",
            "false_negatives", "true_negatives",
        ]:
            print(f"{key}: {metrics[key]}")

        print("Confusion matrix:")
        print(np.array(metrics["confusion_matrix"]))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
