"""Train and evaluate the EduPulse AI weekly-sequence LSTM."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_PATH = DATA_DIR / "lstm_sequences.npz"
METADATA_PATH = DATA_DIR / "lstm_sequence_metadata.csv"

MODEL_PATH = MODEL_DIR / "lstm_risk_classifier.pt"
METRICS_PATH = REPORT_DIR / "lstm_model_metrics.json"
HISTORY_PATH = REPORT_DIR / "lstm_training_history.csv"

SEED = 42
BATCH_SIZE = 256
HIDDEN_SIZE = 32
DROPOUT = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 8

TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATION = "2014B"
TEST_PRESENTATION = "2014J"

DEVICE = torch.device("cpu")


class WeeklyLSTM(nn.Module):
    """LSTM classifier for four weekly student activity sequences."""

    def __init__(self, input_size=13, hidden_size=32, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.output(self.dropout(last_hidden)).squeeze(1)


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
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
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("LSTM sequence dataset or metadata is missing.")

    print("Loading LSTM dataset...")
    data = np.load(DATA_PATH)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    score_mask = data["score_missing_mask"].astype(np.float32)

    metadata = pd.read_csv(METADATA_PATH)

    if len(metadata) != len(X) or len(y) != len(X):
        raise ValueError("Dataset and metadata row counts do not match.")

    if X.shape[1:] != (4, 10):
        raise ValueError(f"Unexpected sequence shape: {X.shape}")

    if score_mask.shape != (len(X), 4, 3):
        raise ValueError("Unexpected score mask shape.")

    if not np.isfinite(X).all() or not np.isfinite(score_mask).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    presentations = metadata["code_presentation"].to_numpy()

    train_idx = np.flatnonzero(
        np.isin(presentations, TRAIN_PRESENTATIONS)
    )
    val_idx = np.flatnonzero(
        presentations == VALIDATION_PRESENTATION
    )
    test_idx = np.flatnonzero(
        presentations == TEST_PRESENTATION
    )

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("One or more chronological splits are empty.")

    if len(np.intersect1d(train_idx, val_idx)):
        raise ValueError("Training and validation overlap.")

    if len(np.intersect1d(train_idx, test_idx)):
        raise ValueError("Training and test overlap.")

    if len(np.intersect1d(val_idx, test_idx)):
        raise ValueError("Validation and test overlap.")

    print("\nChronological split:")
    print("Training:", len(train_idx), TRAIN_PRESENTATIONS)
    print("Validation:", len(val_idx), VALIDATION_PRESENTATION)
    print("Test:", len(test_idx), TEST_PRESENTATION)

    # Log-transform count features to reduce extreme skew.
    X_transformed = X.copy()
    X_transformed[:, :, :7] = np.log1p(
        np.maximum(X_transformed[:, :, :7], 0)
    )

    # Fit normalization using training data only.
    train_values = X_transformed[train_idx].reshape(-1, 10)
    feature_mean = train_values.mean(axis=0)
    feature_std = train_values.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    X_normalized = (
        X_transformed - feature_mean.reshape(1, 1, 10)
    ) / feature_std.reshape(1, 1, 10)

    # Append three score-missing indicators to the ten features.
    X_final = np.concatenate(
        [X_normalized, score_mask],
        axis=2,
    ).astype(np.float32)

    if X_final.shape[2] != 13:
        raise ValueError("LSTM must receive 13 features per week.")

    def make_loader(indices, shuffle):
        inputs = torch.tensor(X_final[indices], dtype=torch.float32)
        labels = torch.tensor(y[indices], dtype=torch.float32)

        dataset = TensorDataset(inputs, labels)

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    model = WeeklyLSTM(
        input_size=13,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    positive_count = float(y[train_idx].sum())
    negative_count = float(len(train_idx) - positive_count)

    if positive_count == 0:
        raise ValueError("Training data contains no positive labels.")

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\nStarting LSTM training on:", DEVICE)
    print("Input features per week:", X_final.shape[2])
    print("Positive class weight:", round(pos_weight.item(), 4))

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            train_loss_total += loss.item() * len(batch_y)

        train_loss = train_loss_total / len(train_idx)

        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss_total += loss.item() * len(batch_y)

        val_loss = val_loss_total / len(val_idx)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": 13,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "best_epoch": best_epoch,
                    "seed": SEED,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    # Load the best validation checkpoint for final test evaluation.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_inputs = torch.tensor(
        X_final[test_idx], dtype=torch.float32
    )

    with torch.no_grad():
        test_logits = model(test_inputs)
        test_probabilities = torch.sigmoid(test_logits).numpy()

    test_metrics = calculate_metrics(
        y[test_idx].astype(int),
        test_probabilities,
    )

    validation_inputs = torch.tensor(
        X_final[val_idx], dtype=torch.float32
    )

    with torch.no_grad():
        val_probabilities = torch.sigmoid(
            model(validation_inputs)
        ).numpy()

    validation_metrics = calculate_metrics(
        y[val_idx].astype(int),
        val_probabilities,
    )

    results = {
        "model": "PyTorch LSTM",
        "dataset": "OULAD",
        "input_shape_per_student": [4, 13],
        "training_presentations": TRAIN_PRESENTATIONS,
        "validation_presentation": VALIDATION_PRESENTATION,
        "test_presentation": TEST_PRESENTATION,
        "training_records": len(train_idx),
        "validation_records": len(val_idx),
        "test_records": len(test_idx),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "classification_threshold": 0.5,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== LSTM TRAINING COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("\n=== HELD-OUT TEST RESULTS: 2014J ===")

    for name, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    print("Confusion matrix:", test_metrics["confusion_matrix"])
    print("\nSaved model:", MODEL_PATH.relative_to(ROOT))
    print("Saved metrics:", METRICS_PATH.relative_to(ROOT))
    print("Saved history:", HISTORY_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
