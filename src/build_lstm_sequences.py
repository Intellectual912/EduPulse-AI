
"""Build leakage-safe weekly sequences for the EduPulse AI LSTM model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FEATURES_PATH = DATA_DIR / "student_features.csv"
ASSESSMENTS_PATH = DATA_DIR / "student_assessment_sequence.csv"

OUTPUT_PATH = DATA_DIR / "lstm_sequences.npz"
METADATA_PATH = DATA_DIR / "lstm_sequence_metadata.csv"
MANIFEST_PATH = DATA_DIR / "lstm_sequence_manifest.json"

KEYS = ["id_student", "code_module", "code_presentation"]
TARGET = "target_at_risk"

CUTOFF = 28
N_WEEKS = 4

ASSESSMENT_TYPES = ["TMA", "CMA", "Exam"]

# Ten features at every weekly time step.
SEQUENCE_FEATURES = [
    "vle_clicks",
    "tma_submissions",
    "cma_submissions",
    "exam_submissions",
    "tma_late_submissions",
    "cma_late_submissions",
    "exam_late_submissions",
    "tma_mean_score",
    "cma_mean_score",
    "exam_mean_score",
]


def main():
    # 1. Check required files.
    for path in [FEATURES_PATH, ASSESSMENTS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if any(path.exists() for path in [
        OUTPUT_PATH, METADATA_PATH, MANIFEST_PATH
    ]):
        raise FileExistsError(
            "An LSTM output already exists. Refusing to overwrite it."
        )

    # 2. Load the existing student feature dataset.
    features = pd.read_csv(FEATURES_PATH)

    required_feature_columns = (
        KEYS
        + [TARGET, "cutoff_day"]
        + [f"vle_clicks_wk{i}" for i in range(N_WEEKS)]
    )

    missing = sorted(set(required_feature_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if features.duplicated(KEYS).any():
        raise ValueError("Duplicate student-course enrolments in feature data.")

    if features["cutoff_day"].nunique() != 1:
        raise ValueError("Expected one common prediction cutoff.")

    if int(features["cutoff_day"].iloc[0]) != CUTOFF:
        raise ValueError("The feature dataset does not use cutoff day 28.")

    if features[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    # 3. Load original assessment event history.
    assessments = pd.read_csv(ASSESSMENTS_PATH)

    required_assessment_columns = (
        KEYS
        + [
            "date",
            "date_submitted",
            "assessment_type",
            "score",
            "is_banked",
        ]
    )

    missing = sorted(
        set(required_assessment_columns) - set(assessments.columns)
    )
    if missing:
        raise ValueError(f"Missing assessment columns: {missing}")

    for column in ["date", "date_submitted", "score", "is_banked"]:
        assessments[column] = pd.to_numeric(
            assessments[column], errors="coerce"
        )

    # 4. Retain only information available by the prediction cutoff.
    # Banked results are carry-overs, not new submission events.
    eligible = assessments.loc[
        assessments["date"].notna()
        & (assessments["date"] <= CUTOFF)
        & (assessments["date_submitted"] <= CUTOFF)
        & (assessments["date_submitted"] >= 0)
        & (assessments["date_submitted"] < CUTOFF)
        & (assessments["is_banked"] == 0)
        & assessments["assessment_type"].isin(ASSESSMENT_TYPES)
    ].copy()

    eligible["week"] = (
        eligible["date_submitted"] // 7
    ).astype(int)

    if not eligible["week"].between(0, N_WEEKS - 1).all():
        raise ValueError("An assessment event falls outside weeks 0–3.")

    # 5. Aggregate assessment events by enrolment, week and type.
    grouped = eligible.groupby(
        KEYS + ["week", "assessment_type"],
        observed=True,
    )

    counts = grouped.size().rename("submission_count")

    late_counts = (
        eligible.assign(
            is_late=(eligible["date_submitted"] > eligible["date"]).astype(int)
        )
        .groupby(
            KEYS + ["week", "assessment_type"],
            observed=True,
        )["is_late"]
        .sum()
        .rename("late_count")
    )

    score_means = (
        grouped["score"]
        .mean()
        .rename("mean_score")
    )

    weekly = pd.concat(
        [counts, late_counts, score_means], axis=1
    ).reset_index()

    # 6. Create a four-week sequence for every model enrolment.
    n_students = len(features)
    n_features = len(SEQUENCE_FEATURES)

    X = np.zeros(
        (n_students, N_WEEKS, n_features),
        dtype=np.float32,
    )

    metadata = features[KEYS + [TARGET]].copy().reset_index(drop=True)

    # Existing weekly VLE click features.
    for week in range(N_WEEKS):
        column_index = SEQUENCE_FEATURES.index("vle_clicks")
        X[:, week, column_index] = (
            pd.to_numeric(
                features[f"vle_clicks_wk{week}"], errors="coerce"
            )
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

    # Map every enrolment to its sequence row.
    row_lookup = {
        tuple(row): index
        for index, row in enumerate(metadata[KEYS].itertuples(
            index=False, name=None
        ))
    }

    # Assessment features are filled only where events exist.
    for row in weekly.itertuples(index=False):
        key = tuple(getattr(row, column) for column in KEYS)

        if key not in row_lookup:
            continue

        student_index = row_lookup[key]
        week = int(row.week)
        assessment_type = row.assessment_type

        count_feature = f"{assessment_type.lower()}_submissions"
        late_feature = f"{assessment_type.lower()}_late_submissions"
        score_feature = f"{assessment_type.lower()}_mean_score"

        X[student_index, week, SEQUENCE_FEATURES.index(count_feature)] = (
            float(row.submission_count)
        )

        X[student_index, week, SEQUENCE_FEATURES.index(late_feature)] = (
            float(row.late_count)
        )

        if pd.notna(row.mean_score):
            X[student_index, week, SEQUENCE_FEATURES.index(score_feature)] = (
                float(row.mean_score)
            )

    
    # 7. Missing scores remain distinguishable using a separate mask.
    # Zero score is valid, so it must not mean "missing".

    score_mask = np.ones(
        (n_students, N_WEEKS, len(ASSESSMENT_TYPES)),
        dtype=np.float32,
    )

    type_to_index = {
        name: index
        for index, name in enumerate(ASSESSMENT_TYPES)
    }

    # A value of 1 means no score was observed.
    # A value of 0 means a score was observed.
    # Check only existing weekly groups instead of repeatedly
    # searching the entire assessment DataFrame.
    for row in weekly.itertuples(index=False):
        if pd.isna(row.mean_score):
            continue

        key = tuple(getattr(row, column) for column in KEYS)

        if key not in row_lookup:
            continue

        student_index = row_lookup[key]
        week = int(row.week)
        assessment_type = row.assessment_type

        type_index = type_to_index[assessment_type]
        score_mask[student_index, week, type_index] = 0.0


    # 8. Save sequences, labels, score mask and enrolment metadata.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    y = metadata[TARGET].to_numpy(dtype=np.int64)

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        score_missing_mask=score_mask,
    )

    metadata.to_csv(METADATA_PATH, index=False)

    manifest = {
        "model": "PyTorch LSTM",
        "dataset": "OULAD",
        "prediction_cutoff_day": CUTOFF,
        "weeks": N_WEEKS,
        "week_definition": "Course-relative days 0-6, 7-13, 14-20, 21-27",
        "sequence_shape": list(X.shape),
        "feature_names": SEQUENCE_FEATURES,
        "target": TARGET,
        "positive_class_meaning": "Fail or Withdrawn",
        "assessment_event_date": "date_submitted",
        "assessment_due_date_limit": CUTOFF,
        "banked_records_included": False,
        "missing_score_handling": (
            "Missing weekly mean scores stored as zero with a separate mask."
        ),
        "test_presentation": "2014J",
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=== LSTM SEQUENCE DATASET CREATED ===")
    print("Students:", n_students)
    print("Sequence shape:", X.shape)
    print("Features per week:", n_features)
    print("Weeks:", N_WEEKS)
    print("Eligible assessment events:", len(eligible))
    print("Assessment events matched to model enrolments:",
          int(eligible[KEYS].apply(tuple, axis=1).isin(row_lookup).sum()))
    print("Students with assessment events:",
          int(metadata[KEYS].apply(tuple, axis=1).isin(
              set(map(tuple, eligible[KEYS].drop_duplicates().to_numpy()))
          ).sum()))
    print("Training presentations:",
          sorted(metadata.loc[
              metadata["code_presentation"] != "2014J",
              "code_presentation"
          ].unique().tolist()))
    print("Test presentation: 2014J")
    print("Saved:", OUTPUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
