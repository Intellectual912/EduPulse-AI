from pathlib import Path
import csv
import io
import zipfile

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_DIR / "data" / "raw" / "anonymisedData.zip"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "student_assessment_records.csv"

def read_csv_from_zip(archive, filename):
    with archive.open(filename) as file:
        return pd.read_csv(file)

def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT_PATH}"
        )

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        assessments = read_csv_from_zip(archive, "assessments.csv")
        student_assessments = read_csv_from_zip(
            archive, "studentAssessment.csv"
        )

    required_assessments = {
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight"
    }
    required_results = {
        "id_assessment", "id_student", "date_submitted",
        "is_banked", "score"
    }

    if not required_assessments.issubset(assessments.columns):
        raise ValueError("assessments.csv is missing required columns.")
    if not required_results.issubset(student_assessments.columns):
        raise ValueError("studentAssessment.csv is missing required columns.")

    records = student_assessments.merge(
        assessments,
        on="id_assessment",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int((records["_merge"] != "both").sum())
    records = records.drop(columns=["_merge"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUTPUT_PATH, index=False)

    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Student assessment records: {len(records):,}")
    print(f"Unique students: {records['id_student'].nunique():,}")
    print(f"Unique assessment IDs in results: {records['id_assessment'].nunique():,}")
    print(f"Rows without matching assessment metadata: {unmatched:,}")
    print(f"Missing scores: {records['score'].isna().sum():,}")

if __name__ == "__main__":
    main()
