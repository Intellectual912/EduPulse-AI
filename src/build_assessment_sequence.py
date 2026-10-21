from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "student_assessment_records.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "student_assessment_sequence.csv"

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    group_cols = ["id_student", "code_module", "code_presentation"]

    # Submission date is available for every record, so use it to order
    # observed assessment history. Keep missing scores missing.
    df["date_submitted"] = pd.to_numeric(
        df["date_submitted"], errors="raise"
    )
    df["score"] = pd.to_numeric(df["score"], errors="raise")

    df = df.sort_values(
        group_cols + ["date_submitted", "id_assessment"],
        kind="stable"
    ).reset_index(drop=True)

    df["sequence_number"] = (
        df.groupby(group_cols, sort=False).cumcount() + 1
    )
    df["score_available"] = df["score"].notna()
    df["assessment_date_available"] = df["date"].notna()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=== ASSESSMENT SEQUENCE CREATED ===")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Total records preserved: {len(df):,}")
    print(f"Student-course-presentation histories: {df.groupby(group_cols).ngroups:,}")
    print(f"Records with scores: {df['score_available'].sum():,}")
    print(f"Records without scores: {(~df['score_available']).sum():,}")
    print(f"Records without assessment due dates: {(~df['assessment_date_available']).sum():,}")
    print(f"Banked records preserved: {(df['is_banked'] == 1).sum():,}")

    print("\n=== SAMPLE CHRONOLOGICAL HISTORY ===")
    sample = df[df["id_student"] == df["id_student"].iloc[0]]
    print(sample[[
        "id_student", "code_module", "code_presentation",
        "sequence_number", "assessment_type", "date_submitted",
        "date", "score", "is_banked"
    ]].head(8).to_string(index=False))

if __name__ == "__main__":
    main()
