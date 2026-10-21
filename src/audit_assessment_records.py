from pathlib import Path
import pandas as pd

path = Path("data/processed/student_assessment_records.csv")
df = pd.read_csv(path)

print("=== ASSESSMENT DATA AUDIT ===")
print(f"Total rows: {len(df):,}")
print(f"Missing scores: {df['score'].isna().sum():,}")
print(f"Duplicate student-assessment pairs: {df.duplicated(['id_student', 'id_assessment']).sum():,}")
print(f"Missing assessment dates: {df['date'].isna().sum():,}")
print(f"Missing submission dates: {df['date_submitted'].isna().sum():,}")
print(f"Missing assessment types: {df['assessment_type'].isna().sum():,}")
print(f"Missing module IDs: {df['code_module'].isna().sum():,}")
print(f"Missing presentation IDs: {df['code_presentation'].isna().sum():,}")
print(f"Banked records: {(df['is_banked'] == 1).sum():,}")

print("\n=== SCORE SUMMARY ===")
print(df["score"].describe().round(2).to_string())

print("\n=== ASSESSMENT TYPES ===")
print(df["assessment_type"].value_counts(dropna=False).to_string())

print("\n=== SAMPLE RECORDS ===")
print(df[[
    "id_student", "code_module", "code_presentation",
    "id_assessment", "assessment_type", "date",
    "date_submitted", "score", "is_banked"
]].head(5).to_string(index=False))
