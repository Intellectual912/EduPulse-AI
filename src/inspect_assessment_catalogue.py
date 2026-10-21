from pathlib import Path
import pandas as pd

path = Path("data/processed/student_assessment_records.csv")
df = pd.read_csv(path)

print("=== ASSESSMENT CATALOGUE SUMMARY ===")
catalog = (
    df.groupby(
        ["code_module", "code_presentation", "assessment_type"],
        dropna=False
    )
    .agg(
        assessment_count=("id_assessment", "nunique"),
        student_records=("id_student", "size"),
        scored_records=("score", "count"),
        mean_score=("score", "mean"),
    )
    .reset_index()
)

print(catalog.to_string(index=False, max_rows=100))

print("\n=== PRESENTATIONS AVAILABLE ===")
print(
    df[["code_module", "code_presentation"]]
    .drop_duplicates()
    .sort_values(["code_module", "code_presentation"])
    .to_string(index=False)
)

print("\n=== MAPPING STATUS ===")
mapping = pd.read_csv(
    "data/processed/assessment_concept_mapping_workbook.csv"
)
print(f"Total mappings: {len(mapping)}")
print(f"Rows with concept names filled: {mapping['concept_name'].fillna('').astype(str).str.strip().ne('').sum()}")
print("No concept labels will be inferred from assessment type.")
