from pathlib import Path
import pandas as pd

MAPPING_PATH = Path("data/processed/assessment_concept_mapping_workbook.csv")

REQUIRED_COLUMNS = [
    "code_module",
    "assessment_type",
    "id_assessment",
    "concept_id",
    "concept_name",
    "prerequisite_concept_ids",
    "mapping_source",
    "verified_by_faculty",
]

TRUE_VALUES = {"yes", "true", "1", "verified"}


def validate_mappings(df):
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.fillna("").copy()
    text_columns = REQUIRED_COLUMNS
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    key_columns = ["code_module", "assessment_type", "id_assessment"]
    duplicate_mask = df.duplicated(key_columns, keep=False)

    complete_mask = (
        df["concept_id"].ne("")
        & df["concept_name"].ne("")
        & df["mapping_source"].ne("")
        & df["verified_by_faculty"].str.lower().isin(TRUE_VALUES)
    )

    concept_ids = set(df.loc[complete_mask, "concept_id"])
    prerequisite_errors = []

    for row_number, value in enumerate(
        df.loc[complete_mask, "prerequisite_concept_ids"], start=2
    ):
        prerequisites = [
            item.strip() for item in value.split(";") if item.strip()
        ]
        for prerequisite in prerequisites:
            if prerequisite not in concept_ids:
                prerequisite_errors.append((row_number, prerequisite))

    return {
        "total_rows": len(df),
        "complete_verified_rows": int(complete_mask.sum()),
        "unverified_or_incomplete_rows": int((~complete_mask).sum()),
        "duplicate_mapping_rows": int(duplicate_mask.sum()),
        "invalid_prerequisite_references": prerequisite_errors,
    }


def main():
    if not MAPPING_PATH.exists():
        raise SystemExit(f"Mapping file not found: {MAPPING_PATH}")

    df = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    result = validate_mappings(df)

    print("CONCEPT MAPPING VALIDATION")
    print("Total mapping rows:", result["total_rows"])
    print("Complete and faculty-verified rows:", result["complete_verified_rows"])
    print("Unverified or incomplete rows:", result["unverified_or_incomplete_rows"])
    print("Rows involved in duplicate mapping keys:", result["duplicate_mapping_rows"])
    print("Invalid prerequisite references:", len(result["invalid_prerequisite_references"]))

    if result["invalid_prerequisite_references"]:
        print("Examples:", result["invalid_prerequisite_references"][:10])

    print("\nOnly complete, faculty-verified mappings qualify for future mastery tracking.")
    print("This audit does not infer or create concept mappings.")


if __name__ == "__main__":
    main()
