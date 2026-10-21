from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "student_assessment_records.csv"
OUTPUT = ROOT / "data" / "processed" / "assessment_concept_mapping_workbook.csv"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    df = pd.read_csv(INPUT)

    # Use only real assessment identifiers and metadata.
    mapping = (
        df[["code_module", "assessment_type", "id_assessment"]]
        .drop_duplicates()
        .sort_values(
            ["code_module", "assessment_type", "id_assessment"]
        )
        .reset_index(drop=True)
    )

    mapping["concept_id"] = ""
    mapping["concept_name"] = ""
    mapping["prerequisite_concept_ids"] = ""
    mapping["mapping_source"] = ""
    mapping["verified_by_faculty"] = ""

    mapping = mapping[[
        "code_module", "assessment_type", "id_assessment",
        "concept_id", "concept_name", "prerequisite_concept_ids",
        "mapping_source", "verified_by_faculty"
    ]]

    mapping.to_csv(OUTPUT, index=False)

    print("=== CONCEPT MAPPING WORKSHEET ===")
    print(f"Output: {OUTPUT.relative_to(ROOT)}")
    print(f"Assessment mapping rows: {len(mapping):,}")
    print(f"Modules: {mapping['code_module'].nunique():,}")
    print("\nRows by module:")
    print(mapping.groupby("code_module").size().to_string())
    print("\nSample rows:")
    print(mapping.head(8).to_string(index=False))
    print("\nConcept fields intentionally left blank for verification.")

if __name__ == "__main__":
    main()
