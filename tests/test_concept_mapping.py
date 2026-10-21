import pandas as pd

from src.validate_concept_mapping import REQUIRED_COLUMNS, validate_mappings


def make_mapping(**overrides):
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update({
        "code_module": "AAA",
        "assessment_type": "TMA",
        "id_assessment": "1752",
    })
    row.update(overrides)
    return row


def test_blank_mapping_is_not_eligible():
    result = validate_mappings(pd.DataFrame([make_mapping()]))

    assert result["total_rows"] == 1
    assert result["complete_verified_rows"] == 0
    assert result["unverified_or_incomplete_rows"] == 1


def test_complete_faculty_verified_mapping_is_eligible():
    row = make_mapping(
        concept_id="C001",
        concept_name="Verified concept",
        mapping_source="Faculty-approved course material",
        verified_by_faculty="yes",
    )
    result = validate_mappings(pd.DataFrame([row]))

    assert result["complete_verified_rows"] == 1
    assert result["unverified_or_incomplete_rows"] == 0


def test_unverified_mapping_is_not_eligible():
    row = make_mapping(
        concept_id="C001",
        concept_name="Example concept",
        mapping_source="Course material",
        verified_by_faculty="no",
    )
    result = validate_mappings(pd.DataFrame([row]))

    assert result["complete_verified_rows"] == 0


def test_duplicate_mapping_keys_are_detected():
    row = make_mapping()
    result = validate_mappings(pd.DataFrame([row, row]))

    assert result["duplicate_mapping_rows"] == 2


def test_unknown_prerequisite_is_reported():
    row = make_mapping(
        concept_id="C001",
        concept_name="Verified concept",
        prerequisite_concept_ids="UNKNOWN",
        mapping_source="Faculty-approved course material",
        verified_by_faculty="yes",
    )
    result = validate_mappings(pd.DataFrame([row]))

    assert len(result["invalid_prerequisite_references"]) == 1
    assert result["invalid_prerequisite_references"][0][1] == "UNKNOWN"


def test_known_verified_prerequisite_is_accepted():
    prerequisite = make_mapping(
        id_assessment="1752",
        concept_id="C001",
        concept_name="Prerequisite concept",
        mapping_source="Faculty-approved course material",
        verified_by_faculty="yes",
    )
    dependent = make_mapping(
        id_assessment="1753",
        concept_id="C002",
        concept_name="Dependent concept",
        prerequisite_concept_ids="C001",
        mapping_source="Faculty-approved course material",
        verified_by_faculty="yes",
    )

    result = validate_mappings(pd.DataFrame([prerequisite, dependent]))

    assert result["complete_verified_rows"] == 2
    assert result["invalid_prerequisite_references"] == []


def test_missing_required_column_is_rejected():
    df = pd.DataFrame([make_mapping()]).drop(columns=["concept_name"])

    try:
        validate_mappings(df)
    except ValueError as exc:
        assert "concept_name" in str(exc)
    else:
        raise AssertionError("Expected missing required column to be rejected")
